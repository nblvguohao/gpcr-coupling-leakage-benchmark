#!/usr/bin/env python3
"""
SGSP: Structure-Guided Sequence Pooling for GPCR ESM-2 embeddings.

Precomputes structure-aware GPCR embeddings by:
1. Loading per-residue ESM-2 650M features (L, 1280)
2. Loading per-residue pLDDT scores as confidence weights
3. Up-weighting ICL2/3 regions (known binding interface)
4. Weighted pooling: embedding_i * (pLDDT_i * ICL_bonus_i)

Output: sgsp_embeddings_650m.json (1280-d per GPCR, replaces mean-pooled)
"""

import json, numpy as np
from pathlib import Path

BASE = Path(__file__).parent
DATA_DIR = BASE / "paired_dataset"
ESM_FILE = BASE / "server_sync" / "extended_data" / "features" / "esm_features_650m_paired.json"
STRUCTURE_FILE = DATA_DIR / "alphafold_structure_features.json"
ICL_FILE = DATA_DIR / "icl_features_650m.json"
UNIPROT_TOPOLOGY_FILE = DATA_DIR / "uniprot_topology.json"
OUTPUT_FILE = DATA_DIR / "sgsp_embeddings_650m.json"

ICL_BONUS = 2.0  # ICL regions get 2x base weight


def get_icl_boundaries(uniprot_id, topology_data):
    """Extract ICL2 and ICL3 residue ranges from UniProt topology."""
    tm_helices = []
    for entry_id, data in topology_data.items():
        features = data if isinstance(data, list) else data.get("features", [])
        for feat in features:
            if feat.get("type") == "Transmembrane" or feat.get("description", "").startswith("Transmembrane"):
                tm_helices.append((feat["location"]["start"]["value"], feat["location"]["end"]["value"]))
    tm_helices = sorted(tm_helices)[:7]
    if len(tm_helices) < 6:
        return None, None  # insufficient TMs
    tm3_end, tm4_start = tm_helices[2][1], tm_helices[3][0]
    tm5_end, tm6_start = tm_helices[4][1], tm_helices[5][0]
    icl2 = (tm3_end + 1, tm4_start - 1)
    icl3 = (tm5_end + 1, tm6_start - 1)
    return icl2, icl3


def get_icl_boundaries_from_icl_data(gid, icl_data):
    """Get ICL start/end from icl_features_650m.json if topology file fails."""
    for key in icl_data:
        if key == gid or key.endswith("_" + gid):
            rec = icl_data[key]
            icl2 = rec.get("ICL2_start"), rec.get("ICL2_end")
            icl3 = rec.get("ICL3_start"), rec.get("ICL3_end")
            if icl2[0] is not None and icl2[1] is not None:
                return icl2, icl3
    return None, None


def compute_sgsp_embedding(esm_tokens, plddt_scores, icl2_boundary, icl3_boundary):
    """Compute structure-guided pooled embedding.

    Weight per residue = pLDDT_score * ICL_bonus
    - ICL_bonus = ICL_BONUS for ICL2/3 residues, 1.0 otherwise
    - pLDDT normalized to [0, 1]
    """
    L = len(esm_tokens)

    # pLDDT weights (base weight for each residue)
    plddt_arr = np.array(plddt_scores[:L], dtype=np.float32)
    weights = plddt_arr / 100.0  # normalize to [0, 1]

    # ICL bonus: double the weight of ICL2/3 residues
    # Input boundaries are 1-indexed (UniProt convention)
    icl_mask = np.zeros(L, dtype=bool)
    for icl_start, icl_end in [icl2_boundary, icl3_boundary]:
        if icl_start is not None and icl_end is not None and icl_end > icl_start:
            i_start = max(0, icl_start - 1)  # 1-indexed to 0-indexed
            i_end = min(L - 1, icl_end - 1)
            if i_end >= i_start:
                icl_mask[i_start:i_end + 1] = True

    if icl_mask.any():
        weights[icl_mask] = weights[icl_mask] * ICL_BONUS

    # Weighted mean pooling
    if weights.sum() > 0:
        pooled = np.average(esm_tokens, axis=0, weights=weights)
    else:
        pooled = np.mean(esm_tokens, axis=0)

    return pooled.tolist(), {
        "n_icl2": int(icl2_boundary[1] - icl2_boundary[0] + 1) if icl2_boundary[0] is not None else 0,
        "n_icl3": int(icl3_boundary[1] - icl3_boundary[0] + 1) if icl3_boundary[0] is not None else 0,
        "mean_plddt": float(plddt_arr.mean()) if len(plddt_arr) > 0 else 0,
        "mean_icl_weight": float(weights[icl_mask].mean()) if icl_mask.any() else 0,
    }


def main():
    print("=" * 70)
    print("  SGSP: Structure-Guided Sequence Pooling")
    print("=" * 70)

    # Load per-residue ESM-2
    print("\nLoading per-residue ESM-2 650M features ...")
    with open(ESM_FILE) as f:
        esm_data = json.load(f)
    print(f"  {len(esm_data)} GPCRs")

    # Load structure features
    print("Loading AlphaFold structure features ...")
    with open(STRUCTURE_FILE) as f:
        struct_data = json.load(f)
    print(f"  {len(struct_data)} GPCRs with pLDDT")

    # Load ICL data
    print("Loading ICL features ...")
    with open(ICL_FILE) as f:
        icl_data = json.load(f)
    print(f"  {len(icl_data)} GPCRs with ICL annotations")

    # Load UniProt topology
    topology_data = {}
    if UNIPROT_TOPOLOGY_FILE.exists():
        with open(UNIPROT_TOPOLOGY_FILE) as f:
            topology_data = json.load(f)

    # Build ICL boundary lookup from TM regions
    icl_boundaries = {}
    for gid, rec in icl_data.items():
        tm_regions = rec.get("tm_regions", [])
        if len(tm_regions) >= 6:
            icl2_s = tm_regions[2][1] + 1 if tm_regions[2][1] else None  # TM3.end + 1
            icl2_e = tm_regions[3][0] - 1 if tm_regions[3][0] else None  # TM4.start - 1
            icl3_s = tm_regions[4][1] + 1 if tm_regions[4][1] else None  # TM5.end + 1
            icl3_e = tm_regions[5][0] - 1 if tm_regions[5][0] else None  # TM6.start - 1
            icl_boundaries[gid] = ((icl2_s, icl2_e), (icl3_s, icl3_e))
        elif "_" in gid:
            base = gid.split("_", 1)[1]
            if base not in [k for k in icl_boundaries]:
                pass  # will try to find from base ID later

    results = {}
    stats = {"with_structure": 0, "without_structure": 0}

    for gid in esm_data:
        tokens = np.array(esm_data[gid])  # (L, 1280)

        # Get pLDDT
        plddt_scores = None
        if gid in struct_data and isinstance(struct_data[gid].get("plddt"), dict):
            pr = struct_data[gid]["plddt"].get("per_residue")
            if pr and len(pr) >= len(tokens):
                plddt_scores = pr[:len(tokens)]
            elif pr:
                plddt_scores = pr
        # Try alternate ID
        if plddt_scores is None:
            for alt_key in struct_data:
                if "_" in alt_key and alt_key.split("_", 1)[1] == gid:
                    pr = struct_data[alt_key]["plddt"].get("per_residue")
                    if pr:
                        plddt_scores = pr[:len(tokens)]
                        break

        if plddt_scores is None:
            stats["without_structure"] += 1
            plddt_scores = np.ones(len(tokens)) * 70.0  # default confidence

        if isinstance(plddt_scores, list):
            plddt_scores = plddt_scores[:len(tokens)]
            # Pad if shorter
            if len(plddt_scores) < len(tokens):
                plddt_scores = list(plddt_scores) + [70.0] * (len(tokens) - len(plddt_scores))

        # Get ICL boundaries from TM regions
        icl_pair = icl_boundaries.get(gid)
        if icl_pair is None:
            # Try alternate ID
            for key in icl_boundaries:
                if "_" in key and key.split("_", 1)[1] == gid:
                    icl_pair = icl_boundaries[key]
                    break
        icl2, icl3 = icl_pair if icl_pair else ((None, None), (None, None))

        # Compute SGSP embedding
        pooled, meta = compute_sgsp_embedding(
            tokens, plddt_scores, icl2, icl3
        )
        results[gid] = {
            "embedding": pooled,
            "sgsp_meta": meta,
        }
        if plddt_scores is not None and not all(s == 70.0 for s in plddt_scores[:5]):
            stats["with_structure"] += 1

    # Save
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w") as f:
        json.dump(results, f)
    print(f"\n[OK] Saved {len(results)} SGSP embeddings to {OUTPUT_FILE}")
    print(f"  With structure info: {stats['with_structure']}")
    print(f"  Without structure (default weights): {stats['without_structure']}")

    # Verify
    sample = list(results.values())[0]
    print(f"\nSample embedding dim: {len(sample['embedding'])}")
    print(f"Sample meta: {sample['sgsp_meta']}")


if __name__ == "__main__":
    main()
