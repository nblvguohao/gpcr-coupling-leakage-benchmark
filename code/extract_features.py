#!/usr/bin/env python3
"""
Comprehensive feature extraction pipeline for GPCR-G protein coupling
prediction.  Merges the functionality of extract_esm.py, extract_gprotein.py,
and extract_icl.py into a single CLI with argparse subcommands.

Subcommands
-----------
esm
    Extract 1280-d ESM-2 650M (esm2_t33_650M_UR50D) per-token embeddings for
    all GPCR sequences listed in merged_dataset/extended_sequences.json.
    Writes per-token features (seq_len x 1280) to
        server_sync/extended_data/features/esm_features_650m_paired.json

gprotein
    Fetch the seven canonical human G protein alpha-subunit sequences from
    UniProt, then extract mean-pooled 1280-d ESM-2 650M embeddings.
    Writes mean-pooled vectors to
        ../data/g_protein_esm_features_650m.json

icl
    Extract ICL2/ICL3 local ESM embeddings and physicochemical statistics
    from the pre-computed per-token ESM features (produced by the "esm"
    subcommand).  Uses ijson to stream the 5.7 GB JSON without hitting
    MemoryError.  Predicts TM helices via Kyte-Doolittle hydropathy when
    UniProt topology is unavailable.  Computes length, mean/std hydropathy,
    net charge, charge ratios, hydrophobic ratio, and aromatic ratio for
    each loop region, together with local mean-pooled ESM embeddings.
    Writes to
        ../data/icl_features_650m.json

all
    Run esm -> gprotein -> icl in sequence.  The ESM-2 model is loaded once
    and shared between the GPCR and G protein extraction steps.

Dependencies
------------
fair-esm, torch, numpy, ijson, requests, tqdm
"""

import argparse
import json
import requests
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
import esm
import ijson
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

# ============================================================================
# Paths — all relative to this file's directory (code/)
# ============================================================================
BASE = Path(__file__).parent

# Shared input
_SEQUENCES_FILE = BASE / "merged_dataset" / "extended_sequences.json"

# esm subcommand paths
_ESM_OUTPUT = (
    BASE / "server_sync" / "extended_data" / "features"
    / "esm_features_650m_paired.json"
)

# gprotein subcommand paths
_GPROTEIN_OUTPUT = BASE.parent / "data" / "g_protein_esm_features_650m.json"

# icl subcommand paths
_TOPOLOGY_FILE = BASE.parent / "data" / "uniprot_topology.json"
_ICL_OUTPUT = BASE.parent / "data" / "icl_features_650m.json"

# G protein UniProt accessions
_UNIPROT_IDS = {
    "GNAQ":  "P50148",
    "GNAI1": "P63096",
    "GNAI2": "P04899",
    "GNAI3": "P08754",
    "GNAS":  "P63092",
    "GNA12": "Q03113",
    "GNA13": "Q14344",
}

# ============================================================================
# Kyte-Doolittle hydropathy scale & amino-acid class sets
# ============================================================================
KD_SCALE = {
    "A": 1.8, "C": 2.5, "D": -3.5, "E": -3.5, "F": 2.8,
    "G": -0.4, "H": -3.2, "I": 4.5, "K": -3.9, "L": 3.8,
    "M": 1.9, "N": -3.5, "P": -1.6, "Q": -3.5, "R": -4.5,
    "S": -0.8, "T": -0.7, "V": 4.2, "W": -0.9, "Y": -1.3,
}

POSITIVE    = {"K", "R", "H"}
NEGATIVE    = {"D", "E"}
HYDROPHOBIC = {"A", "V", "I", "L", "M", "F", "W", "G", "P", "C"}
AROMATIC    = {"F", "W", "Y"}


# ============================================================================
# 1. Shared utilities
# ============================================================================

def load_esm_model():
    """Load esm2_t33_650M_UR50D and return (model, batch_converter)."""
    print("[INFO] Loading ESM-2 model (esm2_t33_650M_UR50D) ...")
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    if torch.cuda.is_available():
        model = model.eval().cuda()
        print("[INFO] Using CUDA.")
    else:
        model = model.eval().cpu()
        print("[INFO] Using CPU (will be very slow).")
    batch_converter = alphabet.get_batch_converter()
    return model, batch_converter


def extract_per_token_esm(model, batch_converter, uid: str, seq: str):
    """Run a single sequence through ESM-2 and return per-token features
    (seq_len x 1280) as a Python list of lists."""
    _, _, batch_tokens = batch_converter([(uid, seq)])
    try:
        if torch.cuda.is_available():
            batch_tokens = batch_tokens.cuda()
        with torch.no_grad():
            results = model(batch_tokens, repr_layers=[33],
                            return_contacts=False)
    except torch.OutOfMemoryError:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            batch_tokens = batch_tokens.cpu()
            with torch.no_grad():
                results = model(batch_tokens, repr_layers=[33],
                                return_contacts=False)
        else:
            raise

    token_representations = results["representations"][33].cpu().numpy()
    seq_len = len(seq)
    features = token_representations[0, 1:seq_len + 1, :]
    return features.tolist()


# ============================================================================
# 2. TM-helix prediction (Kyte-Doolittle sliding window)
# ============================================================================

def predict_tm_helices(
    seq: str,
    window: int = 19,
    threshold: float = 1.4,
    min_len: int = 18,
    max_len: int = 35,
) -> List[Tuple[int, int]]:
    """Predict transmembrane helix regions using the Kyte-Doolittle scale.

    Returns a list of (start_1based, end_1based) tuples, at most 7 regions.
    """
    scores = []
    for i in range(len(seq) - window + 1):
        seg = seq[i:i + window]
        score = sum(KD_SCALE.get(aa, 0) for aa in seg) / window
        scores.append(score)

    regions = []
    in_tm = False
    start = 0
    for i, sc in enumerate(scores):
        if sc >= threshold and not in_tm:
            in_tm = True
            start = i
        elif sc < threshold - 0.4 and in_tm:
            in_tm = False
            end = i + window - 1
            seg_len = end - start + 1
            if min_len <= seg_len <= max_len:
                regions.append((start + 1, end))

    if not regions:
        return []

    # Merge close segments
    merged = [regions[0]]
    for s, e in regions[1:]:
        last_s, last_e = merged[-1]
        if s - last_e < 10:
            merged[-1] = (last_s, e)
        else:
            merged.append((s, e))

    # Keep at most 7 by highest hydropathy
    if len(merged) > 7:
        seg_scores = []
        for s, e in merged:
            seg = seq[s - 1:e]
            sc = sum(KD_SCALE.get(aa, 0) for aa in seg) / len(seg)
            seg_scores.append((sc, s, e))
        seg_scores.sort(reverse=True)
        merged = [(s, e) for _, s, e in seg_scores[:7]]
        merged.sort(key=lambda x: x[0])

    return merged


def derive_loops(
    tm_regions: List[Tuple[int, int]], seq_len: int
) -> Dict[str, Tuple[int, int]]:
    """Derive inter-helical loop boundaries from TM regions.

    Returns a dict mapping loop name -> (start_1based, end_1based).
    """
    loops: Dict[str, Tuple[int, int]] = {}
    if not tm_regions:
        return loops

    loop_names = ["ICL1", "ECL1", "ICL2", "ECL2", "ICL3", "ECL3"]
    for i in range(len(tm_regions) - 1):
        gap_s = tm_regions[i][1] + 1
        gap_e = tm_regions[i + 1][0] - 1
        if gap_e >= gap_s:
            name = loop_names[i] if i < len(loop_names) else f"loop_{i}"
            loops[name] = (gap_s, gap_e)

    if tm_regions[0][0] > 1:
        loops["N-tail"] = (1, tm_regions[0][0] - 1)
    loops["C-tail"] = (tm_regions[-1][1] + 1, seq_len)
    return loops


# ============================================================================
# 3. Physicochemical statistics
# ============================================================================

def sequence_stats(seq: str) -> Dict:
    """Compute physicochemical statistics for an amino-acid sequence.

    Returns a dict with length, mean/std hydropathy, net charge per residue,
    positive/negative charge ratios, hydrophobic ratio, and aromatic ratio.
    """
    n = len(seq)
    if n == 0:
        return {}

    hydro_scores = [KD_SCALE.get(aa, 0) for aa in seq]
    pos = sum(1 for aa in seq if aa in POSITIVE)
    neg = sum(1 for aa in seq if aa in NEGATIVE)

    return {
        "length": n,
        "mean_hydro": float(np.mean(hydro_scores)),
        "std_hydro": float(np.std(hydro_scores)),
        "net_charge": (pos - neg) / n,
        "pos_charge_ratio": pos / n,
        "neg_charge_ratio": neg / n,
        "hydrophobic_ratio": sum(1 for aa in seq if aa in HYDROPHOBIC) / n,
        "aromatic_ratio": sum(1 for aa in seq if aa in AROMATIC) / n,
    }


def extract_local_esm(
    esm_tokens: list, seq: str, start: int, end: int
) -> list:
    """Mean-pool per-token ESM embeddings over the region [start, end] (1-based).

    esm_tokens: list[list[float]]  shape (seq_len, 1280)
    Returns a list[float] of length 1280 (the mean-pooled embedding).
    """
    seq_len = len(seq)
    token_len = len(esm_tokens)
    valid_len = min(seq_len, token_len)
    if end > valid_len:
        end = valid_len
    if start < 1:
        start = 1
    if end >= start:
        local = esm_tokens[start - 1:end]
        dim = len(local[0])
        sums = [0.0] * dim
        for row in local:
            for i, v in enumerate(row):
                sums[i] += float(v)
        n = len(local)
        return [s / n for s in sums]
    return [0.0] * len(esm_tokens[0]) if esm_tokens else []


# ============================================================================
# 4. Subcommand: esm  (GPCR per-token ESM-2 embeddings)
# ============================================================================

def run_esm_extraction(
    model: Optional[torch.nn.Module] = None,
    batch_converter=None,
) -> str:
    """Extract 1280-d ESM-2 650M per-token embeddings for all GPCR sequences.

    Input  : merged_dataset/extended_sequences.json
    Output : server_sync/extended_data/features/esm_features_650m_paired.json

    Parameters
    ----------
    model, batch_converter : optional
        Pre-loaded ESM-2 model.  If None, the model is loaded internally.

    Returns
    -------
    str : path to the output JSON file.
    """
    print("=" * 70)
    print("  ESM-2 650M Feature Extraction")
    print("=" * 70)

    if model is None or batch_converter is None:
        model, batch_converter = load_esm_model()

    with open(_SEQUENCES_FILE, encoding="utf-8") as f:
        sequences = json.load(f)
    print(f"[INFO] Loaded {len(sequences)} sequences")

    features_dict = {}
    for uid, seq in tqdm(list(sequences.items()), desc="Extracting 650M"):
        seq_str = seq["sequence"] if isinstance(seq, dict) else seq
        features_dict[uid] = extract_per_token_esm(
            model, batch_converter, uid, seq_str,
        )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    _ESM_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with open(_ESM_OUTPUT, "w", encoding="utf-8") as f:
        json.dump(features_dict, f)
    print(f"[OK] Saved {len(features_dict)} features to {_ESM_OUTPUT}")

    sample_id = list(features_dict.keys())[0]
    sample_feat = np.array(features_dict[sample_id])
    print(f"[OK] Sample feature shape: {sample_feat.shape}")

    return str(_ESM_OUTPUT)


# ============================================================================
# 5. Subcommand: gprotein  (G protein mean-pooled ESM-2 embeddings)
# ============================================================================

def _fetch_sequence(uniprot_id: str) -> str:
    """Fetch the canonical sequence for a UniProt accession via REST API."""
    url = f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.fasta"
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    lines = resp.text.strip().split("\n")
    return "".join(lines[1:])


def run_gprotein_extraction(
    model: Optional[torch.nn.Module] = None,
    batch_converter=None,
) -> str:
    """Fetch G protein sequences and extract 1280-d mean-pooled ESM-2 embeddings.

    Output : ../data/g_protein_esm_features_650m.json

    Parameters
    ----------
    model, batch_converter : optional
        Pre-loaded ESM-2 model.  If None, the model is loaded internally.

    Returns
    -------
    str : path to the output JSON file.
    """
    print("=" * 60)
    print("  G Protein 650M ESM-2 Feature Extraction")
    print("=" * 60)

    if model is None or batch_converter is None:
        model, batch_converter = load_esm_model()

    sequences = {}
    for name, uid in _UNIPROT_IDS.items():
        seq = _fetch_sequence(uid)
        sequences[name] = seq
        print(f"  {name} ({uid}): {len(seq)} residues")

    output = {}
    for name, seq in sequences.items():
        print(f"  Extracting {name} ...")
        _, _, batch_tokens = batch_converter([(name, seq)])
        if torch.cuda.is_available():
            batch_tokens = batch_tokens.cuda()
        with torch.no_grad():
            results = model(batch_tokens, repr_layers=[33],
                            return_contacts=False)
        token_repr = results["representations"][33].cpu().numpy()
        seq_len = len(seq)
        features = token_repr[0, 1:seq_len + 1, :]
        mean_pooled = features.mean(axis=0).tolist()

        output[name] = {
            "uniprot_id": _UNIPROT_IDS[name],
            "sequence_length": len(seq),
            "mean_pooling": mean_pooled,
        }
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    _GPROTEIN_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with open(_GPROTEIN_OUTPUT, "w") as f:
        json.dump(output, f)
    print(f"[OK] Saved to {_GPROTEIN_OUTPUT}")

    sample = list(output.values())[0]
    print(f"[OK] Sample dimension: {len(sample['mean_pooling'])}")

    return str(_GPROTEIN_OUTPUT)


# ============================================================================
# 6. Subcommand: icl  (ICL2/ICL3 local features via ijson streaming)
# ============================================================================

def run_icl_extraction() -> str:
    """Extract ICL2/ICL3 local ESM embeddings and physicochemical statistics.

    Reads the per-token ESM features via ijson (streaming) to handle the
    large 5.7 GB JSON.  Uses UniProt topology when available, otherwise
    falls back to Kyte-Doolittle TM-helix prediction.

    Input  : merged_dataset/extended_sequences.json
             server_sync/extended_data/features/esm_features_650m_paired.json
             ../data/uniprot_topology.json  (optional)
    Output : ../data/icl_features_650m.json

    Returns
    -------
    str : path to the output JSON file.
    """
    print("=" * 70)
    print("  650M ICL2/ICL3 Local Feature Extraction (streaming)")
    print("=" * 70)

    # ---- load sequences ----
    with open(_SEQUENCES_FILE, encoding="utf-8") as f:
        sequences = json.load(f)

    # ---- load topology cache (optional) ----
    topo_cache = {}
    if _TOPOLOGY_FILE.exists():
        with open(_TOPOLOGY_FILE, encoding="utf-8") as f:
            topo_cache = json.load(f)
        print(f"[INFO] Loaded UniProt topology: {len(topo_cache)} entries")

    def get_topology(uid: str, seq: str):
        """Return (tm_regions, loops) for a given GPCR.

        Tries exact match, then strips a leading prefix before '_',
        then searches all keys.  Falls back to Kyte-Doolittle prediction.
        """
        rec = topo_cache.get(uid)
        if rec is None and "_" in uid:
            base = uid.split("_", 1)[1]
            rec = topo_cache.get(base)
        if rec is None:
            for key in topo_cache:
                if "_" in key and key.split("_", 1)[1] == uid:
                    rec = topo_cache[key]
                    break
        if rec and rec.get("tm_regions"):
            return rec["tm_regions"], rec.get("loops", {})
        return predict_tm_helices(seq), {}

    # ---- stream per-token features ----
    results = {}
    stats = defaultdict(list)
    heuristic_fallback = 0
    count = 0

    print("[INFO] Streaming 650M per-token features via ijson ...")
    with open(_ESM_OUTPUT, "rb") as f:
        for uid, esm_tokens in ijson.kvitems(f, ""):
            seq_rec = sequences.get(uid)
            if seq_rec is None:
                continue
            seq = seq_rec["sequence"] if isinstance(seq_rec, dict) else seq_rec
            if isinstance(seq, dict):
                seq = seq.get("sequence", "")
            if not seq:
                continue

            tm_regions, loops = get_topology(uid, seq)
            if not loops:
                loops = derive_loops(tm_regions, len(seq))
                heuristic_fallback += 1

            feat = {
                "tm_regions": tm_regions,
                "loops": {k: list(v) for k, v in loops.items()},
            }

            for region_name in ["ICL2", "ICL3", "N-tail", "C-tail"]:
                if region_name not in loops:
                    continue
                s, e = loops[region_name]
                subseq = seq[s - 1:e]
                local_stats = sequence_stats(subseq)
                local_esm = extract_local_esm(esm_tokens, seq, s, e)

                feat[f"{region_name}_stats"] = local_stats
                feat[f"{region_name}_esm"] = local_esm

                for k, v in local_stats.items():
                    stats[f"{region_name}_{k}"].append(v)

            results[uid] = feat
            count += 1
            if count % 50 == 0:
                print(f"  ... processed {count} sequences")

    # ---- save ----
    _ICL_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with open(_ICL_OUTPUT, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"[OK] Extracted {len(results)} sequences -> {_ICL_OUTPUT}")
    print(f"[INFO] Heuristic fallback (Kyte-Doolittle): {heuristic_fallback}")

    # ---- summary statistics ----
    print("\n--- ICL2 / ICL3 Summary Statistics ---")
    for key in sorted(stats.keys()):
        vals = stats[key]
        if vals:
            print(
                f"  {key:30s}: mean={np.mean(vals):.3f}, "
                f"std={np.std(vals):.3f}, n={len(vals)}"
            )

    return str(_ICL_OUTPUT)


# ============================================================================
# 7. Subcommand: all  (run esm -> gprotein -> icl)
# ============================================================================

def run_all() -> None:
    """Run all three extraction steps sequentially.

    The ESM-2 model is loaded once and shared between the GPCR and G protein
    extraction steps, then ICL features are streamed from the resulting
    per-token feature file.
    """
    print("=" * 70)
    print("  FULL FEATURE EXTRACTION PIPELINE")
    print("=" * 70)

    # --- load model once ---
    model, batch_converter = load_esm_model()

    # --- 1) GPCR per-token embeddings ---
    run_esm_extraction(model=model, batch_converter=batch_converter)

    # --- 2) G protein mean-pooled embeddings ---
    run_gprotein_extraction(model=model, batch_converter=batch_converter)

    # --- 3) ICL local features (streaming, no model needed) ---
    run_icl_extraction()

    print("\n" + "=" * 70)
    print("  ALL EXTRACTIONS COMPLETE")
    print("=" * 70)


# ============================================================================
# 8. CLI entry point
# ============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Feature extraction pipeline for GPCR-G protein coupling "
            "prediction.  Choose a subcommand or use 'all' to run the "
            "entire pipeline."
        ),
    )
    sub = parser.add_subparsers(
        dest="command",
        title="subcommands",
        description="Which extraction step to run.",
    )

    # esm
    _p_esm = sub.add_parser(
        "esm",
        help="Extract 1280-d ESM-2 650M per-token embeddings for all GPCRs",
    )

    # gprotein
    _p_gp = sub.add_parser(
        "gprotein",
        help="Extract 1280-d mean-pooled ESM-2 embeddings for 7 G proteins",
    )

    # icl
    _p_icl = sub.add_parser(
        "icl",
        help="Extract ICL2/ICL3 local ESM and physicochemical features "
             "(requires pre-computed per-token ESM features)",
    )

    # all
    _p_all = sub.add_parser(
        "all",
        help="Run esm -> gprotein -> icl sequentially",
    )

    args = parser.parse_args()

    if args.command == "esm":
        run_esm_extraction()
    elif args.command == "gprotein":
        run_gprotein_extraction()
    elif args.command == "icl":
        run_icl_extraction()
    elif args.command == "all":
        run_all()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
