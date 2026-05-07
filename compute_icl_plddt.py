#!/usr/bin/env python3
"""
Compute ICL2/ICL3 specific pLDDT and other AlphaFold-derived structure features.

Inputs:
  - paired_dataset/alphafold_structure_features.json
  - paired_dataset/uniprot_topology.json

Outputs:
  - paired_dataset/alphafold_icl_features.json
    { uniprot_id: {
        "icl2_plddt_mean": float,
        "icl2_plddt_std": float,
        "icl3_plddt_mean": float,
        "icl3_plddt_std": float,
        "ntail_plddt_mean": float,
        "ctail_plddt_mean": float,
        "tm_mean_plddt": float,
        "global_plddt_mean": float,
        "global_plddt_std": float,
        "high_confidence_ratio_70": float,
        "high_confidence_ratio_90": float,
        "sasa_mean": float,
        "sasa_buried_ratio": float,
        "contact_density": float,
        "mean_contacts_per_residue": float,
    }}
"""

import json
from pathlib import Path
import numpy as np

BASE = Path(__file__).parent
ALPHA_FILE = BASE / "paired_dataset" / "alphafold_structure_features.json"
GEO_FILE = BASE / "paired_dataset" / "alphafold_geometric_features.json"
PAE_FILE = BASE / "paired_dataset" / "alphafold_pae_features.json"
TOPO_FILE = BASE / "paired_dataset" / "uniprot_topology.json"
OUTPUT_FILE = BASE / "paired_dataset" / "alphafold_icl_features.json"

GEO_KEYS = [
    "tm5_tm6_cyto_ca_distance", "icl2_end_to_end_ca_distance",
    "icl3_end_to_end_ca_distance", "tm5_tm6_cyto_dihedral_angle",
    "icl2_aromatic_centroid_depth", "interface_patch_sasa",
    "interface_patch_sasa_ratio", "icl2_helix_ratio", "icl2_sheet_ratio",
    "icl2_coil_ratio", "icl3_helix_ratio", "icl3_sheet_ratio",
    "icl3_coil_ratio",
]

PAE_KEYS = [
    "icl2_mean_pae", "icl2_intra_pae",
    "icl3_mean_pae", "icl3_intra_pae",
    "icl2_tm5_pae", "icl2_tm6_pae",
    "icl3_tm5_pae", "icl3_tm6_pae",
]


def mean_plddt_for_range(per_residue, start, end):
    """1-indexed inclusive range."""
    arr = np.array(per_residue)
    s = max(0, start - 1)
    e = min(len(arr), end)
    if s >= e:
        return float("nan"), float("nan")
    segment = arr[s:e]
    return float(np.mean(segment)), float(np.std(segment))


def main():
    with open(ALPHA_FILE) as f:
        alpha_data = json.load(f)
    with open(TOPO_FILE) as f:
        topo_data = json.load(f)

    geo_data = {}
    if GEO_FILE.exists():
        with open(GEO_FILE) as f:
            geo_data = json.load(f)

    pae_data = {}
    if PAE_FILE.exists():
        with open(PAE_FILE) as f:
            pae_data = json.load(f)

    results = {}
    for uid, alpha in alpha_data.items():
        topo = topo_data.get(uid)
        if topo is None:
            continue

        plddt = alpha.get("plddt", {})
        per_residue = plddt.get("per_residue", [])
        if not per_residue:
            continue

        loops = topo.get("loops", {})
        rec = {}

        # ICL2 / ICL3 pLDDT
        for loop_name, key_prefix in [("ICL2", "icl2"), ("ICL3", "icl3"),
                                       ("N-tail", "ntail"), ("C-tail", "ctail")]:
            region = loops.get(loop_name)
            if region:
                mean_v, std_v = mean_plddt_for_range(per_residue, region[0], region[1])
                rec[f"{key_prefix}_plddt_mean"] = mean_v if not np.isnan(mean_v) else 0.0
                rec[f"{key_prefix}_plddt_std"] = std_v if not np.isnan(std_v) else 0.0
            else:
                rec[f"{key_prefix}_plddt_mean"] = 0.0
                rec[f"{key_prefix}_plddt_std"] = 0.0

        # TM regions average pLDDT
        tm_regions = topo.get("tm_regions", [])
        tm_plddts = []
        for tm_start, tm_end in tm_regions:
            mean_v, _ = mean_plddt_for_range(per_residue, tm_start, tm_end)
            if not np.isnan(mean_v):
                tm_plddts.append(mean_v)
        rec["tm_mean_plddt"] = float(np.mean(tm_plddts)) if tm_plddts else 0.0

        # Global pLDDT
        rec["global_plddt_mean"] = plddt.get("mean", 0.0)
        rec["global_plddt_std"] = plddt.get("std", 0.0)
        rec["high_confidence_ratio_70"] = plddt.get("high_confidence_ratio_70", 0.0)
        rec["high_confidence_ratio_90"] = plddt.get("high_confidence_ratio_90", 0.0)

        # SASA
        sasa = alpha.get("sasa", {})
        rec["sasa_mean"] = sasa.get("mean_sasa", 0.0)
        rec["sasa_buried_ratio"] = sasa.get("buried_ratio", 0.0)

        # Contact map
        contacts = alpha.get("contact_map", {})
        rec["contact_density"] = contacts.get("contact_density", 0.0)
        rec["mean_contacts_per_residue"] = contacts.get("mean_contacts_per_residue", 0.0)

        # Geometric features
        geo = geo_data.get(uid, {})
        for key in GEO_KEYS:
            rec[key] = geo.get(key, 0.0)

        # PAE features
        pae = pae_data.get(uid, {})
        for key in PAE_KEYS:
            rec[key] = pae.get(key, 0.0)

        results[uid] = rec

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"[OK] Computed AlphaFold ICL features for {len(results)} proteins")
    print(f"     Saved to {OUTPUT_FILE}")

    # Quick summary
    icl2_means = [v["icl2_plddt_mean"] for v in results.values() if v["icl2_plddt_mean"] > 0]
    icl3_means = [v["icl3_plddt_mean"] for v in results.values() if v["icl3_plddt_mean"] > 0]
    print(f"     ICL2 pLDDT mean: {np.mean(icl2_means):.2f} ± {np.std(icl3_means):.2f} (n={len(icl2_means)})")
    print(f"     ICL3 pLDDT mean: {np.mean(icl3_means):.2f} ± {np.std(icl3_means):.2f} (n={len(icl3_means)})")


if __name__ == "__main__":
    main()
