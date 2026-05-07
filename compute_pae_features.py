#!/usr/bin/env python3
"""
下载并提取 AlphaFold Predicted Aligned Error (PAE) 矩阵中的 ICL 区域柔度特征。

输入:
  - merged_dataset/extended_sequences.json
  - paired_dataset/uniprot_topology.json

输出:
  - paired_dataset/alphafold_pae_features.json
    { uniprot_id: {
        "icl2_mean_pae": float,
        "icl2_intra_pae": float,
        "icl3_mean_pae": float,
        "icl3_intra_pae": float,
        "icl2_tm5_pae": float,
        "icl2_tm6_pae": float,
        "icl3_tm5_pae": float,
        "icl3_tm6_pae": float,
    }}
"""

import json
import requests
import numpy as np
from pathlib import Path
from tqdm import tqdm

BASE = Path(__file__).parent
SEQUENCES_FILE = BASE / "merged_dataset" / "extended_sequences.json"
TOPO_FILE = BASE / "paired_dataset" / "uniprot_topology.json"
OUTPUT_FILE = BASE / "paired_dataset" / "alphafold_pae_features.json"

PAE_BASE_URL = "https://alphafold.ebi.ac.uk/files"


def download_pae(uniprot_id: str):
    """Download PAE JSON from AlphaFold EBI. Returns dict or None."""
    for version in [6, 5, 4, 3, 2, 1]:
        filename = f"AF-{uniprot_id}-F1-predicted_aligned_error_v{version}.json"
        url = f"{PAE_BASE_URL}/{filename}"
        try:
            response = requests.get(url, timeout=60)
            if response.status_code == 200:
                data = response.json()
                if isinstance(data, list) and len(data) > 0 and "predicted_aligned_error" in data[0]:
                    return data[0]["predicted_aligned_error"]
                elif isinstance(data, dict) and "predicted_aligned_error" in data:
                    return data["predicted_aligned_error"]
        except Exception:
            continue
    return None


def mean_pae_for_region(pae_matrix: list, indices: list):
    """Compute mean PAE for a set of residue indices (1-indexed)."""
    if not indices:
        return 0.0
    arr = np.array(pae_matrix)
    vals = []
    for i in indices:
        if 1 <= i <= len(arr):
            for j in indices:
                if 1 <= j <= len(arr):
                    vals.append(arr[i - 1, j - 1])
    return float(np.mean(vals)) if vals else 0.0


def mean_pae_cross_region(pae_matrix: list, indices_a: list, indices_b: list):
    """Compute mean PAE between two regions (1-indexed)."""
    if not indices_a or not indices_b:
        return 0.0
    arr = np.array(pae_matrix)
    vals = []
    for i in indices_a:
        if 1 <= i <= len(arr):
            for j in indices_b:
                if 1 <= j <= len(arr):
                    vals.append(arr[i - 1, j - 1])
    return float(np.mean(vals)) if vals else 0.0


def extract_pae_features(pae_matrix: list, tm_regions: list, loops: dict):
    """Extract ICL-related PAE features."""
    icl2 = loops.get("ICL2")
    icl3 = loops.get("ICL3")

    icl2_indices = list(range(icl2[0], icl2[1] + 1)) if icl2 else []
    icl3_indices = list(range(icl3[0], icl3[1] + 1)) if icl3 else []

    tm5_indices = list(range(tm_regions[4][0], tm_regions[4][1] + 1)) if len(tm_regions) >= 5 else []
    tm6_indices = list(range(tm_regions[5][0], tm_regions[5][1] + 1)) if len(tm_regions) >= 6 else []

    return {
        "icl2_mean_pae": mean_pae_for_region(pae_matrix, icl2_indices),
        "icl2_intra_pae": mean_pae_for_region(pae_matrix, icl2_indices),
        "icl3_mean_pae": mean_pae_for_region(pae_matrix, icl3_indices),
        "icl3_intra_pae": mean_pae_for_region(pae_matrix, icl3_indices),
        "icl2_tm5_pae": mean_pae_cross_region(pae_matrix, icl2_indices, tm5_indices),
        "icl2_tm6_pae": mean_pae_cross_region(pae_matrix, icl2_indices, tm6_indices),
        "icl3_tm5_pae": mean_pae_cross_region(pae_matrix, icl3_indices, tm5_indices),
        "icl3_tm6_pae": mean_pae_cross_region(pae_matrix, icl3_indices, tm6_indices),
    }


def main():
    print("=" * 70)
    print("  AlphaFold PAE 特征提取")
    print("=" * 70)

    with open(SEQUENCES_FILE, encoding="utf-8") as f:
        sequences = json.load(f)
    with open(TOPO_FILE, encoding="utf-8") as f:
        topo_data = json.load(f)

    # Filter to proteins with valid topology
    valid_uids = []
    for uid in sequences:
        if uid in topo_data:
            tm_regions = topo_data[uid].get("tm_regions", [])
            if len(tm_regions) >= 7:
                valid_uids.append(uid)

    print(f"[INFO] 有效拓扑蛋白数: {len(valid_uids)}")

    results = {}
    failed = []

    for uid in tqdm(valid_uids, desc="下载并提取 PAE"):
        pae_matrix = download_pae(uid)
        if pae_matrix is None:
            failed.append(uid)
            continue

        tm_regions = topo_data[uid]["tm_regions"]
        loops = topo_data[uid].get("loops", {})
        feats = extract_pae_features(pae_matrix, tm_regions, loops)
        results[uid] = feats

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n[OK] 成功提取: {len(results)} / {len(valid_uids)}")
    print(f"    失败: {len(failed)}")
    if failed:
        print(f"    失败列表 (前10): {failed[:10]}")
    print(f"    输出: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
