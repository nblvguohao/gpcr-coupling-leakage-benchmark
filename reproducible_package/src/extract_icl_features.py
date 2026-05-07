#!/usr/bin/env python3
"""
Extract ICL2/3 local features from sequences and ESM-2 per-token embeddings.

Outputs:
- ICL2/3 physicochemical statistics (16-d)
- ICL2/3 local ESM embeddings (640-d)

Usage:
    python extract_icl_features.py \
        --sequences data/extended_sequences.json \
        --topology data/uniprot_topology.json \
        --esm-features data/gpcr_esm_features.json \
        --output data/icl_features.json
"""

import json
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, Tuple

KD_SCALE = {
    "A": 1.8, "C": 2.5, "D": -3.5, "E": -3.5, "F": 2.8,
    "G": -0.4, "H": -3.2, "I": 4.5, "K": -3.9, "L": 3.8,
    "M": 1.9, "N": -3.5, "P": -1.6, "Q": -3.5, "R": -4.5,
    "S": -0.8, "T": -0.7, "V": 4.2, "W": -0.9, "Y": -1.3,
}
POSITIVE = {"K", "R", "H"}
NEGATIVE = {"D", "E"}
HYDROPHOBIC = {"A", "V", "I", "L", "M", "F", "W", "G", "P", "C"}
AROMATIC = {"F", "W", "Y"}


def compute_loop_stats(seq: str, start: int, end: int) -> Dict[str, float]:
    """Compute physicochemical statistics for a loop region."""
    loop_seq = seq[start - 1:end]
    L = len(loop_seq)
    if L == 0:
        return {k: 0.0 for k in ["length", "mean_hydro", "std_hydro", "net_charge",
                                  "pos_charge_ratio", "neg_charge_ratio", "hydrophobic_ratio", "aromatic_ratio"]}

    hydro = [KD_SCALE.get(aa, 0.0) for aa in loop_seq]
    pos = sum(1 for aa in loop_seq if aa in POSITIVE)
    neg = sum(1 for aa in loop_seq if aa in NEGATIVE)
    hydrophobic = sum(1 for aa in loop_seq if aa in HYDROPHOBIC)
    aromatic = sum(1 for aa in loop_seq if aa in AROMATIC)

    return {
        "length": L,
        "mean_hydro": np.mean(hydro),
        "std_hydro": np.std(hydro) if L > 1 else 0.0,
        "net_charge": (pos - neg) / L,
        "pos_charge_ratio": pos / L,
        "neg_charge_ratio": neg / L,
        "hydrophobic_ratio": hydrophobic / L,
        "aromatic_ratio": aromatic / L,
    }


def extract_local_esm(esm_tokens: np.ndarray, start: int, end: int) -> np.ndarray:
    """Mean-pool ESM tokens within loop boundaries."""
    if start > end or end > len(esm_tokens):
        return np.zeros(esm_tokens.shape[1])
    return esm_tokens[start - 1:end].mean(axis=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequences", required=True)
    parser.add_argument("--topology", required=True)
    parser.add_argument("--esm-features", required=True, help="JSON with per-token ESM features")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    with open(args.sequences) as f:
        sequences = json.load(f)
    with open(args.topology) as f:
        topology = json.load(f)
    with open(args.esm_features) as f:
        esm_data = json.load(f)

    features = {}
    for uid, rec in sequences.items():
        seq = rec["sequence"] if isinstance(rec, dict) else rec
        topo = topology.get(uid, {})
        loops = topo.get("loops", {})

        icl2 = loops.get("ICL2", (0, 0))
        icl3 = loops.get("ICL3", (0, 0))

        icl2_stats = compute_loop_stats(seq, icl2[0], icl2[1])
        icl3_stats = compute_loop_stats(seq, icl3[0], icl3[1])

        # Try to get per-token ESM features for local embedding
        esm_tokens = None
        if uid in esm_data:
            val = esm_data[uid]
            if isinstance(val, dict) and "tokens" in val:
                esm_tokens = np.array(val["tokens"])
            elif isinstance(val, list) and isinstance(val[0], list):
                esm_tokens = np.array(val)

        if esm_tokens is not None and esm_tokens.ndim == 2:
            icl2_local = extract_local_esm(esm_tokens, icl2[0], icl2[1]).tolist()
            icl3_local = extract_local_esm(esm_tokens, icl3[0], icl3[1]).tolist()
        else:
            dim = 320  # Default for esm2_t6_8M
            icl2_local = [0.0] * dim
            icl3_local = [0.0] * dim

        features[uid] = {
            "ICL2_stats": icl2_stats,
            "ICL3_stats": icl3_stats,
            "ICL2_esm": icl2_local,
            "ICL3_esm": icl3_local,
        }

    with open(args.output, "w") as f:
        json.dump(features, f, indent=2)

    print(f"Extracted ICL features for {len(features)} sequences")


if __name__ == "__main__":
    main()
