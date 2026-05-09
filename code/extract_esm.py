#!/usr/bin/env python3
"""
为所有 GPCR 提取 esm2_t33_650M_UR50D 的 1280-d per-token ESM-2 特征。
输出: server_sync/extended_data/features/esm_features_650m_paired.json
"""

import json
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
import esm

BASE = Path(__file__).parent
SEQUENCES_FILE = BASE / "merged_dataset" / "extended_sequences.json"
OUTPUT_FEATURES_FILE = BASE / "server_sync" / "extended_data" / "features" / "esm_features_650m_paired.json"


def load_esm_model():
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


def extract_single(model, batch_converter, uid: str, seq: str):
    _, _, batch_tokens = batch_converter([(uid, seq)])
    try:
        if torch.cuda.is_available():
            batch_tokens = batch_tokens.cuda()
        with torch.no_grad():
            results = model(batch_tokens, repr_layers=[33], return_contacts=False)
    except torch.OutOfMemoryError:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            batch_tokens = batch_tokens.cpu()
            with torch.no_grad():
                results = model(batch_tokens, repr_layers=[33], return_contacts=False)
        else:
            raise

    token_representations = results["representations"][33].cpu().numpy()
    seq_len = len(seq)
    features = token_representations[0, 1:seq_len+1, :]
    return features.tolist()


def main():
    print("=" * 70)
    print("  ESM-2 650M Feature Extraction")
    print("=" * 70)

    with open(SEQUENCES_FILE, encoding="utf-8") as f:
        sequences = json.load(f)
    print(f"[INFO] Loaded {len(sequences)} sequences")

    model, batch_converter = load_esm_model()

    features_dict = {}
    for uid, seq in tqdm(list(sequences.items()), desc="Extracting 650M"):
        seq_str = seq["sequence"] if isinstance(seq, dict) else seq
        features_dict[uid] = extract_single(model, batch_converter, uid, seq_str)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    OUTPUT_FEATURES_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FEATURES_FILE, "w", encoding="utf-8") as f:
        json.dump(features_dict, f)
    print(f"[OK] Saved {len(features_dict)} features to {OUTPUT_FEATURES_FILE}")

    sample_id = list(features_dict.keys())[0]
    sample_feat = np.array(features_dict[sample_id])
    print(f"[OK] Sample feature shape: {sample_feat.shape}")


if __name__ == "__main__":
    main()
