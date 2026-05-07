#!/usr/bin/env python3
"""
为 paired_dataset 中所有 GPCR 提取 ESM-2 per-token 特征。

功能:
1. 读取 merged_dataset/extended_sequences.json (更新后 ~431 条)
2. 与已有特征 server_sync/extended_data/features/esm_features_100samples.json 比对
3. 仅对缺失序列运行 ESM-2 (esm2_t6_8M_UR50D)
4. 合并保存新的特征文件: server_sync/extended_data/features/esm_features_paired.json

要求: 本地已安装 fair-esm + PyTorch + CUDA
"""

import json
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm

import esm

BASE = Path(__file__).parent
SEQUENCES_FILE = BASE / "merged_dataset" / "extended_sequences.json"
EXISTING_FEATURES_FILE = BASE / "server_sync" / "extended_data" / "features" / "esm_features_100samples.json"
OUTPUT_FEATURES_FILE = BASE / "server_sync" / "extended_data" / "features" / "esm_features_100samples.json"
BATCH_SIZE = 1


def load_esm_model():
    print("[INFO] Loading ESM-2 model (esm2_t6_8M_UR50D) ...")
    model, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
    if torch.cuda.is_available():
        model = model.eval().cuda()
        print("[INFO] Using CUDA.")
    else:
        model = model.eval().cpu()
        print("[INFO] Using CPU (will be slow).")
    batch_converter = alphabet.get_batch_converter()
    return model, batch_converter


def extract_single(model, batch_converter, uid: str, seq: str):
    """Extract features for a single sequence; fallback to CPU on OOM."""
    _, _, batch_tokens = batch_converter([(uid, seq)])
    try:
        if torch.cuda.is_available():
            batch_tokens = batch_tokens.cuda()
        with torch.no_grad():
            results = model(batch_tokens, repr_layers=[6], return_contacts=False)
    except torch.OutOfMemoryError:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            batch_tokens = batch_tokens.cpu()
            with torch.no_grad():
                results = model(batch_tokens, repr_layers=[6], return_contacts=False)
        else:
            raise

    token_representations = results["representations"][6].cpu().numpy()
    seq_len = len(seq)
    features = token_representations[0, 1:seq_len+1, :]
    return features.tolist()


def extract_features(sequences_dict, model, batch_converter):
    features_dict = {}
    data = [(uid, seq["sequence"] if isinstance(seq, dict) else seq)
            for uid, seq in sequences_dict.items()]

    for uid, seq in tqdm(data, desc="Extracting ESM-2"):
        features_dict[uid] = extract_single(model, batch_converter, uid, seq)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return features_dict


def main():
    print("=" * 70)
    print("  ESM-2 Feature Extraction for Paired Dataset")
    print("=" * 70)

    with open(SEQUENCES_FILE, encoding="utf-8") as f:
        sequences = json.load(f)
    print(f"[INFO] Loaded {len(sequences)} sequences from {SEQUENCES_FILE}")

    existing_features = {}
    if EXISTING_FEATURES_FILE.exists():
        with open(EXISTING_FEATURES_FILE, encoding="utf-8") as f:
            existing_features = json.load(f)
        print(f"[INFO] Loaded {len(existing_features)} existing features from {EXISTING_FEATURES_FILE}")

    # Determine which sequences need feature extraction
    missing = {uid: seq for uid, seq in sequences.items() if uid not in existing_features}
    print(f"[INFO] Missing features: {len(missing)}")

    if not missing:
        print("[INFO] All sequences already have features. No extraction needed.")
        merged = existing_features
    else:
        model, batch_converter = load_esm_model()
        new_features = extract_features(missing, model, batch_converter)
        print(f"[INFO] Extracted features for {len(new_features)} new sequences.")

        merged = {**existing_features, **new_features}

    OUTPUT_FEATURES_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FEATURES_FILE, "w", encoding="utf-8") as f:
        json.dump(merged, f)
    print(f"[OK] Saved merged features ({len(merged)} total) to {OUTPUT_FEATURES_FILE}")

    # Validate
    sample_id = list(merged.keys())[0]
    sample_feat = np.array(merged[sample_id])
    print(f"[OK] Sample feature shape: {sample_feat.shape}")

    print("=" * 70)
    print("  Next steps:")
    print("  1. Update paired_cross_validation.py to use esm_features_paired.json")
    print("  2. Run paired_cross_validation.py")
    print("=" * 70)


if __name__ == "__main__":
    main()
