#!/usr/bin/env python3
"""本地提取100样本的ESM-2 CLS token特征。"""
import json
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm

import esm

BASE = Path(__file__).parent
DATA_FILE = BASE / "merged_dataset" / "extended_sequences.json"
OUTPUT_FILE = BASE / "server_sync" / "extended_data" / "features" / "esm_features_100samples_cls.json"


def main():
    print("=" * 60)
    print("本地提取 ESM-2 CLS token 特征")
    print("=" * 60)

    with open(DATA_FILE) as f:
        sequences = json.load(f)
    print(f"[INFO] 加载 {len(sequences)} 个序列")

    print("[INFO] 加载 ESM-2 模型 ...")
    model, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
    model = model.eval().cuda()
    batch_converter = alphabet.get_batch_converter()

    features = {}
    data = [(uid, seq["sequence"] if isinstance(seq, dict) else seq)
            for uid, seq in sequences.items()]
    batch_size = 8

    for i in tqdm(range(0, len(data), batch_size), desc="提取 CLS"):
        batch = data[i:i+batch_size]
        _, _, batch_tokens = batch_converter(batch)
        batch_tokens = batch_tokens.cuda()

        with torch.no_grad():
            results = model(batch_tokens, repr_layers=[6], return_contacts=False)

        token_representations = results["representations"][6].cpu().numpy()
        for j, (uid, _) in enumerate(batch):
            features[uid] = token_representations[j, 0, :].tolist()

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w") as f:
        json.dump(features, f)

    print(f"[OK] 已保存: {OUTPUT_FILE}，样本数: {len(features)}")


if __name__ == "__main__":
    main()
