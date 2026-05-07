#!/usr/bin/env python3
"""
提取真实 human GNAQ (P50148) 的 ESM-2 特征。
生成 gnaq_esm_features.json，供后续实验替代"第一个正样本"的虚假 Gαq 特征。
"""
import json
import torch
import numpy as np
from pathlib import Path

import esm

BASE = Path(__file__).parent
GNAQ_SEQ_FILE = BASE / "gnaq_uniprot.json"
OUTPUT_FILE = BASE / "server_sync" / "extended_data" / "features" / "gnaq_esm_features.json"


def main():
    print("=" * 60)
    print("提取真实 GNAQ (P50148) ESM-2 特征")
    print("=" * 60)

    # 加载 GNAQ 序列
    with open(GNAQ_SEQ_FILE) as f:
        gnaq_data = json.load(f)
    sequence = gnaq_data["sequence"]
    print(f"[INFO] GNAQ 序列长度: {len(sequence)}")

    # 加载 ESM-2 模型 (与原文一致: esm2_t6_8M_UR50D)
    print("[INFO] 加载 ESM-2 模型 ...")
    model, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
    model = model.eval().cuda()
    batch_converter = alphabet.get_batch_converter()

    # 转换
    data = [("P50148", sequence)]
    _, _, batch_tokens = batch_converter(data)
    batch_tokens = batch_tokens.cuda()

    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[6], return_contacts=False)

    token_representations = results["representations"][6].cpu().numpy()
    # 去除 BOS/EOS
    seq_tokens = token_representations[0, 1:len(sequence) + 1, :]  # (L, 320)

    # Mean pooling & CLS token
    mean_feature = seq_tokens.mean(axis=0).tolist()
    cls_feature = token_representations[0, 0, :].tolist()

    output = {
        "uniprot_id": "P50148",
        "protein_name": gnaq_data["protein_name"],
        "sequence_length": len(sequence),
        "mean_pooling": mean_feature,
        "cls_token": cls_feature,
        "residue_level": seq_tokens.tolist(),
    }

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w") as f:
        json.dump(output, f, indent=2)

    print(f"[OK] 特征已保存: {OUTPUT_FILE}")
    print(f"    mean_pooling 维度: {len(mean_feature)}")
    print(f"    cls_token 维度: {len(cls_feature)}")


if __name__ == "__main__":
    main()
