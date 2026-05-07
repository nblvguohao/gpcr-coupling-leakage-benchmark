#!/usr/bin/env python3
"""
批量提取所有 G 蛋白亚型的 ESM-2 特征。

覆盖 Gαq, Gαi1/2/3, Gαs, Gα12/13 的代表性人类序列。
自动从 UniProt 下载序列，复用本地 esm2_t6_8M_UR50D 模型。
输出格式与 gnaq_esm_features.json 保持一致，方便后续配对拼接。
"""

import json
import time
import torch
import numpy as np
import requests
from pathlib import Path
from typing import Dict, List

import esm

BASE = Path(__file__).parent
OUTPUT_DIR = BASE / "paired_dataset"
OUTPUT_DIR.mkdir(exist_ok=True)
OUTPUT_FILE = OUTPUT_DIR / "g_protein_esm_features.json"

# G 蛋白亚型 -> UniProt ID (人类)
G_PROTEINS = {
    "GNAQ": "P50148",   # Gαq
    "GNAI1": "P63096",  # Gαi1
    "GNAI2": "P04899",  # Gαi2
    "GNAI3": "P08754",  # Gαi3
    "GNAS": "P63092",   # Gαs (长期亚型)
    "GNA12": "Q03113",  # Gα12
    "GNA13": "Q14344",  # Gα13
}

UNIPROT_API = "https://rest.uniprot.org/uniprotkb/{uniprot_id}.json"


def fetch_sequence(uniprot_id: str) -> Dict[str, str]:
    """从 UniProt REST API 获取序列和元数据。"""
    url = UNIPROT_API.format(uniprot_id=uniprot_id)
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    data = response.json()

    # 序列字段可能在 data["sequence"]["sequence"] 或 ["value"]
    seq_node = data.get("sequence", {})
    sequence = seq_node.get("sequence") or seq_node.get("value", "")

    protein_name = ""
    if "proteinDescription" in data:
        rec_name = data["proteinDescription"].get("recommendedName", {})
        protein_name = rec_name.get("fullName", {}).get("value", "")

    return {
        "uniprot_id": uniprot_id,
        "protein_name": protein_name,
        "sequence": sequence,
        "sequence_length": len(sequence),
    }


def extract_esm_features(sequence: str, model, batch_converter, alphabet) -> Dict[str, List]:
    """利用本地 ESM-2 提取 mean-pooling、CLS token 和残基级特征。"""
    data = [("protein", sequence)]
    _, _, batch_tokens = batch_converter(data)
    batch_tokens = batch_tokens.cuda()

    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[6], return_contacts=False)

    token_representations = results["representations"][6].cpu().numpy()
    seq_tokens = token_representations[0, 1 : len(sequence) + 1, :]  # (L, 320)

    return {
        "mean_pooling": seq_tokens.mean(axis=0).tolist(),
        "cls_token": token_representations[0, 0, :].tolist(),
        "residue_level": seq_tokens.tolist(),
    }


def main():
    print("=" * 70)
    print("  批量提取 G 蛋白 ESM-2 特征")
    print("=" * 70)

    # 加载模型 (仅一次)
    print("[INFO] 加载 ESM-2 模型 (esm2_t6_8M_UR50D) ...")
    model, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
    model = model.eval().cuda()
    batch_converter = alphabet.get_batch_converter()

    all_features = {}

    for subtype, uniprot_id in G_PROTEINS.items():
        print(f"\n[INFO] 处理 {subtype} ({uniprot_id}) ...")
        try:
            info = fetch_sequence(uniprot_id)
            if not info["sequence"]:
                print(f"  [WARN] 未获取到序列，跳过。")
                continue

            print(f"  序列长度: {info['sequence_length']} | 名称: {info['protein_name'][:50]}")

            feats = extract_esm_features(info["sequence"], model, batch_converter, alphabet)

            all_features[subtype] = {
                "uniprot_id": uniprot_id,
                "protein_name": info["protein_name"],
                "sequence_length": info["sequence_length"],
                **feats,
            }
            print(f"  [OK] 特征维度: mean={len(feats['mean_pooling'])}, cls={len(feats['cls_token'])}")

        except Exception as e:
            print(f"  [ERROR] {e}")

        # 礼貌间隔，避免对 UniProt 请求过快
        time.sleep(0.5)

    # 保存结果
    with open(OUTPUT_FILE, "w") as f:
        json.dump(all_features, f, indent=2, ensure_ascii=False)

    print(f"\n[OK] 所有 G 蛋白特征已保存: {OUTPUT_FILE}")
    print(f"    成功提取: {len(all_features)} / {len(G_PROTEINS)}")


if __name__ == "__main__":
    main()
