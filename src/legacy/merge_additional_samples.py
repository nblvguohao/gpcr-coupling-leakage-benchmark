#!/usr/bin/env python3
"""
合并额外的3个负样本到原数据集
"""
import json
from pathlib import Path

# 路径
BASE_DIR = Path('E:/kimi/Kimi_Agent_批判者监督执行')
SERVER_SYNC_DIR = BASE_DIR / 'server_sync'
ADDITIONAL_FILE = BASE_DIR / 'additional_negative_samples.json'

def main():
    print("="*70)
    print("合并额外的负样本到原数据集")
    print("="*70)

    # 加载原有数据
    with open(SERVER_SYNC_DIR / 'real_sequences.json', 'r') as f:
        original_sequences = json.load(f)

    with open(SERVER_SYNC_DIR / 'real_labels.json', 'r') as f:
        original_labels = json.load(f)

    print(f"\n原有数据:")
    print(f"  序列数: {len(original_sequences)}")
    pos_count = sum(1 for v in original_labels.values() if v == 1)
    neg_count = sum(1 for v in original_labels.values() if v == 0)
    print(f"  正样本: {pos_count}")
    print(f"  负样本: {neg_count}")

    # 加载额外样本
    with open(ADDITIONAL_FILE, 'r') as f:
        additional = json.load(f)

    additional_sequences = additional['sequences']
    additional_labels = additional['labels']

    print(f"\n额外样本:")
    print(f"  新增负样本: {len(additional_labels)}")
    for uid, label in additional_labels.items():
        name = additional_sequences[uid]['gpcr_info']['name']
        print(f"    - {uid}: {name}")

    # 合并数据
    merged_sequences = {**original_sequences, **additional_sequences}
    merged_labels = {**original_labels, **additional_labels}

    print(f"\n合并后数据:")
    total = len(merged_labels)
    pos_count = sum(1 for v in merged_labels.values() if v == 1)
    neg_count = sum(1 for v in merged_labels.values() if v == 0)
    print(f"  总样本: {total}")
    print(f"  正样本: {pos_count} ({pos_count/total*100:.1f}%)")
    print(f"  负样本: {neg_count} ({neg_count/total*100:.1f}%)")

    # 保存合并后的数据
    with open(SERVER_SYNC_DIR / 'real_sequences.json', 'w') as f:
        json.dump(merged_sequences, f, indent=2)

    with open(SERVER_SYNC_DIR / 'real_labels.json', 'w') as f:
        json.dump(merged_labels, f, indent=2)

    # 同时保存到data_for_server用于上传
    with open(BASE_DIR / 'data_for_server' / 'real_sequences.json', 'w') as f:
        json.dump(merged_sequences, f, indent=2)

    print(f"\n[OK] 数据已保存:")
    print(f"  - {SERVER_SYNC_DIR / 'real_sequences.json'}")
    print(f"  - {SERVER_SYNC_DIR / 'real_labels.json'}")
    print(f"  - {BASE_DIR / 'data_for_server' / 'real_sequences.json'}")
    print("="*70)

if __name__ == "__main__":
    main()
