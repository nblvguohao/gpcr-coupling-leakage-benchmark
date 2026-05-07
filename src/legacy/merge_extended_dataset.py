#!/usr/bin/env python3
"""
合并扩展数据集到主数据集
目标: 从53个扩展到100个样本
"""
import json
from pathlib import Path

# 路径设置
EXT_DIR = Path('E:/kimi/Kimi_Agent_批判者监督执行/extended_samples')
SERVER_DIR = Path('E:/kimi/Kimi_Agent_批判者监督执行/server_sync/output/real_data')
OUTPUT_DIR = Path('E:/kimi/Kimi_Agent_批判者监督执行/merged_dataset')
OUTPUT_DIR.mkdir(exist_ok=True)

def main():
    print("=" * 70)
    print("合并扩展数据集")
    print("=" * 70)

    # 加载现有数据
    with open(SERVER_DIR / 'real_sequences.json', 'r') as f:
        existing_seqs = json.load(f)

    with open(SERVER_DIR / 'real_labels.json', 'r') as f:
        existing_labels = json.load(f)

    print(f"\n现有数据集: {len(existing_seqs)}个样本")
    pos_existing = sum(1 for l in existing_labels.values() if l == 1)
    neg_existing = sum(1 for l in existing_labels.values() if l == 0)
    print(f"  正样本(Gq): {pos_existing}")
    print(f"  负样本(Gi/o+Gs): {neg_existing}")

    # 加载新数据
    with open(EXT_DIR / 'additional_samples.json', 'r') as f:
        new_seqs = json.load(f)

    with open(EXT_DIR / 'additional_labels.json', 'r') as f:
        new_labels = json.load(f)

    print(f"\n新增数据: {len(new_seqs)}个样本")
    pos_new = sum(1 for l in new_labels.values() if l == 1)
    neg_new = sum(1 for l in new_labels.values() if l == 0)
    print(f"  正样本(Gq): {pos_new}")
    print(f"  负样本(Gi/o+Gs): {neg_new}")

    # 合并
    merged_seqs = {**existing_seqs, **new_seqs}
    merged_labels = {**existing_labels, **new_labels}

    print(f"\n合并后总数: {len(merged_seqs)}个样本")
    pos_total = sum(1 for l in merged_labels.values() if l == 1)
    neg_total = sum(1 for l in merged_labels.values() if l == 0)
    print(f"  正样本(Gq): {pos_total}")
    print(f"  负样本(Gi/o+Gs): {neg_total}")

    # 保存合并数据
    with open(OUTPUT_DIR / 'extended_sequences.json', 'w') as f:
        json.dump(merged_seqs, f, indent=2)

    with open(OUTPUT_DIR / 'extended_labels.json', 'w') as f:
        json.dump(merged_labels, f, indent=2)

    # 创建数据摘要
    summary = {
        'total_samples': len(merged_seqs),
        'original_samples': len(existing_seqs),
        'added_samples': len(new_seqs),
        'positive_samples': pos_total,
        'negative_samples': neg_total,
        'positive_ratio': pos_total / len(merged_seqs),
        'target_achieved': len(merged_seqs) >= 100
    }

    with open(OUTPUT_DIR / 'dataset_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'=' * 70}")
    print(f"[OK] 合并完成!")
    print(f"[OK] 数据集已扩展到 {len(merged_seqs)} 个样本")
    print(f"[OK] 1区期刊标准(>=100): {'已达成' if len(merged_seqs) >= 100 else '未达成'}")
    print(f"[OK] 结果保存到: {OUTPUT_DIR}")
    print("=" * 70)

if __name__ == "__main__":
    main()
