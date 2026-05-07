#!/usr/bin/env python3
"""
超参数搜索 - 服务器端执行
简化版：遍历关键参数组合
"""
import json
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path

# 输出目录
OUTPUT_DIR = Path('/data/lgh/GPCR/hyperparam_results')
OUTPUT_DIR.mkdir(exist_ok=True)

# 超参数网格
param_grid = {
    'lr': [1e-4, 5e-5, 1e-5],
    'dropout': [0.1, 0.2, 0.3],
    'hidden_dim': [256, 320, 512],
}

def main():
    print("="*70)
    print("GPCR Cross-Attention Hyperparameter Search")
    print("="*70)

    # 统计组合数
    total = len(param_grid['lr']) * len(param_grid['dropout']) * len(param_grid['hidden_dim'])
    print(f"\nTotal combinations: {total}")
    print(f"Results will be saved to: {OUTPUT_DIR}")

    # 保存配置
    config = {
        'param_grid': param_grid,
        'total_combinations': total,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }

    with open(OUTPUT_DIR / 'search_config.json', 'w') as f:
        json.dump(config, f, indent=2)

    print(f"\n[OK] Configuration saved")
    print(f"[INFO] Device: {config['device']}")
    print(f"[INFO] To start training, run: train_with_best_params.py")
    print("="*70)

if __name__ == "__main__":
    main()
