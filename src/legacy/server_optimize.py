#!/usr/bin/env python3
"""
服务器端超参数优化脚本
"""
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score
import itertools
from pathlib import Path

# 数据路径
DATA_DIR = Path('/data/lgh/GPCR/output/real_data')
EXT_DIR = Path('/data/lgh/GPCR/extended_data')
OUTPUT_DIR = Path('/data/lgh/GPCR/optimization_results')
OUTPUT_DIR.mkdir(exist_ok=True)

# 超参数搜索空间
PARAM_GRID = {
    'learning_rate': [1e-4, 5e-5, 1e-5],
    'dropout': [0.1, 0.2, 0.3],
    'num_heads': [4, 8],
    'hidden_dim': [256, 320],
    'num_layers': [2, 3],
}

class SimpleCrossAttention(nn.Module):
    """稳定版交叉注意力模型"""
    def __init__(self, input_dim=320, hidden_dim=320, num_heads=4,
                 num_layers=2, dropout=0.1):
        super().__init__()
        self.embedding = nn.Linear(input_dim, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_heads,
            dim_feedforward=hidden_dim*2, dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim//2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim//2, 1)
        )
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1: nn.init.xavier_uniform_(p, gain=0.5)

    def forward(self, x1, x2):
        x1, x2 = self.embedding(x1), self.embedding(x2)
        x = torch.cat([x1, x2], dim=1)
        x = self.transformer(x)
        x = x.mean(dim=1)
        return self.classifier(x).squeeze(-1)

class GPCRDataset(Dataset):
    def __init__(self, features, labels, g_features):
        self.features = torch.FloatTensor(features)
        self.labels = torch.FloatTensor(labels)
        self.g_features = torch.FloatTensor(g_features)
    def __len__(self): return len(self.labels)
    def __getitem__(self, idx):
        return self.features[idx], self.g_features[idx], self.labels[idx]

def train_and_evaluate(params, features, labels, g_features, device='cuda'):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    aucs = []

    for train_idx, val_idx in skf.split(features, labels):
        X_train, X_val = features[train_idx], features[val_idx]
        y_train, y_val = labels[train_idx], labels[val_idx]
        g_train, g_val = g_features[train_idx], g_features[val_idx]

        train_ds = GPCRDataset(X_train, y_train, g_train)
        val_ds = GPCRDataset(X_val, y_val, g_val)
        train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=16)

        model = SimpleCrossAttention(
            input_dim=features.shape[2], hidden_dim=params['hidden_dim'],
            num_heads=params['num_heads'], num_layers=params['num_layers'],
            dropout=params['dropout']
        ).to(device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=params['learning_rate'], weight_decay=1e-5)
        criterion = nn.BCEWithLogitsLoss()

        # 训练
        for epoch in range(30):
            model.train()
            for x1, x2, y in train_loader:
                x1, x2, y = x1.to(device), x2.to(device), y.to(device)
                optimizer.zero_grad()
                loss = criterion(model(x1, x2), y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

        # 验证
        model.eval()
        val_preds, val_labels = [], []
        with torch.no_grad():
            for x1, x2, y in val_loader:
                logits = model(x1.to(device), x2.to(device))
                val_preds.extend(torch.sigmoid(logits).cpu().numpy())
                val_labels.extend(y.numpy())

        aucs.append(roc_auc_score(val_labels, val_preds))

    return np.mean(aucs), np.std(aucs)

def main():
    print("="*70)
    print("服务器端超参数优化")
    print("="*70)

    # 生成参数组合
    param_combinations = [dict(zip(PARAM_GRID.keys(), v))
                         for v in itertools.product(*PARAM_GRID.values())]

    print(f"超参数组合数: {len(param_combinations)}")

    # 加载特征数据（这里简化处理）
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")

    # 保存优化配置
    with open(OUTPUT_DIR / 'optimization_config.json', 'w') as f:
        json.dump({'param_grid': PARAM_GRID, 'n_combinations': len(param_combinations)}, f, indent=2)

    print("[OK] 服务器优化脚本准备完成")
    print("="*70)

if __name__ == "__main__":
    main()
