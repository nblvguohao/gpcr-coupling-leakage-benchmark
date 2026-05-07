#!/usr/bin/env python3
"""
交叉注意力模型超参数优化
目标: AUC>0.88, 测试准确率>80%
"""
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
import itertools
from pathlib import Path

# 数据路径
DATA_DIR = Path('E:/kimi/Kimi_Agent_批判者监督执行/merged_dataset')
OUTPUT_DIR = Path('E:/kimi/Kimi_Agent_批判者监督执行/optimization_results')
OUTPUT_DIR.mkdir(exist_ok=True)

# 超参数搜索空间
PARAM_GRID = {
    'learning_rate': [1e-4, 5e-5, 1e-5],
    'dropout': [0.1, 0.2, 0.3],
    'num_heads': [4, 8],
    'hidden_dim': [256, 320, 512],
    'num_layers': [2, 3, 4],
}

class SimpleCrossAttention(nn.Module):
    """稳定版交叉注意力模型"""
    def __init__(self, input_dim=320, hidden_dim=320, num_heads=4,
                 num_layers=2, dropout=0.1):
        super().__init__()

        self.embedding = nn.Linear(input_dim, hidden_dim)

        # 交叉注意力层
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim*2,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim//2, 1)
        )

        # 初始化
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p, gain=0.5)

    def forward(self, x1, x2):
        # x1, x2: [batch, seq_len, features]
        x1 = self.embedding(x1)
        x2 = self.embedding(x2)

        # 拼接序列
        x = torch.cat([x1, x2], dim=1)

        # Transformer
        x = self.transformer(x)

        # 全局平均池化
        x = x.mean(dim=1)

        # 分类
        logits = self.classifier(x)
        return logits.squeeze(-1)

class GPCRDataset(Dataset):
    def __init__(self, features, labels, g_protein_features):
        self.features = torch.FloatTensor(features)
        self.labels = torch.FloatTensor(labels)
        self.g_features = torch.FloatTensor(g_protein_features)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.g_features[idx], self.labels[idx]

def train_model(model, train_loader, val_loader, lr, epochs=30, device='cuda'):
    """训练模型"""
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)
    criterion = nn.BCEWithLogitsLoss()

    best_val_auc = 0
    patience_counter = 0

    for epoch in range(epochs):
        # 训练
        model.train()
        train_loss = 0
        for x1, x2, y in train_loader:
            x1, x2, y = x1.to(device), x2.to(device), y.to(device)

            optimizer.zero_grad()
            logits = model(x1, x2)
            loss = criterion(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item()

        # 验证
        model.eval()
        val_preds, val_labels = [], []
        with torch.no_grad():
            for x1, x2, y in val_loader:
                x1, x2 = x1.to(device), x2.to(device)
                logits = model(x1, x2)
                probs = torch.sigmoid(logits)
                val_preds.extend(probs.cpu().numpy())
                val_labels.extend(y.numpy())

        val_auc = roc_auc_score(val_labels, val_preds)
        scheduler.step(val_auc)

        # 早停
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= 10:
                break

    return best_val_auc

def evaluate_params(params, features, labels, g_features, n_splits=5):
    """评估一组超参数"""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    aucs = []

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    for fold, (train_idx, val_idx) in enumerate(skf.split(features, labels)):
        X_train, X_val = features[train_idx], features[val_idx]
        y_train, y_val = labels[train_idx], labels[val_idx]
        g_train, g_val = g_features[train_idx], g_features[val_idx]

        train_ds = GPCRDataset(X_train, y_train, g_train)
        val_ds = GPCRDataset(X_val, y_val, g_val)

        train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=16)

        model = SimpleCrossAttention(
            input_dim=features.shape[2],
            hidden_dim=params['hidden_dim'],
            num_heads=params['num_heads'],
            num_layers=params['num_layers'],
            dropout=params['dropout']
        )

        val_auc = train_model(model, train_loader, val_loader,
                             params['learning_rate'], device=device)
        aucs.append(val_auc)

    return np.mean(aucs), np.std(aucs)

def main():
    print("=" * 70)
    print("交叉注意力模型超参数优化")
    print("=" * 70)

    # 生成参数组合
    param_combinations = [
        dict(zip(PARAM_GRID.keys(), v))
        for v in itertools.product(*PARAM_GRID.values())
    ]

    print(f"\n超参数组合数: {len(param_combinations)}")
    print("\n搜索空间:")
    for k, v in PARAM_GRID.items():
        print(f"  {k}: {v}")

    # 这里简化处理，使用模拟数据演示
    print("\n[NOTE] 由于特征数据在服务器上，此处创建优化框架")
    print("[NOTE] 实际运行需要在服务器上执行完整优化")

    # 保存优化配置
    with open(OUTPUT_DIR / 'optimization_config.json', 'w') as f:
        json.dump({
            'param_grid': PARAM_GRID,
            'n_combinations': len(param_combinations),
            'target_auc': 0.88,
            'target_accuracy': 0.80
        }, f, indent=2)

    print(f"\n{'=' * 70}")
    print("[OK] 优化配置已保存")
    print(f"[OK] 配置文件: {OUTPUT_DIR / 'optimization_config.json'}")
    print("=" * 70)

if __name__ == "__main__":
    main()
