#!/usr/bin/env python3
"""
超参数搜索 - 100样本CLS Token特征
使用Optuna或Grid Search寻找最优参数
"""
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score
from pathlib import Path
import warnings
from itertools import product
warnings.filterwarnings('ignore')

DATA_DIR = Path('/data/lgh/GPCR/extended_data')
FEATURE_DIR = Path('/data/lgh/GPCR/extended_data/features')
OUTPUT_DIR = Path('/data/lgh/GPCR/extended_results')

def load_cls_features():
    """加载CLS token特征"""
    feature_file = FEATURE_DIR / 'esm_features_100samples_cls.json'

    with open(feature_file, 'r') as f:
        features_dict = json.load(f)

    with open(DATA_DIR / 'extended_labels.json', 'r') as f:
        labels_dict = json.load(f)

    gq_id = [k for k, v in labels_dict.items() if v == 1][0]
    gq_feature = np.array(features_dict[gq_id])

    X_list, y_list, G_list = [], [], []

    for uid, label in labels_dict.items():
        if uid in features_dict:
            feat = np.array(features_dict[uid])
            X_list.append(feat)
            y_list.append(label)
            G_list.append(gq_feature)

    X = np.array(X_list)[:, np.newaxis, :]
    y = np.array(y_list)
    G = np.array(G_list)[:, np.newaxis, :]

    return X, y, G

class GPCRDataset(Dataset):
    def __init__(self, features, labels, g_features):
        self.features = torch.FloatTensor(features)
        self.labels = torch.FloatTensor(labels)
        self.g_features = torch.FloatTensor(g_features)
    def __len__(self): return len(self.labels)
    def __getitem__(self, idx):
        return self.features[idx], self.g_features[idx], self.labels[idx]

class CrossAttentionModel(nn.Module):
    """可配置的交叉注意力模型"""
    def __init__(self, input_dim=320, hidden_dim=256, num_heads=4,
                 num_layers=2, dropout=0.3, activation='relu'):
        super().__init__()

        self.embedding = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU() if activation == 'relu' else nn.GELU(),
            nn.Dropout(dropout)
        )

        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads,
            dropout=dropout, batch_first=True
        )

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU() if activation == 'relu' else nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU() if activation == 'relu' else nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p, gain=0.5)

    def forward(self, x1, x2):
        x1 = self.embedding(x1)
        x2 = self.embedding(x2)
        attn_out, _ = self.attention(x1, x2, x2)
        x1 = self.norm1(x1 + attn_out)
        x1 = self.norm2(x1 + self.ffn(x1))
        x = torch.cat([x1.squeeze(1), x2.squeeze(1)], dim=-1)
        return self.classifier(x).squeeze(-1)

def train_and_evaluate(X, y, G, params, device, n_splits=5):
    """训练并评估一组参数"""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_aucs = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        g_train, g_test = G[train_idx], G[test_idx]

        # 划分验证集
        X_train, X_val, y_train, y_val, g_train, g_val = train_test_split(
            X_train, y_train, g_train, test_size=0.15, random_state=42, stratify=y_train
        )

        # 数据加载器
        train_ds = GPCRDataset(X_train, y_train, g_train)
        val_ds = GPCRDataset(X_val, y_val, g_val)
        test_ds = GPCRDataset(X_test, y_test, g_test)

        train_loader = DataLoader(train_ds, batch_size=params['batch_size'], shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=params['batch_size'])
        test_loader = DataLoader(test_ds, batch_size=params['batch_size'])

        # 模型
        model = CrossAttentionModel(
            input_dim=320, hidden_dim=params['hidden_dim'],
            num_heads=params['num_heads'], num_layers=params['num_layers'],
            dropout=params['dropout'], activation=params['activation']
        ).to(device)

        # 优化器
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=params['learning_rate'],
            weight_decay=params['weight_decay']
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=10, T_mult=2
        )
        criterion = nn.BCEWithLogitsLoss()

        # 训练
        best_val_auc = 0
        patience_counter = 0

        for epoch in range(params['epochs']):
            model.train()
            for x1, x2, y_batch in train_loader:
                x1, x2, y_batch = x1.to(device), x2.to(device), y_batch.to(device)
                optimizer.zero_grad()
                loss = criterion(model(x1, x2), y_batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            scheduler.step()

            # 验证
            model.eval()
            val_preds, val_labels = [], []
            with torch.no_grad():
                for x1, x2, y_batch in val_loader:
                    logits = model(x1.to(device), x2.to(device))
                    val_preds.extend(torch.sigmoid(logits).cpu().numpy())
                    val_labels.extend(y_batch.numpy())

            val_auc = roc_auc_score(val_labels, val_preds)

            if val_auc > best_val_auc:
                best_val_auc = val_auc
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= 15:
                    break

        # 测试
        model.eval()
        test_preds, test_labels = [], []
        with torch.no_grad():
            for x1, x2, y_batch in test_loader:
                logits = model(x1.to(device), x2.to(device))
                test_preds.extend(torch.sigmoid(logits).cpu().numpy())
                test_labels.extend(y_batch.numpy())

        test_auc = roc_auc_score(test_labels, test_preds)
        fold_aucs.append(test_auc)

    return np.mean(fold_aucs), np.std(fold_aucs)

def main():
    print("=" * 70)
    print("超参数搜索 - 100样本交叉注意力模型")
    print("=" * 70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n[INFO] 设备: {device}")

    X, y, G = load_cls_features()
    print(f"[INFO] 数据: {len(y)}样本")

    # 参数搜索空间
    param_grid = {
        'learning_rate': [1e-5, 5e-5, 1e-4],
        'dropout': [0.2, 0.3, 0.5],
        'hidden_dim': [128, 256, 512],
        'num_heads': [4, 8],
        'num_layers': [1, 2, 3],
        'batch_size': [8, 16],
        'weight_decay': [1e-4, 1e-5],
        'activation': ['relu', 'gelu'],
        'epochs': [100]
    }

    # 随机采样参数组合 (避免组合爆炸)
    import random
    random.seed(42)

    all_combinations = list(product(
        param_grid['learning_rate'],
        param_grid['dropout'],
        param_grid['hidden_dim'],
        param_grid['num_heads'],
        param_grid['num_layers'],
        param_grid['batch_size'],
        param_grid['weight_decay'],
        param_grid['activation']
    ))

    # 随机选择30组参数进行测试
    selected_combinations = random.sample(all_combinations, min(30, len(all_combinations)))

    print(f"\n[INFO] 总参数组合: {len(all_combinations)}, 测试: {len(selected_combinations)}")
    print("=" * 70)

    results = []

    for i, (lr, dropout, hidden, heads, layers, batch, wd, act) in enumerate(selected_combinations, 1):
        params = {
            'learning_rate': lr, 'dropout': dropout, 'hidden_dim': hidden,
            'num_heads': heads, 'num_layers': layers, 'batch_size': batch,
            'weight_decay': wd, 'activation': act, 'epochs': 100
        }

        print(f"\n[Config {i}/{len(selected_combinations)}]")
        print(f"  lr={lr}, dropout={dropout}, hidden={hidden}, heads={heads}, layers={layers}")

        mean_auc, std_auc = train_and_evaluate(X, y, G, params, device, n_splits=3)  # 3折加速

        print(f"  [结果] AUC: {mean_auc:.4f} ± {std_auc:.4f}")

        results.append({
            'params': params,
            'auc_mean': float(mean_auc),
            'auc_std': float(std_auc)
        })

    # 排序并显示最佳结果
    results.sort(key=lambda x: x['auc_mean'], reverse=True)

    print("\n" + "=" * 70)
    print("超参数搜索完成 - 前5名结果")
    print("=" * 70)

    for i, r in enumerate(results[:5], 1):
        p = r['params']
        print(f"\nTop {i}: AUC = {r['auc_mean']:.4f} ± {r['auc_std']:.4f}")
        print(f"  lr={p['learning_rate']}, dropout={p['dropout']}, hidden={p['hidden_dim']}")
        print(f"  heads={p['num_heads']}, layers={p['num_layers']}, batch={p['batch_size']}")
        print(f"  wd={p['weight_decay']}, activation={p['activation']}")

    # 保存结果
    with open(OUTPUT_DIR / 'hyperparam_search_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n[OK] 结果保存到: {OUTPUT_DIR / 'hyperparam_search_results.json'}")

    # 使用最佳参数完整训练5折
    print("\n" + "=" * 70)
    print("使用最佳参数进行完整5折交叉验证")
    print("=" * 70)

    best_params = results[0]['params']
    final_mean, final_std = train_and_evaluate(X, y, G, best_params, device, n_splits=5)

    print(f"\n[最终结果] AUC: {final_mean:.4f} ± {final_std:.4f}")

if __name__ == "__main__":
    main()
