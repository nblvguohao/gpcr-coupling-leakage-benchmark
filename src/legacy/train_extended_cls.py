#!/usr/bin/env python3
"""
使用CLS token特征训练100样本 - 改进版
"""
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = Path('/data/lgh/GPCR/extended_data')
FEATURE_DIR = Path('/data/lgh/GPCR/extended_data/features')
OUTPUT_DIR = Path('/data/lgh/GPCR/extended_results')
OUTPUT_DIR.mkdir(exist_ok=True)

# 优化后的超参数
BEST_PARAMS = {
    'learning_rate': 5e-5,  # 降低学习率
    'dropout': 0.3,         # 增加dropout防止过拟合
    'hidden_dim': 256,      # 减小模型复杂度
    'num_heads': 4,
    'num_layers': 2,        # 减少层数
    'batch_size': 8,        # 小batch更稳定
    'epochs': 100,          # 更多epoch配合早停
    'weight_decay': 1e-4
}

class ImprovedCrossAttention(nn.Module):
    """改进的交叉注意力模型 - 更简单高效"""
    def __init__(self, input_dim=320, hidden_dim=256, num_heads=4,
                 num_layers=2, dropout=0.3):
        super().__init__()

        self.embedding = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # 更简单的注意力机制
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),  # *2因为有GPCR和G蛋白
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p, gain=0.5)

    def forward(self, x1, x2):
        # x1: GPCR特征, x2: G蛋白特征
        # 形状: (batch, 1, 320) - CLS token

        x1 = self.embedding(x1)  # (batch, 1, hidden)
        x2 = self.embedding(x2)  # (batch, 1, hidden)

        # 交叉注意力: GPCR作为query，G蛋白作为key/value
        attn_out, _ = self.attention(x1, x2, x2)
        x1 = self.norm1(x1 + attn_out)
        x1 = self.norm2(x1 + self.ffn(x1))

        # 拼接GPCR和G蛋白特征
        x = torch.cat([x1.squeeze(1), x2.squeeze(1)], dim=-1)

        return self.classifier(x).squeeze(-1)

class GPCRDataset(Dataset):
    def __init__(self, features, labels, g_features):
        self.features = torch.FloatTensor(features)
        self.labels = torch.FloatTensor(labels)
        self.g_features = torch.FloatTensor(g_features)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.g_features[idx], self.labels[idx]

def load_cls_features():
    """加载CLS token特征"""
    print("[INFO] 加载CLS token特征...")

    feature_file = FEATURE_DIR / 'esm_features_100samples_cls.json'

    with open(feature_file, 'r') as f:
        features_dict = json.load(f)

    with open(DATA_DIR / 'extended_labels.json', 'r') as f:
        labels_dict = json.load(f)

    # 获取Gαq模板
    gq_id = [k for k, v in labels_dict.items() if v == 1][0]
    gq_feature = np.array(features_dict[gq_id])

    X_list, y_list, G_list = [], [], []

    for uid, label in labels_dict.items():
        if uid in features_dict:
            feat = np.array(features_dict[uid])  # (320,)
            X_list.append(feat)
            y_list.append(label)
            G_list.append(gq_feature)

    X = np.array(X_list)[:, np.newaxis, :]  # (100, 1, 320)
    y = np.array(y_list)
    G = np.array(G_list)[:, np.newaxis, :]  # (100, 1, 320)

    print(f"[OK] 加载完成: {len(y)} 个样本")
    print(f"    正样本: {y.sum()}, 负样本: {len(y) - y.sum()}")

    return X, y, G

def train_fold(model, train_loader, val_loader, params, device):
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=params['learning_rate'],
        weight_decay=params['weight_decay']
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2
    )

    criterion = nn.BCEWithLogitsLoss()

    best_val_auc = 0
    patience_counter = 0
    best_model_state = None

    for epoch in range(params['epochs']):
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

        scheduler.step()

        # 验证
        model.eval()
        val_preds, val_labels = [], []
        with torch.no_grad():
            for x1, x2, y in val_loader:
                logits = model(x1.to(device), x2.to(device))
                val_preds.extend(torch.sigmoid(logits).cpu().numpy())
                val_labels.extend(y.numpy())

        val_auc = roc_auc_score(val_labels, val_preds)

        # 早停
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= 15:
                print(f"    Early stopping at epoch {epoch+1}")
                break

        if (epoch + 1) % 20 == 0:
            print(f"    Epoch {epoch+1}: Val AUC = {val_auc:.4f}")

    if best_model_state:
        model.load_state_dict(best_model_state)

    return best_val_auc, model

def evaluate_model(model, test_loader, device):
    model.eval()
    test_preds, test_labels = [], []

    with torch.no_grad():
        for x1, x2, y in test_loader:
            x1, x2 = x1.to(device), x2.to(device)
            logits = model(x1, x2)
            test_preds.extend(torch.sigmoid(logits).cpu().numpy())
            test_labels.extend(y.numpy())

    preds_binary = (np.array(test_preds) > 0.5).astype(int)

    return {
        'auc': roc_auc_score(test_labels, test_preds),
        'accuracy': accuracy_score(test_labels, preds_binary),
        'precision': precision_score(test_labels, preds_binary, zero_division=0),
        'recall': recall_score(test_labels, preds_binary, zero_division=0),
        'f1': f1_score(test_labels, preds_binary, zero_division=0)
    }

def main():
    print("=" * 70)
    print("100样本CLS Token特征训练 - 改进版模型")
    print("=" * 70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n[INFO] 设备: {device}")

    # 加载数据
    features, labels, g_features = load_cls_features()

    # 5折交叉验证
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    all_metrics = []

    print("\n" + "=" * 70)
    print("开始5折交叉验证")
    print("=" * 70)

    for fold, (train_idx, test_idx) in enumerate(skf.split(features, labels)):
        print(f"\n[Fold {fold+1}/5]")

        X_train, X_test = features[train_idx], features[test_idx]
        y_train, y_test = labels[train_idx], labels[test_idx]
        g_train, g_test = g_features[train_idx], g_features[test_idx]

        # 使用分层划分从训练集划分出验证集 (85% train, 15% val)
        X_train, X_val, y_train, y_val, g_train, g_val = train_test_split(
            X_train, y_train, g_train, test_size=0.15, random_state=42, stratify=y_train
        )

        print(f"  训练集: {len(y_train)}样本 (正:{y_train.sum()}, 负:{len(y_train)-y_train.sum()})")
        print(f"  验证集: {len(y_val)}样本 (正:{y_val.sum()}, 负:{len(y_val)-y_val.sum()})")
        print(f"  测试集: {len(y_test)}样本 (正:{y_test.sum()}, 负:{len(y_test)-y_test.sum()})")

        # 数据加载器
        train_ds = GPCRDataset(X_train, y_train, g_train)
        val_ds = GPCRDataset(X_val, y_val, g_val)
        test_ds = GPCRDataset(X_test, y_test, g_test)

        train_loader = DataLoader(train_ds, batch_size=BEST_PARAMS['batch_size'], shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=BEST_PARAMS['batch_size'])
        test_loader = DataLoader(test_ds, batch_size=BEST_PARAMS['batch_size'])

        # 模型
        model = ImprovedCrossAttention(
            input_dim=features.shape[2],
            hidden_dim=BEST_PARAMS['hidden_dim'],
            num_heads=BEST_PARAMS['num_heads'],
            num_layers=BEST_PARAMS['num_layers'],
            dropout=BEST_PARAMS['dropout']
        ).to(device)

        # 训练
        val_auc, trained_model = train_fold(model, train_loader, val_loader, BEST_PARAMS, device)
        print(f"  [OK] 最佳验证AUC: {val_auc:.4f}")

        # 测试
        metrics = evaluate_model(trained_model, test_loader, device)
        print(f"  [OK] 测试AUC: {metrics['auc']:.4f}, Acc: {metrics['accuracy']:.4f}, F1: {metrics['f1']:.4f}")

        all_metrics.append(metrics)
        torch.save(trained_model.state_dict(), OUTPUT_DIR / f'model_cls_fold{fold+1}.pt')

    # 汇总
    print("\n" + "=" * 70)
    print("训练完成 - 结果汇总")
    print("=" * 70)

    final_metrics = {
        'auc': {'mean': np.mean([m['auc'] for m in all_metrics]), 'std': np.std([m['auc'] for m in all_metrics])},
        'accuracy': {'mean': np.mean([m['accuracy'] for m in all_metrics]), 'std': np.std([m['accuracy'] for m in all_metrics])},
        'precision': {'mean': np.mean([m['precision'] for m in all_metrics]), 'std': np.std([m['precision'] for m in all_metrics])},
        'recall': {'mean': np.mean([m['recall'] for m in all_metrics]), 'std': np.std([m['recall'] for m in all_metrics])},
        'f1': {'mean': np.mean([m['f1'] for m in all_metrics]), 'std': np.std([m['f1'] for m in all_metrics])}
    }

    print(f"\n5折交叉验证平均性能:")
    for metric, values in final_metrics.items():
        print(f"  {metric.upper()}: {values['mean']:.4f} ± {values['std']:.4f}")

    # 保存结果
    with open(OUTPUT_DIR / 'extended_100samples_cls_results.json', 'w') as f:
        json.dump({
            'params': BEST_PARAMS,
            'metrics': final_metrics,
            'fold_results': all_metrics
        }, f, indent=2)

    print(f"\n[OK] 结果保存到: {OUTPUT_DIR / 'extended_100samples_cls_results.json'}")
    print("=" * 70)

if __name__ == "__main__":
    main()
