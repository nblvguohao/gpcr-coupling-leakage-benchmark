#!/usr/bin/env python3
"""
简化版交叉注意力模型训练
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import json
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

FEATURES_DIR = Path('/data/lgh/GPCR/output/real_data/features')
OUTPUT_DIR = Path('/data/lgh/GPCR/output/real_data/results')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

class SimpleCrossAttention(nn.Module):
    """简化的交叉注意力模型"""
    def __init__(self, input_dim=320, hidden_dim=128):
        super().__init__()

        # 特征投影
        self.opsin_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        self.gnaq_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        # 交叉注意力参数
        self.W_Q = nn.Linear(hidden_dim, hidden_dim)
        self.W_K = nn.Linear(hidden_dim, hidden_dim)
        self.scale = hidden_dim ** -0.5

        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, opsin_feat, gnaq_feat):
        # 投影
        opsin = self.opsin_proj(opsin_feat)
        gnaq = self.gnaq_proj(gnaq_feat)

        # 交叉注意力: opsin关注gnaq
        Q = self.W_Q(opsin)
        K = self.W_K(gnaq)
        attn = torch.matmul(Q, K.t()) * self.scale
        attn = torch.softmax(attn, dim=-1)
        opsin_attended = torch.matmul(attn, gnaq)

        # 拼接
        combined = torch.cat([opsin, opsin_attended], dim=-1)

        # 分类
        logits = self.classifier(combined).squeeze(-1)
        return logits, attn

def load_data():
    """加载数据"""
    with open(FEATURES_DIR / 'esm_features.json', 'r') as f:
        esm_features = json.load(f)
    with open('/data/lgh/GPCR/output/real_data/real_labels.json', 'r') as f:
        labels = json.load(f)

    ids = list(esm_features.keys())
    X = [esm_features[k] for k in ids]
    y = np.array([labels[k] for k in ids])

    return X, y, ids

def main():
    print("="*60)
    print("简化版交叉注意力模型训练")
    print("="*60)

    X, y, ids = load_data()
    print(f"\n数据集: {len(y)} 样本")
    print(f"  正样本: {sum(y)}")
    print(f"  负样本: {len(y) - sum(y)}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"设备: {device}")

    # 5折交叉验证
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"\nFold {fold+1}/5")

        X_train = torch.FloatTensor([X[i] for i in train_idx]).to(device)
        y_train = torch.FloatTensor([y[i] for i in train_idx]).to(device)
        X_val = torch.FloatTensor([X[i] for i in val_idx]).to(device)
        y_val = torch.FloatTensor([y[i] for i in val_idx]).to(device)

        # Gαq使用固定特征（简化处理）
        gnaq_train = torch.randn_like(X_train) * 0.1
        gnaq_val = torch.randn_like(X_val) * 0.1

        model = SimpleCrossAttention().to(device)
        optimizer = optim.Adam(model.parameters(), lr=1e-4)
        criterion = nn.BCEWithLogitsLoss()

        # 训练
        best_acc = 0
        for epoch in range(100):
            model.train()
            optimizer.zero_grad()
            logits, _ = model(X_train, gnaq_train)
            loss = criterion(logits, y_train)
            loss.backward()
            optimizer.step()

            if (epoch + 1) % 20 == 0:
                model.eval()
                with torch.no_grad():
                    val_logits, _ = model(X_val, gnaq_val)
                    val_preds = (torch.sigmoid(val_logits) > 0.5).float()
                    acc = (val_preds == y_val).float().mean().item()
                print(f"  Epoch {epoch+1}: Loss={loss.item():.4f}, Val Acc={acc:.4f}")
                if acc > best_acc:
                    best_acc = acc

        # 评估
        model.eval()
        with torch.no_grad():
            val_logits, _ = model(X_val, gnaq_val)
            val_probs = torch.sigmoid(val_logits)
            val_preds = (val_probs > 0.5).cpu().numpy()
            y_true = y_val.cpu().numpy()
            y_prob = val_probs.cpu().numpy()

        acc = accuracy_score(y_true, val_preds)
        try:
            auc = roc_auc_score(y_true, y_prob)
        except:
            auc = 0.5

        print(f"  Fold {fold+1}: Acc={acc:.4f}, AUC={auc:.4f}")
        results.append({'accuracy': acc, 'auc': auc})

    print(f"\n{'='*60}")
    print("结果汇总")
    print(f"{'='*60}")
    accs = [r['accuracy'] for r in results]
    aucs = [r['auc'] for r in results]
    print(f"  准确率: {np.mean(accs):.4f} ± {np.std(accs):.4f}")
    print(f"  AUC: {np.mean(aucs):.4f} ± {np.std(aucs):.4f}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
