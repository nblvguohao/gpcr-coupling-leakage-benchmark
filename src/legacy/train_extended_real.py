#!/usr/bin/env python3
"""
使用真实ESM-2特征训练交叉注意力模型 - 100样本
服务器端执行
"""
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# 路径
DATA_DIR = Path('/data/lgh/GPCR/extended_data')
FEATURE_DIR = Path('/data/lgh/GPCR/extended_data/features')
OUTPUT_DIR = Path('/data/lgh/GPCR/extended_results')
OUTPUT_DIR.mkdir(exist_ok=True)

# 最优参数
BEST_PARAMS = {
    'learning_rate': 1e-4,
    'dropout': 0.2,
    'hidden_dim': 320,
    'num_heads': 4,
    'num_layers': 3,
    'batch_size': 16,
    'epochs': 50
}

class CrossAttentionModel(nn.Module):
    """改进的交叉注意力模型"""
    def __init__(self, input_dim=320, hidden_dim=320, num_heads=4,
                 num_layers=3, dropout=0.2):
        super().__init__()

        self.embedding = nn.Linear(input_dim, hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_heads,
            dim_feedforward=hidden_dim*4, dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.attention_pool = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim//2), nn.Tanh(),
            nn.Linear(hidden_dim//2, 1)
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim//2), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(hidden_dim//2, 1)
        )
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1: nn.init.xavier_uniform_(p, gain=0.5)

    def forward(self, x1, x2):
        x1, x2 = self.embedding(x1), self.embedding(x2)
        x = torch.cat([x1, x2], dim=1)
        x = self.transformer(x)
        attn_weights = torch.softmax(self.attention_pool(x), dim=1)
        x = (x * attn_weights).sum(dim=1)
        return self.classifier(x).squeeze(-1)

class GPCRDataset(Dataset):
    def __init__(self, features, labels, g_features):
        self.features = torch.FloatTensor(features)
        self.labels = torch.FloatTensor(labels)
        self.g_features = torch.FloatTensor(g_features)
    def __len__(self): return len(self.labels)
    def __getitem__(self, idx):
        return self.features[idx], self.g_features[idx], self.labels[idx]

def load_real_features():
    """加载真实的ESM-2特征（对变长序列进行平均池化）"""
    print("[INFO] 加载真实ESM-2特征...")

    feature_file = FEATURE_DIR / 'esm_features_100samples.json'

    if not feature_file.exists():
        print("[ERROR] 特征文件不存在，请先运行 extract_esm_features_server.py")
        return None, None, None

    with open(feature_file, 'r') as f:
        features_dict = json.load(f)

    with open(DATA_DIR / 'extended_labels.json', 'r') as f:
        labels_dict = json.load(f)

    # 获取Gαq特征（使用第一个正样本）
    gq_id = [k for k, v in labels_dict.items() if v == 1][0]
    gq_feature_raw = np.array(features_dict[gq_id])
    # 平均池化得到固定长度特征
    gq_feature = gq_feature_raw.mean(axis=0)  # (320,)

    # 构建数据矩阵
    X_list, y_list, G_list = [], [], []

    for uid, label in labels_dict.items():
        if uid in features_dict:
            feat_raw = np.array(features_dict[uid])  # (seq_len, 320)
            # 平均池化: (seq_len, 320) -> (320,)
            feat = feat_raw.mean(axis=0)
            X_list.append(feat)
            y_list.append(label)
            G_list.append(gq_feature)

    X = np.array(X_list)  # (100, 320)
    y = np.array(y_list)  # (100,)
    G = np.array(G_list)  # (100, 320)

    # 扩展维度以适应交叉注意力模型: (batch, seq_len=1, dim)
    X = X[:, np.newaxis, :]  # (100, 1, 320)
    G = G[:, np.newaxis, :]  # (100, 1, 320)

    print(f"[OK] 加载完成: {len(y)} 个样本")
    print(f"    正样本: {y.sum()}")
    print(f"    负样本: {len(y) - y.sum()}")
    print(f"    特征形状: {X.shape}")

    return X, y, G

def train_fold(model, train_loader, val_loader, params, device):
    optimizer = torch.optim.AdamW(model.parameters(), lr=params['learning_rate'], weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=5, factor=0.5)
    criterion = nn.BCEWithLogitsLoss()

    best_val_auc = 0
    patience_counter = 0
    best_model_state = None

    for epoch in range(params['epochs']):
        model.train()
        for x1, x2, y in train_loader:
            x1, x2, y = x1.to(device), x2.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x1, x2), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        model.eval()
        val_preds, val_labels = [], []
        with torch.no_grad():
            for x1, x2, y in val_loader:
                logits = model(x1.to(device), x2.to(device))
                val_preds.extend(torch.sigmoid(logits).cpu().numpy())
                val_labels.extend(y.numpy())

        val_auc = roc_auc_score(val_labels, val_preds)
        scheduler.step(val_auc)

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= 10:
                print(f"    Early stopping at epoch {epoch+1}")
                break

        if (epoch + 1) % 10 == 0:
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
    print("100样本扩展数据集训练 - 使用真实ESM-2特征")
    print("=" * 70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n[INFO] 设备: {device}")

    # 加载真实特征
    features, labels, g_features = load_real_features()
    if features is None:
        print("[ERROR] 无法加载特征，退出")
        return

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

        # 划分验证集
        val_split = int(0.85 * len(X_train))
        X_train, X_val = X_train[:val_split], X_train[val_split:]
        y_train, y_val = y_train[:val_split], y_train[val_split:]
        g_train, g_val = g_train[:val_split], g_train[val_split:]

        # 数据加载器
        train_ds = GPCRDataset(X_train, y_train, g_train)
        val_ds = GPCRDataset(X_val, y_val, g_val)
        test_ds = GPCRDataset(X_test, y_test, g_test)

        train_loader = DataLoader(train_ds, batch_size=BEST_PARAMS['batch_size'], shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=BEST_PARAMS['batch_size'])
        test_loader = DataLoader(test_ds, batch_size=BEST_PARAMS['batch_size'])

        # 模型
        model = CrossAttentionModel(
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
        torch.save(trained_model.state_dict(), OUTPUT_DIR / f'model_extended_fold{fold+1}.pt')

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
    with open(OUTPUT_DIR / 'extended_100samples_results.json', 'w') as f:
        json.dump({'params': BEST_PARAMS, 'metrics': final_metrics, 'fold_results': all_metrics}, f, indent=2)

    print(f"\n[OK] 结果保存到: {OUTPUT_DIR / 'extended_100samples_results.json'}")
    print("=" * 70)

if __name__ == "__main__":
    main()
