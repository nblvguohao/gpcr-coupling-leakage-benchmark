#!/usr/bin/env python3
"""
多模态特征融合训练 - ESM-2 + 结构特征
目标: 突破当前AUC 0.9072
"""
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = Path('/data/lgh/GPCR/extended_data')
FEATURE_DIR = Path('/data/lgh/GPCR/extended_data/features')
OUTPUT_DIR = Path('/data/lgh/GPCR/extended_results')
OUTPUT_DIR.mkdir(exist_ok=True)

class MultimodalGPCRDataset(Dataset):
    """多模态数据集: ESM-2 + 结构特征"""
    def __init__(self, esm_features, struct_features, labels, g_esm, g_struct):
        self.esm_features = torch.FloatTensor(esm_features)
        self.struct_features = torch.FloatTensor(struct_features)
        self.labels = torch.FloatTensor(labels)
        self.g_esm = torch.FloatTensor(g_esm)
        self.g_struct = torch.FloatTensor(g_struct)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (self.esm_features[idx], self.struct_features[idx],
                self.g_esm[idx], self.g_struct[idx], self.labels[idx])

class MultimodalCrossAttention(nn.Module):
    """多模态交叉注意力模型"""
    def __init__(self, esm_dim=320, struct_dim=16, hidden_dim=256,
                 num_heads=4, dropout=0.5, activation='gelu'):
        super().__init__()

        # ESM-2特征投影
        self.esm_embedding = nn.Sequential(
            nn.Linear(esm_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU() if activation == 'gelu' else nn.ReLU(),
            nn.Dropout(dropout)
        )

        # 结构特征投影
        self.struct_embedding = nn.Sequential(
            nn.Linear(struct_dim, hidden_dim // 4),
            nn.LayerNorm(hidden_dim // 4),
            nn.GELU() if activation == 'gelu' else nn.ReLU(),
            nn.Dropout(dropout)
        )

        # 融合层
        fusion_dim = hidden_dim + hidden_dim // 4
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU() if activation == 'gelu' else nn.ReLU(),
            nn.Dropout(dropout)
        )

        # 交叉注意力
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads,
            dropout=dropout, batch_first=True
        )

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU() if activation == 'gelu' else nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )

        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU() if activation == 'gelu' else nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p, gain=0.5)

    def forward(self, esm_feat, struct_feat, g_esm, g_struct):
        # 投影各模态特征
        esm_emb = self.esm_embedding(esm_feat).unsqueeze(1)  # (batch, 1, hidden)
        struct_emb = self.struct_embedding(struct_feat).unsqueeze(1)  # (batch, 1, hidden/4)

        g_esm_emb = self.esm_embedding(g_esm).unsqueeze(1)
        g_struct_emb = self.struct_embedding(g_struct).unsqueeze(1)

        # 拼接多模态特征
        combined = torch.cat([esm_emb, struct_emb], dim=-1)  # (batch, 1, hidden*1.25)
        g_combined = torch.cat([g_esm_emb, g_struct_emb], dim=-1)

        # 融合到统一维度
        x1 = self.fusion(combined)  # (batch, 1, hidden)
        x2 = self.fusion(g_combined)

        # 交叉注意力: GPCR作为query, G蛋白作为key/value
        attn_out, _ = self.attention(x1, x2, x2)
        x1 = self.norm1(x1 + attn_out)
        x1 = self.norm2(x1 + self.ffn(x1))

        # 拼接原始和注意力输出
        x = torch.cat([x1.squeeze(1), attn_out.squeeze(1)], dim=-1)

        return self.classifier(x).squeeze(-1)

def load_multimodal_features(use_cls_token=True):
    """加载多模态特征"""
    print("[INFO] 加载多模态特征...")

    # 1. 加载ESM-2特征
    if use_cls_token:
        esm_file = FEATURE_DIR / 'esm_features_100samples_cls.json'
    else:
        esm_file = FEATURE_DIR / 'esm_features_100samples.json'

    with open(esm_file, 'r') as f:
        esm_dict = json.load(f)

    # 2. 加载结构特征
    struct_file = FEATURE_DIR / 'structure_features_100samples.json'
    with open(struct_file, 'r') as f:
        struct_dict = json.load(f)

    # 3. 加载标签
    with open(DATA_DIR / 'extended_labels.json', 'r') as f:
        labels_dict = json.load(f)

    # 获取Gαq模板特征
    gq_id = [k for k, v in labels_dict.items() if v == 1][0]
    gq_esm = np.array(esm_dict[gq_id])
    gq_struct = np.array(list(struct_dict[gq_id].values()))

    if not use_cls_token:
        gq_esm = gq_esm.mean(axis=0)

    # 整合特征
    X_esm, X_struct, y_list, G_esm, G_struct = [], [], [], [], []

    for uid, label in labels_dict.items():
        if uid in esm_dict and uid in struct_dict:
            # ESM特征
            esm_feat = np.array(esm_dict[uid])
            if not use_cls_token:
                esm_feat = esm_feat.mean(axis=0)

            # 结构特征
            struct_feat = np.array(list(struct_dict[uid].values()))

            X_esm.append(esm_feat)
            X_struct.append(struct_feat)
            y_list.append(label)
            G_esm.append(gq_esm)
            G_struct.append(gq_struct)

    X_esm = np.array(X_esm)
    X_struct = np.array(X_struct)
    y = np.array(y_list)
    G_esm = np.array(G_esm)
    G_struct = np.array(G_struct)

    print(f"[OK] 加载完成: {len(y)} 个样本")
    print(f"    ESM特征维度: {X_esm.shape[1]}")
    print(f"    结构特征维度: {X_struct.shape[1]}")
    print(f"    总特征维度: {X_esm.shape[1] + X_struct.shape[1]}")
    print(f"    正样本: {y.sum()}, 负样本: {len(y) - y.sum()}")

    return X_esm, X_struct, y, G_esm, G_struct

def train_multimodal_model(X_esm, X_struct, y, G_esm, G_struct, params, device, n_splits=5):
    """训练多模态模型"""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    all_metrics = []

    print(f"\n[INFO] 多模态模型参数:")
    print(f"    hidden_dim={params['hidden_dim']}, dropout={params['dropout']}")
    print(f"    lr={params['learning_rate']}, activation={params['activation']}")

    for fold, (train_idx, test_idx) in enumerate(skf.split(X_esm, y)):
        print(f"\n[Fold {fold+1}/{n_splits}]")

        # 划分数据
        X_esm_train, X_esm_test = X_esm[train_idx], X_esm[test_idx]
        X_struct_train, X_struct_test = X_struct[train_idx], X_struct[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        G_esm_train, G_esm_test = G_esm[train_idx], G_esm[test_idx]
        G_struct_train, G_struct_test = G_struct[train_idx], G_struct[test_idx]

        # 从训练集划分验证集
        X_esm_train, X_esm_val, X_struct_train, X_struct_val, y_train, y_val, \
            G_esm_train, G_esm_val, G_struct_train, G_struct_val = train_test_split(
            X_esm_train, X_struct_train, y_train, G_esm_train, G_struct_train,
            test_size=0.15, random_state=42, stratify=y_train
        )

        # 数据加载器
        train_ds = MultimodalGPCRDataset(X_esm_train, X_struct_train, y_train,
                                          G_esm_train, G_struct_train)
        val_ds = MultimodalGPCRDataset(X_esm_val, X_struct_val, y_val,
                                        G_esm_val, G_struct_val)
        test_ds = MultimodalGPCRDataset(X_esm_test, X_struct_test, y_test,
                                         G_esm_test, G_struct_test)

        train_loader = DataLoader(train_ds, batch_size=params['batch_size'], shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=params['batch_size'])
        test_loader = DataLoader(test_ds, batch_size=params['batch_size'])

        # 模型
        model = MultimodalCrossAttention(
            esm_dim=X_esm.shape[1],
            struct_dim=X_struct.shape[1],
            hidden_dim=params['hidden_dim'],
            num_heads=params['num_heads'],
            dropout=params['dropout'],
            activation=params['activation']
        ).to(device)

        # 优化器
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=params['learning_rate'],
            weight_decay=params['weight_decay']
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=10, T_mult=2
        )
        criterion = nn.BCEWithLogitsLoss()

        # 训练
        best_val_auc = 0
        patience_counter = 0
        best_model_state = None

        for epoch in range(params['epochs']):
            model.train()
            for esm_f, struct_f, g_esm, g_struct, y_batch in train_loader:
                esm_f = esm_f.to(device)
                struct_f = struct_f.to(device)
                g_esm = g_esm.to(device)
                g_struct = g_struct.to(device)
                y_batch = y_batch.to(device)

                optimizer.zero_grad()
                logits = model(esm_f, struct_f, g_esm, g_struct)
                loss = criterion(logits, y_batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            scheduler.step()

            # 验证
            model.eval()
            val_preds, val_labels = [], []
            with torch.no_grad():
                for esm_f, struct_f, g_esm, g_struct, y_batch in val_loader:
                    logits = model(esm_f.to(device), struct_f.to(device),
                                  g_esm.to(device), g_struct.to(device))
                    val_preds.extend(torch.sigmoid(logits).cpu().numpy())
                    val_labels.extend(y_batch.numpy())

            val_auc = roc_auc_score(val_labels, val_preds)

            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_model_state = model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= 15:
                    break

        # 加载最佳模型并测试
        if best_model_state:
            model.load_state_dict(best_model_state)

        model.eval()
        test_preds, test_labels = [], []
        with torch.no_grad():
            for esm_f, struct_f, g_esm, g_struct, y_batch in test_loader:
                logits = model(esm_f.to(device), struct_f.to(device),
                              g_esm.to(device), g_struct.to(device))
                test_preds.extend(torch.sigmoid(logits).cpu().numpy())
                test_labels.extend(y_batch.numpy())

        preds_binary = (np.array(test_preds) > 0.5).astype(int)

        metrics = {
            'auc': roc_auc_score(test_labels, test_preds),
            'accuracy': accuracy_score(test_labels, preds_binary),
            'precision': precision_score(test_labels, preds_binary, zero_division=0),
            'recall': recall_score(test_labels, preds_binary, zero_division=0),
            'f1': f1_score(test_labels, preds_binary, zero_division=0)
        }

        all_metrics.append(metrics)
        print(f"  Test AUC: {metrics['auc']:.4f}, Acc: {metrics['accuracy']:.4f}, F1: {metrics['f1']:.4f}")

        # 保存模型
        torch.save(model.state_dict(), OUTPUT_DIR / f'model_multimodal_fold{fold+1}.pt')

    return all_metrics

def train_svm_multimodal(X_esm, X_struct, y, G_esm, G_struct):
    """训练SVM多模态基线"""
    print("\n" + "=" * 70)
    print("SVM多模态基线 - 拼接ESM-2 + 结构特征")
    print("=" * 70)

    # 拼接GPCR和G蛋白特征
    X_combined = np.concatenate([X_esm, X_struct, G_esm, G_struct], axis=1)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    all_metrics = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(X_combined, y)):
        X_train, X_test = X_combined[train_idx], X_combined[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # 标准化
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # 训练SVM (使用之前找到的最佳参数)
        svm = SVC(kernel='rbf', C=10.0, probability=True, random_state=42)
        svm.fit(X_train_scaled, y_train)

        # 预测
        y_proba = svm.predict_proba(X_test_scaled)[:, 1]
        y_pred = svm.predict(X_test_scaled)

        metrics = {
            'auc': roc_auc_score(y_test, y_proba),
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1': f1_score(y_test, y_pred, zero_division=0)
        }

        all_metrics.append(metrics)
        print(f"[Fold {fold+1}] AUC: {metrics['auc']:.4f}, Acc: {metrics['accuracy']:.4f}, F1: {metrics['f1']:.4f}")

    return all_metrics

def main():
    print("=" * 70)
    print("多模态特征融合训练 - ESM-2 + 结构特征")
    print("=" * 70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n[INFO] 设备: {device}")

    # 加载多模态特征
    X_esm, X_struct, y, G_esm, G_struct = load_multimodal_features(use_cls_token=True)

    # ========== 1. SVM多模态基线 ==========
    svm_metrics = train_svm_multimodal(X_esm, X_struct, y, G_esm, G_struct)

    svm_final = {
        'auc': {'mean': np.mean([m['auc'] for m in svm_metrics]),
                'std': np.std([m['auc'] for m in svm_metrics])},
        'accuracy': {'mean': np.mean([m['accuracy'] for m in svm_metrics]),
                     'std': np.std([m['accuracy'] for m in svm_metrics])},
        'f1': {'mean': np.mean([m['f1'] for m in svm_metrics]),
               'std': np.std([m['f1'] for m in svm_metrics])}
    }

    print("\n" + "=" * 70)
    print("SVM多模态 - 5折平均性能")
    print("=" * 70)
    print(f"  AUC: {svm_final['auc']['mean']:.4f} ± {svm_final['auc']['std']:.4f}")
    print(f"  Accuracy: {svm_final['accuracy']['mean']:.4f} ± {svm_final['accuracy']['std']:.4f}")
    print(f"  F1: {svm_final['f1']['mean']:.4f} ± {svm_final['f1']['std']:.4f}")

    # ========== 2. 多模态交叉注意力 ==========
    print("\n" + "=" * 70)
    print("多模态交叉注意力模型")
    print("=" * 70)

    # 使用超参数搜索找到的最佳配置
    best_params = {
        'learning_rate': 1e-4,
        'dropout': 0.5,
        'hidden_dim': 256,
        'num_heads': 4,
        'num_layers': 2,  # 实际实现中固定为2
        'batch_size': 8,
        'weight_decay': 1e-4,
        'activation': 'gelu',
        'epochs': 100
    }

    multimodal_metrics = train_multimodal_model(
        X_esm, X_struct, y, G_esm, G_struct, best_params, device, n_splits=5
    )

    multimodal_final = {
        'auc': {'mean': np.mean([m['auc'] for m in multimodal_metrics]),
                'std': np.std([m['auc'] for m in multimodal_metrics])},
        'accuracy': {'mean': np.mean([m['accuracy'] for m in multimodal_metrics]),
                     'std': np.std([m['accuracy'] for m in multimodal_metrics])},
        'f1': {'mean': np.mean([m['f1'] for m in multimodal_metrics]),
               'std': np.std([m['f1'] for m in multimodal_metrics])}
    }

    print("\n" + "=" * 70)
    print("多模态交叉注意力 - 5折平均性能")
    print("=" * 70)
    print(f"  AUC: {multimodal_final['auc']['mean']:.4f} ± {multimodal_final['auc']['std']:.4f}")
    print(f"  Accuracy: {multimodal_final['accuracy']['mean']:.4f} ± {multimodal_final['accuracy']['std']:.4f}")
    print(f"  F1: {multimodal_final['f1']['mean']:.4f} ± {multimodal_final['f1']['std']:.4f}")

    # ========== 3. 结果汇总 ==========
    print("\n" + "=" * 70)
    print("方法对比总结")
    print("=" * 70)
    print(f"\n1. SVM (仅ESM-2):      AUC = 0.9072 ± 0.0765 (基线)")
    print(f"2. SVM (ESM+Struct):   AUC = {svm_final['auc']['mean']:.4f} ± {svm_final['auc']['std']:.4f}")
    print(f"3. CrossAttn (ESM):    AUC = 0.8489 ± 0.0580")
    print(f"4. CrossAttn (Multi):  AUC = {multimodal_final['auc']['mean']:.4f} ± {multimodal_final['auc']['std']:.4f}")

    # 保存结果
    results = {
        'svm_multimodal': {
            'params': {'kernel': 'rbf', 'C': 10.0},
            'metrics': svm_final,
            'fold_results': svm_metrics
        },
        'crossattn_multimodal': {
            'params': best_params,
            'metrics': multimodal_final,
            'fold_results': multimodal_metrics
        }
    }

    with open(OUTPUT_DIR / 'multimodal_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n[OK] 结果保存: {OUTPUT_DIR / 'multimodal_results.json'}")

    # 判断是否突破目标
    best_auc = max(svm_final['auc']['mean'], multimodal_final['auc']['mean'])
    if best_auc > 0.92:
        print(f"\n🎉 目标达成! AUC突破0.92: {best_auc:.4f}")
    elif best_auc > 0.9072:
        print(f"\n✅ 有提升! AUC从0.9072提升到: {best_auc:.4f}")
    else:
        print(f"\n⚠️ 无显著改进。当前最佳: {best_auc:.4f}")

    print("=" * 70)

if __name__ == "__main__":
    main()
