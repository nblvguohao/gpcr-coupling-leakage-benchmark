#!/usr/bin/env python3
"""
稳定的交叉注意力模型训练
简化版：只使用ESM特征 + 交叉注意力
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import json
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

# 路径
DATA_DIR = Path('/data/lgh/GPCR/output/real_data')
FEATURES_DIR = DATA_DIR / 'features'
OUTPUT_DIR = DATA_DIR / 'results'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


class SimpleCrossAttention(nn.Module):
    """简化版交叉注意力"""
    def __init__(self, dim=320, num_heads=4, dropout=0.1):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

        # 初始化
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.1)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, opsin_emb, gnaq_emb):
        batch_size = opsin_emb.size(0)

        Q = self.q_proj(opsin_emb).view(batch_size, self.num_heads, self.head_dim).transpose(0, 1)
        K = self.k_proj(gnaq_emb).view(batch_size, self.num_heads, self.head_dim).transpose(0, 1)
        V = self.v_proj(gnaq_emb).view(batch_size, self.num_heads, self.head_dim).transpose(0, 1)

        attn = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, V)

        out = out.transpose(0, 1).contiguous().view(batch_size, self.dim)
        out = self.out_proj(out)

        return out + opsin_emb, attn.mean(0)


class SimplePredictor(nn.Module):
    """简化版预测器"""
    def __init__(self, esm_dim=320, hidden_dim=128, num_heads=4, dropout=0.2):
        super().__init__()

        self.gnaq_encoder = nn.Sequential(
            nn.Linear(esm_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, esm_dim)
        )

        self.cross_attn = SimpleCrossAttention(esm_dim, num_heads, dropout)

        self.classifier = nn.Sequential(
            nn.Linear(esm_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, opsin_esm, gnaq_esm):
        gnaq_encoded = self.gnaq_encoder(gnaq_esm)
        opsin_attended, attn_weights = self.cross_attn(opsin_esm, gnaq_encoded)

        combined = torch.cat([opsin_attended, gnaq_encoded], dim=-1)
        logits = self.classifier(combined).squeeze(-1)

        return logits, attn_weights


class GPCRDataset(Dataset):
    def __init__(self, esm_features, labels):
        self.esm_features = esm_features
        self.labels = labels
        self.ids = list(esm_features.keys())

        # 计算Gαq特征
        positive_features = [
            esm_features[k] for k, label in zip(self.ids, labels.values()) if label == 1
        ]
        self.gnaq_feature = np.mean(positive_features, axis=0) if positive_features else np.zeros(320)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        uid = self.ids[idx]
        esm_feat = torch.tensor(self.esm_features[uid], dtype=torch.float32)
        gnaq_feat = torch.tensor(self.gnaq_feature, dtype=torch.float32)
        label = torch.tensor(self.labels[uid], dtype=torch.float32)

        return {
            'opsin_esm': esm_feat,
            'gnaq_esm': gnaq_feat,
            'label': label,
            'id': uid
        }


def train_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0

    for batch in dataloader:
        opsin_esm = batch['opsin_esm'].to(device)
        gnaq_esm = batch['gnaq_esm'].to(device)
        labels = batch['label'].to(device)

        optimizer.zero_grad()
        logits, _ = model(opsin_esm, gnaq_esm)
        loss = F.binary_cross_entropy_with_logits(logits, labels)

        # 检查NaN
        if torch.isnan(loss):
            print("Warning: NaN loss detected, skipping batch")
            continue

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def evaluate(model, dataloader, device):
    model.eval()
    all_labels = []
    all_probs = []
    all_preds = []

    with torch.no_grad():
        for batch in dataloader:
            opsin_esm = batch['opsin_esm'].to(device)
            gnaq_esm = batch['gnaq_esm'].to(device)
            labels = batch['label'].to(device)

            logits, _ = model(opsin_esm, gnaq_esm)
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()

            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    all_preds = np.array(all_preds)

    return {
        'accuracy': accuracy_score(all_labels, all_preds),
        'precision': precision_score(all_labels, all_preds, zero_division=0),
        'recall': recall_score(all_labels, all_preds, zero_division=0),
        'f1': f1_score(all_labels, all_preds, zero_division=0),
        'auc': roc_auc_score(all_labels, all_probs) if len(set(all_labels)) > 1 else 0.5
    }


def main():
    print("="*70)
    print("稳定版交叉注意力模型训练")
    print("="*70)

    # 加载数据
    with open(FEATURES_DIR / 'esm_features.json', 'r') as f:
        esm_features = json.load(f)
    with open(DATA_DIR / 'real_labels.json', 'r') as f:
        labels = json.load(f)

    print(f"\n数据集: {len(labels)} 个样本")
    print(f"  正样本: {sum(labels.values())}")
    print(f"  负样本: {len(labels) - sum(labels.values())}")

    ids = list(labels.keys())
    y = np.array([labels[k] for k in ids])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n使用设备: {device}")

    # 5折交叉验证
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_results = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(ids, y)):
        print(f"\n{'='*70}")
        print(f"Fold {fold + 1}/5")
        print(f"{'='*70}")

        train_ids = [ids[i] for i in train_idx]
        test_ids = [ids[i] for i in test_idx]

        train_esm = {k: esm_features[k] for k in train_ids}
        train_labels = {k: labels[k] for k in train_ids}
        test_esm = {k: esm_features[k] for k in test_ids}
        test_labels = {k: labels[k] for k in test_ids}

        train_dataset = GPCRDataset(train_esm, train_labels)
        test_dataset = GPCRDataset(test_esm, test_labels)

        train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, drop_last=False)
        test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

        model = SimplePredictor(esm_dim=320, hidden_dim=128, num_heads=4, dropout=0.2).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-5, weight_decay=1e-4)

        print(f"模型参数: {sum(p.numel() for p in model.parameters()):,}")
        print(f"训练样本: {len(train_dataset)}, 测试样本: {len(test_dataset)}")

        best_auc = 0
        best_state = None
        patience = 15
        patience_counter = 0

        for epoch in range(80):
            loss = train_epoch(model, train_loader, optimizer, device)

            if (epoch + 1) % 10 == 0:
                metrics = evaluate(model, test_loader, device)
                print(f"Epoch {epoch+1:3d}: Loss={loss:.4f}, AUC={metrics['auc']:.4f}, Acc={metrics['accuracy']:.4f}")

                if metrics['auc'] > best_auc:
                    best_auc = metrics['auc']
                    best_state = model.state_dict().copy()
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= patience:
                    print(f"早停: {epoch+1}轮")
                    break

        if best_state is not None:
            model.load_state_dict(best_state)

        final_metrics = evaluate(model, test_loader, device)
        fold_results.append(final_metrics)

        print(f"\nFold {fold+1} 最佳结果: AUC={final_metrics['auc']:.4f}, Acc={final_metrics['accuracy']:.4f}")
        torch.save(model.state_dict(), OUTPUT_DIR / f'cross_attn_stable_fold{fold+1}.pt')

    # 汇总
    print(f"\n{'='*70}")
    print("交叉验证汇总")
    print(f"{'='*70}")

    for metric in ['accuracy', 'precision', 'recall', 'f1', 'auc']:
        values = [r[metric] for r in fold_results]
        print(f"{metric:12s}: {np.mean(values):.4f} ± {np.std(values):.4f}")

    results = {
        'model': 'Stable Cross-Attention',
        'fold_results': fold_results,
        'summary': {
            metric: {'mean': float(np.mean([r[metric] for r in fold_results])),
                    'std': float(np.std([r[metric] for r in fold_results]))}
            for metric in ['accuracy', 'precision', 'recall', 'f1', 'auc']
        }
    }

    with open(OUTPUT_DIR / 'cross_attention_stable_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n结果保存到: {OUTPUT_DIR / 'cross_attention_stable_results.json'}")
    print("="*70)


if __name__ == "__main__":
    main()
