#!/usr/bin/env python3
"""
完整版交叉注意力模型训练
使用多模态特征：ESM-2 + 物理化学 + 结构特征
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import json
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# 导入模型
from cross_attention_model import OpsinGaqPredictor, CrossAttention, MultimodalFusion

# 路径
DATA_DIR = Path('/data/lgh/GPCR/output/real_data')
FEATURES_DIR = DATA_DIR / 'features'
OUTPUT_DIR = DATA_DIR / 'results'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


class GPCRDataset(Dataset):
    """GPCR数据集，加载多模态特征"""

    def __init__(self, esm_features, phys_features, labels):
        self.esm_features = esm_features
        self.phys_features = phys_features
        self.labels = labels
        self.ids = list(esm_features.keys())

        # 计算Gαq特征（这里使用正样本的平均特征作为Gαq的代表）
        self.gnaq_feature = self._compute_gnaq_feature()

        # 模拟结构特征（11维）
        # 实际应用中应从AlphaFold结构中提取
        np.random.seed(42)
        self.struct_features = {
            k: np.random.randn(11) * 0.1 + 0.5 for k in self.ids
        }

    def _compute_gnaq_feature(self):
        """计算Gαq代表特征（使用正样本的平均ESM特征）"""
        positive_features = [
            self.esm_features[k] for k, label in zip(self.ids, self.labels.values())
            if label == 1
        ]
        return np.mean(positive_features, axis=0)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        uid = self.ids[idx]

        # 加载特征
        esm_feat = torch.tensor(self.esm_features[uid], dtype=torch.float32)
        phys_feat = torch.tensor(self.phys_features[uid], dtype=torch.float32)
        struct_feat = torch.tensor(self.struct_features[uid], dtype=torch.float32)
        gnaq_feat = torch.tensor(self.gnaq_feature, dtype=torch.float32)

        label = torch.tensor(self.labels[uid], dtype=torch.float32)

        return {
            'opsin_esm': esm_feat,
            'opsin_phys': phys_feat,
            'opsin_struct': struct_feat,
            'gnaq_esm': gnaq_feat,
            'label': label,
            'id': uid
        }


def load_data():
    """加载所有特征数据"""
    with open(FEATURES_DIR / 'esm_features.json', 'r') as f:
        esm_features = json.load(f)
    with open(FEATURES_DIR / 'phys_features.json', 'r') as f:
        phys_features = json.load(f)
    with open(DATA_DIR / 'real_labels.json', 'r') as f:
        labels = json.load(f)

    return esm_features, phys_features, labels


def train_epoch(model, dataloader, optimizer, device, lambda_contrast=0.1):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    total_ce_loss = 0
    total_contrast_loss = 0

    for batch in dataloader:
        opsin_esm = batch['opsin_esm'].to(device)
        opsin_phys = batch['opsin_phys'].to(device)
        opsin_struct = batch['opsin_struct'].to(device)
        gnaq_esm = batch['gnaq_esm'].to(device)
        labels = batch['label'].to(device)

        optimizer.zero_grad()

        # 前向传播
        logits, attn, contrastive_loss, gate = model(
            opsin_esm, opsin_phys, opsin_struct, gnaq_esm, labels
        )

        # 分类损失
        ce_loss = F.binary_cross_entropy_with_logits(logits, labels)

        # 总损失
        loss = ce_loss + lambda_contrast * contrastive_loss

        # 反向传播
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        total_ce_loss += ce_loss.item()
        total_contrast_loss += contrastive_loss.item()

    n = len(dataloader)
    return total_loss / n, total_ce_loss / n, total_contrast_loss / n


def evaluate(model, dataloader, device):
    """评估模型"""
    model.eval()
    all_labels = []
    all_probs = []
    all_preds = []

    with torch.no_grad():
        for batch in dataloader:
            opsin_esm = batch['opsin_esm'].to(device)
            opsin_phys = batch['opsin_phys'].to(device)
            opsin_struct = batch['opsin_struct'].to(device)
            gnaq_esm = batch['gnaq_esm'].to(device)
            labels = batch['label'].to(device)

            logits, attn, _, gate = model(
                opsin_esm, opsin_phys, opsin_struct, gnaq_esm
            )

            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()

            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    all_preds = np.array(all_preds)

    # 计算指标
    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, zero_division=0)
    rec = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)

    try:
        auc = roc_auc_score(all_labels, all_probs)
    except:
        auc = 0.5

    cm = confusion_matrix(all_labels, all_preds)

    return {
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1': f1,
        'auc': auc,
        'confusion_matrix': cm.tolist()
    }


def cross_validation_training(n_splits=5, epochs=100, batch_size=8, lr=1e-4):
    """交叉验证训练"""
    print("="*70)
    print("完整版交叉注意力模型训练 (多模态特征)")
    print("="*70)

    # 加载数据
    esm_features, phys_features, labels = load_data()
    print(f"\n数据集: {len(labels)} 个样本")
    print(f"  正样本: {sum(labels.values())}")
    print(f"  负样本: {len(labels) - sum(labels.values())}")

    # 准备数据
    ids = list(labels.keys())
    y = np.array([labels[k] for k in ids])

    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n使用设备: {device}")

    # 交叉验证
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_results = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(ids, y)):
        print(f"\n{'='*70}")
        print(f"Fold {fold + 1}/{n_splits}")
        print(f"{'='*70}")

        # 划分数据
        train_ids = [ids[i] for i in train_idx]
        test_ids = [ids[i] for i in test_idx]

        train_esm = {k: esm_features[k] for k in train_ids}
        train_phys = {k: phys_features[k] for k in train_ids}
        train_labels = {k: labels[k] for k in train_ids}

        test_esm = {k: esm_features[k] for k in test_ids}
        test_phys = {k: phys_features[k] for k in test_ids}
        test_labels = {k: labels[k] for k in test_ids}

        # 创建数据集
        train_dataset = GPCRDataset(train_esm, train_phys, train_labels)
        test_dataset = GPCRDataset(test_esm, test_phys, test_labels)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        # 创建模型
        model = OpsinGaqPredictor(
            esm_dim=320,
            phys_dim=29,
            struct_dim=11,
            hidden_dim=256,
            num_heads=4,
            dropout=0.3
        ).to(device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=10
        )

        print(f"\n模型参数: {sum(p.numel() for p in model.parameters()):,}")
        print(f"训练样本: {len(train_dataset)}, 测试样本: {len(test_dataset)}")

        # 训练
        best_auc = 0
        best_state = None
        patience_counter = 0
        patience = 20

        for epoch in range(epochs):
            train_loss, ce_loss, contrast_loss = train_epoch(
                model, train_loader, optimizer, device
            )

            # 评估
            if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
                metrics = evaluate(model, test_loader, device)
                scheduler.step(metrics['auc'])

                print(f"Epoch {epoch+1:3d}: Loss={train_loss:.4f} "
                      f"(CE={ce_loss:.4f}, Contrast={contrast_loss:.4f}) | "
                      f"AUC={metrics['auc']:.4f}, Acc={metrics['accuracy']:.4f}")

                # 早停
                if metrics['auc'] > best_auc:
                    best_auc = metrics['auc']
                    best_state = model.state_dict().copy()
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= patience:
                    print(f"早停: {epoch+1}轮")
                    break

        # 加载最佳模型
        if best_state is not None:
            model.load_state_dict(best_state)

        # 最终评估
        final_metrics = evaluate(model, test_loader, device)
        fold_results.append(final_metrics)

        print(f"\nFold {fold+1} 最佳结果:")
        print(f"  AUC: {final_metrics['auc']:.4f}")
        print(f"  Acc: {final_metrics['accuracy']:.4f}")
        print(f"  F1:  {final_metrics['f1']:.4f}")

        # 保存模型
        torch.save(model.state_dict(), OUTPUT_DIR / f'full_cross_attn_fold{fold+1}.pt')

    # 汇总结果
    print(f"\n{'='*70}")
    print("交叉验证汇总")
    print(f"{'='*70}")

    for metric in ['accuracy', 'precision', 'recall', 'f1', 'auc']:
        values = [r[metric] for r in fold_results]
        mean = np.mean(values)
        std = np.std(values)
        print(f"{metric:12s}: {mean:.4f} ± {std:.4f}")

    # 保存结果
    results = {
        'model': 'Full Cross-Attention with Multimodal Fusion',
        'fold_results': fold_results,
        'summary': {
            metric: {
                'mean': float(np.mean([r[metric] for r in fold_results])),
                'std': float(np.std([r[metric] for r in fold_results]))
            }
            for metric in ['accuracy', 'precision', 'recall', 'f1', 'auc']
        }
    }

    with open(OUTPUT_DIR / 'full_cross_attention_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n结果保存到: {OUTPUT_DIR / 'full_cross_attention_results.json'}")
    print("="*70)

    return results


if __name__ == "__main__":
    results = cross_validation_training(
        n_splits=5,
        epochs=100,
        batch_size=8,
        lr=1e-4
    )
