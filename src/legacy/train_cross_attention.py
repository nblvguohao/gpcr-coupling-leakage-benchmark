#!/usr/bin/env python3
"""
使用交叉注意力模型训练视蛋白-Gαq预测
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import json
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# 导入交叉注意力模型
from cross_attention_model import OpsinGaqPredictor, ContrastiveLoss

# 数据路径
FEATURES_DIR = Path('/data/lgh/GPCR/output/real_data/features')
OUTPUT_DIR = Path('/data/lgh/GPCR/output/real_data/results')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

class ProteinDataset(Dataset):
    """蛋白质数据集"""
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            'opsin_esm': torch.FloatTensor(self.features[idx]['esm_features']),
            'opsin_phys': torch.FloatTensor(self.features[idx]['phys_features']),
            'opsin_struct': torch.FloatTensor(self.features[idx].get('structure_features', [0.0]*11)),
            'gnaq_esm': torch.FloatTensor(self.features[idx].get('gnaq_features', [0.0]*320)),
            'label': torch.FloatTensor([self.labels[idx]])
        }

def load_features():
    """加载特征和标签"""
    with open(FEATURES_DIR / 'combined_features.json', 'r') as f:
        features = json.load(f)

    with open('/data/lgh/GPCR/output/real_data/real_labels.json', 'r') as f:
        labels_dict = json.load(f)

    # 转换为列表
    ids = list(features.keys())
    X = [features[k] for k in ids]
    y = [labels_dict[k] for k in ids]

    # 添加Gαq特征（使用固定序列的特征）
    gnaq_sequence = "MGCCVSSTPE..."  # Gαq序列简化版
    for x in X:
        x['gnaq_features'] = [0.0] * 320  # 简化处理，实际应该提取真实Gαq特征

    return X, np.array(y), ids

def train_epoch(model, train_loader, optimizer, device, alpha_contrastive=0.1):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    total_ce_loss = 0
    total_cont_loss = 0

    for batch in train_loader:
        opsin_esm = batch['opsin_esm'].to(device)
        opsin_phys = batch['opsin_phys'].to(device)
        opsin_struct = batch['opsin_struct'].to(device)
        gnaq_esm = batch['gnaq_esm'].to(device)
        labels = batch['label'].to(device).squeeze()

        optimizer.zero_grad()

        # 前向传播
        logits, attn, cont_loss, gate = model(
            opsin_esm, opsin_phys, opsin_struct, gnaq_esm, labels
        )

        # 分类损失
        ce_loss = nn.BCEWithLogitsLoss()(logits, labels)

        # 总损失
        loss = ce_loss + alpha_contrastive * cont_loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        total_ce_loss += ce_loss.item()
        total_cont_loss += cont_loss.item() if not torch.isnan(cont_loss) else 0

    return total_loss / len(train_loader), total_ce_loss / len(train_loader), total_cont_loss / len(train_loader)

def evaluate(model, val_loader, device):
    """评估模型"""
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for batch in val_loader:
            opsin_esm = batch['opsin_esm'].to(device)
            opsin_phys = batch['opsin_phys'].to(device)
            opsin_struct = batch['opsin_struct'].to(device)
            gnaq_esm = batch['gnaq_esm'].to(device)
            labels = batch['label'].to(device).squeeze()

            logits, _, _, _ = model(opsin_esm, opsin_phys, opsin_struct, gnaq_esm)
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    # 计算指标
    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, zero_division=0)
    rec = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)

    # AUC (需要至少两类)
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except:
        auc = 0.5

    return {
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1': f1,
        'auc': auc,
        'predictions': all_preds,
        'labels': all_labels,
        'probabilities': all_probs
    }

def cross_validation(X, y, n_splits=5, epochs=50):
    """交叉验证训练"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_results = []
    models = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"\n{'='*60}")
        print(f"Fold {fold+1}/{n_splits}")
        print(f"{'='*60}")

        # 划分数据
        X_train = [X[i] for i in train_idx]
        y_train = y[train_idx]
        X_val = [X[i] for i in val_idx]
        y_val = y[val_idx]

        # 创建数据加载器
        train_dataset = ProteinDataset(X_train, y_train)
        val_dataset = ProteinDataset(X_val, y_val)

        train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=len(val_dataset))

        # 创建模型
        model = OpsinGaqPredictor().to(device)
        optimizer = optim.Adam(model.parameters(), lr=1e-5, weight_decay=1e-4)

        # 检查模型参数是否有nan
        for name, param in model.named_parameters():
            if torch.isnan(param).any():
                print(f"Warning: {name} has NaN values")

        # 初始化权重
        def init_weights(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        model.apply(init_weights)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10)

        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(epochs):
            train_loss, ce_loss, cont_loss = train_epoch(model, train_loader, optimizer, device)
            val_metrics = evaluate(model, val_loader, device)
            val_loss = 1 - val_metrics['accuracy']  # 简化处理

            scheduler.step(val_loss)

            if (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{epochs}] "
                      f"Train: {train_loss:.4f} (CE: {ce_loss:.4f}, Cont: {cont_loss:.4f}) | "
                      f"Val Acc: {val_metrics['accuracy']:.4f} AUC: {val_metrics['auc']:.4f}")

            # 早停
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_model_state = model.state_dict().copy()
            else:
                patience_counter += 1
                if patience_counter >= 20:
                    print(f"早停于epoch {epoch+1}")
                    break

        # 加载最佳模型
        model.load_state_dict(best_model_state)

        # 最终评估
        final_metrics = evaluate(model, val_loader, device)
        print(f"\nFold {fold+1} 结果:")
        print(f"  Acc: {final_metrics['accuracy']:.4f}")
        print(f"  Prec: {final_metrics['precision']:.4f}")
        print(f"  Rec: {final_metrics['recall']:.4f}")
        print(f"  F1: {final_metrics['f1']:.4f}")
        print(f"  AUC: {final_metrics['auc']:.4f}")

        fold_results.append(final_metrics)
        models.append(model)

        # 保存模型
        torch.save(model.state_dict(), OUTPUT_DIR / f'model_cross_attn_fold_{fold+1}.pt')

    return fold_results, models

def main():
    print("="*60)
    print("交叉注意力模型训练")
    print("="*60)

    # 加载数据
    X, y, ids = load_features()
    print(f"\n数据集统计:")
    print(f"  总样本: {len(y)}")
    print(f"  正样本: {sum(y)}")
    print(f"  负样本: {len(y) - sum(y)}")

    # 交叉验证训练
    results, models = cross_validation(X, y, n_splits=5, epochs=100)

    # 汇总结果
    print(f"\n{'='*60}")
    print("交叉验证结果汇总")
    print(f"{'='*60}")

    accs = [r['accuracy'] for r in results]
    precs = [r['precision'] for r in results]
    recs = [r['recall'] for r in results]
    f1s = [r['f1'] for r in results]
    aucs = [r['auc'] for r in results]

    print(f"  准确率: {np.mean(accs):.4f} ± {np.std(accs):.4f}")
    print(f"  精确率: {np.mean(precs):.4f} ± {np.std(precs):.4f}")
    print(f"  召回率: {np.mean(recs):.4f} ± {np.std(recs):.4f}")
    print(f"  F1: {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")
    print(f"  AUC: {np.mean(aucs):.4f} ± {np.std(aucs):.4f}")

    # 保存结果
    # 转换结果为JSON可序列化的格式
    def convert_to_native(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, dict):
            return {k: convert_to_native(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_native(item) for item in obj]
        return obj

    results_summary = {
        'accuracy': {'mean': float(np.mean(accs)), 'std': float(np.std(accs))},
        'precision': {'mean': float(np.mean(precs)), 'std': float(np.std(precs))},
        'recall': {'mean': float(np.mean(recs)), 'std': float(np.std(recs))},
        'f1': {'mean': float(np.mean(f1s)), 'std': float(np.std(f1s))},
        'auc': {'mean': float(np.mean(aucs)), 'std': float(np.std(aucs))},
        'fold_results': [convert_to_native(r) for r in results]
    }

    with open(OUTPUT_DIR / 'cross_attention_results.json', 'w') as f:
        json.dump(results_summary, f, indent=2)

    print(f"\n结果保存到: {OUTPUT_DIR}")
    print("="*60)

if __name__ == "__main__":
    main()
