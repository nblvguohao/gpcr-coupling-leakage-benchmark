#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
真实数据模型训练
使用真实提取的特征训练深度学习模型
"""

import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime

# 设置路径
DATA_DIR = Path("/mnt/okcomputer/output/real_data")
FEATURES_DIR = DATA_DIR / "features"
STRUCT_FEATURES_DIR = DATA_DIR / "structure_features"
RESULTS_DIR = DATA_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

print("="*80)
print("🤖 真实数据模型训练")
print("="*80)

# ==============================================================================
# 第一部分：数据加载
# ==============================================================================

def load_real_data():
    """加载真实提取的特征数据"""
    
    print("\n📊 加载真实数据...")
    
    # 1. 加载序列数据和标签
    with open(DATA_DIR / "real_sequences.json", 'r') as f:
        sequences_data = json.load(f)
    
    # 2. 加载ESM特征
    with open(FEATURES_DIR / "combined_features.json", 'r') as f:
        features_data = json.load(f)
    
    # 3. 加载结构特征（如果有）
    struct_features = {}
    if (STRUCT_FEATURES_DIR / "structure_features.json").exists():
        with open(STRUCT_FEATURES_DIR / "structure_features.json", 'r') as f:
            struct_features = json.load(f)
    
    # 4. 构建数据集
    data_list = []
    
    for uniprot_id in features_data.keys():
        if uniprot_id in sequences_data:
            # 基础特征
            esm_feat = np.array(features_data[uniprot_id]['esm_features'])
            phys_feat = np.array(features_data[uniprot_id]['phys_features'])
            
            # 结构特征（如果有）
            struct_feat = []
            if uniprot_id in struct_features:
                sf = struct_features[uniprot_id]
                struct_feat = [
                    sf.get('plddt', {}).get('mean', 0),
                    sf.get('plddt', {}).get('high_confidence_ratio', 0),
                    sf.get('secondary_structure', {}).get('helix', 0),
                    sf.get('secondary_structure', {}).get('sheet', 0),
                    sf.get('sasa', {}).get('mean_sasa', 0),
                ]
            else:
                struct_feat = [0, 0, 0, 0, 0]
            
            # 组合所有特征
            combined_feat = np.concatenate([esm_feat, phys_feat, struct_feat])
            
            data_list.append({
                'uniprot_id': uniprot_id,
                'protein_name': sequences_data[uniprot_id]['name'],
                'features': combined_feat,
                'label': sequences_data[uniprot_id]['label'],
                'sequence_length': sequences_data[uniprot_id]['length']
            })
    
    df = pd.DataFrame(data_list)
    
    print(f"  ✅ 加载了 {len(df)} 个样本")
    print(f"  正样本 (Gq偶联): {sum(df['label'] == 1)}")
    print(f"  负样本 (非Gq偶联): {sum(df['label'] == 0)}")
    print(f"  特征维度: {len(df['features'].iloc[0])}")
    
    return df


# ==============================================================================
# 第二部分：模型定义
# ==============================================================================

class PPINet(nn.Module):
    """蛋白质相互作用预测网络"""
    
    def __init__(self, input_dim: int, hidden_dim: int = 512, dropout: float = 0.3):
        super(PPINet, self).__init__()
        
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.LayerNorm(hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim // 4, 2)
        )
    
    def forward(self, x):
        return self.classifier(x)


class ProteinDataset(Dataset):
    """蛋白质数据集"""
    
    def __init__(self, features: np.ndarray, labels: np.ndarray):
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


# ==============================================================================
# 第三部分：训练函数
# ==============================================================================

def train_model(model, train_loader, val_loader, epochs=100, lr=0.001, device='cpu'):
    """训练模型"""
    
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=15, factor=0.5)
    criterion = nn.CrossEntropyLoss()
    
    best_val_acc = 0
    best_model_state = None
    train_losses = []
    val_losses = []
    
    for epoch in range(epochs):
        # 训练
        model.train()
        train_loss = 0
        
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # 验证
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                
                logits = model(batch_x)
                loss = criterion(logits, batch_y)
                
                val_loss += loss.item()
                _, predicted = torch.max(logits, 1)
                val_total += batch_y.size(0)
                val_correct += (predicted == batch_y).sum().item()
        
        val_loss /= len(val_loader)
        val_acc = val_correct / val_total
        val_losses.append(val_loss)
        
        scheduler.step(val_loss)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        
        if (epoch + 1) % 20 == 0:
            print(f"  Epoch [{epoch+1}/{epochs}] Train Loss: {train_loss:.4f}, Val Acc: {val_acc:.4f}")
    
    # 加载最佳模型
    if best_model_state:
        model.load_state_dict(best_model_state)
    
    return model, train_losses, val_losses


def evaluate_model(model, X_test, y_test):
    """评估模型"""
    
    model.eval()
    with torch.no_grad():
        logits = model(torch.FloatTensor(X_test))
        probs = F.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1).numpy()
    
    # 计算指标
    acc = accuracy_score(y_test, preds)
    precision = precision_score(y_test, preds, zero_division=0)
    recall = recall_score(y_test, preds, zero_division=0)
    f1 = f1_score(y_test, preds, zero_division=0)
    auc = roc_auc_score(y_test, probs[:, 1].numpy())
    cm = confusion_matrix(y_test, preds)
    
    return {
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc,
        'confusion_matrix': cm,
        'predictions': preds,
        'probabilities': probs[:, 1].numpy()
    }


# ==============================================================================
# 第四部分：5-Fold交叉验证
# ==============================================================================

def cross_validation(X, y, n_splits=5, hidden_dim=512, dropout=0.3, lr=0.001, epochs=100):
    """5-Fold交叉验证"""
    
    print(f"\n📊 进行{n_splits}-Fold交叉验证...")
    
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    fold_results = []
    fold_models = []
    
    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        print(f"\n  Fold {fold+1}/{n_splits}:")
        
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # 划分验证集
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.15, random_state=42, stratify=y_train
        )
        
        # 创建数据加载器
        train_dataset = ProteinDataset(X_train, y_train)
        val_dataset = ProteinDataset(X_val, y_val)
        
        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, drop_last=True)
        val_loader = DataLoader(val_dataset, batch_size=len(X_val))
        
        # 创建和训练模型
        model = PPINet(input_dim=X.shape[1], hidden_dim=hidden_dim, dropout=dropout)
        model, _, _ = train_model(model, train_loader, val_loader, epochs=epochs, lr=lr)
        
        # 评估
        results = evaluate_model(model, X_test, y_test)
        fold_results.append(results)
        fold_models.append(model)
        
        print(f"    Test Acc: {results['accuracy']:.4f}, AUC: {results['auc']:.4f}")
    
    # 汇总结果
    print("\n" + "="*80)
    print("📊 交叉验证结果汇总")
    print("="*80)
    print(f"  准确率: {np.mean([r['accuracy'] for r in fold_results]):.4f} ± {np.std([r['accuracy'] for r in fold_results]):.4f}")
    print(f"  AUC:    {np.mean([r['auc'] for r in fold_results]):.4f} ± {np.std([r['auc'] for r in fold_results]):.4f}")
    print(f"  精确率: {np.mean([r['precision'] for r in fold_results]):.4f} ± {np.std([r['precision'] for r in fold_results]):.4f}")
    print(f"  召回率: {np.mean([r['recall'] for r in fold_results]):.4f} ± {np.std([r['recall'] for r in fold_results]):.4f}")
    print(f"  F1:     {np.mean([r['f1'] for r in fold_results]):.4f} ± {np.std([r['f1'] for r in fold_results]):.4f}")
    
    return fold_results, fold_models


# ==============================================================================
# 第五部分：主程序
# ==============================================================================

def main():
    """主程序"""
    
    print("\n" + "="*80)
    print("🚀 真实数据模型训练流程")
    print("="*80)
    
    # 1. 加载数据
    df = load_real_data()
    
    # 2. 准备特征和标签
    X = np.array(df['features'].tolist())
    y = df['label'].values
    
    print(f"\n📊 特征矩阵: {X.shape}")
    print(f"📊 标签分布: {np.bincount(y)}")
    
    # 3. 5-Fold交叉验证
    fold_results, fold_models = cross_validation(
        X, y, n_splits=5, 
        hidden_dim=512, dropout=0.3, 
        lr=0.001, epochs=100
    )
    
    # 4. 保存结果
    print("\n💾 保存结果...")
    
    # 保存交叉验证结果
    cv_results = {
        'accuracy_mean': float(np.mean([r['accuracy'] for r in fold_results])),
        'accuracy_std': float(np.std([r['accuracy'] for r in fold_results])),
        'auc_mean': float(np.mean([r['auc'] for r in fold_results])),
        'auc_std': float(np.std([r['auc'] for r in fold_results])),
        'precision_mean': float(np.mean([r['precision'] for r in fold_results])),
        'recall_mean': float(np.mean([r['recall'] for r in fold_results])),
        'f1_mean': float(np.mean([r['f1'] for r in fold_results])),
        'fold_details': [
            {
                'fold': i+1,
                'accuracy': r['accuracy'],
                'auc': r['auc'],
                'precision': r['precision'],
                'recall': r['recall'],
                'f1': r['f1']
            }
            for i, r in enumerate(fold_results)
        ]
    }
    
    with open(RESULTS_DIR / "cv_results.json", 'w') as f:
        json.dump(cv_results, f, indent=2)
    
    # 保存模型
    for i, model in enumerate(fold_models):
        torch.save(model.state_dict(), RESULTS_DIR / f"model_fold_{i+1}.pt")
    
    print(f"  ✅ 结果保存到: {RESULTS_DIR}/")
    
    # 5. 生成可视化
    print("\n📊 生成可视化...")
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 交叉验证结果
    ax1 = axes[0, 0]
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1', 'AUC']
    means = [cv_results['accuracy_mean'], cv_results['precision_mean'], 
             cv_results['recall_mean'], cv_results['f1_mean'], cv_results['auc_mean']]
    stds = [cv_results['accuracy_std'], 
            np.std([r['precision'] for r in fold_results]),
            np.std([r['recall'] for r in fold_results]),
            np.std([r['f1'] for r in fold_results]),
            cv_results['auc_std']]
    
    x_pos = np.arange(len(metrics))
    ax1.bar(x_pos, means, yerr=stds, capsize=5, alpha=0.7, color='steelblue', edgecolor='black')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(metrics)
    ax1.set_ylabel('Score')
    ax1.set_title('5-Fold Cross-Validation Results', fontweight='bold')
    ax1.set_ylim(0, 1.1)
    ax1.axhline(y=0.8, color='red', linestyle='--', label='Target (0.8)')
    ax1.legend()
    
    # 各Fold表现
    ax2 = axes[0, 1]
    fold_nums = [f'Fold {i+1}' for i in range(5)]
    accuracies = [r['accuracy'] for r in fold_results]
    ax2.bar(fold_nums, accuracies, color='coral', alpha=0.7, edgecolor='black')
    ax2.axhline(y=np.mean(accuracies), color='blue', linestyle='--', linewidth=2, 
                label=f'Mean: {np.mean(accuracies):.3f}')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Individual Fold Performance', fontweight='bold')
    ax2.legend()
    ax2.set_ylim(0, 1.1)
    
    # 混淆矩阵（最后一个fold）
    ax3 = axes[1, 0]
    cm = fold_results[-1]['confusion_matrix']
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax3,
                xticklabels=['Non-Gq', 'Gq'],
                yticklabels=['Non-Gq', 'Gq'])
    ax3.set_xlabel('Predicted')
    ax3.set_ylabel('True')
    ax3.set_title('Confusion Matrix (Last Fold)', fontweight='bold')
    
    # ROC曲线（最后一个fold）
    ax4 = axes[1, 1]
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(y, fold_results[-1]['probabilities'])
    ax4.plot(fpr, tpr, linewidth=2, label=f"AUC = {fold_results[-1]['auc']:.3f}")
    ax4.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    ax4.set_xlabel('False Positive Rate')
    ax4.set_ylabel('True Positive Rate')
    ax4.set_title('ROC Curve', fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "training_results.png", dpi=300, bbox_inches='tight')
    print(f"  ✅ 可视化保存到: {RESULTS_DIR / 'training_results.png'}")
    
    plt.close()
    
    print("\n" + "="*80)
    print("✅ 模型训练完成!")
    print("="*80)
    
    return cv_results, fold_models


if __name__ == "__main__":
    results, models = main()
