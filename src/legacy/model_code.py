#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
蛋白质相互作用预测 - GNAQ与Opsin4相互作用分析
Protein-Protein Interaction Prediction: GNAQ and Opsin4 Interaction Analysis

作者: Bioinformatics AI Assistant
日期: 2025
描述: 使用ESM-2预训练模型和深度学习预测GNAQ与Opsin4蛋白的相互作用
"""

import os
import re
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from Bio import SeqIO
from Bio.SeqUtils import ProtParam
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold, LeaveOneOut
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                            f1_score, roc_auc_score, roc_curve, confusion_matrix,
                            classification_report)
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics.pairwise import cosine_similarity
import esm
import warnings
warnings.filterwarnings('ignore')

# 设置随机种子确保可复现
np.random.seed(42)
torch.manual_seed(42)

# ============================================
# 常量定义
# ============================================
AMINO_ACIDS = 'ACDEFGHIKLMNPQRSTVWY'
AA_PROPERTIES = {
    'A': {'hydrophobicity': 1.8, 'size': 67, 'charge': 0},
    'C': {'hydrophobicity': 2.5, 'size': 86, 'charge': 0},
    'D': {'hydrophobicity': -3.5, 'size': 91, 'charge': -1},
    'E': {'hydrophobicity': -3.5, 'size': 109, 'charge': -1},
    'F': {'hydrophobicity': 2.8, 'size': 135, 'charge': 0},
    'G': {'hydrophobicity': -0.4, 'size': 48, 'charge': 0},
    'H': {'hydrophobicity': -3.2, 'size': 118, 'charge': 0.5},
    'I': {'hydrophobicity': 4.5, 'size': 124, 'charge': 0},
    'K': {'hydrophobicity': -3.9, 'size': 135, 'charge': 1},
    'L': {'hydrophobicity': 3.8, 'size': 124, 'charge': 0},
    'M': {'hydrophobicity': 1.9, 'size': 124, 'charge': 0},
    'N': {'hydrophobicity': -3.5, 'size': 96, 'charge': 0},
    'P': {'hydrophobicity': -1.6, 'size': 90, 'charge': 0},
    'Q': {'hydrophobicity': -3.5, 'size': 114, 'charge': 0},
    'R': {'hydrophobicity': -4.5, 'size': 148, 'charge': 1},
    'S': {'hydrophobicity': -0.8, 'size': 73, 'charge': 0},
    'T': {'hydrophobicity': -0.7, 'size': 93, 'charge': 0},
    'V': {'hydrophobicity': 4.2, 'size': 105, 'charge': 0},
    'W': {'hydrophobicity': -0.9, 'size': 163, 'charge': 0},
    'Y': {'hydrophobicity': -1.3, 'size': 141, 'charge': 0}
}

# ============================================
# 数据预处理函数
# ============================================
def parse_fasta(file_path):
    """解析FASTA文件，返回序列字典"""
    sequences = {}
    with open(file_path, 'r') as f:
        content = f.read()

    entries = content.split('>')
    for entry in entries[1:]:
        lines = entry.strip().split('\n')
        header = lines[0].strip()
        sequence = ''.join(lines[1:]).replace('\n', '').replace(' ', '')
        seq_id = header.split()[0]
        sequences[seq_id] = {
            'header': header,
            'sequence': sequence,
            'length': len(sequence)
        }
    return sequences

def calculate_aa_composition(sequence):
    """计算氨基酸组成"""
    composition = {}
    seq_len = len(sequence)
    for aa in AMINO_ACIDS:
        composition[f'aa_{aa}'] = sequence.count(aa) / seq_len
    return composition

def calculate_physicochemical_features(sequence):
    """计算物理化学特征"""
    features = {}
    seq_len = len(sequence)

    # 氨基酸组成
    aa_comp = calculate_aa_composition(sequence)
    features.update(aa_comp)

    # 疏水性相关
    hydrophobic_aas = 'AILMFWV'
    hydrophilic_aas = 'RKDENQ'
    features['hydrophobic_ratio'] = sum(sequence.count(aa) for aa in hydrophobic_aas) / seq_len
    features['hydrophilic_ratio'] = sum(sequence.count(aa) for aa in hydrophilic_aas) / seq_len

    # 电荷相关
    positive_aas = 'KRH'
    negative_aas = 'DE'
    features['positive_ratio'] = sum(sequence.count(aa) for aa in positive_aas) / seq_len
    features['negative_ratio'] = sum(sequence.count(aa) for aa in negative_aas) / seq_len
    features['net_charge'] = features['positive_ratio'] - features['negative_ratio']

    # 芳香族氨基酸
    aromatic_aas = 'FWY'
    features['aromatic_ratio'] = sum(sequence.count(aa) for aa in aromatic_aas) / seq_len

    # 极性氨基酸
    polar_aas = 'STCNQ'
    features['polar_ratio'] = sum(sequence.count(aa) for aa in polar_aas) / seq_len

    # 平均疏水性
    avg_hydrophobicity = np.mean([AA_PROPERTIES.get(aa, {'hydrophobicity': 0})['hydrophobicity'] 
                                   for aa in sequence if aa in AA_PROPERTIES])
    features['avg_hydrophobicity'] = avg_hydrophobicity

    # 序列长度
    features['length'] = seq_len

    return features

# ============================================
# ESM-2 Embedding提取
# ============================================
def get_esm_embedding(sequence, model, batch_converter, device):
    """使用ESM-2获取蛋白质序列的embedding"""
    max_len = 1022
    if len(sequence) > max_len:
        sequence = sequence[:max_len]

    data = [("protein", sequence)]
    batch_labels, batch_strs, batch_tokens = batch_converter(data)
    batch_tokens = batch_tokens.to(device)

    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[6], return_contacts=False)

    token_representations = results["representations"][6]
    seq_embedding = token_representations[0, 1:-1].mean(dim=0).cpu().numpy()

    return seq_embedding

# ============================================
# 蛋白质对特征构建
# ============================================
def build_pair_features(gnaq_emb, opsin_emb, gnaq_phy, opsin_phy):
    """构建蛋白质对特征"""
    features = {}

    # ESM embedding特征
    features['esm_concat'] = np.concatenate([gnaq_emb, opsin_emb])
    features['esm_diff'] = np.abs(gnaq_emb - opsin_emb)
    features['esm_product'] = gnaq_emb * opsin_emb

    # 物理化学特征
    phy_keys = [k for k in gnaq_phy.keys() if k not in ['seq_id', 'label']]
    gnaq_phy_vec = np.array([gnaq_phy[k] for k in phy_keys])
    opsin_phy_vec = np.array([opsin_phy[k] for k in phy_keys])

    features['phy_concat'] = np.concatenate([gnaq_phy_vec, opsin_phy_vec])
    features['phy_diff'] = np.abs(gnaq_phy_vec - opsin_phy_vec)

    # 组合所有特征
    combined_features = np.concatenate([
        features['esm_concat'],
        features['esm_diff'],
        features['esm_product'],
        features['phy_concat'],
        features['phy_diff']
    ])

    return combined_features, features

# ============================================
# PyTorch数据集
# ============================================
class ProteinInteractionDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# ============================================
# 注意力机制
# ============================================
class AttentionLayer(nn.Module):
    def __init__(self, hidden_dim):
        super(AttentionLayer, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, x):
        attention_weights = F.softmax(self.attention(x), dim=1)
        return attention_weights

# ============================================
# 深度学习模型
# ============================================
class ProteinInteractionNet(nn.Module):
    def __init__(self, input_dim, hidden_dims=[512, 256, 128], dropout_rate=0.5):
        super(ProteinInteractionNet, self).__init__()

        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim

        self.feature_extractor = nn.Sequential(*layers)
        self.attention = AttentionLayer(hidden_dims[-1])

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dims[-1], 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        features = self.feature_extractor(x)
        attention_weights = self.attention(features)
        output = self.classifier(features)
        return output, attention_weights, features

# ============================================
# 数据增强
# ============================================
def augment_data(X, y, noise_factor=0.05, n_augment=2):
    """通过添加噪声进行数据增强"""
    X_augmented = [X]
    y_augmented = [y]

    for _ in range(n_augment):
        noise = np.random.normal(0, noise_factor, X.shape)
        X_noisy = X + noise
        X_augmented.append(X_noisy)
        y_augmented.append(y)

    return np.vstack(X_augmented), np.hstack(y_augmented)

# ============================================
# 训练函数
# ============================================
def train_model(model, train_loader, val_loader, epochs=100, lr=0.001, patience=15):
    """训练模型"""
    device = torch.device('cpu')
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', 
                                                           factor=0.5, patience=5)

    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0

    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            outputs, _, _ = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs, _, _ = model(X_batch)
                loss = criterion(outputs, y_batch)
                val_loss += loss.item()

        val_loss /= len(val_loader) if len(val_loader) > 0 else 1
        val_losses.append(val_loss)

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model, train_losses, val_losses

# ============================================
# 评估函数
# ============================================
def evaluate_model(model, X_test, y_test):
    """评估模型"""
    device = torch.device('cpu')
    model.eval()

    with torch.no_grad():
        X_tensor = torch.FloatTensor(X_test).to(device)
        outputs, attention_weights, features = model(X_tensor)
        probabilities = F.softmax(outputs, dim=1)
        predictions = torch.argmax(outputs, dim=1).cpu().numpy()
        probs = probabilities[:, 1].cpu().numpy()

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, zero_division=0)
    recall = recall_score(y_test, predictions, zero_division=0)
    f1 = f1_score(y_test, predictions, zero_division=0)

    try:
        auc = roc_auc_score(y_test, probs)
    except:
        auc = 0.5

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc,
        'predictions': predictions,
        'probabilities': probs,
        'attention_weights': attention_weights.cpu().numpy(),
        'features': features.cpu().numpy()
    }

# ============================================
# 主函数
# ============================================
def main():
    """主函数"""
    print("=" * 60)
    print("蛋白质相互作用预测 - GNAQ与Opsin4相互作用分析")
    print("=" * 60)

    # 数据路径
    gnaq_file = '/mnt/okcomputer/upload/GNAQ_human.txt'
    opsin_file = '/mnt/okcomputer/upload/Opsin4_sequences.txt'
    labels_file = '/mnt/okcomputer/upload/OPN4_Gq_interaction_labels.xlsx'

    # 读取数据
    gnaq_data = parse_fasta(gnaq_file)
    gnaq_seq = list(gnaq_data.values())[0]['sequence']

    opsin_data = parse_fasta(opsin_file)
    labels_df = pd.read_excel(labels_file)

    # 构建标签映射
    label_mapping = {}
    for _, row in labels_df.iterrows():
        seq_id = row['NCBI ID']
        label = 1 if row['Activate human Galpha_q?'] == 'Yes' else 0
        label_mapping[seq_id] = label

    # 加载ESM-2模型
    print("\n加载ESM-2模型...")
    model, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
    batch_converter = alphabet.get_batch_converter()
    model.eval()
    device = torch.device('cpu')
    model = model.to(device)

    # 提取embedding
    gnaq_embedding = get_esm_embedding(gnaq_seq, model, batch_converter, device)
    opsin_embeddings = {}
    for seq_id, data in opsin_data.items():
        opsin_embeddings[seq_id] = get_esm_embedding(data['sequence'], model, batch_converter, device)

    # 计算物理化学特征
    gnaq_features = calculate_physicochemical_features(gnaq_seq)
    opsin_features_list = []
    for seq_id, data in opsin_data.items():
        features = calculate_physicochemical_features(data['sequence'])
        features['seq_id'] = seq_id
        features['label'] = label_mapping.get(seq_id, -1)
        opsin_features_list.append(features)
    opsin_features_df = pd.DataFrame(opsin_features_list)

    # 构建蛋白质对特征
    X_list = []
    y_list = []
    sample_ids = []

    for seq_id, opsin_emb in opsin_embeddings.items():
        label = label_mapping[seq_id]
        opsin_phy = opsin_features_df[opsin_features_df['seq_id'] == seq_id].iloc[0].to_dict()
        combined_features, _ = build_pair_features(gnaq_embedding, opsin_emb, gnaq_features, opsin_phy)
        X_list.append(combined_features)
        y_list.append(label)
        sample_ids.append(seq_id)

    X = np.array(X_list)
    y = np.array(y_list)

    # 数据标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Leave-One-Out交叉验证
    loo = LeaveOneOut()
    all_predictions = []
    all_probabilities = []
    all_labels = []

    print("\n开始LOO交叉验证...")
    for fold, (train_idx, val_idx) in enumerate(loo.split(X_scaled)):
        X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # 数据增强
        X_train_aug, y_train_aug = augment_data(X_train, y_train, noise_factor=0.05, n_augment=3)

        train_dataset = ProteinInteractionDataset(X_train_aug, y_train_aug)
        val_dataset = ProteinInteractionDataset(X_val, y_val)

        train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

        # 创建和训练模型
        net = ProteinInteractionNet(input_dim=X.shape[1], hidden_dims=[512, 256, 128], dropout_rate=0.3)
        net, _, _ = train_model(net, train_loader, val_loader, epochs=150, lr=0.001, patience=20)

        # 评估
        result = evaluate_model(net, X_val, y_val)
        all_predictions.extend(result['predictions'])
        all_probabilities.extend(result['probabilities'])
        all_labels.extend(y_val)

    # 计算性能指标
    all_predictions = np.array(all_predictions)
    all_probabilities = np.array(all_probabilities)
    all_labels = np.array(all_labels)

    accuracy = accuracy_score(all_labels, all_predictions)
    precision = precision_score(all_labels, all_predictions, zero_division=0)
    recall = recall_score(all_labels, all_predictions, zero_division=0)
    f1 = f1_score(all_labels, all_predictions, zero_division=0)
    auc = roc_auc_score(all_labels, all_probabilities)

    print(f"\n性能指标:")
    print(f"  Accuracy: {accuracy:.3f}")
    print(f"  Precision: {precision:.3f}")
    print(f"  Recall: {recall:.3f}")
    print(f"  F1: {f1:.3f}")
    print(f"  AUC: {auc:.3f}")

    return accuracy, precision, recall, f1, auc

if __name__ == '__main__':
    main()
