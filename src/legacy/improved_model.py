#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
改进的蛋白质相互作用预测模型
包含交叉注意力机制和对比学习
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class CrossAttention(nn.Module):
    """交叉注意力机制 - 建模opsin和GNAQ之间的相互作用"""
    def __init__(self, dim=320, dropout=0.1):
        super(CrossAttention, self).__init__()
        self.dim = dim
        self.scale = torch.sqrt(torch.FloatTensor([dim]))
        
        # Q, K, V 投影矩阵
        self.W_Q = nn.Linear(dim, dim)
        self.W_K = nn.Linear(dim, dim)
        self.W_V = nn.Linear(dim, dim)
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(dim)
        
    def forward(self, opsin_emb, gnaq_emb):
        """
        Args:
            opsin_emb: [batch, dim]
            gnaq_emb: [batch, dim]
        Returns:
            out: [batch, dim]
            attention: [batch, batch]
        """
        batch_size = opsin_emb.size(0)
        
        # 计算Q, K, V
        Q = self.W_Q(opsin_emb)  # [batch, dim]
        K = self.W_K(gnaq_emb)   # [batch, dim]
        V = self.W_V(gnaq_emb)   # [batch, dim]
        
        # 计算注意力分数
        attention = torch.matmul(Q, K.transpose(-2, -1)) / self.scale.to(Q.device)
        attention = F.softmax(attention, dim=-1)  # [batch, batch]
        attention = self.dropout(attention)
        
        # 应用注意力
        out = torch.matmul(attention, V)  # [batch, dim]
        
        # 残差连接和层归一化
        out = self.layer_norm(opsin_emb + out)
        
        return out, attention


class ContrastiveLoss(nn.Module):
    """对比学习损失 - InfoNCE"""
    def __init__(self, temperature=0.5):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature
        
    def forward(self, z_i, z_j, labels):
        """
        Args:
            z_i: 样本i的表示 [batch, dim]
            z_j: 样本j的表示 [batch, dim]
            labels: 1表示正样本对，0表示负样本对 [batch]
        Returns:
            loss: 标量
        """
        # 计算余弦相似度
        similarity = F.cosine_similarity(z_i, z_j, dim=-1) / self.temperature
        
        # 正样本对的损失
        pos_loss = -torch.mean(labels * torch.log(torch.sigmoid(similarity) + 1e-8))
        
        # 负样本对的损失
        neg_loss = -torch.mean((1 - labels) * torch.log(1 - torch.sigmoid(similarity) + 1e-8))
        
        loss = pos_loss + neg_loss
        return loss


class ImprovedPPIPredictor(nn.Module):
    """改进的蛋白质相互作用预测模型"""
    def __init__(self, esm_dim=320, phys_dim=29, hidden_dim=512, output_dim=2, dropout=0.3):
        super(ImprovedPPIPredictor, self).__init__()
        
        self.esm_dim = esm_dim
        self.phys_dim = phys_dim
        
        # 交叉注意力机制
        self.cross_attention = CrossAttention(esm_dim, dropout)
        
        # 特征融合层
        fusion_dim = esm_dim * 3 + phys_dim * 2  # 640 + 320 + 320 + 58 + 29 = 1367
        
        # 门控融合机制
        self.gate = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim),
            nn.Sigmoid()
        )
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.BatchNorm1d(hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim // 4, output_dim)
        )
        
        # 投影头（用于对比学习）
        self.projection_head = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )
        
    def forward(self, opsin_esm, gnaq_esm, opsin_phys, gnaq_phys, return_embedding=False):
        """
        Args:
            opsin_esm: [batch, 320]
            gnaq_esm: [batch, 320]
            opsin_phys: [batch, 29]
            gnaq_phys: [batch, 29]
            return_embedding: 是否返回embedding用于对比学习
        Returns:
            logits: [batch, 2]
            attention: [batch, batch]
            embedding: [batch, 128] (if return_embedding=True)
        """
        # 交叉注意力
        attended_opsin, attention = self.cross_attention(opsin_esm, gnaq_esm)
        
        # 组合所有特征
        # ESM特征: 拼接 + 差值 + 点积
        esm_concat = torch.cat([opsin_esm, gnaq_esm], dim=-1)  # [batch, 640]
        esm_diff = opsin_esm - gnaq_esm  # [batch, 320]
        esm_product = opsin_esm * gnaq_esm  # [batch, 320]
        
        # 物理化学特征: 拼接 + 差值
        phys_concat = torch.cat([opsin_phys, gnaq_phys], dim=-1)  # [batch, 58]
        phys_diff = opsin_phys - gnaq_phys  # [batch, 29]
        
        # 组合所有特征
        combined = torch.cat([
            esm_concat,
            esm_diff,
            esm_product,
            phys_concat,
            phys_diff
        ], dim=-1)  # [batch, 1367]
        
        # 门控融合
        gate = self.gate(combined)
        fused_features = gate * combined
        
        # 分类预测
        logits = self.classifier(fused_features)
        
        if return_embedding:
            embedding = self.projection_head(fused_features)
            return logits, attention, embedding
        
        return logits, attention


def train_improved_model(model, train_loader, val_loader, epochs=100, lr=0.001, device='cpu'):
    """训练改进的模型"""
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
    
    criterion_ce = nn.CrossEntropyLoss()
    criterion_contrastive = ContrastiveLoss(temperature=0.5)
    
    best_val_loss = float('inf')
    best_model_state = None
    
    train_losses = []
    val_losses = []
    
    for epoch in range(epochs):
        # 训练阶段
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for batch in train_loader:
            opsin_esm, gnaq_esm, opsin_phys, gnaq_phys, labels = batch
            opsin_esm = opsin_esm.to(device)
            gnaq_esm = gnaq_esm.to(device)
            opsin_phys = opsin_phys.to(device)
            gnaq_phys = gnaq_phys.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            # 前向传播
            logits, attention, embeddings = model(
                opsin_esm, gnaq_esm, opsin_phys, gnaq_phys, return_embedding=True
            )
            
            # 分类损失
            loss_ce = criterion_ce(logits, labels)
            
            # 对比学习损失（使用同类样本作为正样本对）
            # 简化版：使用相同标签的样本作为正样本对
            loss_contrastive = 0
            if len(labels) > 1:
                for i in range(len(labels)):
                    for j in range(i+1, len(labels)):
                        label_sim = 1 if labels[i] == labels[j] else 0
                        loss_contrastive += criterion_contrastive(
                            embeddings[i:i+1], embeddings[j:j+1], 
                            torch.tensor([label_sim], dtype=torch.float32).to(device)
                        )
                loss_contrastive = loss_contrastive / (len(labels) * (len(labels) - 1) / 2)
            
            # 总损失
            loss = loss_ce + 0.1 * loss_contrastive
            
            # 反向传播
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(logits.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        
        train_loss = train_loss / len(train_loader)
        train_acc = train_correct / train_total
        train_losses.append(train_loss)
        
        # 验证阶段
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch in val_loader:
                opsin_esm, gnaq_esm, opsin_phys, gnaq_phys, labels = batch
                opsin_esm = opsin_esm.to(device)
                gnaq_esm = gnaq_esm.to(device)
                opsin_phys = opsin_phys.to(device)
                gnaq_phys = gnaq_phys.to(device)
                labels = labels.to(device)
                
                logits, _ = model(opsin_esm, gnaq_esm, opsin_phys, gnaq_phys)
                loss = criterion_ce(logits, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(logits.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_loss = val_loss / len(val_loader)
        val_acc = val_correct / val_total
        val_losses.append(val_loss)
        
        scheduler.step(val_loss)
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}] "
                  f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
    
    # 加载最佳模型
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return model, train_losses, val_losses


if __name__ == "__main__":
    print("改进的蛋白质相互作用预测模型")
    print("包含交叉注意力机制和对比学习")
    print("模型架构:")
    model = ImprovedPPIPredictor()
    print(model)
