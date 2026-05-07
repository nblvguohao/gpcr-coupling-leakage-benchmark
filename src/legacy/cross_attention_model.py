#!/usr/bin/env python3
"""
视蛋白-Gαq蛋白相互作用预测模型
包含真正的交叉注意力机制
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import json

# ==============================================================================
# 1. 交叉注意力机制 (Cross-Attention)
# ==============================================================================

class CrossAttention(nn.Module):
    """
    双蛋白质交叉注意力机制
    计算视蛋白和Gαq之间的注意力权重
    """
    def __init__(self, dim=320, num_heads=4, dropout=0.1):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        # Q, K, V 投影
        self.opsin_q = nn.Linear(dim, dim)
        self.opsin_k = nn.Linear(dim, dim)
        self.opsin_v = nn.Linear(dim, dim)

        self.gnaq_q = nn.Linear(dim, dim)
        self.gnaq_k = nn.Linear(dim, dim)
        self.gnaq_v = nn.Linear(dim, dim)

        self.dropout = nn.Dropout(dropout)
        self.opsin_out = nn.Linear(dim, dim)
        self.gnaq_out = nn.Linear(dim, dim)

    def forward(self, opsin_emb, gnaq_emb):
        """
        Args:
            opsin_emb: [batch, dim] 视蛋白嵌入
            gnaq_emb: [batch, dim] Gαq嵌入
        Returns:
            opsin_out: [batch, dim] 更新后的视蛋白表示
            gnaq_out: [batch, dim] 更新后的Gαq表示
            attention_weights: [batch, batch] 注意力权重矩阵
        """
        batch_size = opsin_emb.size(0)

        # 计算Q, K, V
        # 视蛋白作为Query，Gαq作为Key和Value
        Q_opsin = self.opsin_q(opsin_emb).view(batch_size, self.num_heads, self.head_dim).transpose(0, 1)
        K_gnaq = self.gnaq_k(gnaq_emb).view(batch_size, self.num_heads, self.head_dim).transpose(0, 1)
        V_gnaq = self.gnaq_v(gnaq_emb).view(batch_size, self.num_heads, self.head_dim).transpose(0, 1)

        # Gαq作为Query，视蛋白作为Key和Value
        Q_gnaq = self.gnaq_q(gnaq_emb).view(batch_size, self.num_heads, self.head_dim).transpose(0, 1)
        K_opsin = self.opsin_k(opsin_emb).view(batch_size, self.num_heads, self.head_dim).transpose(0, 1)
        V_opsin = self.opsin_v(opsin_emb).view(batch_size, self.num_heads, self.head_dim).transpose(0, 1)

        # 计算注意力: 视蛋白 → Gαq
        attn_opsin = torch.matmul(Q_opsin, K_gnaq.transpose(-2, -1)) * self.scale
        attn_opsin = F.softmax(attn_opsin, dim=-1)
        attn_opsin = self.dropout(attn_opsin)
        out_opsin = torch.matmul(attn_opsin, V_gnaq)

        # 计算注意力: Gαq → 视蛋白
        attn_gnaq = torch.matmul(Q_gnaq, K_opsin.transpose(-2, -1)) * self.scale
        attn_gnaq = F.softmax(attn_gnaq, dim=-1)
        attn_gnaq = self.dropout(attn_gnaq)
        out_gnaq = torch.matmul(attn_gnaq, V_opsin)

        # 重塑并投影
        out_opsin = out_opsin.transpose(0, 1).contiguous().view(batch_size, self.dim)
        out_gnaq = out_gnaq.transpose(0, 1).contiguous().view(batch_size, self.dim)

        out_opsin = self.opsin_out(out_opsin)
        out_gnaq = self.gnaq_out(out_gnaq)

        # 残差连接
        out_opsin = out_opsin + opsin_emb
        out_gnaq = out_gnaq + gnaq_emb

        # 返回平均注意力权重用于可视化
        avg_attn = (attn_opsin.mean(0) + attn_gnaq.mean(0).t()) / 2

        return out_opsin, out_gnaq, avg_attn


# ==============================================================================
# 2. 对比学习损失 (Contrastive Loss)
# ==============================================================================

class ContrastiveLoss(nn.Module):
    """
    InfoNCE对比学习损失
    用于跨物种的蛋白质对比学习
    """
    def __init__(self, temperature=0.5):
        super().__init__()
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, embeddings, labels, species_ids=None):
        """
        Args:
            embeddings: [batch, dim] 蛋白质嵌入
            labels: [batch] 偶联标签 (1=Gq, 0=非Gq)
            species_ids: [batch] 物种ID (用于构造正样本对)
        Returns:
            loss: 对比学习损失
        """
        batch_size = embeddings.size(0)

        # 计算相似度矩阵
        similarity = torch.matmul(embeddings, embeddings.t()) / self.temperature

        # 创建正样本掩码
        if species_ids is not None:
            # 同一物种、相同标签的为正样本
            label_mask = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
            species_mask = (species_ids.unsqueeze(0) == species_ids.unsqueeze(1)).float()
            positive_mask = label_mask * (1 - species_mask)  # 同功能不同物种
        else:
            # 仅基于标签构造正样本
            positive_mask = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()

        # 移除对角线
        mask = torch.eye(batch_size, device=embeddings.device).bool()
        positive_mask = positive_mask.masked_fill(mask, 0)

        # 对于每个样本，正样本是相似度最高的同类样本
        # 使用InfoNCE损失
        exp_sim = torch.exp(similarity)

        # 正样本的相似度
        pos_sim = (exp_sim * positive_mask).sum(dim=1)

        # 所有负样本的相似度
        neg_sim = exp_sim.sum(dim=1) - exp_sim.diag()

        # 损失 = -log(正样本相似度 / 所有负样本相似度)
        loss = -torch.log(pos_sim / (pos_sim + neg_sim + 1e-8) + 1e-8)

        return loss.mean()


# ==============================================================================
# 3. 多模态融合网络
# ==============================================================================

class MultimodalFusion(nn.Module):
    """
    多模态特征融合网络
    融合ESM-2特征、物理化学特征和结构特征
    """
    def __init__(self, esm_dim=320, phys_dim=29, struct_dim=11, hidden_dim=256):
        super().__init__()
        self.esm_dim = esm_dim
        self.phys_dim = phys_dim
        self.struct_dim = struct_dim

        # 门控融合机制
        self.gate = nn.Sequential(
            nn.Linear(esm_dim + phys_dim + struct_dim, 3),
            nn.Softmax(dim=-1)
        )

        # 特征投影
        self.esm_proj = nn.Linear(esm_dim, hidden_dim)
        self.phys_proj = nn.Linear(phys_dim, hidden_dim)
        self.struct_proj = nn.Linear(struct_dim, hidden_dim)

        # 融合层
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

    def forward(self, esm_feat, phys_feat, struct_feat):
        """
        Args:
            esm_feat: [batch, esm_dim]
            phys_feat: [batch, phys_dim]
            struct_feat: [batch, struct_dim]
        Returns:
            fused: [batch, hidden_dim]
            gate_weights: [batch, 3] 用于可解释性
        """
        # 计算门控权重
        concat_feat = torch.cat([esm_feat, phys_feat, struct_feat], dim=-1)
        gate_weights = self.gate(concat_feat)  # [batch, 3]

        # 投影各模态特征
        esm_proj = self.esm_proj(esm_feat)
        phys_proj = self.phys_proj(phys_feat)
        struct_proj = self.struct_proj(struct_feat)

        # 门控加权融合
        fused = torch.cat([
            esm_proj * gate_weights[:, 0:1],
            phys_proj * gate_weights[:, 1:2],
            struct_proj * gate_weights[:, 2:3]
        ], dim=-1)

        fused = self.fusion(fused)

        return fused, gate_weights


# ==============================================================================
# 4. 完整模型
# ==============================================================================

class OpsinGaqPredictor(nn.Module):
    """
    视蛋白-Gαq相互作用预测器
    包含交叉注意力、多模态融合和对比学习
    """
    def __init__(
        self,
        esm_dim=320,
        phys_dim=29,
        struct_dim=11,
        hidden_dim=256,
        num_heads=4,
        num_layers=2,
        dropout=0.3
    ):
        super().__init__()

        # Gαq序列特征提取器 (Gαq是固定的)
        self.gnaq_encoder = nn.Sequential(
            nn.Linear(esm_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # 视蛋白多模态融合
        self.multimodal_fusion = MultimodalFusion(
            esm_dim=esm_dim,
            phys_dim=phys_dim,
            struct_dim=struct_dim,
            hidden_dim=hidden_dim
        )

        # 交叉注意力
        self.cross_attention = CrossAttention(
            dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout
        )

        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )

        # 对比学习损失
        self.contrastive_loss = ContrastiveLoss(temperature=0.5)

    def forward(
        self,
        opsin_esm,
        opsin_phys,
        opsin_struct,
        gnaq_esm,
        labels=None,
        species_ids=None
    ):
        """
        Args:
            opsin_esm: [batch, esm_dim] 视蛋白ESM特征
            opsin_phys: [batch, phys_dim] 视蛋白物理化学特征
            opsin_struct: [batch, struct_dim] 视蛋白结构特征
            gnaq_esm: [batch, esm_dim] Gαq ESM特征
            labels: [batch] 用于计算对比损失
            species_ids: [batch] 用于计算对比损失
        Returns:
            logits: [batch] 预测logits
            attention_weights: [batch, batch] 注意力权重
            contrastive_loss: 对比学习损失
        """
        # 融合视蛋白多模态特征
        opsin_fused, gate_weights = self.multimodal_fusion(
            opsin_esm, opsin_phys, opsin_struct
        )  # [batch, hidden_dim]

        # 编码Gαq特征
        gnaq_encoded = self.gnaq_encoder(gnaq_esm)  # [batch, hidden_dim]

        # 交叉注意力
        opsin_attended, gnaq_attended, attention_weights = self.cross_attention(
            opsin_fused, gnaq_encoded
        )

        # 拼接特征
        combined = torch.cat([opsin_attended, gnaq_attended], dim=-1)

        # 分类
        logits = self.classifier(combined).squeeze(-1)

        # 计算对比学习损失
        contrastive_loss = 0
        if labels is not None:
            # 使用融合后的视蛋白特征进行对比学习
            contrastive_loss = self.contrastive_loss(
                opsin_fused, labels, species_ids
            )

        return logits, attention_weights, contrastive_loss, gate_weights


# ==============================================================================
# 5. 数据加载和训练
# ==============================================================================

def load_data(data_dir):
    """加载数据集"""
    with open(f"{data_dir}/real_sequences.json", 'r') as f:
        sequences = json.load(f)
    with open(f"{data_dir}/real_labels.json", 'r') as f:
        labels = json.load(f)

    # 转换为numpy数组
    ids = list(sequences.keys())
    X = np.array([sequences[k]['length'] for k in ids]).reshape(-1, 1)
    y = np.array([labels[k] for k in ids])

    return X, y, ids


if __name__ == "__main__":
    print("="*60)
    print("视蛋白-Gαq相互作用预测模型")
    print("包含交叉注意力和对比学习")
    print("="*60)

    # 测试模型
    model = OpsinGaqPredictor()
    print(f"\n模型参数: {sum(p.numel() for p in model.parameters()):,}")

    # 测试前向传播
    batch_size = 4
    opsin_esm = torch.randn(batch_size, 320)
    opsin_phys = torch.randn(batch_size, 29)
    opsin_struct = torch.randn(batch_size, 11)
    gnaq_esm = torch.randn(batch_size, 320)
    labels = torch.randint(0, 2, (batch_size,))

    logits, attn, contrastive_loss, gate = model(
        opsin_esm, opsin_phys, opsin_struct, gnaq_esm, labels
    )

    print(f"输出logits形状: {logits.shape}")
    print(f"注意力权重形状: {attn.shape}")
    print(f"对比学习损失: {contrastive_loss.item():.4f}")
    print(f"门控权重形状: {gate.shape}")
    print("\n模型测试成功!")
