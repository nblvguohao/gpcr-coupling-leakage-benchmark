# 技术实现细节文档

## 视蛋白-Gαq蛋白相互作用预测

---

## 一、数据预处理流程

### 1.1 序列数据加载

```python
import pandas as pd
from Bio import SeqIO
import numpy as np

# 加载视蛋白序列
opsin_sequences = {}
for record in SeqIO.parse("Opsin4_sequences.txt", "fasta"):
    opsin_sequences[record.id] = str(record.seq)

# 加载Gαq序列
gnaq_record = next(SeqIO.parse("GNAQ_human.txt", "fasta"))
gnaq_sequence = str(gnaq_record.seq)

# 加载标签
labels_df = pd.read_excel("OPN4_Gq_interaction_labels.xlsx")
```

### 1.2 数据增强策略

```python
class SequenceAugmentation:
    """序列数据增强"""
    
    def __init__(self):
        # 定义相似氨基酸替换矩阵
        self.similar_aa = {
            'A': ['G', 'S', 'T'],  # 小疏水
            'V': ['I', 'L', 'M'],  # 大疏水
            'F': ['Y', 'W'],       # 芳香族
            'K': ['R', 'H'],       # 正电荷
            'D': ['E', 'N'],       # 负电荷/极性
            'S': ['T', 'N'],       # 极性
        }
    
    def conservative_mutation(self, sequence, mutation_rate=0.05):
        """保守突变增强"""
        seq_list = list(sequence)
        n_mutations = int(len(sequence) * mutation_rate)
        positions = np.random.choice(len(sequence), n_mutations, replace=False)
        
        for pos in positions:
            aa = seq_list[pos]
            if aa in self.similar_aa:
                seq_list[pos] = np.random.choice(self.similar_aa[aa])
        
        return ''.join(seq_list)
    
    def random_crop(self, sequence, crop_ratio=0.9):
        """随机裁剪（保留功能域）"""
        crop_length = int(len(sequence) * crop_ratio)
        start = np.random.randint(0, len(sequence) - crop_length + 1)
        return sequence[start:start + crop_length]
```

---

## 二、特征提取模块

### 2.1 ESM-2特征提取

```python
import torch
import esm

class ESM2FeatureExtractor:
    """ESM-2蛋白质语言模型特征提取"""
    
    def __init__(self, model_name="esm2_t33_650M_UR50D"):
        self.model, self.alphabet = esm.pretrained.load_model_and_alphabet(model_name)
        self.model.eval()
        self.batch_converter = self.alphabet.get_batch_converter()
        
    def extract_features(self, sequences):
        """
        提取ESM-2嵌入特征
        
        Args:
            sequences: 蛋白质序列列表
            
        Returns:
            embeddings: [batch_size, seq_len, 1280] 张量
        """
        data = [(f"protein_{i}", seq) for i, seq in enumerate(sequences)]
        batch_labels, batch_strs, batch_tokens = self.batch_converter(data)
        
        with torch.no_grad():
            results = self.model(batch_tokens, repr_layers=[33])
            # 提取第33层的表示
            embeddings = results["representations"][33]
            
        return embeddings
    
    def extract_contact_map(self, sequences):
        """提取预测的接触图"""
        data = [(f"protein_{i}", seq) for i, seq in enumerate(sequences)]
        batch_labels, batch_strs, batch_tokens = self.batch_converter(data)
        
        with torch.no_grad():
            results = self.model(batch_tokens, repr_layers=[33], return_contacts=True)
            contacts = results["contacts"]
            
        return contacts
```

### 2.2 物理化学特征编码

```python
import numpy as np

class PhysicochemicalEncoder:
    """物理化学特征编码器"""
    
    def __init__(self):
        # Kyte-Doolittle疏水性标度
        self.hydrophobicity = {
            'I': 4.5, 'V': 4.2, 'L': 3.8, 'F': 2.8, 'C': 2.5,
            'M': 1.9, 'A': 1.8, 'G': -0.4, 'T': -0.7, 'S': -0.8,
            'W': -0.9, 'Y': -1.3, 'P': -1.6, 'H': -3.2, 'E': -3.5,
            'Q': -3.5, 'D': -3.5, 'N': -3.5, 'K': -3.9, 'R': -4.5
        }
        
        # 电荷性质
        self.charge = {
            'K': 1, 'R': 1, 'H': 0.5,  # 正电荷
            'D': -1, 'E': -1,           # 负电荷
        }
        
        # 分子量
        self.molecular_weight = {
            'A': 89.09, 'R': 174.20, 'N': 132.12, 'D': 133.10, 'C': 121.16,
            'E': 147.13, 'Q': 146.15, 'G': 75.07, 'H': 155.16, 'I': 131.17,
            'L': 131.17, 'K': 146.19, 'M': 149.21, 'F': 165.19, 'P': 115.13,
            'S': 105.09, 'T': 119.12, 'W': 204.23, 'Y': 181.19, 'V': 117.15
        }
        
    def encode(self, sequence):
        """
        编码蛋白质序列为物理化学特征
        
        Args:
            sequence: 氨基酸序列
            
        Returns:
            features: [seq_len, 4] 特征矩阵
        """
        features = np.zeros((len(sequence), 4))
        
        for i, aa in enumerate(sequence):
            features[i, 0] = self.hydrophobicity.get(aa, 0)
            features[i, 1] = self.charge.get(aa, 0)
            features[i, 2] = self.molecular_weight.get(aa, 0) / 200  # 归一化
            features[i, 3] = 1 if aa in 'FWY' else 0  # 芳香性
            
        return features
```

---

## 三、模型架构实现

### 3.1 交叉注意力模块

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossAttention(nn.Module):
    """双蛋白质交叉注意力模块"""
    
    def __init__(self, d_model=1280, n_heads=8, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        # Query, Key, Value 投影
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        
    def forward(self, query_seq, key_seq, value_seq, mask=None):
        """
        交叉注意力前向传播
        
        Args:
            query_seq: [batch, q_len, d_model]
            key_seq: [batch, k_len, d_model]
            value_seq: [batch, v_len, d_model]
            mask: 可选的注意力掩码
            
        Returns:
            output: [batch, q_len, d_model]
            attention_weights: [batch, n_heads, q_len, k_len]
        """
        batch_size = query_seq.size(0)
        
        # 线性投影并分头
        Q = self.W_q(query_seq).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key_seq).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value_seq).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        
        # 计算注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # 应用注意力到Value
        context = torch.matmul(attention_weights, V)
        
        # 拼接多头
        context = context.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model
        )
        
        # 残差连接和层归一化
        output = self.layer_norm(query_seq + context)
        
        return output, attention_weights


class CoAttentionFusion(nn.Module):
    """共注意力融合模块"""
    
    def __init__(self, d_model=1280, n_heads=8):
        super().__init__()
        self.attn_opsin_to_gaq = CrossAttention(d_model, n_heads)
        self.attn_gaq_to_opsin = CrossAttention(d_model, n_heads)
        
        # 融合层
        self.fusion_layer = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
    def forward(self, opsin_features, gaq_features):
        """
        共注意力融合
        
        Args:
            opsin_features: [batch, opsin_len, d_model]
            gaq_features: [batch, gaq_len, d_model]
            
        Returns:
            fused_features: [batch, d_model]
        """
        # 双向交叉注意力
        opsin_attended, attn_weights_1 = self.attn_opsin_to_gaq(
            opsin_features, gaq_features, gaq_features
        )
        gaq_attended, attn_weights_2 = self.attn_gaq_to_opsin(
            gaq_features, opsin_features, opsin_features
        )
        
        # 全局平均池化
        opsin_pooled = opsin_attended.mean(dim=1)
        gaq_pooled = gaq_attended.mean(dim=1)
        
        # 融合
        fused = self.fusion_layer(torch.cat([opsin_pooled, gaq_pooled], dim=-1))
        
        return fused, (attn_weights_1, attn_weights_2)
```

### 3.2 图神经网络模块

```python
import torch_geometric.nn as gnn
from torch_geometric.data import Data

class ProteinGraphNet(nn.Module):
    """蛋白质图神经网络"""
    
    def __init__(self, in_channels=1280, hidden_channels=512, num_layers=3):
        super().__init__()
        
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        # 第一层
        self.convs.append(gnn.GATConv(in_channels, hidden_channels, heads=4, concat=False))
        self.batch_norms.append(nn.BatchNorm1d(hidden_channels))
        
        # 隐藏层
        for _ in range(num_layers - 2):
            self.convs.append(gnn.GATConv(hidden_channels, hidden_channels, heads=4, concat=False))
            self.batch_norms.append(nn.BatchNorm1d(hidden_channels))
        
        # 输出层
        self.convs.append(gnn.GATConv(hidden_channels, hidden_channels, heads=1))
        
    def forward(self, x, edge_index, batch):
        """
        图神经网络前向传播
        
        Args:
            x: 节点特征 [num_nodes, in_channels]
            edge_index: 边索引 [2, num_edges]
            batch: 批处理索引 [num_nodes]
            
        Returns:
            graph_embedding: [batch_size, hidden_channels]
        """
        for i, (conv, bn) in enumerate(zip(self.convs[:-1], self.batch_norms)):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=0.1, training=self.training)
        
        # 最后一层
        x = self.convs[-1](x, edge_index)
        
        # 全局平均池化
        x = gnn.global_mean_pool(x, batch)
        
        return x
    
    def build_graph(self, sequence, embedding, contact_threshold=8.0):
        """
        从序列和嵌入构建蛋白质图
        
        Args:
            sequence: 氨基酸序列
            embedding: ESM-2嵌入 [seq_len, 1280]
            contact_threshold: 接触距离阈值
            
        Returns:
            graph: PyG Data对象
        """
        seq_len = len(sequence)
        
        # 构建K近邻边（基于序列位置）
        edge_list = []
        k = 5  # 每个节点连接k个最近邻
        
        for i in range(seq_len):
            for j in range(max(0, i-k), min(seq_len, i+k+1)):
                if i != j:
                    edge_list.append([i, j])
        
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        
        return Data(x=embedding, edge_index=edge_index)
```

### 3.3 完整模型

```python
class OpsinGaqPredictor(nn.Module):
    """视蛋白-Gαq相互作用预测模型"""
    
    def __init__(self, 
                 d_model=1280,
                 n_heads=8,
                 hidden_dim=512,
                 num_gnn_layers=3,
                 dropout=0.3):
        super().__init__()
        
        # 共注意力融合
        self.co_attention = CoAttentionFusion(d_model, n_heads)
        
        # 图神经网络
        self.graph_net = ProteinGraphNet(d_model, hidden_dim, num_gnn_layers)
        
        # 物理化学特征编码
        self.physio_encoder = nn.Sequential(
            nn.Linear(4, 64),
            nn.ReLU(),
            nn.Linear(64, 128)
        )
        
        # 特征融合
        self.feature_fusion = nn.Sequential(
            nn.Linear(d_model + hidden_dim * 2 + 128 * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim // 2, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
    def forward(self, opsin_data, gaq_data):
        """
        前向传播
        
        Args:
            opsin_data: dict with keys 'embedding', 'graph', 'physio'
            gaq_data: dict with keys 'embedding', 'graph', 'physio'
            
        Returns:
            prediction: [batch_size, 1]
            attention_weights: tuple of attention matrices
        """
        # 共注意力融合
        attended_features, attn_weights = self.co_attention(
            opsin_data['embedding'], 
            gaq_data['embedding']
        )
        
        # 图神经网络处理
        opsin_graph_emb = self.graph_net(
            opsin_data['graph'].x,
            opsin_data['graph'].edge_index,
            opsin_data['graph'].batch
        )
        gaq_graph_emb = self.graph_net(
            gaq_data['graph'].x,
            gaq_data['graph'].edge_index,
            gaq_data['graph'].batch
        )
        
        # 物理化学特征
        opsin_physio = self.physio_encoder(opsin_data['physio']).mean(dim=1)
        gaq_physio = self.physio_encoder(gaq_data['physio']).mean(dim=1)
        
        # 特征拼接
        combined = torch.cat([
            attended_features,
            opsin_graph_emb,
            gaq_graph_emb,
            opsin_physio,
            gaq_physio
        ], dim=-1)
        
        # 融合和分类
        fused = self.feature_fusion(combined)
        prediction = self.classifier(fused)
        
        return prediction, attn_weights
```

---

## 四、对比学习损失

```python
class ContrastiveLoss(nn.Module):
    """物种感知对比损失"""
    
    def __init__(self, temperature=0.5):
        super().__init__()
        self.temperature = temperature
        
    def forward(self, embeddings, labels, species_ids):
        """
        计算对比损失
        
        Args:
            embeddings: [batch_size, dim] 蛋白质嵌入
            labels: [batch_size] 偶联标签 (0/1)
            species_ids: [batch_size] 物种标识
            
        Returns:
            loss: 标量损失值
        """
        batch_size = embeddings.size(0)
        
        # 计算相似度矩阵
        similarity_matrix = F.cosine_similarity(
            embeddings.unsqueeze(1), 
            embeddings.unsqueeze(0), 
            dim=2
        ) / self.temperature
        
        # 创建正样本掩码（同标签不同物种）
        positive_mask = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        species_mask = (species_ids.unsqueeze(0) != species_ids.unsqueeze(1)).float()
        positive_mask = positive_mask * species_mask
        
        # 排除自身
        mask = torch.eye(batch_size, device=embeddings.device).bool()
        positive_mask = positive_mask.masked_fill(mask, 0)
        
        # 计算InfoNCE损失
        exp_sim = torch.exp(similarity_matrix)
        
        # 正样本相似度
        pos_sim = (exp_sim * positive_mask).sum(dim=1)
        
        # 所有负样本相似度
        neg_sim = exp_sim.sum(dim=1) - exp_sim.diag()
        
        loss = -torch.log(pos_sim / (pos_sim + neg_sim + 1e-8) + 1e-8)
        
        return loss.mean()
```

---

## 五、训练流程

```python
import torch.optim as optim
from torch.utils.data import DataLoader

class Trainer:
    """模型训练器"""
    
    def __init__(self, model, device='cuda'):
        self.model = model.to(device)
        self.device = device
        self.contrastive_loss = ContrastiveLoss(temperature=0.5)
        self.bce_loss = nn.BCELoss()
        
    def pretrain(self, train_loader, epochs=50, lr=1e-4):
        """在大规模PPI数据上预训练"""
        optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
        
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            
            for batch in train_loader:
                opsin_data, gaq_data, labels = batch
                opsin_data = {k: v.to(self.device) for k, v in opsin_data.items()}
                gaq_data = {k: v.to(self.device) for k, v in gaq_data.items()}
                labels = labels.to(self.device)
                
                optimizer.zero_grad()
                
                predictions, embeddings = self.model(opsin_data, gaq_data)
                loss = self.bce_loss(predictions.squeeze(), labels.float())
                
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            scheduler.step()
            print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.4f}")
    
    def finetune(self, train_loader, val_loader, epochs=100, lr=1e-5):
        """在目标数据集上微调"""
        # 冻结ESM-2参数
        for param in self.model.esm_parameters():
            param.requires_grad = False
        
        optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=lr, weight_decay=1e-5
        )
        
        best_val_loss = float('inf')
        patience = 20
        patience_counter = 0
        
        for epoch in range(epochs):
            # 训练
            self.model.train()
            train_loss = 0
            
            for batch in train_loader:
                opsin_data, gaq_data, labels, species_ids = batch
                opsin_data = {k: v.to(self.device) for k, v in opsin_data.items()}
                gaq_data = {k: v.to(self.device) for k, v in gaq_data.items()}
                labels = labels.to(self.device)
                species_ids = species_ids.to(self.device)
                
                optimizer.zero_grad()
                
                predictions, embeddings, attn_weights = self.model(opsin_data, gaq_data)
                
                # 组合损失
                cls_loss = self.bce_loss(predictions.squeeze(), labels.float())
                contra_loss = self.contrastive_loss(embeddings, labels, species_ids)
                
                loss = cls_loss + 0.3 * contra_loss
                
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            # 验证
            val_loss = self.evaluate(val_loader)
            
            print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss/len(train_loader):.4f}, "
                  f"Val Loss: {val_loss:.4f}")
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(self.model.state_dict(), 'best_model.pt')
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
    
    def evaluate(self, data_loader):
        """评估模型"""
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch in data_loader:
                opsin_data, gaq_data, labels = batch
                opsin_data = {k: v.to(self.device) for k, v in opsin_data.items()}
                gaq_data = {k: v.to(self.device) for k, v in gaq_data.items()}
                labels = labels.to(self.device)
                
                predictions, _ = self.model(opsin_data, gaq_data)
                loss = self.bce_loss(predictions.squeeze(), labels.float())
                total_loss += loss.item()
        
        return total_loss / len(data_loader)
```

---

## 六、评估指标

```python
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, matthews_corrcoef, confusion_matrix,
    precision_recall_curve, average_precision_score
)

class Evaluator:
    """模型评估器"""
    
    def __init__(self):
        self.metrics = {}
    
    def evaluate(self, y_true, y_pred, y_prob):
        """
        计算所有评估指标
        
        Args:
            y_true: 真实标签
            y_pred: 预测标签
            y_prob: 预测概率
            
        Returns:
            metrics: 指标字典
        """
        self.metrics = {
            'Accuracy': accuracy_score(y_true, y_pred),
            'Precision': precision_score(y_true, y_pred, zero_division=0),
            'Recall': recall_score(y_true, y_pred, zero_division=0),
            'F1': f1_score(y_true, y_pred, zero_division=0),
            'AUC-ROC': roc_auc_score(y_true, y_prob),
            'MCC': matthews_corrcoef(y_true, y_pred),
            'AP': average_precision_score(y_true, y_prob)
        }
        
        return self.metrics
    
    def print_report(self):
        """打印评估报告"""
        print("=" * 50)
        print("Evaluation Report")
        print("=" * 50)
        for metric, value in self.metrics.items():
            print(f"{metric:15s}: {value:.4f}")
        print("=" * 50)
```

---

## 七、可解释性分析

```python
import matplotlib.pyplot as plt
import seaborn as sns

class InterpretabilityAnalyzer:
    """可解释性分析器"""
    
    def __init__(self, model):
        self.model = model
    
    def visualize_attention(self, attn_weights, opsin_seq, gaq_seq, save_path=None):
        """
        可视化注意力权重
        
        Args:
            attn_weights: 注意力权重矩阵
            opsin_seq: 视蛋白序列
            gaq_seq: Gαq序列
            save_path: 保存路径
        """
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Opsin -> Gαq 注意力
        sns.heatmap(attn_weights[0][0].cpu().numpy(), 
                    cmap='viridis', ax=axes[0])
        axes[0].set_title('Opsin → Gαq Attention')
        axes[0].set_xlabel('Gαq Residues')
        axes[0].set_ylabel('Opsin Residues')
        
        # Gαq -> Opsin 注意力
        sns.heatmap(attn_weights[1][0].cpu().numpy(), 
                    cmap='viridis', ax=axes[1])
        axes[1].set_title('Gαq → Opsin Attention')
        axes[1].set_xlabel('Opsin Residues')
        axes[1].set_ylabel('Gαq Residues')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def identify_key_residues(self, attn_weights, opsin_seq, gaq_seq, top_k=10):
        """
        识别关键残基
        
        Args:
            attn_weights: 注意力权重
            opsin_seq: 视蛋白序列
            gaq_seq: Gαq序列
            top_k: 返回前k个关键残基
            
        Returns:
            key_residues: 关键残基列表
        """
        # 计算每个残基的平均注意力权重
        opsin_attention = attn_weights[0].mean(dim=(0, 1)).cpu().numpy()
        gaq_attention = attn_weights[1].mean(dim=(0, 1)).cpu().numpy()
        
        # 获取top-k残基
        opsin_topk = np.argsort(opsin_attention)[-top_k:]
        gaq_topk = np.argsort(gaq_attention)[-top_k:]
        
        key_residues = {
            'opsin': [(i, opsin_seq[i], opsin_attention[i]) for i in opsin_topk],
            'gaq': [(i, gaq_seq[i], gaq_attention[i]) for i in gaq_topk]
        }
        
        return key_residues
```

---

## 八、实验配置

```yaml
# config.yaml
model:
  d_model: 1280
  n_heads: 8
  hidden_dim: 512
  num_gnn_layers: 3
  dropout: 0.3

training:
  pretrain_epochs: 50
  finetune_epochs: 100
  pretrain_lr: 1e-4
  finetune_lr: 1e-5
  batch_size: 32
  weight_decay: 1e-5
  early_stopping_patience: 20

data:
  esm_model: "esm2_t33_650M_UR50D"
  max_seq_length: 500
  contact_threshold: 8.0
  augmentation_rate: 0.05

evaluation:
  n_folds: 5
  random_seeds: [42, 123, 456, 789, 1011]
```

---

*技术文档版本：v1.0*
