# 实验设计与评估方案

## 视蛋白-Gαq蛋白相互作用预测

---

## 一、实验设计概述

### 1.1 数据集划分策略

由于样本量极小（9个样本），采用以下策略：

```
原始数据：
├── 正样本（4个）
│   ├── 人类melanopsin isoform 1 (NP_150598.1)
│   ├── 人类melanopsin isoform 2 (NP_001025186.1)
│   ├── 鱼类melanopsin (AAO38857.1)
│   └── 蜘蛛kumopsin1 (BAG14330.1)
│
└── 负样本（5个）
    ├── 蜜蜂UV opsin (BAH04514.1)
    ├── 蜜蜂blue opsin (BAH04515.1)
    ├── 蜜蜂long-wavelength opsin (BAH04516.1)
    ├── 蝴蝶opsin (BAA31723.1)
    └── 鱿鱼opsin (ACB05672.1)
```

### 1.2 交叉验证方案

#### A. Leave-One-Species-Out (LOSO) 交叉验证

```
Fold 1: 留人类
├── 训练：鱼类melanopsin, 蜘蛛kumopsin1, 蜜蜂3种, 蝴蝶, 鱿鱼
├── 验证：人类melanopsin isoform 1/2
└── 测试物种：Homo sapiens

Fold 2: 留鱼类
├── 训练：人类2种, 蜘蛛kumopsin1, 蜜蜂3种, 蝴蝶, 鱿鱼
├── 验证：鱼类melanopsin
└── 测试物种：Rutilus rutilus

Fold 3: 留蜘蛛
├── 训练：人类2种, 鱼类melanopsin, 蜜蜂3种, 蝴蝶, 鱿鱼
├── 验证：蜘蛛kumopsin1
└── 测试物种：Hasarius adansoni

... (其他fold类似)
```

#### B. Leave-One-Out (LOO) 交叉验证

```
每次留1个样本作为测试集，其余8个作为训练集
共9次实验，确保每个样本都被测试一次
```

#### C. 分层K折交叉验证

```
考虑到类别平衡，采用分层抽样
K=3或K=5（视数据增强后的样本量而定）
确保每折中正负样本比例一致
```

---

## 二、数据增强实验

### 2.1 增强策略对比

| 策略 | 描述 | 预期效果 |
|------|------|----------|
| 无增强 | 原始9个样本 | 基线 |
| 保守突变 | 每个序列生成5个变体 | 中等提升 |
| 同源序列 | BLAST检索相似序列 | 显著提升 |
| 结构增强 | AlphaFold结构扰动 | 中等提升 |
| 组合增强 | 上述方法组合 | 最大提升 |

### 2.2 增强参数设置

```python
augmentation_config = {
    "conservative_mutation": {
        "mutation_rate": 0.05,  # 5%残基突变
        "n_variants_per_sequence": 5
    },
    "homolog_retrieval": {
        "database": "UniRef90",
        "evalue_threshold": 1e-5,
        "identity_range": [30, 90],  # 30%-90%相似度
        "max_homologs_per_sequence": 10
    },
    "structure_perturbation": {
        "noise_std": 0.5,  # 坐标扰动标准差
        "n_perturbations": 3
    }
}
```

---

## 三、基线方法对比

### 3.1 传统机器学习方法

| 方法 | 特征 | 实现 |
|------|------|------|
| SVM | 序列k-mer + 物理化学 | scikit-learn |
| Random Forest | 序列特征 + 结构特征 | scikit-learn |
| XGBoost | ESM-2嵌入 | xgboost |
| Logistic Regression | 手工特征 | scikit-learn |

### 3.2 深度学习方法

| 方法 | 架构 | 特征 | 来源 |
|------|------|------|------|
| PIPR | Siamese RCNN | 序列 | Chen et al., 2019 |
| D-SCRIPT | CNN | 序列 | Sledzieski et al., 2021 |
| DeepFE-PPI | FCNN | ESM-2 | 改编 |
| EGRET | GAT | ProtBERT | Mahbub & Bayzid, 2022 |
| **本研究** | Cross-Attn + GNN | ESM-2 + 多模态 | 原创 |

### 3.3 序列相似性方法

| 方法 | 原理 |
|------|------|
| BLAST | 序列比对相似度 |
| HMMER | 隐马尔可夫模型 |
| SPRINT | 序列相似性网络 |

---

## 四、消融实验设计

### 4.1 模块消融

```
完整模型: ESM-2 + Cross-Attn + GNN + Physio + Contrastive

消融实验:
├── Abl-1: 移除Cross-Attn (仅用ESM-2平均池化)
├── Abl-2: 移除GNN (仅用序列特征)
├── Abl-3: 移除Physio (仅用ESM-2)
├── Abl-4: 移除Contrastive (仅用BCE损失)
├── Abl-5: 冻结ESM-2 (不微调)
└── Abl-6: 仅使用ESM-2 + 简单分类器
```

### 4.2 特征消融

```
特征组合实验:
├── ESM-2 only
├── ESM-2 + Physico
├── ESM-2 + Structure
├── ESM-2 + Physico + Structure
└── ESM-2 + Physico + Structure + Contrastive (完整)
```

### 4.3 架构消融

```
架构变体:
├── Variant-1: Cross-Attn → Concat → FC
├── Variant-2: Bi-LSTM → Attention → FC
├── Variant-3: Transformer Encoder → FC
├── Variant-4: GNN only
└── Variant-5: Cross-Attn + GNN (本研究)
```

---

## 五、评估指标

### 5.1 分类指标

| 指标 | 公式 | 说明 |
|------|------|------|
| Accuracy | (TP+TN)/(TP+TN+FP+FN) | 整体准确率 |
| Precision | TP/(TP+FP) | 精确率 |
| Recall | TP/(TP+FN) | 召回率 |
| F1 Score | 2×Precision×Recall/(Precision+Recall) | 调和平均 |
| MCC | (TP×TN-FP×FN)/√[(TP+FP)(TP+FN)(TN+FP)(TN+FN)] | 马修斯相关系数 |
| AUC-ROC | ROC曲线下面积 | 分类能力 |
| AUC-PR | PR曲线下面积 | 不平衡数据适用 |

### 5.2 统计显著性检验

```python
# 配对t检验比较不同方法
from scipy import stats

def paired_t_test(model1_scores, model2_scores):
    """配对t检验"""
    t_stat, p_value = stats.ttest_rel(model1_scores, model2_scores)
    return t_stat, p_value

# Wilcoxon符号秩检验（非参数）
def wilcoxon_test(model1_scores, model2_scores):
    """Wilcoxon符号秩检验"""
    w_stat, p_value = stats.wilcoxon(model1_scores, model2_scores)
    return w_stat, p_value
```

### 5.3 置信区间估计

```python
import numpy as np
from scipy import stats

def confidence_interval(scores, confidence=0.95):
    """计算置信区间"""
    mean = np.mean(scores)
    sem = stats.sem(scores)  # 标准误
    ci = stats.t.interval(confidence, len(scores)-1, loc=mean, scale=sem)
    return mean, ci
```

---

## 六、可解释性实验

### 6.1 注意力可视化

```
实验:
1. 提取所有样本的交叉注意力权重
2. 计算每个残基的平均注意力分数
3. 识别高注意力区域（top-10%）
4. 与已知功能域比对
```

### 6.2 关键残基识别

```
方法:
1. 基于注意力权重排序
2. 基于SHAP值分析
3. 基于梯度归因（Integrated Gradients）
4. 基于消融实验（逐个残基mask）
```

### 6.3 结构定位

```
步骤:
1. AlphaFold2预测3D结构
2. 将关键残基映射到结构
3. 分析空间位置和相互作用
4. 与文献报道的相互作用位点比对
```

---

## 七、跨物种泛化实验

### 7.1 物种间相似性分析

```python
# 计算物种间序列相似度矩阵
from Bio import pairwise2

def compute_species_similarity(sequences):
    """计算物种间序列相似度"""
    n = len(sequences)
    similarity_matrix = np.zeros((n, n))
    
    for i in range(n):
        for j in range(i+1, n):
            alignments = pairwise2.align.globalxx(sequences[i], sequences[j])
            identity = alignments[0][2] / max(len(sequences[i]), len(sequences[j]))
            similarity_matrix[i, j] = identity
            similarity_matrix[j, i] = identity
    
    return similarity_matrix
```

### 7.2 进化距离分析

```
实验:
1. 构建系统发育树
2. 分析预测性能与进化距离的关系
3. 检验：进化距离越近，预测越准确？
```

### 7.3 功能域保守性

```
分析:
1. 识别GPCR保守功能域（TM1-7, ICL, ECL）
2. 计算各功能域的保守性分数
3. 关联保守性与预测置信度
```

---

## 八、实验流程

### 8.1 完整实验流程图

```
开始
  │
  ▼
┌─────────────────────────┐
│ 1. 数据预处理            │
│ • 序列加载和清洗          │
│ • 标签编码               │
│ • 序列对齐（可选）        │
└─────────────────────────┘
  │
  ▼
┌─────────────────────────┐
│ 2. 特征提取              │
│ • ESM-2嵌入              │
│ • 物理化学特征           │
│ • AlphaFold结构预测      │
└─────────────────────────┘
  │
  ▼
┌─────────────────────────┐
│ 3. 数据增强              │
│ • 保守突变               │
│ • 同源序列检索           │
│ • 结构扰动               │
└─────────────────────────┘
  │
  ▼
┌─────────────────────────┐
│ 4. 模型训练              │
│ • 预训练（大规模PPI数据） │
│ • 微调（目标数据集）      │
│ • 早停和模型选择         │
└─────────────────────────┘
  │
  ▼
┌─────────────────────────┐
│ 5. 交叉验证              │
│ • LOSO验证               │
│ • LOO验证                │
│ • 分层K折验证            │
└─────────────────────────┘
  │
  ▼
┌─────────────────────────┐
│ 6. 基线对比              │
│ • 传统ML方法             │
│ • 深度学习方法           │
│ • 统计显著性检验         │
└─────────────────────────┘
  │
  ▼
┌─────────────────────────┐
│ 7. 消融实验              │
│ • 模块消融               │
│ • 特征消融               │
│ • 架构消融               │
└─────────────────────────┘
  │
  ▼
┌─────────────────────────┐
│ 8. 可解释性分析          │
│ • 注意力可视化           │
│ • 关键残基识别           │
│ • 结构定位               │
└─────────────────────────┘
  │
  ▼
┌─────────────────────────┐
│ 9. 结果汇总              │
│ • 性能指标汇总           │
│ • 图表生成               │
│ • 生物学解释             │
└─────────────────────────┘
  │
  ▼
结束
```

### 8.2 实验配置

```yaml
# experiment_config.yaml
experiment:
  name: "opsin_gaq_prediction"
  random_seed: 42
  device: "cuda"
  
data:
  train_test_split: "LOSO"  # LOSO, LOO, StratifiedKFold
  n_folds: 9  # for LOSO
  augmentation: true
  
model:
  esm_model: "esm2_t33_650M_UR50D"
  hidden_dim: 512
  num_layers: 3
  dropout: 0.3
  
training:
  pretrain:
    epochs: 50
    lr: 1e-4
    batch_size: 32
  finetune:
    epochs: 100
    lr: 1e-5
    batch_size: 8
    early_stopping_patience: 20

evaluation:
  metrics: ["accuracy", "precision", "recall", "f1", "mcc", "auc_roc", "auc_pr"]
  n_bootstrap: 1000
  confidence_level: 0.95
```

---

## 九、预期结果

### 9.1 性能预期

| 方法 | 预期Accuracy | 预期F1 | 预期AUC |
|------|-------------|--------|---------|
| SVM | 60-70% | 0.55-0.65 | 0.65-0.75 |
| Random Forest | 65-75% | 0.60-0.70 | 0.70-0.80 |
| PIPR | 70-80% | 0.65-0.75 | 0.75-0.85 |
| D-SCRIPT | 75-85% | 0.70-0.80 | 0.80-0.90 |
| **本研究** | **>85%** | **>0.80** | **>0.90** |

### 9.2 消融实验预期

| 配置 | 预期Accuracy | 相对完整模型下降 |
|------|-------------|-----------------|
| 完整模型 | 85% | - |
| 无Cross-Attn | 78% | -7% |
| 无GNN | 80% | -5% |
| 无Physio | 82% | -3% |
| 无Contrastive | 81% | -4% |
| 仅ESM-2 | 75% | -10% |

---

## 十、风险与应对

| 风险 | 概率 | 影响 | 应对策略 |
|------|------|------|----------|
| 过拟合 | 高 | 严重 | 数据增强+正则化+早停 |
| 负样本不足 | 中 | 中等 | 合成负样本+加权损失 |
| 跨物种泛化差 | 中 | 严重 | 对比学习+物种无关特征 |
| 可解释性不足 | 低 | 中等 | 多重验证+文献支持 |
| 计算资源不足 | 低 | 轻微 | 模型压缩+云计算 |

---

*实验设计版本：v1.0*
