# AlphaFold结构特征整合计划

**日期**: 2026年4月8日  
**阶段**: Phase 4 - 结构特征增强

---

## 1. 概述

### 1.1 目标

整合AlphaFold预测的结构信息到GPCR-Gαq耦合预测模型，提升预测准确性。

### 1.2 当前状态

- 已有ESM-2序列特征：AUC 0.8489 (CLS token)
- 服务器无法直接下载AlphaFold DB
- 需要替代方案获取结构信息

---

## 2. 结构特征提取方案

### 2.1 方案对比

| 方案 | 优点 | 缺点 | 优先级 |
|------|------|------|--------|
| A. 本地AlphaFold2预测 | 完整结构、高精度 | 计算昂贵、耗时长 | 中 |
| B. 序列衍生结构特征 | 快速、无需外部依赖 | 信息较间接 | 高 |
| C. ColabFold API | 免费、准确 | 依赖外部服务、批量受限 | 中 |
| D. 结构数据库查询 | 现成数据 | 覆盖率有限 | 高 |

### 2.2 推荐实施方案

**主方案**: B + D 组合
- 优先查询已有结构数据库
- 对缺失结构使用序列衍生特征

---

## 3. 序列衍生结构特征

### 3.1 特征类别

#### 3.1.1 物理化学特征

```python
physicochemical_features = {
    'hydrophobicity': {
        'description': '残基疏水性评分',
        'method': 'Kyte-Doolittle或Hopp-Woods标度',
        'window_size': [5, 10, 15],  # 滑动窗口平均
        'dimension': 3
    },
    'charge': {
        'description': '残基电荷状态',
        'method': 'pH 7.4条件下的电荷',
        'features': ['正电荷密度', '负电荷密度', '净电荷']
    },
    'size': {
        'description': '残基侧链体积',
        'method': 'van der Waals体积'
    },
    'flexibility': {
        'description': '骨架柔性',
        'method': 'B因子预测或柔性指数'
    }
}
```

#### 3.1.2 二级结构预测

```python
secondary_structure_features = {
    'method': 'Chou-Fasman + GOR算法',
    'features': {
        'helix_propensity': 'α-螺旋形成倾向 (0-1)',
        'sheet_propensity': 'β-折叠形成倾向 (0-1)',
        'coil_propensity': '无规卷曲倾向 (0-1)',
        'tm_regions': '跨膜区段预测 (7个TM helix)'
    },
    'implementation': '''
    # 使用biopython或自实现
    from Bio.SeqUtils.ProtParam import ProteinAnalysis

    def predict_ss_features(sequence):
        analyzer = ProteinAnalysis(sequence)

        # 二级结构比例
        ss_frac = analyzer.secondary_structure_fraction()

        # 跨膜区段预测 (简化版)
        tm_regions = predict_tm_helices(sequence)  # 使用TMHMM或简化算法

        return {
            'helix': ss_frac[0],
            'sheet': ss_frac[1],
            'coil': ss_frac[2],
            'tm_count': len(tm_regions)
        }
    '''
}
```

#### 3.1.3 接触图特征

```python
contact_map_features = {
    'description': '残基间接触概率',
    'method': 'ESM-2 contact prediction',
    'features': {
        'intra_contacts': '分子内接触密度',
        'interface_contacts': '预测的界面接触',
        'contact_order': '接触顺序参数'
    }
}
```

### 3.2 特征提取实现

```python
#!/usr/bin/env python3
"""
序列衍生结构特征提取
不依赖外部数据库，纯序列计算
"""
import numpy as np
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from pathlib import Path

class StructureFeatureExtractor:
    """从序列提取结构相关特征"""

    # 疏水性标度 (Kyte-Doolittle)
    hydrophobicity_scale = {
        'I': 4.5, 'V': 4.2, 'L': 3.8, 'F': 2.8, 'C': 2.5,
        'M': 1.9, 'A': 1.8, 'G': -0.4, 'T': -0.7, 'S': -0.8,
        'W': -0.9, 'Y': -1.3, 'P': -1.6, 'H': -3.2, 'E': -3.5,
        'Q': -3.5, 'D': -3.5, 'N': -3.5, 'K': -3.9, 'R': -4.5
    }

    # 残基体积 (Å³)
    volume_scale = {
        'A': 88.6, 'C': 108.5, 'D': 111.1, 'E': 138.4, 'F': 189.9,
        'G': 60.1, 'H': 153.2, 'I': 166.7, 'K': 168.6, 'L': 166.7,
        'M': 162.9, 'N': 114.1, 'P': 112.7, 'Q': 143.8, 'R': 173.4,
        'S': 89.0, 'T': 116.1, 'V': 140.0, 'W': 227.8, 'Y': 193.6
    }

    def __init__(self, sequence):
        self.sequence = sequence
        self.analyzer = ProteinAnalysis(sequence)
        self.length = len(sequence)

    def extract_all_features(self):
        """提取所有结构特征"""
        features = {}

        # 物理化学特征
        features.update(self.physicochemical_features())

        # 二级结构特征
        features.update(self.secondary_structure_features())

        # 残基组成特征
        features.update(self.composition_features())

        # 结构域特征
        features.update(self.domain_features())

        return features

    def physicochemical_features(self):
        """物理化学特征"""
        # 疏水性分析
        hydro_scores = [self.hydrophobicity_scale.get(aa, 0) for aa in self.sequence]

        # 滑动窗口平均疏水性 (预测表面/内部区域)
        window_sizes = [5, 10, 15]
        hydro_features = {}
        for w in window_sizes:
            smoothed = np.convolve(hydro_scores, np.ones(w)/w, mode='valid')
            hydro_features[f'hydro_max_w{w}'] = np.max(smoothed)
            hydro_features[f'hydro_min_w{w}'] = np.min(smoothed)
            hydro_features[f'hydro_mean_w{w}'] = np.mean(smoothed)

        # 疏水性矩 (预测跨膜区)
        hydro_features['hydrophobic_moment'] = self._hydrophobic_moment()

        # 电荷分析
        charged = sum(1 for aa in self.sequence if aa in 'DEKR')
        positive = sum(1 for aa in self.sequence if aa in 'KR')
        negative = sum(1 for aa in self.sequence if aa in 'DE')

        # 体积分析
        volumes = [self.volume_scale.get(aa, 100) for aa in self.sequence]

        return {
            **hydro_features,
            'charge_density': charged / self.length,
            'positive_density': positive / self.length,
            'negative_density': negative / self.length,
            'net_charge': positive - negative,
            'mean_volume': np.mean(volumes),
            'volume_variance': np.var(volumes)
        }

    def _hydrophobic_moment(self, angle=100):
        """计算疏水性矩 (用于预测两亲性螺旋)"""
        hydro_scores = [self.hydrophobicity_scale.get(aa, 0) for aa in self.sequence]
        radians = np.deg2rad(angle * np.arange(len(hydro_scores)))

        moment_x = sum(h * np.cos(r) for h, r in zip(hydro_scores, radians))
        moment_y = sum(h * np.sin(r) for h, r in zip(hydro_scores, radians))

        return np.sqrt(moment_x**2 + moment_y**2) / self.length

    def secondary_structure_features(self):
        """二级结构特征"""
        # BioPython二级结构预测
        ss_frac = self.analyzer.secondary_structure_fraction()

        # 简化版跨膜区预测 (基于疏水性峰)
        tm_regions = self._predict_tm_regions()

        # 柔性分析
        flexibility = list(self.analyzer.flexibility().values())

        return {
            'helix_fraction': ss_frac[0],
            'sheet_fraction': ss_frac[1],
            'coil_fraction': ss_frac[2],
            'tm_region_count': len(tm_regions),
            'mean_flexibility': np.mean(flexibility),
            'max_flexibility': np.max(flexibility),
            'min_flexibility': np.min(flexibility)
        }

    def _predict_tm_regions(self, threshold=1.5):
        """基于疏水性预测跨膜区段"""
        hydro_scores = [self.hydrophobicity_scale.get(aa, 0) for aa in self.sequence]

        # 21残基窗口 (典型跨膜螺旋长度)
        window = 21
        tm_regions = []

        for i in range(len(hydro_scores) - window + 1):
            avg_hydro = np.mean(hydro_scores[i:i+window])
            if avg_hydro > threshold:
                tm_regions.append((i, i+window))

        # 合并重叠区域
        merged = []
        for start, end in tm_regions:
            if merged and start <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], end))
            else:
                merged.append((start, end))

        return merged

    def composition_features(self):
        """残基组成特征"""
        aa_counts = self.analyzer.count_amino_acids()
        total = sum(aa_counts.values())

        # 按类别分组
        categories = {
            'hydrophobic': 'AVLIMCWFY',
            'polar': 'STNQ',
            'charged': 'DEKR',
            'special': 'GP'
        }

        features = {}
        for cat, aas in categories.items():
            count = sum(aa_counts.get(aa, 0) for aa in aas)
            features[f'{cat}_fraction'] = count / total

        return features

    def domain_features(self):
        """结构域相关特征"""
        # N端信号肽/膜定位预测 (简化)
        n_term = self.sequence[:30]
        n_hydro = [self.hydrophobicity_scale.get(aa, 0) for aa in n_term]

        # C端特征 (G蛋白偶联相关)
        c_term = self.sequence[-30:] if len(self.sequence) > 30 else self.sequence
        c_hydro = [self.hydrophobicity_scale.get(aa, 0) for aa in c_term]

        return {
            'n_term_hydrophobicity': np.mean(n_hydro),
            'c_term_hydrophobicity': np.mean(c_hydro),
            'sequence_length': self.length,
            'length_category': 1 if self.length > 400 else 0  # 长序列标记
        }


def extract_structure_features_for_dataset(sequences_dict):
    """为整个数据集提取结构特征"""
    features_dict = {}

    for uid, seq_data in sequences_dict.items():
        sequence = seq_data['sequence'] if isinstance(seq_data, dict) else seq_data

        extractor = StructureFeatureExtractor(sequence)
        features = extractor.extract_all_features()

        features_dict[uid] = features

    return features_dict


# 使用示例
if __name__ == "__main__":
    # 测试序列 (GPCR)
    test_seq = "MTLPT..."  # 实际序列

    extractor = StructureFeatureExtractor(test_seq)
    features = extractor.extract_all_features()

    print(f"提取特征数量: {len(features)}")
    for k, v in list(features.items())[:10]:
        print(f"  {k}: {v:.4f}")
```

---

## 4. 多尺度特征融合

### 4.1 特征整合策略

```
ESM-2序列特征 (320-dim)
         ↓
    [CLS] token
         ↓
    ┌─────────┐
    │  融合层  │
    └─────────┘
         ↑
结构特征 (50-dim)
    - 物理化学: 15-dim
    - 二级结构: 8-dim
    - 组成特征: 6-dim
    - 结构域: 4-dim
         ↑
    序列衍生计算
```

### 4.2 模型修改

```python
class EnhancedCrossAttention(nn.Module):
    """增强版交叉注意力，整合结构特征"""

    def __init__(self, seq_dim=320, struct_dim=50, hidden_dim=256):
        super().__init__()

        # 序列特征投影
        self.seq_embedding = nn.Sequential(
            nn.Linear(seq_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.3)
        )

        # 结构特征投影
        self.struct_embedding = nn.Sequential(
            nn.Linear(struct_dim, hidden_dim // 4),  # 较小维度
            nn.LayerNorm(hidden_dim // 4),
            nn.GELU(),
            nn.Dropout(0.3)
        )

        # 融合层
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim // 4, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )

        # 交叉注意力 (同上)
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=4,
            dropout=0.3, batch_first=True
        )

        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, seq_feat, struct_feat, g_seq_feat, g_struct_feat):
        # seq_feat: (batch, 1, 320)
        # struct_feat: (batch, 50)

        # 投影序列特征
        seq_emb = self.seq_embedding(seq_feat)  # (batch, 1, hidden)
        g_seq_emb = self.seq_embedding(g_seq_feat)

        # 投影并扩展结构特征
        struct_emb = self.struct_embedding(struct_feat).unsqueeze(1)  # (batch, 1, hidden/4)
        g_struct_emb = self.struct_embedding(g_struct_feat).unsqueeze(1)

        # 拼接特征
        combined = torch.cat([seq_emb, struct_emb], dim=-1)  # (batch, 1, hidden*1.25)
        g_combined = torch.cat([g_seq_emb, g_struct_emb], dim=-1)

        # 融合
        x1 = self.fusion(combined)
        x2 = self.fusion(g_combined)

        # 交叉注意力
        attn_out, _ = self.attention(x1, x2, x2)
        x = torch.cat([x1.squeeze(1), attn_out.squeeze(1)], dim=-1)

        return self.classifier(x).squeeze(-1)
```

---

## 5. 实施计划

### 5.1 实施步骤

```
Week 1:  实现序列衍生结构特征提取
        └─ 完成StructureFeatureExtractor类
        └─ 测试特征质量

Week 2:  修改训练脚本整合结构特征
        └─ 更新数据加载器
        └─ 修改模型架构
        └─ 运行对比实验

Week 3:  评估与优化
        └─ 分析结构特征贡献
        └─ 特征选择优化
        └─ 消融实验
```

### 5.2 预期改进

| 指标 | 当前(纯序列) | 目标(融合结构) | 提升 |
|------|-------------|---------------|------|
| AUC | 0.8489 | > 0.88 | +3% |
| 准确率 | 0.68 | > 0.75 | +7% |
| 召回率 | 0.42 | > 0.60 | +18% |

---

## 6. 下一步行动

- [ ] 实现 `extract_structure_features.py` 脚本
- [ ] 上传服务器提取100样本的结构特征
- [ ] 修改训练脚本支持多模态特征
- [ ] 运行对比实验验证改进效果

---

**文档版本**: v1.0  
**创建日期**: 2026年4月8日
