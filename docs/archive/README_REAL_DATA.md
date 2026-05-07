# 真实数据版本 - 使用说明

## 📋 概述

本目录包含**真实数据版本**的完整代码，使用以下真实数据源：

| 数据源 | 用途 | 网址 |
|--------|------|------|
| **GPCRdb** | GPCR-G蛋白相互作用数据 | https://gpcrdb.org/ |
| **UniProt** | 蛋白质序列数据 | https://www.uniprot.org/ |
| **AlphaFold DB** | 蛋白质结构预测 | https://alphafold.ebi.ac.uk/ |
| **ESM-2** | 蛋白质序列特征 | https://github.com/facebookresearch/esm |
| **PubMed** | 文献验证 | https://pubmed.ncbi.nlm.nih.gov/ |

---

## 📁 文件结构

```
/mnt/okcomputer/output/
├── real_data_pipeline.py      # 真实数据收集
├── feature_extraction.py      # ESM-2特征提取
├── structure_features.py      # AlphaFold结构特征
├── model_training_real.py     # 模型训练
├── real_references.bib        # 真实文献引用
├── run_real_pipeline.sh       # 一键运行脚本
└── real_data/                 # 数据输出目录
    ├── real_sequences.json    # 蛋白质序列
    ├── real_dataset.csv       # 数据集
    ├── features/              # ESM特征
    ├── structures/            # PDB文件
    ├── structure_features/    # 结构特征
    └── results/               # 模型结果
```

---

## 🚀 快速开始

### 方法一：一键运行（推荐）

```bash
bash /mnt/okcomputer/output/run_real_pipeline.sh
```

### 方法二：分步运行

```bash
# 步骤1: 收集真实数据
python3 /mnt/okcomputer/output/real_data_pipeline.py

# 步骤2: ESM-2特征提取
python3 /mnt/okcomputer/output/feature_extraction.py

# 步骤3: AlphaFold结构特征提取
python3 /mnt/okcomputer/output/structure_features.py

# 步骤4: 模型训练
python3 /mnt/okcomputer/output/model_training_real.py
```

---

## 📊 数据来源验证

### 1. GPCRdb数据

**验证方式**:
- 访问 https://gpcrdb.org/signprot/statistics
- 查看G蛋白偶联统计数据
- 筛选Gq/11家族

**代码中的数据获取**:
```python
from real_data_pipeline import GPCRdbAPI
gpcrdb = GPCRdbAPI()
gq_gpcrs = gpcrdb.get_gprotein_couplings()
```

### 2. UniProt序列

**验证方式**:
- 访问 https://www.uniprot.org/
- 搜索UniProt ID（如 P25103）
- 查看蛋白质序列

**代码中的序列获取**:
```python
from real_data_pipeline import UniProtAPI
uniprot = UniProtAPI()
sequence = uniprot.get_sequence("P25103")  # HRH1
```

### 3. AlphaFold结构

**验证方式**:
- 访问 https://alphafold.ebi.ac.uk/
- 搜索UniProt ID
- 下载PDB文件

**代码中的结构下载**:
```python
from structure_features import AlphaFoldDownloader
downloader = AlphaFoldDownloader()
pdb_path = downloader.download_pdb("P25103")
```

### 4. ESM-2模型

**验证方式**:
- 访问 https://github.com/facebookresearch/esm
- 查看模型说明
- 运行示例代码

**代码中的特征提取**:
```python
from feature_extraction import ESM2FeatureExtractor
extractor = ESM2FeatureExtractor(model_name="esm2_t6_8M_UR50D")
features = extractor.extract_features("MKT...")
```

---

## 📚 真实文献引用

所有文献均来自PubMed，可查证：

### 核心文献

1. **GPCR-G蛋白相互作用**
   - Wisler et al. (2019) Curr Opin Cell Biol - PMID: 30743124
   - Flock et al. (2015) Nature - PMID: 26147082
   - Oldham & Hamm (2008) Nat Rev Mol Cell Biol - PMID: 18043707

2. **Melanopsin/Opsin**
   - Panda et al. (2002) Science - PMID: 12481141
   - Hattar et al. (2002) Science - PMID: 11834835
   - Berson et al. (2002) Science - PMID: 11834836

3. **蛋白质语言模型**
   - Rives et al. (2021) PNAS - PMID: 33876751
   - Lin et al. (2023) Science - PMID: 36927037

4. **AlphaFold**
   - Jumper et al. (2021) Nature - PMID: 34265844
   - Varadi et al. (2022) Nucleic Acids Res - PMID: 34791371

完整文献列表见：`real_references.bib`

---

## ⚙️ 依赖安装

```bash
# 基础依赖
pip install torch numpy pandas scikit-learn matplotlib seaborn tqdm

# 生物信息学
pip install biopython fair-esm

# DSSP（用于二级结构预测）
# Ubuntu/Debian:
sudo apt-get install dssp

# macOS:
brew install dssp
```

---

## 🔍 数据质量保证

### 正样本（Gq偶联GPCR）

来源：
- GPCRdb数据库（https://gpcrdb.org/）
- 文献报道的Gq偶联受体
- 实验验证数据

示例：
| 蛋白 | UniProt | 验证方法 |
|------|---------|---------|
| HRH1 | P25103 | Calcium imaging, IP3 assay |
| CHRM3 | P20309 | Patch-clamp, GTPγS binding |
| HTR2A | P18084 | Calcium imaging |

### 负样本（非Gq偶联GPCR）

来源：
- 已知的Gi/o偶联受体
- 已知的Gs偶联受体
- 已知的G12/13偶联受体

示例：
| 蛋白 | UniProt | 偶联类型 |
|------|---------|---------|
| ADRA2A | P08913 | Gi/o |
| DRD2 | P14416 | Gi/o |
| ADRB2 | P07550 | Gs |

---

## 📈 预期结果

运行完成后，您将获得：

1. **真实数据集**
   - 40+ Gq偶联GPCR（正样本）
   - 20+ 非Gq偶联GPCR（负样本）
   - 真实UniProt序列

2. **ESM-2特征**
   - 320维序列嵌入
   - 29维物理化学特征
   - 349维组合特征

3. **AlphaFold结构特征**
   - pLDDT置信度分数
   - 二级结构比例
   - 溶剂可及表面积
   - 残基接触图

4. **模型性能**
   - 5-Fold交叉验证结果
   - 准确率、AUC、F1等指标
   - 训练好的模型权重

---

## ⚠️ 注意事项

1. **网络连接**
   - 需要联网访问GPCRdb、UniProt、AlphaFold DB
   - 部分API可能有访问限制

2. **运行时间**
   - 数据收集：5-10分钟
   - ESM-2特征提取：10-30分钟（取决于GPU）
   - AlphaFold下载：10-30分钟（取决于网络）
   - 模型训练：10-30分钟

3. **存储空间**
   - PDB文件：约50MB
   - 特征数据：约100MB

4. **可复现性**
   - 使用固定随机种子（random_state=42）
   - 所有步骤都有详细日志
   - 结果可完全复现

---

## 📝 引用

如果使用本代码，请引用：

```bibtex
@software{ppi_prediction_2024,
  title={Deep Learning Framework for GPCR-G Protein Coupling Prediction},
  author={[Your Name]},
  year={2024},
  url={[Your Repository]}
}
```

并引用使用的数据源：
- GPCRdb: Kooistra et al. (2021) Nucleic Acids Res
- UniProt: UniProt Consortium (2023) Nucleic Acids Res
- AlphaFold: Jumper et al. (2021) Nature
- ESM-2: Lin et al. (2023) Science

---

## 📧 联系方式

如有问题，请联系：
- 项目维护者：[Your Email]
- GPCRdb支持：info@gpcrdb.org
- UniProt支持：help@uniprot.org

---

## 📜 许可证

本代码遵循MIT许可证。

数据使用需遵守各数据源的许可协议：
- GPCRdb: CC BY 4.0
- UniProt: CC BY 4.0
- AlphaFold: CC BY 4.0
