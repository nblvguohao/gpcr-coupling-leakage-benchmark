# FeatureEngineering 改进建议执行报告

**执行日期**: 2026-04-09  
**执行内容**: 优先落实 FeatureEngineering 批判审查中的 5 项改进建议

---

## 1. 纠正 Gαq 特征 — 使用真实 human GNAQ (P50148)

### 问题
原始代码将"第一个正样本 GPCR"的 ESM-2 embedding 冒充为 Gαq 特征：
```python
# svm_baseline_100samples.py / homology_clustered_cv.py
gq_id = [k for k, v in labels_dict.items() if v == 1][0]
gq_feature = np.array(features_dict[gq_id])
```

### 执行
- 从 UniProt REST API 获取真实 human GNAQ (P50148) 序列
  - 序列长度: 359 aa
  - 蛋白名: Guanine nucleotide-binding protein G(q) subunit alpha
- 使用 `esm2_t6_8M_UR50D` 提取真实 GNAQ 的 ESM-2 embedding
  - 生成文件: `server_sync/extended_data/features/gnaq_esm_features.json`
  - 包含: `mean_pooling` (320d), `cls_token` (320d), `residue_level` (359×320)
- 修改 `homology_clustered_cv.py`
  - 引入 `GNAQ_FEATURE_FILE` 路径常量
  - 将虚假 `gq_feature = X[gq_idx]` 替换为真实的 `gq_feature = np.array(gnaq_esm["mean_pooling"])`

### 结果
重新运行同源聚类交叉验证后，AUC 数值**未发生变化**（去重后随机 CV 仍为 0.842，LOCO 仍为 0.831）。
这说明原 AUC 数值对"第一个正样本"与"真实 GNAQ"的替换并不敏感，但方法论上已彻底消除"用 GPCR 冒充 G 蛋白"的学术诚信风险。

---

## 2. 统一结构特征描述 — 修正维度

### 问题
- 论文 3.2.2 声称 physicochemical features 有 **29 维**
- 实际代码 `extract_structure_features.py` 输出 **16 维**
- 论文 3.2.3 声称总维度 **1,367**
- 实际训练管道 `train_multimodal.py` 使用 **672 维** (320+16+320+16)

### 执行
- 修改 `PAPER_DRAFT.md` 3.2.2，将 29 维修正为 16 维，并明确列出：
  - 疏水性 (4)
  - 跨膜区预测 (1)
  - 电荷 (3)
  - 组成 (5，含长度)
  - 二级结构倾向 (2)
  - 末端疏水性 (已并入上述 4 维疏水性，总数 16)
- 修改 3.2.3，将总维度修正为 **1,328**（若使用全部拼接/差值/点乘）并注明：
  - SVM 基线仅使用 640-d (320 GPCR + 320 GNAQ)
  - 多模态模型实际输入 672-d (320+16+320+16)

---

## 3. 控制变量比较 mean pooling vs CLS token

### 问题
原论文声称 "Mean pooling performed better than CLS token"，但这是一个**混淆变量比较**：
- SVM 使用了 mean pooling (AUC 0.907)
- Cross-Attention NN 使用了 CLS token (AUC 0.849)
- 变量同时包含**模型架构**与**特征聚合方式**

### 执行
- 本地提取 100 样本的 CLS token 特征
  - 生成 `server_sync/extended_data/features/esm_features_100samples_cls.json`
- 编写 `svm_cls_vs_mean_corrected.py`
  - 在去重后的 86 样本上
  - 使用**同一 SVM 分类器**（RBF, C=10, balanced）
  - 使用**真实 GNAQ P50148 特征**
  - 分别测试 mean pooling 与 CLS token

### 结果 (`svm_cls_vs_mean_corrected.json`)

| 聚合方式 | AUC | Acc | F1 |
|---------|------|------|------|
| **CLS token** | **0.856 ± 0.064** | 0.825 | 0.825 |
| Mean pooling | 0.842 ± 0.084 | 0.801 | 0.811 |
| Δ (CLS - Mean) | **+0.014** | — | — |

**结论**: 在 SVM 上，CLS token 与 mean pooling 表现相近，CLS token 略优。原论文的"mean pooling > CLS"结论**不再成立**，该差异主要源于模型架构（SVM vs 深度神经网络）的混淆。

### 文档更新
- `PAPER_DRAFT.md` 4.1 添加了 "(updated after data-leakage correction and controlled comparison)" 的 key findings。
- 明确删除了 "Mean pooling performed better than CLS token" 的断言。
- 5.1 和结论部分同步修正。

---

## 4. 升级结构特征 — 现状说明

### 问题
代码中使用的结构特征停留在 1980 年代水平：21 残基窗口 Kyte-Doolittle 疏水性预测跨膜区，未使用 TMHMM/Phobius 或真实 AlphaFold 结构。

### 执行
- 代码库中**已存在** `structure_features.py`，支持从 AlphaFold PDB 提取：
  - pLDDT 分数
  - DSSP 二级结构
  - SASA（溶剂可及表面积）
  - 残基接触图
- **尚未执行**的改造工程（超出当前 immediate fix 范畴）：
  - 将 `structure_features.py` 整合进 `homology_clustered_cv.py` 的训练管道
  - 使用 TMHMM2/Phobius 获取 7-TM 拓扑并映射到 ESM-2 残基位置

**建议**: 此项需要额外时间（约 1–2 小时）完成 AlphaFold PDB 批量下载、特征对齐、重新实验。当前优先完成了更紧急的 GNAQ 修正和混淆变量澄清。

---

## 5. 生物学可解释性 — SHAP 归因准备

已将真实 GNAQ 特征和受控比较脚本准备就绪。下一步 SHAP 归因可以在修正后的 86 样本 + SVM (C=10, CLS token) 上执行，高重要性维度可映射回 ICL2/ICL3/TM 区域。

---

## 文件变更清单

| 文件 | 操作 | 说明 |
|------|------|------|
| `homology_clustered_cv.py` | 修改 | 使用真实 GNAQ P50148 替代虚假模板 |
| `PAPER_DRAFT.md` | 修改 | 修正维度描述、删除虚假结论、更新 AUC |
| `fetch_gnaq.py` | 新建 | 从 UniProt 获取 GNAQ 序列 |
| `extract_gnaq_esm_feature.py` | 新建 | 提取真实 GNAQ 的 ESM-2 embedding |
| `extract_cls_features_local.py` | 新建 | 本地提取 100 样本 CLS token |
| `svm_cls_vs_mean_corrected.py` | 新建 | 控制变量比较脚本 |
| `svm_cls_vs_mean_corrected.json` | 生成 | 比较结果 |
| `server_sync/extended_data/features/gnaq_esm_features.json` | 生成 | 真实 GNAQ ESM-2 特征 |
| `server_sync/extended_data/features/esm_features_100samples_cls.json` | 生成 | 100 样本 CLS token |
| `homology_cv_results.json` | 覆盖 | 使用真实 GNAQ 重新运行后的结果（数值未变） |
| `FIX_EXECUTION_REPORT.md` | 新建 | 本报告 |

---

## 下一步建议

1. **SHAP 归因** (`/execute 特征重要性归因`) — 已在修正数据上准备就绪
2. **手稿叙事重构** (`/summarize [目标期刊]`) — 基于 AUC ≈ 0.83 和"混淆变量被澄清"的新故事线
3. **AlphaFold 结构特征整合** — 作为可选增强项，可显著提升生物学可信度
