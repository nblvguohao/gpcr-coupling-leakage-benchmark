# 项目文件组织方案 & 一区发表差距评估

> 生成日期: 2026-05-07

---

## 第一部分：文件分类与组织方案

### 一、根目录保留（核心论文文件）

| 文件 | 说明 |
|------|------|
| `MANUSCRIPT.md` | 完整论文Markdown版本 |
| `MANUSCRIPT.tex` | LaTeX源码（主稿件） |
| `MANUSCRIPT.pdf` | 编译后的论文PDF |
| `README.md` | 项目总README（合并现有多个README） |
| `real_references.bib` | 参考文献BibTeX |

### 二、`src/` — 源代码按功能分组

#### `src/models/` — 模型架构与训练
| 文件 | 优先级 | 说明 |
|------|--------|------|
| `cross_attention_model.py` | ★★★ | 核心Cross-Attention模型架构 |
| `train_paired_cross_attention_650m.py` | ★★★ | 650M CA训练（论文主要结果） |
| `train_paired_baselines_650m.py` | ★★★ | MLP/RF/XGBoost基线（论文Table 1补充） |
| `train_paired_cross_attention.py` | ★★ | 8M CA训练（早期消融） |
| `run_gprot_650m_experiment.py` | ★★★ | 650M G蛋白全实验入口 |
| `run_final_ablation.py` | ★★★ | 最终消融实验 |
| `train_gsca.py` | ★★ | GSCA模型（探索性） |
| `train_ipl.py` / `train_ipl_v2.py` | ★★ | IPL模型（探索性） |

#### `src/features/` — 特征提取
| 文件 | 优先级 | 说明 |
|------|--------|------|
| `extract_650m_features.py` | ★★★ | 650M ESM-2特征提取 |
| `extract_icl_features_650m.py` | ★★★ | 650M ICL2/3特征提取（论文核心） |
| `extract_alphafold_features_paired.py` | ★★★ | AlphaFold结构特征提取 |
| `extract_gprotein_650m.py` | ★★★ | G蛋白650M特征提取 |
| `compute_geometric_alphafold_features.py` | ★★★ | AlphaFold几何描述符计算 |
| `compute_pae_features.py` | ★★★ | PAE矩阵特征计算 |
| `compute_icl_plddt.py` | ★★ | ICL pLDDT计算 |
| `compute_sgsp_embeddings.py` | ★★ | SGSP嵌入计算 |
| `compute_prauc.py` | ★★★ | PRAUC计算 |
| `convert_650m_to_meanpool.py` | ★★ | 嵌入mean-pool转换 |
| `convert_650m_to_meanpool_streaming.py` | ★★ | streaming版转换 |
| `extract_icl_features.py` | ★★ | 8M ICL特征提取（早期） |
| `extract_all_gpcr_esm_features.py` | ★★ | 批量GPCR ESM提取 |
| `extract_all_g_protein_features.py` | ★★ | 批量G蛋白特征提取 |

#### `src/cv/` — 交叉验证与评估
| 文件 | 优先级 | 说明 |
|------|--------|------|
| `paired_cross_validation_enhanced_v2_650m.py` | ★★★ | **主CV脚本**（产生论文核心结果） |
| `paired_cross_validation_enhanced.py` | ★★ | Enhanced CV v1 |
| `paired_cross_validation_enhanced_v2.py` | ★★ | Enhanced CV v2 (8M) |
| `paired_cross_validation_650m.py` | ★★ | 650M CV |
| `paired_cross_validation.py` | ★★ | 8M CV |
| `independent_test_eval.py` | ★★ | 独立测试评估 |
| `method_comparison.py` | ★★ | 方法比较 |
| `statistical_significance_test_paired.py` | ★★★ | 统计显著性检验（Table S4） |
| `ablation_study.py` | ★★ | 早期消融研究 |

#### `src/analysis/` — 可解释性分析
| 文件 | 优先级 | 说明 |
|------|--------|------|
| `gradient_attribution_650m.py` | ★★★ | 梯度归因分析（Figure 6） |
| `shap_attribution_paired.py` | ★★★ | SHAP配对分析（论文结果） |
| `shap_attribution_icl.py` | ★★★ | SHAP ICL分析 |
| `map_shap_to_residues.py` | ★★ | SHAP残基映射 |
| `analyze_logpso_failure.py` | ★★ | LOGPSO失败分析 |

#### `src/data/` — 数据获取与处理
| 文件 | 优先级 | 说明 |
|------|--------|------|
| `fetch_gpcrdb_couplings.py` | ★★★ | GPCRdb耦合数据获取 |
| `fetch_uniprot_sequences.py` | ★★ | UniProt序列获取 |
| `fetch_uniprot_tm_annotations.py` | ★★★ | TM拓扑注释（论文方法） |
| `fetch_additional_gpcr_samples.py` | ★★ | 额外样本获取 |
| `fetch_samples_biopython.py` | ★★ | Biopython样本获取 |
| `build_paired_matrix.py` | ★★★ | 配对矩阵构建（论文Data） |
| `merge_additional_samples.py` | ★★ | 数据合并 |
| `fetch_gnaq.py` | ★ | GNAQ专用获取 |
| `run_tmhmm_batch.py` | ★ | TMHMM批量运行 |
| `run_hyperparam_search.py` | ★ | 超参数搜索入口 |

#### `src/figures/` — 图表生成
| 文件 | 优先级 | 说明 |
|------|--------|------|
| `generate_figures_for_manuscript.py` | ★★★ | 论文图表生成 |
| `generate_gpcr_ipn_figures.py` | ★★ | GPCR IPN图表 |
| `generate_schematic_figure.py` | ★★★ | 示意图生成（Figure 1） |
| `generate_supplementary_materials.py` | ★★★ | 补充材料生成 |
| `generate_result_report.py` | ★★ | 结果报告生成 |
| `generate_wetlab_candidates_650m.py` | ★★★ | 湿实验候选生成 |
| `generate_wetlab_candidates.py` | ★★ | 湿实验候选（8M版） |

### 三、数据目录（保留）

| 目录 | 说明 |
|------|------|
| `paired_dataset/` | **核心数据集**（1,639 pairs, 特征, 结果） |
| `data_for_server/` | 服务器数据集 |
| `figures/` | 已生成的论文图表 |
| `results/` | 实验结果 |

### 四、submission_package/（保留，已完整）

```
submission_package/
├── main_text/          # 主稿件LaTeX源码
├── supplementary/      # 补充材料LaTeX源码
├── cover_letter/       # 投稿信
├── figures/            # 投稿用图表
├── compile_latex.py
├── generate_figures.py
├── highlights.txt
└── README.md
```

### 五、reproducible_package/（保留，已完整）

```
reproducible_package/
├── src/                # 精简后的可复现代码
├── data/               # 核心数据（已脱敏）
├── figures/            # 可复现图表
├── results/            # 可复现结果
└── README.md
```

### 六、建议清理的文件

| 文件/目录 | 原因 |
|-----------|------|
| `__pycache__/` | Python编译缓存 |
| `*.aux`, `*.log`, `*.out`（根目录） | LaTeX临时文件 |
| `dssp-package.conda` / `mkdssp.exe` / `mkdssp_conda.exe` | 二进制包（可存dssp_data/） |
| `flushend.sty` / `stfloats.sty` | LaTeX模板（已在submission_package/） |
| `gproteindb_couplings.html` (8.7MB) | 原始HTML数据（若已处理则可删） |
| `submission_package.zip` / `submission_package_complete.zip` | 已解压，ZIP可删 |
| `gproteindb_coupling_datasets.html` | 原始HTML |
| `gproteindb_parse_debug.txt` | 调试日志 |
| `fetch_test.log` / `fetch_uniprot.log` | 日志 |
| `tmp_path.txt` | 临时路径 |
| `execution_status.json` / `iteration_log.json` | 执行状态日志 |
| `OPN4_Gq_interaction_labels.xlsx` / `Opsin4_sequences.txt` / `GNAQ_human.txt` / `gnaq_uniprot.json` | 原始数据（若已整合则删） |

### 七、建议归档（移入 `docs/`）的文档

| 文件 | 说明 |
|------|------|
| 各轮Critic Review（`critic_*.md`） | AI评审记录 |
| 各轮执行计划（`execution_plan.md`, `EXECUTION_PLAN_NEXT_PHASE.md`, `STRATEGY_C_EXECUTION_PLAN.md`） | 开发计划 |
| 实验设计文档（`experiment_design.md`, `experimental_evidence.json`） | 早期设计 |
| 各阶段报告（`PHASE1_COMPLETION_REPORT.md`, `PROGRESS_REPORT_20260408.md`, `PROJECT_COMPLETION_SUMMARY.md`等） | 阶段报告 |
| 技术文档（`technical_implementation.md`, `paper_framework_design.md`, `ALPHAFOLD_INTEGRATION_PLAN.md`, `ANALYSIS_MULTIMODAL.md`） | 技术设计 |
| 多个README变体（`README_FINAL.md`, `README_REAL_DATA.md`, `FINAL_SUMMARY.md`等） | 只保留根README |

---

## 第二部分：一区（Briefings in Bioinformatics）发表差距评估

### 目标期刊要求分析

**Briefings in Bioinformatics**（牛津出版社，2024 IF ~13+，中科院一区）：
- 要求：**方法创新 + 生物学洞见 + 可获取软件**
- 典型论文：开发新算法/数据库，在多个benchmark上展示优势
- 不接受：纯应用报告、增量改进

### 当前状态评分（1-10）

| 维度 | 评分 | 说明 |
|------|------|------|
| 方法新颖性 | 6/10 | 配对公式较新，但Cross-Attention非原创（借鉴D-SCRIPT） |
| 结果强度 | 7/10 | AUC 0.8619合理，但无显著超SOTA |
| 实验验证 | **2/10** | ⚠️ 无水实验证，仅有候选列表 |
| 数据规模 | 5/10 | 1,639 pairs、431 GPCRs，偏小 |
| 可重复性 | 8/10 | 有reproducible_package，但缺Docker/CI |
| 写作质量 | 7/10 | 结构完整，但语言需要润色 |
| 生物学洞见 | 6/10 | ICL对齐、维度匹配有洞见，但主要是负面发现 |
| 泛化能力 | 3/10 | LOGPSO仅~0.60 AUC，跨家族泛化差 |

### 关键差距

#### 差距1（最严重）：无水实验证
- 论文提到"wet-lab candidates"但**完全没有实验数据**
- 一区生物信息学期刊越来越期望实验验证
- **整改建议**：
  - **最小方案**：在论文中明确标注"验证进行中"，讨论预期结果
  - **推荐方案**：至少完成2-3个候选对的BRET/Co-IP验证，将结果加入论文
  - **理想方案**：系统性验证5-10个预测，分析真/假阳性模式

#### 差距2：方法创新不足
- Cross-attention直接借鉴现有PPI文献（D-SCRIPT 2021, EGRET 2022）
- ESM-2特征提取为标准做法
- ICL特征为简单统计量
- **整改建议**：
  - **强制**：与最相关的SOTA方法（Miglionico 2025）进行直接基准对比
  - **推荐**：开发更创新的架构（残基级attention、G蛋白家族embedding层）
  - **推荐**：增加多任务学习（同时预测耦合家族 + 结合强度）
  - **可选**：对比AlphaFold3/MultiFOLD结构预测方法

#### 差距3：LOGPSO泛化能力差
- AUC仅~0.60，说明学到的主要是family-specific模式
- **整改建议**：
  - **推荐**：实现G蛋白家族embedding + multi-task learning
  - **推荐**：增加few-shot learning策略（每个家族少量样本微调）
  - **推荐**：尝试使用交错数据（interleaved data）训练
  - **最小方案**：深入分析为何LOGPSO失败，转化为methodological insight

#### 差距4：结果对比不够充分
- Table 2中对Miglionico 2025的对比缺少直接benchmark
- 缺少与最新方法（AlphaFold3-based, graph-based）的公平比较
- **整改建议**：
  - **强制**：在统一的数据集/评估协议上比较SOTA方法
  - **推荐**：使用GPCRdb最新release的数据重新评估
  - **推荐**：增加AlphaFold3结构特征对比实验

#### 差距5：代码可重复性不足
- 没有Docker/Singularity容器
- 没有CI/CD测试
- environment.yml（conda）比requirements.txt更好
- **整改建议**：
  - **推荐**：提供Dockerfile或devcontainer
  - **推荐**：GitHub Actions自动运行核心实验
  - **推荐**：Zenodo归档完整数据和模型权重（论文要求）

#### 差距6：写作和呈现
- 文稿使用markdown而非期刊模板
- 图的分辨率和样式需要符合期刊要求
- 补充材料格式
- **整改建议**：
  - **强制**：使用Briefings in Bioinformatics官方LaTeX模板
  - **推荐**：专业润色（native speaker proofreading）
  - **推荐**：所有图表统一风格，分辨率≥300 DPI

#### 差距7：数据规模
- 1,639 pairs偏小，特别是与近期方法比较时
- **整改建议**：
  - **推荐**：从GPCRdb + IUPHAR + 文献挖掘扩展到3,000+ pairs
  - **推荐**：集成多物种数据（human + mouse + rat）
  - **最小方案**：明确讨论数据限制

### 优先级整改路线

```
紧急（投稿前必须完成）：
├── ■ 与Miglionico 2025公平对比
├── ■ 官方LaTeX模板重排
├── ■ 补充LOGPSO失败分析
├── ■ 英文润色
└── ■ Zenodo归档代码+数据

重要（显著提高接受率）：
├── ■ 湿实验验证（>3个候选对）
├── ■ Docker/CI配置
├── ■ 数据扩展（>2,500 pairs）
├── ■ 多任务学习改进
└── ■ 高分辨率专业图表

加分（边缘翻转为可接受）：
├── ■ 残基级cross-attention
├── ■ 与AlphaFold3 benchmark对比
├── ■ GitHub Actions自动复现
└── ■ 交互式Web demo
```

### 时间估计

| 整改级别 | 预计工作量 | 预计时间 |
|----------|-----------|----------|
| 最小（补实验对比 + 格式化） | ~2周 | 高概率可提交 |
| 推荐（+ 湿实验验证 + 数据扩展） | ~2-3月 | 大幅提高接受率 |
| 理想（+ 模型创新 + 全流程） | ~3-6月 | 一区顶刊级别 |

### 当前最薄弱的三个环节

1. **无水实验证** ← 这是审稿人最可能拒稿的原因
2. **方法创新不足** ← Cross-attention + ESM-2是成熟技术组合
3. **LOGPSO泛化差** ← 审稿人会质疑方法的实用性

---

*本文件由Claude Code生成，基于项目文件完整扫描和论文内容分析。*
