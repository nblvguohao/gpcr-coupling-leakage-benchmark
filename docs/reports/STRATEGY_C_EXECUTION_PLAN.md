# Strategy C 详细补强执行方案

**目标**: 在 3–4 个月内将项目从当前 "不可辩护的 AUC 0.83 单方分类器" 升级至可冲击 *Bioinformatics* / *Briefings in Bioinformatics* (IF 4–10) 的高质量工作。

**核心问题必须解决**:
1. **GNAQ 特征零贡献问题** — 当前所有样本使用固定 GNAQ 向量，模型退化为 GPCR 单方分类
2. **样本量不足** — 86 独立样本已接近 SVM 天花板，无法展示深度学习价值
3. **结构特征过时** — 使用 21 残基疏水性窗口，未整合真实 3D 结构
4. **缺乏实验验证** — 纯计算预测在 1区/2区 顶刊中竞争力弱

---

## 一、成功标准（Go/No-Go Criteria）

| 里程碑 | 最低标准 (Go) | 优秀标准 (Stretch) |
|--------|--------------|-------------------|
| **最终 AUC** | LOCO / Cluster-aware CV ≥ 0.88 | ≥ 0.90 |
| **G蛋白信息利用率** | Gαq vs Gi/Gs 差异特征 SHAP \|importance\| > 0 且排名前 50 中占 ≥ 5% | 占 ≥ 15% |
| **独立测试集** | 构建时间/序列分离的独立测试集 ≥ 20 样本 | ≥ 30 样本 |
| **实验验证** | 3 个预测的湿实验 (BRET/Co-IP) | 5 个预测 + 1 个机制突变验证 |
| **样本总量** | ≥ 150 独立蛋白质 | ≥ 200 独立蛋白质 |

---

## 二、三阶段时间线与详细任务

### Phase 1: 架构重塑 — 真实配对与结构整合 (Week 1–4)

**目标**: 从 "GPCR单方分类" 转变为 "GPCR-G蛋白配对特异性预测"

#### Week 1: 数据收集 — 引入 G 蛋白配对变化
**负责人/工具**: 你自己 + Python脚本 + GPCRdb/UniProt API

**具体任务**:
1. **获取 GPCR-G protein coupling 全景数据**
   - 从 **GPCRdb** (https://www.gpcrdb.org/) 下载 `coupling` 数据（human + multiple species）
   - 或爬取最新的 GPCR-G protein profiling 数据集（例如 Inoue et al., 2019; 及后续扩展数据集）
   - 重点收集：与 **Gαq**、**Gαi/o**、**Gαs**、**Gα12/13** 有明确偶联记录的 GPCR

2. **构建 "配对矩阵"**
   - 每一行 = 一个 (GPCR, G_protein) 配对
   - 标签 = binary: 1 如果该 GPCR 偶联该 G蛋白，0 如果不偶联
   - 关键：同一个 GPCR 可能同时出现在多个配对中（例如偶联 Gαq=1，偶联 Gi=0）
   - 这样 GPCR 特征不再固定，G蛋白特征也不再固定，模型必须**同时利用两者**

3. **数据清洗规则（必须执行）**
   - 去除完全重复的 (GPCR_seq, G_protein_seq) 对
   - 按 30% 序列同源性聚类（复用现有 `sequence_identity` 函数）
   - **关键约束**: 同一个 GPCR 的所有配对必须落在**同一个 cluster** 里（防止信息泄露）

**交付物**:
- `paired_dataset/pairing_matrix_raw.csv` — 原始配对矩阵
- `paired_dataset/sequence_clusters.json` — GPCR 聚类结果
- `paired_dataset/deduplicated_pairs.json` — 去重后配对

**Go/No-Go Checkpoint (Week 1 周末)**:
- ✅ 去重后配对数 ≥ 400（这样即使聚类拆分，仍有足够训练数据）
- ❌ 如果 < 300，则需要扩大物种范围（加入 mouse, rat, zebrafish orthologs）或寻找额外数据源

---

#### Week 2: 特征工程升级 — 真实结构与 TM 拓扑
**负责人/工具**: Python + Biopython + TMHMM2 + AlphaFold DB

**具体任务**:
1. **TM 拓扑预测（替换掉 21-residue 窗口）**
   - 安装/使用 **TMHMM2** (http://www.cbs.dtu.dk/services/TMHMM-2.0/) 或 **Phobius** 批量预测所有 GPCR 的跨膜拓扑
   - 输出解析：提取 TM1-7 的起止位置、ICL1-3 起止位置、ECL1-3 起止位置
   - 更新 `shap_attribution_corrected.py` 中的 `approximate_tm_regions` 函数，改用 TMHMM2 的真实结果

2. **AlphaFold 真实结构特征提取（真正启用）**
   - 使用现有 `structure_features.py` 中的 `AlphaFoldDownloader` 批量下载所有新样本的 AlphaFold PDB
   - 提取以下特征并保存为结构化 JSON:
     - **pLDDT**: mean, std, high-confidence ratio (>70, >90)
     - **DSSP 二级结构**: 螺旋/折叠/转角/无规卷曲比例
     - **SASA**: mean, std, buried residues ratio
     - **残基接触图**: contact density, mean contacts per residue
   - **新增**：计算 **胞内环 (ICL1/2/3) 的平均 pLDDT**，作为结构可靠性指标

3. **G蛋白结构特征**
   - 收集 Gαi1/2/3、Gαs、Gα12/13 的代表性序列（UniProt IDs: P63096, P04899, P50148 等）
   - 用 `extract_gnaq_esm_feature.py` 的模板批量提取所有 G 蛋白的 ESM-2 mean/CLS embedding

**交付物**:
- `paired_dataset/tmhmm_topology.json`
- `paired_dataset/alphafold_structure_features.json`
- `paired_dataset/g_protein_esm_features.json`

**Go/No-Go Checkpoint (Week 2 周末)**:
- ✅ AlphaFold PDB 下载成功率 ≥ 90%
- ✅ TMHMM2 成功预测 ≥ 95% 的 GPCR 序列
- ❌ 任一失败率 > 20% → 改用 DeepTMHMM (在线/本地) 或 AlphaFold 结构中的 B-factor 估计 TM 区域

---

#### Week 3: 模型架构重构 — 从静态模板到动态配对
**负责人/工具**: PyTorch + sklearn

**具体任务**:
1. **设计新的特征拼接逻辑**
   - 输入样本 = (GPCR_esm_320, GPCR_struct_16, G_protein_esm_320, G_protein_struct_N)
   - SVM 基线: 拼接 `[GPCR_esm, G_prot_esm]` = 640-d（此时 G 蛋白不同，向量有意义了）
   - 多模态: `[GPCR_esm, GPCR_struct, G_prot_esm, G_prot_struct]`

2. **重写训练脚本**
   - 新建 `paired_cross_validation.py`
   - **Cluster-aware CV 逻辑升级**:
     - Fold 拆分必须基于 **GPCR cluster**（同一个 GPCR 的所有配对必须在同一 fold 中）
     - 测试集中的 GPCR cluster 在训练集中**完全不可见**
     - 这保证了模型必须泛化到**全新的 GPCR**，而不仅仅是新的配对
   - **新增 Leave-One-G-Protein-Subtype-Out (LOGPSO)**:
     - 例如：训练集包含 Gαq/Gαi/Gα12 的配对，测试集只包含 Gαs 的配对
     - 如果模型在此设置下仍表现良好，说明它学到了**跨 G 蛋白 subtype 的通用偶联规则**

3. **SHAP 兼容性预留**
   - 确保新模型输出 `predict_proba` 接口，方便后续 SHAP 分析

**交付物**:
- `paired_cross_validation.py` — 新 CV 脚本
- `train_paired_svm.py` — 配对 SVM 基线
- `train_paired_cross_attention.py` — 配对神经网络（可选，若样本量≥150时启用）

---

#### Week 4: Phase 1 验证与 Baseline 跑通
**具体任务**:
1. 使用当前已收集到的所有数据（即使样本量暂未满 150）跑通新 pipeline
2. 记录以下指标:
   - Random CV AUC
   - Cluster-aware CV AUC
   - LOCO CV AUC
   - LOGPSO CV AUC
3. **检查 G 蛋白特征的 SHAP 重要性**:
   - 运行 `shap_attribution_paired.py`
   - 验证 G蛋白维度是否有非零重要性
   - 检查不同 G 蛋白 subtype 对预测的贡献差异

**Go/No-Go Checkpoint (Week 4 周末)**:
- ✅ G蛋白维度 SHAP 重要性 **非零**
- ✅ Cluster-aware AUC ≥ 0.78（早期数据可能样本不足，但不允许低于 0.75）
- ❌ 如果 G蛋白维度仍为零 → 诊断问题（可能是 GPCR 特征过于主导，需尝试特征差值/点乘，或用 Siamese 架构强制学习配对交互）
- ❌ 如果 AUC < 0.75 → 检查标签噪声和数据源可靠性

---

### Phase 2: 数据扩增与深度学习方法验证 (Week 5–8)

**目标**: 样本量 ≥ 150 独立蛋白，展示深度学习在真实配对任务上的价值

#### Week 5–6: 数据集扩充
**数据来源**:
1. **GPCRdb 全库耦合数据** — 最新的大规模 profiling 数据集（许多 2022–2024 年的研究提供了 300+ GPCR × 4 G protein 的配对矩阵）
2. **文献挖掘** — 使用已写好的 `fetch_additional_gpcr_samples.py` 和 `merge_additional_samples.py` 扩展
3. **Ortholog 扩充** — 对已有 human GPCR，加入 mouse、rat、zebrafish 的直系同源序列作为独立样本（确保它们有自己的 cluster）

**质量控制**:
- 每个新样本必须检查序列完整性（去除含有大量 'X' 的序列）
- 去除全长 < 250 aa 的短序列（不完整的 GPCR）
- 使用现有的 `sequence_identity` 函数去重

**交付物**:
- `paired_dataset/extended_pairing_matrix_v2.csv` — 扩展后配对矩阵
- 统计报告：独立 GPCR 数、总配对数、各 G 蛋白 subtype 的配对分布

**Go/No-Go Checkpoint (Week 6 周末)**:
- ✅ 独立 GPCR 数 ≥ 120（向 150 努力）
- ✅ 总配对数 ≥ 600
- ✅ 每个 G 蛋白 subtype 的配对数 ≥ 100（保证 balance）

---

#### Week 7: 超参数搜索与模型对比
**具体任务**:
1. **SVM Grid Search**: C ∈ [0.1, 1, 10, 100], kernel ∈ [rbf, linear], class_weight ∈ [balanced, None]
2. **Cross-Attention Network 调参**:
   - 由于样本量增加，适当降低 dropout（0.3–0.5）
   - 尝试 `esm2_t33_650M`（如果 GPU 内存允许） vs `esm2_t6_8M`
   - 注意力头数、隐藏层大小搜索
3. **Feature ablation**:
   - ESM-2 only
   - ESM-2 + AlphaFold structure
   - ESM-2 + TMHMM topology
   - Full multimodal

**交付物**:
- `hyperparam_search_results.json`
- 最佳模型配置

---

#### Week 8: 独立测试集构建与最终评估
**关键原则**：测试集必须在时间和信息上完全隔离

**独立测试集来源（三选一或组合）**:
1. **时间隔离集**: 2023–2025 年新发表的 GPCR-G protein coupling 数据（在训练集中不可用）
2. **物种隔离集**: 仅包含非哺乳动物（如鸟类、两栖类、某些鱼类）的 GPCR
3. **家族隔离集**: 从训练中完全移除某一 GPCR 家族（如全部 peptide receptors），在测试集中评估

**具体任务**:
1. 构建 ≥ 20 样本的独立测试集
2. 用全部训练数据重新训练最佳模型
3. 在独立测试集上报告 AUC、Accuracy、Precision、Recall、F1
4. 分析错误案例：哪些家族的 GPCR 预测失败？是否与结构特殊性有关？

**Go/No-Go Checkpoint (Week 8 周末)**:
- ✅ Cluster-aware CV AUC ≥ 0.85
- ✅ 独立测试集 AUC ≥ 0.82
- ✅ 深度学习模型 AUC ≥ SVM（否则无法支撑 "深度学习" 卖点）
- ❌ 任一未达标 → 进入 2 周缓冲期，尝试更大的 ESM 模型或图神经网络（GNN + AlphaFold 结构）

---

### Phase 3: 湿实验验证与手稿定稿 (Week 9–12)

**目标**: 3–5 个预测的实验验证 + 论文全文重写

#### Week 9: 预测结果筛选与实验设计
**预测筛选标准（用于湿实验验证）**:
1. **高置信度正例**: model probability > 0.85，但目前文献中尚无明确 Gαq 偶联记录（ novel prediction ）
2. **高置信度负例**: model probability < 0.15，但结构上与正例相似（hard negative）
3. **边界案例**: probability ≈ 0.5，预期实验结果可能有细微差异（适合探索机制）

**具体任务**:
1. 从模型输出中筛选 5–8 个候选蛋白
2. 与湿实验合作者确认可及性（是否能获得 cDNA / 细胞系）
3. 设计 BRET 或 Co-IP 实验方案（使用已有的 `EXECUTION_PLAN_NEXT_PHASE.md` 作为基础）

**交付物**:
- `wetlab_candidates.json` — 候选蛋白列表及预测分数
- `wetlab_experimental_design.md` — 实验方案细化

---

#### Week 10–11: 湿实验执行与结果收集
**实验内容**:
- **BRET assay**: 检测 GPCR 激活后 Gαq 的招募效率
- **Co-IP**: 验证 GPCR 与 Gαq 的物理相互作用
- **对照**: 使用已知 Gαq-coupling 受体（如 CHRM3, HRH1）作为阳性对照；已知 Gi-coupling 受体（如 DRD2）作为阴性对照

**风险缓冲**:
- 如果湿实验合作者在 Week 11 仍无法完成，至少获取到 **初步的转染表达结果**（Western blot 确认蛋白表达），可作为 "ongoing experimental validation" 写入论文和补充材料。

**Go/No-Go Checkpoint (Week 11 周末)**:
- ✅ 至少 3 个候选蛋白获得可重复的实验结果（阳/阴性均可）
- ⚠️ 如果仅完成 1–2 个 → 论文中降低 "experimental validation" 的权重，改为 "validation pipeline established"

---

#### Week 12: 论文重写与投稿准备
**具体任务**:
1. **全文重写 `PAPER_DRAFT.md`**:
   - 引言：强调 "从配对视角理解 GPCR-G protein 选择性"，而非单方分类
   - 方法：详细描述配对矩阵构建、TMHMM2、AlphaFold 特征、Cluster-aware CV、LOGPSO
   - 结果：报告所有基线（Random / Cluster-aware / LOCO / LOGPSO / 独立测试集）
   - 讨论：坦诚讨论早期版本的数据泄露和修正过程（作为研究改进的正面叙事）

2. **Figure 定稿**:
   - Fig 1: 配对矩阵构建与数据泄露控制流程
   - Fig 2: AlphaFold + TMHMM2 特征示例（2–3 个代表性 GPCR）
   - Fig 3: 性能对比（SVM vs Cross-Attention，多种 CV 策略）
   - Fig 4: SHAP 归因（GPCR 维度 + G蛋白维度同时展示，ICL2/ICL3 高亮）
   - Fig 5: 独立测试集表现 + 湿实验验证结果（BRET/Co-IP）

3. **期刊选择最终决策**:
   - 如果最终 AUC ≥ 0.88 且有 3+ 湿实验验证 → *Briefings in Bioinformatics* 或 *PLOS Computational Biology*
   - 如果 AUC 0.85–0.88 但方法学创新强（LOGPSO、多模态、结构整合）→ *Bioinformatics*
   - 如果湿实验未完全完成 → *BMC Bioinformatics*（保底）或先投 bioRxiv

4. **预印本发布**:
   - 在提交 peer review 前，将修正后的方法学和主要结果发布到 **bioRxiv**
   - 这能保护优先权，并获得同行反馈以改进正式投稿版本

---

## 三、关键技术风险与应对措施

| 风险 | 影响 | 应对措施 |
|------|------|---------|
| GPCRdb 限制下载/API不稳定 | 数据收集受阻 | 提前下载完整数据库快照；备用数据源：UniProt + 文献手动整理 |
| AlphaFold PDB 缺失率高 | 结构特征无法提取 | 使用 ColabFold/AlphaFold2 本地预测缺失结构（需 GPU） |
| 深度学习仍无法超越 SVM | 核心卖点缺失 | 重点转向 "多模态可解释性" 和 "结构特征贡献量化"，淡化 SVM vs DL 对比 |
| 湿实验时间超预期 | 错过投稿窗口 | 并行推进：在实验进行的同时完成计算部分手稿，将湿实验作为后续 letter/revision 补充 |
| 配对矩阵极度稀疏 | 正负样本不平衡 | 使用 focal loss / 类别权重 / 过采样；或者改用多任务学习（同时预测 4 个 G protein subtype） |

---

## 四、每周执行检查清单模板

建议在每周结束时检查以下项目：

- [ ] 本周数据/代码是否已提交到 git（哪怕是本地 repo）?
- [ ] 新增的样本是否通过了去重和同源聚类检查?
- [ ] 本周是否有至少一个 Go/No-Go Checkpoint 被评估?
- [ ] 如果发现数据问题，是否已在文档中记录并通知合作者?
- [ ] 湿实验进度是否与计算进度同步?

---

## 五、费用与资源估算

| 项目 | 估算 | 备注 |
|------|------|------|
| GPU 计算 | 现有 CUDA 环境已够用 | `esm2_t33_650M` 需约 16GB VRAM；若不足可用 `esm2_t12_35M` 折中 |
| AlphaFold2 本地运行 | 可选 | 若大量缺失 PDB，可租用云端 A100（约 ¥200–400/天）跑 1–2 天即可 |
| TMHMM2/Phobius | 免费（学术） | TMHMM2 在线版限制批量；Phobius 本地版可免费用于学术 |
| 湿实验 (BRET/Co-IP) | ¥3万–8万 | 取决于合作者是否有现有试剂和细胞系 |
| 论文润色/投稿 | ¥3000–6000 | 若投 Briefings in Bioinformatics，建议专业润色 |

---

## 六、快速启动检查表（你可以立刻开始做的事）

**今天就做（30 分钟内可启动）**:
1. [ ] 打开 GPCRdb (https://www.gpcrdb.org/)，在 "Couplings" 页面下载最新的 `gpcr-g_protein_couplings.xlsx`
2. [ ] 检查文件中有多少条目涉及 Gαq / Gαi / Gαs / Gα12/13
3. [ ] 用邮件/微信联系湿实验合作者，确认未来 2 个月内是否有档期做 3–5 个样本的 BRET/Co-IP

**本周内完成**:
4. [ ] 跑通 TMHMM2 或 Phobius 的本地安装/在线批量提交脚本
5. [ ] 批量下载现有 86 样本 + 新增候选样本的 AlphaFold PDB（使用 `structure_features.py`）
6. [ ] 用 `extract_gnaq_esm_feature.py` 的模板批量提取 Gαi1、Gαs、Gα12 的 ESM-2 特征

---

*方案至此结束。只要你按周推进本计划，每一步都有明确的交付物和 checkpoint，可以将项目从当前状态提升到可发表水平。*

**需要我帮你执行 Phase 1 第一周的具体代码（下载 GPCRdb 数据、构建配对矩阵）吗？**
