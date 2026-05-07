# 手稿叙事重构与期刊策略报告

**日期**: 2026-04-09  
**依据**: DataSplit 修正 (AUC 0.915→0.831) + FeatureEngineering 批判执行结果 + SHAP 归因结果

---

## 一、核心发现对原有叙事的冲击

| 原有叙事 | 修正后事实 | 冲击等级 |
|---------|-----------|---------|
| AUC = 0.907，显著优于深度学习 | LOCO AUC = 0.831，深度学习差异不再显著 | 🔴 致命 |
| 预测 GPCR-Gαq 偶联特异性 | GNAQ 特征固定不变，SHAP 重要性全为零；模型仅基于 GPCR 单方序列做二分类 | 🔴 致命 |
| Mean pooling > CLS token | 控制变量后 CLS (0.856) ≈ Mean (0.842) | 🟡 重大 |
| 29-d 结构特征 + 1,367 总维度 | 实际 16-d 结构特征，SVM 640-d、多模态 672-d | 🟡 重大 |
| 跨物种对比学习 + 多模态融合深度神经网络 | 无真实对比学习目标，深度神经网络无优势，Gαq  partner 无信息贡献 | 🔴 致命 |

**结论**: 原稿的核心卖点（"SVM 显著优于深度学习的小样本 PPI 基准"）在修正后**全部崩塌**。继续沿用原叙事投稿将面临严重的学术诚信和方法学质疑风险。

---

## 二、三种重构策略

### 策略 A：保守诚实路线（推荐）
**定位**: 方法学校正 / 数据泄露警示 / 保守基线建立  
**适合期刊**: *BMC Bioinformatics*, *PeerJ*, *F1000Research*, 或作为预印本 (bioRxiv)  
**风险**: 较低（叙事诚实，审稿人无法攻击数据造假）  
**收益**: 中等偏低（可发表，但影响力有限）

**核心叙事**: 
> "我们在构建 GPCR-Gαq 偶联预测数据集时发现了数据泄露（14 个重复序列）和特征构造缺陷（Gαq 模板非真实 G 蛋白序列）。在修正这些问题并引入同源感知交叉验证后，AUC 从 0.915 降至 0.831。进一步分析表明，当 Gαq 配偶体固定不变时，模型退化为单方序列分类器。本研究提出了一套数据清洗、去重、同源控制的工作流，为小样本蛋白质相互作用预测的方法学审查提供了警示案例。"

**标题选项**:
1. *Data leakage and feature construction pitfalls in small-sample GPCR-Gαq coupling prediction: a cautionary benchmark study*
2. *Correcting homology leakage in GPCR-Gαq coupling classifiers: a conservative baseline and methodological commentary*
3. *From AUC 0.91 to 0.83: identifying data leakage and partner-feature artifacts in protein-coupling prediction*

---

### 策略 B：机制探索路线
**定位**: ESM-2 生物学可解释性 + GPCR 序列签名发现  
**适合期刊**: *PLOS Computational Biology* (如果生物学故事够强), *Briefings in Bioinformatics* (需补充实验)  
**风险**: 中高（需要把"失败"的 PPI 预测包装成"成功的序列签名挖掘"）  
**收益**: 中高（若能说服审稿人 ESM-2 捕捉到了 ICL2/ICL3 的偶联签名）

**核心叙事**:
> "本研究利用 ESM-2 嵌入探索了 GPCR 序列中决定 Gαq 偶联的隐含签名。在经过去重和同源控制的 86 独立样本上，ESM-2 特征支持约 0.83 的 AUC。SHAP 分析显示，特定 ESM-2 维度（如 dim 193, 99, 113）携带了 strongest 的分类信号，其残基级投影在 ICL1、TM6、ICL2 区域具有较高权重。这些发现与 G 蛋白结合界面的结构生物学知识部分吻合，提示预训练蛋白语言模型在小样本场景下能够捕捉功能相关的序列模式。"

**必需补充**:
- 必须**淡化** "PPI 预测" 和 "Gαq partner 建模" 的表述，改为 "GPCR sequence signature prediction"
- 必须加入更多生物学解读（如将 top ESM-2 维度映射到氨基酸类型偏好、长度分布等）
- 必须承认模型不依赖 Gαq 特征，并解释这是当前数据限制下的简化假设

**标题选项**:
1. *ESM-2 embeddings capture Gαq-coupling sequence signatures in GPCRs: a leakage-corrected small-sample study*
2. *Decoding GPCR-Gαq coupling specificity from pre-trained protein language model representations*
3. *Sequence-level signatures of Gαq coupling in GPCRs revealed by ESM-2 and homology-aware validation*

---

### 策略 C：暂缓投稿，加速补强（高风险高回报）
**定位**: 如果用户仍有冲击 2区/1区 的野心，必须在投稿前完成硬性补强  
**适合期刊**: *Bioinformatics*, *Briefings in Bioinformatics* (补强后), *Nature Communications* (需重大创新和实验)  
**建议时间线**: 3–6 个月

**必须完成的补强项**:
1. **引入真正的配对变化**:
   - 使用多个 G 蛋白作为负对照（Gi/o, Gs, G12/13），让 Gαq 与 GPCR 的配对特征产生真实变化。这样模型必须同时利用 GPCR 和 G 蛋白信息才能区分 "Gαq-coupling vs Gi-coupling"。
   - 或者：构建 GPCR-Gαq 复合物的 AlphaFold-Multimer 结构，提取界面接触残基作为特征。
2. **样本量扩充至 150–200+**:
   - 目前的 86 独立样本已接近 SVM 表现上限。要展示深度学习的价值，必须有足够独立的簇来支撑神经网络训练。
3. **结构特征升级**:
   - 真正整合 AlphaFold PDB 特征（pLDDT, DSSP, SASA, 接触图）。
   - 使用 TMHMM2/Phobius 代替 21-residue hydrophobicity 窗口。
4. **实验验证（哪怕少量）**:
   - 3–5 个 top 预测的湿实验验证（BRET/Co-IP）将大幅提升可信度。

如果完成以上 2–3 项，AUC 有望回到 0.88–0.90+（且是真实的），届时可以重建可信的 "小样本多模态 PPI 预测" 叙事。

---

## 三、重构 Abstract（策略 B 版本，可供直接使用）

> **Background**: G protein-coupled receptors (GPCRs) signal through specific G protein subtypes, yet the sequence determinants of GPCR-Gαq coupling remain incompletely understood. Pre-trained protein language models such as ESM-2 offer powerful sequence representations, but their utility in small-sample, homology-biased datasets is poorly characterized.
>
> **Methods**: We curated a dataset of 86 independent GPCR sequences with experimentally determined Gαq coupling labels. After removing 14 duplicate entries and enforcing homology-aware cross-validation (cluster-aware CV and leave-one-cluster-out), we evaluated support vector machines and cross-attention networks using ESM-2 embeddings. We extracted the genuine human GNAQ (P50148) ESM-2 reference and performed controlled comparisons of CLS token versus mean pooling. SHAP attribution was used to map predictive signals onto residue-level regions.
>
> **Results**: The leakage-corrected conservative baseline achieved LOCO AUC = 0.831 and cluster-aware AUC = 0.844. A controlled comparison found no meaningful difference between CLS token (AUC = 0.856) and mean pooling (AUC = 0.842) when evaluated on the same classifier. SHAP analysis revealed that the fixed GNAQ partner contributed zero predictive importance, indicating that the model relied entirely on intrinsic GPCR sequence signatures. Post-hoc projection of top ESM-2 dimensions localized signal enrichment to ICL1, TM6, and ICL2—regions consistent with known G protein binding interfaces.
>
> **Conclusions**: While deep-learning cross-attention did not outperform SVM in this small-sample regime, ESM-2 embeddings successfully encoded Gαq-coupling-related sequence signatures within GPCRs alone. Our work highlights the necessity of rigorous data-leakage control and homology-aware validation in small-sample protein interaction benchmarks.

---

## 四、Figure 故事线重构

**Figure 1**: 数据清洗与同源控制流程图  
- (a) 原始 100 样本中的 14 个重复对
- (b) 基于 3-mer Jaccard 同源聚类 (threshold 0.30)
- (c) Cluster-aware CV 与 LOCO CV 的 fold 分配

**Figure 2**: 性能修正对比  
- 原始随机 CV (0.915) → 去重后随机 CV (0.842) → Cluster-aware CV (0.844) → LOCO CV (0.831)
- 用柱状图或箭头图展示 AUC 下跌

**Figure 3**: CLS vs Mean 控制变量实验  
- 同一 SVM 上两种聚合方式的 5-fold AUC 箱线图
- 结论：差异不显著，原结论为混淆变量假象

**Figure 4**: SHAP 归因与生物学投影  
- (a) 全局 Top 20 GPCR ESM-2 维度的 SHAP 条形图
- (b) GNAQ 维度 SHAP = 0 的示意图（强调模型未利用 partner 信息）
- (c) 2–3 个代表性 GPCR 的残基级伪热图，ICL2 / TM6 区域高亮
- (d) 区域平均重要性柱状图

**Figure 5**: 方法学缺陷总结与未来工作路线图（如果需要）

---

## 五、期刊对齐策略

| 期刊 | IF | 适合度（修正后） | 投稿建议 |
|------|-----|----------------|---------|
| **BMC Bioinformatics** | 2.9 | ★★★★☆ | 最稳妥，接受方法学校正和负结果文章 |
| **PeerJ** | 2.3 | ★★★★☆ | 开放获取，审稿较快，适合警示性研究 |
| **PLOS Comp Biol** | 4.3 | ★★★☆☆ | 如果能把 ESM-2 生物学签名故事讲好，可尝试 |
| **Bioinformatics** | 4.4 | ★★☆☆☆ | 除非完成策略 C 的补强（真实配对变化 + 结构特征），否则基线太弱 |
| **Briefings in Bioinformatics** | 9.5 | ★☆☆☆☆ | 必须有实验验证或重大方法学突破，当前结果无法支撑 |
| **bioRxiv 预印本** | — | ★★★★★ | 强烈建议在投稿前发布预印本，建立修正结果优先权，获取同行反馈 |

---

## 六、最终建议

1. **不要**用原叙事（SVM 暴打深度学习、AUC 0.91）投稿任何期刊。这已构成不可辩护的方法学缺陷集合。
2. **优先推荐策略 A+B 的混合版本**：诚实地报告数据泄露和 AUC 修正，同时将 ESM-2 签名发现作为建设性的科学贡献包装进去。
3. **投稿前务必发布预印本**（bioRxiv），标题突出 "leakage correction" 和 "conservative baseline"。
4. 如果目标是高质量期刊（IF > 5），必须执行策略 C 的至少两项补强（特别是**引入真实 G 蛋白配对变化**和**样本量扩充**）。

---

*报告生成完毕。下一步建议：选定策略 (A/B/C)，据此修改 PAPER_DRAFT.md 全文。*
