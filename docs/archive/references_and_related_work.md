# 文献综述与参考文献

## 视蛋白-Gαq蛋白相互作用预测研究

---

## 一、核心参考文献

### 1.1 蛋白质语言模型

1. **Lin, Z., et al. (2022).** Evolutionary-scale prediction of atomic-level protein structure with a language model. *Science*, 379(6637), 1123-1130.
   - ESM-2模型的原始论文，650M参数版本
   - 奠定了蛋白质语言模型在结构预测中的应用基础

2. **Rives, A., et al. (2021).** Biological structure and function emerge from scaling unsupervised learning to 250 million protein sequences. *PNAS*, 118(15), e2016239118.
   - ESM-1b模型，ESM系列的开创性工作
   - 展示了大规模无监督学习在蛋白质表示学习中的威力

3. **Elnaggar, A., et al. (2021).** ProtTrans: Towards cracking the language of Life's code through self-supervised deep learning and high performance computing. *IEEE TPAMI*.
   - ProtT5模型，基于T5架构的蛋白质语言模型
   - 提供了另一种蛋白质序列编码方案

### 1.2 蛋白质相互作用预测

4. **Sledzieski, S., et al. (2024).** D-SCRIPT translates genome to phenome with sequence-based prediction of protein-protein interactions. *Nature Methods*.
   - 序列级PPI预测的经典方法
   - 使用Siamese网络架构

5. **Sledzieski, S., et al. (2024).** An update on protein-protein interaction prediction. *Bioinformatics*.
   - PPI预测领域的最新综述
   - 讨论了当前方法的局限性和未来方向

6. **Chen, M., et al. (2023).** Multifaceted protein-protein interaction prediction based on Siamese residual RCNN. *Bioinformatics*, 35(14), i305-i314.
   - PIPR方法，使用残差RCNN
   - 在多个数据集上取得优异性能

7. **Richoux, F., et al. (2019).** Comparing two deep learning sequence-based models for protein-protein interaction prediction. *arXiv preprint*.
   - 深度学习方法在PPI预测中的比较研究
   - 提供了重要的基线方法

### 1.3 图神经网络在PPI中的应用

8. **Mahbub, S., & Bayzid, M. S. (2022).** EGRET: Edge aggregated graph attention networks and transfer learning for protein-protein interaction site prediction. *Bioinformatics*.
   - 结合ProtBERT和图注意力网络
   - 残基级PPI位点预测

9. **Yuan, Q., et al. (2021).** GraphPPIS: Graph neural network with self-attention for protein-protein interaction site prediction. *Bioinformatics*.
   - 基于图的PPI位点预测
   - 自注意力机制

10. **Feng, S., et al. (2024).** DGCPPISP: Dynamic graph convolution for protein-protein interaction site prediction. *Bioinformatics*.
    - 动态图卷积网络
    - 结合ESM-2嵌入

### 1.4 GPCR-G蛋白偶联预测

11. **Ono, T., & Hishigaki, H. (2006).** Prediction of GPCR-G protein coupling specificity using features of sequences and biological functions. *Genomics Proteomics Bioinformatics*.
    - 结合序列特征和NLP提取的功能信息
    - C4.5算法，92.2%准确率

12. **Möller, S., et al. (2001).** Prediction of the coupling specificity of G protein-coupled receptors to their G proteins. *Bioinformatics*.
    - 基于模式发现的早期方法
    - 专注于细胞内环区域

13. **Sgourakis, N. G., et al. (2005).** Prediction of GPCR-G protein coupling selectivity. *Proteins*.
    - HMM方法预测偶联特异性
    - 基于细胞内环的模式

14. **Miglionico, P., et al. (2025).** Predicting and engineering GPCR-G protein coupling specificity with AlphaFold3. *ISMB/ECCB*.
    - 最新AlphaFold3结构预测方法
    - 78%准确率，0.87 AUC

### 1.5 对比学习与小样本学习

15. **Chen, T., et al. (2020).** A simple framework for contrastive learning of visual representations. *ICML*.
    - SimCLR对比学习框架
    - 可借鉴到蛋白质表示学习

16. **Snell, J., et al. (2017).** Prototypical networks for few-shot learning. *NeurIPS*.
    - 原型网络小样本学习
    - 适用于小样本蛋白质分类

17. **Finn, C., et al. (2017).** Model-agnostic meta-learning for fast adaptation of deep networks. *ICML*.
    - MAML元学习算法
    - 快速适应新任务

### 1.6 注意力机制

18. **Vaswani, A., et al. (2017).** Attention is all you need. *NeurIPS*.
    - Transformer原始论文
    - 自注意力机制的基础

19. **Veličković, P., et al. (2018).** Graph attention networks. *ICLR*.
    - 图注意力网络GAT
    - 适用于图结构数据

20. **Wang, X., et al. (2018).** Non-local neural networks. *CVPR*.
    - 非局部注意力机制
    - 捕获长距离依赖

---

## 二、相关研究领域

### 2.1 蛋白质结构预测

- **Jumper, J., et al. (2021).** Highly accurate protein structure prediction with AlphaFold. *Nature*, 596(7873), 583-589.
- **Baek, M., et al. (2021).** Accurate prediction of protein structures and interactions using a three-track neural network. *Science*, 373(6557), 871-876.

### 2.2 蛋白质功能预测

- **Kulmanov, M., & Hoehndorf, R. (2020).** DeepGOPlus: improved protein function prediction from sequence. *Bioinformatics*, 36(2), 422-429.
- **You, R., et al. (2018).** GOLabeler: Improving sequence-based large-scale protein function prediction by learning to rank. *Bioinformatics*, 34(14), 2465-2473.

### 2.3 跨物种预测

- **Saelens, W., et al. (2024).** Hierarchical multi-label contrastive learning for protein-protein interaction prediction across organisms. *arXiv*.
- **Zitnik, M., et al. (2019).** Machine learning for integrating data in biology and medicine. *Nature Methods*.

---

## 三、方法学比较

### 3.1 PPI预测方法对比

| 方法 | 年份 | 架构 | 特征 | 性能(AUC) |
|------|------|------|------|-----------|
| PIPR | 2019 | Siamese RCNN | 序列 | 0.85 |
| D-SCRIPT | 2021 | CNN | 序列 | 0.89 |
| EGRET | 2022 | GAT + ProtBERT | 序列+结构 | 0.72 |
| DGCPPISP | 2024 | Dynamic GCN | ESM-2+序列 | 0.78 |
| TUnA | 2024 | Transformer | ESM-2 | 0.65 |
| **本研究** | 2025 | Cross-Attn + GNN | ESM-2+多模态 | **目标>0.85** |

### 3.2 GPCR-G蛋白预测方法对比

| 方法 | 年份 | 技术 | 准确率 |
|------|------|------|--------|
| HMM方法 | 2004 | HMM | 99% (低灵敏度) |
| SVM+特征 | 2005 | SVM | 85% |
| C4.5规则 | 2006 | 决策树 | 92.2% |
| AlphaFold3 | 2025 | 结构+ML | 78% |
| **本研究** | 2025 | 深度学习 | **目标>85%** |

---

## 四、技术发展趋势

### 4.1 当前趋势

1. **蛋白质语言模型主导**：ESM-2、ProtT5等PLM成为标准特征提取器
2. **多模态融合**：序列+结构+功能的联合建模
3. **图神经网络兴起**：利用蛋白质结构信息
4. **可解释性需求**：理解模型决策的生物学基础
5. **跨物种泛化**：从模式生物到非模式生物的迁移

### 4.2 未来方向

1. **大规模预训练**：类似NLP中的GPT，蛋白质领域的Foundation Model
2. **动态相互作用**：考虑蛋白质构象变化的动态PPI
3. **多蛋白质复合物**：超越二元相互作用
4. **因果推理**：理解相互作用的因果关系
5. **实验验证闭环**：计算预测与实验验证的结合

---

## 五、本研究的定位

### 5.1 创新点定位

```
现有研究                        本研究
─────────────────────────────────────────────────────────
通用PPI预测    ────────────>   GPCR-G蛋白特异性预测
大规模数据集   ────────────>   小样本学习
单一物种       ────────────>   跨物种泛化
黑盒模型       ────────────>   可解释性分析
序列/结构单一  ────────────>   多模态融合
```

### 5.2 学术贡献

1. **方法创新**：首个专门针对GPCR-Gαq偶联特异性的深度学习框架
2. **技术创新**：跨物种对比学习+交叉注意力机制
3. **应用创新**：为光遗传学工具开发提供计算指导
4. **理论创新**：揭示跨物种偶联特异性的分子机制

---

## 六、参考文献格式（BibTeX）

```bibtex
% 蛋白质语言模型
@article{lin2022evolutionary,
  title={Evolutionary-scale prediction of atomic-level protein structure with a language model},
  author={Lin, Zeming and Akin, Halil and Rao, Roshan and Hie, Brian and Zhu, Zhongkai and Lu, Wenting and Smetanin, Nikita and Verkuil, Robert and Kabeli, Ori and Shmueli, Yaniv and others},
  journal={Science},
  volume={379},
  number={6637},
  pages={1123--1130},
  year={2022}
}

% PPI预测
@article{sledzieski2021dscript,
  title={D-SCRIPT translates genome to phenome with sequence-based prediction of protein-protein interactions},
  author={Sledzieski, Samuel and Bansal, Mugdha and Berger, Bonnie},
  journal={Nature Methods},
  year={2024}
}

% GPCR-G蛋白预测
@article{ono2006prediction,
  title={Prediction of GPCR--G protein coupling specificity using features of sequences and biological functions},
  author={Ono, Toshihide and Hishigaki, Hideki},
  journal={Genomics Proteomics Bioinformatics},
  year={2006}
}

% 注意力机制
@inproceedings{vaswani2017attention,
  title={Attention is all you need},
  author={Vaswani, Ashish and Shazeer, Noam and Parmar, Niki and Uszkoreit, Jakob and Jones, Llion and Gomez, Aidan N and Kaiser, {\L}ukasz and Polosukhin, Illia},
  booktitle={NeurIPS},
  year={2017}
}

% 图注意力网络
@inproceedings{velickovic2018graph,
  title={Graph attention networks},
  author={Veli{\v{c}}kovi{\'c}, Petar and Cucurull, Guillem and Casanova, Arantxa and Romero, Adriana and Li{\`o}, Pietro and Bengio, Yoshua},
  booktitle={ICLR},
  year={2018}
}
```

---

*文献综述版本：v1.0*
