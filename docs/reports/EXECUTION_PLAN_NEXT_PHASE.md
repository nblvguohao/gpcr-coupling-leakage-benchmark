# GPCR项目后续执行计划方案

**制定日期**: 2026年4月8日
**项目状态**: 2区标准达成，冲刺1区标准
**当前样本**: 100个（47正/53负）✅

---

## 一、执行概览

```
Phase 1 (1-2周): 论文完善与图表优化
Phase 2 (2-4周): 服务器优化与模型提升
Phase 3 (1-2月): 实验验证与结构调整
Phase 4 (投稿前): 最终整合与投稿准备
```

---

## 二、Phase 1: 论文完善与图表优化（1-2周）

### 2.1 论文图表制作（优先级：高）

| 图表类型 | 工具 | 预计时间 | 状态 | 负责人 |
|---------|------|---------|------|--------|
| 方法流程图 | Gamma | 已完成 | ✅ | Claude |
| 性能对比图 | Gamma | 已完成 | ✅ | Claude |
| ROC曲线对比 | Python/matplotlib | 2小时 | ⏳ 待做 | Claude |
| 消融实验图 | Python/matplotlib | 2小时 | ⏳ 待做 | Claude |
| 特征重要性热图 | Python/seaborn | 3小时 | ⏳ 待做 | Claude |
| 数据集分布图 | Python | 1小时 | ⏳ 待做 | Claude |

**ROC曲线代码框架**:
```python
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

def plot_roc_curves(results_dict):
    plt.figure(figsize=(8, 6))
    for method, scores in results_dict.items():
        fpr, tpr, _ = roc_curve(y_true, scores)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'{method} (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves Comparison')
    plt.legend()
    plt.savefig('roc_curves.png', dpi=300)
```

### 2.2 论文内容完善（优先级：高）

| 章节 | 当前状态 | 需要完善 | 预计时间 |
|------|---------|---------|---------|
| Abstract | 已撰写 | 更新100样本数据 | 30分钟 |
| Introduction | 框架完整 | 补充最新文献 | 2小时 |
| Methods | 已完成 | 增加扩展数据集说明 | 1小时 |
| Results | 已完成 | 更新100样本结果 | 2小时 |
| Discussion | 框架完整 | 深入讨论局限性和展望 | 3小时 |
| Conclusion | 已撰写 | 更新成果总结 | 30分钟 |

**需要更新的关键数据**:
- 数据集: 53 → 100个样本
- 正样本: 29 → 47个
- 负样本: 24 → 53个
- 补充: 超参数优化正在进行

### 2.3 参考文献更新（优先级：中）

```
新增文献建议:
1. AlphaFold3相关文献（2024）
2. ESM-3蛋白质语言模型（2024）
3. 最新GPCR-G蛋白耦合预测研究
4. 交叉注意力在蛋白质相互作用中的应用

目标: 从15篇增加到20-25篇
```

---

## 三、Phase 2: 服务器优化与模型提升（2-4周）

### 3.1 超参数优化执行（优先级：高）

**配置确认**:
```json
{
  "param_grid": {
    "learning_rate": [1e-4, 5e-5, 1e-5],
    "dropout": [0.1, 0.2, 0.3],
    "hidden_dim": [256, 320, 512],
    "num_heads": [4, 8],
    "num_layers": [2, 3]
  },
  "total_combinations": 162,
  "target_auc": 0.88,
  "target_accuracy": 0.80
}
```

**执行步骤**:
1. 等待服务器GPU空闲
2. 运行超参数搜索脚本
3. 监控训练进度
4. 分析最优参数
5. 使用最优参数重新训练最终模型

**预计时间**: 12-24小时（服务器连续运行）

### 3.2 扩展数据集特征提取（优先级：高）

**任务分解**:
```
□ 上传100个样本到服务器
□ 运行ESM-2特征提取（100样本 × 2蛋白 = 200个序列）
□ 生成物理化学特征
□ 生成结构特征（基于序列预测）
□ 合并特征矩阵
□ 保存到output/real_data/features/
```

**预计时间**: 3-4小时

### 3.3 完整版交叉注意力训练（优先级：中）

**使用100样本重新训练**:
- 训练集: 68样本
- 验证集: 15样本
- 测试集: 17样本

**目标性能**:
- AUC > 0.88
- Accuracy > 80%
- F1-score > 0.82

---

## 四、Phase 3: 实验验证与结构调整（1-2个月）

### 4.1 实验验证设计（优先级：高）

**BRET实验方案**:
```
实验设计:
1. 选择3-5个高置信度预测:
   - 2个高置信度阳性预测
   - 2个高置信度阴性预测
   - 1个边界案例

2. 构建BRET质粒:
   - GPCR-Rluc8 (供体)
   - Gαq-YFP (受体)

3. 细胞系选择:
   - HEK293T细胞

4. 检测指标:
   - BRET比值
   - 剂量-反应曲线
   - 阳性对照: 已知Gq偶联GPCR
   - 阴性对照: 已知Gi偶联GPCR
```

**Co-IP实验方案**:
```
实验设计:
1. 构建表达质粒:
   - Flag-tagged GPCR
   - HA-tagged Gαq

2. 免疫共沉淀:
   - 抗Flag抗体沉淀
   - Western blot检测HA信号

3. 定量分析:
   - 相对结合强度
```

**时间安排**:
- 方案设计: 1周
- 质粒构建: 2-3周
- 细胞实验: 2-3周
- 数据分析: 1周
- **总计**: 6-8周

### 4.2 AlphaFold结构整合（优先级：中）

**执行步骤**:
```
□ 解决服务器网络连接问题
□ 批量下载100个GPCR的AlphaFold结构
□ 解析PDB文件提取特征:
   - 二级结构（螺旋/折叠/无规卷曲）
   - 接触图（残基-残基接触）
   - 溶剂可及性
   - 结构柔性
□ 整合结构特征到现有特征集
□ 重新训练模型评估性能提升
```

**备选方案**（如果下载失败）:
- 使用本地AlphaFold2预测
- 使用基于序列的结构特征（当前已实现）

---

## 五、Phase 4: 最终整合与投稿准备（2-4周）

### 5.1 论文最终润色（优先级：高）

**检查清单**:
- [ ] 数据一致性检查
- [ ] 图表质量检查（300 dpi）
- [ ] 参考文献格式统一
- [ ] 补充材料整理
- [ ] 作者信息和贡献声明
- [ ] 利益冲突声明
- [ ] 数据可用性声明

### 5.2 目标期刊选择（优先级：高）

**候选期刊（1区）**:

| 期刊 | IF | 命中率 | 审稿周期 | 要求 |
|------|-----|--------|---------|------|
| Bioinformatics | 4.4 | 中等 | 2-3月 | 方法创新 |
| Briefings in Bioinformatics | 9.5 | 较低 | 2-3月 | 综述+方法 |
| PLOS Computational Biology | 4.3 | 中等 | 2-3月 | 生物学意义 |
| Journal of Chemical Information and Modeling | 5.6 | 中等 | 2-3月 | 化学应用 |
| BMC Bioinformatics | 3.0 | 较高 | 1-2月 | 方法学 |

**推荐策略**:
1. 首选: Briefings in Bioinformatics（IF高，适合方法学）
2. 备选: Journal of Chemical Information and Modeling
3. 保底: BMC Bioinformatics

### 5.3 投稿材料准备（优先级：高）

```
投稿包内容:
├── Manuscript.docx (主文档)
├── Figures/
│   ├── Figure1_method.pdf
│   ├── Figure2_performance.pdf
│   ├── Figure3_ablation.pdf
│   └── Figure4_roc_curves.pdf
├── Supplementary/
│   ├── Table_S1_dataset.xlsx
│   ├── Figure_S1_attention_maps.pdf
│   └── Methods_S1_details.pdf
├── Cover_Letter.docx (投稿信)
├── Highlights.txt (3-5条亮点)
└── Author_Statement.pdf
```

---

## 六、时间节点与里程碑

### 甘特图

```
周次    1    2    3    4    5    6    7    8    9    10   11   12
       |----|----|----|----|----|----|----|----|----|----|----|----|
Phase 1: 论文完善
  图表  [====]
  内容  [====]
  文献  [==  ]

Phase 2: 服务器优化
  超参      [========]
  特征      [====    ]
  训练          [========]

Phase 3: 实验验证
  方案          [==]
  质粒              [========]
  实验                  [========]
  分析                      [==  ]

Phase 4: 投稿准备
  润色                              [====]
  投稿                                  [====]
```

### 关键里程碑

| 日期 | 里程碑 | 交付物 |
|------|--------|--------|
| 4月15日 | Phase 1完成 | 完整论文初稿+图表 |
| 4月22日 | 超参数优化完成 | 最优参数+模型 |
| 4月29日 | 100样本训练完成 | 最终性能报告 |
| 5月13日 | 实验方案确定 | 实验方案文档 |
| 6月10日 | 实验完成 | 实验数据 |
| 6月24日 | 论文终稿 | 投稿包 |
| 7月1日 | 投稿 | 提交确认 |

---

## 七、风险评估与应对

| 风险 | 概率 | 影响 | 应对措施 |
|------|------|------|----------|
| 超参数优化效果不佳 | 中 | 高 | 扩大搜索空间，尝试Transformer变体 |
| 实验验证周期长 | 高 | 中 | 提前联系合作实验室，准备备选方案 |
| AlphaFold下载失败 | 中 | 低 | 使用序列预测结构作为替代 |
| 期刊拒稿 | 中 | 高 | 准备3个梯队期刊，快速转投 |
| 服务器故障 | 低 | 高 | 定期备份，本地保留副本 |

---

## 八、资源需求

### 计算资源
- GPU服务器: 100小时（V100/A100）
- 存储: 50GB

### 实验资源
- BRET实验: 约5万元
- Co-IP实验: 约3万元
- 总计: 8万元（预算申请中）

### 人力资源
- 数据分析: Claude（AI助手）
- 实验执行: 合作实验室
- 论文撰写: 主要研究者
- 论文审校: 合作导师

---

## 九、今日立即执行任务

### 高优先级（接下来2小时）

1. **下载Gamma图表**
   - 链接1: https://gamma.app/docs/tw601h488b1jp65
   - 链接2: https://gamma.app/docs/glu9ib8ebucxu9b
   - 操作: Export → PNG/PDF

2. **创建ROC曲线图**
   - 使用本地Python运行
   - 保存到figures/目录

3. **更新论文数据**
   - 将100样本数据写入Methods
   - 更新Abstract中的数字

### 中优先级（本周内）

4. 服务器超参数优化启动
5. 100样本特征提取
6. 补充材料表格整理

---

## 十、联系方式与协作

### 关键文件位置
```
本地:
E:/kimi/Kimi_Agent_批判者监督执行/
├── GPCR_PAPER_FINAL.md (论文主文档)
├── merged_dataset/ (100样本数据集)
├── server_results/ (服务器结果)
└── figures/ (图表目录)

服务器:
/data/lgh/GPCR/
├── output/real_data/ (原始数据)
├── extended_data/ (扩展数据)
└── hyperparam_results/ (优化结果)
```

---

**计划制定**: 2026年4月8日
**下次评审**: 2026年4月15日
**目标投稿日期**: 2026年6月底

**备注**: 本计划将根据实际进展动态调整，建议每周评审一次进度。
