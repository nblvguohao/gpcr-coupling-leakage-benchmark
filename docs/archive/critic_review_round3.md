# 批判者第三轮优化要求

**日期**: 2026年4月8日  
**目标**: 提高测试集准确率至0.80+，达到1区期刊标准  
**时间**: 1-2周  
**优先级**: 最高

---

## 一、问题诊断

### 1.1 当前问题

| 指标 | 当前值 | 目标值 | 差距 |
|------|--------|--------|------|
| 测试集准确率 | 0.647 | 0.80+ | -0.153 (-19.1%) |
| 测试集AUC | 0.586 | 0.75+ | -0.164 (-28.0%) |

### 1.2 可能原因分析

1. **过拟合**: 训练集表现好，测试集表现差
2. **数据不平衡**: 正负样本比例不均衡
3. **特征不够区分性**: 需要更强的特征工程
4. **模型复杂度不够**: 需要更深的网络或集成方法
5. **超参数未优化**: 使用默认参数，未调优

### 1.3 优化方向

```
优化策略优先级:
1. 数据增强 (最高优先级) - 解决样本不足和过拟合
2. 超参数调优 (高优先级) - 找到最优参数组合
3. 集成学习 (高优先级) - 提高泛化能力
4. 特征工程 (中优先级) - 增强特征区分性
```

---

## 二、P0级优化任务（必须完成）

### 任务1: 数据增强

**目标**: 通过数据增强增加训练样本多样性，减少过拟合

#### 1.1 SMOTE过采样

**方法**: 使用SMOTE生成合成样本

```python
from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42, k_neighbors=5)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
```

**预期效果**:
- 正负样本比例平衡
- 训练样本量增加
- 减少类别不平衡影响

**验收标准**:
- [ ] 正负样本比例达到1:1
- [ ] 训练样本量 ≥ 100

---

#### 1.2 高斯噪声增强

**方法**: 在特征上添加高斯噪声

```python
def add_gaussian_noise(X, noise_factor=0.05):
    noise = np.random.normal(0, noise_factor, X.shape)
    return X + noise
```

**参数**:
- noise_factor: 0.01, 0.05, 0.1

**预期效果**:
- 增加数据多样性
- 提高模型鲁棒性
- 减少过拟合

**验收标准**:
- [ ] 每个原始样本生成3-5个增强样本
- [ ] 噪声参数经过验证

---

#### 1.3 Mixup数据增强

**方法**: 混合两个样本的特征和标签

```python
def mixup_data(X, y, alpha=0.2):
    lam = np.random.beta(alpha, alpha)
    batch_size = X.shape[0]
    index = np.random.permutation(batch_size)
    
    mixed_X = lam * X + (1 - lam) * X[index]
    mixed_y = lam * y + (1 - lam) * y[index]
    return mixed_X, mixed_y
```

**预期效果**:
- 生成更平滑的决策边界
- 提高泛化能力

**验收标准**:
- [ ] 实现Mixup数据增强
- [ ] alpha参数调优

---

### 任务2: 超参数调优

**目标**: 找到最优的超参数组合

#### 2.1 Grid Search

**搜索空间**:

| 参数 | 搜索范围 |
|------|---------|
| learning_rate | [0.0001, 0.0005, 0.001, 0.005] |
| hidden_dim | [256, 512, 1024] |
| dropout | [0.2, 0.3, 0.4, 0.5] |
| batch_size | [8, 16, 32] |
| weight_decay | [1e-6, 1e-5, 1e-4] |

**方法**:
```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'learning_rate': [0.0001, 0.001, 0.01],
    'hidden_dim': [256, 512, 1024],
    'dropout': [0.2, 0.3, 0.5],
}

grid_search = GridSearchCV(model, param_grid, cv=5, scoring='roc_auc')
```

**验收标准**:
- [ ] 完成Grid Search
- [ ] 找到最优参数组合
- [ ] 验证集性能提升 > 5%

---

#### 2.2 贝叶斯优化

**方法**: 使用Optuna进行贝叶斯优化

```python
import optuna

def objective(trial):
    lr = trial.suggest_float('lr', 1e-5, 1e-2, log=True)
    hidden_dim = trial.suggest_categorical('hidden_dim', [256, 512, 1024])
    dropout = trial.suggest_float('dropout', 0.1, 0.5)
    
    model = create_model(hidden_dim, dropout)
    val_auc = train_and_evaluate(model, lr)
    return val_auc

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)
```

**验收标准**:
- [ ] 完成100次trial
- [ ] 找到最优参数
- [ ] 验证集AUC提升 > 10%

---

### 任务3: 集成学习

**目标**: 通过集成多个模型提高泛化能力

#### 3.1 5-Fold集成

**方法**:
1. 将训练集分成5份
2. 每份作为验证集，其余作为训练集
3. 训练5个独立的模型
4. 预测时取5个模型的平均

```python
from sklearn.model_selection import StratifiedKFold

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
models = []

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
    model = create_model()
    train_model(model, X_train[train_idx], y_train[train_idx])
    models.append(model)

# 预测时集成
predictions = [model.predict(X_test) for model in models]
final_prediction = np.mean(predictions, axis=0)
```

**验收标准**:
- [ ] 训练5个独立模型
- [ ] 集成预测性能提升 > 10%

---

#### 3.2 模型多样性集成

**方法**: 集成不同类型的模型

```python
models = {
    'DNN': ImprovedPPIPredictor(),
    'RandomForest': RandomForestClassifier(n_estimators=100),
    'XGBoost': XGBClassifier(),
    'SVM': SVC(probability=True),
}

# 加权平均
weights = {'DNN': 0.4, 'RandomForest': 0.2, 'XGBoost': 0.3, 'SVM': 0.1}
final_prediction = sum(w * models[m].predict(X_test) for m, w in weights.items())
```

**验收标准**:
- [ ] 至少3种不同类型的模型
- [ ] 集成性能优于单个模型

---

### 任务4: 5-Fold交叉验证

**目标**: 获得更可靠的性能评估

**方法**:
```python
from sklearn.model_selection import cross_val_score

scores = cross_val_score(model, X_all, y_all, cv=5, scoring='roc_auc')
print(f"5-Fold AUC: {scores.mean():.4f} ± {scores.std():.4f}")
```

**验收标准**:
- [ ] 完成5-fold交叉验证
- [ ] 报告mean ± std
- [ ] 与测试集性能对比

---

## 三、P1级优化任务（推荐完成）

### 任务5: 特征工程增强

#### 5.1 进化特征（PSSM）

**方法**: 使用PSI-BLAST生成PSSM矩阵

**预期效果**:
- 捕获进化保守性信息
- 提高特征区分性

---

#### 5.2 物理化学特征扩展

**新增特征**:
- 氨基酸理化性质（疏水性、电荷、大小）
- 序列复杂度
- 重复序列比例

---

### 任务6: 正则化增强

#### 6.1 更强的Dropout

**调整**: dropout从0.3提高到0.4-0.5

#### 6.2 早停策略

**方法**:
```python
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=20,
    restore_best_weights=True
)
```

#### 6.3 标签平滑

**方法**:
```python
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
```

---

## 四、时间安排（1-2周）

### 第1周

| 天数 | 任务 | 目标 |
|------|------|------|
| 1-2 | 数据增强（SMOTE + 噪声） | 训练样本翻倍 |
| 3-4 | 超参数调优（Grid Search） | 找到最优参数 |
| 5-7 | 集成学习（5-fold） | 训练5个模型 |

### 第2周

| 天数 | 任务 | 目标 |
|------|------|------|
| 8-10 | 5-fold交叉验证 | 可靠性能评估 |
| 11-12 | 最终测试集评估 | 准确率 ≥ 0.80 |
| 13-14 | 论文修改完善 | 准备投稿 |

---

## 五、验收标准

### 最低标准（达到1区要求）
- [ ] 测试集准确率 ≥ 0.80
- [ ] 测试集AUC ≥ 0.75
- [ ] 5-fold交叉验证完成
- [ ] 集成学习实现

### 理想标准（高质量1区论文）
- [ ] 测试集准确率 ≥ 0.85
- [ ] 测试集AUC ≥ 0.80
- [ ] 统计显著性 p < 0.01
- [ ] 消融实验证明各模块贡献

---

## 六、风险与应对

| 风险 | 可能性 | 影响 | 应对策略 |
|------|--------|------|---------|
| 数据增强无效 | 低 | 高 | 尝试多种增强方法组合 |
| 超参数调优耗时 | 中 | 中 | 使用贝叶斯优化加速 |
| 集成学习复杂 | 低 | 中 | 简化集成策略 |
| 性能提升有限 | 中 | 高 | 考虑特征工程或模型架构调整 |

---

## 七、成功标准

**必须达成**:
1. 测试集准确率 ≥ 0.80
2. 测试集AUC ≥ 0.75

**建议达成**:
1. 测试集准确率 ≥ 0.85
2. 测试集AUC ≥ 0.80
3. 5-fold交叉验证 AUC ≥ 0.78

---

**批判者**: 制定优化要求  
**执行者**: 执行优化任务  
**统筹者**: 监督进度和质量  

**制定日期**: 2026年4月8日  
**预计完成**: 2026年4月22日
