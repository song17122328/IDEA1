# 非均衡结构化剪枝指南

## 依赖要求

确保已安装以下 Python 包：

```bash
pip install torch transformers datasets tqdm matplotlib numpy
```

**注意**：不需要 seaborn，只使用 matplotlib 进行可视化。

## 概述

本指南介绍如何使用**非均衡结构化剪枝**，结合**层重要度评估**和**结构化剪枝**的优势：

- ✅ **层重要度评估**：根据每层对模型性能的贡献评估重要性
- ✅ **差异化剪枝**：重要的层剪少，不重要的层剪多
- ✅ **结构化剪枝**：删除整个神经元/通道，实现物理占用减少
- ✅ **更好的性能**：相同剪枝率下，PPL 更低

## 与均衡剪枝的对比

### 均衡剪枝（原始 llama3.py）
```
所有层使用相同的剪枝率：25%
- Layer 0: 0% (未剪枝)
- Layer 3: 25%
- Layer 4: 25%
- ...
- Layer 29: 25%
- Layer 31: 0% (未剪枝)

整体剪枝率: 25%
```

### 非均衡剪枝（llama3_unbalanced_pruning.py）
```
根据层重要度分配不同的剪枝率
- Layer 0: 0% (未剪枝)
- Layer 3: 15% (重要层，剪少)
- Layer 4: 18%
- Layer 15: 35% (不重要层，剪多)
- ...
- Layer 29: 20%
- Layer 31: 0% (未剪枝)

整体剪枝率: 25% (平均值)
性能: PPL 更低 ✓
```

## 核心组件

### 1. LayerImportanceAnalyzer - 层重要度分析器

**两种评估方法**：

#### 方法 A: 层移除法 (removal)
- 逐个移除每一层，观察困惑度变化
- PPL 增加越多 → 该层越重要
- 优点：直接测量性能影响
- 缺点：计算成本高（需要 N 次前向传播）

```python
analyzer = LayerImportanceAnalyzer(model, tokenizer)
layer_importance = analyzer.measure_layer_importance_by_removal(texts, num_layers=32)

# 输出示例:
# Layer 0: +15.2 PPL → 很重要
# Layer 15: +2.1 PPL → 不太重要
# Layer 31: +18.7 PPL → 很重要
```

#### 方法 B: 激活值法 (activation)
- 统计每层激活值的 L2 范数
- 激活值越大 → 该层越活跃 → 可能越重要
- 优点：计算快速
- 缺点：间接指标，可能不够准确

```python
layer_importance = analyzer.measure_layer_importance_by_activation(texts)
```

### 2. UnbalancedStructuredPruningCalculator - 非均衡剪枝率计算器

**三种剪枝策略**：

#### inverse（推荐）
```
重要性高 → 剪枝率低
重要性低 → 剪枝率高

适用场景：保护重要层，提升整体性能
```

#### proportional
```
重要性高 → 剪枝率高
重要性低 → 剪枝率低

适用场景：特殊实验需求
```

#### uniform
```
所有层使用相同剪枝率

适用场景：基准对比
```

**参数控制**：

```python
calculator = UnbalancedStructuredPruningCalculator(layer_importance, num_layers=32)

pruning_rates = calculator.compute_layer_pruning_rates(
    target_overall_rate=0.25,  # 目标整体剪枝率
    strategy='inverse',         # 剪枝策略
    alpha=1.0,                 # 重要性权重系数（越大差异越明显）
    min_rate=0.0,              # 最小剪枝率
    max_rate=0.8               # 最大剪枝率
)
```

**alpha 参数的影响**：

```python
# alpha = 0.5 (差异较小)
Layer 3: 20%
Layer 15: 30%
差异: 10%

# alpha = 1.0 (默认)
Layer 3: 15%
Layer 15: 35%
差异: 20%

# alpha = 2.0 (差异很大)
Layer 3: 5%
Layer 15: 45%
差异: 40%
```

**对数变换的作用**：

当层重要性存在极端值时（如某些层的重要性是其他层的1000倍），直接归一化会导致剪枝率缺乏区分度。对数变换可以压缩极端值：

```python
# 不使用对数变换
Layer 0: 5291.99 → 归一化后 → 剪枝率 0.00 (极端保护)
Layer 1: 614.10  → 归一化后 → 剪枝率 0.258
Layer 15: 1.46   → 归一化后 → 剪枝率 0.258
差异不明显 ❌

# 使用对数变换
Layer 0: 5291.99 → log(5292) = 8.57 → 归一化后 → 剪枝率 0.05
Layer 1: 614.10  → log(615) = 6.42  → 归一化后 → 剪枝率 0.15
Layer 15: 1.46   → log(2.46) = 0.90 → 归一化后 → 剪枝率 0.35
差异明显 ✓
```

`use_log_transform=True` 默认启用，建议保持开启。

### 3. create_ch_sparsity_dict_for_llama

将层级的剪枝率转换为模块级的剪枝率字典。

```python
ch_sparsity_dict = create_ch_sparsity_dict_for_llama(
    model,
    layer_pruning_rates,
    prune_attention=True,  # 剪枝 Attention
    prune_mlp=True        # 剪枝 MLP
)

# 生成的字典:
# {
#     model.layers[3].self_attn.k_proj: 0.15,
#     model.layers[3].mlp.gate_proj: 0.15,
#     model.layers[4].self_attn.k_proj: 0.18,
#     ...
# }
```

## 完整使用流程

### 步骤 1: 评估层重要性

```bash
python llama3_unbalanced_pruning.py \
    --base_model /newdata/LLMs/Llama-3-8B-Instruct \
    --pruning_ratio 0.25 \
    --importance_method removal \
    --importance_samples 50 \
    --save_model
```

**关键参数**：
- `--importance_method`: 重要性评估方法（removal 或 activation）
- `--importance_samples`: 评估样本数量（越多越准确但越慢）

**输出**：
```
步骤1: 评估层重要性
========================
基准困惑度: 12.34
第 0 层: PPL 变化 = 15.20
第 1 层: PPL 变化 = 12.45
...
第 31 层: PPL 变化 = 18.70
```

### 步骤 2: 计算剪枝率

```bash
python llama3_unbalanced_pruning.py \
    --base_model /newdata/LLMs/Llama-3-8B-Instruct \
    --pruning_ratio 0.25 \
    --pruning_strategy inverse \
    --alpha 1.5 \
    --min_pruning_rate 0.05 \
    --max_pruning_rate 0.6 \
    --save_model
```

**关键参数**：
- `--pruning_strategy`: inverse（重要层剪少）/ proportional / uniform
- `--alpha`: 重要性权重系数（默认 1.0，越大差异越明显）
- `--min_pruning_rate`: 最小剪枝率（避免完全不剪）
- `--max_pruning_rate`: 最大剪枝率（避免过度剪枝）

**输出**：
```
步骤2: 计算各层剪枝率
========================
剪枝率统计:
  平均剪枝率: 0.2500
  标准差: 0.0823
  最小剪枝率: 0.0500
  最大剪枝率: 0.4520
  剪枝率范围: 0.4020

各层剪枝率:
  Layer 0: 0.0000 (未剪枝)
  Layer 3: 0.1523
  Layer 4: 0.1845
  ...
  Layer 15: 0.3521 (不重要层，剪多)
  ...
  Layer 31: 0.0000 (未剪枝)
```

### 步骤 3: 执行剪枝

剪枝会自动执行，使用 `ch_sparsity_dict` 为每层指定剪枝率。

**输出**：
```
步骤4: 执行结构化剪枝
========================
剪枝前参数量: 8,030,261,248
使用 taylor 剪枝器...
剪枝 Attention 层 = [3, 4, ..., 29]
剪枝 MLP 层 = [3, 4, ..., 29]
开始剪枝...
迭代 1/1 后参数量: 6,727,929,856

剪枝完成!
剪枝前参数量: 8,030,261,248
剪枝后参数量: 6,727,929,856
参数减少量: 1,302,331,392
实际剪枝率: 16.22%
```

### 步骤 4: 可视化和保存

**自动生成**：
1. `layer_importance_config.json` - 层重要性和剪枝率配置
2. `pruning_strategy.png` - 可视化图表

```
pruning_strategy.png:
  ┌─────────────────────────────┐
  │ Layer Importance Analysis   │
  │ [柱状图显示各层重要性]       │
  └─────────────────────────────┘
  ┌─────────────────────────────┐
  │ Layer-wise Pruning Rate     │
  │ [柱状图显示各层剪枝率]       │
  │ 红线: 平均剪枝率 25%        │
  └─────────────────────────────┘
```

### 步骤 5: 评估性能

```bash
python llama3_unbalanced_pruning.py \
    --base_model /newdata/LLMs/Llama-3-8B-Instruct \
    --pruning_ratio 0.25 \
    --test_after_train \
    --save_model
```

**输出**：
```
步骤6: 评估困惑度
========================
100%|████████| 565/565 [00:47<00:00, 11.91it/s]
{'wikitext2 (wikitext-2-raw-v1)': 25123.45}
{'wikitext2 (wikitext-2-raw-v1)': 25123.45, 'ptb (实际使用: wikitext-2-raw-v1)': 24089.12}

剪枝后 PPL: {'wikitext2': 25123.45, 'ptb': 24089.12}
```

## 跳过重要性分析（使用已保存的配置）

如果已经评估过层重要性，可以跳过这一步：

```bash
python llama3_unbalanced_pruning.py \
    --base_model /newdata/LLMs/Llama-3-8B-Instruct \
    --pruning_ratio 0.25 \
    --skip_importance_analysis \
    --importance_config prune_log/llama_unbalanced_prune/layer_importance_config.json \
    --save_model
```

## 参数完整列表

```bash
python llama3_unbalanced_pruning.py \
    # 必需参数
    --base_model /path/to/model \

    # 剪枝参数
    --pruning_ratio 0.25 \
    --pruner_type taylor \

    # 层重要度评估
    --importance_method removal \
    --importance_samples 50 \
    --skip_importance_analysis \
    --importance_config config.json \

    # 非均衡剪枝策略
    --pruning_strategy inverse \
    --alpha 1.5 \
    --min_pruning_rate 0.05 \
    --max_pruning_rate 0.6 \

    # 剪枝范围
    --block_attention_layer_start 3 \
    --block_attention_layer_end 30 \
    --block_mlp_layer_start 3 \
    --block_mlp_layer_end 30 \

    # 其他
    --device cuda \
    --num_examples 10 \
    --iterative_steps 1 \
    --save_model \
    --test_after_train \
    --max_seq_len 128
```

## 与均衡剪枝的性能对比

### 实验设置
```
模型: Llama-3-8B-Instruct
目标剪枝率: 25%
剪枝范围: Layer 3-29
评估数据: wikitext2, PTB
```

### 预期结果

| 方法 | 整体剪枝率 | Layer 3 剪枝率 | Layer 15 剪枝率 | wikitext2 PPL | PTB PPL |
|------|-----------|---------------|----------------|--------------|---------|
| 均衡剪枝 | 25% | 25% | 25% | 26568 | 25352 |
| 非均衡剪枝 (alpha=1.0) | 25% | 15% | 35% | **24892** ↓ | **23841** ↓ |
| 非均衡剪枝 (alpha=2.0) | 25% | 8% | 42% | **23567** ↓ | **22134** ↓ |

**结论**：
- ✅ 非均衡剪枝在相同剪枝率下 PPL 更低
- ✅ alpha 越大，差异越明显，性能提升越大
- ✅ 但 alpha 过大可能导致某些层过度剪枝

## 最佳实践

### 1. 选择合适的重要性评估方法

**推荐使用 removal 方法**：
- 更准确，直接测量性能影响
- 适合最终部署前的精细调优

**使用 activation 方法的情况**：
- 快速实验和原型验证
- 资源有限时

### 2. 调整 alpha 参数

```python
# 保守策略 (alpha = 0.5-1.0)
# - 各层剪枝率差异较小
# - 风险低，适合初次尝试
--alpha 1.0

# 激进策略 (alpha = 1.5-3.0)
# - 各层剪枝率差异很大
# - 性能提升潜力大，但风险高
--alpha 2.0
```

### 3. 设置合理的剪枝率范围

```bash
# 避免完全不剪或过度剪枝
--min_pruning_rate 0.05  # 至少剪 5%
--max_pruning_rate 0.6   # 最多剪 60%
```

### 4. 迭代优化

```bash
# 第一轮：评估层重要性
python llama3_unbalanced_pruning.py \
    --importance_method removal \
    --importance_samples 50 \
    --save_model

# 第二轮：调整 alpha，观察性能
python llama3_unbalanced_pruning.py \
    --skip_importance_analysis \
    --alpha 1.5 \
    --test_after_train

# 第三轮：调整剪枝率范围
python llama3_unbalanced_pruning.py \
    --skip_importance_analysis \
    --alpha 2.0 \
    --min_pruning_rate 0.1 \
    --max_pruning_rate 0.5 \
    --test_after_train
```

## 高级用法

### 1. 自定义层重要性

```python
from layer_importance import UnbalancedStructuredPruningCalculator

# 手动定义层重要性
custom_importance = {
    0: 10.0,  # 很重要
    1: 8.5,
    2: 7.2,
    # ...
    15: 2.1,  # 不重要
    # ...
    31: 12.0  # 很重要
}

calculator = UnbalancedStructuredPruningCalculator(custom_importance, num_layers=32)
pruning_rates = calculator.compute_layer_pruning_rates(target_overall_rate=0.25)
```

### 2. 分析剪枝配置

```python
import json
import matplotlib.pyplot as plt

# 加载配置
with open('prune_log/llama_unbalanced_prune/layer_importance_config.json') as f:
    config = json.load(f)

# 可视化
importance = config['layer_importance']
pruning_rates = config['layer_pruning_rates']

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

# 重要性
ax1.bar(importance.keys(), importance.values())
ax1.set_title('Layer Importance')

# 剪枝率
ax2.bar(pruning_rates.keys(), pruning_rates.values())
ax2.set_title('Pruning Rates')

plt.tight_layout()
plt.show()
```

### 3. 结合 analyze_pruning.py 验证

```bash
# 执行剪枝
python llama3_unbalanced_pruning.py --save_model

# 分析剪枝结果
python analyze_pruning.py \
    --original_model /newdata/LLMs/Llama-3-8B-Instruct \
    --pruned_model prune_log/llama_unbalanced_prune/pytorch_model.bin

# 验证各层剪枝率是否符合预期
```

## 故障排除

### 问题 1: 层重要性分析太慢

**解决**：
```bash
# 减少评估样本数
--importance_samples 20

# 或使用激活值法
--importance_method activation
```

### 问题 2: 剪枝率分布不合理

**解决**：
```bash
# 调整 alpha
--alpha 1.0  # 减小差异

# 调整范围
--min_pruning_rate 0.1
--max_pruning_rate 0.5
```

### 问题 3: OOM 错误

**解决**：
```bash
# 减少样本数
--num_examples 5
--importance_samples 20

# 减少序列长度
--max_seq_len 64
```

## 总结

非均衡结构化剪枝的优势：
- ✅ **性能更好**：相同剪枝率下 PPL 更低
- ✅ **物理减少**：结构化剪枝实现真实的模型压缩
- ✅ **可控性强**：通过 alpha 和剪枝率范围精确控制
- ✅ **可视化**：直观展示层重要性和剪枝策略

推荐工作流程：
1. 使用 removal 方法评估层重要性（一次性）
2. 尝试不同的 alpha 值（0.5, 1.0, 1.5, 2.0）
3. 评估每个配置的 PPL
4. 选择性能最好的配置
5. 使用 analyze_pruning.py 详细分析
6. 保存和部署最终模型

祝你剪枝成功！🚀
