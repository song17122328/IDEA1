# 剪枝分析工具使用指南

## 功能说明

这个工具用于详细分析原始模型和剪枝后模型的差异，包括：

1. **每一层的参数维度对比**：展示每个模块（Attention 和 MLP）的权重维度变化
2. **剪枝度（保留率）**：剪枝后参数量 ÷ 原始参数量
3. **结构化稀疏度**：1 - 剪枝度，表示被剪掉的参数比例
4. **逐模块分析**：分别分析 Q/K/V/O projection 和 MLP 的 gate/up/down projection
5. **CSV 导出**：保存详细分析结果供后续使用

## Llama-3-8B-Instruct 模型结构

### 基本信息
- **总层数**：32 个 Transformer 层
- **隐藏维度**：4096
- **中间维度（MLP）**：14336
- **注意力头数**：32
- **KV 头数**：8（使用 GQA - Grouped Query Attention）

### 每层包含的模块

#### 1. Attention 模块 (self_attn)
```python
q_proj: Linear(4096 → 4096)   # Query projection
k_proj: Linear(4096 → 1024)   # Key projection (GQA)
v_proj: Linear(4096 → 1024)   # Value projection (GQA)
o_proj: Linear(4096 → 4096)   # Output projection
```

#### 2. MLP 模块 (mlp)
```python
gate_proj: Linear(4096 → 14336)  # Gate projection (SwiGLU)
up_proj:   Linear(4096 → 14336)  # Up projection (SwiGLU)
down_proj: Linear(14336 → 4096)  # Down projection
```

### 剪枝的目标模块

根据你的剪枝配置：
```bash
--block_attention_layer_start 3 --block_attention_layer_end 30
--block_mlp_layer_start 3 --block_mlp_layer_end 30
```

**剪枝范围**：
- 第 3-29 层的 Attention 模块（共 27 层）
- 第 3-29 层的 MLP 模块（共 27 层）
- 第 0-2 层和第 30-31 层**不剪枝**

## 使用方法

### 基本用法

```bash
python analyze_pruning.py \
    --original_model /newdata/LLMs/Llama-3-8B-Instruct \
    --pruned_model prune_log/llama_prune/pytorch_model.bin
```

### 指定输出文件

```bash
python analyze_pruning.py \
    --original_model /newdata/LLMs/Llama-3-8B-Instruct \
    --pruned_model prune_log/llama_prune/pytorch_model.bin \
    --output my_analysis.csv
```

### 只分析特定层

```bash
# 分析第 0-5 层
python analyze_pruning.py \
    --original_model /newdata/LLMs/Llama-3-8B-Instruct \
    --pruned_model prune_log/llama_prune/pytorch_model.bin \
    --layers "0-5"

# 分析第 3, 10, 20 层
python analyze_pruning.py \
    --original_model /newdata/LLMs/Llama-3-8B-Instruct \
    --pruned_model prune_log/llama_prune/pytorch_model.bin \
    --layers "3,10,20"
```

## 输出说明

### 1. 控制台输出

#### 每层汇总统计
```
==================================================================================================
每层汇总统计
==================================================================================================

 层编号  原始参数量     剪枝后参数量    保留率    稀疏度
     0  135,266,304   135,266,304   1.0000   0.0000
     1  135,266,304   135,266,304   1.0000   0.0000
     2  135,266,304   135,266,304   1.0000   0.0000
     3  135,266,304   113,397,760   0.8383   0.1617
     4  135,266,304   113,397,760   0.8383   0.1617
   ...
```

**解释**：
- **层编号**：Transformer 层的索引（0-31）
- **原始参数量**：该层在原始模型中的总参数数量
- **剪枝后参数量**：该层在剪枝后的参数数量
- **保留率**：剪枝后参数量 ÷ 原始参数量（即剪枝度）
- **稀疏度**：1 - 保留率（被剪掉的参数比例）

#### 全局统计
```
==================================================================================================
全局统计
==================================================================================================
原始模型总参数量: 4,328,521,728
剪枝后模型总参数量: 3,632,906,240
参数减少量: 695,615,488
整体保留率: 0.8393 (83.93%)
整体稀疏度: 0.1607 (16.07%)
```

#### 详细模块分析
```
────────────────────────────────────────────────────────────────────────────────────────────────
📊 Attention Q Projection
────────────────────────────────────────────────────────────────────────────────────────────────
 层  原始维度      剪枝后维度    保留率    稀疏度
  0  4096 × 4096  4096 × 4096  1.0000   0.0000
  1  4096 × 4096  4096 × 4096  1.0000   0.0000
  2  4096 × 4096  4096 × 4096  1.0000   0.0000
  3  4096 × 4096  3432 × 4096  0.8379   0.1621
  4  4096 × 4096  3432 × 4096  0.8379   0.1621
...

✅ 所有层的 Attention Q Projection 剪枝度一致: 0.8379
```

**解释**：
- **原始维度**：权重矩阵的形状（输出维度 × 输入维度）
- **剪枝后维度**：剪枝后的权重矩阵形状
- **保留率**：该模块的参数保留比例
- **稀疏度**：该模块的结构化稀疏度

### 2. CSV 文件输出

生成的 CSV 文件包含所有层所有模块的详细信息：

| 层编号 | 模块名称 | 原始维度 | 剪枝后维度 | 保留率 | 稀疏度 |
|--------|----------|----------|------------|---------|---------|
| 0 | attn_q_proj | 4096 × 4096 | 4096 × 4096 | 1.0000 | 0.0000 |
| 0 | attn_k_proj | 1024 × 4096 | 1024 × 4096 | 1.0000 | 0.0000 |
| ... | ... | ... | ... | ... | ... |
| 3 | attn_q_proj | 4096 × 4096 | 3432 × 4096 | 0.8379 | 0.1621 |
| 3 | mlp_gate_proj | 14336 × 4096 | 12016 × 4096 | 0.8382 | 0.1618 |

可以使用 Excel、pandas 或其他工具打开进行进一步分析。

## 预期分析结果

### 未剪枝的层（第 0-2 层和第 30-31 层）
```
保留率: 1.0000 (100%)
稀疏度: 0.0000 (0%)
维度: 保持不变
```

### 剪枝的层（第 3-29 层）
```
保留率: ~0.838 (83.8%)
稀疏度: ~0.162 (16.2%)
维度变化示例:
  - q_proj: 4096 × 4096 → 3432 × 4096  (输出维度减少)
  - k_proj: 1024 × 4096 → 1024 × 4096  (可能不变，取决于剪枝策略)
  - gate_proj: 14336 × 4096 → 12016 × 4096  (中间维度减少)
```

## 理解剪枝策略

### 结构化剪枝
- **不是**随机删除权重中的个别参数
- **而是**删除整个神经元/通道/注意力头
- 保持权重矩阵的结构，只改变维度

### 示例
原始 Q projection: `[4096, 4096]`
- 4096 个输出神经元，每个接收 4096 维输入
- 剪枝后: `[3432, 4096]`
- 删除了 664 个输出神经元（整行）
- 保留了 3432 个神经元
- 保留率: 3432/4096 = 83.79%

### 为什么要保留第 0-2 层和第 30-31 层？
- **浅层**（0-2）：学习基础特征，对模型性能影响大
- **深层**（30-31）：负责最终输出，剪枝影响大
- **中间层**（3-29）：有更多冗余，可以安全剪枝

## 高级分析

### 使用 pandas 分析 CSV

```python
import pandas as pd
import matplotlib.pyplot as plt

# 读取CSV
df = pd.read_csv('pruning_analysis.csv')

# 按层分组，计算平均稀疏度
layer_sparsity = df.groupby('层编号')['稀疏度'].apply(
    lambda x: pd.to_numeric(x, errors='coerce').mean()
)

# 可视化
plt.figure(figsize=(15, 5))
plt.bar(layer_sparsity.index, layer_sparsity.values)
plt.xlabel('Layer Index')
plt.ylabel('Sparsity')
plt.title('Layer-wise Sparsity')
plt.savefig('layer_sparsity.png')

# 分析不同模块的稀疏度
module_sparsity = df.groupby('模块名称')['稀疏度'].apply(
    lambda x: pd.to_numeric(x, errors='coerce').mean()
)
print(module_sparsity)
```

### 验证剪枝一致性

```python
# 检查被剪枝的层是否剪枝度一致
pruned_layers = df[df['层编号'].between(3, 29)]
consistency = pruned_layers.groupby('模块名称')['保留率'].nunique()
print("不同保留率的数量 (应该为1表示一致):")
print(consistency)
```

## 故障排除

### 问题：加载模型时 CUDA OOM
**解决**：
```bash
# 设置环境变量，使用 CPU 加载
export CUDA_VISIBLE_DEVICES=""
python analyze_pruning.py ...
```

### 问题：weights_only 错误
**解决**：代码已经添加了 `weights_only=False`，如果仍有问题，确保使用 PyTorch 2.0+

### 问题：找不到层
**解决**：确保模型路径正确，剪枝后的模型应该是 `.bin` 文件

## 参考资料

- [Llama 3 模型结构](https://github.com/meta-llama/llama3)
- [结构化剪枝原理](https://arxiv.org/abs/2301.00774)
- [Grouped Query Attention (GQA)](https://arxiv.org/abs/2305.13245)
