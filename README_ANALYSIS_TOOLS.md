# LLM 剪枝分析工具集

这个工具集提供了完整的模型剪枝分析功能，帮助你理解剪枝前后模型的变化。

## 🛠️ 工具列表

### 1. `check_model_size.py` - 模型大小检查工具
快速检查模型文件的物理大小和参数数量。

**功能**：
- 显示模型文件的物理大小
- 统计模型参数数量
- 对比原始模型和剪枝模型的大小差异

**使用**：
```bash
# 检查所有剪枝后的模型
python check_model_size.py

# 对比原始模型和剪枝模型
python check_model_size.py \
    --original_model /path/to/original/model \
    --pruned_model prune_log/llama_prune/pytorch_model.bin
```

### 2. `analyze_pruning.py` - 剪枝详细分析工具 ⭐
**重点推荐！** 逐层分析模型剪枝的详细信息。

**功能**：
- 对比每一层的参数维度变化
- 计算每个模块的剪枝度（保留率）
- 计算结构化稀疏度（1 - 剪枝度）
- 分析 Attention 和 MLP 各个子模块
- 导出详细的 CSV 报告

**使用**：
```bash
python analyze_pruning.py \
    --original_model /newdata/LLMs/Llama-3-8B-Instruct \
    --pruned_model prune_log/llama_prune/pytorch_model.bin
```

**输出示例**：
```
==================================================================================================
每层汇总统计
==================================================================================================
 层编号  原始参数量     剪枝后参数量    保留率    稀疏度
     0  135,266,304   135,266,304   1.0000   0.0000  # 未剪枝
     3  135,266,304   113,397,760   0.8383   0.1617  # 已剪枝
    30  135,266,304   135,266,304   1.0000   0.0000  # 未剪枝

全局统计
整体保留率: 0.8393 (83.93%)
整体稀疏度: 0.1607 (16.07%)
```

### 3. `example_analysis.sh` - 快速分析脚本
一键运行完整分析的便捷脚本。

**使用**：
```bash
# 编辑脚本，设置正确的路径
vim example_analysis.sh

# 运行分析
./example_analysis.sh
```

## 📊 分析流程

### 步骤 1：运行剪枝
```bash
python llama3.py --pruning_ratio 0.25 \
    --base_model /newdata/LLMs/Llama-3-8B-Instruct \
    --block_wise \
    --block_mlp_layer_start 3 --block_mlp_layer_end 30 \
    --block_attention_layer_start 3 --block_attention_layer_end 30 \
    --pruner_type taylor \
    --save_model
```

### 步骤 2：检查模型大小
```bash
python check_model_size.py \
    --original_model /newdata/LLMs/Llama-3-8B-Instruct \
    --pruned_model prune_log/llama_prune/pytorch_model.bin
```

预期输出：
```
原始模型总大小: 14.97 GB
剪枝后模型大小: 12.54 GB
物理空间减少: 2.43 GB (16.22%)
```

### 步骤 3：详细剪枝分析
```bash
python analyze_pruning.py \
    --original_model /newdata/LLMs/Llama-3-8B-Instruct \
    --pruned_model prune_log/llama_prune/pytorch_model.bin \
    --output detailed_analysis.csv
```

### 步骤 4：分析 CSV 结果
```python
import pandas as pd
import matplotlib.pyplot as plt

# 读取分析结果
df = pd.read_csv('detailed_analysis.csv')

# 查看特定层的信息
layer_3 = df[df['层编号'] == 3]
print(layer_3)

# 计算每层的平均稀疏度
layer_sparsity = df.groupby('层编号')['稀疏度'].apply(
    lambda x: pd.to_numeric(x, errors='coerce').mean()
)

# 可视化
plt.figure(figsize=(15, 5))
plt.bar(layer_sparsity.index, layer_sparsity.values)
plt.xlabel('Layer Index')
plt.ylabel('Sparsity')
plt.title('Layer-wise Sparsity Distribution')
plt.axhline(y=0.162, color='r', linestyle='--', label='Target Sparsity (16.2%)')
plt.legend()
plt.savefig('sparsity_visualization.png', dpi=150)
plt.show()
```

## 🔍 理解输出

### Llama-3-8B-Instruct 模型结构

**总体信息**：
- 32 个 Transformer 层
- 隐藏维度：4096
- MLP 中间维度：14336
- 注意力头数：32
- KV 头数：8 (GQA)

**每层模块**：
```
Attention:
  ├─ q_proj: [4096, 4096]
  ├─ k_proj: [1024, 4096]  # GQA，维度较小
  ├─ v_proj: [1024, 4096]  # GQA，维度较小
  └─ o_proj: [4096, 4096]

MLP:
  ├─ gate_proj: [14336, 4096]  # SwiGLU
  ├─ up_proj:   [14336, 4096]  # SwiGLU
  └─ down_proj: [4096, 14336]
```

### 剪枝后的变化

根据你的配置（`--block_*_layer_start 3 --block_*_layer_end 30`）：

**未剪枝的层**（第 0-2 层和第 30-31 层）：
```
保留率: 1.0000 (100%)
稀疏度: 0.0000 (0%)
所有维度保持不变
```

**剪枝的层**（第 3-29 层，共 27 层）：
```
保留率: ~0.838 (83.8%)
稀疏度: ~0.162 (16.2%)

示例维度变化:
  q_proj: [4096, 4096] → [3432, 4096]
  gate_proj: [14336, 4096] → [12016, 4096]
```

### 关键指标

**剪枝度（保留率）**：
```
剪枝度 = 剪枝后参数量 / 原始参数量
```
- 表示保留了多少参数
- 1.0 = 未剪枝
- 0.838 = 保留了 83.8% 的参数

**结构化稀疏度**：
```
稀疏度 = 1 - 剪枝度
```
- 表示删除了多少参数
- 0.0 = 未剪枝
- 0.162 = 删除了 16.2% 的参数

## 📝 常见问题

### Q1: 为什么不剪枝第 0-2 层和第 30-31 层？
**A**:
- **浅层**（0-2）负责学习基础特征，对性能影响大
- **深层**（30-31）负责最终输出，剪枝风险高
- **中间层**（3-29）有更多冗余，可以安全剪枝

### Q2: 每层的剪枝度一样吗？
**A**: 使用相同的剪枝比例（如 25%），每层被剪枝的模块保留率应该相近，但：
- 不同模块（Q/K/V/O/Gate/Up/Down）可能略有差异
- 依赖于 Taylor 重要性评分的结果
- 工具会标注是否一致

### Q3: 如何验证剪枝是否成功？
**A**:
1. 检查未剪枝层（0-2, 30-31）的保留率应为 1.0
2. 检查剪枝层（3-29）的稀疏度应接近目标（16.2%）
3. 整体稀疏度应匹配预期
4. 维度变化应该合理（输出维度减少，输入维度保持）

### Q4: 物理文件大小为什么没有减少 16.22%？
**A**:
- 模型文件包含元数据、配置、tokenizer 等
- PyTorch 序列化有额外开销
- 只有权重参数部分减少了 16.22%
- 实际文件大小减少约 2.43 GB（从 14.97 GB → 12.54 GB）

### Q5: 如何使用分析结果？
**A**:
```python
import pandas as pd

df = pd.read_csv('detailed_analysis.csv')

# 1. 查看特定层
print(df[df['层编号'] == 3])

# 2. 比较不同模块
print(df.groupby('模块名称')['稀疏度'].describe())

# 3. 找出稀疏度异常的层
df['稀疏度'] = pd.to_numeric(df['稀疏度'], errors='coerce')
outliers = df[df['稀疏度'] > 0.2]  # 稀疏度超过 20%
print(outliers)

# 4. 验证剪枝范围
unpruned = df[df['稀疏度'] == 0.0]['层编号'].unique()
print(f"未剪枝的层: {sorted(unpruned)}")
```

## 📚 相关文档

- [`MODEL_SAVE_STRUCTURE.md`](MODEL_SAVE_STRUCTURE.md) - 模型保存结构说明
- [`PRUNING_ANALYSIS_GUIDE.md`](PRUNING_ANALYSIS_GUIDE.md) - 剪枝分析详细指南
- [`check_model_size.py`](check_model_size.py) - 模型大小检查工具
- [`analyze_pruning.py`](analyze_pruning.py) - 剪枝分析工具

## 🔧 依赖要求

```bash
pip install torch transformers pandas
```

## 💡 最佳实践

1. **剪枝前**：先用 `check_model_size.py` 确认原始模型大小
2. **剪枝后**：
   - 用 `check_model_size.py` 快速检查文件大小
   - 用 `analyze_pruning.py` 详细分析每层变化
3. **验证**：检查未剪枝层和剪枝层的稀疏度是否符合预期
4. **记录**：保存 CSV 文件用于后续分析和论文写作

## 🎯 示例输出

运行 `analyze_pruning.py` 后的典型输出：

```
==================================================================================================
全局统计
==================================================================================================
原始模型总参数量: 8,030,261,248
剪枝后模型总参数量: 6,727,929,856
参数减少量: 1,302,331,392
整体保留率: 0.8378 (83.78%)
整体稀疏度: 0.1622 (16.22%)

==================================================================================================
详细模块分析
==================================================================================================
📊 Attention Q Projection
 层  原始维度      剪枝后维度    保留率    稀疏度
  0  4096 × 4096  4096 × 4096  1.0000   0.0000
  1  4096 × 4096  4096 × 4096  1.0000   0.0000
  2  4096 × 4096  4096 × 4096  1.0000   0.0000
  3  4096 × 4096  3432 × 4096  0.8379   0.1621
  4  4096 × 4096  3432 × 4096  0.8379   0.1621
  ...
 30  4096 × 4096  4096 × 4096  1.0000   0.0000
 31  4096 × 4096  4096 × 4096  1.0000   0.0000

✅ 所有层的 Attention Q Projection 剪枝度一致: 0.8379
```

## 🚀 快速开始

```bash
# 1. 运行剪枝（如果还没运行）
python llama3.py --pruning_ratio 0.25 --save_model ...

# 2. 快速检查大小
python check_model_size.py

# 3. 详细分析
python analyze_pruning.py \
    --original_model /newdata/LLMs/Llama-3-8B-Instruct \
    --pruned_model prune_log/llama_prune/pytorch_model.bin

# 4. 查看结果
head -20 pruning_analysis.csv
```

祝你剪枝成功！🎉
