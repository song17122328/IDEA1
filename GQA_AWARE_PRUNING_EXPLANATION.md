# GQA感知剪枝机制说明

## 问题背景

### 原始问题
在Llama-3-8B的GQA (Grouped Query Attention)结构中：
- **Q heads**: 32个（4096通道 = 32 × 128）
- **KV heads**: 8个（1024通道 = 8 × 128）
- **比例要求**: Q:KV = 4:1（`num_q_heads % num_kv_heads == 0`）

### torch_pruning的局限性

torch_pruning的依赖图传播机制是**通道对通道**的传播：

```python
# 原始模式（仅k_proj为root）：
ch_sparsity_dict[layer.self_attn.k_proj] = 0.25  # 只设置k_proj
# q_proj通过依赖图自动传播

# 实际剪枝过程：
k_proj: 1024通道 → 剪枝0.25 → 移除256通道（2个KV heads）→ 剩余768通道（6 KV heads）
q_proj: 4096通道 → 依赖图传播相同索引 → 也移除256通道 → 剩余3840通道（30 Q heads）

# 结果：30:6 = 5:1 ✗ (不是4:1)
```

**根本原因**：依赖图按"相同通道索引"传播，不理解GQA的4:1结构。当k_proj剪掉256通道时，q_proj也剪掉256通道，而不是1024通道（4倍）。

## 解决方案：GQA感知剪枝模式

### 核心思想

将q_proj、k_proj、v_proj都设置为**独立的root modules**，并使用**比例匹配的consecutive_groups**：

```python
# 1. 所有投影层都在ch_sparsity_dict中（相同剪枝率）
ch_sparsity_dict[layer.self_attn.q_proj] = 0.25
ch_sparsity_dict[layer.self_attn.k_proj] = 0.25
ch_sparsity_dict[layer.self_attn.v_proj] = 0.25

# 2. 设置比例匹配的consecutive_groups
consecutive_groups[layer.self_attn.q_proj] = 512   # 4个Q heads为一组
consecutive_groups[layer.self_attn.k_proj] = 128   # 1个KV head为一组
consecutive_groups[layer.self_attn.v_proj] = 128   # 1个KV head为一组

# 3. 都加入root_instances（独立剪枝）
root_instances = [q_proj, k_proj, v_proj, gate_proj, ...]
```

### 工作原理

以剪枝率0.25为例：

**q_proj剪枝**：
- 原始：32 Q heads = 8组 × 4 heads/组 = 4096通道
- consecutive_groups=512 → 按512通道（4个Q heads）为单位分组
- 剪枝率0.25 → 剪掉2组 → 移除1024通道（8个Q heads）
- 结果：**24 Q heads**（6组 × 4 heads）

**k_proj剪枝**：
- 原始：8 KV heads = 8组 × 1 head/组 = 1024通道
- consecutive_groups=128 → 按128通道（1个KV head）为单位分组
- 剪枝率0.25 → 剪掉2组 → 移除256通道（2个KV heads）
- 结果：**6 KV heads**

**验证比例**：24:6 = 4:1 ✓

### 代码修改

**layer_importance.py**:
```python
def create_ch_sparsity_dict_for_llama(..., gqa_aware=True):
    if gqa_aware:
        # 同时设置q/k/v为root
        ch_sparsity_dict[layer.self_attn.q_proj] = pruning_rate
        ch_sparsity_dict[layer.self_attn.k_proj] = pruning_rate
        ch_sparsity_dict[layer.self_attn.v_proj] = pruning_rate
```

**llama3_unbalanced_pruning.py**:
```python
# 1. 启用GQA感知模式
ch_sparsity_dict = create_ch_sparsity_dict_for_llama(
    model, effective_pruning_rates,
    gqa_aware=True  # ✓
)

# 2. 设置比例匹配的consecutive_groups
consecutive_groups = {}
for layer in model.model.layers:
    consecutive_groups[layer.self_attn.q_proj] = 512  # 4 Q heads
    consecutive_groups[layer.self_attn.k_proj] = 128  # 1 KV head
    consecutive_groups[layer.self_attn.v_proj] = 128  # 1 KV head

# 3. 所有投影层都作为root
root_instances = []
for layer_idx in actual_pruning_layers:
    root_instances.append(model.layers[layer_idx].self_attn.q_proj)
    root_instances.append(model.layers[layer_idx].self_attn.k_proj)
    root_instances.append(model.layers[layer_idx].self_attn.v_proj)
    root_instances.append(model.layers[layer_idx].mlp.gate_proj)
```

## 重要说明

### 1. 独立剪枝的含义

q_proj、k_proj、v_proj会**独立**根据重要性得分选择要剪枝的heads：

- **优点**：每个投影层保留各自最重要的heads
- **限制**：可能剪枝不同位置的heads

例如：
- q_proj可能剪掉组0和组1（Q heads 0-7）
- k_proj可能剪掉KV heads 2和5

这意味着剪枝后的Q heads和KV heads可能不是位置对齐的。

### 2. 后处理修正

后处理阶段会进一步调整以确保GQA约束：

```python
# 如果剪枝后：num_heads % num_kv_heads != 0
adjusted_num_heads = (num_heads // num_kv_heads) * num_kv_heads

# 调整q_proj权重
layer.self_attn.q_proj.weight.data =
    layer.self_attn.q_proj.weight.data[:adjusted_q_channels, :]
```

### 3. 为什么不精确对齐头位置？

torch_pruning的设计理念是**基于重要性的自动剪枝**，不支持精确控制剪枝哪些具体的heads。

要实现精确对齐，需要：
1. 自定义GQA Pruner（理解Q:KV的4:1关系）
2. 修改依赖图传播逻辑（当k_proj剪枝时，自动让q_proj剪枝对应的4倍通道）

这需要深度修改torch_pruning的核心代码，复杂度高。

## 预期结果

启用GQA感知剪枝后：

1. **head数量比例正确**：剪枝后Q:KV应该保持4:1（或更精确的比例）
2. **后处理微调减少**：由于比例已经接近正确，后处理只需要小幅调整
3. **剪枝质量提升**：每个投影层保留各自最重要的heads，而不是被动接受依赖图传播

## 使用方法

运行V2剪枝脚本（已启用GQA感知模式）：

```bash
bash run_pruning_v2.sh
```

检查日志中的：
- "GQA感知剪枝模式已启用" 确认已启用
- 每层的剪枝后head数量和比例
- 后处理的调整幅度（应该比之前小）

## 总结

| 模式 | k_proj剪枝 | q_proj剪枝 | 结果比例 | 后处理 |
|------|-----------|-----------|----------|--------|
| **传统模式** | 2 KV heads (256通道) | 依赖图传播256通道 ≈ 2 Q heads | 30:6 = 5:1 ✗ | 大幅调整 |
| **GQA感知** | 2 KV heads (256通道) | 独立剪枝2组 = 8 Q heads | 24:6 = 4:1 ✓ | 微调或不需要 |

**关键改进**：从"通道对通道传播"变为"比例匹配的独立剪枝"。
