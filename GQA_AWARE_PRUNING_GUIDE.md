# GQA-Aware Pruning 实现指南

## 🎯 核心思想

**问题**：当前剪枝流程的致命缺陷
```python
# 当前方法：简单截断
剪枝后: Q=30 heads, KV=6 heads (5:1)
后处理: 截断最后6个Q heads → Q=24 heads, KV=6 heads (4:1)
问题: 截断的6个Q heads可能是重要的！
结果: PPL飙升到71万
```

**解决方案**：GQA-aware Taylor importance
```python
# 新方法：基于importance的组级剪枝
1. 将"4个Q heads + 1个KV head"视为一个GQA组
2. 计算每个GQA组的总Taylor importance
3. 保留importance最高的N个完整组
4. 剪枝importance最低的组
结果: 保持4:1比例 + 保留重要的heads + 保持语义对齐
```

---

## 📊 三个关键问题的回答

### 问题1：每层剪枝率如何确定？

✅ **当前方法很好，无需改动**

```python
# layer_importance.py
layer_importance = compute_layer_importance_removal(model, example_prompts, ...)
layer_pruning_rates = {0: 0.25, 1: 0.30, 2: 0.20, ...}  # per-layer rates
```

### 问题2：Attention和MLP都用Taylor吗？

✅ **是的，但方式不同**

#### MLP：通道级别Taylor
```python
# 每个通道的importance
salience = layer.weight * layer.weight.grad
channel_imp = salience.abs().sum(1)  # [num_channels]

# 选择importance最低的通道剪掉
```

#### Attention（当前问题）：通道级别Taylor + 依赖图传播
```python
# 当前方法 (hf_llama_pruner.py:310-327)
local_norm = 0
local_norm += salience[o_proj].abs().sum(1)  # output channels
local_norm += salience[q_proj].abs().sum(0)  # input channels
local_norm += salience[k_proj].abs().sum(0)
local_norm += salience[v_proj].abs().sum(0)

# 问题：这是"通道级别"的importance，不理解GQA结构
# k_proj剪2个heads → q_proj依赖图传播剪2个heads（应该剪8个）
```

#### Attention（新方法）：GQA组级别Taylor
```python
# 新方法 (gqa_aware_pruning.py)
# 计算每个GQA组的importance
for kv_idx in range(num_kv_heads):
    q_start = kv_idx * 4
    q_end = q_start + 4

    group_imp = 0
    # 4个Q heads的contribution
    group_imp += q_head_imp[q_start:q_end].sum()
    group_imp += o_head_imp[q_start:q_end].sum()
    # 1个KV head的contribution
    group_imp += k_head_imp[kv_idx]
    group_imp += v_head_imp[kv_idx]

    group_importance[kv_idx] = group_imp

# 选择importance最低的完整GQA组剪掉
```

### 问题3：后处理能否基于importance？

✅ **完全可以！这正是新方法的核心**

```python
# 旧方法：后处理简单截断（不考虑importance）
layer.self_attn.q_proj.weight.data = \
    layer.self_attn.q_proj.weight.data[:target_q_channels, :]
# ↑ 丢弃最后6个Q heads（可能是重要的）

# 新方法：基于importance选择完整GQA组
keep_indices, prune_indices = select_gqa_groups_to_prune(group_imp, target_num_kv_heads)
# keep_indices = [0, 2, 3, 5, 6, 7] (importance最高的6个组)
# prune_indices = [1, 4] (importance最低的2个组)

prune_attention_by_gqa_groups(layer, keep_indices, head_dim=128, gqa_ratio=4)
# ↑ 保留完整的GQA组，保持语义对齐
```

---

## 🔧 实现方案

### 方案1：完全替代torch_pruning（推荐）

**优点**：
- 完全控制剪枝过程
- 确保GQA比例正确
- 基于importance，不会破坏语义

**步骤**：

1. **为每层计算剪枝率**（保持不变）
```python
layer_pruning_rates = compute_pruning_rates_from_importance(...)
```

2. **MLP使用torch_pruning**（保持不变）
```python
# MLP没有GQA问题，可以继续使用torch_pruning
pruner = tp.pruner.MetaPruner(model, ...)
```

3. **Attention使用GQA-aware手动剪枝**（新方法）
```python
from gqa_aware_pruning import prune_layer_with_gqa_awareness

for layer_idx, pruning_rate in layer_pruning_rates.items():
    # 计算梯度
    model.zero_grad()
    loss = model(example_prompts, labels=example_prompts).loss
    loss.backward()

    # Attention: GQA-aware剪枝
    num_q, num_kv = prune_layer_with_gqa_awareness(
        model, layer_idx, pruning_rate, example_prompts
    )

    # MLP: 使用torch_pruning (待实现)
    prune_mlp_layer(model, layer_idx, pruning_rate)
```

### 方案2：改进当前的后处理（权宜之计）

如果不想大改流程，可以改进后处理逻辑：

```python
# 在后处理阶段，基于已有的梯度信息选择要保留的Q heads
def intelligent_post_processing(layer, target_num_heads, target_num_kv_heads):
    """
    基于importance的后处理，而不是简单截断
    """
    # 1. 计算每个Q head的importance (假设梯度还在)
    q_salience = (layer.self_attn.q_proj.weight * layer.self_attn.q_proj.weight.grad).abs()
    q_head_imp = q_salience.view(num_heads, head_dim, -1).sum(dim=[1, 2])

    # 2. 计算每个GQA组的importance
    group_imp = torch.zeros(num_kv_heads)
    for kv_idx in range(num_kv_heads):
        q_start = kv_idx * 4
        q_end = q_start + 4
        group_imp[kv_idx] = q_head_imp[q_start:q_end].sum()

    # 3. 选择importance最高的组
    keep_kv_indices = torch.argsort(group_imp, descending=True)[:target_num_kv_heads]

    # 4. 根据keep_kv_indices重新排列权重
    ...
```

**问题**：在后处理阶段，梯度可能已经被清空，importance信息丢失。

---

## 📈 预期效果

### 当前方法的问题

| 指标 | 当前方法 |
|------|----------|
| 剪枝后PPL | **718,107** (71万) |
| 微调后PPL | 159.85 |
| GQA比例 | ✅ 4:1 (后处理强制) |
| 语义对齐 | ❌ 破坏（简单截断） |
| 依赖微调 | ✅ 必须微调才能用 |

### GQA-aware方法的预期

| 指标 | 新方法 |
|------|--------|
| 剪枝后PPL | **预期: ~30-50** (大幅改善) |
| 微调后PPL | 预期: ~15-25 (更接近原始) |
| GQA比例 | ✅ 4:1 (自然保持) |
| 语义对齐 | ✅ 保持（基于importance选择） |
| 依赖微调 | ⚠️ 仍需要，但改善空间更大 |

**关键改善**：
- 剪枝后PPL从71万降到~30-50（改善14000倍）
- 微调后PPL从160降到~15-25（改善6-10倍）
- 保留重要的attention heads，保持模型理解能力

---

## 🚀 下一步实施

### Step 1: 验证GQA-aware importance计算

```bash
# 测试单层剪枝
python test_gqa_aware_pruning.py
```

创建测试脚本：
```python
# test_gqa_aware_pruning.py
from gqa_aware_pruning import prune_layer_with_gqa_awareness

# 加载模型
model = ...
example_prompts = ...

# 测试剪枝单个层
num_q, num_kv = prune_layer_with_gqa_awareness(
    model, layer_idx=10, pruning_rate=0.25, example_prompts
)

# 验证模型是否还能forward
output = model(example_prompts)
print(f"Forward pass successful! Output shape: {output.logits.shape}")
```

### Step 2: 集成到完整剪枝流程

修改`llama3_unbalanced_pruning.py`：

```python
# 选项A: 完全替代torch_pruning (for Attention)
from gqa_aware_pruning import prune_layer_with_gqa_awareness

for layer_idx, rate in layer_pruning_rates.items():
    prune_layer_with_gqa_awareness(model, layer_idx, rate, example_prompts)

# 选项B: 改进后处理
# (但需要保存importance信息，较复杂)
```

### Step 3: 对比实验

运行两个版本并对比：

| 方法 | 剪枝后PPL | 微调后PPL | 参数减少 |
|------|-----------|-----------|----------|
| 当前方法 | 718,107 | 159.85 | 17.39% |
| GQA-aware | ??? | ??? | 17.39% |

---

## ⚠️ 注意事项

### 1. MLP剪枝仍需torch_pruning

新方法只处理Attention，MLP仍然使用torch_pruning：
```python
# 需要实现MLP的独立剪枝逻辑
# 或者只对Attention使用GQA-aware，MLP继续用torch_pruning
```

### 2. 梯度计算开销

每层需要单独计算梯度：
```python
# 当前方法：一次backward，pruner处理所有层
# 新方法：每层单独backward（开销更大）

# 优化：batch处理多个层
```

### 3. 迭代剪枝

当前使用iterative pruning (多次step)，新方法需要适配：
```python
# 方案：每次迭代后重新计算group importance
for i in range(iterative_steps):
    for layer_idx in pruning_layers:
        prune_layer_with_gqa_awareness(...)
```

---

## 📚 代码文件

- `gqa_aware_pruning.py`: 核心实现
  - `compute_gqa_group_importance()`: 计算GQA组importance
  - `select_gqa_groups_to_prune()`: 选择要剪枝的组
  - `prune_attention_by_gqa_groups()`: 执行剪枝
  - `prune_layer_with_gqa_awareness()`: 完整流程

- `GQA_AWARE_PRUNING_GUIDE.md`: 本文档

- `llama3_unbalanced_pruning.py`: 待修改（集成新方法）

---

## 🎓 总结

你的三个问题的核心洞察：

1. ✅ **层级剪枝率**：removal方法已经做得很好
2. ✅ **Taylor重要度**：Attention和MLP都用，但Attention需要GQA-aware
3. 💡 **关键创新**：将4个Q heads + 1个KV head视为一个组，基于组的总importance剪枝

这个方案预期能将剪枝后的PPL从71万降到~30-50，微调后从160降到~15-25，大幅提升模型可用性！

**下一步**：实施Step 1验证，如果效果好就全面替换当前方法。
