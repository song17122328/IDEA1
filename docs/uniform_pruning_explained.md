# 原始均匀剪枝详细解析

## 🎯 核心思想

**所有指定的层使用相同的剪枝率**，通过一个全局参数 `ch_sparsity` 控制。

---

## 📝 完整代码流程（llama3.py）

### 第1步：初始化重要性评估器

```python
# llama3.py:98-107
pruner_type = args.pruner_type.lower()  # 例如: 'taylor'

if pruner_type == 'taylor':
    imp = llama_pruner.TaylorImportance(
        group_reduction=args.grouping_strategy,  # 'sum' 或 'mean'
        taylor=args.taylor  # 'param_first' 或 'param_mix'
    )
```

**Taylor 重要性：**
```
importance = |∂L/∂W × W|

其中:
- ∂L/∂W: 损失对权重的梯度（通过反向传播得到）
- W: 权重本身
- |·|: 绝对值
```

这个重要性会用于**在每一层内部**选择哪些 head 要剪枝。

---

### 第2步：配置剪枝参数（关键！）

```python
# llama3.py:112-129
kwargs = {
    "importance": imp,                          # Taylor 重要性评估器
    "global_pruning": args.global_pruning,      # False: 局部剪枝，True: 全局剪枝
    "iterative_steps": args.iterative_steps,    # 迭代次数（通常为1）

    # ⭐ 核心参数：全局剪枝率
    "ch_sparsity": args.pruning_ratio,          # 例如: 0.25 (所有层都用这个)

    "ignored_layers": [],                       # 忽略的层（空列表）

    # channel_groups: 空字典（不使用）
    "channel_groups": {},

    # ⭐ consecutive_groups: 强制 head-level 剪枝
    "consecutive_groups": {
        layer.self_attn.k_proj: layer.self_attn.head_dim  # 每个 k_proj: 128
        for layer in model.model.layers
    },

    "customized_pruners": {
        LlamaRMSNorm: llama_pruner.hf_rmsnorm_pruner,
    },

    "root_module_types": None,

    # ⭐ root_instances: 指定哪些层要剪枝
    "root_instances": [
        model.model.layers[i].self_attn.k_proj
        for i in range(args.block_attention_layer_start, args.block_attention_layer_end)
    ] + [
        model.model.layers[i].mlp.gate_proj
        for i in range(args.block_mlp_layer_start, args.block_mlp_layer_end)
    ]
}
```

---

### 第3步：关键参数详解

#### 3.1 `ch_sparsity: 0.25`

**含义：** 所有在 `root_instances` 中的模块都剪枝 25%

```python
# 如果 ch_sparsity = 0.25
对于每个 root_instance（例如 Layer 5 的 k_proj）:
  - 原始输出通道: 1024 (8 个 heads × 128)
  - 剪枝后通道: 1024 × (1 - 0.25) = 768 (6 个 heads × 128)
  - 剪枝的通道: 256 (2 个 heads × 128)
```

**重点：**
- ✅ 所有层使用**相同的剪枝率** (0.25)
- ❌ 没有 `ch_sparsity_dict`，无法为每层指定不同的剪枝率
- ✅ 简单、可预测

#### 3.2 `consecutive_groups`

**含义：** 强制每个 k_proj 必须剪枝完整的 head（128 个连续通道）

```python
"consecutive_groups": {
    model.layers[0].self_attn.k_proj: 128,
    model.layers[1].self_attn.k_proj: 128,
    ...
    model.layers[31].self_attn.k_proj: 128,
}
```

**工作原理：**

```python
# 对于 Layer 5 的 k_proj
k_proj: [4096, 1024]  # 8 个 heads

consecutive_group_size = 128  # head_dim

# 剪枝时的约束:
剪枝的通道索引必须是连续的 128 个通道的倍数

例如:
  ✅ 剪枝 head 2 和 head 5: 通道 [256:384] 和 [640:768]
  ❌ 剪枝零散的通道: [10, 25, 67, 89, ...]
```

**为什么需要？**

Head-level 剪枝才能保持 attention 机制的完整性！

#### 3.3 `root_instances`

**含义：** 指定从哪些模块开始剪枝

```python
# 默认参数:
# --block_attention_layer_start 3
# --block_attention_layer_end 30

root_instances = [
    model.layers[3].self_attn.k_proj,   # Layer 3 Attention
    model.layers[4].self_attn.k_proj,   # Layer 4 Attention
    ...
    model.layers[29].self_attn.k_proj,  # Layer 29 Attention
    model.layers[3].mlp.gate_proj,      # Layer 3 MLP
    model.layers[4].mlp.gate_proj,      # Layer 4 MLP
    ...
    model.layers[29].mlp.gate_proj,     # Layer 29 MLP
]

# 总共: 27 层 × 2 模块 = 54 个 root modules
```

**剪枝流程：**

1. **从 root_instances 开始**
   - k_proj 被剪枝 → 自动传播到 q_proj, v_proj, o_proj
   - gate_proj 被剪枝 → 自动传播到 up_proj, down_proj

2. **不在 root_instances 中的层不会被剪枝**
   - Layer 0, 1, 2: 不剪枝（保护）
   - Layer 30, 31: 不剪枝（保护）

---

### 第4步：MetaPruner 执行剪枝

```python
# llama3.py:133-177
pruner = tp.pruner.MetaPruner(model, forward_prompts, **kwargs)
model.zero_grad()

logger.log("Start Pruning")
for i in range(args.iterative_steps):  # 通常只有 1 次迭代

    # 如果使用 Taylor 重要性，需要计算梯度
    if pruner_type == 'taylor':
        example_prompts = get_examples('bookcorpus', tokenizer, 10, seq_len=64)
        loss = model(example_prompts, labels=example_prompts).loss
        loss.backward()  # 计算梯度

    # 执行剪枝
    pruner.step()

    # 更新 attention 配置
    for layer in model.model.layers:
        layer.self_attn.num_heads = layer.self_attn.q_proj.weight.shape[0] // 128
        layer.self_attn.num_key_value_heads = layer.self_attn.k_proj.weight.shape[0] // 128
```

---

## 🔍 MetaPruner 内部是如何工作的？

### 阶段1：构建依赖图

```python
# 使用 forward_prompts 执行一次前向传播
forward_prompts = torch.tensor([
    [1, 306, 4658, 278, 6593, 310, 2834, 338],
    [1, 3439, 17632, 1925, 29892, 278, 6368, 310],
]).to(device)

output = model(forward_prompts)
```

**目的：** 追踪模块之间的连接关系

```
Layer 5 示例:

输入 (hidden_states)
  ├─→ q_proj [4096, 4096]
  ├─→ k_proj [4096, 1024]  ← root module
  └─→ v_proj [4096, 1024]
       ↓
   Attention 计算
       ↓
   o_proj [4096, 4096]
       ↓
   输出 (hidden_states)
```

依赖图记录：
- k_proj 的输出连接到 Attention
- q_proj, v_proj 的输出也连接到 Attention
- Attention 的输出连接到 o_proj
- o_proj 的输出连接到下一层的输入

### 阶段2：计算每个通道的重要性

对于每个 **root_instance**（例如 Layer 5 的 k_proj）：

```python
# k_proj: [4096, 1024]
# 1024 个输出通道 = 8 个 heads

# 使用 Taylor 重要性
for head_idx in range(8):
    channels = range(head_idx * 128, (head_idx + 1) * 128)  # 128 个连续通道

    # 计算这个 head 的重要性
    importance[head_idx] = sum(|∂L/∂W[c] × W[c]| for c in channels)

# 结果示例:
head_importance = [
    0.523,  # head 0
    0.891,  # head 1
    0.156,  # head 2 ← 最不重要
    0.734,  # head 3
    0.621,  # head 4
    0.445,  # head 5 ← 第二不重要
    0.812,  # head 6
    0.678,  # head 7
]
```

### 阶段3：选择要剪枝的 heads

```python
# ch_sparsity = 0.25
num_heads = 8
num_to_prune = int(8 * 0.25) = 2  # 剪枝 2 个 heads

# 选择重要性最低的 2 个 heads
sorted_heads = [2, 5, 4, 7, 0, 3, 6, 1]  # 按重要性升序
heads_to_prune = sorted_heads[:2] = [2, 5]  # 最不重要的 2 个

# 剪枝的通道索引
pruning_indices = [
    range(2 * 128, 3 * 128),  # head 2: [256, 257, ..., 383]
    range(5 * 128, 6 * 128),  # head 5: [640, 641, ..., 767]
]
```

### 阶段4：传播剪枝决策

```python
# 根据依赖图自动传播

k_proj 剪枝 heads [2, 5]:
  → v_proj 也剪枝 heads [2, 5]  (同步，因为都是 KV heads)
  → q_proj 剪枝对应的 Q heads
      (因为 GQA 4:1，剪枝 2 个 KV heads → 剪枝 8 个 Q heads)
      例如: Q heads [8, 9, 20, 21]（对应 KV heads 2 和 5）
  → o_proj 的输入维度相应减少
```

**GQA 比例传播：**

```
原始:
  k_proj: 1024 (8 KV heads)
  q_proj: 4096 (32 Q heads)
  比例: 32:8 = 4:1

剪枝 2 个 KV heads:
  k_proj: 768 (6 KV heads)
  q_proj: 3072 (24 Q heads)
  比例: 24:6 = 4:1 ✅ 保持不变！
```

### 阶段5：物理执行剪枝

```python
# 对于 k_proj
original_weight = k_proj.weight  # [1024, 4096]

# 删除要剪枝的 heads (head 2 和 head 5)
keep_indices = [0,1, 3,4, 6,7]  # 保留的 heads
keep_channels = []
for head in keep_indices:
    keep_channels.extend(range(head * 128, (head + 1) * 128))

# 新的权重矩阵
new_weight = original_weight[keep_channels, :]  # [768, 4096]
k_proj.weight = nn.Parameter(new_weight)

# 更新 k_proj 的配置
k_proj.out_features = 768  # 从 1024 → 768
```

---

## 📊 完整示例：剪枝 Layer 5 的 25%

### 原始状态

```
Layer 5:
  q_proj: [4096, 4096]  → 32 Q heads × 128 = 4096
  k_proj: [4096, 1024]  → 8 KV heads × 128 = 1024
  v_proj: [4096, 1024]  → 8 KV heads × 128 = 1024
  o_proj: [4096, 4096]  → 输出投影

参数量: 16,777,216 + 4,194,304 + 4,194,304 + 16,777,216 = 41,943,040
```

### 剪枝过程

```
步骤1: 计算重要性
  k_proj 的 8 个 heads 重要性: [0.52, 0.89, 0.16, 0.73, 0.62, 0.45, 0.81, 0.68]

步骤2: 选择最不重要的 2 个 heads（25% = 2/8）
  剪枝 heads: [2, 5]

步骤3: 剪枝 k_proj
  k_proj: [4096, 1024] → [4096, 768]
  剪枝通道: 256 (2 heads)

步骤4: 自动传播
  v_proj: [4096, 1024] → [4096, 768]  (同步 KV)
  q_proj: [4096, 4096] → [4096, 3072] (GQA 4:1)
  o_proj: [4096, 4096] → [3072, 4096] (输入维度匹配)
```

### 剪枝后状态

```
Layer 5:
  q_proj: [4096, 3072]  → 24 Q heads × 128 = 3072
  k_proj: [4096, 768]   → 6 KV heads × 128 = 768
  v_proj: [4096, 768]   → 6 KV heads × 128 = 768
  o_proj: [3072, 4096]  → 输出投影

参数量: 12,582,912 + 3,145,728 + 3,145,728 + 12,582,912 = 31,457,280

减少: 41,943,040 - 31,457,280 = 10,485,760 (25%)
```

---

## 🎯 关键要点总结

### 1. 单一剪枝率

```python
"ch_sparsity": 0.25  # 所有层都用这个
```

- ✅ **简单**: 只需要一个参数
- ✅ **可预测**: 所有层行为一致
- ❌ **不灵活**: 无法根据层重要性调整

### 2. Head-level 剪枝

```python
"consecutive_groups": {
    layer.self_attn.k_proj: 128  # 强制 128 的倍数
}
```

- ✅ **保持完整性**: 每个 head 是独立的 attention 单元
- ✅ **GQA 友好**: 天然支持 4:1 比例
- ❌ **粒度限制**: 只能按 12.5% 的倍数剪枝

### 3. Taylor 重要性

```python
importance = |∂L/∂W × W|
```

- ✅ **准确**: 考虑梯度和权重
- ✅ **一阶近似**: 计算效率高
- ❌ **需要梯度**: 必须执行反向传播

### 4. 依赖图传播

```
k_proj 剪枝 → 自动传播到 q_proj, v_proj, o_proj
```

- ✅ **自动化**: 不需要手动指定每个模块
- ✅ **保证一致性**: 维度自动匹配
- ✅ **支持 GQA**: 自动处理 Q/KV 比例

---

## 🆚 与非均衡剪枝的对比

| 特性 | 均匀剪枝 | 非均衡剪枝 |
|------|---------|-----------|
| **剪枝率控制** | `ch_sparsity: 0.25` | `ch_sparsity_dict: {...}` |
| **每层剪枝率** | 全部相同 | 每层不同（基于重要性） |
| **层重要性** | 不评估 | 评估并使用 |
| **实现复杂度** | 简单 | 复杂 |
| **性能/剪枝率** | 一般 | 更好 |

---

## 🚀 运行示例

```bash
# 均匀剪枝：所有层 25%
python llama3.py \
    --base_model /path/to/Llama-3-8B-Instruct \
    --pruning_ratio 0.25 \
    --pruner_type taylor \
    --block_attention_layer_start 3 \
    --block_attention_layer_end 30 \
    --block_mlp_layer_start 3 \
    --block_mlp_layer_end 30
```

**结果：**
- Layer 0-2: 不剪枝
- Layer 3-29: 每层都剪枝 25%
- Layer 30-31: 不剪枝
- 实际总剪枝率: ~17-18%（因为保护了 5 层）
