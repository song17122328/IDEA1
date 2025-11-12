# 均匀剪枝 vs 非均衡剪枝详细对比

## 📊 核心区别总结

| 特性 | 均匀剪枝 (llama3.py) | 非均衡剪枝 (llama3_unbalanced_pruning.py) |
|------|---------------------|-------------------------------------------|
| **剪枝率** | 所有层相同 | 每层不同（基于层重要性） |
| **层重要性评估** | ❌ 无 | ✅ 有（removal/activation方法） |
| **ch_sparsity_dict** | ❌ 无 | ✅ 有（每层自定义剪枝率） |
| **剪枝粒度** | Head-level (128通道) | Head-level (128通道) |
| **GQA约束** | ✅ 有 (consecutive_groups) | ✅ 有 (consecutive_groups + channel_groups) |
| **保护关键层** | ❌ 手动指定范围 | ✅ 自动过滤低重要性层 |
| **实际剪枝率** | 接近目标 | 可能低于目标（保护层） |

---

## 🔧 原来的均匀剪枝 (llama3.py)

### 关键代码

```python
# llama3.py:112-129
kwargs = {
    "importance": imp,                    # Taylor/L1/L2
    "global_pruning": args.global_pruning,
    "iterative_steps": args.iterative_steps,
    "ch_sparsity": args.pruning_ratio,   # ⭐ 全局剪枝率（所有层相同）
    "ignored_layers": [],
    "channel_groups": {},                 # 空字典
    "consecutive_groups": {
        layer.self_attn.k_proj: layer.self_attn.head_dim
        for layer in model.model.layers   # 所有层的 head_dim = 128
    },
    "customized_pruners": {
        LlamaRMSNorm: llama_pruner.hf_rmsnorm_pruner,
    },
    "root_module_types": None,
    "root_instances": [
        model.model.layers[i].self_attn.k_proj
        for i in range(args.block_attention_layer_start, args.block_attention_layer_end)
    ] + [
        model.model.layers[i].mlp.gate_proj
        for i in range(args.block_mlp_layer_start, args.block_mlp_layer_end)
    ]
}
```

### 工作原理

#### 1️⃣ 所有层使用相同的剪枝率

```python
"ch_sparsity": 0.25  # 所有层都剪枝 25%
```

**示例：** 如果设置 `--pruning_ratio 0.25`
- Layer 0: 剪枝 25%
- Layer 1: 剪枝 25%
- Layer 2: 剪枝 25%
- ...
- Layer 31: 剪枝 25%

**问题：**
- ❌ 不考虑层的重要性差异
- ❌ 可能过度剪枝重要层
- ❌ 可能欠剪枝不重要层

#### 2️⃣ Head-level 剪枝（通过 consecutive_groups）

```python
"consecutive_groups": {
    layer.self_attn.k_proj: 128  # 每个 head 有 128 个通道
}
```

**含义：**
- k_proj 的 1024 个输出通道被分为 8 个 head
- 每次必须剪枝完整的 head（128 个连续通道）
- 剪枝率必须是 128/1024 的倍数

**有效剪枝率：**
```
12.5%  → 剪枝 1 个 head (8 → 7)
25.0%  → 剪枝 2 个 head (8 → 6)
37.5%  → 剪枝 3 个 head (8 → 5)
50.0%  → 剪枝 4 个 head (8 → 4)
```

#### 3️⃣ 选择要剪枝的层范围

```python
"root_instances": [
    # Attention: 层 3-29
    model.model.layers[i].self_attn.k_proj
    for i in range(3, 30)
] + [
    # MLP: 层 3-29
    model.model.layers[i].mlp.gate_proj
    for i in range(3, 30)
]
```

**保护的层：**
- Layer 0-2: 前 3 层不剪枝
- Layer 30-31: 后 2 层不剪枝

**原因：** 经验性选择，保护关键层

#### 4️⃣ 重要性评估（Taylor）

```python
# llama3.py:143-165
# 在每次迭代中，使用示例数据计算 Taylor 重要性
example_prompts = get_examples('bookcorpus', tokenizer, 10, seq_len=64)
loss = model(example_prompts, labels=example_prompts).loss
loss.backward()

# Taylor 重要性 = |∂L/∂W × W|
```

**用途：**
- 在每一层内部，选择哪些 head 要剪枝
- **不是** 用来决定每层的剪枝率（所有层都是 25%）

---

## 🚀 新的非均衡剪枝 (llama3_unbalanced_pruning.py)

### 关键代码

```python
# llama3_unbalanced_pruning.py:313-330
kwargs = {
    "importance": imp,
    "global_pruning": False,
    "iterative_steps": args.iterative_steps,
    "ch_sparsity": args.pruning_ratio,        # 默认剪枝率（备用）
    "ch_sparsity_dict": ch_sparsity_dict,     # ⭐ 每层的自定义剪枝率
    "ignored_layers": [],
    "channel_groups": {
        layer.self_attn.q_proj: 4             # ⭐ GQA 比例约束
        for layer in model.model.layers
    },
    "consecutive_groups": {
        layer.self_attn.k_proj: layer.self_attn.head_dim
        for layer in model.model.layers
    },
    "customized_pruners": {
        LlamaRMSNorm: llama_pruner.hf_rmsnorm_pruner,
    },
    "root_module_types": None,
    "root_instances": [...]  # 同上
}
```

### 工作原理

#### 1️⃣ 每层使用不同的剪枝率（基于层重要性）

```python
# 步骤1: 评估层重要性
layer_importance = {
    0: 5291.99,  # 非常重要
    1: 614.10,   # 很重要
    2: 2.43,
    ...
    31: 31.60
}

# 步骤2: 计算非均衡剪枝率
layer_pruning_rates = {
    0: 0.0000,   # 最重要 → 不剪枝
    1: 0.0967,   # 很重要 → 剪枝 9.67%
    2: 0.2612,   # 一般 → 剪枝 26.12%
    11: 0.2750,  # 不太重要 → 剪枝 27.5%
    31: 0.2039,  # 重要 → 剪枝 20.39%
}
```

**示例对比：**

| 层 | 重要性 | 剪枝率 | 说明 |
|---|--------|--------|------|
| Layer 0 | 5291.99 | 0.00% | 最重要，完全保护 |
| Layer 1 | 614.10 | 9.67% | 很重要，轻度剪枝 |
| Layer 11 | 0.22 | 27.50% | 不重要，重度剪枝 |
| Layer 31 | 31.60 | 20.39% | 较重要，中度剪枝 |

#### 2️⃣ ch_sparsity_dict：每层自定义剪枝率

```python
# layer_importance.py:312-327
ch_sparsity_dict = {}

for layer_idx, pruning_rate in layer_pruning_rates.items():
    layer = model.model.layers[layer_idx]

    # Attention: k_proj 作为 root module
    ch_sparsity_dict[layer.self_attn.k_proj] = pruning_rate

    # MLP: gate_proj 作为 root module
    ch_sparsity_dict[layer.mlp.gate_proj] = pruning_rate
```

**生成的字典内容：**
```python
{
    <Layer 2 的 k_proj>: 0.2612,
    <Layer 2 的 gate_proj>: 0.2612,
    <Layer 3 的 k_proj>: 0.2651,
    <Layer 3 的 gate_proj>: 0.2651,
    ...
}
```

这个字典告诉 MetaPruner：
- Layer 2 的 k_proj 剪枝 26.12%
- Layer 3 的 k_proj 剪枝 26.51%
- 每层都有自己的剪枝率！

#### 3️⃣ 增强的 GQA 约束（channel_groups）

```python
"channel_groups": {
    layer.self_attn.q_proj: 4  # q_heads : kv_heads = 4:1
    for layer in model.model.layers
}
```

**作用：**
- 确保 q_proj 和 k_proj 按 4:1 比例剪枝
- 例如：k_proj 剪枝 2 个 head → q_proj 剪枝 8 个 head

**对比：**
- 均匀剪枝：没有 channel_groups，依赖依赖图传播（可能不够精确）
- 非均衡剪枝：显式指定 channel_groups，确保 GQA 比例

#### 4️⃣ 自动过滤低剪枝率层

```python
# llama3_unbalanced_pruning.py:232-246
min_effective_rate = 0.15  # 最小有效剪枝率 15%

effective_pruning_rates = {
    idx: rate for idx, rate in filtered_pruning_rates.items()
    if rate >= min_effective_rate
}

# 自动跳过剪枝率 < 15% 的层
# 例如：Layer 0 (0%) 和 Layer 1 (9.67%) 被跳过
```

**原因：**
- k_proj 有 1024 通道 = 8 个 head
- 每个 head 有 128 通道
- 至少需要剪枝 1 个 head = 12.5%
- 设置为 15% 确保安全

---

## 📈 实际效果对比

### 场景1：均匀剪枝（llama3.py）

```bash
python llama3.py \
    --base_model /path/to/Llama-3-8B-Instruct \
    --pruning_ratio 0.25 \
    --block_attention_layer_start 3 \
    --block_attention_layer_end 30
```

**结果：**
- 剪枝 Layer 3-29（27 层）
- 每层都剪枝 25%
- 实际剪枝率：~17-18%（因为保护了 5 层）
- PPL：假设为 X

**问题：**
- Layer 11（不重要）只剪枝了 25%，可以剪更多
- Layer 31（重要）剪枝了 25%，可能太多

### 场景2：非均衡剪枝（llama3_unbalanced_pruning.py）

```bash
python llama3_unbalanced_pruning.py \
    --base_model /path/to/Llama-3-8B-Instruct \
    --pruning_ratio 0.25 \
    --block_attention_layer_start 0 \
    --block_attention_layer_end 32
```

**结果：**
- 自动过滤 Layer 0-1（剪枝率太低）
- 剪枝 Layer 2-31（30 层）
- 每层剪枝率不同：20.39% ~ 27.50%
- 实际剪枝率：~20%
- PPL：预期比均匀剪枝更低（因为保护了重要层）

**优势：**
- ✅ Layer 11（不重要）剪枝 27.5%（更激进）
- ✅ Layer 0（最重要）不剪枝（完全保护）
- ✅ Layer 31（重要）剪枝 20.39%（比 25% 更保守）
- ✅ 更好的性能/剪枝率权衡

---

## 🎯 如何选择？

### 使用均匀剪枝（llama3.py）当：
- ✅ 想要简单、可预测的剪枝率
- ✅ 对所有层一视同仁
- ✅ 快速实验，不需要层重要性评估
- ✅ 已知哪些层要保护（手动设置范围）

### 使用非均衡剪枝（llama3_unbalanced_pruning.py）当：
- ✅ 想要更好的性能/剪枝率权衡
- ✅ 愿意花时间评估层重要性
- ✅ 希望自动保护重要层
- ✅ 追求更低的 PPL
- ✅ 做学术研究，需要创新方法

---

## 📝 共同点

两种方法都：
1. ✅ 使用 **Head-level 剪枝**（consecutive_groups）
2. ✅ 保持 **GQA 约束**（q_heads : kv_heads = 4:1）
3. ✅ 使用 **Taylor 重要性** 选择每层内部哪些 head 要剪枝
4. ✅ 支持 **迭代式剪枝**（iterative_steps）
5. ✅ 使用相同的 **Torch-Pruning** 库和 **MetaPruner**

**核心区别只在于：** 每层的剪枝率是否相同！

---

## 🔬 实验建议

运行对比实验：

```bash
# 实验1：均匀剪枝
python llama3.py \
    --pruning_ratio 0.25 \
    --save_ckpt_log_name "llama_uniform_25"

# 实验2：非均衡剪枝
python llama3_unbalanced_pruning.py \
    --pruning_ratio 0.25 \
    --save_ckpt_log_name "llama_unbalanced_25"
```

对比指标：
- 实际剪枝率
- PPL (wikitext2)
- 模型大小
- 生成质量

预期：非均衡剪枝的 PPL 更低（性能更好）！
