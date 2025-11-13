# GQA比例问题的根本解决方案

## 问题分析

经过详细分析torch_pruning的源码，我发现了问题的根本原因：

### torch_pruning的依赖图传播机制
```python
# dependency.py:620-634
for out_node in node.outputs:
    trigger = self.get_pruner_of_module(node.module).prune_out_channels
    handler = self.get_pruner_of_module(out_node.module).prune_in_channels
    dep = Dependency(trigger=trigger, handler=handler, source=node, target=out_node)
```

依赖图传播是**通道对通道**的：
- k_proj剪掉通道[0-127] → q_proj也剪掉通道[0-127]
- 这导致剪掉1个KV head → 也只剪掉1个Q head（而不是4个）

### _ExpandIndexMapping的局限性

虽然torch_pruning有`_ExpandIndexMapping`来处理GQA的repeat：
```python
# _helpers.py:58-66
new_idxs = [
    i * self.repeat + j  # 对于每个KV head，生成4个Q head索引
    for i in idxs
    for j in range(self.repeat)
]
```

但它需要满足两个条件：
1. 依赖图必须检测到expand操作（`node.grad_fn._saved_self_sym_sizes`有5维）
2. Llama-3的forward实现必须使用标准的expand操作

**实际情况**：Llama-3可能使用了其他方式（如reshape+transpose）来实现KV repeat，导致torch_pruning检测不到。

### GQA感知模式失败的原因

我尝试的方案（将q/k/v都设置为root）失败了，因为：
1. 即使q/k/v都是root instances，依赖图仍然在它们之间建立连接
2. 当处理k_proj时，它的剪枝会通过依赖图传播到q_proj，覆盖q_proj的独立剪枝
3. consecutive_groups虽然强制head-level剪枝，但不能改变传播的通道索引

## 解决方案

### 方案1：后处理强制修正（推荐）

**原理**：不依赖torch_pruning的自动GQA处理，在剪枝后手动调整Q heads数量

**步骤**：
1. 使用原始方式剪枝（只有k_proj作为root）
2. 剪枝完成后，检查每层的Q:KV比例
3. 如果比例不正确：
   - 根据KV head数量计算应该保留的Q head数量：`target_q = kv_heads * 4`
   - 如果当前Q heads > target_q：截断q_proj权重到target_q个heads
   - 如果当前Q heads < target_q：这种情况不应该发生（KV剪得更多）

**实现**：
```python
# 剪枝后的后处理
for layer in model.model.layers:
    q_channels = layer.self_attn.q_proj.weight.shape[0]
    kv_channels = layer.self_attn.k_proj.weight.shape[0]

    num_q_heads = q_channels // 128
    num_kv_heads = kv_channels // 128

    # 计算目标Q head数量（保持4:1）
    target_q_heads = num_kv_heads * 4

    if num_q_heads != target_q_heads:
        # 截断q_proj权重
        target_q_channels = target_q_heads * 128
        layer.self_attn.q_proj.weight.data = layer.self_attn.q_proj.weight.data[:target_q_channels, :]

        # 同时调整o_proj的输入维度
        layer.self_attn.o_proj.weight.data = layer.self_attn.o_proj.weight.data[:, :target_q_channels]
```

**优点**：
- 简单可靠，完全控制GQA比例
- 不依赖torch_pruning的expand检测
- 已经在当前代码中部分实现（llama3_unbalanced_pruning.py:454-479）

**缺点**：
- 可能丢弃一些本来重要的Q heads（如果它们在target_q之外）

### 方案2：自定义Attention Pruner

**原理**：创建一个理解GQA结构的自定义pruner

**实现**：
```python
class GQAAttentionPruner(BasePruningFunc):
    def prune_out_channels(self, layer, idxs):
        # idxs是要剪枝的KV head索引（基于k_proj）

        # 1. 剪枝k_proj和v_proj
        keep_kv_idxs = [i for i in range(k_out_channels) if i not in idxs]
        layer.self_attn.k_proj.weight.data = layer.self_attn.k_proj.weight.data[keep_kv_idxs, :]
        layer.self_attn.v_proj.weight.data = layer.self_attn.v_proj.weight.data[keep_kv_idxs, :]

        # 2. 计算对应的Q head索引（4:1映射）
        q_idxs = []
        for kv_idx in idxs:
            for i in range(4):
                q_idxs.append(kv_idx * 4 + i)

        # 3. 剪枝q_proj
        keep_q_idxs = [i for i in range(q_out_channels) if i not in q_idxs]
        layer.self_attn.q_proj.weight.data = layer.self_attn.q_proj.weight.data[keep_q_idxs, :]
        layer.self_attn.o_proj.weight.data = layer.self_attn.o_proj.weight.data[:, keep_q_idxs]
```

**优点**：
- 从源头解决问题，剪枝时就保证正确的比例
- 保留了基于重要性的剪枝逻辑

**缺点**：
- 需要修改torch_pruning的核心逻辑
- 实现复杂度高

### 方案3：使用HFAttentionPruner（已存在但未启用）

torch_pruning已经有一个`HFAttentionPruner`（hf_llama_pruner.py:39-69），但在llama3.py中被注释掉了：

```python
# llama3.py:200
"customized_pruners": {
    LlamaRMSNorm: llama_pruner.hf_rmsnorm_pruner,
    #LlamaAttention: llama_pruner.hf_attention_pruner,  # 被注释掉
},
```

这个pruner会同时处理q/k/v/o_proj，但它也是按相同的idxs剪枝，不理解GQA的4:1结构。

## 推荐方案

**使用方案1（后处理修正）**，原因：
1. 已经部分实现，只需要完善
2. 简单可靠，不依赖torch_pruning的复杂机制
3. 完全控制最终的GQA比例

**改进当前的后处理逻辑**：
```python
# 当前：只在比例不正确时调整
if num_heads % num_kv_heads != 0:
    adjusted_num_heads = (num_heads // num_kv_heads) * num_kv_heads

# 改进：总是强制执行4:1比例
target_q_heads = num_kv_heads * 4
if num_q_heads != target_q_heads:
    logger.log(f"Layer {layer_idx}: 调整 Q heads {num_q_heads} → {target_q_heads}")
    # 截断到目标数量
    ...
```

这样即使torch_pruning剪出31:7，我们也会强制调整为28:7（4:1）。

## 测试建议

1. 先用原始llama3.py测试均匀剪枝，验证它是否也有同样的GQA比例问题
2. 如果原始llama3.py也有问题，说明这是torch_pruning对Llama-3的已知限制
3. 使用改进的后处理方案，确保所有层都严格保持4:1比例
