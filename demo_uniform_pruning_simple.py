#!/usr/bin/env python3
"""
演示均匀剪枝的工作流程（无外部依赖）
模拟 MetaPruner 如何根据单一的 ch_sparsity 剪枝 Attention
"""


def demonstrate_layer_pruning():
    """
    演示 Layer 5 的剪枝过程
    """
    print("="*80)
    print("均匀剪枝演示：Layer 5")
    print("="*80)
    print()

    # 配置
    ch_sparsity = 0.25
    hidden_size = 4096
    num_q_heads = 32
    num_kv_heads = 8
    head_dim = 128

    q_out = num_q_heads * head_dim  # 4096
    kv_out = num_kv_heads * head_dim  # 1024

    print("配置:")
    print(f"  ch_sparsity (全局剪枝率): {ch_sparsity*100:.0f}%")
    print(f"  hidden_size: {hidden_size}")
    print(f"  num_q_heads: {num_q_heads}, num_kv_heads: {num_kv_heads}")
    print(f"  head_dim: {head_dim}")
    print()

    # 步骤1: 原始状态
    print("步骤1: 原始状态")
    print("-" * 80)
    print(f"  q_proj: [{hidden_size}, {q_out}]  → {num_q_heads} Q heads × {head_dim}")
    print(f"  k_proj: [{hidden_size}, {kv_out}]  → {num_kv_heads} KV heads × {head_dim}")
    print(f"  v_proj: [{hidden_size}, {kv_out}]  → {num_kv_heads} KV heads × {head_dim}")
    print(f"  o_proj: [{q_out}, {hidden_size}]")
    print()

    # 步骤2: 计算 Taylor 重要性
    print("步骤2: 计算 Taylor 重要性")
    print("-" * 80)
    print("  对于 k_proj 的 8 个 KV heads，计算每个 head 的重要性:")
    print()

    # 模拟的重要性值
    kv_importance = [0.523, 0.891, 0.156, 0.734, 0.621, 0.445, 0.812, 0.678]

    print("  KV Head | Taylor 重要性 | 说明")
    print("  " + "-" * 50)
    for i, imp in enumerate(kv_importance):
        note = ""
        if imp < 0.3:
            note = "← 最不重要"
        elif imp < 0.5:
            note = "← 第二不重要"
        print(f"  Head {i}  |    {imp:.4f}     | {note}")
    print()

    print("  公式: Taylor 重要性 = |∂L/∂W × W|")
    print("  其中:")
    print("    - ∂L/∂W: 损失对权重的梯度（通过反向传播计算）")
    print("    - W: 权重本身")
    print("    - 对每个 head 的 128 个通道求和")
    print()

    # 步骤3: 选择要剪枝的 heads
    print(f"步骤3: 选择要剪枝的 heads (ch_sparsity={ch_sparsity})")
    print("-" * 80)

    num_to_prune = int(num_kv_heads * ch_sparsity)
    print(f"  需要剪枝: {num_kv_heads} × {ch_sparsity} = {num_to_prune} 个 KV heads")
    print()

    # 按重要性排序
    sorted_heads = sorted(enumerate(kv_importance), key=lambda x: x[1])
    heads_to_prune = [h[0] for h in sorted_heads[:num_to_prune]]

    print("  按重要性排序:")
    for rank, (head_idx, imp) in enumerate(sorted_heads):
        marker = "✂️ 剪枝" if head_idx in heads_to_prune else ""
        print(f"    {rank+1}. Head {head_idx}: {imp:.4f} {marker}")
    print()

    print(f"  选择最不重要的 {num_to_prune} 个: {heads_to_prune}")
    print()

    # 步骤4: GQA 比例传播
    print("步骤4: GQA 比例传播 (4:1)")
    print("-" * 80)

    gqa_ratio = num_q_heads // num_kv_heads
    print(f"  GQA 比例: {num_q_heads}:{num_kv_heads} = {gqa_ratio}:1")
    print(f"  剪枝 {num_to_prune} 个 KV heads → 剪枝 {num_to_prune * gqa_ratio} 个 Q heads")
    print()

    # 对应的 Q heads
    q_heads_to_prune = []
    for kv_head in heads_to_prune:
        for i in range(gqa_ratio):
            q_heads_to_prune.append(kv_head * gqa_ratio + i)

    print(f"  KV heads {heads_to_prune} 对应的 Q heads:")
    for kv_head in heads_to_prune:
        q_heads = [kv_head * gqa_ratio + i for i in range(gqa_ratio)]
        print(f"    KV Head {kv_head} → Q Heads {q_heads}")
    print()

    # 步骤5: 物理剪枝
    print("步骤5: 物理执行剪枝")
    print("-" * 80)

    pruned_kv_heads = num_kv_heads - num_to_prune
    pruned_q_heads = num_q_heads - len(q_heads_to_prune)
    pruned_kv_out = pruned_kv_heads * head_dim
    pruned_q_out = pruned_q_heads * head_dim

    print("  k_proj:")
    print(f"    原始维度: [{hidden_size}, {kv_out}]")
    print(f"    删除 heads: {heads_to_prune}")
    print(f"    删除通道: ", end="")
    for h in heads_to_prune:
        print(f"[{h*128}:{(h+1)*128}] ", end="")
    print()
    print(f"    剪枝后维度: [{hidden_size}, {pruned_kv_out}]")
    print()

    print("  v_proj:")
    print(f"    同步剪枝相同的 heads: {heads_to_prune}")
    print(f"    剪枝后维度: [{hidden_size}, {pruned_kv_out}]")
    print()

    print("  q_proj:")
    print(f"    原始维度: [{hidden_size}, {q_out}]")
    print(f"    删除 Q heads: {q_heads_to_prune}")
    print(f"    剪枝后维度: [{hidden_size}, {pruned_q_out}]")
    print()

    print("  o_proj:")
    print(f"    原始维度: [{q_out}, {hidden_size}]")
    print(f"    输入维度匹配 q_proj 输出: [{pruned_q_out}, {hidden_size}]")
    print()

    # 步骤6: 参数统计
    print("步骤6: 参数量统计")
    print("-" * 80)

    modules = {
        'q_proj': {
            'orig': [hidden_size, q_out],
            'pruned': [hidden_size, pruned_q_out]
        },
        'k_proj': {
            'orig': [hidden_size, kv_out],
            'pruned': [hidden_size, pruned_kv_out]
        },
        'v_proj': {
            'orig': [hidden_size, kv_out],
            'pruned': [hidden_size, pruned_kv_out]
        },
        'o_proj': {
            'orig': [q_out, hidden_size],
            'pruned': [pruned_q_out, hidden_size]
        }
    }

    print(f"  {'模块':<10} {'原始参数量':<15} {'剪枝后参数量':<15} {'减少量':<15} {'减少率'}")
    print("  " + "-" * 75)

    total_orig = 0
    total_pruned = 0

    for name, dims in modules.items():
        orig_params = dims['orig'][0] * dims['orig'][1]
        pruned_params = dims['pruned'][0] * dims['pruned'][1]
        reduction = orig_params - pruned_params
        ratio = reduction / orig_params * 100

        total_orig += orig_params
        total_pruned += pruned_params

        print(f"  {name:<10} {orig_params:>14,} {pruned_params:>14,} {reduction:>14,} {ratio:>6.2f}%")

    total_reduction = total_orig - total_pruned
    total_ratio = total_reduction / total_orig * 100

    print("  " + "-" * 75)
    print(f"  {'总计':<10} {total_orig:>14,} {total_pruned:>14,} {total_reduction:>14,} {total_ratio:>6.2f}%")
    print()


def compare_uniform_vs_unbalanced():
    """
    对比均匀剪枝和非均衡剪枝
    """
    print("\n" + "="*80)
    print("均匀剪枝 vs 非均衡剪枝对比")
    print("="*80)
    print()

    print("假设目标剪枝率为 25%，有 32 层")
    print()

    print("┌" + "─"*78 + "┐")
    print("│ 均匀剪枝 (llama3.py)                                                     │")
    print("├" + "─"*78 + "┤")
    print("│                                                                          │")
    print("│  配置:                                                                   │")
    print("│    ch_sparsity: 0.25        ← 所有层使用这个参数                        │")
    print("│    root_instances: Layer 3-29 (27 层)                                   │")
    print("│                                                                          │")
    print("│  剪枝率:                                                                 │")
    print("│    Layer 0:  0.00% (保护)                                               │")
    print("│    Layer 1:  0.00% (保护)                                               │")
    print("│    Layer 2:  0.00% (保护)                                               │")
    print("│    Layer 3:  25.00% ← 所有层都相同                                      │")
    print("│    Layer 4:  25.00%                                                     │")
    print("│    ...                                                                   │")
    print("│    Layer 29: 25.00%                                                     │")
    print("│    Layer 30: 0.00% (保护)                                               │")
    print("│    Layer 31: 0.00% (保护)                                               │")
    print("│                                                                          │")
    print("│  特点:                                                                   │")
    print("│    ✅ 简单、可预测                                                       │")
    print("│    ❌ 不考虑层重要性差异                                                 │")
    print("│    ❌ 可能过度剪枝重要层                                                 │")
    print("│                                                                          │")
    print("└" + "─"*78 + "┘")
    print()

    print("┌" + "─"*78 + "┐")
    print("│ 非均衡剪枝 (llama3_unbalanced_pruning.py)                               │")
    print("├" + "─"*78 + "┤")
    print("│                                                                          │")
    print("│  配置:                                                                   │")
    print("│    ch_sparsity_dict: {...}  ← 每层有自己的剪枝率                        │")
    print("│    基于层重要性评估                                                     │")
    print("│                                                                          │")
    print("│  剪枝率:                                                                 │")
    print("│    Layer 0:  0.00%   (重要性: 5291.99, 最重要)                          │")
    print("│    Layer 1:  9.67%   (重要性: 614.10, < 15% 自动过滤)                   │")
    print("│    Layer 2:  26.12%  (重要性: 2.43)                                     │")
    print("│    Layer 3:  26.51%  (重要性: 1.76)                                     │")
    print("│    ...                                                                   │")
    print("│    Layer 11: 27.50%  (重要性: 0.22, 最不重要)  ← 最高剪枝率             │")
    print("│    ...                                                                   │")
    print("│    Layer 29: 24.96%  (重要性: 4.97)                                     │")
    print("│    Layer 30: 24.76%  (重要性: 5.52)                                     │")
    print("│    Layer 31: 20.39%  (重要性: 31.60, 较重要)   ← 保守剪枝               │")
    print("│                                                                          │")
    print("│  特点:                                                                   │")
    print("│    ✅ 根据层重要性自适应调整                                             │")
    print("│    ✅ 保护重要层，激进剪枝不重要层                                       │")
    print("│    ✅ 在相同剪枝率下获得更好的性能                                       │")
    print("│    ❌ 需要额外的层重要性评估时间                                         │")
    print("│                                                                          │")
    print("└" + "─"*78 + "┘")
    print()

    print("关键区别:")
    print("-" * 80)
    print("  1. 剪枝率分配:")
    print("     - 均匀: 所有层 25%")
    print("     - 非均衡: 20.39% ~ 27.50% (根据重要性)")
    print()
    print("  2. 层保护:")
    print("     - 均匀: 手动指定范围 (3-29)")
    print("     - 非均衡: 自动过滤低剪枝率层 (< 15%)")
    print()
    print("  3. 性能:")
    print("     - 均匀: PPL 可能较高")
    print("     - 非均衡: PPL 更低（保护了重要层）")
    print()


def main():
    demonstrate_layer_pruning()
    compare_uniform_vs_unbalanced()

    print("="*80)
    print("总结")
    print("="*80)
    print()
    print("均匀剪枝的核心:")
    print("  1. 使用单一的 ch_sparsity 参数控制所有层的剪枝率")
    print("  2. MetaPruner 对每个 root_instance 应用相同的剪枝率")
    print("  3. Taylor 重要性用于选择每层内部哪些 heads 要剪枝")
    print("  4. 但剪枝的 head 数量在所有层都相同")
    print()
    print("Attention 剪枝流程:")
    print("  k_proj 剪枝 → v_proj 同步 → q_proj 按 GQA 比例 → o_proj 匹配")
    print()


if __name__ == '__main__':
    main()
