#!/usr/bin/env python3
"""
分析 Attention 模块中不同维度的剪枝率
解释为什么会看到 0.0625 和 0.25 两种比例
"""

def analyze_single_layer_pruning(target_rate=0.25):
    """
    分析单层剪枝时各个模块的维度变化
    """
    print("="*80)
    print("单层 Attention 剪枝分析")
    print("="*80)
    print()

    # 原始维度
    hidden_size = 4096
    num_q_heads = 32
    num_kv_heads = 8
    head_dim = 128

    q_out = num_q_heads * head_dim  # 4096
    kv_out = num_kv_heads * head_dim  # 1024

    print(f"原始配置:")
    print(f"  hidden_size = {hidden_size}")
    print(f"  num_q_heads = {num_q_heads}")
    print(f"  num_kv_heads = {num_kv_heads}")
    print(f"  head_dim = {head_dim}")
    print(f"  GQA 比例 = {num_q_heads // num_kv_heads}:1")
    print()

    # 剪枝 k_proj
    pruned_kv_heads = int(num_kv_heads * (1 - target_rate))
    pruned_q_heads = int(num_q_heads * (1 - target_rate))

    kv_out_pruned = pruned_kv_heads * head_dim
    q_out_pruned = pruned_q_heads * head_dim

    print(f"剪枝 {target_rate*100:.0f}% 后的配置:")
    print(f"  pruned_kv_heads = {pruned_kv_heads} (剪枝 {num_kv_heads - pruned_kv_heads} 个)")
    print(f"  pruned_q_heads = {pruned_q_heads} (剪枝 {num_q_heads - pruned_q_heads} 个)")
    print()

    # 各个模块的维度
    modules = {
        "q_proj": {
            "original": [hidden_size, q_out],
            "pruned": [hidden_size, q_out_pruned]
        },
        "k_proj": {
            "original": [hidden_size, kv_out],
            "pruned": [hidden_size, kv_out_pruned]
        },
        "v_proj": {
            "original": [hidden_size, kv_out],
            "pruned": [hidden_size, kv_out_pruned]
        },
        "o_proj": {
            "original": [q_out, hidden_size],
            "pruned": [q_out_pruned, hidden_size]
        }
    }

    print("各模块维度变化:")
    print("-" * 80)
    print(f"{'模块':<10} {'原始维度':<20} {'剪枝后维度':<20} {'输入维度变化':<15} {'输出维度变化'}")
    print("-" * 80)

    for name, dims in modules.items():
        orig_in, orig_out = dims["original"]
        prun_in, prun_out = dims["pruned"]

        in_change = orig_in - prun_in
        out_change = orig_out - prun_out

        in_ratio = in_change / orig_in if in_change != 0 else 0
        out_ratio = out_change / orig_out if out_change != 0 else 0

        print(f"{name:<10} [{orig_in:>4},{orig_out:>4}] → [{prun_in:>4},{prun_out:>4}]   "
              f"{in_ratio:>6.2%} ({in_change:>4})   {out_ratio:>6.2%} ({out_change:>4})")

    print()

    # 参数量统计
    print("参数量统计:")
    print("-" * 80)
    total_orig = 0
    total_pruned = 0

    for name, dims in modules.items():
        orig_params = dims["original"][0] * dims["original"][1]
        prun_params = dims["pruned"][0] * dims["pruned"][1]
        reduction = orig_params - prun_params
        ratio = reduction / orig_params

        total_orig += orig_params
        total_pruned += prun_params

        print(f"{name:<10} {orig_params:>12,} → {prun_params:>12,}   减少 {reduction:>12,} ({ratio:>6.2%})")

    print("-" * 80)
    total_reduction = total_orig - total_pruned
    total_ratio = total_reduction / total_orig
    print(f"{'总计':<10} {total_orig:>12,} → {total_pruned:>12,}   减少 {total_reduction:>12,} ({total_ratio:>6.2%})")
    print()


def analyze_cross_layer_pruning(target_rate=0.25):
    """
    分析跨层剪枝时的维度变化
    """
    print("="*80)
    print("跨层剪枝分析（Layer N 影响 Layer N+1）")
    print("="*80)
    print()

    hidden_size = 4096
    num_q_heads = 32
    num_kv_heads = 8
    head_dim = 128

    # Layer N 剪枝后
    pruned_q_heads = int(num_q_heads * (1 - target_rate))
    q_out_pruned = pruned_q_heads * head_dim  # 3072

    print(f"Layer N 剪枝 {target_rate*100:.0f}% 后:")
    print(f"  o_proj 输出维度: {hidden_size} → {q_out_pruned}")
    print()

    print(f"Layer N+1 的维度变化:")
    print("-" * 80)

    # Layer N+1 的输入维度变化
    # 如果 Layer N 没有被剪枝
    print("情况1: Layer N+1 本身不被剪枝")
    print(f"  q_proj: [{hidden_size}, {hidden_size*num_q_heads//num_q_heads}] → [{q_out_pruned}, {hidden_size}]")
    print(f"  k_proj: [{hidden_size}, {hidden_size*num_kv_heads//num_q_heads}] → [{q_out_pruned}, {hidden_size*num_kv_heads//num_q_heads}]")
    print(f"  输入维度剪枝率: {(hidden_size - q_out_pruned)/hidden_size:.4f} = {(hidden_size - q_out_pruned)/hidden_size}")
    print()

    # 如果 Layer N+1 也被剪枝
    print("情况2: Layer N+1 本身也被剪枝 {:.0f}%".format(target_rate*100))
    kv_out_pruned_n1 = int(num_kv_heads * (1 - target_rate)) * head_dim
    q_out_pruned_n1 = int(num_q_heads * (1 - target_rate)) * head_dim

    print(f"  q_proj: [{q_out_pruned}, {q_out_pruned_n1}]")
    print(f"  k_proj: [{q_out_pruned}, {kv_out_pruned_n1}]")
    print(f"  输入维度剪枝率: {(hidden_size - q_out_pruned)/hidden_size:.4f}")
    print(f"  输出维度剪枝率: {target_rate:.4f}")
    print()


def find_ratio_sources():
    """
    寻找 0.0625 和 0.25 的来源
    """
    print("="*80)
    print("剪枝比例来源分析")
    print("="*80)
    print()

    hidden_size = 4096
    num_q_heads = 32
    num_kv_heads = 8
    head_dim = 128

    target_rate = 0.25

    # 计算剪枝后的维度
    pruned_kv_heads = int(num_kv_heads * (1 - target_rate))
    pruned_q_heads = int(num_q_heads * (1 - target_rate))

    kv_out_pruned = pruned_kv_heads * head_dim  # 768
    q_out_pruned = pruned_q_heads * head_dim    # 3072

    print("可能的剪枝比例:")
    print("-" * 80)

    # 各种可能的比例
    ratios = [
        ("k_proj 输出维度剪枝", (1024 - kv_out_pruned) / 1024),
        ("q_proj 输出维度剪枝", (4096 - q_out_pruned) / 4096),
        ("下一层输入维度剪枝 (from 4096)", (4096 - q_out_pruned) / 4096),
        ("k_proj 输出通道数变化", (1024 - kv_out_pruned) / 4096),
        ("GQA 比例变化", (num_kv_heads - pruned_kv_heads) / num_q_heads),
        ("单个 KV head 占总 Q 维度", 128 / 4096),
        ("2个 KV head 占总 Q 维度", 256 / 4096),
    ]

    for name, ratio in ratios:
        print(f"{name:<40} = {ratio:.6f} = {ratio:.4f} = {ratio*100:.2f}%")
        if abs(ratio - 0.0625) < 0.001:
            print(f"  ★★★ 这可能是 0.0625 的来源！")
        if abs(ratio - 0.25) < 0.001:
            print(f"  ★★★ 这可能是 0.25 的来源！")

    print()
    print("结论:")
    print(f"  0.25 (25%)   = k_proj/q_proj 的输出维度剪枝率")
    print(f"  0.0625 (6.25%) = 单个 KV head (256 通道) 占整个 hidden_size (4096) 的比例")
    print(f"                 = 2 个 KV heads / 总 Q 维度")
    print()


if __name__ == "__main__":
    analyze_single_layer_pruning(target_rate=0.25)
    print("\n" * 2)
    analyze_cross_layer_pruning(target_rate=0.25)
    print("\n" * 2)
    find_ratio_sources()
