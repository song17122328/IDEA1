#!/usr/bin/env python3
"""
详细分析非均衡结构化剪枝过程（无需加载模型）
展示步骤3（创建模块级剪枝率字典）和步骤4（结构化剪枝）的细节
"""

import json
from typing import Dict


def load_pruning_config(config_path: str) -> Dict:
    """加载剪枝配置"""
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config['layer_pruning_rates']


def analyze_layer_filtering(layer_pruning_rates: Dict[str, float],
                            block_attention_start: int = 3,
                            block_attention_end: int = 30,
                            block_mlp_start: int = 3,
                            block_mlp_end: int = 30):
    """
    分析步骤3：层过滤逻辑

    解释为什么虽然计算了所有层（0-31）的剪枝率，
    但实际只剪枝3-29层
    """
    print("=" * 80)
    print("步骤3详解：层过滤和模块级剪枝率字典创建")
    print("=" * 80)
    print()

    # 转换key为int
    layer_pruning_rates = {int(k): v for k, v in layer_pruning_rates.items()}

    # 1. 显示所有层的计算结果
    print("1️⃣  所有层的剪枝率（步骤2计算结果）:")
    print("-" * 80)
    for layer_idx in sorted(layer_pruning_rates.keys()):
        rate = layer_pruning_rates[layer_idx]
        print(f"   Layer {layer_idx:2d}: {rate:.4f}")
    print()

    # 2. 显示过滤逻辑
    print("2️⃣  层过滤逻辑（llama3_unbalanced_pruning.py:222-228行）:")
    print("-" * 80)
    print(f"   参数配置:")
    print(f"     --block_attention_layer_start = {block_attention_start}")
    print(f"     --block_attention_layer_end   = {block_attention_end}")
    print(f"     --block_mlp_layer_start       = {block_mlp_start}")
    print(f"     --block_mlp_layer_end         = {block_mlp_end}")
    print()
    print(f"   Python range() 规则:")
    print(f"     range({block_attention_start}, {block_attention_end}) = [{block_attention_start}, {block_attention_start+1}, ..., {block_attention_end-1}]  (不包含{block_attention_end})")
    print()
    print("   代码:")
    print("     pruning_layers = set(range(args.block_attention_layer_start,")
    print("                                args.block_attention_layer_end)) | \\")
    print("                      set(range(args.block_mlp_layer_start,")
    print("                                args.block_mlp_layer_end))")
    print()

    # 3. 显示过滤结果
    pruning_layers = set(range(block_attention_start, block_attention_end)) | \
                     set(range(block_mlp_start, block_mlp_end))

    filtered_rates = {
        idx: rate for idx, rate in layer_pruning_rates.items()
        if idx in pruning_layers
    }

    print("3️⃣  过滤后实际参与剪枝的层:")
    print("-" * 80)
    print(f"   实际剪枝层集合: {sorted(pruning_layers)}")
    print(f"   共 {len(pruning_layers)} 层")
    print()

    # 4. 显示被保护的层
    all_layers = set(layer_pruning_rates.keys())
    protected_layers = all_layers - pruning_layers

    print("4️⃣  被保护（不剪枝）的层:")
    print("-" * 80)
    print(f"   不剪枝层集合: {sorted(protected_layers)}")
    print()
    print("   🛡️  保护原因:")
    early_layers = [i for i in protected_layers if i < block_attention_start]
    late_layers = [i for i in protected_layers if i >= block_attention_end]

    if early_layers:
        print(f"     - 前 {len(early_layers)} 层 {early_layers}: 底层特征提取层，对模型性能影响大")
    if late_layers:
        print(f"     - 后 {len(late_layers)} 层 {late_layers}: 高层语义理解层，对模型性能影响大")
    print()

    return filtered_rates, pruning_layers


def analyze_ch_sparsity_dict_creation(filtered_rates: Dict[int, float],
                                      prune_attention: bool = True,
                                      prune_mlp: bool = True):
    """
    分析 ch_sparsity_dict 的创建过程
    """
    print("=" * 80)
    print("步骤3详解：ch_sparsity_dict（模块级剪枝率字典）创建")
    print("=" * 80)
    print()

    print("5️⃣  ch_sparsity_dict 是什么？")
    print("-" * 80)
    print("   ch_sparsity_dict 是一个 Python 字典，将 PyTorch 模块对象映射到其剪枝率")
    print("   格式: {module_object: pruning_rate}")
    print()
    print("   作用: 告诉 MetaPruner 每个模块应该剪枝多少比例的通道数")
    print()

    print("6️⃣  为什么选择 k_proj 和 gate_proj 作为 root modules？")
    print("-" * 80)
    print("   Llama 模型架构:")
    print("     每层包含两个主要组件:")
    print("       1. Self-Attention: q_proj, k_proj, v_proj, o_proj")
    print("       2. MLP:           gate_proj, up_proj, down_proj")
    print()
    print("   结构化剪枝规则:")
    print("     - Attention: 剪枝 k_proj 的输出通道 → 自动传播到 q_proj, v_proj, o_proj")
    print("     - MLP:       剪枝 gate_proj 的输出通道 → 自动传播到 up_proj, down_proj")
    print()
    print("   这样只需设置 2 个 root module，就能剪枝整层的 7 个线性层！")
    print()

    print("7️⃣  ch_sparsity_dict 创建代码（layer_importance.py:312-327）:")
    print("-" * 80)
    print("   def create_ch_sparsity_dict_for_llama(model, layer_pruning_rates, ...):")
    print("       ch_sparsity_dict = {}")
    print()
    print("       for layer_idx, pruning_rate in layer_pruning_rates.items():")
    print("           layer = model.model.layers[layer_idx]")
    print()
    print("           # Attention 模块")
    print("           if prune_attention:")
    print("               ch_sparsity_dict[layer.self_attn.k_proj] = pruning_rate")
    print()
    print("           # MLP 模块")
    print("           if prune_mlp:")
    print("               ch_sparsity_dict[layer.mlp.gate_proj] = pruning_rate")
    print()
    print("       return ch_sparsity_dict")
    print()

    print("8️⃣  ch_sparsity_dict 内容示例:")
    print("-" * 80)

    layer_module_info = []
    for layer_idx, pruning_rate in sorted(filtered_rates.items()):
        modules_in_layer = []
        if prune_attention:
            modules_in_layer.append(('k_proj', pruning_rate))
        if prune_mlp:
            modules_in_layer.append(('gate_proj', pruning_rate))
        layer_module_info.append((layer_idx, modules_in_layer))

    # 显示前3层和后3层作为示例
    print("   示例（前3层）:")
    for layer_idx, modules in layer_module_info[:3]:
        print(f"\n   Layer {layer_idx} (剪枝率: {filtered_rates[layer_idx]:.4f}):")
        for mod_name, rate in modules:
            print(f"     - model.model.layers[{layer_idx}].self_attn.{mod_name}: {rate:.4f}" if 'proj' in mod_name and mod_name != 'gate_proj' else f"     - model.model.layers[{layer_idx}].mlp.{mod_name}: {rate:.4f}")

    print("\n   ...")

    print("\n   示例（后3层）:")
    for layer_idx, modules in layer_module_info[-3:]:
        print(f"\n   Layer {layer_idx} (剪枝率: {filtered_rates[layer_idx]:.4f}):")
        for mod_name, rate in modules:
            print(f"     - model.model.layers[{layer_idx}].self_attn.{mod_name}: {rate:.4f}" if 'proj' in mod_name and mod_name != 'gate_proj' else f"     - model.model.layers[{layer_idx}].mlp.{mod_name}: {rate:.4f}")

    num_modules = len(filtered_rates) * (int(prune_attention) + int(prune_mlp))
    print()
    print(f"   总计: {num_modules} 个模块设置了自定义剪枝率")
    print(f"   计算: {len(filtered_rates)} 层 × {int(prune_attention) + int(prune_mlp)} 模块/层 = {num_modules} 个模块")
    print()


def explain_metapruner_workflow():
    """
    解释步骤4：MetaPruner 的工作流程
    """
    print("=" * 80)
    print("步骤4详解：MetaPruner 结构化剪枝工作流程")
    print("=" * 80)
    print()

    print("9️⃣  什么是结构化剪枝？")
    print("-" * 80)
    print("   非结构化剪枝（稀疏剪枝）:")
    print("     - 将权重矩阵中的某些元素设为 0")
    print("     - 参数逻辑上减少，但物理内存不减少")
    print("     - 需要稀疏矩阵运算支持才能加速")
    print()
    print("   结构化剪枝（通道剪枝）:")
    print("     - 删除整个通道（神经元）")
    print("     - 物理上减小模型尺寸")
    print("     - 直接加速，无需特殊硬件支持")
    print()
    print("   示例:")
    print("     原始:    Linear(4096 → 4096)")
    print("              权重矩阵: [4096, 4096]")
    print("              参数量: 16,777,216")
    print()
    print("     剪枝 25%: Linear(4096 → 3072)  ✅ 真正减少参数和显存")
    print("              权重矩阵: [3072, 4096]")
    print("              参数量: 12,582,912")
    print("              减少: 4,194,304 (25%)")
    print()

    print("🔟  MetaPruner 是什么？")
    print("-" * 80)
    print("   MetaPruner 是 Torch-Pruning 库的核心剪枝器")
    print("   特点:")
    print("     1. 自动追踪模块之间的依赖关系")
    print("     2. 确保剪枝后模型结构一致性")
    print("     3. 支持各种重要性评估方法（Taylor、L1、L2等）")
    print()

    print("1️⃣1️⃣  MetaPruner 工作流程:")
    print("-" * 80)
    print()
    print("   第1步: 构建依赖图（Dependency Graph）")
    print("   ─────────────────────────────────────────")
    print("     输入: forward_prompts（示例输入张量）")
    print("     过程: 执行一次前向传播，记录所有模块的输入输出关系")
    print("     输出: 依赖图，记录哪些模块的输出连接到哪些模块的输入")
    print()
    print("     示例（Layer 3）:")
    print("       ")
    print("       ┌────────────────────────────────────────────┐")
    print("       │          Layer 3 Self-Attention            │")
    print("       └────────────────────────────────────────────┘")
    print("       ")
    print("       输入: hidden_states [batch, seq_len, 4096]")
    print("          │")
    print("          ├──→ q_proj: Linear(4096 → 4096)")
    print("          │")
    print("          ├──→ k_proj: Linear(4096 → 1024) ← root module")
    print("          │")
    print("          └──→ v_proj: Linear(4096 → 1024)")
    print("          ")
    print("          → Attention 计算")
    print("          ")
    print("          → o_proj: Linear(4096 → 4096)")
    print()

    print("   第2步: 计算通道重要性")
    print("   ─────────────────────────────────────────")
    print("     对于每个要剪枝的模块:")
    print("       - Taylor 重要性: importance = |∂L/∂W × W|")
    print("       - L1 重要性:     importance = |W|")
    print("       - L2 重要性:     importance = ||W||₂")
    print()
    print("     为每个输出通道计算重要性分数")
    print()
    print("     示例（Layer 5 的 k_proj）:")
    print("       k_proj 权重: [1024, 4096]  (1024个输出通道)")
    print()
    print("       计算每个输出通道的重要性:")
    print("         Channel 0:   importance = 0.523")
    print("         Channel 1:   importance = 0.891")
    print("         Channel 2:   importance = 0.156  ← 不重要")
    print("         ...")
    print("         Channel 1023: importance = 0.734")
    print()

    print("   第3步: 选择要剪枝的通道")
    print("   ─────────────────────────────────────────")
    print("     对于每个模块 (根据 ch_sparsity_dict):")
    print()
    print("       例如: Layer 5 的 k_proj, 剪枝率 = 0.2629")
    print()
    print("         原始通道数: 1024")
    print("         保留通道数: 1024 × (1 - 0.2629) ≈ 755")
    print("         剪枝通道数: 1024 - 755 = 269")
    print()
    print("         选择重要性最低的 269 个通道进行剪枝")
    print("         例如: [2, 15, 37, 89, ...]  (269个通道索引)")
    print()

    print("   第4步: 传播剪枝决策")
    print("   ─────────────────────────────────────────")
    print("     根据依赖图，自动传播剪枝:")
    print()
    print("       Layer 5 示例:")
    print()
    print("         k_proj 输出: 1024 → 755 通道")
    print("           ↓")
    print("         (GQA: q_proj 和 k_proj 的比例关系)")
    print("           ↓")
    print("         q_proj 输出: 4096 → 3020 通道  (4:1 比例)")
    print("         v_proj 输出: 1024 → 755 通道   (1:1 比例)")
    print("           ↓")
    print("         o_proj 输入: 4096 → 3020 通道  (必须匹配)")
    print()
    print("       这保证了 Attention 机制的维度一致性！")
    print()
    print("       同理，MLP 的传播:")
    print("         gate_proj 输出: 14336 → 10570 通道")
    print("         up_proj   输出: 14336 → 10570 通道")
    print("         down_proj 输入: 14336 → 10570 通道")
    print()

    print("   第5步: 物理执行剪枝")
    print("   ─────────────────────────────────────────")
    print("     对于每个被标记剪枝的通道:")
    print("       1. 从权重矩阵中删除对应的行/列")
    print("       2. 更新模块的 in_features 和 out_features")
    print("       3. 释放显存")
    print()
    print("     示例:")
    print("       原始: k_proj = Linear(in=4096, out=1024)")
    print("         weight.shape = [1024, 4096]")
    print("         参数量 = 4,194,304")
    print("         显存占用 (FP16) = 8.4 MB")
    print()
    print("       剪枝后: k_proj = Linear(in=4096, out=755)")
    print("         weight.shape = [755, 4096]")
    print("         参数量 = 3,092,480")
    print("         显存占用 (FP16) = 6.2 MB")
    print()
    print("         减少参数 = 1,101,824 (26.29%)")
    print("         减少显存 = 2.2 MB")
    print()

    print("1️⃣2️⃣  迭代式剪枝（Iterative Pruning）")
    print("-" * 80)
    print("   为什么需要多次迭代？")
    print("     - 一次性大幅剪枝会严重损害模型性能")
    print("     - 逐步剪枝允许模型在每步后重新计算重要性")
    print()
    print("   示例（iterative_steps=1，目标剪枝率25%）:")
    print("     迭代 1: 直接剪枝 25%")
    print()
    print("   示例（iterative_steps=3，目标剪枝率25%）:")
    print("     迭代 1: 剪枝 8.33%  → 重新计算重要性")
    print("     迭代 2: 剪枝 8.33%  → 重新计算重要性")
    print("     迭代 3: 剪枝 8.34%")
    print("     总计:   25%")
    print()
    print("   从日志看，您使用 iterative_steps=1（一次性剪枝）")
    print()

    print("1️⃣3️⃣  剪枝器配置参数（llama3_unbalanced_pruning.py:272-289）")
    print("-" * 80)
    print("   kwargs = {")
    print("       'importance': imp,              # Taylor/L1/L2 重要性")
    print("       'global_pruning': False,        # 不使用全局剪枝")
    print("       'iterative_steps': 1,           # 迭代次数")
    print("       'ch_sparsity': 0.25,            # 默认剪枝率（备用）")
    print("       'ch_sparsity_dict': {...},      # ⭐ 每个模块的剪枝率")
    print("       'ignored_layers': [],           # 忽略的层")
    print("       'consecutive_groups': {...},    # 连续分组约束（GQA）")
    print("       'root_instances': [             # 剪枝入口模块")
    print("           model.layers[3].self_attn.k_proj,")
    print("           model.layers[3].mlp.gate_proj,")
    print("           ...,")
    print("           model.layers[29].self_attn.k_proj,")
    print("           model.layers[29].mlp.gate_proj,")
    print("       ]")
    print("   }")
    print()
    print("   pruner = tp.pruner.MetaPruner(model, forward_prompts, **kwargs)")
    print("   pruner.step()  # 执行剪枝")
    print()


def explain_why_3_to_29():
    """
    总结：为什么只剪枝3-29层
    """
    print("=" * 80)
    print("🎯 核心问题：为什么只剪枝 3-29 层？")
    print("=" * 80)
    print()

    print("答案:")
    print("-" * 80)
    print()
    print("  1️⃣  步骤2 确实计算了所有层（0-31）的剪枝率")
    print("      基于层重要性评估，每层都有一个剪枝率")
    print()
    print("  2️⃣  步骤3 通过参数过滤，只保留 3-29 层")
    print("      代码 (line 222-228):")
    print("        pruning_layers = set(range(3, 30)) | set(range(3, 30))")
    print("        → 结果: [3, 4, 5, ..., 29]")
    print()
    print("  3️⃣  为什么要过滤？保护关键层！")
    print("      🛡️  Layer 0-2:  底层特征提取，不剪枝")
    print("      ✂️  Layer 3-29: 中间层，根据重要性剪枝")
    print("      🛡️  Layer 30-31: 高层语义，不剪枝")
    print()
    print("  4️⃣  这是一个经验性的设计选择")
    print("      - 借鉴了很多 LLM 剪枝论文的做法")
    print("      - 在剪枝率和性能之间取得平衡")
    print("      - 您可以通过修改参数来调整:")
    print("        --block_attention_layer_start 0  ← 从第0层开始")
    print("        --block_attention_layer_end 32    ← 到第31层结束")
    print()

    print("对比:")
    print("-" * 80)
    print()
    print("  场景A: 剪枝所有层（0-31）")
    print("    优点: 更高的参数减少率")
    print("    缺点: PPL 显著上升，性能下降明显")
    print()
    print("  场景B: 只剪枝 3-29 层（当前方案）")
    print("    优点: 保持较好性能，PPL 上升较小")
    print("    缺点: 参数减少率略低")
    print()
    print("  从您的日志看:")
    print("    实际剪枝率: 17.19% (目标 25%)")
    print("    → 因为保护了 5 层，实际剪枝的层数减少")
    print("    → 如果剪枝所有层，实际剪枝率会更接近 25%")
    print()


def main():
    import argparse
    parser = argparse.ArgumentParser(description='分析非均衡结构化剪枝的详细过程')
    parser.add_argument('--config', type=str,
                       default='prune_log/llama_unbalanced_prune/layer_importance_config.json',
                       help='层重要性配置文件路径')
    parser.add_argument('--block_attention_layer_start', type=int, default=3)
    parser.add_argument('--block_attention_layer_end', type=int, default=30)
    parser.add_argument('--block_mlp_layer_start', type=int, default=3)
    parser.add_argument('--block_mlp_layer_end', type=int, default=30)

    args = parser.parse_args()

    print()
    print("🔍 非均衡结构化剪枝详细分析")
    print("=" * 80)
    print()

    # 加载配置
    print("加载剪枝配置...")
    layer_pruning_rates = load_pruning_config(args.config)
    print(f"✅ 成功加载 {len(layer_pruning_rates)} 层的剪枝率配置")
    print()

    # 分析层过滤
    filtered_rates, pruning_layers = analyze_layer_filtering(
        layer_pruning_rates,
        args.block_attention_layer_start,
        args.block_attention_layer_end,
        args.block_mlp_layer_start,
        args.block_mlp_layer_end
    )

    # 分析 ch_sparsity_dict 创建
    analyze_ch_sparsity_dict_creation(
        filtered_rates,
        prune_attention=True,
        prune_mlp=True
    )

    # 解释 MetaPruner 工作流程
    explain_metapruner_workflow()

    # 回答核心问题
    explain_why_3_to_29()

    print("=" * 80)
    print("✅ 分析完成！")
    print("=" * 80)
    print()


if __name__ == '__main__':
    main()
