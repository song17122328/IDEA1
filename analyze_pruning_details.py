#!/usr/bin/env python3
"""
详细分析非均衡结构化剪枝过程
展示步骤3（创建模块级剪枝率字典）和步骤4（结构化剪枝）的细节
"""

import torch
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Dict


def load_pruning_config(config_path: str) -> Dict:
    """加载剪枝配置"""
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config['layer_pruning_rates']


def analyze_layer_filtering(layer_pruning_rates: Dict[int, float],
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

    # 1. 显示所有层的计算结果
    print("1️⃣  所有层的剪枝率（步骤2计算结果）:")
    print("-" * 80)
    for layer_idx in sorted(layer_pruning_rates.keys()):
        rate = layer_pruning_rates[layer_idx]
        print(f"   Layer {layer_idx:2d}: {rate:.4f}")
    print()

    # 2. 显示过滤逻辑
    print("2️⃣  层过滤逻辑（代码第222-228行）:")
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


def analyze_ch_sparsity_dict_creation(model, filtered_rates: Dict[int, float],
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

    # 创建 ch_sparsity_dict
    ch_sparsity_dict = {}

    print("7️⃣  ch_sparsity_dict 内容（layer_importance.py:312-327）:")
    print("-" * 80)

    layer_module_info = []

    for layer_idx, pruning_rate in sorted(filtered_rates.items()):
        layer = model.model.layers[layer_idx]

        modules_in_layer = []

        # Attention
        if prune_attention and hasattr(layer, 'self_attn'):
            k_proj = layer.self_attn.k_proj
            ch_sparsity_dict[k_proj] = pruning_rate
            modules_in_layer.append(('k_proj', k_proj, pruning_rate))

        # MLP
        if prune_mlp and hasattr(layer, 'mlp'):
            gate_proj = layer.mlp.gate_proj
            ch_sparsity_dict[gate_proj] = pruning_rate
            modules_in_layer.append(('gate_proj', gate_proj, pruning_rate))

        layer_module_info.append((layer_idx, modules_in_layer))

    # 显示前3层和后3层作为示例
    print("   示例（前3层）:")
    for layer_idx, modules in layer_module_info[:3]:
        print(f"\n   Layer {layer_idx} (剪枝率: {filtered_rates[layer_idx]:.4f}):")
        for mod_name, mod_obj, rate in modules:
            print(f"     - {mod_name}: {type(mod_obj).__name__} → 剪枝率 {rate:.4f}")
            print(f"       内存地址: {hex(id(mod_obj))}")

    print("\n   ...")

    print("\n   示例（后3层）:")
    for layer_idx, modules in layer_module_info[-3:]:
        print(f"\n   Layer {layer_idx} (剪枝率: {filtered_rates[layer_idx]:.4f}):")
        for mod_name, mod_obj, rate in modules:
            print(f"     - {mod_name}: {type(mod_obj).__name__} → 剪枝率 {rate:.4f}")
            print(f"       内存地址: {hex(id(mod_obj))}")

    print()
    print(f"   总计: {len(ch_sparsity_dict)} 个模块设置了自定义剪枝率")
    print(f"   计算: {len(filtered_rates)} 层 × 2 模块/层 = {len(filtered_rates) * 2} 个模块")
    print()

    return ch_sparsity_dict


def explain_metapruner_workflow(ch_sparsity_dict):
    """
    解释步骤4：MetaPruner 的工作流程
    """
    print("=" * 80)
    print("步骤4详解：MetaPruner 结构化剪枝工作流程")
    print("=" * 80)
    print()

    print("8️⃣  什么是结构化剪枝？")
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
    print("     原始: Linear(4096 → 4096)")
    print("     剪枝 25%: Linear(4096 → 3072)  ✅ 真正减少参数和显存")
    print()

    print("9️⃣  MetaPruner 是什么？")
    print("-" * 80)
    print("   MetaPruner 是 Torch-Pruning 库的核心剪枝器")
    print("   特点:")
    print("     1. 自动追踪模块之间的依赖关系")
    print("     2. 确保剪枝后模型结构一致性")
    print("     3. 支持各种重要性评估方法（Taylor、L1、L2等）")
    print()

    print("🔟  MetaPruner 工作流程:")
    print("-" * 80)
    print()
    print("   第1步: 构建依赖图（Dependency Graph）")
    print("   ─────────────────────────────────────────")
    print("     输入: forward_prompts（示例输入张量）")
    print("     过程: 执行一次前向传播，记录所有模块的输入输出关系")
    print("     输出: 依赖图，记录哪些模块的输出连接到哪些模块的输入")
    print()
    print("     示例:")
    print("       Layer 3:")
    print("         k_proj (4096→1024) → RMSNorm → Attention")
    print("                 ↓")
    print("         q_proj (4096→4096) ──┘")
    print("         v_proj (4096→1024) ──┘")
    print("                 ↓")
    print("         o_proj (4096→4096)")
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

    print("   第3步: 选择要剪枝的通道")
    print("   ─────────────────────────────────────────")
    print("     对于每个模块 (根据 ch_sparsity_dict):")
    print()
    print("       例如: Layer 5 的 k_proj, 剪枝率 = 0.2629")
    print()
    print("         原始通道数: 1024")
    print("         保留通道数: 1024 × (1 - 0.2629) = 755")
    print("         剪枝通道数: 1024 - 755 = 269")
    print()
    print("         选择重要性最低的 269 个通道进行剪枝")
    print()

    print("   第4步: 传播剪枝决策")
    print("   ─────────────────────────────────────────")
    print("     根据依赖图，自动传播剪枝:")
    print()
    print("       Layer 5 示例:")
    print("         k_proj 输出: 1024 → 755 通道")
    print("           ↓ (依赖关系自动传播)")
    print("         q_proj 输出: 4096 → 3020 通道  (同比例)")
    print("         v_proj 输出: 1024 → 755 通道")
    print("         o_proj 输入: 4096 → 3020 通道")
    print()
    print("       这保证了 Attention 机制的一致性！")
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
    print()
    print("       剪枝后: k_proj = Linear(in=4096, out=755)")
    print("         weight.shape = [755, 4096]")
    print("         参数量 = 3,092,480")
    print("         减少 = 1,101,824 (26.29%)")
    print()

    print("1️⃣1️⃣  迭代式剪枝（Iterative Pruning）")
    print("-" * 80)
    print("   为什么需要多次迭代？")
    print("     - 一次性大幅剪枝会严重损害模型性能")
    print("     - 逐步剪枝允许模型在每步后适应")
    print()
    print("   示例（3次迭代，目标剪枝率25%）:")
    print("     迭代 1: 剪枝 8.33%")
    print("     迭代 2: 剪枝 8.33%")
    print("     迭代 3: 剪枝 8.34%")
    print("     总计:   25%")
    print()

    print("1️⃣2️⃣  为什么日志显示 3-29 层？")
    print("-" * 80)
    print("   原因总结:")
    print("     ✅ 步骤2 计算了所有层（0-31）的剪枝率")
    print("     ✅ 步骤3 根据参数过滤，只保留 3-29 层")
    print("     ✅ 步骤4 只对过滤后的层执行实际剪枝")
    print()
    print("   好处:")
    print("     🛡️  保护关键层（前3层和后2层）")
    print("     📊 保持更好的性能（PPL 更低）")
    print("     ⚖️  在剪枝率和性能之间取得平衡")
    print()


def main():
    import argparse
    parser = argparse.ArgumentParser(description='分析非均衡结构化剪枝的详细过程')
    parser.add_argument('--config', type=str,
                       default='prune_log/llama_unbalanced_prune/layer_importance_config.json',
                       help='层重要性配置文件路径')
    parser.add_argument('--model', type=str,
                       default='/mnt/sharedata/song/models/Meta-Llama-3-8B-Instruct',
                       help='模型路径')
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

    # 加载模型（仅用于演示，不执行实际剪枝）
    print("加载模型结构（仅用于演示）...")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            device_map='cpu',
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True
        )
        print("✅ 模型加载成功")
        print()

        # 分析 ch_sparsity_dict 创建
        ch_sparsity_dict = analyze_ch_sparsity_dict_creation(
            model, filtered_rates,
            prune_attention=True,
            prune_mlp=True
        )

    except Exception as e:
        print(f"⚠️  模型加载失败: {e}")
        print("继续进行原理讲解...")
        print()
        ch_sparsity_dict = None

    # 解释 MetaPruner 工作流程
    explain_metapruner_workflow(ch_sparsity_dict)

    print("=" * 80)
    print("✅ 分析完成！")
    print("=" * 80)
    print()
    print("总结:")
    print("  1. 步骤2 计算所有层的剪枝率（基于层重要性）")
    print("  2. 步骤3 过滤层 + 创建模块级字典（保护关键层）")
    print("  3. 步骤4 MetaPruner 执行结构化剪枝（物理减小模型）")
    print()
    print("这就是为什么:")
    print("  📊 看到 32 层的剪枝率")
    print("  ✂️  但只剪枝 27 层（3-29）")
    print()


if __name__ == '__main__':
    main()
