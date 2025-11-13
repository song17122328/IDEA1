#!/usr/bin/env python3
"""
Llama-3 非均衡结构化剪枝脚本
结合层重要度评估和结构化剪枝
"""

import os
import gc
import sys
import json
import torch
import argparse
import numpy as np
from transformers import AutoTokenizer, LlamaForCausalLM
from transformers.models.llama.modeling_llama import LlamaRMSNorm

import LLMPruner.torch_pruning as tp
from LLMPruner.pruner import hf_llama_pruner as llama_pruner
from LLMPruner.utils.logger import LoggerWithDepth
from LLMPruner.evaluator.ppl import PPLMetric
from LLMPruner.datasets.example_samples import get_examples

from layer_importance import (
    LayerImportanceAnalyzer,
    UnbalancedStructuredPruningCalculator,
    create_ch_sparsity_dict_for_llama
)


def load_evaluation_data(tokenizer, num_samples=100):
    """加载评估数据"""
    from datasets import load_dataset

    print("加载评估数据...")
    dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')

    texts = []
    for i, item in enumerate(dataset):
        if i >= num_samples:
            break
        text = item['text'].strip()
        if len(text) > 50:  # 只使用足够长的文本
            texts.append(text)

    return texts[:num_samples]


def main():
    parser = argparse.ArgumentParser(description='Llama-3 非均衡结构化剪枝')

    # 模型参数
    parser.add_argument('--base_model', type=str, required=True,
                       help='原始模型路径')
    parser.add_argument('--save_ckpt_log_name', type=str, default='llama_unbalanced_prune',
                       help='日志和模型保存目录名称')

    # 剪枝参数
    parser.add_argument('--pruning_ratio', type=float, default=0.25,
                       help='目标剪枝率（整体平均）')
    parser.add_argument('--pruner_type', type=str, default='taylor',
                       choices=['random', 'l1', 'l2', 'taylor'],
                       help='剪枝重要性评估方法')

    # 层重要度评估
    parser.add_argument('--importance_method', type=str, default='removal',
                       choices=['removal', 'activation'],
                       help='层重要度评估方法：removal(移除层) 或 activation(激活值)')
    parser.add_argument('--importance_samples', type=int, default=50,
                       help='用于评估层重要度的样本数量')
    parser.add_argument('--skip_importance_analysis', action='store_true',
                       help='跳过层重要度分析，使用已保存的配置')
    parser.add_argument('--importance_config', type=str, default='layer_importance_config.json',
                       help='层重要度配置文件路径')

    # 非均衡剪枝策略
    parser.add_argument('--pruning_strategy', type=str, default='inverse',
                       choices=['inverse', 'proportional', 'uniform'],
                       help='剪枝策略：inverse(重要层剪少), proportional(重要层剪多), uniform(均匀)')
    parser.add_argument('--alpha', type=float, default=1.0,
                       help='重要性权重系数，越大差异越明显')
    parser.add_argument('--min_pruning_rate', type=float, default=0.0,
                       help='最小剪枝率')
    parser.add_argument('--max_pruning_rate', type=float, default=0.8,
                       help='最大剪枝率')

    # 剪枝范围（默认剪枝所有层）
    parser.add_argument('--block_attention_layer_start', type=int, default=0,
                       help='Attention 剪枝起始层')
    parser.add_argument('--block_attention_layer_end', type=int, default=32,
                       help='Attention 剪枝结束层')
    parser.add_argument('--block_mlp_layer_start', type=int, default=0,
                       help='MLP 剪枝起始层')
    parser.add_argument('--block_mlp_layer_end', type=int, default=32,
                       help='MLP 剪枝结束层')

    # 其他参数
    parser.add_argument('--device', type=str, default='cuda',
                       help='设备')
    parser.add_argument('--num_examples', type=int, default=10,
                       help='Taylor 重要性评估的样本数')
    parser.add_argument('--iterative_steps', type=int, default=1,
                       help='迭代剪枝步数')
    parser.add_argument('--save_model', action='store_true',
                       help='是否保存模型')
    parser.add_argument('--test_after_train', action='store_true',
                       help='剪枝后是否评估')
    parser.add_argument('--max_seq_len', type=int, default=128,
                       help='最大序列长度')

    args = parser.parse_args()

    # 设置设备
    print(f"默认设备: {args.device}")
    if args.device == "cuda":
        from get_best_gpu import get_best_gpu
        args.device = "cuda:" + str(get_best_gpu())
    print(f"最优设备为: {args.device}")

    # 创建日志
    logger = LoggerWithDepth(
        env_name=args.save_ckpt_log_name,
        config=args.__dict__,
        root_dir='prune_log',
        setup_sublogger=True
    )

    # 加载模型和分词器
    print("加载模型和分词器...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    model = LlamaForCausalLM.from_pretrained(
        args.base_model,
        device_map=args.device,
        torch_dtype=torch.float16,
    )
    model.half()

    # 获取层数
    num_layers = len(model.model.layers)
    logger.log(f"模型总层数: {num_layers}")

    # ==================== 步骤1: 评估层重要性 ====================
    if not args.skip_importance_analysis:
        logger.log("=" * 80)
        logger.log("步骤1: 评估层重要性")
        logger.log("=" * 80)

        # 加载评估数据
        eval_texts = load_evaluation_data(tokenizer, num_samples=args.importance_samples)
        logger.log(f"加载了 {len(eval_texts)} 个评估样本")

        # 创建分析器
        analyzer = LayerImportanceAnalyzer(model, tokenizer, device=args.device)

        # 评估层重要性
        if args.importance_method == 'removal':
            logger.log("使用层移除法评估重要性...")
            layer_importance = analyzer.measure_layer_importance_by_removal(
                eval_texts, num_layers=num_layers
            )
        else:
            logger.log("使用激活值法评估重要性...")
            layer_importance = analyzer.measure_layer_importance_by_activation(eval_texts)

        # 显示重要性结果
        logger.log("\n层重要性评分:")
        for layer_idx, importance in sorted(layer_importance.items()):
            logger.log(f"  Layer {layer_idx}: {importance:.6f}")

    else:
        logger.log("跳过层重要度分析，加载已保存的配置...")
        calculator = UnbalancedStructuredPruningCalculator({}, num_layers)
        layer_pruning_rates = calculator.load_pruning_rates(args.importance_config)

        # 反推重要性（简化处理）
        layer_importance = {i: 1.0 for i in range(num_layers)}

    # ==================== 步骤2: 计算各层剪枝率 ====================
    logger.log("=" * 80)
    logger.log("步骤2: 计算各层剪枝率")
    logger.log("=" * 80)

    calculator = UnbalancedStructuredPruningCalculator(layer_importance, num_layers)

    # 计算剪枝率
    layer_pruning_rates = calculator.compute_layer_pruning_rates(
        target_overall_rate=args.pruning_ratio,
        strategy=args.pruning_strategy,
        alpha=args.alpha,
        min_rate=args.min_pruning_rate,
        max_rate=args.max_pruning_rate,
        use_log_transform=True  # 使用对数变换处理极端值
    )

    # 验证剪枝率
    stats = calculator.verify_average_pruning_rate(layer_pruning_rates)
    logger.log(f"\n剪枝率统计:")
    logger.log(f"  平均剪枝率: {stats['average_pruning_rate']:.4f}")
    logger.log(f"  标准差: {stats['std_pruning_rate']:.4f}")
    logger.log(f"  最小剪枝率: {stats['min_pruning_rate']:.4f}")
    logger.log(f"  最大剪枝率: {stats['max_pruning_rate']:.4f}")
    logger.log(f"  剪枝率范围: {stats['rate_range']:.4f}")

    # 显示各层剪枝率
    logger.log("\n各层剪枝率:")
    for layer_idx in range(num_layers):
        rate = layer_pruning_rates.get(layer_idx, 0.0)
        logger.log(f"  Layer {layer_idx}: {rate:.4f}")

    # 保存配置
    config_path = os.path.join(logger.log_dir, args.importance_config)
    calculator.save_pruning_rates(layer_pruning_rates, config_path)

    # 可视化
    viz_path = os.path.join(logger.log_dir, 'pruning_strategy.png')
    calculator.visualize_pruning_strategy(layer_pruning_rates, save_path=viz_path)

    # ==================== 步骤3: 创建 ch_sparsity_dict ====================
    logger.log("=" * 80)
    logger.log("步骤3: 创建模块级剪枝率字典")
    logger.log("=" * 80)

    # 只为要剪枝的层创建 ch_sparsity_dict
    pruning_layers = set(range(args.block_attention_layer_start, args.block_attention_layer_end)) | \
                    set(range(args.block_mlp_layer_start, args.block_mlp_layer_end))

    # 过滤：只保留在剪枝范围内的层
    filtered_pruning_rates = {
        idx: rate for idx, rate in layer_pruning_rates.items()
        if idx in pruning_layers
    }

    # 进一步过滤：移除剪枝率太低的层（避免剪枝0个通道导致错误）
    # 对于 k_proj (1024通道，head_dim=128)，至少需要剪枝 1个head = 128/1024 = 12.5%
    # 为了安全，设置为 15%
    min_effective_rate = 0.15  # 最小有效剪枝率：15%
    effective_pruning_rates = {
        idx: rate for idx, rate in filtered_pruning_rates.items()
        if rate >= min_effective_rate
    }

    # 记录被过滤掉的层
    skipped_layers = set(filtered_pruning_rates.keys()) - set(effective_pruning_rates.keys())
    if skipped_layers:
        logger.log(f"警告：以下层的剪枝率 < {min_effective_rate:.2%}，已跳过：{sorted(skipped_layers)}")
        for idx in sorted(skipped_layers):
            logger.log(f"  Layer {idx}: {filtered_pruning_rates[idx]:.4f} (需要 >= {min_effective_rate:.2%} 才能剪枝至少1个attention head)")

    ch_sparsity_dict = create_ch_sparsity_dict_for_llama(
        model,
        effective_pruning_rates,
        prune_attention=True,
        prune_mlp=True
    )

    logger.log(f"为 {len(ch_sparsity_dict)} 个模块设置了自定义剪枝率")

    # 保存 ch_sparsity_dict 到文件（转换为可读格式）
    ch_sparsity_readable = {}
    for module, rate in ch_sparsity_dict.items():
        # 获取模块的完整名称
        module_name = None
        for name, m in model.named_modules():
            if m is module:
                module_name = name
                break
        if module_name:
            ch_sparsity_readable[module_name] = float(rate)

    ch_sparsity_path = os.path.join(logger.log_dir, 'ch_sparsity_dict.json')
    with open(ch_sparsity_path, 'w') as f:
        json.dump(ch_sparsity_readable, f, indent=2, ensure_ascii=False)
    logger.log(f"ch_sparsity_dict 已保存到: {ch_sparsity_path}")

    # 获取实际参与剪枝的层列表
    actual_pruning_layers = sorted(effective_pruning_rates.keys())
    logger.log(f"实际参与剪枝的层: {actual_pruning_layers}")

    # ==================== 注意：关于 GQA 比例 ====================
    logger.log("=" * 80)
    logger.log("关于 GQA 比例的说明")
    logger.log("=" * 80)
    logger.log(f"⚠️  torch_pruning 的依赖图传播是'通道对通道'的，不理解 GQA 的 4:1 结构")
    logger.log(f"   示例：剪枝率 0.25")
    logger.log(f"   - k_proj: 剪掉 1024 × 0.25 = 256 通道（2个KV heads）→ 剩余 6 KV heads")
    logger.log(f"   - q_proj: 依赖图传播相同的 256 通道（2个Q heads）→ 剩余 30 Q heads")
    logger.log(f"   - 结果：30:6 = 5:1 ✗ (不是原始的 4:1)")
    logger.log(f"\n✅ 解决方案：剪枝后的后处理阶段会强制修正为 4:1")
    logger.log(f"   - 30:6 (5:1) → 24:6 (4:1)：截断 6 个 Q heads")
    logger.log(f"   - 这确保所有层都保持原始的 GQA 比例")
    logger.log("=" * 80)

    # ==================== 步骤4: 执行结构化剪枝 ====================
    logger.log("=" * 80)
    logger.log("步骤4: 执行结构化剪枝")
    logger.log("=" * 80)

    # 启用梯度
    for param in model.parameters():
        param.requires_grad_(True)

    before_pruning_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.log(f"剪枝前参数量: {before_pruning_parameters:,}")

    # 创建 forward prompts
    forward_prompts = torch.tensor([
        [1, 306, 4658, 278, 6593, 310, 2834, 338],
        [1, 3439, 17632, 1925, 29892, 278, 6368, 310],
    ]).to(args.device)

    # 选择重要性评估方法
    if args.pruner_type == 'random':
        imp = tp.importance.RandomImportance()
    elif args.pruner_type == 'l1':
        imp = llama_pruner.MagnitudeImportance(p=1)
    elif args.pruner_type == 'l2':
        imp = llama_pruner.MagnitudeImportance(p=2)
    elif args.pruner_type == 'taylor':
        imp = llama_pruner.TaylorImportance(group_reduction='sum', taylor='param_first')
    else:
        raise NotImplementedError

    logger.log(f"使用 {args.pruner_type} 剪枝器...")

    # ==================== 新方案：手动控制 GQA 剪枝 ====================
    # 策略：让 q_proj 和 o_proj 不参与自动依赖图，手动剪枝以确保4:1比例
    # 原因：torch_pruning的依赖图是"通道对通道"传播，无法理解GQA的4:1结构
    #
    # 步骤：
    # 1. pruner 只管理 k_proj, v_proj, 和 MLP（不包括 q_proj, o_proj）
    # 2. 每次 pruner.step() 后，手动剪枝 q_proj（确保是 k_proj 的4倍）
    # 3. 同时手动调整 o_proj 输入维度（匹配 q_proj 输出）
    # 4. 这样所有模块从始至终保持一致的维度，LoRA 不会遇到配置冲突

    root_instances = []
    for layer_idx in actual_pruning_layers:
        root_instances.append(model.model.layers[layer_idx].self_attn.k_proj)  # KV attention
        root_instances.append(model.model.layers[layer_idx].mlp.gate_proj)     # MLP

    # 获取 GQA 配置
    num_heads = model.config.num_attention_heads  # 32
    num_key_value_heads = model.config.num_key_value_heads  # 8
    head_dim = 128
    gqa_ratio = num_heads // num_key_value_heads  # 4

    # 配置 consecutive_groups：让 k_proj 按 head 级别剪枝
    consecutive_groups = {}
    for layer in model.model.layers:
        consecutive_groups[layer.self_attn.k_proj] = head_dim  # 1个KV head = 128通道

    # 关键：将 q_proj 和 o_proj 加入 ignored_layers，不让它们参与自动剪枝
    # 我们会手动控制它们的剪枝
    ignored_layers = []
    for layer in model.model.layers:
        ignored_layers.append(layer.self_attn.q_proj)
        ignored_layers.append(layer.self_attn.o_proj)

    logger.log("=" * 80)
    logger.log("🔧 新的 GQA-aware 剪枝策略")
    logger.log("=" * 80)
    logger.log(f"GQA 配置: Q heads={num_heads}, KV heads={num_key_value_heads}, 比例={gqa_ratio}:1")
    logger.log(f"\n剪枝配置:")
    logger.log(f"  - root_instances: k_proj 和 gate_proj")
    logger.log(f"  - consecutive_groups: k_proj (128通道/head)")
    logger.log(f"  - ignored_layers: q_proj 和 o_proj（手动控制以确保4:1比例）")
    logger.log(f"\n工作流程:")
    logger.log(f"  1. pruner.step() 剪枝 k_proj 和 MLP")
    logger.log(f"  2. 手动剪枝 q_proj（确保 Q heads = KV heads × 4）")
    logger.log(f"  3. 手动调整 o_proj 输入维度（匹配 q_proj 输出）")
    logger.log(f"  4. 所有模块维度一致，无需后处理")
    logger.log("=" * 80 + "\n")

    kwargs = {
        "importance": imp,
        "global_pruning": False,
        "iterative_steps": args.iterative_steps,
        "ch_sparsity": args.pruning_ratio,  # 默认剪枝率（不应该被使用）
        "ch_sparsity_dict": ch_sparsity_dict,  # ⭐ 每层的剪枝率
        "ignored_layers": ignored_layers,  # ⭐ 忽略 q_proj 和 o_proj，我们会手动处理
        "channel_groups": {},  # ⭐ 空字典，与原始 llama3.py 一致
        "consecutive_groups": consecutive_groups,  # ⭐ 强制 k_proj 按 128 通道分组
        "customized_pruners": {
            LlamaRMSNorm: llama_pruner.hf_rmsnorm_pruner,
        },
        "root_module_types": None,
        "root_instances": root_instances  # ⭐ 只有 k_proj 和 gate_proj（与llama3.py一致）
    }

    logger.log(f"实际剪枝 Attention 和 MLP 的层: {actual_pruning_layers}")
    logger.log(f"root_instances 数量: {len(root_instances)} (每层2个模块: k_proj + gate_proj)")

    # 创建剪枝器
    pruner = tp.pruner.MetaPruner(model, forward_prompts, **kwargs)
    model.zero_grad()

    logger.log("开始剪枝...")

    # 迭代剪枝
    for i in range(args.iterative_steps):
        if args.pruner_type == 'taylor':
            example_prompts = get_examples('wikitext', tokenizer, args.num_examples, seq_len=64).to(args.device)
            logger.log(f"迭代步骤 {i}: 开始反向传播...")

            loss = model(example_prompts, labels=example_prompts).loss
            logger.log(f"Loss = {loss.item()}")
            loss.backward()

        # 执行剪枝（只剪枝 k_proj, v_proj, MLP）
        pruner.step()

        logger.log(f"\n{'='*80}")
        logger.log(f"迭代 {i+1}/{args.iterative_steps}: 手动处理 Q-projection 和 O-projection")
        logger.log(f"{'='*80}")

        # 手动剪枝 q_proj 和调整 o_proj（确保 4:1 GQA 比例）
        original_gqa_ratio = model.config.num_attention_heads // model.config.num_key_value_heads  # 4
        head_dim = 128

        for layer_idx, layer in enumerate(model.model.layers):
            # 1. 读取 k_proj 剪枝后的维度（由 pruner 处理）
            k_out_channels = layer.self_attn.k_proj.weight.data.shape[0]
            assert k_out_channels % head_dim == 0, f"Layer {layer_idx}: k_proj 维度 {k_out_channels} 不是 head_dim {head_dim} 的倍数"
            num_kv_heads = k_out_channels // head_dim

            # 2. 计算目标 q_proj 维度（保持 4:1 比例）
            target_num_heads = num_kv_heads * original_gqa_ratio
            target_q_channels = target_num_heads * head_dim

            # 3. 获取当前 q_proj 维度
            current_q_channels = layer.self_attn.q_proj.weight.data.shape[0]
            current_q_in_channels = layer.self_attn.q_proj.weight.data.shape[1]

            # 4. 如果需要，手动剪枝 q_proj（选择前 N 个 heads）
            if current_q_channels != target_q_channels:
                logger.log(f"Layer {layer_idx}: 手动剪枝 q_proj")
                logger.log(f"  当前 Q heads: {current_q_channels // head_dim}")
                logger.log(f"  当前 KV heads: {num_kv_heads}")
                logger.log(f"  目标 Q heads: {target_num_heads} (KV {num_kv_heads} × {original_gqa_ratio})")

                # 剪枝 q_proj 权重：只保留前 target_q_channels 个输出通道
                layer.self_attn.q_proj.weight.data = layer.self_attn.q_proj.weight.data[:target_q_channels, :]
                if layer.self_attn.q_proj.bias is not None:
                    layer.self_attn.q_proj.bias.data = layer.self_attn.q_proj.bias.data[:target_q_channels]

                # 调整 o_proj 的输入维度（它接收 q_proj 的输出）
                layer.self_attn.o_proj.weight.data = layer.self_attn.o_proj.weight.data[:, :target_q_channels]

                logger.log(f"  ✅ 已调整: Q {target_num_heads} heads, KV {num_kv_heads} heads → {original_gqa_ratio}:1")

            # 5. 更新层配置
            layer.self_attn.num_heads = target_num_heads
            layer.self_attn.num_key_value_heads = num_kv_heads
            layer.self_attn.num_key_value_groups = target_num_heads // num_kv_heads

        after_pruning_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.log(f"\n迭代 {i+1}/{args.iterative_steps} 完成，参数量: {after_pruning_parameters:,}")

    # 清理梯度
    model.zero_grad()
    for name, module in model.named_parameters():
        if 'weight' in name:
            module.grad = None

    del pruner
    torch.cuda.empty_cache()

    # 最终统计
    final_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.log(f"\n剪枝完成!")
    logger.log(f"剪枝前参数量: {before_pruning_parameters:,}")
    logger.log(f"剪枝后参数量: {final_parameters:,}")
    logger.log(f"参数减少量: {before_pruning_parameters - final_parameters:,}")
    logger.log(f"实际剪枝率: {(1 - final_parameters/before_pruning_parameters)*100:.2f}%")

    # 验证和打印每层的 head 数量
    logger.log("\n每层的 attention heads 配置:")
    for idx, layer in enumerate(model.model.layers):
        logger.log(f"  Layer {idx}: q_heads={layer.self_attn.num_heads}, kv_heads={layer.self_attn.num_key_value_heads}, ratio={layer.self_attn.num_heads/layer.self_attn.num_key_value_heads:.1f}:1")

    # ==================== 步骤5: 保存模型 ====================
    if args.save_model:
        logger.log("=" * 80)
        logger.log("步骤5: 保存模型")
        logger.log("=" * 80)

        # 保存前最终配置确认和修正（确保所有配置正确，避免 LoRA 维度不匹配）
        logger.log("\n保存前最终配置检查...")
        head_dim = 128
        for i, layer in enumerate(model.model.layers):
            # 从实际权重推断配置
            actual_q_heads = layer.self_attn.q_proj.weight.shape[0] // head_dim
            actual_kv_heads = layer.self_attn.k_proj.weight.shape[0] // head_dim

            # 强制更新所有相关配置
            layer.self_attn.num_heads = actual_q_heads
            layer.self_attn.num_key_value_heads = actual_kv_heads
            layer.self_attn.num_key_value_groups = actual_q_heads // actual_kv_heads

            logger.log(f"  Layer {i}: {actual_q_heads} Q heads, {actual_kv_heads} KV heads, ratio {actual_q_heads/actual_kv_heads:.1f}:1")

        logger.log("✅ 配置检查完成\n")

        model.half()
        torch.save({
            'model': model,
            'tokenizer': tokenizer,
            'layer_pruning_rates': layer_pruning_rates,
            'layer_importance': layer_importance,
        }, logger.best_checkpoint_path)

        logger.log(f"模型已保存到: {logger.best_checkpoint_path}")

    # ==================== 步骤6: 评估 PPL ====================
    if args.test_after_train:
        logger.log("=" * 80)
        logger.log("步骤6: 评估困惑度")
        logger.log("=" * 80)

        # 在评估前再次更新 head 配置（确保正确）
        for layer in model.model.layers:
            layer.self_attn.num_heads = layer.self_attn.q_proj.weight.data.shape[0] // layer.self_attn.head_dim
            layer.self_attn.num_key_value_heads = layer.self_attn.k_proj.weight.data.shape[0] // layer.self_attn.head_dim

        model.to(args.device)
        model.eval()

        ppl = PPLMetric(model, tokenizer, ['wikitext2', 'ptb'],
                       args.max_seq_len, device=args.device)
        logger.log(f"剪枝后 PPL: {ppl}")

    logger.log("\n完成！")


if __name__ == "__main__":
    main()
