#!/usr/bin/env python3
"""
Llama-3 非均衡结构化剪枝脚本
结合层重要度评估和结构化剪枝
"""

import os
import gc
import sys
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

    # 剪枝范围
    parser.add_argument('--block_attention_layer_start', type=int, default=3,
                       help='Attention 剪枝起始层')
    parser.add_argument('--block_attention_layer_end', type=int, default=30,
                       help='Attention 剪枝结束层')
    parser.add_argument('--block_mlp_layer_start', type=int, default=3,
                       help='MLP 剪枝起始层')
    parser.add_argument('--block_mlp_layer_end', type=int, default=30,
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
        max_rate=args.max_pruning_rate
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

    filtered_pruning_rates = {
        idx: rate for idx, rate in layer_pruning_rates.items()
        if idx in pruning_layers
    }

    ch_sparsity_dict = create_ch_sparsity_dict_for_llama(
        model,
        filtered_pruning_rates,
        prune_attention=True,
        prune_mlp=True
    )

    logger.log(f"为 {len(ch_sparsity_dict)} 个模块设置了自定义剪枝率")

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

    # 配置剪枝参数
    kwargs = {
        "importance": imp,
        "global_pruning": False,
        "iterative_steps": args.iterative_steps,
        "ch_sparsity": args.pruning_ratio,  # 默认剪枝率
        "ch_sparsity_dict": ch_sparsity_dict,  # ⭐ 每层的剪枝率
        "ignored_layers": [],
        "channel_groups": {},
        "consecutive_groups": {
            layer.self_attn.k_proj: layer.self_attn.head_dim for layer in model.model.layers
        },
        "customized_pruners": {
            LlamaRMSNorm: llama_pruner.hf_rmsnorm_pruner,
        },
        "root_module_types": None,
        "root_instances": [model.model.layers[i].self_attn.k_proj for i in range(args.block_attention_layer_start, args.block_attention_layer_end)] +
                         [model.model.layers[i].mlp.gate_proj for i in range(args.block_mlp_layer_start, args.block_mlp_layer_end)]
    }

    logger.log(f"剪枝 Attention 层 = {list(range(args.block_attention_layer_start, args.block_attention_layer_end))}")
    logger.log(f"剪枝 MLP 层 = {list(range(args.block_mlp_layer_start, args.block_mlp_layer_end))}")

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

        # 执行剪枝
        pruner.step()

        after_pruning_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.log(f"迭代 {i+1}/{args.iterative_steps} 后参数量: {after_pruning_parameters:,}")

        # 更新推理相关属性
        for layer in model.model.layers:
            layer.self_attn.num_heads = layer.self_attn.q_proj.weight.data.shape[0] // layer.self_attn.head_dim
            layer.self_attn.num_key_value_heads = layer.self_attn.k_proj.weight.data.shape[0] // layer.self_attn.head_dim

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

    # ==================== 步骤5: 保存模型 ====================
    if args.save_model:
        logger.log("=" * 80)
        logger.log("步骤5: 保存模型")
        logger.log("=" * 80)

        model.half()
        torch.save({
            'model': model,
            'tokenizer': tokenizer,
            'layer_pruning_rates': layer_pruning_rates,
            'layer_importance': layer_importance,
        }, logger.best_checkpoint_path, weights_only=False)

        logger.log(f"模型已保存到: {logger.best_checkpoint_path}")

    # ==================== 步骤6: 评估 PPL ====================
    if args.test_after_train:
        logger.log("=" * 80)
        logger.log("步骤6: 评估困惑度")
        logger.log("=" * 80)

        model.to(args.device)
        model.eval()

        ppl = PPLMetric(model, tokenizer, ['wikitext2', 'ptb'],
                       args.max_seq_len, device=args.device)
        logger.log(f"剪枝后 PPL: {ppl}")

    logger.log("\n完成！")


if __name__ == "__main__":
    main()
