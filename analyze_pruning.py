#!/usr/bin/env python3
"""
剪枝分析工具
详细分析原始模型和剪枝后模型的每一层参数维度、剪枝度和结构化稀疏度
"""

import torch
import argparse
from pathlib import Path
from collections import defaultdict
import pandas as pd


def load_model(model_path, is_pruned=False):
    """加载模型"""
    print(f"正在加载模型: {model_path}")

    if is_pruned:
        # 剪枝后的模型保存格式
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        if 'model' in checkpoint:
            model = checkpoint['model']
        else:
            model = checkpoint
    else:
        # 原始 HuggingFace 模型
        from transformers import LlamaForCausalLM
        model = LlamaForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map='cpu'
        )

    return model


def analyze_layer(layer, layer_idx, model_type="original"):
    """分析单个 transformer 层的参数"""
    stats = {
        'layer_idx': layer_idx,
        'model_type': model_type,
    }

    # Attention 模块
    if hasattr(layer, 'self_attn'):
        attn = layer.self_attn

        # Q, K, V, O projections
        if hasattr(attn, 'q_proj'):
            q_shape = attn.q_proj.weight.shape
            stats['attn_q_proj'] = f"{q_shape[0]} × {q_shape[1]}"
            stats['attn_q_proj_params'] = q_shape[0] * q_shape[1]

        if hasattr(attn, 'k_proj'):
            k_shape = attn.k_proj.weight.shape
            stats['attn_k_proj'] = f"{k_shape[0]} × {k_shape[1]}"
            stats['attn_k_proj_params'] = k_shape[0] * k_shape[1]

        if hasattr(attn, 'v_proj'):
            v_shape = attn.v_proj.weight.shape
            stats['attn_v_proj'] = f"{v_shape[0]} × {v_shape[1]}"
            stats['attn_v_proj_params'] = v_shape[0] * v_shape[1]

        if hasattr(attn, 'o_proj'):
            o_shape = attn.o_proj.weight.shape
            stats['attn_o_proj'] = f"{o_shape[0]} × {o_shape[1]}"
            stats['attn_o_proj_params'] = o_shape[0] * o_shape[1]

    # MLP 模块
    if hasattr(layer, 'mlp'):
        mlp = layer.mlp

        if hasattr(mlp, 'gate_proj'):
            gate_shape = mlp.gate_proj.weight.shape
            stats['mlp_gate_proj'] = f"{gate_shape[0]} × {gate_shape[1]}"
            stats['mlp_gate_proj_params'] = gate_shape[0] * gate_shape[1]

        if hasattr(mlp, 'up_proj'):
            up_shape = mlp.up_proj.weight.shape
            stats['mlp_up_proj'] = f"{up_shape[0]} × {up_shape[1]}"
            stats['mlp_up_proj_params'] = up_shape[0] * up_shape[1]

        if hasattr(mlp, 'down_proj'):
            down_shape = mlp.down_proj.weight.shape
            stats['mlp_down_proj'] = f"{down_shape[0]} × {down_shape[1]}"
            stats['mlp_down_proj_params'] = down_shape[0] * down_shape[1]

    return stats


def compare_models(original_model, pruned_model):
    """对比两个模型的每一层"""
    print("\n" + "="*100)
    print("开始逐层分析...")
    print("="*100)

    # 获取层数
    original_layers = original_model.model.layers
    pruned_layers = pruned_model.model.layers

    num_layers = len(original_layers)
    print(f"\n模型总层数: {num_layers}")

    # 收集所有层的统计信息
    comparison_data = []

    for i in range(num_layers):
        print(f"\n分析第 {i} 层...")

        # 分析原始层
        orig_stats = analyze_layer(original_layers[i], i, "original")

        # 分析剪枝后的层
        pruned_stats = analyze_layer(pruned_layers[i], i, "pruned")

        # 合并统计信息
        layer_comparison = {
            'Layer': i,
        }

        # 对比每个模块
        modules = ['attn_q_proj', 'attn_k_proj', 'attn_v_proj', 'attn_o_proj',
                  'mlp_gate_proj', 'mlp_up_proj', 'mlp_down_proj']

        total_orig_params = 0
        total_pruned_params = 0

        for module in modules:
            if module in orig_stats and module in pruned_stats:
                # 维度信息
                layer_comparison[f'{module}_original'] = orig_stats[module]
                layer_comparison[f'{module}_pruned'] = pruned_stats[module]

                # 参数数量
                orig_params = orig_stats.get(f'{module}_params', 0)
                pruned_params = pruned_stats.get(f'{module}_params', 0)

                total_orig_params += orig_params
                total_pruned_params += pruned_params

                # 计算剪枝度和稀疏度
                if orig_params > 0:
                    retention_rate = pruned_params / orig_params
                    sparsity = 1 - retention_rate
                    layer_comparison[f'{module}_retention'] = f"{retention_rate:.4f}"
                    layer_comparison[f'{module}_sparsity'] = f"{sparsity:.4f}"

        # 计算整层的剪枝度和稀疏度
        if total_orig_params > 0:
            layer_retention = total_pruned_params / total_orig_params
            layer_sparsity = 1 - layer_retention

            layer_comparison['total_original_params'] = total_orig_params
            layer_comparison['total_pruned_params'] = total_pruned_params
            layer_comparison['layer_retention_rate'] = f"{layer_retention:.4f}"
            layer_comparison['layer_sparsity'] = f"{layer_sparsity:.4f}"

        comparison_data.append(layer_comparison)

    return comparison_data


def print_summary_table(comparison_data):
    """打印汇总表格"""
    print("\n" + "="*100)
    print("每层汇总统计")
    print("="*100)

    # 创建汇总表格
    summary = []
    for data in comparison_data:
        summary.append({
            '层编号': data['Layer'],
            '原始参数量': f"{data['total_original_params']:,}",
            '剪枝后参数量': f"{data['total_pruned_params']:,}",
            '保留率': data['layer_retention_rate'],
            '稀疏度': data['layer_sparsity'],
        })

    df = pd.DataFrame(summary)
    print("\n" + df.to_string(index=False))

    # 打印总体统计
    total_orig = sum(d['total_original_params'] for d in comparison_data)
    total_pruned = sum(d['total_pruned_params'] for d in comparison_data)
    overall_retention = total_pruned / total_orig if total_orig > 0 else 0
    overall_sparsity = 1 - overall_retention

    print("\n" + "="*100)
    print("全局统计")
    print("="*100)
    print(f"原始模型总参数量: {total_orig:,}")
    print(f"剪枝后模型总参数量: {total_pruned:,}")
    print(f"参数减少量: {total_orig - total_pruned:,}")
    print(f"整体保留率: {overall_retention:.4f} ({overall_retention*100:.2f}%)")
    print(f"整体稀疏度: {overall_sparsity:.4f} ({overall_sparsity*100:.2f}%)")


def print_detailed_module_analysis(comparison_data):
    """打印详细的模块分析"""
    print("\n" + "="*100)
    print("详细模块分析")
    print("="*100)

    modules = ['attn_q_proj', 'attn_k_proj', 'attn_v_proj', 'attn_o_proj',
              'mlp_gate_proj', 'mlp_up_proj', 'mlp_down_proj']

    module_names = {
        'attn_q_proj': 'Attention Q Projection',
        'attn_k_proj': 'Attention K Projection',
        'attn_v_proj': 'Attention V Projection',
        'attn_o_proj': 'Attention O Projection',
        'mlp_gate_proj': 'MLP Gate Projection',
        'mlp_up_proj': 'MLP Up Projection',
        'mlp_down_proj': 'MLP Down Projection',
    }

    for module in modules:
        print(f"\n{'─'*100}")
        print(f"📊 {module_names[module]}")
        print(f"{'─'*100}")

        module_data = []
        for data in comparison_data:
            if f'{module}_original' in data:
                module_data.append({
                    '层': data['Layer'],
                    '原始维度': data.get(f'{module}_original', 'N/A'),
                    '剪枝后维度': data.get(f'{module}_pruned', 'N/A'),
                    '保留率': data.get(f'{module}_retention', 'N/A'),
                    '稀疏度': data.get(f'{module}_sparsity', 'N/A'),
                })

        if module_data:
            df = pd.DataFrame(module_data)
            print(df.to_string(index=False))

            # 检查是否所有层都一样
            retention_rates = [d['保留率'] for d in module_data if d['保留率'] != 'N/A']
            if retention_rates:
                unique_rates = set(retention_rates)
                if len(unique_rates) == 1:
                    print(f"\n✅ 所有层的 {module_names[module]} 剪枝度一致: {retention_rates[0]}")
                else:
                    print(f"\n⚠️  不同层的 {module_names[module]} 剪枝度不同")
                    print(f"   最小保留率: {min(retention_rates)}")
                    print(f"   最大保留率: {max(retention_rates)}")


def save_to_csv(comparison_data, output_file):
    """保存分析结果到CSV文件"""
    print(f"\n保存详细分析结果到: {output_file}")

    # 展开数据
    rows = []
    for data in comparison_data:
        layer_idx = data['Layer']

        modules = ['attn_q_proj', 'attn_k_proj', 'attn_v_proj', 'attn_o_proj',
                  'mlp_gate_proj', 'mlp_up_proj', 'mlp_down_proj']

        for module in modules:
            if f'{module}_original' in data:
                rows.append({
                    '层编号': layer_idx,
                    '模块名称': module,
                    '原始维度': data.get(f'{module}_original', ''),
                    '剪枝后维度': data.get(f'{module}_pruned', ''),
                    '保留率': data.get(f'{module}_retention', ''),
                    '稀疏度': data.get(f'{module}_sparsity', ''),
                })

    df = pd.DataFrame(rows)
    df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"✅ 已保存 {len(rows)} 条记录到 {output_file}")


def main():
    parser = argparse.ArgumentParser(description='分析原始模型和剪枝后模型的差异')
    parser.add_argument('--original_model', type=str, required=True,
                       help='原始模型路径 (HuggingFace 模型目录)')
    parser.add_argument('--pruned_model', type=str, required=True,
                       help='剪枝后模型路径 (.bin 文件)')
    parser.add_argument('--output', type=str, default='pruning_analysis.csv',
                       help='输出CSV文件路径 (默认: pruning_analysis.csv)')
    parser.add_argument('--layers', type=str, default=None,
                       help='指定要分析的层，例如 "0,1,2" 或 "0-5" (默认: 所有层)')

    args = parser.parse_args()

    print("="*100)
    print("剪枝分析工具")
    print("="*100)
    print(f"原始模型: {args.original_model}")
    print(f"剪枝模型: {args.pruned_model}")
    print("="*100)

    # 加载模型
    print("\n加载原始模型...")
    original_model = load_model(args.original_model, is_pruned=False)

    print("\n加载剪枝后模型...")
    pruned_model = load_model(args.pruned_model, is_pruned=True)

    # 对比分析
    comparison_data = compare_models(original_model, pruned_model)

    # 打印汇总表格
    print_summary_table(comparison_data)

    # 打印详细模块分析
    print_detailed_module_analysis(comparison_data)

    # 保存到CSV
    save_to_csv(comparison_data, args.output)

    print("\n" + "="*100)
    print("分析完成！")
    print("="*100)


if __name__ == "__main__":
    main()
