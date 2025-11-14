#!/usr/bin/env python3
"""
检查剪枝模型和微调模型的路径和参数量
"""

import os
import sys
import torch
from pathlib import Path

def check_model_file(model_path):
    """检查模型文件是否存在并显示参数量"""
    print(f"\n检查模型: {model_path}")

    if not os.path.exists(model_path):
        print(f"  ❌ 文件不存在")
        return None

    file_size = os.path.getsize(model_path) / (1024**3)  # GB
    print(f"  ✅ 文件存在")
    print(f"  文件大小: {file_size:.2f} GB")

    try:
        print(f"  正在加载模型...")
        model_dict = torch.load(model_path, map_location='cpu', weights_only=False)

        if 'model' in model_dict:
            model = model_dict['model']
            num_params = sum(p.numel() for p in model.parameters())
            print(f"  ✅ 参数量: {num_params:,}")

            # 检查第一层的配置
            if hasattr(model, 'model') and hasattr(model.model, 'layers'):
                first_layer = model.model.layers[0]
                q_heads = first_layer.self_attn.num_heads
                kv_heads = first_layer.self_attn.num_key_value_heads
                print(f"  Attention配置: Q={q_heads}, KV={kv_heads}, ratio={q_heads//kv_heads}:1")

                # 检查MLP维度
                mlp_dim = first_layer.mlp.gate_proj.weight.shape[0]
                print(f"  MLP中间层维度: {mlp_dim}")

            return num_params
        else:
            print(f"  ⚠️  模型字典中没有'model'键")
            return None
    except Exception as e:
        print(f"  ❌ 加载失败: {e}")
        return None

def main():
    print("=" * 80)
    print("检查剪枝和微调模型")
    print("=" * 80)

    # 可能的剪枝模型路径
    possible_pruned_paths = [
        "prune_log/llama3_gqa_aware_pruned_v3/pytorch_model.bin",
        "prune_log/llama_gqa_aware_prune/pytorch_model.bin",
        "prune_log/llama_unbalanced_prune/pytorch_model.bin",
        "prune_log/llama_unbalanced_prune_v2/pytorch_model.bin",
    ]

    print("\n1. 搜索剪枝模型...")
    print("-" * 80)

    found_models = []
    for path in possible_pruned_paths:
        if os.path.exists(path):
            print(f"\n找到: {path}")
            params = check_model_file(path)
            if params:
                found_models.append((path, params))

    if not found_models:
        print("\n⚠️  没有找到任何剪枝模型")
        print("\n请检查以下路径是否正确，或指定正确的路径：")
        for path in possible_pruned_paths:
            print(f"  - {path}")
    else:
        print("\n" + "=" * 80)
        print("找到的剪枝模型总结:")
        print("-" * 80)
        for path, params in found_models:
            reduction = (1 - params / 8030261248) * 100
            print(f"{path}")
            print(f"  参数量: {params:,} (减少 {reduction:.1f}%)")

            # 判断是否是GQA-aware剪枝的模型
            if reduction > 80:
                print(f"  ✅ 这可能是GQA-aware剪枝的模型（参数减少>80%）")
            elif reduction > 50:
                print(f"  ⚠️  这可能是v2剪枝的模型（参数减少50-80%）")
            else:
                print(f"  ❌ 参数减少较少，可能不是完整剪枝")
            print()

    # 检查微调模型目录
    print("\n2. 检查微调模型目录...")
    print("-" * 80)

    finetuned_dirs = [
        "./finetuned_llama3_gqa_aware_v3",
        "finetuned_llama3_gqa_aware_v3",
    ]

    for ft_dir in finetuned_dirs:
        if os.path.exists(ft_dir):
            print(f"\n找到微调目录: {ft_dir}")

            # 检查是否有合并模型
            merged_path = os.path.join(ft_dir, "merged_model")
            if os.path.exists(merged_path):
                print(f"  ✅ 有合并模型: {merged_path}")
                # 检查模型文件
                model_files = list(Path(merged_path).glob("*.safetensors")) + \
                             list(Path(merged_path).glob("*.bin"))
                if model_files:
                    print(f"  模型文件: {[f.name for f in model_files[:3]]}")
            else:
                print(f"  ⚠️  没有合并模型")

            # 检查是否有LoRA权重
            adapter_files = list(Path(ft_dir).glob("adapter_*.bin")) + \
                           list(Path(ft_dir).glob("adapter_*.safetensors"))
            if adapter_files:
                print(f"  ✅ 有LoRA权重: {[f.name for f in adapter_files[:3]]}")
            else:
                print(f"  ⚠️  没有找到LoRA权重")

    # 检查原始模型
    print("\n3. 检查原始模型...")
    print("-" * 80)

    original_model_path = "/newdata/LLMs/Llama-3-8B-Instruct"
    if os.path.exists(original_model_path):
        print(f"✅ 原始模型存在: {original_model_path}")
    else:
        print(f"❌ 原始模型不存在: {original_model_path}")

    print("\n" + "=" * 80)
    print("检查完成")
    print("=" * 80)

    # 给出建议
    if found_models:
        print("\n建议:")
        print("-" * 80)

        # 找到参数减少最多的模型
        best_model = max(found_models, key=lambda x: (1 - x[1] / 8030261248))
        best_path, best_params = best_model
        best_reduction = (1 - best_params / 8030261248) * 100

        print(f"\n推荐使用以下剪枝模型（参数减少{best_reduction:.1f}%）:")
        print(f"  {best_path}")
        print(f"\n运行PPL对比:")
        print(f"  python compare_model_ppl.py \\")
        print(f"    --pruned_model {best_path} \\")
        print(f"    --finetuned_dir ./finetuned_llama3_gqa_aware_v3 \\")
        print(f"    --device cuda:0")

if __name__ == "__main__":
    main()
