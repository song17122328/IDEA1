#!/usr/bin/env python3
"""
对比评估原始模型、剪枝模型和微调模型的PPL

使用相同的数据集（wikitext2）公平对比三个模型的性能
"""

import os
import sys
import argparse
import torch
from transformers import AutoTokenizer, LlamaForCausalLM

from LLMPruner.evaluator.ppl import PPLMetric


def load_original_model(model_path, device):
    """加载原始未剪枝的模型"""
    print("\n" + "=" * 80)
    print("1. 加载原始模型...")
    print("=" * 80)

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = LlamaForCausalLM.from_pretrained(
        model_path,
        device_map=device,
        torch_dtype=torch.float16,
    )
    model.half()
    model.eval()

    num_params = sum(p.numel() for p in model.parameters())
    print(f"✅ 原始模型加载完成")
    print(f"   参数量: {num_params:,}")

    return tokenizer, model


def load_pruned_model(pruned_path, device):
    """加载剪枝后的模型"""
    print("\n" + "=" * 80)
    print("2. 加载剪枝模型...")
    print("=" * 80)

    pruned_dict = torch.load(pruned_path, map_location='cpu', weights_only=False)
    tokenizer = pruned_dict['tokenizer']
    model = pruned_dict['model']

    model.to(device)
    model.half()
    model.eval()

    num_params = sum(p.numel() for p in model.parameters())
    print(f"✅ 剪枝模型加载完成")
    print(f"   参数量: {num_params:,}")

    # 显示各层配置
    print(f"\n   各层Attention配置:")
    for idx, layer in enumerate(model.model.layers):
        q_heads = layer.self_attn.num_heads
        kv_heads = layer.self_attn.num_key_value_heads
        ratio = q_heads // kv_heads
        if idx < 3 or idx >= len(model.model.layers) - 2:  # 只显示前3层和后2层
            print(f"     Layer {idx}: Q={q_heads}, KV={kv_heads}, ratio={ratio}:1")
        elif idx == 3:
            print(f"     ...")

    return tokenizer, model


def load_finetuned_model_from_lora(pruned_path, finetuned_dir, device):
    """从剪枝模型加载并应用LoRA权重"""
    from LLMPruner.peft import PeftModel

    # 加载剪枝模型
    pruned_dict = torch.load(pruned_path, map_location='cpu', weights_only=False)
    tokenizer = pruned_dict['tokenizer']
    base_model = pruned_dict['model']

    # 修正配置
    head_dim = 128
    print(f"   修正剪枝模型的层配置...")
    for i, layer in enumerate(base_model.model.layers):
        layer_q = layer.self_attn.q_proj.weight.shape[0] // head_dim
        layer_kv = layer.self_attn.k_proj.weight.shape[0] // head_dim

        layer.self_attn.num_heads = layer_q
        layer.self_attn.num_key_value_heads = layer_kv
        layer.self_attn.num_key_value_groups = layer_q // layer_kv

    # 移动到设备
    base_model.to(device)
    base_model.half()

    # 加载LoRA权重
    print(f"   加载LoRA权重从: {finetuned_dir}")
    model = PeftModel.from_pretrained(base_model, finetuned_dir)

    # 合并LoRA权重
    print(f"   合并LoRA权重...")
    model = model.merge_and_unload()

    model.eval()

    num_params = sum(p.numel() for p in model.parameters())
    print(f"✅ 微调模型加载完成（剪枝模型+LoRA）")
    print(f"   参数量: {num_params:,}")

    return tokenizer, model


def load_finetuned_model(pruned_path, finetuned_dir, device):
    """加载微调后的模型（合并后的完整模型或剪枝模型+LoRA）"""
    print("\n" + "=" * 80)
    print("3. 加载微调模型...")
    print("=" * 80)

    # 尝试加载合并后的模型
    merged_model_path = os.path.join(finetuned_dir, "merged_model")

    if os.path.exists(merged_model_path):
        print(f"   尝试从合并模型加载: {merged_model_path}")
        try:
            tokenizer = AutoTokenizer.from_pretrained(merged_model_path)
            model = LlamaForCausalLM.from_pretrained(
                merged_model_path,
                device_map=device,
                torch_dtype=torch.float16,
            )
            print(f"   ✅ 成功从合并模型加载")
        except Exception as e:
            print(f"   ⚠️  合并模型加载失败: {e}")
            print(f"   回退到加载剪枝模型+LoRA权重...")
            return load_finetuned_model_from_lora(pruned_path, finetuned_dir, device)
    else:
        print(f"   未找到合并模型，加载剪枝模型+LoRA权重...")
        return load_finetuned_model_from_lora(pruned_path, finetuned_dir, device)

    model.to(device)
    model.half()
    model.eval()

    num_params = sum(p.numel() for p in model.parameters())
    print(f"✅ 微调模型加载完成")
    print(f"   参数量: {num_params:,}")

    return tokenizer, model


def evaluate_ppl(model, tokenizer, datasets, seq_len, device):
    """评估模型的PPL"""
    ppl_metric = PPLMetric(
        model=model,
        tokenizer=tokenizer,
        datasets=datasets,
        seq_len=seq_len,
        device=device
    )

    return ppl_metric


def main():
    parser = argparse.ArgumentParser(description='对比评估原始/剪枝/微调模型的PPL')

    parser.add_argument('--original_model', type=str,
                       default='/newdata/LLMs/Llama-3-8B-Instruct',
                       help='原始模型路径')
    parser.add_argument('--pruned_model', type=str,
                       default='prune_log/llama3_gqa_aware_pruned_v3/pytorch_model.bin',
                       help='剪枝模型路径')
    parser.add_argument('--finetuned_dir', type=str,
                       default='./finetuned_llama3_gqa_aware_v3',
                       help='微调模型目录（包含LoRA权重或合并后的模型）')

    parser.add_argument('--datasets', type=str, nargs='+',
                       default=['wikitext2'],
                       help='评估数据集列表')
    parser.add_argument('--seq_len', type=int, default=128,
                       help='序列长度')
    parser.add_argument('--device', type=str, default='cuda:0',
                       help='使用的设备')

    args = parser.parse_args()

    print("=" * 80)
    print("GQA-Aware剪枝模型 PPL 对比评估")
    print("=" * 80)
    print(f"\n配置:")
    print(f"  原始模型: {args.original_model}")
    print(f"  剪枝模型: {args.pruned_model}")
    print(f"  微调模型: {args.finetuned_dir}")
    print(f"  评估数据集: {args.datasets}")
    print(f"  序列长度: {args.seq_len}")
    print(f"  设备: {args.device}")

    results = {}

    # ==================== 评估原始模型 ====================
    if os.path.exists(args.original_model):
        try:
            tokenizer, original_model = load_original_model(args.original_model, args.device)

            print("\n" + "=" * 80)
            print("评估原始模型 PPL...")
            print("=" * 80)

            original_ppl = evaluate_ppl(original_model, tokenizer, args.datasets, args.seq_len, args.device)
            results['original'] = {
                'ppl': original_ppl,
                'params': sum(p.numel() for p in original_model.parameters())
            }

            print(f"\n✅ 原始模型 PPL: {original_ppl}")

            # 清理内存
            del original_model
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"\n❌ 原始模型评估失败: {e}")
            results['original'] = {'ppl': None, 'params': None}
    else:
        print(f"\n⚠️  跳过原始模型评估（路径不存在）: {args.original_model}")
        results['original'] = {'ppl': None, 'params': None}

    # ==================== 评估剪枝模型 ====================
    if os.path.exists(args.pruned_model):
        try:
            tokenizer, pruned_model = load_pruned_model(args.pruned_model, args.device)

            print("\n" + "=" * 80)
            print("评估剪枝模型 PPL...")
            print("=" * 80)

            pruned_ppl = evaluate_ppl(pruned_model, tokenizer, args.datasets, args.seq_len, args.device)
            results['pruned'] = {
                'ppl': pruned_ppl,
                'params': sum(p.numel() for p in pruned_model.parameters())
            }

            print(f"\n✅ 剪枝模型 PPL: {pruned_ppl}")

            # 清理内存
            del pruned_model
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"\n❌ 剪枝模型评估失败: {e}")
            import traceback
            traceback.print_exc()
            results['pruned'] = {'ppl': None, 'params': None}
    else:
        print(f"\n⚠️  跳过剪枝模型评估（路径不存在）: {args.pruned_model}")
        results['pruned'] = {'ppl': None, 'params': None}

    # ==================== 评估微调模型 ====================
    if os.path.exists(args.finetuned_dir):
        try:
            tokenizer, finetuned_model = load_finetuned_model(
                args.pruned_model, args.finetuned_dir, args.device
            )

            print("\n" + "=" * 80)
            print("评估微调模型 PPL...")
            print("=" * 80)

            finetuned_ppl = evaluate_ppl(finetuned_model, tokenizer, args.datasets, args.seq_len, args.device)
            results['finetuned'] = {
                'ppl': finetuned_ppl,
                'params': sum(p.numel() for p in finetuned_model.parameters())
            }

            print(f"\n✅ 微调模型 PPL: {finetuned_ppl}")

            # 清理内存
            del finetuned_model
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"\n❌ 微调模型评估失败: {e}")
            import traceback
            traceback.print_exc()
            results['finetuned'] = {'ppl': None, 'params': None}
    else:
        print(f"\n⚠️  跳过微调模型评估（路径不存在）: {args.finetuned_dir}")
        results['finetuned'] = {'ppl': None, 'params': None}

    # ==================== 输出对比结果 ====================
    print("\n" + "=" * 80)
    print("PPL 对比结果")
    print("=" * 80)

    # 提取wikitext2的PPL值
    def extract_ppl(ppl_dict):
        if ppl_dict is None:
            return None
        if isinstance(ppl_dict, dict):
            for key, value in ppl_dict.items():
                if 'wikitext2' in key.lower():
                    return value
        return ppl_dict

    original_ppl_value = extract_ppl(results['original']['ppl'])
    pruned_ppl_value = extract_ppl(results['pruned']['ppl'])
    finetuned_ppl_value = extract_ppl(results['finetuned']['ppl'])

    # 打印表格
    print(f"\n{'模型':<15} {'参数量':<20} {'PPL (wikitext2)':<20} {'vs 原始':<15}")
    print("-" * 80)

    # 原始模型
    if results['original']['params'] is not None:
        print(f"{'原始模型':<15} {results['original']['params']:>19,} {original_ppl_value:>19.2f} {'-':<15}")
    else:
        print(f"{'原始模型':<15} {'N/A':<20} {'N/A':<20} {'-':<15}")

    # 剪枝模型
    if results['pruned']['params'] is not None:
        param_reduction = (1 - results['pruned']['params'] / results['original']['params']) * 100 if results['original']['params'] else 0
        ppl_change = pruned_ppl_value / original_ppl_value if original_ppl_value else 0
        print(f"{'剪枝模型':<15} {results['pruned']['params']:>19,} {pruned_ppl_value:>19.2f} {f'{ppl_change:.2f}×':<15}")
        print(f"{'  (减少)':<15} {f'-{param_reduction:.1f}%':<20}")
    else:
        print(f"{'剪枝模型':<15} {'N/A':<20} {'N/A':<20} {'N/A':<15}")

    # 微调模型
    if results['finetuned']['params'] is not None:
        param_reduction = (1 - results['finetuned']['params'] / results['original']['params']) * 100 if results['original']['params'] else 0
        ppl_change = finetuned_ppl_value / original_ppl_value if original_ppl_value else 0
        ppl_recovery = (pruned_ppl_value - finetuned_ppl_value) / (pruned_ppl_value - original_ppl_value) * 100 if pruned_ppl_value and original_ppl_value else 0
        print(f"{'微调模型':<15} {results['finetuned']['params']:>19,} {finetuned_ppl_value:>19.2f} {f'{ppl_change:.2f}×':<15}")
        print(f"{'  (减少)':<15} {f'-{param_reduction:.1f}%':<20}")
        if pruned_ppl_value and original_ppl_value:
            print(f"{'  (恢复度)':<15} {f'{ppl_recovery:.1f}%':<20}")
    else:
        print(f"{'微调模型':<15} {'N/A':<20} {'N/A':<20} {'N/A':<15}")

    print("\n" + "=" * 80)

    # 计算改善
    if original_ppl_value and pruned_ppl_value and finetuned_ppl_value:
        print("\n关键指标:")
        print(f"  剪枝后PPL退化: {pruned_ppl_value/original_ppl_value:.2f}× (从 {original_ppl_value:.2f} 到 {pruned_ppl_value:.2f})")
        print(f"  微调后PPL恢复: {finetuned_ppl_value/original_ppl_value:.2f}× (从 {original_ppl_value:.2f} 到 {finetuned_ppl_value:.2f})")
        print(f"  微调改善: {pruned_ppl_value/finetuned_ppl_value:.2f}× (从 {pruned_ppl_value:.2f} 到 {finetuned_ppl_value:.2f})")

        if results['original']['params'] and results['finetuned']['params']:
            param_reduction = (1 - results['finetuned']['params'] / results['original']['params']) * 100
            speedup = 1 / (1 - param_reduction/100)
            print(f"\n  参数减少: {param_reduction:.1f}%")
            print(f"  预期加速: {speedup:.1f}×")

            # 判断成功标准
            print(f"\n成功标准:")
            if finetuned_ppl_value < 25:
                print(f"  ✅ 微调后PPL < 25: {finetuned_ppl_value:.2f}")
            elif finetuned_ppl_value < 30:
                print(f"  ⚠️  微调后PPL < 30: {finetuned_ppl_value:.2f} (可接受)")
            else:
                print(f"  ❌ 微调后PPL >= 30: {finetuned_ppl_value:.2f} (需要改进)")

            if param_reduction > 80:
                print(f"  ✅ 参数减少 > 80%: {param_reduction:.1f}%")
            elif param_reduction > 60:
                print(f"  ⚠️  参数减少 > 60%: {param_reduction:.1f}% (良好)")
            else:
                print(f"  ❌ 参数减少 < 60%: {param_reduction:.1f}% (偏低)")

    print("\n" + "=" * 80)
    print("评估完成!")
    print("=" * 80)

    # 保存结果到JSON
    import json
    output_file = "ppl_comparison_results.json"

    # 准备可序列化的结果
    serializable_results = {}
    for model_name, data in results.items():
        serializable_results[model_name] = {
            'params': data['params'],
            'ppl': extract_ppl(data['ppl'])
        }

    with open(output_file, 'w') as f:
        json.dump(serializable_results, f, indent=2)

    print(f"\n结果已保存到: {output_file}")


if __name__ == "__main__":
    main()
