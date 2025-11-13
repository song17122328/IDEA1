#!/usr/bin/env python3
"""
评估剪枝后未微调的模型（用于诊断）
"""
import torch
import argparse
from LLMPruner.evaluator.ppl import PPLMetric

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pruned_model', type=str, required=True,
                      help='剪枝后的模型路径')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--max_seq_len', type=int, default=128)
    args = parser.parse_args()

    print("=" * 80)
    print("评估剪枝后未微调的模型")
    print("=" * 80)

    # 加载剪枝后的模型
    print(f"\n1. 加载剪枝后的模型: {args.pruned_model}")
    pruned_dict = torch.load(args.pruned_model, map_location='cpu', weights_only=False)
    model = pruned_dict['model']
    tokenizer = pruned_dict['tokenizer']
    print("✅ 模型加载完成")

    # 修正配置
    print(f"\n2. 修正模型配置...")
    head_dim = 128
    for i, layer in enumerate(model.model.layers):
        layer_q = layer.self_attn.q_proj.weight.shape[0] // head_dim
        layer_kv = layer.self_attn.k_proj.weight.shape[0] // head_dim

        layer.self_attn.num_heads = layer_q
        layer.self_attn.num_key_value_heads = layer_kv
        layer.self_attn.num_key_value_groups = layer_q // layer_kv
    print("✅ 配置修正完成")

    # 移动到设备
    print(f"\n3. 移动模型到 {args.device}")
    model = model.to(args.device)
    model.eval()

    # 评估 PPL
    print("\n4. 开始评估困惑度...")
    print("-" * 80)
    ppl_results = PPLMetric(
        model,
        tokenizer,
        ['wikitext2', 'ptb'],
        args.max_seq_len,
        device=args.device
    )

    print("\n" + "=" * 80)
    print("评估结果（剪枝后，未微调）:")
    print("=" * 80)
    for dataset, ppl in ppl_results.items():
        print(f"  {dataset}: {ppl:.2f}")
    print("=" * 80)

if __name__ == "__main__":
    main()
