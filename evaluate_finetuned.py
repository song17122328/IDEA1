#!/usr/bin/env python3
"""
评估微调后的模型
"""
import torch
import argparse
from transformers import AutoTokenizer
from LLMPruner.evaluator.ppl import PPLMetric

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True,
                      help='微调后的模型路径')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--max_seq_len', type=int, default=128)
    args = parser.parse_args()

    print("=" * 80)
    print("评估微调后的模型")
    print("=" * 80)

    # 加载模型
    print(f"\n加载模型: {args.model_path}")
    checkpoint = torch.load(args.model_path + '/pytorch_model.bin',
                           map_location='cpu', weights_only=False)
    model = checkpoint['model']
    tokenizer = checkpoint['tokenizer']

    model.to(args.device)
    model.eval()

    # 评估 PPL
    print("\n开始评估困惑度...")
    ppl_results = PPLMetric(
        model,
        tokenizer,
        ['wikitext2', 'ptb'],
        args.max_seq_len,
        device=args.device
    )

    print("\n" + "=" * 80)
    print("评估结果:")
    print("=" * 80)
    for dataset, ppl in ppl_results.items():
        print(f"{dataset}: {ppl:.2f}")
    print("=" * 80)

if __name__ == "__main__":
    main()
