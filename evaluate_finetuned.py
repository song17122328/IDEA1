#!/usr/bin/env python3
"""
评估微调后的模型
"""
import torch
import argparse
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
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

    # 加载模型 - 从 HuggingFace 格式的 merged_model 目录
    merged_model_path = os.path.join(args.model_path, 'merged_model')

    if not os.path.exists(merged_model_path):
        print(f"❌ 错误：找不到合并后的模型目录: {merged_model_path}")
        print("   请确保训练已完成并成功保存了 merged_model")
        return

    print(f"\n加载模型: {merged_model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        merged_model_path,
        torch_dtype=torch.float16,
        device_map=args.device
    )
    tokenizer = AutoTokenizer.from_pretrained(merged_model_path)

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
