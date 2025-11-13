#!/usr/bin/env python3
"""
直接评估 LoRA 微调后的模型（无需合并）
"""
import torch
import argparse
from peft import PeftModel
from LLMPruner.evaluator.ppl import PPLMetric

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pruned_model', type=str, required=True,
                      help='原始剪枝模型路径')
    parser.add_argument('--lora_dir', type=str, required=True,
                      help='LoRA adapter 目录')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--max_seq_len', type=int, default=128)
    args = parser.parse_args()

    print("=" * 80)
    print("评估 LoRA 微调后的模型")
    print("=" * 80)

    # 1. 加载剪枝后的基础模型
    print(f"\n1. 加载剪枝后的基础模型: {args.pruned_model}")
    pruned_dict = torch.load(args.pruned_model, map_location='cpu', weights_only=False)
    base_model = pruned_dict['model']
    tokenizer = pruned_dict['tokenizer']
    print("✅ 基础模型加载完成")

    # 2. 加载 LoRA adapter
    print(f"\n2. 加载 LoRA adapter: {args.lora_dir}")
    model = PeftModel.from_pretrained(base_model, args.lora_dir)
    print("✅ LoRA adapter 加载完成")

    # 3. 移动到指定设备
    print(f"\n3. 移动模型到 {args.device}")
    model = model.to(args.device)
    model.eval()

    # 4. 评估 PPL
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
    print("评估结果:")
    print("=" * 80)
    for dataset, ppl in ppl_results.items():
        print(f"  {dataset}: {ppl:.2f}")
    print("=" * 80)

if __name__ == "__main__":
    main()
