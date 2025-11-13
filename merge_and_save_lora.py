#!/usr/bin/env python3
"""
合并 LoRA 权重并保存为完整模型
用于修复训练后保存失败的模型
"""
import torch
import argparse
import os
from peft import PeftModel
from transformers import AutoTokenizer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pruned_model', type=str, required=True,
                      help='原始剪枝模型路径')
    parser.add_argument('--lora_dir', type=str, required=True,
                      help='LoRA adapter 目录')
    parser.add_argument('--output_dir', type=str, required=True,
                      help='输出目录')
    args = parser.parse_args()

    print("=" * 80)
    print("合并 LoRA 权重并保存完整模型")
    print("=" * 80)

    # 1. 加载剪枝后的基础模型
    print(f"\n1. 加载剪枝后的基础模型: {args.pruned_model}")
    pruned_dict = torch.load(args.pruned_model, map_location='cpu', weights_only=False)
    base_model = pruned_dict['model']
    tokenizer = pruned_dict['tokenizer']
    print("✅ 基础模型加载完成")

    # 2. 加载 LoRA adapter 并合并
    print(f"\n2. 加载 LoRA adapter: {args.lora_dir}")
    model_with_lora = PeftModel.from_pretrained(base_model, args.lora_dir)
    print("✅ LoRA adapter 加载完成")

    print(f"\n3. 合并 LoRA 权重到基础模型...")
    merged_model = model_with_lora.merge_and_unload()
    print("✅ 权重合并完成")

    # 3. 保存合并后的模型
    print(f"\n4. 保存合并后的模型到: {args.output_dir}")
    os.makedirs(args.output_dir, exist_ok=True)

    merged_model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    print(f"✅ 完整模型已保存到: {args.output_dir}")
    print("=" * 80)
    print("\n现在可以使用 evaluate_finetuned.py 评估模型了")
    print(f"命令: python evaluate_finetuned.py --model_path {os.path.dirname(args.output_dir)} --device cuda:0")
    print("=" * 80)

if __name__ == "__main__":
    main()
