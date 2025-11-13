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

    # 加载模型 - 使用剪枝后的模型结构，加载合并后的权重
    merged_model_path = os.path.join(args.model_path, 'merged_model')

    if not os.path.exists(merged_model_path):
        print(f"❌ 错误：找不到合并后的模型目录: {merged_model_path}")
        print("   请确保训练已完成并成功保存了 merged_model")
        return

    print(f"\n加载模型: {merged_model_path}")
    print("   策略: 使用剪枝后的模型结构 + 微调后的权重")

    # 从 merged_model 加载权重
    from safetensors.torch import load_file
    import glob

    # 查找权重文件（可能是 safetensors 或 pytorch 格式）
    safetensor_files = glob.glob(os.path.join(merged_model_path, "*.safetensors"))

    if safetensor_files:
        print(f"   找到 {len(safetensor_files)} 个 safetensors 文件")
        state_dict = {}
        for shard_file in sorted(safetensor_files):
            print(f"   加载: {os.path.basename(shard_file)}")
            state_dict.update(load_file(shard_file))
    else:
        print("   未找到 safetensors 文件，尝试加载 pytorch_model.bin")
        pytorch_file = os.path.join(merged_model_path, "pytorch_model.bin")
        state_dict = torch.load(pytorch_file, map_location='cpu', weights_only=False)

    # 直接加载为完整模型字典（假设这是完整的模型保存格式）
    # 如果 state_dict 包含 'model' 和 'tokenizer' 键，提取它们
    if 'model' in state_dict:
        model = state_dict['model']
        if 'tokenizer' in state_dict:
            tokenizer = state_dict['tokenizer']
        else:
            tokenizer = AutoTokenizer.from_pretrained(merged_model_path)
    else:
        # 否则，这是一个 state_dict，需要加载到模型结构中
        # 这种情况下，我们需要从原始剪枝模型加载结构
        print("   检测到 state_dict 格式，需要原始模型结构...")
        print("   错误：无法从 state_dict 重建剪枝后的模型结构")
        print("   建议：使用原始的评估方法或等待修复")
        return

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
