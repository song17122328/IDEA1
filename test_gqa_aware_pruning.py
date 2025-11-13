"""
GQA-Aware Pruning 测试脚本

验证新方法是否能改善剪枝后的PPL

使用方法:
python test_gqa_aware_pruning.py --layer_idx 10 --pruning_rate 0.25
"""

import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from gqa_aware_pruning import (
    compute_gqa_group_importance,
    select_gqa_groups_to_prune,
    prune_attention_by_gqa_groups
)

def get_example_prompts(tokenizer, device, num_examples=5, seq_len=64):
    """生成测试用的prompts"""
    texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is a subset of artificial intelligence.",
        "Python is a popular programming language.",
        "The capital of France is Paris.",
        "Deep learning models require large amounts of data."
    ][:num_examples]

    inputs = tokenizer(texts, return_tensors='pt', padding=True, truncation=True, max_length=seq_len)
    return inputs.input_ids.to(device)


def evaluate_ppl(model, tokenizer, device):
    """简单评估困惑度"""
    from LLMPruner.evaluator.ppl import PPLMetric

    try:
        ppl = PPLMetric(model, tokenizer, ['wikitext2'], max_seq_len=128, device=device)
        return ppl['wikitext2']
    except Exception as e:
        print(f"PPL评估失败: {e}")
        return None


def test_single_layer_pruning(args):
    """测试单层GQA-aware剪枝"""

    print("=" * 80)
    print("GQA-Aware Pruning 单层测试")
    print("=" * 80)

    # 1. 加载模型
    print("\n1. 加载模型...")
    model_path = "/newdata/LLMs/Llama-3-8B-Instruct"
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map=args.device
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token

    print(f"✅ 模型加载完成")

    # 2. 评估原始PPL（可选）
    if args.eval_ppl:
        print("\n2. 评估原始模型PPL...")
        original_ppl = evaluate_ppl(model, tokenizer, args.device)
        print(f"原始PPL: {original_ppl:.2f}")

    # 3. 准备测试数据
    print("\n3. 准备测试数据...")
    example_prompts = get_example_prompts(tokenizer, args.device)
    print(f"测试数据: {example_prompts.shape}")

    # 4. 记录原始配置
    layer = model.model.layers[args.layer_idx]
    original_q_heads = layer.self_attn.num_heads
    original_kv_heads = layer.self_attn.num_key_value_heads

    print(f"\n4. Layer {args.layer_idx} 原始配置:")
    print(f"   Q heads: {original_q_heads}")
    print(f"   KV heads: {original_kv_heads}")
    print(f"   Ratio: {original_q_heads // original_kv_heads}:1")

    # 5. 计算GQA组的importance
    print(f"\n5. 计算GQA组importance...")
    model.zero_grad()
    loss = model(example_prompts, labels=example_prompts).loss
    print(f"   Loss: {loss.item():.4f}")
    loss.backward()

    group_imp = compute_gqa_group_importance(layer, head_dim=128, gqa_ratio=4)
    print(f"   GQA组importance: {group_imp}")
    print(f"   最高importance组: {group_imp.argmax().item()}")
    print(f"   最低importance组: {group_imp.argmin().item()}")

    # 6. 确定要剪枝的组
    num_groups_to_prune = int(original_kv_heads * args.pruning_rate)
    target_num_kv_heads = original_kv_heads - num_groups_to_prune

    print(f"\n6. 剪枝策略:")
    print(f"   剪枝率: {args.pruning_rate:.2%}")
    print(f"   剪枝 {num_groups_to_prune} 个GQA组")
    print(f"   保留 {target_num_kv_heads} 个GQA组")

    keep_indices, prune_indices = select_gqa_groups_to_prune(group_imp, target_num_kv_heads)
    print(f"   保留组索引: {keep_indices}")
    print(f"   剪枝组索引: {prune_indices}")

    # 显示被剪枝组的importance
    print(f"\n   被剪枝组的importance:")
    for idx in prune_indices:
        print(f"     组 {idx}: {group_imp[idx].item():.4f}")
    print(f"   保留组的importance:")
    for idx in keep_indices[:3]:  # 只显示前3个
        print(f"     组 {idx}: {group_imp[idx].item():.4f}")

    # 7. 执行剪枝
    print(f"\n7. 执行GQA-aware剪枝...")
    num_q, num_kv = prune_attention_by_gqa_groups(layer, keep_indices, head_dim=128, gqa_ratio=4)

    print(f"   ✅ 剪枝完成!")
    print(f"   剪枝后: Q heads={num_q}, KV heads={num_kv}, ratio={num_q//num_kv}:1")

    # 8. 验证模型forward
    print(f"\n8. 验证模型forward...")
    model.zero_grad()
    try:
        output = model(example_prompts)
        print(f"   ✅ Forward成功! Output shape: {output.logits.shape}")

        # 计算loss
        loss = model(example_prompts, labels=example_prompts).loss
        print(f"   剪枝后Loss: {loss.item():.4f}")
    except Exception as e:
        print(f"   ❌ Forward失败: {e}")
        return

    # 9. 评估剪枝后PPL（可选）
    if args.eval_ppl:
        print("\n9. 评估剪枝后PPL...")
        pruned_ppl = evaluate_ppl(model, tokenizer, args.device)
        print(f"   剪枝后PPL: {pruned_ppl:.2f}")
        if original_ppl:
            print(f"   PPL变化: {original_ppl:.2f} → {pruned_ppl:.2f} (×{pruned_ppl/original_ppl:.2f})")

    print("\n" + "=" * 80)
    print("测试完成!")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description='Test GQA-aware pruning')
    parser.add_argument('--layer_idx', type=int, default=10,
                        help='要测试的层索引 (default: 10)')
    parser.add_argument('--pruning_rate', type=float, default=0.25,
                        help='剪枝率 (default: 0.25)')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='设备 (default: cuda:0)')
    parser.add_argument('--eval_ppl', action='store_true',
                        help='是否评估PPL（耗时）')

    args = parser.parse_args()

    test_single_layer_pruning(args)


if __name__ == '__main__':
    main()
