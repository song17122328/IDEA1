"""
多层GQA-Aware Pruning测试脚本

快速验证GQA-aware方法在多层剪枝时的效果

使用方法:
python test_multi_layer_gqa_pruning.py \
    --layers 5,10,15,20,25 \
    --pruning_rate 0.25 \
    --device cuda:0
"""

import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from gqa_aware_pruning import (
    compute_gqa_group_importance,
    select_gqa_groups_to_prune,
    prune_attention_by_gqa_groups
)


def get_example_prompts(tokenizer, device, num_examples=10, seq_len=64):
    """生成测试用的prompts"""
    texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is a subset of artificial intelligence.",
        "Python is a popular programming language.",
        "The capital of France is Paris.",
        "Deep learning models require large amounts of data.",
        "Natural language processing is a challenging task.",
        "Neural networks are inspired by biological neurons.",
        "Transformers have revolutionized NLP.",
        "Attention mechanisms are key to modern models.",
        "Large language models can generate human-like text."
    ][:num_examples]

    inputs = tokenizer(texts, return_tensors='pt', padding=True, truncation=True, max_length=seq_len)
    return inputs.input_ids.to(device)


def evaluate_ppl(model, tokenizer, device):
    """评估困惑度"""
    from LLMPruner.evaluator.ppl import PPLMetric

    try:
        ppl_dict = PPLMetric(model, tokenizer, ['wikitext2'], seq_len=128, device=device)
        ppl_value = list(ppl_dict.values())[0] if ppl_dict else None
        return ppl_value
    except Exception as e:
        print(f"PPL评估失败: {e}")
        return None


def prune_single_layer(model, layer_idx, pruning_rate, example_prompts, head_dim=128, gqa_ratio=4):
    """剪枝单个层"""
    layer = model.model.layers[layer_idx]

    # 1. 计算GQA组的importance
    group_imp = compute_gqa_group_importance(layer, head_dim, gqa_ratio)

    # 2. 确定要保留的GQA组数量
    num_kv_heads = len(group_imp)
    num_groups_to_prune = int(num_kv_heads * pruning_rate)
    target_num_kv_heads = num_kv_heads - num_groups_to_prune
    target_num_kv_heads = max(1, target_num_kv_heads)  # 至少保留1个组

    # 3. 选择要保留的组
    keep_indices, prune_indices = select_gqa_groups_to_prune(group_imp, target_num_kv_heads)

    # 4. 执行剪枝
    num_q, num_kv = prune_attention_by_gqa_groups(layer, keep_indices, head_dim, gqa_ratio)

    return num_q, num_kv, group_imp, keep_indices, prune_indices


def test_multi_layer_pruning(args):
    """测试多层GQA-aware剪枝"""

    print("=" * 80)
    print("多层 GQA-Aware Pruning 测试")
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

    # 2. 评估原始PPL
    print("\n2. 评估原始模型PPL...")
    original_ppl = evaluate_ppl(model, tokenizer, args.device)
    if original_ppl is not None:
        print(f"原始PPL: {original_ppl:.2f}")
    else:
        print(f"原始PPL: 评估失败")
        return

    # 3. 准备测试数据
    print("\n3. 准备测试数据...")
    example_prompts = get_example_prompts(tokenizer, args.device)
    print(f"测试数据: {example_prompts.shape}")

    # 4. 剪枝多个层
    print(f"\n4. 剪枝多个层...")
    print(f"目标层: {args.layers}")
    print(f"剪枝率: {args.pruning_rate:.2%}")
    print("-" * 80)

    layer_results = {}

    for layer_idx in args.layers:
        print(f"\n处理 Layer {layer_idx}...")

        # 前向+反向传播计算梯度
        model.zero_grad()
        loss = model(example_prompts, labels=example_prompts).loss
        print(f"  Loss: {loss.item():.4f}")
        loss.backward()

        # 剪枝该层
        num_q, num_kv, group_imp, keep_indices, prune_indices = prune_single_layer(
            model, layer_idx, args.pruning_rate, example_prompts
        )

        layer_results[layer_idx] = {
            'num_q': num_q,
            'num_kv': num_kv,
            'group_imp': group_imp,
            'keep_indices': keep_indices,
            'prune_indices': prune_indices
        }

        print(f"  剪枝前: 32 Q heads, 8 KV heads")
        print(f"  剪枝后: {num_q} Q heads, {num_kv} KV heads")
        print(f"  保留组: {keep_indices}")
        print(f"  剪枝组: {prune_indices}")
        print(f"  GQA比例: {num_q//num_kv}:1 ✓")

    print("\n" + "-" * 80)

    # 5. 验证模型forward
    print("\n5. 验证模型forward...")
    try:
        model.zero_grad()
        output = model(example_prompts)
        print(f"✅ Forward成功! Output shape: {output.logits.shape}")

        loss = model(example_prompts, labels=example_prompts).loss
        print(f"剪枝后Loss: {loss.item():.4f}")
    except Exception as e:
        print(f"❌ Forward失败: {e}")
        return

    # 6. 评估剪枝后PPL
    print("\n6. 评估剪枝后PPL...")
    pruned_ppl = evaluate_ppl(model, tokenizer, args.device)

    if pruned_ppl is not None:
        print(f"剪枝后PPL: {pruned_ppl:.2f}")
        print(f"PPL变化: {original_ppl:.2f} → {pruned_ppl:.2f} (×{pruned_ppl/original_ppl:.2f})")
    else:
        print(f"剪枝后PPL: 评估失败")

    # 7. 总结
    print("\n" + "=" * 80)
    print("测试总结")
    print("=" * 80)
    print(f"剪枝层数: {len(args.layers)}")
    print(f"剪枝率: {args.pruning_rate:.2%}")
    print(f"\n逐层详情:")
    print(f"{'Layer':<10} {'原始':<15} {'剪枝后':<15} {'比例':<10}")
    print("-" * 80)

    for layer_idx in args.layers:
        result = layer_results[layer_idx]
        print(f"{layer_idx:<10} 32Q:8KV         {result['num_q']}Q:{result['num_kv']}KV         {result['num_q']//result['num_kv']}:1")

    print("-" * 80)
    print(f"\nPPL对比:")
    print(f"  原始模型: {original_ppl:.2f}")
    print(f"  剪枝后:   {pruned_ppl:.2f}")
    print(f"  变化:     {((pruned_ppl/original_ppl - 1) * 100):+.2f}%")

    # 8. 计算参数减少量
    total_layers = len(model.model.layers)
    pruned_layers = len(args.layers)
    avg_kv_reduction = sum([1 - result['num_kv']/8 for result in layer_results.values()]) / pruned_layers

    # 估算attention参数减少（粗略）
    # Q, K, V, O projections
    attention_params_per_layer = (4096 * 4096 * 4)  # q_proj, k_proj, v_proj, o_proj
    attention_reduction_per_layer = avg_kv_reduction * 0.5  # 粗略估计

    total_attention_reduction = attention_reduction_per_layer * pruned_layers / total_layers

    print(f"\n估算参数减少:")
    print(f"  剪枝 {pruned_layers}/{total_layers} 层")
    print(f"  平均KV head减少: {avg_kv_reduction:.1%}")
    print(f"  估算Attention参数减少: ~{total_attention_reduction:.1%}")

    print("\n" + "=" * 80)
    print("测试完成!")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description='Test multi-layer GQA-aware pruning')
    parser.add_argument('--layers', type=str, default='5,10,15,20,25',
                        help='要剪枝的层（逗号分隔），default: 5,10,15,20,25')
    parser.add_argument('--pruning_rate', type=float, default=0.25,
                        help='剪枝率 (default: 0.25)')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='设备 (default: cuda:0)')

    args = parser.parse_args()

    # 解析层索引
    args.layers = [int(x.strip()) for x in args.layers.split(',')]

    test_multi_layer_pruning(args)


if __name__ == '__main__':
    main()
