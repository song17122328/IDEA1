#!/usr/bin/env python3
"""
检查剪枝后模型的实际配置
"""
import torch

# 加载剪枝模型
pruned_dict = torch.load('prune_log/llama_unbalanced_prune_all_layers/pytorch_model.bin',
                         map_location='cpu', weights_only=False)
model = pruned_dict['model']

print("=" * 80)
print("检查剪枝后模型的配置")
print("=" * 80)

# 检查模型 config
print("\n模型 config:")
print(f"  num_attention_heads: {model.config.num_attention_heads}")
print(f"  num_key_value_heads: {model.config.num_key_value_heads}")
print(f"  hidden_size: {model.config.hidden_size}")
print(f"  num_hidden_layers: {model.config.num_hidden_layers}")

# 检查每层的实际维度
print("\n" + "=" * 80)
print("各层实际的投影维度:")
print("=" * 80)
print(f"{'Layer':<10} {'Q Out':<12} {'K Out':<12} {'V Out':<12} {'O In':<12} {'Q Heads':<12} {'KV Heads':<12}")
print("-" * 80)

head_dim = 128  # Llama-3 的 head dimension

for i, layer in enumerate(model.model.layers):
    q_out = layer.self_attn.q_proj.weight.shape[0]
    k_out = layer.self_attn.k_proj.weight.shape[0]
    v_out = layer.self_attn.v_proj.weight.shape[0]
    o_in = layer.self_attn.o_proj.weight.shape[1]

    num_q_heads = q_out // head_dim
    num_kv_heads = k_out // head_dim

    print(f"Layer {i:<5} {q_out:<12} {k_out:<12} {v_out:<12} {o_in:<12} {num_q_heads:<12} {num_kv_heads:<12}")

print("=" * 80)

# 统计不同的配置
layer_configs = {}
for i, layer in enumerate(model.model.layers):
    q_out = layer.self_attn.q_proj.weight.shape[0]
    k_out = layer.self_attn.k_proj.weight.shape[0]

    num_q_heads = q_out // head_dim
    num_kv_heads = k_out // head_dim
    ratio = num_q_heads // num_kv_heads if num_kv_heads > 0 else 0

    config_key = (num_q_heads, num_kv_heads, ratio)
    if config_key not in layer_configs:
        layer_configs[config_key] = []
    layer_configs[config_key].append(i)

print("\n配置分布:")
for (num_q, num_kv, ratio), layers in sorted(layer_configs.items()):
    print(f"  {num_q} Q heads, {num_kv} KV heads (ratio {ratio}:1) - Layers: {layers}")

print("=" * 80)
