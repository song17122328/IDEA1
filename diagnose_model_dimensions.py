#!/usr/bin/env python3
"""
诊断剪枝后模型的所有维度，包括 Attention 和 MLP
"""
import torch
import sys

# 加载剪枝模型
pruned_dict = torch.load('prune_log/llama_unbalanced_prune_all_layers/pytorch_model.bin',
                         map_location='cpu', weights_only=False)
model = pruned_dict['model']

print("=" * 100)
print("剪枝后模型维度诊断")
print("=" * 100)

# 检查模型 config
print(f"\n全局 Config:")
print(f"  num_attention_heads: {model.config.num_attention_heads}")
print(f"  num_key_value_heads: {model.config.num_key_value_heads}")
print(f"  hidden_size: {model.config.hidden_size}")
print(f"  intermediate_size: {model.config.intermediate_size}")

# 检查每层的实际维度
print("\n" + "=" * 100)
print("各层实际维度:")
print("=" * 100)

head_dim = 128

for i, layer in enumerate(model.model.layers):
    # Attention 投影
    q_out = layer.self_attn.q_proj.weight.shape[0]
    k_out = layer.self_attn.k_proj.weight.shape[0]
    v_out = layer.self_attn.v_proj.weight.shape[0]
    o_in = layer.self_attn.o_proj.weight.shape[1]

    # Attention 输入维度（从 q_proj 的输入维度推断）
    attn_input = layer.self_attn.q_proj.weight.shape[1]

    # MLP 维度
    gate_out = layer.mlp.gate_proj.weight.shape[0]
    up_out = layer.mlp.up_proj.weight.shape[0]
    down_in = layer.mlp.down_proj.weight.shape[1]
    down_out = layer.mlp.down_proj.weight.shape[0]

    # MLP 输入维度
    mlp_input = layer.mlp.gate_proj.weight.shape[1]

    num_q_heads = q_out // head_dim
    num_kv_heads = k_out // head_dim

    print(f"\nLayer {i}:")
    print(f"  Attention:")
    print(f"    输入维度: {attn_input}")
    print(f"    Q projection: {attn_input} -> {q_out} ({num_q_heads} heads)")
    print(f"    K projection: {attn_input} -> {k_out} ({num_kv_heads} heads)")
    print(f"    V projection: {attn_input} -> {v_out}")
    print(f"    O projection: {o_in} -> {layer.self_attn.o_proj.weight.shape[0]}")
    print(f"    GQA ratio: {num_q_heads // num_kv_heads}:1")

    print(f"  MLP:")
    print(f"    输入维度: {mlp_input}")
    print(f"    Gate projection: {mlp_input} -> {gate_out}")
    print(f"    Up projection: {mlp_input} -> {up_out}")
    print(f"    Down projection: {down_in} -> {down_out}")

    print(f"  LayerNorm:")
    print(f"    input_layernorm: {layer.input_layernorm.weight.shape[0]}")
    print(f"    post_attention_layernorm: {layer.post_attention_layernorm.weight.shape[0]}")

    # 检查不一致性
    inconsistencies = []
    if attn_input != mlp_input:
        inconsistencies.append(f"Attention输入({attn_input}) != MLP输入({mlp_input})")
    if o_in != q_out:
        inconsistencies.append(f"O projection输入({o_in}) != Q projection输出({q_out})")
    if down_out != attn_input:
        inconsistencies.append(f"Down projection输出({down_out}) != Attention输入({attn_input})")

    if inconsistencies:
        print(f"  ⚠️  维度不一致:")
        for issue in inconsistencies:
            print(f"      - {issue}")

print("\n" + "=" * 100)
print("诊断完成")
print("=" * 100)
