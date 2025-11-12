#!/usr/bin/env python3
"""
演示均匀剪枝的工作流程
模拟 MetaPruner 如何根据单一的 ch_sparsity 剪枝 Attention
"""

import numpy as np


class UniformPruningDemo:
    def __init__(self, num_layers=32, ch_sparsity=0.25):
        self.num_layers = num_layers
        self.ch_sparsity = ch_sparsity

        # Llama-3-8B 配置
        self.hidden_size = 4096
        self.num_q_heads = 32
        self.num_kv_heads = 8
        self.head_dim = 128

        # 每层的配置
        self.q_out = self.num_q_heads * self.head_dim  # 4096
        self.kv_out = self.num_kv_heads * self.head_dim  # 1024

    def simulate_taylor_importance(self, layer_idx, num_heads):
        """
        模拟 Taylor 重要性计算
        返回每个 head 的重要性分数
        """
        # 使用随机值模拟（实际是通过梯度计算的）
        np.random.seed(layer_idx + 42)
        importance = np.random.rand(num_heads)
        return importance

    def select_heads_to_prune(self, importance, num_to_prune):
        """
        选择要剪枝的 heads（重要性最低的）
        """
        # 按重要性升序排序，选择最不重要的
        sorted_indices = np.argsort(importance)
        heads_to_prune = sorted_indices[:num_to_prune]
        return sorted(heads_to_prune)

    def prune_single_layer(self, layer_idx):
        """
        演示对单层的剪枝过程
        """
        print(f"\n{'='*80}")
        print(f"剪枝 Layer {layer_idx} (剪枝率: {self.ch_sparsity*100:.0f}%)")
        print(f"{'='*80}\n")

        # 步骤1: 原始配置
        print("步骤1: 原始配置")
        print("-" * 80)
        print(f"  q_proj: [{self.hidden_size}, {self.q_out}]  → {self.num_q_heads} Q heads × {self.head_dim}")
        print(f"  k_proj: [{self.hidden_size}, {self.kv_out}]  → {self.num_kv_heads} KV heads × {self.head_dim}")
        print(f"  v_proj: [{self.hidden_size}, {self.kv_out}]  → {self.num_kv_heads} KV heads × {self.head_dim}")
        print(f"  o_proj: [{self.q_out}, {self.hidden_size}]")
        print()

        # 步骤2: 计算重要性
        print("步骤2: 计算 Taylor 重要性")
        print("-" * 80)
        kv_importance = self.simulate_taylor_importance(layer_idx, self.num_kv_heads)

        print("  KV heads 重要性:")
        for head_idx, imp in enumerate(kv_importance):
            print(f"    Head {head_idx}: {imp:.4f}")
        print()

        # 步骤3: 选择要剪枝的 heads
        num_kv_to_prune = int(self.num_kv_heads * self.ch_sparsity)
        heads_to_prune = self.select_heads_to_prune(kv_importance, num_kv_to_prune)

        print(f"步骤3: 选择要剪枝的 heads (ch_sparsity={self.ch_sparsity})")
        print("-" * 80)
        print(f"  需要剪枝: {num_kv_to_prune} 个 KV heads ({num_kv_to_prune}/{self.num_kv_heads} = {self.ch_sparsity*100:.0f}%)")
        print(f"  选择最不重要的 {num_kv_to_prune} 个: {heads_to_prune}")
        print()

        # 步骤4: 计算剪枝后的维度
        pruned_kv_heads = self.num_kv_heads - num_kv_to_prune
        pruned_q_heads = self.num_q_heads - num_kv_to_prune * (self.num_q_heads // self.num_kv_heads)

        pruned_kv_out = pruned_kv_heads * self.head_dim
        pruned_q_out = pruned_q_heads * self.head_dim

        print("步骤4: GQA 比例传播")
        print("-" * 80)
        print(f"  GQA 比例: {self.num_q_heads // self.num_kv_heads}:1")
        print(f"  剪枝 {num_kv_to_prune} 个 KV heads → 剪枝 {num_kv_to_prune * 4} 个 Q heads")
        print()

        # 步骤5: 物理剪枝
        print("步骤5: 物理执行剪枝")
        print("-" * 80)
        print(f"  k_proj: [{self.hidden_size}, {self.kv_out}] → [{self.hidden_size}, {pruned_kv_out}]")
        print(f"    删除 heads {heads_to_prune}")
        print(f"    删除通道: {[f'[{h*128}:{(h+1)*128}]' for h in heads_to_prune]}")
        print()

        print(f"  v_proj: [{self.hidden_size}, {self.kv_out}] → [{self.hidden_size}, {pruned_kv_out}]")
        print(f"    同步删除 heads {heads_to_prune}")
        print()

        # 对应的 Q heads
        q_heads_to_prune = []
        for kv_head in heads_to_prune:
            for i in range(4):  # GQA 4:1
                q_heads_to_prune.append(kv_head * 4 + i)

        print(f"  q_proj: [{self.hidden_size}, {self.q_out}] → [{self.hidden_size}, {pruned_q_out}]")
        print(f"    删除对应的 Q heads: {q_heads_to_prune}")
        print()

        print(f"  o_proj: [{self.q_out}, {self.hidden_size}] → [{pruned_q_out}, {self.hidden_size}]")
        print(f"    输入维度匹配 q_proj 的输出")
        print()

        # 步骤6: 参数统计
        print("步骤6: 参数量变化")
        print("-" * 80)

        orig_params = {
            'q_proj': self.hidden_size * self.q_out,
            'k_proj': self.hidden_size * self.kv_out,
            'v_proj': self.hidden_size * self.kv_out,
            'o_proj': self.q_out * self.hidden_size,
        }

        pruned_params = {
            'q_proj': self.hidden_size * pruned_q_out,
            'k_proj': self.hidden_size * pruned_kv_out,
            'v_proj': self.hidden_size * pruned_kv_out,
            'o_proj': pruned_q_out * self.hidden_size,
        }

        for name in ['q_proj', 'k_proj', 'v_proj', 'o_proj']:
            orig = orig_params[name]
            pruned = pruned_params[name]
            reduction = orig - pruned
            ratio = reduction / orig * 100

            print(f"  {name:<8} {orig:>12,} → {pruned:>12,}   减少 {reduction:>12,} ({ratio:>5.2f}%)")

        total_orig = sum(orig_params.values())
        total_pruned = sum(pruned_params.values())
        total_reduction = total_orig - total_pruned
        total_ratio = total_reduction / total_orig * 100

        print("-" * 80)
        print(f"  {'总计':<8} {total_orig:>12,} → {total_pruned:>12,}   减少 {total_reduction:>12,} ({total_ratio:>5.2f}%)")
        print()

        return {
            'layer_idx': layer_idx,
            'heads_pruned': heads_to_prune,
            'kv_heads': f"{self.num_kv_heads} → {pruned_kv_heads}",
            'q_heads': f"{self.num_q_heads} → {pruned_q_heads}",
            'params_reduction': total_ratio
        }

    def demonstrate_uniform_pruning(self, layers_to_show=[5, 11, 27]):
        """
        演示多层的均匀剪枝
        """
        print("\n" + "="*80)
        print("均匀剪枝演示")
        print("="*80)
        print(f"\n配置:")
        print(f"  模型: Llama-3-8B")
        print(f"  总层数: {self.num_layers}")
        print(f"  全局剪枝率 (ch_sparsity): {self.ch_sparsity*100:.0f}%")
        print(f"  剪枝层范围: 3-29 (共27层)")
        print()

        print("关键点:")
        print("  ✅ 所有层使用相同的剪枝率")
        print("  ✅ 每层根据 Taylor 重要性选择不同的 heads")
        print("  ✅ 但剪枝的 head 数量相同")
        print()

        results = []
        for layer_idx in layers_to_show:
            result = self.prune_single_layer(layer_idx)
            results.append(result)

        # 总结
        print("\n" + "="*80)
        print("总结")
        print("="*80)
        print()
        print(f"{'Layer':<8} {'剪枝的 heads':<20} {'KV heads':<15} {'Q heads':<15} {'参数减少'}")
        print("-" * 80)
        for r in results:
            print(f"{r['layer_idx']:<8} {str(r['heads_pruned']):<20} {r['kv_heads']:<15} {r['q_heads']:<15} {r['params_reduction']:.2f}%")

        print()
        print("观察:")
        print(f"  1. 所有层的参数减少率都是 {results[0]['params_reduction']:.2f}% (相同)")
        print(f"  2. 所有层剪枝的 head 数量相同: {len(results[0]['heads_pruned'])} 个 KV heads")
        print(f"  3. 但每层选择的具体 heads 不同（基于各自的 Taylor 重要性）")
        print()


def main():
    demo = UniformPruningDemo(num_layers=32, ch_sparsity=0.25)
    demo.demonstrate_uniform_pruning(layers_to_show=[5, 11, 27])

    print("\n" + "="*80)
    print("与非均衡剪枝的对比")
    print("="*80)
    print()
    print("均匀剪枝 (当前演示):")
    print("  - Layer 5:  剪枝 25% (2 个 KV heads)")
    print("  - Layer 11: 剪枝 25% (2 个 KV heads)")
    print("  - Layer 27: 剪枝 25% (2 个 KV heads)")
    print()
    print("非均衡剪枝 (基于层重要性):")
    print("  - Layer 5:  剪枝 26.29% (重要性: 2.13)")
    print("  - Layer 11: 剪枝 27.50% (重要性: 0.22, 最不重要)")
    print("  - Layer 27: 剪枝 25.40% (重要性: 3.89)")
    print()
    print("优势:")
    print("  ✅ 非均衡剪枝可以更激进地剪枝不重要的层")
    print("  ✅ 同时保护重要的层")
    print("  ✅ 在相同剪枝率下获得更好的性能")
    print()


if __name__ == '__main__':
    main()
