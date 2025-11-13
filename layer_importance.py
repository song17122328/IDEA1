#!/usr/bin/env python3
"""
层重要度分析工具 - 用于评估 Transformer 各层的重要性
结合结构化剪枝，实现非均衡剪枝
"""

import torch
import numpy as np
from typing import List, Dict
from tqdm import tqdm
import json
import matplotlib.pyplot as plt


class LayerImportanceAnalyzer:
    """分析Transformer各层的重要性"""

    def __init__(self, model, tokenizer, device='cuda'):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.eval()

    def compute_perplexity(self, texts: List[str]) -> float:
        """计算困惑度"""
        total_loss = 0
        total_tokens = 0

        with torch.no_grad():
            for text in tqdm(texts, desc="计算困惑度"):
                inputs = self.tokenizer(text, return_tensors="pt",
                                       truncation=True, max_length=512)
                # 将inputs移动到模型第一层所在的设备
                first_device = next(self.model.parameters()).device
                inputs = {k: v.to(first_device) for k, v in inputs.items()}

                outputs = self.model(**inputs, labels=inputs["input_ids"])
                total_loss += outputs.loss.item() * inputs["input_ids"].size(1)
                total_tokens += inputs["input_ids"].size(1)

        return np.exp(total_loss / total_tokens)

    def measure_layer_importance_by_removal(self, texts: List[str],
                                           num_layers: int) -> Dict[int, float]:
        """
        通过移除层来评估重要性（困惑度变化）
        重要性 = 移除该层后的困惑度增加量
        """
        baseline_ppl = self.compute_perplexity(texts)
        layer_importance = {}

        print(f"基准困惑度: {baseline_ppl:.4f}")

        for layer_idx in tqdm(range(num_layers), desc="分析层重要性"):
            # 保存原始forward函数
            original_forward = self.model.model.layers[layer_idx].forward

            # 定义恒等映射函数
            def identity_forward(hidden_states, *args, **kwargs):
                # 直接返回输入的hidden_states，跳过该层的计算
                # Llama 的 DecoderLayer forward 返回格式：
                # - 如果不返回额外信息：hidden_states
                # - 如果返回注意力权重：(hidden_states, self_attn_weights, present_key_value)

                # 检查是否需要返回额外信息
                output_attentions = kwargs.get('output_attentions', False)
                use_cache = kwargs.get('use_cache', False)

                if output_attentions or use_cache:
                    # 返回元组格式
                    outputs = (hidden_states,)
                    if output_attentions:
                        outputs += (None,)  # self_attn_weights
                    if use_cache:
                        outputs += (None,)  # present_key_value
                    return outputs
                else:
                    # 只返回 hidden_states
                    return hidden_states

            # 临时替换该层的forward
            self.model.model.layers[layer_idx].forward = identity_forward

            try:
                ppl = self.compute_perplexity(texts)
                importance = ppl - baseline_ppl  # 困惑度增加越多，该层越重要
                layer_importance[layer_idx] = importance

                # print(f"第 {layer_idx} 层: PPL 变化 = {importance:.4f}")
            finally:
                # 无论是否出错，都要恢复该层
                self.model.model.layers[layer_idx].forward = original_forward

        return layer_importance

    def measure_layer_importance_by_activation(self, texts: List[str]) -> Dict[int, float]:
        """通过激活值统计评估重要性"""
        layer_activations = {}

        def hook_fn(layer_idx):
            def hook(module, input, output):
                if isinstance(output, tuple):
                    activation = output[0]
                else:
                    activation = output

                # 计算激活值的L2范数
                activation_norm = torch.norm(activation, p=2, dim=-1).mean().item()
                if layer_idx not in layer_activations:
                    layer_activations[layer_idx] = []
                layer_activations[layer_idx].append(activation_norm)
            return hook

        # 注册hooks
        hooks = []
        for idx, layer in enumerate(self.model.model.layers):
            hooks.append(layer.register_forward_hook(hook_fn(idx)))

        # 前向传播
        with torch.no_grad():
            for text in tqdm(texts, desc="收集激活值"):
                inputs = self.tokenizer(text, return_tensors="pt",
                                       truncation=True, max_length=512)
                first_device = next(self.model.parameters()).device
                inputs = {k: v.to(first_device) for k, v in inputs.items()}
                self.model(**inputs)

        # 移除hooks
        for hook in hooks:
            hook.remove()

        # 计算平均激活值
        layer_importance = {idx: np.mean(acts) for idx, acts in layer_activations.items()}
        return layer_importance


class UnbalancedStructuredPruningCalculator:
    """
    非均衡结构化剪枝率计算器
    结合层重要度和目标剪枝率，计算每层的剪枝率
    """

    def __init__(self, layer_importance: Dict[int, float], num_layers: int):
        self.layer_importance = layer_importance
        self.num_layers = num_layers

    def compute_layer_pruning_rates(self,
                                    target_overall_rate: float,
                                    strategy: str = 'inverse',
                                    alpha: float = 1.0,
                                    min_rate: float = 0.0,
                                    max_rate: float = 0.8,
                                    use_log_transform: bool = True) -> Dict[int, float]:
        """
        根据层重要性计算各层剪枝率

        Args:
            target_overall_rate: 目标整体剪枝率（例如 0.25 表示减少25%的参数）
            strategy: 剪枝策略
                - 'inverse': 重要层剪少，不重要层剪多（默认）
                - 'proportional': 重要层剪多，不重要层剪少（反向）
                - 'uniform': 所有层使用相同剪枝率
            alpha: 重要性权重系数，越大差异越明显
            min_rate: 最小剪枝率
            max_rate: 最大剪枝率
            use_log_transform: 是否使用对数变换处理极端值（推荐）

        Returns:
            Dict[int, float]: 每层的剪枝率
        """
        if strategy == 'uniform':
            # 均匀剪枝
            return {idx: target_overall_rate for idx in range(self.num_layers)}

        importance_values = np.array(list(self.layer_importance.values()))

        # 对数变换处理极端值
        if use_log_transform:
            # 平移使所有值为正（最小值+1），然后取对数
            min_val = importance_values.min()
            shifted_importance = importance_values - min_val + 1.0
            log_importance = np.log(shifted_importance)
            importance_values = log_importance

        if strategy == 'inverse':
            # 重要性高 -> 剪枝率低
            # 归一化到 [0, 1]
            normalized_importance = (importance_values - importance_values.min()) / \
                                   (importance_values.max() - importance_values.min() + 1e-8)

            # 应用alpha系数增强差异
            normalized_importance = normalized_importance ** alpha

            # 反转：重要性高 -> 剪枝率低
            inverse_importance = 1 - normalized_importance

            # 缩放使得平均剪枝率等于目标剪枝率
            pruning_rates = inverse_importance * (target_overall_rate * self.num_layers / inverse_importance.sum())

        elif strategy == 'proportional':
            # 重要性高 -> 剪枝率高
            normalized_importance = (importance_values - importance_values.min()) / \
                                   (importance_values.max() - importance_values.min() + 1e-8)

            normalized_importance = normalized_importance ** alpha

            # 直接使用：重要性高 -> 剪枝率高
            pruning_rates = normalized_importance * (target_overall_rate * self.num_layers / normalized_importance.sum())

        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        # 限制在合理范围内
        pruning_rates = np.clip(pruning_rates, min_rate, max_rate)

        # 重新归一化以确保平均剪枝率等于目标
        current_avg = pruning_rates.mean()
        if current_avg > 0:
            pruning_rates = pruning_rates * (target_overall_rate / current_avg)
            pruning_rates = np.clip(pruning_rates, min_rate, max_rate)

        return {idx: float(rate) for idx, rate in enumerate(pruning_rates)}

    def verify_average_pruning_rate(self, layer_pruning_rates: Dict[int, float]) -> Dict[str, float]:
        """验证各层平均剪枝率"""
        rates = list(layer_pruning_rates.values())
        avg_rate = np.mean(rates)
        std_rate = np.std(rates)
        min_rate = np.min(rates)
        max_rate = np.max(rates)

        return {
            'average_pruning_rate': avg_rate,
            'std_pruning_rate': std_rate,
            'min_pruning_rate': min_rate,
            'max_pruning_rate': max_rate,
            'rate_range': max_rate - min_rate
        }

    def save_pruning_rates(self, layer_pruning_rates: Dict[int, float], filepath: str):
        """保存剪枝率配置到JSON文件"""
        config = {
            'layer_pruning_rates': {str(k): v for k, v in layer_pruning_rates.items()},
            'layer_importance': {str(k): v for k, v in self.layer_importance.items()},
            'statistics': self.verify_average_pruning_rate(layer_pruning_rates)
        }

        with open(filepath, 'w') as f:
            json.dump(config, f, indent=2)

        print(f"剪枝率配置已保存到: {filepath}")

    @staticmethod
    def load_pruning_rates(filepath: str) -> Dict[int, float]:
        """从JSON文件加载剪枝率配置"""
        with open(filepath, 'r') as f:
            config = json.load(f)

        return {int(k): v for k, v in config['layer_pruning_rates'].items()}

    def visualize_pruning_strategy(self, layer_pruning_rates: Dict[int, float],
                                   save_path: str = None):
        """可视化剪枝策略"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))

        # 绘制层重要性
        layers = sorted(self.layer_importance.keys())
        importance_values = [self.layer_importance[i] for i in layers]

        ax1.bar(layers, importance_values, alpha=0.7, color='steelblue')
        ax1.set_xlabel('Layer Index', fontsize=12)
        ax1.set_ylabel('Importance Score', fontsize=12)
        ax1.set_title('Layer Importance Analysis', fontsize=14)
        ax1.grid(True, alpha=0.3)

        # 绘制剪枝率
        pruning_values = [layer_pruning_rates[i] for i in layers]

        ax2.bar(layers, pruning_values, alpha=0.7, color='coral')
        ax2.axhline(y=np.mean(pruning_values), color='r', linestyle='--',
                   label=f'Average: {np.mean(pruning_values):.4f}')
        ax2.set_xlabel('Layer Index', fontsize=12)
        ax2.set_ylabel('Pruning Rate', fontsize=12)
        ax2.set_title('Layer-wise Pruning Rate Distribution', fontsize=14)
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"可视化图表已保存到: {save_path}")

        plt.show()


def create_ch_sparsity_dict_for_llama(model, layer_pruning_rates: Dict[int, float],
                                      prune_attention: bool = True,
                                      prune_mlp: bool = True) -> Dict:
    """
    为 Llama 模型创建 ch_sparsity_dict

    Args:
        model: Llama 模型
        layer_pruning_rates: 每层的剪枝率
        prune_attention: 是否剪枝 Attention 模块
        prune_mlp: 是否剪枝 MLP 模块

    Returns:
        ch_sparsity_dict: 模块级别的剪枝率字典
    """
    ch_sparsity_dict = {}

    for layer_idx, pruning_rate in layer_pruning_rates.items():
        layer = model.model.layers[layer_idx]

        # Attention 模块 - 只为 k_proj 设置剪枝率（作为 root）
        # q_proj 通过依赖图自动从 k_proj 接收剪枝索引
        # torch_pruning的_ExpandIndexMapping会将KV head索引×4转换为Q head索引
        if prune_attention and hasattr(layer, 'self_attn'):
            ch_sparsity_dict[layer.self_attn.k_proj] = pruning_rate

        # MLP 模块
        if prune_mlp and hasattr(layer, 'mlp'):
            ch_sparsity_dict[layer.mlp.gate_proj] = pruning_rate

    return ch_sparsity_dict


if __name__ == "__main__":
    print("""
    层重要度分析工具
    ================

    用法示例:

    1. 评估层重要性:
        analyzer = LayerImportanceAnalyzer(model, tokenizer)
        importance = analyzer.measure_layer_importance_by_removal(texts, num_layers=32)

    2. 计算非均衡剪枝率:
        calculator = UnbalancedStructuredPruningCalculator(importance, num_layers=32)
        pruning_rates = calculator.compute_layer_pruning_rates(
            target_overall_rate=0.25,
            strategy='inverse',
            alpha=1.0
        )

    3. 创建 ch_sparsity_dict:
        ch_sparsity_dict = create_ch_sparsity_dict_for_llama(
            model, pruning_rates,
            prune_attention=True,
            prune_mlp=True
        )

    4. 在 llama3.py 中使用:
        pruner = tp.pruner.MetaPruner(
            model,
            forward_prompts,
            importance=imp,
            ch_sparsity=0.25,  # 默认剪枝率
            ch_sparsity_dict=ch_sparsity_dict,  # 每层的剪枝率
            ...
        )
    """)
