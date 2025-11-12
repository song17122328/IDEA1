#!/usr/bin/env python3
"""
可视化非均衡结构化剪枝流程
"""

import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def load_pruning_config(config_path: str):
    """加载剪枝配置"""
    with open(config_path, 'r') as f:
        config = json.load(f)
    layer_rates = {int(k): v for k, v in config['layer_pruning_rates'].items()}
    return layer_rates


def create_overview_diagram(layer_pruning_rates, output_path):
    """创建整体流程概览图"""
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle('非均衡结构化剪枝流程详解', fontsize=20, fontweight='bold')

    # ============ 子图1: 层剪枝率分布 ============
    ax1 = axes[0, 0]
    layers = sorted(layer_pruning_rates.keys())
    rates = [layer_pruning_rates[i] for i in layers]

    # 标记不同类型的层
    protected_early = [i for i in layers if i < 3]
    pruned_layers = [i for i in layers if 3 <= i < 30]
    protected_late = [i for i in layers if i >= 30]

    colors = []
    for i in layers:
        if i in protected_early or i in protected_late:
            colors.append('#FF6B6B')  # 红色 - 保护的层
        else:
            colors.append('#4ECDC4')  # 青色 - 剪枝的层

    bars = ax1.bar(layers, rates, color=colors, edgecolor='black', linewidth=1)

    # 添加分界线
    ax1.axvline(x=2.5, color='red', linestyle='--', linewidth=2, alpha=0.7)
    ax1.axvline(x=29.5, color='red', linestyle='--', linewidth=2, alpha=0.7)

    # 添加标注
    ax1.text(1, 0.28, '前3层\n不剪枝', ha='center', va='top',
            bbox=dict(boxstyle='round', facecolor='#FF6B6B', alpha=0.5),
            fontsize=10, fontweight='bold')
    ax1.text(16, 0.28, '中间27层\n根据重要性剪枝', ha='center', va='top',
            bbox=dict(boxstyle='round', facecolor='#4ECDC4', alpha=0.5),
            fontsize=10, fontweight='bold')
    ax1.text(30.5, 0.28, '后2层\n不剪枝', ha='center', va='top',
            bbox=dict(boxstyle='round', facecolor='#FF6B6B', alpha=0.5),
            fontsize=10, fontweight='bold')

    ax1.set_xlabel('层索引', fontsize=12, fontweight='bold')
    ax1.set_ylabel('剪枝率', fontsize=12, fontweight='bold')
    ax1.set_title('步骤2: 计算所有层的剪枝率\n(基于层重要性)', fontsize=14, fontweight='bold')
    ax1.set_ylim(0, 0.3)
    ax1.grid(True, alpha=0.3)

    # 添加图例
    legend_elements = [
        mpatches.Patch(color='#FF6B6B', label='保护的层（不剪枝）'),
        mpatches.Patch(color='#4ECDC4', label='剪枝的层')
    ]
    ax1.legend(handles=legend_elements, loc='upper right', fontsize=10)

    # ============ 子图2: 过滤逻辑 ============
    ax2 = axes[0, 1]
    ax2.axis('off')

    # 绘制过滤流程
    y_start = 0.9
    step_height = 0.15

    # 步骤1: 输入
    box1 = FancyBboxPatch((0.1, y_start - step_height * 1), 0.8, 0.12,
                          boxstyle="round,pad=0.01", edgecolor='black',
                          facecolor='#FFE66D', linewidth=2)
    ax2.add_patch(box1)
    ax2.text(0.5, y_start - step_height * 1 + 0.06, '输入: 32层的剪枝率\n(Layer 0-31)',
            ha='center', va='center', fontsize=11, fontweight='bold')

    # 箭头1
    ax2.annotate('', xy=(0.5, y_start - step_height * 1), xytext=(0.5, y_start - step_height * 0.8),
                arrowprops=dict(arrowstyle='->', lw=2, color='black'))

    # 步骤2: 参数配置
    box2 = FancyBboxPatch((0.1, y_start - step_height * 2.2), 0.8, 0.15,
                          boxstyle="round,pad=0.01", edgecolor='black',
                          facecolor='#A8DADC', linewidth=2)
    ax2.add_patch(box2)
    ax2.text(0.5, y_start - step_height * 2.2 + 0.075,
            '参数配置:\n--block_attention_layer_start=3\n--block_attention_layer_end=30',
            ha='center', va='center', fontsize=10, fontweight='bold')

    # 箭头2
    ax2.annotate('', xy=(0.5, y_start - step_height * 2.2), xytext=(0.5, y_start - step_height * 1.9),
                arrowprops=dict(arrowstyle='->', lw=2, color='black'))

    # 步骤3: 过滤
    box3 = FancyBboxPatch((0.1, y_start - step_height * 3.5), 0.8, 0.15,
                          boxstyle="round,pad=0.01", edgecolor='black',
                          facecolor='#F4A261', linewidth=2)
    ax2.add_patch(box3)
    ax2.text(0.5, y_start - step_height * 3.5 + 0.075,
            '过滤逻辑:\npruning_layers = set(range(3, 30))\n→ [3, 4, 5, ..., 29]',
            ha='center', va='center', fontsize=10, fontweight='bold')

    # 箭头3
    ax2.annotate('', xy=(0.5, y_start - step_height * 3.5), xytext=(0.5, y_start - step_height * 3.2),
                arrowprops=dict(arrowstyle='->', lw=2, color='black'))

    # 步骤4: 输出
    box4 = FancyBboxPatch((0.1, y_start - step_height * 4.7), 0.8, 0.12,
                          boxstyle="round,pad=0.01", edgecolor='black',
                          facecolor='#95E1D3', linewidth=2)
    ax2.add_patch(box4)
    ax2.text(0.5, y_start - step_height * 4.7 + 0.06,
            '输出: 27层的剪枝率\n(Layer 3-29)',
            ha='center', va='center', fontsize=11, fontweight='bold')

    # 箭头4
    ax2.annotate('', xy=(0.5, y_start - step_height * 4.7), xytext=(0.5, y_start - step_height * 4.4),
                arrowprops=dict(arrowstyle='->', lw=2, color='black'))

    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.set_title('步骤3: 层过滤逻辑', fontsize=14, fontweight='bold')

    # ============ 子图3: ch_sparsity_dict 创建 ============
    ax3 = axes[1, 0]
    ax3.axis('off')

    # 示例：显示几个层的模块映射
    example_layers = [3, 5, 11, 29]
    y_pos = 0.9
    y_step = 0.2

    ax3.text(0.5, 0.95, 'ch_sparsity_dict 创建示例', ha='center', va='top',
            fontsize=14, fontweight='bold')

    for idx, layer_idx in enumerate(example_layers):
        rate = layer_pruning_rates[layer_idx]
        y = y_pos - idx * y_step

        # 层标签
        ax3.text(0.05, y, f'Layer {layer_idx}:', ha='left', va='center',
                fontsize=11, fontweight='bold')

        # k_proj
        box_k = FancyBboxPatch((0.2, y - 0.04), 0.25, 0.05,
                              boxstyle="round,pad=0.005", edgecolor='blue',
                              facecolor='lightblue', linewidth=1.5)
        ax3.add_patch(box_k)
        ax3.text(0.325, y - 0.015, 'k_proj', ha='center', va='center',
                fontsize=9, fontweight='bold')

        # 箭头
        ax3.annotate('', xy=(0.55, y - 0.015), xytext=(0.46, y - 0.015),
                    arrowprops=dict(arrowstyle='->', lw=1.5, color='black'))

        # 剪枝率
        ax3.text(0.7, y - 0.015, f'{rate:.4f}', ha='center', va='center',
                fontsize=10, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))

        # gate_proj
        box_g = FancyBboxPatch((0.2, y - 0.09), 0.25, 0.05,
                              boxstyle="round,pad=0.005", edgecolor='green',
                              facecolor='lightgreen', linewidth=1.5)
        ax3.add_patch(box_g)
        ax3.text(0.325, y - 0.065, 'gate_proj', ha='center', va='center',
                fontsize=9, fontweight='bold')

        # 箭头
        ax3.annotate('', xy=(0.55, y - 0.065), xytext=(0.46, y - 0.065),
                    arrowprops=dict(arrowstyle='->', lw=1.5, color='black'))

        # 剪枝率
        ax3.text(0.7, y - 0.065, f'{rate:.4f}', ha='center', va='center',
                fontsize=10, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))

    # 总结
    ax3.text(0.5, 0.05, '总计: 27层 × 2模块 = 54个模块', ha='center', va='center',
            fontsize=12, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='orange', alpha=0.5))

    ax3.set_xlim(0, 1)
    ax3.set_ylim(0, 1)

    # ============ 子图4: MetaPruner 工作流程 ============
    ax4 = axes[1, 1]
    ax4.axis('off')

    ax4.text(0.5, 0.95, 'MetaPruner 工作流程', ha='center', va='top',
            fontsize=14, fontweight='bold')

    steps = [
        ('1. 构建依赖图', '前向传播追踪模块关系', '#FFE66D'),
        ('2. 计算重要性', 'Taylor: |∂L/∂W × W|', '#A8DADC'),
        ('3. 选择通道', '保留重要性高的通道', '#F4A261'),
        ('4. 传播决策', '自动传播到依赖模块', '#95E1D3'),
        ('5. 物理剪枝', '删除权重矩阵行/列', '#88D8B0'),
    ]

    y_pos = 0.85
    y_step = 0.16

    for idx, (title, desc, color) in enumerate(steps):
        y = y_pos - idx * y_step

        # 步骤框
        box = FancyBboxPatch((0.1, y - 0.06), 0.8, 0.1,
                            boxstyle="round,pad=0.01", edgecolor='black',
                            facecolor=color, linewidth=2)
        ax4.add_patch(box)

        # 文字
        ax4.text(0.15, y - 0.01, title, ha='left', va='center',
                fontsize=11, fontweight='bold')
        ax4.text(0.5, y - 0.045, desc, ha='center', va='center',
                fontsize=9, style='italic')

        # 箭头（除了最后一个）
        if idx < len(steps) - 1:
            ax4.annotate('', xy=(0.5, y - 0.06), xytext=(0.5, y - 0.03),
                        arrowprops=dict(arrowstyle='->', lw=2, color='black'))

    ax4.set_xlim(0, 1)
    ax4.set_ylim(0, 1)

    # 保存图表
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ 流程图已保存到: {output_path}")


def create_module_detail_diagram(output_path):
    """创建模块级剪枝详细图"""
    fig, ax = plt.subplots(1, 1, figsize=(16, 10))
    fig.suptitle('Layer 5 结构化剪枝详细流程（剪枝率 26.29%）', fontsize=18, fontweight='bold')

    ax.axis('off')
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)

    # ====== 左侧：原始结构 ======
    ax.text(1.5, 9.5, '原始结构', ha='center', va='top',
           fontsize=14, fontweight='bold',
           bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))

    # Attention
    y_att = 8.5
    ax.text(0.2, y_att, 'Attention:', ha='left', va='center',
           fontsize=11, fontweight='bold')

    modules_orig = [
        ('q_proj', '4096→4096', 16777216),
        ('k_proj', '4096→1024', 4194304),
        ('v_proj', '4096→1024', 4194304),
        ('o_proj', '4096→4096', 16777216),
    ]

    y = y_att - 0.5
    for name, shape, params in modules_orig:
        box = FancyBboxPatch((0.3, y - 0.15), 2.2, 0.25,
                            boxstyle="round,pad=0.01", edgecolor='black',
                            facecolor='lightcyan', linewidth=1.5)
        ax.add_patch(box)
        ax.text(0.5, y, name, ha='left', va='center', fontsize=10, fontweight='bold')
        ax.text(1.5, y, shape, ha='center', va='center', fontsize=9)
        ax.text(2.3, y - 0.05, f'{params:,}', ha='right', va='center',
               fontsize=8, style='italic')
        y -= 0.4

    # MLP
    y_mlp = y - 0.3
    ax.text(0.2, y_mlp, 'MLP:', ha='left', va='center',
           fontsize=11, fontweight='bold')

    modules_mlp_orig = [
        ('gate_proj', '4096→14336', 58720256),
        ('up_proj', '4096→14336', 58720256),
        ('down_proj', '14336→4096', 58720256),
    ]

    y = y_mlp - 0.5
    for name, shape, params in modules_mlp_orig:
        box = FancyBboxPatch((0.3, y - 0.15), 2.2, 0.25,
                            boxstyle="round,pad=0.01", edgecolor='black',
                            facecolor='lightgreen', linewidth=1.5)
        ax.add_patch(box)
        ax.text(0.5, y, name, ha='left', va='center', fontsize=10, fontweight='bold')
        ax.text(1.5, y, shape, ha='center', va='center', fontsize=9)
        ax.text(2.3, y - 0.05, f'{params:,}', ha='right', va='center',
               fontsize=8, style='italic')
        y -= 0.4

    # 总参数
    total_orig = 217841664
    ax.text(1.5, 1.5, f'总参数: {total_orig:,}', ha='center', va='center',
           fontsize=11, fontweight='bold',
           bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

    # ====== 中间：剪枝操作 ======
    ax.text(5, 9.5, '剪枝操作', ha='center', va='top',
           fontsize=14, fontweight='bold',
           bbox=dict(boxstyle='round', facecolor='orange', alpha=0.7))

    # 剪枝说明
    prune_info = [
        ('k_proj 剪枝:', '1024 → 755'),
        ('→ q_proj 传播:', '4096 → 3020'),
        ('→ v_proj 传播:', '1024 → 755'),
        ('→ o_proj 传播:', '4096 → 3020'),
        '',
        ('gate_proj 剪枝:', '14336 → 10570'),
        ('→ up_proj 传播:', '14336 → 10570'),
        ('→ down_proj 传播:', '14336 → 10570'),
    ]

    y = 8.5
    for info in prune_info:
        if info:
            parts = info.split(':')
            if len(parts) == 2:
                ax.text(4, y, parts[0], ha='left', va='center',
                       fontsize=10, fontweight='bold')
                ax.text(5.5, y, parts[1], ha='left', va='center',
                       fontsize=10, color='red')
        y -= 0.4

    # ====== 右侧：剪枝后结构 ======
    ax.text(8.5, 9.5, '剪枝后结构', ha='center', va='top',
           fontsize=14, fontweight='bold',
           bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))

    # Attention
    y_att = 8.5
    ax.text(7.2, y_att, 'Attention:', ha='left', va='center',
           fontsize=11, fontweight='bold')

    modules_pruned = [
        ('q_proj', '4096→3020', 12369920),
        ('k_proj', '4096→755', 3092480),
        ('v_proj', '4096→755', 3092480),
        ('o_proj', '3020→4096', 12369920),
    ]

    y = y_att - 0.5
    for name, shape, params in modules_pruned:
        box = FancyBboxPatch((7.3, y - 0.15), 2.2, 0.25,
                            boxstyle="round,pad=0.01", edgecolor='black',
                            facecolor='lightcyan', linewidth=1.5)
        ax.add_patch(box)
        ax.text(7.5, y, name, ha='left', va='center', fontsize=10, fontweight='bold')
        ax.text(8.5, y, shape, ha='center', va='center', fontsize=9)
        ax.text(9.3, y - 0.05, f'{params:,}', ha='right', va='center',
               fontsize=8, style='italic')
        y -= 0.4

    # MLP
    y_mlp = y - 0.3
    ax.text(7.2, y_mlp, 'MLP:', ha='left', va='center',
           fontsize=11, fontweight='bold')

    modules_mlp_pruned = [
        ('gate_proj', '4096→10570', 43294720),
        ('up_proj', '4096→10570', 43294720),
        ('down_proj', '10570→4096', 43294720),
    ]

    y = y_mlp - 0.5
    for name, shape, params in modules_mlp_pruned:
        box = FancyBboxPatch((7.3, y - 0.15), 2.2, 0.25,
                            boxstyle="round,pad=0.01", edgecolor='black',
                            facecolor='lightgreen', linewidth=1.5)
        ax.add_patch(box)
        ax.text(7.5, y, name, ha='left', va='center', fontsize=10, fontweight='bold')
        ax.text(8.5, y, shape, ha='center', va='center', fontsize=9)
        ax.text(9.3, y - 0.05, f'{params:,}', ha='right', va='center',
               fontsize=8, style='italic')
        y -= 0.4

    # 总参数
    total_pruned = 160809960
    reduction = total_orig - total_pruned
    reduction_pct = (reduction / total_orig) * 100

    ax.text(8.5, 1.5, f'总参数: {total_pruned:,}', ha='center', va='center',
           fontsize=11, fontweight='bold',
           bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

    ax.text(8.5, 1.0, f'减少: {reduction:,} ({reduction_pct:.2f}%)', ha='center', va='center',
           fontsize=10, fontweight='bold', color='red')

    # 保存
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ 模块详细图已保存到: {output_path}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description='可视化非均衡结构化剪枝流程')
    parser.add_argument('--config', type=str,
                       default='prune_log/llama_unbalanced_prune/layer_importance_config.json',
                       help='层重要性配置文件路径')
    parser.add_argument('--output_dir', type=str,
                       default='prune_log/llama_unbalanced_prune',
                       help='输出目录')

    args = parser.parse_args()

    print("=" * 80)
    print("可视化非均衡结构化剪枝流程")
    print("=" * 80)
    print()

    # 加载配置
    layer_pruning_rates = load_pruning_config(args.config)
    print(f"✅ 成功加载 {len(layer_pruning_rates)} 层的剪枝率配置")
    print()

    # 创建流程概览图
    print("生成流程概览图...")
    overview_path = f"{args.output_dir}/pruning_flow_overview.png"
    create_overview_diagram(layer_pruning_rates, overview_path)
    print()

    # 创建模块详细图
    print("生成模块详细图...")
    detail_path = f"{args.output_dir}/module_pruning_detail.png"
    create_module_detail_diagram(detail_path)
    print()

    print("=" * 80)
    print("✅ 所有图表生成完成！")
    print("=" * 80)


if __name__ == '__main__':
    main()
