#!/usr/bin/env python3
"""
模型大小检查工具
用于比较原始模型和剪枝后模型的物理占用空间
"""

import os
import torch
import argparse
from pathlib import Path


def format_size(size_bytes):
    """将字节转换为人类可读的格式"""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} PB"


def get_file_size(file_path):
    """获取文件大小（字节）"""
    if os.path.exists(file_path):
        return os.path.getsize(file_path)
    return None


def get_model_params(model_path):
    """加载模型并统计参数数量"""
    try:
        print(f"正在加载模型: {model_path}")
        checkpoint = torch.load(model_path, map_location='cpu')

        if 'model' in checkpoint:
            model = checkpoint['model']
        else:
            model = checkpoint

        # 统计参数数量
        total_params = 0
        trainable_params = 0

        if hasattr(model, 'parameters'):
            for param in model.parameters():
                total_params += param.numel()
                if param.requires_grad:
                    trainable_params += param.numel()

        return total_params, trainable_params
    except Exception as e:
        print(f"加载模型时出错: {e}")
        return None, None


def check_pruned_model(prune_log_dir='prune_log'):
    """检查剪枝后的模型"""
    print("=" * 80)
    print("模型大小检查工具")
    print("=" * 80)

    # 查找所有的 pytorch_model.bin 文件
    prune_log_path = Path(prune_log_dir)
    if not prune_log_path.exists():
        print(f"错误：目录 {prune_log_dir} 不存在")
        return

    model_files = list(prune_log_path.glob('**/pytorch_model.bin'))

    if not model_files:
        print(f"在 {prune_log_dir} 中没有找到任何 pytorch_model.bin 文件")
        return

    print(f"\n找到 {len(model_files)} 个模型文件：\n")

    for i, model_file in enumerate(sorted(model_files), 1):
        file_size = get_file_size(model_file)

        print(f"{i}. 文件路径: {model_file}")
        print(f"   物理大小: {format_size(file_size)}")

        # 尝试加载模型并获取参数信息
        total_params, trainable_params = get_model_params(model_file)

        if total_params is not None:
            print(f"   总参数量: {total_params:,}")
            print(f"   可训练参数: {trainable_params:,}")
            print(f"   参数大小 (FP16): {format_size(total_params * 2)}")
            print(f"   参数大小 (FP32): {format_size(total_params * 4)}")

        print()

    # 如果有多个文件，显示对比
    if len(model_files) > 1:
        print("-" * 80)
        print("大小对比：")
        sizes = [(f, get_file_size(f)) for f in model_files]
        sizes.sort(key=lambda x: x[1], reverse=True)

        largest = sizes[0]
        print(f"\n最大文件: {largest[0].name}")
        print(f"大小: {format_size(largest[1])}")

        for file, size in sizes[1:]:
            reduction = (1 - size / largest[1]) * 100
            print(f"\n{file.name}")
            print(f"大小: {format_size(size)}")
            print(f"相比最大文件减少: {reduction:.2f}%")


def compare_with_original(original_model_path, pruned_model_path):
    """对比原始模型和剪枝后模型"""
    print("=" * 80)
    print("原始模型 vs 剪枝模型对比")
    print("=" * 80)

    # 检查原始模型
    if not os.path.exists(original_model_path):
        print(f"错误：原始模型路径不存在: {original_model_path}")
        return

    # 如果是目录，查找模型文件
    if os.path.isdir(original_model_path):
        # 对于 HuggingFace 模型，查找 .bin 或 .safetensors 文件
        model_files = list(Path(original_model_path).glob('*.bin')) + \
                     list(Path(original_model_path).glob('*.safetensors'))
        if model_files:
            original_size = sum(get_file_size(f) for f in model_files)
            print(f"\n原始模型目录: {original_model_path}")
            print(f"包含 {len(model_files)} 个权重文件")
            print(f"总大小: {format_size(original_size)}")
        else:
            print(f"在 {original_model_path} 中找不到模型文件")
            return
    else:
        original_size = get_file_size(original_model_path)
        print(f"\n原始模型: {original_model_path}")
        print(f"大小: {format_size(original_size)}")

    # 检查剪枝后的模型
    if not os.path.exists(pruned_model_path):
        print(f"\n错误：剪枝模型路径不存在: {pruned_model_path}")
        return

    pruned_size = get_file_size(pruned_model_path)
    print(f"\n剪枝后模型: {pruned_model_path}")
    print(f"大小: {format_size(pruned_size)}")

    # 计算减少量
    reduction = original_size - pruned_size
    reduction_pct = (reduction / original_size) * 100

    print(f"\n{'=' * 80}")
    print(f"物理空间减少: {format_size(reduction)} ({reduction_pct:.2f}%)")
    print(f"剩余大小占比: {100 - reduction_pct:.2f}%")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='检查模型文件大小')
    parser.add_argument('--prune_log_dir', type=str, default='prune_log',
                      help='剪枝日志目录 (默认: prune_log)')
    parser.add_argument('--original_model', type=str,
                      help='原始模型路径 (用于对比)')
    parser.add_argument('--pruned_model', type=str,
                      help='剪枝后模型路径 (用于对比)')

    args = parser.parse_args()

    if args.original_model and args.pruned_model:
        # 对比模式
        compare_with_original(args.original_model, args.pruned_model)
    else:
        # 检查模式
        check_pruned_model(args.prune_log_dir)
