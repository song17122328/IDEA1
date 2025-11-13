#!/bin/bash
# 微调GQA-aware剪枝后的 Llama-3-8B 模型 (v3)
#
# 模型特点：
# - GQA-aware Taylor importance 剪枝
# - MLP Taylor importance 剪枝
# - 参数减少84.64%（8B → 1.2B）
# - 剪枝后PPL: 37.82 (vs 原始19.66)
#
# 微调策略：
# - 低学习率（剪枝后模型更脆弱）
# - LoRA r=8, alpha=16
# - 目标模块：Attention + MLP全覆盖
# - 训练数据：alpaca-cleaned

# 配置
PRUNE_MODEL="prune_log/llama3_gqa_aware_pruned_v3/pytorch_model.bin"
OUTPUT_DIR="./finetuned_llama3_gqa_aware_v3"
GPU_ID=0  # 根据需要修改

# 检查模型是否存在
if [ ! -f "$PRUNE_MODEL" ]; then
    echo "错误：剪枝模型不存在: $PRUNE_MODEL"
    echo "请先运行: ./run_gqa_aware_pruning.sh"
    exit 1
fi

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

# 开始训练
echo "=========================================="
echo "微调 GQA-aware 剪枝模型 (v3)"
echo "=========================================="
echo "剪枝模型: $PRUNE_MODEL"
echo "输出目录: $OUTPUT_DIR"
echo "使用 GPU: $GPU_ID"
echo ""
echo "模型信息："
echo "  - 剪枝方法: GQA-aware Taylor + MLP Taylor"
echo "  - 参数量: 1.2B (原始8B，减少84.64%)"
echo "  - 剪枝后PPL: 37.82"
echo "  - GQA比例: 所有层保持4:1 ✓"
echo ""
echo "微调配置："
echo "  - 学习率: 5e-5 (低学习率，保护剪枝后的结构)"
echo "  - LoRA r: 8, alpha: 16"
echo "  - 训练轮数: 2 epochs"
echo "  - Batch size: 128 (micro: 4)"
echo "  - 目标模块: Attention + MLP 全覆盖"
echo "=========================================="

CUDA_VISIBLE_DEVICES=$GPU_ID python post_training.py \
    --prune_model "$PRUNE_MODEL" \
    --data_path yahma/alpaca-cleaned \
    --output_dir "$OUTPUT_DIR" \
    --wandb_project llama3-gqa-aware-finetune-v3 \
    --num_epochs 2 \
    --learning_rate 5e-5 \
    --batch_size 128 \
    --micro_batch_size 4 \
    --cutoff_len 256 \
    --val_set_size 2000 \
    --lora_r 8 \
    --lora_alpha 16 \
    --lora_target_modules q_proj,k_proj,v_proj,o_proj,gate_proj,down_proj,up_proj \
    --extra_val_dataset wikitext2 \
    --cache_dataset

echo ""
echo "=========================================="
echo "微调完成！"
echo "模型保存在: $OUTPUT_DIR"
echo "=========================================="

# 评估微调后的模型
echo ""
echo "开始评估微调后的模型..."
python evaluate_lora_model.py \
    --pruned_model "$PRUNE_MODEL" \
    --lora_dir "$OUTPUT_DIR" \
    --device cuda:$GPU_ID

echo ""
echo "=========================================="
echo "全部完成！"
echo "=========================================="
echo ""
echo "预期结果："
echo "  - 剪枝后PPL: 37.82"
echo "  - 微调后PPL: 预期 20-25 (接近原始模型的19.66)"
echo "  - vs v2方法微调后PPL: 159.85"
echo "  - 改善: 6-8倍 ✓"
echo ""
echo "后续步骤："
echo "  1. 合并LoRA权重: python merge_and_save_lora.py ..."
echo "  2. 部署推理: 使用合并后的模型"
echo ""
