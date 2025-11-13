#!/bin/bash
# 微调剪枝后的 Llama-3-8B 模型 V2 - 修复版（降低学习率）
#
# 关键修复：
# 1. 学习率从 3e-4 降低到 5e-5（剪枝后的模型更脆弱）
# 2. 可选：减少 epochs 从 3 到 2（避免过拟合）
# 3. LoRA alpha 从 16 降低到 8（减少 adapter 的影响）

# 配置
PRUNE_MODEL="prune_log/llama_unbalanced_prune_v2/pytorch_model.bin"  # 剪枝脚本的输出路径
OUTPUT_DIR="./finetuned_llama3_v2_fixed"
GPU_ID=6

# 检查模型是否存在
if [ ! -f "$PRUNE_MODEL" ]; then
    echo "错误：剪枝模型不存在: $PRUNE_MODEL"
    echo "请先运行: bash run_pruning_v2.sh"
    exit 1
fi

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

# 开始训练
echo "=================================="
echo "微调剪枝后的 Llama-3 模型 V2 (修复版)"
echo "=================================="
echo "剪枝模型: $PRUNE_MODEL"
echo "输出目录: $OUTPUT_DIR"
echo "使用 GPU: $GPU_ID"
echo "LoRA 目标: Attention + MLP 层"
echo ""
echo "关键修复："
echo "  - 学习率: 3e-4 → 5e-5 (降低6倍)"
echo "  - LoRA alpha: 16 → 8 (减半)"
echo "  - 有效学习率: 6e-4 → 5e-5 (降低12倍)"
echo "=================================="

CUDA_VISIBLE_DEVICES=$GPU_ID python post_training.py \
    --prune_model "$PRUNE_MODEL" \
    --data_path yahma/alpaca-cleaned \
    --output_dir "$OUTPUT_DIR" \
    --wandb_project llama3-pruned-finetune-v2-fixed \
    --num_epochs 2 \
    --learning_rate 5e-5 \
    --batch_size 128 \
    --micro_batch_size 4 \
    --cutoff_len 256 \
    --val_set_size 2000 \
    --lora_r 8 \
    --lora_alpha 8 \
    --lora_target_modules q_proj,k_proj,v_proj,o_proj,gate_proj,down_proj,up_proj \
    --extra_val_dataset wikitext2 \
    --cache_dataset

echo "=================================="
echo "微调完成！"
echo "模型保存在: $OUTPUT_DIR"
echo "=================================="

# 评估微调后的模型
echo "开始评估微调后的模型..."
python evaluate_lora_model.py \
    --pruned_model "$PRUNE_MODEL" \
    --lora_dir "$OUTPUT_DIR" \
    --device cuda:$GPU_ID

echo "全部完成！"
