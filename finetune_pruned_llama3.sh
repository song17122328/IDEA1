#!/bin/bash
# 微调剪枝后的 Llama-3-8B 模型

# 配置
PRUNE_MODEL="prune_log/llama_unbalanced_prune_all_layers/pytorch_model.bin"
OUTPUT_DIR="./finetuned_llama3_unbalanced"
GPU_ID=6

# 检查模型是否存在
if [ ! -f "$PRUNE_MODEL" ]; then
    echo "错误：剪枝模型不存在: $PRUNE_MODEL"
    exit 1
fi

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

# 开始训练
echo "=================================="
echo "开始微调剪枝后的 Llama-3 模型"
echo "=================================="
echo "剪枝模型: $PRUNE_MODEL"
echo "输出目录: $OUTPUT_DIR"
echo "使用 GPU: $GPU_ID"
echo "=================================="

CUDA_VISIBLE_DEVICES=$GPU_ID python post_training.py \
    --prune_model "$PRUNE_MODEL" \
    --data_path yahma/alpaca-cleaned \
    --output_dir "$OUTPUT_DIR" \
    --disable_wandb \
    --num_epochs 3 \
    --learning_rate 3e-4 \
    --batch_size 128 \
    --micro_batch_size 4 \
    --cutoff_len 256 \
    --val_set_size 2000 \
    --lora_r 8 \
    --lora_alpha 16 \
    --lora_target_modules q_proj,k_proj,v_proj,o_proj,gate_proj,down_proj,up_proj \
    --extra_val_dataset wikitext2 \
    --cache_dataset

echo "=================================="
echo "微调完成！"
echo "模型保存在: $OUTPUT_DIR"
echo "=================================="

# 评估微调后的模型
echo "开始评估微调后的模型..."
python evaluate_finetuned.py \
    --model_path "$OUTPUT_DIR" \
    --device cuda:$GPU_ID

echo "全部完成！"
