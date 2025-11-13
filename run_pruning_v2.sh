#!/bin/bash
# 重新剪枝 Llama-3-8B（使用改进后的脚本，支持 Attention 层 LoRA）

# 配置
BASE_MODEL="/newdata/LLMs/Llama-3-8B-Instruct"
OUTPUT_NAME="llama_unbalanced_prune_v2"  # Logger 会自动添加 prune_log/ 前缀

# 剪枝参数（与之前相同）
PRUNING_RATIO=0.25
MIN_PRUNING_RATE=0.0
MAX_PRUNING_RATE=0.3

echo "=================================="
echo "重新剪枝 Llama-3-8B"
echo "=================================="
echo "基础模型: $BASE_MODEL"
echo "输出名称: $OUTPUT_NAME"
echo "剪枝比例: $PRUNING_RATIO"
echo "最小剪枝率: $MIN_PRUNING_RATE"
echo "最大剪枝率: $MAX_PRUNING_RATE"
echo "=================================="

python llama3_unbalanced_pruning.py \
    --base_model $BASE_MODEL \
    --pruning_ratio $PRUNING_RATIO \
    --save_model \
    --save_ckpt_log_name $OUTPUT_NAME \
    --min_pruning_rate $MIN_PRUNING_RATE \
    --max_pruning_rate $MAX_PRUNING_RATE \
    --test_after_train

echo "=================================="
echo "剪枝完成！"
echo "模型保存在: prune_log/$OUTPUT_NAME/pytorch_model.bin"
echo "=================================="
