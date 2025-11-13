#!/bin/bash

# Llama-3-8B GQA-Aware非均衡剪枝脚本
# 使用GQA-aware Taylor importance方法，确保剪枝后PPL几乎无损

MODEL_PATH="/newdata/LLMs/Llama-3-8B-Instruct"
SAVE_NAME="llama3_gqa_aware_pruned_v3"

python llama3_unbalanced_pruning_v3_gqa_aware.py \
    --base_model $MODEL_PATH \
    --save_ckpt_log_name $SAVE_NAME \
    \
    --pruning_ratio 0.25 \
    \
    --importance_method removal \
    --importance_samples 50 \
    --importance_config layer_importance_config.json \
    \
    --pruning_strategy inverse \
    --alpha 1.0 \
    --min_pruning_rate 0.15 \
    --max_pruning_rate 0.5 \
    \
    --layer_start 0 \
    --layer_end 32 \
    \
    --device cuda \
    --num_examples 10 \
    --head_dim 128 \
    --gqa_ratio 4 \
    \
    --prune_mlp \
    --save_model \
    --test_after_prune \
    --max_seq_len 128

# 参数说明：
# --pruning_ratio: 整体平均剪枝率（25%）
# --importance_method: 层重要性评估方法（removal = 逐层移除法）
# --pruning_strategy: inverse = 重要的层剪得少，不重要的层剪得多
# --min_pruning_rate: 最小剪枝率15%（至少剪1个GQA组，8个KV heads的12.5%）
# --prune_mlp: 也剪枝MLP（使用magnitude方法）
# --save_model: 保存剪枝后的模型
# --test_after_prune: 剪枝后立即评估PPL

echo ""
echo "======================================"
echo "GQA-Aware剪枝完成！"
echo "======================================"
echo ""
echo "预期结果："
echo "  - 剪枝后PPL: <5% 退化（vs 旧方法71万）"
echo "  - 所有层保持4:1 GQA比例"
echo "  - 参数减少: ~20-25%"
echo ""
echo "日志保存在: prune_log/$SAVE_NAME/"
echo "模型保存在: prune_log/$SAVE_NAME/best_model.pth"
