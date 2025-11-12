#!/bin/bash
# Llama-3 非均衡结构化剪枝示例脚本

echo "=========================================="
echo "Llama-3 非均衡结构化剪枝示例"
echo "=========================================="

# 设置路径（请根据实际情况修改）
ORIGINAL_MODEL="/newdata/LLMs/Llama-3-8B-Instruct"
LOG_NAME="llama_unbalanced_prune_example"

echo ""
echo "原始模型: $ORIGINAL_MODEL"
echo "日志目录: prune_log/$LOG_NAME"
echo ""

# ==================== 示例 1: 完整流程（推荐） ====================
echo "=========================================="
echo "示例 1: 完整的非均衡剪枝流程"
echo "=========================================="
echo ""
echo "步骤："
echo "1. 评估层重要性（使用层移除法）"
echo "2. 计算非均衡剪枝率"
echo "3. 执行结构化剪枝"
echo "4. 评估 PPL"
echo "5. 保存模型"
echo ""
read -p "是否执行示例 1? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]
then
    python llama3_unbalanced_pruning.py \
        --base_model "$ORIGINAL_MODEL" \
        --save_ckpt_log_name "$LOG_NAME" \
        --pruning_ratio 0.25 \
        --importance_method removal \
        --importance_samples 50 \
        --pruning_strategy inverse \
        --alpha 1.5 \
        --min_pruning_rate 0.05 \
        --max_pruning_rate 0.6 \
        --save_model \
        --test_after_train
fi

# ==================== 示例 2: 快速测试（使用激活值法） ====================
echo ""
echo "=========================================="
echo "示例 2: 快速测试（使用激活值法）"
echo "=========================================="
echo ""
echo "更快但可能不够准确"
echo ""
read -p "是否执行示例 2? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]
then
    python llama3_unbalanced_pruning.py \
        --base_model "$ORIGINAL_MODEL" \
        --save_ckpt_log_name "${LOG_NAME}_fast" \
        --pruning_ratio 0.25 \
        --importance_method activation \
        --importance_samples 20 \
        --pruning_strategy inverse \
        --alpha 1.0 \
        --save_model
fi

# ==================== 示例 3: 使用已保存的配置 ====================
echo ""
echo "=========================================="
echo "示例 3: 使用已保存的重要性配置"
echo "=========================================="
echo ""
echo "跳过层重要性分析，直接使用之前的结果"
echo ""
read -p "是否执行示例 3? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]
then
    IMPORTANCE_CONFIG="prune_log/$LOG_NAME/layer_importance_config.json"

    if [ -f "$IMPORTANCE_CONFIG" ]; then
        python llama3_unbalanced_pruning.py \
            --base_model "$ORIGINAL_MODEL" \
            --save_ckpt_log_name "${LOG_NAME}_reuse" \
            --pruning_ratio 0.25 \
            --skip_importance_analysis \
            --importance_config "$IMPORTANCE_CONFIG" \
            --pruning_strategy inverse \
            --alpha 2.0 \
            --save_model \
            --test_after_train
    else
        echo "错误: 配置文件不存在: $IMPORTANCE_CONFIG"
        echo "请先运行示例 1"
    fi
fi

# ==================== 示例 4: 对比不同 alpha 值 ====================
echo ""
echo "=========================================="
echo "示例 4: 对比不同 alpha 值的影响"
echo "=========================================="
echo ""
echo "测试 alpha = 0.5, 1.0, 1.5, 2.0"
echo ""
read -p "是否执行示例 4? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]
then
    IMPORTANCE_CONFIG="prune_log/$LOG_NAME/layer_importance_config.json"

    if [ -f "$IMPORTANCE_CONFIG" ]; then
        for alpha in 0.5 1.0 1.5 2.0
        do
            echo ""
            echo "测试 alpha = $alpha..."
            python llama3_unbalanced_pruning.py \
                --base_model "$ORIGINAL_MODEL" \
                --save_ckpt_log_name "${LOG_NAME}_alpha_${alpha}" \
                --pruning_ratio 0.25 \
                --skip_importance_analysis \
                --importance_config "$IMPORTANCE_CONFIG" \
                --pruning_strategy inverse \
                --alpha $alpha \
                --save_model \
                --test_after_train
        done

        echo ""
        echo "=========================================="
        echo "对比结果（查看各个日志目录的 PPL）"
        echo "=========================================="
        find prune_log/${LOG_NAME}_alpha_* -name "training.log" -exec echo {} \; -exec grep "剪枝后 PPL" {} \;
    else
        echo "错误: 配置文件不存在: $IMPORTANCE_CONFIG"
        echo "请先运行示例 1"
    fi
fi

# ==================== 后续分析 ====================
echo ""
echo "=========================================="
echo "后续分析建议"
echo "=========================================="
echo ""
echo "1. 检查模型大小:"
echo "   python check_model_size.py --original_model $ORIGINAL_MODEL --pruned_model prune_log/$LOG_NAME/pytorch_model.bin"
echo ""
echo "2. 详细剪枝分析:"
echo "   python analyze_pruning.py --original_model $ORIGINAL_MODEL --pruned_model prune_log/$LOG_NAME/pytorch_model.bin"
echo ""
echo "3. 查看可视化结果:"
echo "   ls -lh prune_log/$LOG_NAME/pruning_strategy.png"
echo "   # 使用图片查看器打开"
echo ""
echo "4. 查看剪枝配置:"
echo "   cat prune_log/$LOG_NAME/layer_importance_config.json | python -m json.tool"
echo ""
echo "完成！"
