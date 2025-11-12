#!/bin/bash
# 剪枝分析示例脚本

echo "======================================"
echo "Llama-3-8B 剪枝分析示例"
echo "======================================"

# 设置路径（请根据实际情况修改）
ORIGINAL_MODEL="/newdata/LLMs/Llama-3-8B-Instruct"
PRUNED_MODEL="prune_log/llama_prune/pytorch_model.bin"
OUTPUT_CSV="pruning_analysis_llama3.csv"

# 检查文件是否存在
if [ ! -d "$ORIGINAL_MODEL" ]; then
    echo "错误：原始模型目录不存在: $ORIGINAL_MODEL"
    exit 1
fi

if [ ! -f "$PRUNED_MODEL" ]; then
    echo "错误：剪枝模型文件不存在: $PRUNED_MODEL"
    exit 1
fi

echo ""
echo "原始模型: $ORIGINAL_MODEL"
echo "剪枝模型: $PRUNED_MODEL"
echo "输出文件: $OUTPUT_CSV"
echo ""

# 运行分析
python analyze_pruning.py \
    --original_model "$ORIGINAL_MODEL" \
    --pruned_model "$PRUNED_MODEL" \
    --output "$OUTPUT_CSV"

echo ""
echo "======================================"
echo "分析完成！"
echo "======================================"
echo "结果已保存到: $OUTPUT_CSV"
echo ""
echo "你可以使用以下命令查看 CSV 文件："
echo "  head -20 $OUTPUT_CSV"
echo ""
echo "或使用 Python 进行进一步分析："
echo "  python -c \"import pandas as pd; df = pd.read_csv('$OUTPUT_CSV'); print(df.head(20))\""
