#!/bin/bash
# 对比评估原始模型、剪枝模型和微调模型的PPL
#
# 使用相同的数据集（wikitext2）公平对比三个模型的性能

GPU_ID=0  # 根据需要修改

echo "=========================================="
echo "PPL 对比评估"
echo "=========================================="
echo "将对比以下三个模型的PPL："
echo "  1. 原始模型 (Llama-3-8B-Instruct)"
echo "  2. 剪枝模型 (GQA-aware v3)"
echo "  3. 微调模型 (LoRA微调后)"
echo ""
echo "数据集: wikitext2"
echo "序列长度: 128"
echo "设备: cuda:$GPU_ID"
echo "=========================================="
echo ""

CUDA_VISIBLE_DEVICES=$GPU_ID python compare_model_ppl.py \
    --original_model /newdata/LLMs/Llama-3-8B-Instruct \
    --pruned_model prune_log/llama3_gqa_aware_pruned_v3/pytorch_model.bin \
    --finetuned_dir ./finetuned_llama3_gqa_aware_v3 \
    --datasets wikitext2 \
    --seq_len 128 \
    --device cuda:$GPU_ID

echo ""
echo "=========================================="
echo "评估完成！"
echo "=========================================="
echo ""
echo "结果已保存到: ppl_comparison_results.json"
echo ""
