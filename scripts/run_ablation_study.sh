#!/bin/bash
# RFDM 消融实验脚本
# 用于系统化测试各个改进的效果

# 设置随机种子（确保可复现）
export PYTHONHASHSEED=2024

# 数据集
DATASET="amazon"
CUDA_DEVICE=0

# 实验配置
EXPERIMENTS=(
    "baseline:rfdm_ablation:RFDMBaselineStrategy"
    "reflow_only:rfdm_ablation:RFDMReflowOnlyStrategy"
    "adaptive_only:rfdm_ablation:RFDMAdaptiveOnlyStrategy"
    "vectorfield_only:rfdm_ablation:RFDMVectorFieldOnlyStrategy"
    "reflow_adaptive:rfdm_ablation:RFDMReflowAdaptiveStrategy"
    "all:rfdm:RectifiedFMDMStrategy"
)

# 结果文件
RESULTS_FILE="ablation_results.md"

echo "# RFDM Ablation Study Results" > $RESULTS_FILE
echo "" >> $RESULTS_FILE
echo "Date: $(date)" >> $RESULTS_FILE
echo "" >> $RESULTS_FILE
echo "| Experiment | Recall@10 | Recall@20 | NDCG@10 | NDCG@20 | Best Epoch |" >> $RESULTS_FILE
echo "|------------|-----------|-----------|---------|---------|------------|" >> $RESULTS_FILE

# 运行每个实验
for exp in "${EXPERIMENTS[@]}"; do
    IFS=':' read -r exp_name strategy_module strategy_class <<< "$exp"
    
    echo "========================================="
    echo "Running experiment: $exp_name"
    echo "Strategy: $strategy_module.$strategy_class"
    echo "========================================="
    
    # 创建临时配置文件或使用环境变量
    # 这里假设可以通过修改策略名称来控制
    
    # 运行训练
    python train_encoder.py \
        --base_model mult_vae \
        --strategy $strategy_module \
        --strategy_class $strategy_class \
        --dataset $DATASET \
        --cuda $CUDA_DEVICE \
        > "logs/${exp_name}.log" 2>&1
    
    # 提取结果（需要根据实际日志格式调整）
    # 这里需要解析日志文件提取指标
    echo "Extracting results from logs/${exp_name}.log..."
    
    # 示例：从日志中提取结果（需要根据实际格式调整）
    # recall_10=$(grep "recall" logs/${exp_name}.log | ...)
    # 将结果追加到结果文件
    
    echo "Experiment $exp_name completed"
    echo ""
done

echo "All experiments completed!"
echo "Results saved to $RESULTS_FILE"

