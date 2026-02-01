#!/bin/bash
# Simple bash script to run multiple strategies sequentially
# Usage: bash run_experiments.sh [data] [strategies...]
# Example: bash run_experiments.sh cifar100 entropy combined

DATA=${1:-cifar10}
shift

# Default to all strategies if none specified
if [ $# -eq 0 ]; then
    STRATEGIES=("random" "entropy" "least_confidence" "margin" "cp_size" "cp_v_shaped" "combined" "combined_v_shaped")
else
    STRATEGIES=("$@")
fi

echo "Running experiments on $DATA with strategies: ${STRATEGIES[*]}"

for strategy in "${STRATEGIES[@]}"; do
    echo ""
    echo "========================================"
    echo "Running: $strategy on $DATA"
    echo "========================================"
    python src/train.py data=$DATA strategy=$strategy
done

echo ""
echo "All experiments completed!"
echo "To plot results: python plot_results.py outputs/"
