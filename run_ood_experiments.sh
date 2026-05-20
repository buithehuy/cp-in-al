#!/bin/bash
# Run OOD noise experiments
# Usage:
#   bash run_ood_experiments.sh                     # All strategies, default ood_ratio=0.5, MNIST
#   bash run_ood_experiments.sh 0.3                  # Custom ood_ratio
#   bash run_ood_experiments.sh 0.5 SVHN             # Custom OOD source
#   bash run_ood_experiments.sh 0.5 MNIST entropy cp_size  # Specific strategies

OOD_RATIO=${1:-0.5}
OOD_DATASET=${2:-MNIST}
shift 2 2>/dev/null

# Default strategies if none specified
if [ $# -eq 0 ]; then
    STRATEGIES=("random" "entropy" "least_confidence" "margin" "cp_size" "cp_aps" "cp_v_shaped" "combined")
else
    STRATEGIES=("$@")
fi

echo "============================================================"
echo "OOD Noise Experiment"
echo "  OOD Source : $OOD_DATASET"
echo "  OOD Ratio  : $OOD_RATIO"
echo "  Strategies : ${STRATEGIES[*]}"
echo "============================================================"

for strategy in "${STRATEGIES[@]}"; do
    echo ""
    echo "========================================"
    echo "Running: $strategy (ood_ratio=$OOD_RATIO, ood=$OOD_DATASET)"
    echo "========================================"

    # --- Without CP correction ---
    python src/train.py \
        data=cifar10_ood \
        strategy=$strategy \
        data.ood_ratio=$OOD_RATIO \
        data.ood_dataset=$OOD_DATASET \
        output_dir="./outputs/ood_${OOD_DATASET}_${OOD_RATIO}/${strategy}"

    # --- With CP correction (only for CP strategies) ---
    if [[ $strategy == cp_* ]]; then
        echo "  >> Re-running with cp_correction=true"
        python src/train.py \
            data=cifar10_ood \
            strategy=$strategy \
            data.ood_ratio=$OOD_RATIO \
            data.ood_dataset=$OOD_DATASET \
            cp_correction=true \
            output_dir="./outputs/ood_${OOD_DATASET}_${OOD_RATIO}/${strategy}_corrected"
    fi
done

echo ""
echo "All OOD experiments completed!"
echo "Results saved in: outputs/ood_${OOD_DATASET}_${OOD_RATIO}/"
