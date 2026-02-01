# Active Learning with Conformal Prediction

This project implements Active Learning combined with Conformal Prediction for efficient data labeling on CIFAR-10/100.

## Installation

```bash
cd d:\Reseach\main\cp-in-al

# Install dependencies
pip install -r requirements.txt

# Install package in editable mode
pip install -e .
```

## Quick Start

### Single Experiment

Run a single strategy with default settings (CIFAR-10):

```bash
python src/train.py strategy=entropy
```

Run with CIFAR-100:

```bash
python src/train.py data=cifar100 strategy=entropy
```

### Common Use Cases

```bash
# Quick test (2 rounds, small data)
python src/train.py strategy=entropy num_rounds=2 initial_labeled=100 calibration_size=100 trainer.epochs_per_round=1

# CIFAR-100 with Combined strategy
python src/train.py data=cifar100 strategy=combined

# Custom hyperparameters
python src/train.py strategy=cp_v_shaped trainer.lr=0.001 trainer.epochs_per_round=20

# Override multiple settings
python src/train.py data=cifar100 strategy=combined num_rounds=10 budget_per_round=1000
```

### Run Multiple Strategies

**Linux/Mac:**
```bash
# All strategies on CIFAR-10
bash run_experiments.sh cifar10

# Specific strategies on CIFAR-100
bash run_experiments.sh cifar100 entropy combined cp_v_shaped
```

**Windows:**
```cmd
# All strategies on CIFAR-10
run_experiments.bat cifar10

# Specific strategies on CIFAR-100
run_experiments.bat cifar100 entropy combined cp_v_shaped
```

Or manually in a loop:
```bash
# Windows PowerShell
foreach ($s in @("entropy", "combined", "cp_v_shaped")) {
    python src/train.py data=cifar100 strategy=$s
}

# Linux/Mac
for s in entropy combined cp_v_shaped; do
    python src/train.py data=cifar100 strategy=$s
done
```

## Available Strategies

- **Uncertainty-based:**
  - `random` - Random sampling (baseline)
  - `entropy` - Entropy-based uncertainty
  - `least_confidence` - Least confidence sampling
  - `margin` - Margin sampling

- **Conformal Prediction:**
  - `cp_size` - CP prediction set size
  - `cp_v_shaped` - CP V-shaped scoring

- **Combined:**
  - `combined` - Entropy + CP Size
  - `combined_v_shaped` - Entropy + CP V-shaped

## Visualization

### Plot Results

After running experiments, visualize results:

```bash
# Plot all metrics from outputs directory
python plot_results.py outputs/

# Plot only accuracy
python plot_results.py outputs/ --plot accuracy

# Save plots without showing
python plot_results.py outputs/ --save-dir plots/ --no-show

# Print summary table only
python plot_results.py outputs/ --table-only
```

**Generated plots include:**
- Accuracy vs Training Samples
- Accuracy Gap vs Random Baseline
- AULC (Area Under Learning Curve) Comparison
- CP Coverage, Set Size, Zero Sets (optional)

## Configuration

### Dataset Configs (`configs/data/`)

- `cifar10.yaml` - CIFAR-10 (10 classes)
- `cifar100.yaml` - CIFAR-100 (100 classes)

### Model Configs (`configs/model/`)

- `resnet18.yaml` - ResNet18 architecture

### Strategy Configs (`configs/strategy/`)

One config file per acquisition strategy.

### Trainer Configs (`configs/trainer/`)

- `default.yaml` - Training hyperparameters

### Override Options

Key parameters you can override:

```bash
# Data
data=cifar10|cifar100                    # Dataset
initial_labeled=5000                     # Initial labeled samples
calibration_size=5000                    # Calibration set size
budget_per_round=2000                    # Samples to acquire per round
num_rounds=21                            # Number of AL rounds

# Training
trainer.epochs_per_round=10              # Epochs per round
trainer.lr=0.01                          # Learning rate
trainer.batch_size=128                   # Batch size

# Conformal Prediction
cp_alpha=0.1                             # CP significance level (1-coverage)

# Other
seed=42                                  # Random seed
save_results=true                        # Save results to disk
output_dir=outputs                       # Output directory
```

## Project Structure

```
├── configs/              # Hydra configuration files
│   ├── data/            # Dataset configs
│   ├── model/           # Model configs
│   ├── strategy/        # Acquisition strategy configs
│   ├── trainer/         # Training configs
│   └── train.yaml       # Main config
├── src/
│   ├── data/            # Data module
│   ├── models/          # Model definitions
│   ├── strategies/      # Acquisition strategies
│   ├── utils/           # Utilities (CP, training, visualization)
│   └── train.py         # Main training script
├── plot_results.py      # Plot saved results
├── run_experiments.sh   # Bash script for multiple runs
├── run_experiments.bat  # Windows script for multiple runs
├── requirements.txt
└── README.md
```

## Examples

### Full Experiment on CIFAR-10

```bash
# Run all 8 strategies
for s in random entropy least_confidence margin cp_size cp_v_shaped combined combined_v_shaped; do
    python src/train.py strategy=$s
done

# Plot results
python plot_results.py outputs/
```

### CIFAR-100 Comparison

```bash
# Compare best strategies on CIFAR-100
for s in entropy combined cp_v_shaped; do
    python src/train.py data=cifar100 strategy=$s
done

# Plot
python plot_results.py outputs/ --save-dir plots_cifar100/
```

### Quick Ablation Study

```bash
# Test different learning rates
for lr in 0.001 0.01 0.1; do
    python src/train.py strategy=combined trainer.lr=$lr num_rounds=5
done
```

## Development

See [DEVELOPER_GUIDE.md](DEVELOPER_GUIDE.md) for:
- Adding new datasets
- Creating custom strategies
- Extending the framework

## License

MIT
