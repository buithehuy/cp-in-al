# Active Learning with Conformal Prediction

This project implements Active Learning combined with Conformal Prediction for efficient data labeling on CIFAR-10.

## Installation

```bash
# Clone repository
cd d:\Reseach\main\cp-in-al

# Install dependencies
pip install -r requirements.txt

# Install package in editable mode
pip install -e .
```

## Usage

### Basic Training

Run with default configuration (Entropy strategy):

```bash
python src/train.py
```

### Change Acquisition Strategy

```bash
# Random baseline
python src/train.py strategy=random

# Uncertainty-based
python src/train.py strategy=entropy
python src/train.py strategy=least_confidence
python src/train.py strategy=margin

# Conformal Prediction-based
python src/train.py strategy=cp_size
python src/train.py strategy=cp_v_shaped

# Combined
python src/train.py strategy=combined
python src/train.py strategy=combined_v_shaped
```

### Run Multiple Strategies

Use `run_all_strategies.py` to run multiple strategies automatically:

```bash
# Run specific strategies (recommended for comparison)
python run_all_strategies.py -s entropy combined cp_v_shaped

# Run all 8 strategies
python run_all_strategies.py --all

# Quick test with specific strategies (2 rounds, small data)
python run_all_strategies.py -s entropy cp_v_shaped --quick

# Custom number of rounds
python run_all_strategies.py -s entropy combined --num_rounds 10

# See all options
python run_all_strategies.py --help
```

**Note**: Strategies run sequentially. For 8 strategies × 21 rounds, expect ~2-4 hours on GPU.

### Quick Test

Run a quick test with 2 rounds and small data:

```bash
python src/train.py num_rounds=2 initial_labeled=500 budget_per_round=200 trainer.epochs_per_round=2
```

### Override Configuration

```bash
# Change hyperparameters
python src/train.py num_rounds=10 budget_per_round=1000 trainer.lr=0.001

# Change CP alpha
python src/train.py cp_alpha=0.05

# Disable pretrained weights
python src/train.py model.pretrained=false
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
│   ├── utils/           # Utilities (CP, training)
│   └── train.py         # Main training script
├── requirements.txt
├── setup.py
└── README.md
```

## Strategies

### Uncertainty-based
- **Random**: Random sampling baseline
- **Entropy**: Select samples with highest prediction entropy
- **Least Confidence**: Select samples with lowest confidence
- **Margin**: Select samples with smallest margin between top-2 predictions

### Conformal Prediction-based
- **CP Size**: Select samples with largest prediction sets
- **CP V-shaped**: Prioritize both empty sets (overconfident errors) and large sets
- **Combined**: Entropy + CP Size (equal weighting)
- **Combined V-shaped**: Entropy + V-shaped CP score

## Results

Expected results on CIFAR-10 (from experiments):
- All strategies reach ~95% accuracy with full dataset (45k samples)
- **Best strategies**: Combined, Entropy, Combined V-shaped
- CP-based methods reach target accuracies faster (fewer labeled samples needed)

## Citation

If you use this code, please cite:
```
@software{cp_in_al_2026,
  title={Active Learning with Conformal Prediction},
  author={Your Name},
  year={2026}
}
```

## Documentation

- **[README.md](README.md)** - Quick start guide
- **[NOTES.md](NOTES.md)** - Usage tips and troubleshooting (Vietnamese)
- **[DEVELOPER_GUIDE.md](DEVELOPER_GUIDE.md)** - Extending the project (datasets, strategies, CP methods)

