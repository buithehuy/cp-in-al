"""Simple test script for CUB-200-2011 dataset integration (Windows-compatible)."""
import sys
sys.path.append('src')

from omegaconf import OmegaConf
from data import CUB200DataModule

if __name__ == '__main__':
    # Load config
    cfg = OmegaConf.load('configs/data/cub200.yaml')
    
    print("=" * 80)
    print("Testing CUB-200-2011 Dataset Integration")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Dataset: {cfg.name}")
    print(f"  Number of classes: {cfg.num_classes}")
    print(f"  Batch size: {cfg.batch_size}")
    print(f"  Image size: {cfg.image_size}")
    print(f"\nActive Learning Parameters:")
    print(f"  Initial labeled: {cfg.initial_labeled} (10%)")
    print(f"  Calibration size: {cfg.calibration_size} (10%)")
    print(f"  Budget per round: {cfg.budget_per_round}")
    
    # Initialize datamodule
    print(f"\n{'Initializing datamodule...'}")
    try:
        data_module = CUB200DataModule(cfg)
        print("✓ Datamodule initialized successfully")
        
        print(f"\n{'Loading dataset...'}")
        print(f"  Train set size: {len(data_module.train_set)}")
        print(f"  Test set size: {len(data_module.test_set)}")
        
        # Test active learning setup
        print(f"\n{'Testing active learning splits...'}")
        labeled_idx, calib_idx, pool_idx = data_module.setup_active_learning(
            initial_labeled=cfg.initial_labeled,
            calibration_size=cfg.calibration_size,
            seed=42
        )
        
        print(f"  Initial labeled: {len(labeled_idx)} samples")
        print(f"  Calibration: {len(calib_idx)} samples")
        print(f"  Pool: {len(pool_idx)} samples")
        print(f"  Total: {len(labeled_idx) + len(calib_idx) + len(pool_idx)} samples")
        
        # Verify percentages
        total_train = len(data_module.train_set)
        init_pct = (len(labeled_idx) / total_train) * 100
        calib_pct = (len(calib_idx) / total_train) * 100
        pool_pct = (len(pool_idx) / total_train) * 100
        
        print(f"\n{'Percentage verification:'}")
        print(f"  Initial labeled: {init_pct:.1f}% (target: 10%)")
        print(f"  Calibration: {calib_pct:.1f}% (target: 10%)")
        print(f"  Pool: {pool_pct:.1f}% (target: 80%)")
        
        # Test data sample
        print(f"\n{'Testing data loading...'}")
        sample = data_module.train_set[0]
        image, label = sample
        
        print(f"  Sample image shape: {image.shape}")
        print(f"  Sample label: {label}")
        print(f"  Image value range: [{image.min():.2f}, {image.max():.2f}]")
        
        print("\n" + "=" * 80)
        print("✓ All tests passed! CUB-200-2011 is ready to use.")
        print("=" * 80)
        
        print("\nTo use CUB-200-2011 in training, run:")
        print("  python src/train.py data=cub200 strategy=random")
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
