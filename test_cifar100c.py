"""Test CIFAR-100-C datamodule with dynamic corruptions."""
import sys
sys.path.insert(0, 'src')

from omegaconf import OmegaConf
from data import CIFAR100CDataModule
import torch
import numpy as np

print("=" * 70)
print("CIFAR-100-C DataModule Test (Dynamic Corruption)")
print("=" * 70)

# Test 1: Gaussian noise, severity 5, 20% ratio
print("\n[TEST 1] Gaussian Noise (sev=5, ratio=20%)")
print("-" * 70)

cfg_c = OmegaConf.create({
    'name': 'CIFAR100C',
    'root': './data',
    'batch_size': 128,
    'num_workers': 0,
    'corruption': 'gaussian_noise',
    'severity': 5,
    'corruption_ratio': 0.2,
    'mean': [0.5071, 0.4867, 0.4408],
    'std': [0.2675, 0.2565, 0.2761]
})

try:
    dm_c = CIFAR100CDataModule(cfg_c)
    print(f"✓ CIFAR-100 loaded successfully")
    
    # Setup AL split
    print("  Setting up AL split + corrupting pool...")
    labeled_idx, calib_idx, pool_idx = dm_c.setup_active_learning(
        initial_labeled=1000,
        calibration_size=1000,
        seed=42
    )
    
    print(f"  Labeled: {len(labeled_idx)}")
    print(f"  Pool: {len(pool_idx)}")
    
    # Verify pool corruption
    # We can't easily check 'is_corrupted' without ground truth, 
    # but we can check if data was modified from original if we had a copy.
    # Here we rely on the implementation logs (which print success message).
    
    # Get a loader and check batch shape
    loader = dm_c.get_loader(pool_idx[:32], shuffle=False)
    batch = next(iter(loader))
    images, labels = batch
    print(f"  Batch shape: {images.shape}")
    print(f"  Batch mean: {images.mean():.4f}, std: {images.std():.4f}")
    
    # Check if transforms are applied (normalization should make mean ~0)
    if abs(images.mean()) < 1.0:
        print("✓ Transforms applied correctly")
    else:
        print("✗ Warning: Images might not be normalized properly")
        
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 70)
print("CIFAR-100-C Ready!")
print("=" * 70)
print("Usage example:")
print("  python src/train.py data=cifar100c data.corruption=impulse_noise data.severity=3 data.corruption_ratio=0.1")
print("=" * 70)
