"""Demo script to show how cp_v_shaped_entropy works.

This demonstrates the difference between cp_v_shaped and cp_v_shaped_entropy
for samples with zero setsize.
"""
import torch
import numpy as np
from src.strategies import get_strategy

# Create sample data
# Sample 1: Very confident (max prob = 0.99) -> low entropy
# Sample 2: Moderately uncertain (max prob = 0.6) -> medium entropy  
# Sample 3: Very uncertain (uniform distribution) -> high entropy
probs = torch.tensor([
    [0.99, 0.005, 0.005],  # Very confident
    [0.6, 0.3, 0.1],       # Moderately uncertain
    [0.33, 0.34, 0.33],    # Very uncertain (high entropy)
])

# Set qhat such that all samples have zero setsize
# (i.e., all probs < 1 - qhat)
qhat = 0.02  # This means prediction set includes probs >= 0.98

print("=" * 60)
print("Demo: cp_v_shaped vs cp_v_shaped_entropy")
print("=" * 60)
print("\nSample probabilities:")
for i, prob in enumerate(probs):
    entropy = -(prob * torch.log(prob + 1e-9)).sum()
    print(f"Sample {i}: {prob.numpy()} (entropy: {entropy:.3f})")

print(f"\nqhat = {qhat:.2f}")
print(f"Prediction sets threshold: {1-qhat:.2f}")

# Calculate set sizes
set_sizes = (probs >= (1 - qhat)).sum(dim=1)
print(f"\nSet sizes: {set_sizes.numpy()}")
print("All samples have zero setsize (overconfident errors)")

print("\n" + "-" * 60)

# Test cp_v_shaped strategy
cp_v_shaped = get_strategy("cp_v_shaped")
indices_v = cp_v_shaped.select(probs, budget=3, qhat=qhat)
print(f"\ncp_v_shaped selection order: {indices_v.numpy()}")
print("→ All zero-setsize samples have equal priority")

# Test cp_v_shaped_entropy strategy
cp_v_entropy = get_strategy("cp_v_shaped_entropy")
indices_ve = cp_v_entropy.select(probs, budget=3, qhat=qhat)
print(f"\ncp_v_shaped_entropy selection order: {indices_ve.numpy()}")
print("→ Zero-setsize samples ranked by entropy (highest first)")
print(f"   Expected: Sample 2 (highest entropy) should be prioritized")

print("\n" + "=" * 60)
print("Key difference:")
print("- cp_v_shaped: Treats all zero-setsize samples equally")
print("- cp_v_shaped_entropy: Prioritizes by entropy among zero-setsize")
print("=" * 60)
