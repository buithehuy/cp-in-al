"""Verify the quantile direction fix for RMCP."""
import torch
import numpy as np

print("=" * 70)
print("RMCP Quantile Direction - Verification")
print("=" * 70)

# Simulate RMCP scores
# High score = confident (easy to cover)
# Low score = uncertain (hard to cover)

torch.manual_seed(42)
n = 100
alpha = 0.1

# Simulate: 90% samples are confident (high scores), 10% uncertain (low scores)
high_scores = torch.rand(90) * 0.5 + 0.3  # [0.3, 0.8]
low_scores = torch.rand(10) * 0.3 - 0.2   # [-0.2, 0.1]
scores = torch.cat([high_scores, low_scores])
scores = scores[torch.randperm(n)]  # Shuffle

print(f"\nSimulated RMCP scores:")
print(f"  90% confident samples: scores in [0.3, 0.8]")
print(f"  10% uncertain samples: scores in [-0.2, 0.1]")
print(f"  Score range: [{scores.min():.2f}, {scores.max():.2f}]")

# OLD WAY (WRONG): Using (1-alpha) quantile
k_old = int(np.ceil((n + 1) * (1 - alpha)))
k_old = min(k_old - 1, n - 1)
qhat_old = torch.sort(scores)[0][k_old].item()

coverage_old = (scores >= qhat_old).float().mean().item()

print(f"\nOLD METHOD (WRONG - using 1-alpha = 90th percentile):")
print(f"  qhat = {qhat_old:.3f}")
print(f"  Coverage = {coverage_old:.1%}")
print(f"  Problem: qhat TOO HIGH! Only {coverage_old:.0%} of test samples covered")
print(f"  → Most test samples will have empty prediction sets!")

# NEW WAY (CORRECT): Using alpha quantile
k_new = int(np.ceil((n + 1) * alpha))
k_new = min(max(k_new - 1, 0), n - 1)
qhat_new = torch.sort(scores)[0][k_new].item()

coverage_new = (scores >= qhat_new).float().mean().item()

print(f"\nNEW METHOD (CORRECT - using alpha = 10th percentile):")
print(f"  qhat = {qhat_new:.3f}")
print(f"  Coverage = {coverage_new:.1%}")
print(f"  ✓ qhat is LOWER, allowing {coverage_new:.0%} of samples to be covered")
print(f"  → Properly achieves ~90% coverage!")

print("\n" + "=" * 70)
print("EXPLANATION:")
print("=" * 70)
print("RMCP score = p_true - mean(others)")
print("  High score (+0.7) → p_true >> mean → CONFIDENT → easy to cover")
print("  Low score (-0.2)  → p_true << mean → UNCERTAIN → hard to cover")
print()
print("This is OPPOSITE to standard CP where:")
print("  Standard CP score = 1 - p_true")
print("  High score → low p_true → UNCERTAIN")
print("  Low score → high p_true → CONFIDENT")
print()
print("Therefore RMCP needs ALPHA quantile (like APS), not (1-alpha)!")
print("=" * 70)
