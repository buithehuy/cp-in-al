"""Demo showing RMCP score calculation with intuitive examples."""
import torch

print("=" * 70)
print("RMCP (Relative Margin CP) - Score Intuition")
print("=" * 70)

# Example 1: High confidence sample
print("\n[Example 1] High Confidence Sample")
print("-" * 70)
probs1 = torch.tensor([0.8, 0.1, 0.05, 0.05])
print(f"Probabilities: {probs1.tolist()}")

for k in range(4):
    other_probs = torch.cat([probs1[:k], probs1[k+1:]])
    score = probs1[k] - other_probs.mean()
    print(f"  Class {k}: p={probs1[k]:.2f}, mean_others={other_probs.mean():.3f}, score={score:+.3f}")

print("\nInterpretation:")
print("  Class 0 has LARGE positive margin (+0.733) → very confident")
print("  Other classes have negative margins → not predicted")

# Example 2: Low confidence (uniform)
print("\n[Example 2] Low Confidence (Uniform)")
print("-" * 70)
probs2 = torch.tensor([0.25, 0.25, 0.25, 0.25])
print(f"Probabilities: {probs2.tolist()}")

for k in range(4):
    other_probs = torch.cat([probs2[:k], probs2[k+1:]])
    score = probs2[k] - other_probs.mean()
    print(f"  Class {k}: p={probs2[k]:.2f}, mean_others={other_probs.mean():.3f}, score={score:+.3f}")

print("\nInterpretation:")
print("  All classes have score ≈ 0 → very uncertain")
print("  Model cannot distinguish between classes")

# Example 3: Two strong candidates
print("\n[Example 3] Two Strong Candidates")
print("-" * 70)
probs3 = torch.tensor([0.45, 0.45, 0.05, 0.05])
print(f"Probabilities: {probs3.tolist()}")

for k in range(4):
    other_probs = torch.cat([probs3[:k], probs3[k+1:]])
    score = probs3[k] - other_probs.mean()
    print(f"  Class {k}: p={probs3[k]:.2f}, mean_others={other_probs.mean():.3f}, score={score:+.3f}")

print("\nInterpretation:")
print("  Class 0 & 1 have moderate positive margins (+0.267)")
print("  Class 2 & 3 have negative margins")
print("  → Model uncertain between top 2 classes")

# Comparison with standard CP
print("\n" + "=" * 70)
print("COMPARISON: RMCP vs Standard CP")
print("=" * 70)

examples = [
    ("High Confidence", torch.tensor([0.8, 0.1, 0.05, 0.05])),
    ("Uniform", torch.tensor([0.25, 0.25, 0.25, 0.25])),
    ("Two Candidates", torch.tensor([0.45, 0.45, 0.05, 0.05])),
]

for name, probs in examples:
    print(f"\n{name}: {probs.tolist()}")
    
    # Standard CP score (1 - p_max)
    cp_score = 1 - probs.max()
    print(f"  Standard CP:  1 - p_max = {cp_score:.3f}")
    
    # RMCP score (p_max - mean_others)
    max_idx = probs.argmax()
    other_probs = torch.cat([probs[:max_idx], probs[max_idx+1:]])
    rmcp_score = probs[max_idx] - other_probs.mean()
    print(f"  RMCP:         p_max - mean(others) = {rmcp_score:.3f}")
    
    print(f"  → Higher score = more confident in both methods")

print("\n" + "=" * 70)
print("KEY INSIGHT:")
print("=" * 70)
print("RMCP considers the DISTRIBUTION of probabilities, not just max prob")
print("  - Uniform [0.25, 0.25, 0.25, 0.25]: RMCP ≈ 0")
print("  - Strong [0.8, 0.1, 0.05, 0.05]:    RMCP ≈ +0.73")
print("  - Bimodal [0.45, 0.45, 0.05, 0.05]: RMCP ≈ +0.27 (intermediate)")
print("\nThis makes RMCP sensitive to multi-modal distributions!")
print("=" * 70)
