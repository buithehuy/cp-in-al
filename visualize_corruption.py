"""
Visualize CIFAR-10 Clean vs Corrupted (50% noise injection).

Hiển thị 2 hàng:
  Row 1 - Clean  : ảnh CIFAR-10 gốc
  Row 2 - Corrupted: cùng ảnh đó sau khi bị gaussian_noise severity=5

Corruption được inject giống hệt logic trong CIFAR10CDataModule:
  - 50% pool ngẫu nhiên bị corrupt (corruption_ratio=0.5)
  - imagecorruptions.corrupt()
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from torchvision import datasets
from imagecorruptions import corrupt

# ── Config ───────────────────────────────────────────────────────────────────
DATA_ROOT       = "./data"
CORRUPTION_NAME = "gaussian_noise"
SEVERITY        = 5          # 1‑5, severity 5 = nặng nhất ≈ 50% quality loss
N_SHOW          = 8          # số ảnh hiển thị
SEED            = 42
CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

# ── Load CIFAR-10 (raw uint8 numpy, không normalize) ────────────────────────
dataset = datasets.CIFAR10(root=DATA_ROOT, train=True, download=True, transform=None)

rng = np.random.default_rng(SEED)
indices = rng.choice(len(dataset), N_SHOW, replace=False)

clean_imgs  = []
corrupt_imgs = []
labels       = []

for idx in indices:
    img, label = dataset[idx]               # PIL Image, int
    img_np = np.array(img)                  # uint8 [32,32,3]
    clean_imgs.append(img_np)
    corrupt_imgs.append(
        corrupt(img_np, severity=SEVERITY, corruption_name=CORRUPTION_NAME)
    )
    labels.append(CIFAR10_CLASSES[label])

# ── Plot ─────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(
    2, N_SHOW,
    figsize=(N_SHOW * 1.6, 4),
    gridspec_kw={"hspace": 0.05, "wspace": 0.05}
)

for col in range(N_SHOW):
    # Row 0 – Clean
    axes[0, col].imshow(clean_imgs[col])
    axes[0, col].axis("off")
    axes[0, col].set_title(labels[col], fontsize=8, pad=2)

    # Row 1 – Corrupted
    axes[1, col].imshow(corrupt_imgs[col])
    axes[1, col].axis("off")

# Row labels bên trái
for row_idx, row_label in enumerate(["Clean", "Corrupted\n(50% noise)"]):
    axes[row_idx, 0].set_ylabel(
        row_label, fontsize=10, fontweight="bold",
        rotation=90, labelpad=6, va="center"
    )
    axes[row_idx, 0].yaxis.set_label_position("left")
    axes[row_idx, 0].tick_params(left=False, labelleft=False)

fig.suptitle(
    f"CIFAR-10  |  Corruption: {CORRUPTION_NAME}  |  Severity: {SEVERITY}/5  |  50% pool corrupted",
    fontsize=11, y=1.01
)

plt.savefig("cifar10_corruption_preview.png", dpi=150, bbox_inches="tight")
print("✓ Saved → cifar10_corruption_preview.png")
plt.show()
