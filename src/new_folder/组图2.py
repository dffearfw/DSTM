import matplotlib.pyplot as plt
from pathlib import Path

# ==========================
# 图片文件夹
# ==========================
img_dir = Path("/root/autodl-tmp/Untitled Folder")

# ==========================
# 图片及标题
# 注意：你的文件名是 piont.png，不是 point.png
# ==========================
images = [
    ("lastlayer.png", "(a) Output Layer FT"),
    ("fusion.png", "(b) Fusion-Layer FT"),
    ("piont.png", "(c) Point-Branch FT"),
    ("spatial.png", "(d) Spatial-Branch FT"),
    ("partial.png", "(e) Top-Layer FT"),
    ("none.png", "(f) Full FT"),
]

# ==========================
# 创建 2×3 组图
# ==========================
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

for ax, (filename, title) in zip(axes, images):
    img_path = img_dir / filename

    if img_path.exists():
        img = plt.imread(img_path)
        ax.imshow(img)
    else:
        print(f"⚠ Missing: {img_path}")

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.axis("off")

# ==========================
# 调整间距
# ==========================
plt.subplots_adjust(
    left=0.02,
    right=0.98,
    bottom=0.03,
    top=0.95,
    wspace=0.08,
    hspace=0.12
)

# ==========================
# 保存
# ==========================
save_path = img_dir / "all_strategy_scatter.png"

plt.savefig(
    save_path,
    dpi=600,
    bbox_inches="tight"
)

plt.close()

print(f"✅ 已保存: {save_path}")