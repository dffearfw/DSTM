import matplotlib.pyplot as plt
from pathlib import Path

# ============ 配置 ============
exp_dir = Path("/root/autodl-tmp/experiments/swe_full_temporal_20260609_085603")

# 输入：每折的 PNG 文件
fold_images = []
for fold in range(1, 11):
    img_path = exp_dir / f"density_scatter_chinese_fold_{fold}.png"
    if img_path.exists():
        fold_images.append(img_path)
    else:
        print(f"⚠ 警告: 折{fold}图片不存在: {img_path}")

if not fold_images:
    print("❌ 没有找到任何图片")
    exit()

# ============ 4行：前3行各3张，第4行第1列放Fold10 ============
fig = plt.figure(figsize=(12, 14))

# 前9张：3行3列
for i in range(9):
    ax = fig.add_subplot(4, 3, i+1)  # 位置 1-9
    img = plt.imread(fold_images[i])
    ax.imshow(img)
    ax.axis('off')
    ax.set_title(f'Fold {i+1}', fontsize=10)

# 第10张：第4行第1列（位置10）
ax = fig.add_subplot(4, 3, 10)
img = plt.imread(fold_images[9])
ax.imshow(img)
ax.axis('off')
ax.set_title(f'Fold 10', fontsize=10)

# 隐藏多余子图（位置11和12）
ax = fig.add_subplot(4, 3, 11)
ax.axis('off')
ax = fig.add_subplot(4, 3, 12)
ax.axis('off')

plt.tight_layout()
plt.savefig(exp_dir / "all_folds_combined.png", dpi=300, bbox_inches='tight')
plt.close()

print(f"✅ 已保存: {exp_dir / 'all_folds_combined.png'}")