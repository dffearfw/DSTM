import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os

# ============================================================
# 强制配置中文字体（WenQuanYi Zen Hei）
# ============================================================
def setup_chinese_font():
    font_name = 'WenQuanYi Zen Hei'
    plt.rcParams['font.sans-serif'] = [font_name]
    plt.rcParams['axes.unicode_minus'] = False
    
    try:
        fp = fm.FontProperties(family=font_name)
        if fp.get_name() != font_name:
            raise ValueError("Font not found")
        print(f"已设置中文字体：{font_name}")
        return True
    except:
        font_paths = [
            '/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc',
            '/usr/share/fonts/wqy-zenhei/wqy-zenhei.ttc',
            '/usr/share/fonts/truetype/wqy/wqy-zenhei.ttf'
        ]
        for path in font_paths:
            if os.path.exists(path):
                fm.fontManager.addfont(path)
                plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei']
                plt.rcParams['axes.unicode_minus'] = False
                print(f"通过路径 {path} 添加字体成功")
                return True
        
        print("未找到 WenQuanYi Zen Hei 字体文件")
        return False

has_chinese = setup_chinese_font()
title_text = '十折交叉验证 性能指标分布箱线图' if has_chinese else '10-Fold CV Metrics Boxplot'

# ============================================================
# 10 折交叉验证数据
# ============================================================
r2 = [
    0.929, 0.925, 0.94,
    0.94, 0.935, 0.939,
    0.941, 0.941, 0.941,
    0.937
]

mae = [
    0.89, 0.96, 0.83, 0.83,
    0.85, 0.84, 0.82, 0.83,
    0.81, 0.84
]

rmse = [
    3.02, 3.02, 2.71, 2.7,
    2.86, 2.79, 2.71, 2.72,
    2.7, 2.88
]

metrics = {
    'R²': r2,
    'MAE (mm)': mae,
    'RMSE (mm)': rmse
}

# 学术界常用高级莫兰迪/柔和配色
colors = ['#E2B13C', '#3CAEA3', '#F67280']

# 🔥 自定义中位数函数（偶数时取较小的那个实际值）
def get_actual_median(values):
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    if n % 2 == 1:
        return sorted_vals[n // 2]
    else:
        return sorted_vals[n // 2 - 1]

# ============================================================
# 绘制三个并列箱线图
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(15, 5.2)) # 微调高度，去掉说明后比例更紧凑
box_width = 0.45

for ax, (name, values), color in zip(axes, metrics.items(), colors):
    median_val = get_actual_median(values)
    
    # 优雅淡化背景网格，并置于底层（zorder=1）
    ax.grid(axis='y', linestyle=':', color='#CCCCCC', alpha=0.7, zorder=1)
    
    bp = ax.boxplot(
        values,
        labels=[name],
        showmeans=True,
        # 优化均值菱形：使用显眼饱满的红色，带白色边缘，使其更加有立体感
        meanprops=dict(marker='D', markerfacecolor='#D9383A', markersize=7, markeredgecolor='white', markeredgewidth=1, zorder=11),
        medianprops=dict(color='none'),  # 隐藏默认，交由下方紫线重绘
        whiskerprops=dict(color='#333333', linewidth=1.2, linestyle='-'),
        capprops=dict(color='#333333', linewidth=1.5),
        # 异常值圈：改用半透明深灰，略微缩小，更显内敛
        flierprops=dict(marker='o', markerfacecolor='#555555', markersize=5, markeredgecolor='none', alpha=0.6),
        showfliers=True,
        widths=box_width,
        patch_artist=True,
        zorder=2  # 让箱体盖在网格线上
    )
    
    # 设置箱体质感
    for box in bp['boxes']:
        box.set_facecolor(color)
        box.set_alpha(0.85)
        box.set_edgecolor('#222222')
        box.set_linewidth(1.5)
    
    # 🔥 手动重画紫色的中位数线（完美闭合，线宽加粗至 3.0）
    ax.hlines(y=median_val, 
              xmin=1 - box_width/2, 
              xmax=1 + box_width/2, 
              colors='#7D26CD', 
              linewidth=3.0, 
              zorder=10)
    
    # 视觉微调：去除上方和右方的多余边框线
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#888888')
    ax.spines['bottom'].set_color('#888888')
    
    ax.set_ylabel(name, fontsize=12, fontweight='bold', labelpad=8)
    ax.tick_params(axis='x', labelsize=12, bottom=False) # 隐藏横坐标无意义的刻度小短线
    ax.tick_params(axis='y', labelsize=10, colors='#333333')

    # 智能 Y 轴视窗缩放逻辑（针对每组数据提供黄金比例的上下空气感留白）
    v_min, v_max = min(values), max(values)
    v_range = v_max - v_min
    padding = v_range * 0.25 if v_range > 0 else 0.05
    ax.set_ylim(v_min - padding, v_max + padding)

# 全局标题美化（去掉了原有的底部 fig.text 说明段落）
fig.suptitle(title_text, fontsize=15, fontweight='bold', color='#111111', y=0.96)

plt.tight_layout()
# 优化边距平衡，因为没有底部说明，bottom 边距可以收紧
plt.subplots_adjust(bottom=0.08, top=0.88, wspace=0.3)
plt.savefig('boxplot_r2_mae_rmse_elegant.png', dpi=300, bbox_inches='tight')
plt.show()

# ============================================================
# 计算完整箱体要素并保存为 JSON
# ============================================================
def compute_box_stats(data_array):
    q1 = np.percentile(data_array, 25)
    q3 = np.percentile(data_array, 75)
    iqr = q3 - q1
    lower_fence = q1 - 1.5 * iqr
    upper_fence = q3 + 1.5 * iqr
    
    valid_below = data_array[data_array >= lower_fence]
    valid_above = data_array[data_array <= upper_fence]
    whisker_low = np.min(valid_below) if len(valid_below) > 0 else np.min(data_array)
    whisker_high = np.max(valid_above) if len(valid_above) > 0 else np.max(data_array)
    
    outliers = data_array[(data_array < whisker_low) | (data_array > whisker_high)].tolist()
    
    sorted_vals = sorted(data_array)
    n = len(sorted_vals)
    median_actual = sorted_vals[n // 2] if n % 2 == 1 else sorted_vals[n // 2 - 1]
    
    return {
        "min": float(np.min(data_array)),
        "max": float(np.max(data_array)),
        "mean": float(np.mean(data_array)),
        "median_actual": float(median_actual),
        "median_average": float(np.median(data_array)),
        "q1": float(q1),
        "q3": float(q3),
        "iqr": float(iqr),
        "lower_whisker": float(whisker_low),
        "upper_whisker": float(whisker_high),
        "outliers": outliers
    }

box_stats = {}
for name, values in metrics.items():
    box_stats[name] = compute_box_stats(np.array(values))

with open('boxplot_stats.json', 'w') as f:
    json.dump(box_stats, f, indent=4)

print("\n" + "="*50)
print("✅ 高级箱线图已成功保存为: boxplot_r2_mae_rmse_elegant.png")
print("📊 统计要素 JSON 已更新保存为: boxplot_stats.json")
print("="*50)