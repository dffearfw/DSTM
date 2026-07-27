#!/bin/bash

# ============================================
# SWE模型多策略对比实验 - 多随机种子版本
# ============================================

PRETRAINED_MODEL="/root/autodl-tmp/experiments/swe_full_temporal_20260609_085603/best_model.pth"
STATION_DATA="/root/autodl-tmp/combined_station.csv"

# ============ 定义不同的随机种子 ============
SEEDS=(


    2521011747
)

# ============ 固定划分比例 ============
TEST_RATIO=0.10
VAL_RATIO=0.20

# ============ 策略列表 ============
STRATEGIES=(
    "fusion_ft"
    "point_ft"
    "spatial_ft"
    "partial"
    "none"
)

# ============ 学习率配置 ============
declare -A LR_HEAD=(
    ["fusion_ft"]="5e-4"
    ["point_ft"]="5e-4"
    ["spatial_ft"]="1e-4"
    ["partial"]="6e-4"
    ["none"]="5e-4"
)

declare -A LR_TRANS=(
    ["fusion_ft"]="5e-4"
    ["point_ft"]="5e-4"
    ["spatial_ft"]="1e-4"
    ["partial"]="4e-5"
    ["none"]="3e-5"
)

declare -A LR_ENC=(
    ["fusion_ft"]=""
    ["point_ft"]="1e-4"
    ["spatial_ft"]="1e-4"
    ["partial"]="1e-6"
    ["none"]="3e-5"
)

# ============ 冻结主干配置 ============
declare -A FREEZE_BACKBONE=(
    ["fusion_ft"]="--freeze_backbone"
    ["point_ft"]="--freeze_backbone"
    ["spatial_ft"]="--freeze_backbone"
    ["partial"]="--freeze_backbone"
    ["none"]=""
)

# ============ 统一配置 ============
EPOCHS=45
BATCH_SIZE=32
STATION_RATIO=0.7
NUM_WORKERS=8

# 🔥 统一实验根目录（所有策略的文件夹都放这里）
EXPERIMENTS_BASE_DIR="/root/autodl-tmp/experiments"
mkdir -p ${EXPERIMENTS_BASE_DIR}

# 🔥 本次实验的统一时间戳
MAIN_TIMESTAMP=$(date +'%Y%m%d_%H%M%S')

# 🔥 本次实验的根目录
RUN_ROOT_DIR="${EXPERIMENTS_BASE_DIR}/run_${MAIN_TIMESTAMP}"
mkdir -p ${RUN_ROOT_DIR}

BASE_LOG_DIR="./training_runs"
mkdir -p ${BASE_LOG_DIR}

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo "=========================================="
echo "🚀 SWE模型多策略对比实验"
echo "   实验根目录: ${RUN_ROOT_DIR}"
echo "   时间戳: ${MAIN_TIMESTAMP}"
echo "=========================================="

# ============ 主循环：遍历不同的随机种子 ============
TOTAL_SEEDS=${#SEEDS[@]}
CURRENT_SEED_IDX=0

for SEED in "${SEEDS[@]}"; do
    CURRENT_SEED_IDX=$((CURRENT_SEED_IDX + 1))
    
    echo ""
    echo "██████████████████████████████████████████████████████████████████████████████"
    echo "███  🎲 随机种子 ${CURRENT_SEED_IDX}/${TOTAL_SEEDS}: SEED=${SEED}  ███"
    echo "███  测试集比例=${TEST_RATIO}, 验证集比例=${VAL_RATIO}  ███"
    echo "██████████████████████████████████████████████████████████████████████████████"
    echo ""
    
    # 🔥 为当前随机种子创建独立的缓存目录
    SPLIT_CACHE_DIR="./split_cache/seed_${SEED}"
    SPLIT_CACHE_FILE="${SPLIT_CACHE_DIR}/shared_split.pkl"
    mkdir -p ${SPLIT_CACHE_DIR}
    
    # 🔥 日志文件：包含时间戳和种子
    LOG_FILE="${BASE_LOG_DIR}/seed_${SEED}_${MAIN_TIMESTAMP}.log"
    
    echo "=========================================="
    echo "🚀 SWE模型多策略对比实验"
    echo "   随机种子: ${SEED}"
    echo "   测试集比例: ${TEST_RATIO}"
    echo "   验证集比例: ${VAL_RATIO}"
    echo "   统一轮次: ${EPOCHS}"
    echo "   策略数量: ${#STRATEGIES[@]}"
    echo "   日志文件: ${LOG_FILE}"
    echo "   🔥 划分缓存: ${SPLIT_CACHE_FILE}"
    echo "=========================================="
    
    # 初始化日志文件
    {
        echo "=========================================="
        echo "SWE模型多策略对比实验"
        echo "实验ID: ${MAIN_TIMESTAMP}"
        echo "随机种子: ${SEED}"
        echo "测试集比例: ${TEST_RATIO}"
        echo "验证集比例: ${VAL_RATIO}"
        echo "开始时间: $(date '+%Y-%m-%d %H:%M:%S')"
        echo "=========================================="
        echo ""
    } > ${LOG_FILE}
    
    START_TIME=$(date +%s)
    FIRST_STRATEGY=true
    
    # ============ 内层循环：遍历所有策略 ============
    for idx in "${!STRATEGIES[@]}"; do
        STRATEGY="${STRATEGIES[$idx]}"
        
        echo ""
        echo "=========================================="
        echo -e "${BLUE}📊 [SEED=${SEED}] 运行策略: ${STRATEGY} (第 $((idx+1))/${#STRATEGIES[@]})${NC}"
        echo "=========================================="
        
        # 🔥 为每个策略创建独立的实验目录（按种子分组）
        STRATEGY_EXP_DIR="${RUN_ROOT_DIR}/seed_${SEED}/${STRATEGY}"
        mkdir -p ${STRATEGY_EXP_DIR}
        
        # 写入日志
        {
            echo ""
            echo "=========================================="
            echo "📊 策略: ${STRATEGY} (第 $((idx+1))/${#STRATEGIES[@]})"
            echo "开始时间: $(date '+%Y-%m-%d %H:%M:%S')"
            echo "实验目录: ${STRATEGY_EXP_DIR}"
            echo "=========================================="
        } >> ${LOG_FILE}
        
        # 构建学习率参数
        LR_ARGS=""
        [ -n "${LR_HEAD[$STRATEGY]}" ] && LR_ARGS="${LR_ARGS} --lr_head ${LR_HEAD[$STRATEGY]}"
        [ -n "${LR_TRANS[$STRATEGY]}" ] && LR_ARGS="${LR_ARGS} --lr_transformer ${LR_TRANS[$STRATEGY]}"
        [ -n "${LR_ENC[$STRATEGY]}" ] && LR_ARGS="${LR_ARGS} --lr_encoder ${LR_ENC[$STRATEGY]}"
        
        # 🔥 构建命令 - 添加 --save_dir 参数
        CMD="python main_tune.py \
            --mode fine_tune \
            --model_type full \
            --batch_size ${BATCH_SIZE} \
            --fine_tune_epochs ${EPOCHS} \
            ${FREEZE_BACKBONE[$STRATEGY]} \
            --freeze_strategy ${STRATEGY} \
            --pretrained_model ${PRETRAINED_MODEL} \
            --station_data_path ${STATION_DATA} \
            --mixed_mode \
            --use_amp \
            --station_ratio ${STATION_RATIO} \
            ${LR_ARGS} \
            --cv_mode station_cv \
            --seed ${SEED} \
            --num_workers ${NUM_WORKERS} \
            --split_cache_file ${SPLIT_CACHE_FILE} \
            --save_dir ${STRATEGY_EXP_DIR}"
        
        # 第一个策略强制重新计算划分
        if [ "$FIRST_STRATEGY" = true ]; then
            CMD="${CMD} --force_recompute_split"
            echo "   🔥 第一个策略将计算并保存划分 (seed=${SEED})"
            FIRST_STRATEGY=false
        else
            echo "   📦 复用已有划分缓存"
        fi
        
        echo "执行命令: ${CMD}"
        echo "实验目录: ${STRATEGY_EXP_DIR}"
        echo ""
        
        # 写入日志并执行
        {
            echo "命令: ${CMD}"
            echo "实验目录: ${STRATEGY_EXP_DIR}"
            echo ""
            echo "========== 训练输出 =========="
        } >> ${LOG_FILE}
        
        eval ${CMD} 2>&1 | tee -a "${LOG_FILE}"
        
        {
            echo ""
            echo "========== 策略 ${STRATEGY} 结束 =========="
            echo "结束时间: $(date '+%Y-%m-%d %H:%M:%S')"
            echo ""
        } >> ${LOG_FILE}
        
        if [ ${PIPESTATUS[0]} -eq 0 ]; then
            echo -e "${GREEN}✅ [SEED=${SEED}] 策略 ${STRATEGY} 成功完成${NC}"
        else
            echo -e "${RED}❌ [SEED=${SEED}] 策略 ${STRATEGY} 失败${NC}"
        fi
        
        # 策略之间等待10秒
        if [ $idx -ne $((${#STRATEGIES[@]} - 1)) ]; then
            echo ""
            echo -e "${YELLOW}⏳ 等待 10 秒后开始下一个策略...${NC}"
            sleep 10
        fi
        echo ""
    done
    
    END_TIME=$(date +%s)
    TOTAL_DURATION=$((END_TIME - START_TIME))
    
    # 追加汇总信息到日志末尾
    {
        echo ""
        echo "=========================================="
        echo "📊 实验汇总 (SEED=${SEED})"
        echo "=========================================="
        echo "总耗时: $((TOTAL_DURATION / 60))分 $((TOTAL_DURATION % 60))秒"
        echo "结束时间: $(date '+%Y-%m-%d %H:%M:%S')"
        echo "=========================================="
        echo ""
        echo "策略执行结果:"
        for STRATEGY in "${STRATEGIES[@]}"; do
            R2=$(grep -E "R²: [0-9.]+" "${LOG_FILE}" | grep -A2 "策略: ${STRATEGY}" | tail -1 | grep -oP "R²: \K[0-9.]+" || echo "N/A")
            RMSE=$(grep -E "RMSE: [0-9.]+" "${LOG_FILE}" | grep -A2 "策略: ${STRATEGY}" | tail -1 | grep -oP "RMSE: \K[0-9.]+" || echo "N/A")
            printf "  %-20s R²=%s  RMSE=%s\n" "${STRATEGY}" "${R2}" "${RMSE}"
        done
        echo "=========================================="
    } >> ${LOG_FILE}
    
    echo ""
    echo "📁 日志文件: ${LOG_FILE}"
    echo "📦 划分缓存: ${SPLIT_CACHE_FILE}"
    echo "📁 实验目录: ${RUN_ROOT_DIR}/seed_${SEED}/"
    echo "=========================================="
    
    # 种子之间等待15秒
    if [ $CURRENT_SEED_IDX -lt $TOTAL_SEEDS ]; then
        echo ""
        echo -e "${YELLOW}⏳ 等待 15 秒后开始下一个随机种子...${NC}"
        sleep 15
    fi
    
done

echo ""
echo "██████████████████████████████████████████████████████████████████████████████"
echo "███  🏆 所有随机种子实验完成！  🏆  ███"
echo "██████████████████████████████████████████████████████████████████████████████"
echo ""
echo "📁 实验根目录: ${RUN_ROOT_DIR}"
echo "📁 日志目录: ${BASE_LOG_DIR}"
echo "📄 本次实验日志文件:"
for SEED in "${SEEDS[@]}"; do
    echo "   - ${BASE_LOG_DIR}/seed_${SEED}_${MAIN_TIMESTAMP}.log"
done
echo ""
echo "📂 实验目录结构:"
echo "   ${RUN_ROOT_DIR}/"
for SEED in "${SEEDS[@]}"; do
    echo "   ├── seed_${SEED}/"
    for STRATEGY in "${STRATEGIES[@]}"; do
        echo "   │   └── ${STRATEGY}/"
    done
done
echo "=========================================="

# ============ 🔥 新增：导出所有指标到 CSV ============
echo ""
echo "=========================================="
echo "📊 导出所有指标到 CSV..."
echo "=========================================="

python << 'PYTHON_CSV'
import os
import re
import pandas as pd
from pathlib import Path

log_dir = Path("./training_runs")
strategy_order = ['fusion_ft', 'point_ft', 'spatial_ft', 'partial', 'none']

print("正在解析日志文件...")

# 查找日志文件
log_files = list(log_dir.glob("seed_*.log"))
if not log_files:
    print("⚠️ 没有找到日志文件")
    exit(0)

all_records = []

for log_file in log_files:
    match = re.search(r'seed_(\d+)_', log_file.name)
    if not match:
        continue
    seed = int(match.group(1))
    
    with open(log_file, 'r') as f:
        content = f.read()
    
    print(f"\n处理 seed={seed}: {log_file.name}")
    
    # 查找每个策略的聚合结果
    for strategy in strategy_order:
        # 匹配模式: 策略: xxx ... R²=0.xxx  RMSE=xx.xx
        pattern = rf'策略: {strategy}.*?R²=([0-9.]+)\s+RMSE=([0-9.]+)'
        match_result = re.search(pattern, content, re.DOTALL)
        
        if match_result:
            r2 = float(match_result.group(1))
            rmse = float(match_result.group(2))
            
            all_records.append({
                'seed': seed,
                'strategy': strategy,
                'R2': r2,
                'RMSE': rmse,
            })
            print(f"  ✓ {strategy}: R²={r2}, RMSE={rmse}")
        else:
            print(f"  ⚠ {strategy}: 未找到数据")

# 保存到 CSV
if all_records:
    df = pd.DataFrame(all_records)
    csv_path = log_dir / f"all_metrics_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv"
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    
    print(f"\n✅ 已保存到: {csv_path}")
    print("\n数据内容:")
    print(df.to_string())
    
    # 按策略分组统计
    print("\n" + "="*60)
    print("📊 各策略统计 (所有种子汇总):")
    print("="*60)
    for strategy in strategy_order:
        subset = df[df['strategy'] == strategy]
        if not subset.empty:
            r2_mean = subset['R2'].mean()
            r2_std = subset['R2'].std()
            rmse_mean = subset['RMSE'].mean()
            rmse_std = subset['RMSE'].std()
            print(f"  {strategy:20s}: R²={r2_mean:.4f}±{r2_std:.4f}, RMSE={rmse_mean:.2f}±{rmse_std:.2f}")
else:
    print("⚠️ 没有找到任何有效数据")

PYTHON_CSV

echo ""
echo "✅ CSV 导出完成"
echo "=========================================="

# ============ 🔥 绘制多种子策略 Violin+Box 对比图 ============
echo ""
echo "=========================================="
echo "📊 绘制多种子策略 Violin+Box 对比图..."
echo "=========================================="

python << 'EOF'
import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# 设置中文字体（兼容 Linux 服务器）
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 日志目录
log_dir = Path("./training_runs")

# 策略配置
strategy_order = ['fusion_ft', 'point_ft', 'spatial_ft', 'partial', 'none']
strategy_labels = {
    'fusion_ft': 'Fusion-Layer FT',
    'point_ft': 'Point-Branch FT',
    'spatial_ft': 'Spatial-Branch FT',
    'partial': 'Top-Layer FT',
    'none': 'Full FT'
}

print("正在解析日志文件...")

# 获取当前实验的时间戳
main_timestamp = None
for log_file in log_dir.glob("seed_*_*.log"):
    match = re.search(r'seed_\d+_(\d{8}_\d{6})\.log', log_file.name)
    if match:
        main_timestamp = match.group(1)
        break

if main_timestamp is None:
    print("⚠️ 无法确定实验时间戳，使用最新日志")
    log_files = sorted(log_dir.glob("seed_*.log"), key=os.path.getmtime, reverse=True)
else:
    log_files = log_dir.glob(f"seed_*_{main_timestamp}.log")

# 收集所有种子的结果
all_results = {}

for log_file in log_files:
    match = re.search(r'seed_(\d+)_', log_file.name)
    if not match:
        continue
    
    seed = int(match.group(1))
    
    with open(log_file, 'r') as f:
        content = f.read()
    
    results = {}
    for strategy in strategy_order:
        # 先定位到该策略的区域
        section_pattern = rf'(📊?\s*策略:\s*{strategy}.*?)(?=📊?\s*策略:|$)'
        section_match = re.search(section_pattern, content, re.DOTALL)
        
        if section_match:
            section_content = section_match.group(1)
            best_pattern = r'BEST_FOLD_R2:\s*([0-9.]+).*?BEST_FOLD_RMSE:\s*([0-9.]+).*?BEST_FOLD_MAE:\s*([0-9.]+)'
            best_match = re.search(best_pattern, section_content, re.DOTALL)
            
            if best_match:
                results[strategy] = {
                    'r2': float(best_match.group(1)),
                    'rmse': float(best_match.group(2)),
                    'mae': float(best_match.group(3))
                }
    
    if results:
        all_results[seed] = results
        print(f"  ✓ seed={seed}: {len(results)} 个策略")

# 如果没有找到任何结果，报错退出
if not all_results:
    print("❌ 错误：没有找到任何有效结果！")
    print("   请检查日志文件是否包含正确的 BEST_FOLD_R2 数据")
    print("   日志文件路径: ./training_runs/seed_*.log")
    exit(1)

print(f"\n✅ 共收集到 {len(all_results)} 个种子的结果")

# 统计每个策略的有效数据点数量
for strategy in strategy_order:
    count = sum(1 for seed, res in all_results.items() if strategy in res)
    print(f"  {strategy}: {count} 个有效种子")

# ============ 准备绘图数据 ============
all_data = []
for seed, results in all_results.items():
    for strategy_name, metrics in results.items():
        all_data.append({
            'Strategy': strategy_name,
            'Metric': 'R²',
            'Value': metrics['r2']
        })
        all_data.append({
            'Strategy': strategy_name,
            'Metric': 'RMSE (mm)',
            'Value': metrics['rmse']
        })
        all_data.append({
            'Strategy': strategy_name,
            'Metric': 'MAE (mm)',
            'Value': metrics['mae']
        })

df_plot = pd.DataFrame(all_data)
df_plot['Strategy_Label'] = df_plot['Strategy'].map(strategy_labels)

# ============================================================
# 1:1 Paper Style Violin + Box Plot (最终修正版)
# ============================================================

# 柔和配色
strategy_colors = {
        'Fusion-Layer FT': '#AEECEC',
    'Point-Branch FT': '#B8E6B8',
    'Spatial-Branch FT': '#FFE699',
    'Top-Layer FT': '#F8C88C',
    'Full FT': '#FFB3B3'
}

metrics_order = ['R²', 'RMSE (mm)', 'MAE (mm)']

metric_titles = {
    'R²': 'R² (↑)',
    'RMSE (mm)': 'RMSE (mm) (↓)',
    'MAE (mm)': 'MAE (mm) (↓)'
}

# 获取策略标签列表（按顺序）
strategy_labels_list = [strategy_labels[s] for s in strategy_order]

fig, axes = plt.subplots(3, 1, figsize=(10, 14))

fig.suptitle(
    "Multi-Seed CV: Fine-tuning Strategy Comparison",
    fontsize=22,
    fontweight='bold',
    y=0.985
)

fig.text(
    0.5,
    0.955,
    "(Best fold R² across 10-fold CV, 2 seeds)",
    ha='center',
    fontsize=17,
    style='italic',
    color='dimgray'
)

for ax, metric in zip(axes, metrics_order):
    subset = df_plot[df_plot["Metric"] == metric]
    positions = np.arange(len(strategy_labels_list))
    
    violin_data = []
    for strat in strategy_labels_list:
        vals = subset[subset["Strategy_Label"] == strat]["Value"].values
        violin_data.append(vals)
    
    # ==================================================
    # 先设置坐标轴范围（避免文字飞出）
    # ==================================================
    if metric == 'R²':
        ax.set_ylim(0, 1.0)
    elif metric == 'RMSE (mm)':
        ax.set_ylim(9, 20)
    elif metric == 'MAE (mm)':
        ax.set_ylim(6, 12)
    
    # 获取坐标轴范围用于文字定位
    ymin, ymax = ax.get_ylim()
    yrange = ymax - ymin
    
    # ==================================================
    # Violin (宽度适中，更自然)
    # ==================================================
    vp = ax.violinplot(
        violin_data,
        positions=positions,
        widths=0.42,
        showmeans=False,
        showmedians=False,
        showextrema=False
    )
    
    for body, strat in zip(vp['bodies'], strategy_labels_list):
        color = strategy_colors[strat]
        body.set_facecolor(color)
        body.set_edgecolor(color)
        body.set_linewidth(1.0)
        body.set_alpha(0.45)
    
    # ==================================================
    # Box
    # ==================================================
    bp = ax.boxplot(
        violin_data,
        positions=positions,
        widths=0.12,
        patch_artist=True,
        showfliers=False,
        medianprops=dict(color='red', linewidth=2),
        whiskerprops=dict(color='black', linewidth=1),
        capprops=dict(color='black', linewidth=1)
    )
    
    for box in bp['boxes']:
        box.set_facecolor('white')
        box.set_edgecolor('black')
        box.set_linewidth(1)
    
    # ==================================================
    # 竖直连接线（不抖动）
    # ==================================================
    for x, vals in enumerate(violin_data):
        vals = np.sort(vals)
        n_vals = len(vals)
        
        # 竖直连接线（黑色，不抖动）
        ax.plot(
            [x] * n_vals,
            vals,
            color='black',
            linewidth=0.8,
            alpha=0.7,
            zorder=3
        )
        
        # 散点加轻微抖动
        jitter = np.random.uniform(-0.02, 0.02, n_vals)
        ax.scatter(
            x + jitter,
            vals,
            s=35,
            facecolors='none',
            edgecolors='black',
            linewidths=1,
            zorder=4
        )
        
        # 中位数文字（跟随每个 violin 的顶部）
        median_val = np.median(vals)
        text_y = np.max(vals) + yrange * 0.03
        
        # 防止文字飞出上边界
        if text_y > ymax - yrange * 0.02:
            text_y = ymax - yrange * 0.02
        
        ax.text(
            x,
            text_y,
            f"{median_val:.3f}",
            ha='center',
            va='bottom',
            fontsize=12,
            color='darkred',
            fontweight='medium'
        )
    
    # ==================================================
    # Axis style
    # ==================================================
    ax.set_title(metric_titles[metric], fontsize=18, fontweight='medium', pad=12)
    ax.set_ylabel(metric, fontsize=18, fontweight='medium')
    ax.set_xlabel("")
    ax.set_xticks(positions)
    ax.set_xticklabels(strategy_labels_list, rotation=25, ha='right', fontsize=14)
    ax.tick_params(axis='y', labelsize=13)
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    ax.set_axisbelow(True)
    
    # ==================================================
    # 不添加任何参考线
    # ==================================================

# ==================================================
# Layout - 增加子图间距
# ==================================================
plt.subplots_adjust(
    top=0.91,
    bottom=0.08,
    left=0.09,
    right=0.98,
    hspace=0.45
)

save_path = "./training_runs/multi_seed_violin_box_paper_style.png"
plt.savefig(save_path, dpi=600, bbox_inches='tight')
plt.close()

print(f"\n📊 Figure saved to: {save_path}")
EOF

echo ""
echo "✅ Violin+Box 对比图生成完成"
echo "=========================================="