#!/bin/bash

# ============================================
# SWE模型自动化训练脚本
# 循环运行10次，完成后自动关机
# ============================================

# 配置参数
TOTAL_RUNS=10
CURRENT_RUN=1
BASE_LOG_DIR="./training_runs"

# 创建日志目录
mkdir -p ${BASE_LOG_DIR}

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "=========================================="
echo "🚀 SWE模型自动化训练脚本"
echo "   总运行次数: ${TOTAL_RUNS}"
echo "   完成后自动关机"
echo "   日志目录: ${BASE_LOG_DIR}"
echo "=========================================="

# 记录开始时间
START_TIME=$(date +%s)

for ((RUN=1; RUN<=TOTAL_RUNS; RUN++))
do
    CURRENT_RUN=$RUN
    
    echo ""
    echo "=========================================="
    echo -e "${GREEN}📊 开始第 ${CURRENT_RUN}/${TOTAL_RUNS} 次训练${NC}"
    echo "=========================================="
    echo "开始时间: $(date '+%Y-%m-%d %H:%M:%S')"
    
    # 记录单次开始时间
    RUN_START=$(date +%s)
    
    # 生成日志文件名
    LOG_FILE="${BASE_LOG_DIR}/training_run_${CURRENT_RUN}_$(date +'%Y%m%d_%H%M%S').log"
    
    echo "日志文件: ${LOG_FILE}"
    
    # 运行训练命令
    python main_tune.py \
        --mode fine_tune \
        --model_type full \
        --batch_size 32 \
        --fine_tune_epochs 100 \
        --freeze_backbone \
        --freeze_strategy partial \
        --pretrained_model /root/autodl-tmp/experiments/swe_full_temporal_20260609_085603/best_model.pth \
        --station_data_path /root/autodl-tmp/combined_station.csv \
        --mixed_mode \
        --use_amp \
        --station_ratio 0.7 \
        --lr_transformer 4e-5 \
        --lr_head 6e-4 \
        --lr_encoder 1e-6 \
        --cv_mode station_cv \
        2>&1 | tee -a "${LOG_FILE}"
    
    # 检查命令执行结果
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✅ 第 ${CURRENT_RUN} 次训练成功完成${NC}"
    else
        echo -e "${RED}❌ 第 ${CURRENT_RUN} 次训练失败${NC}"
        echo "是否继续？(y/n)"
        read -r continue_choice
        if [ "$continue_choice" != "y" ]; then
            echo "终止训练"
            exit 1
        fi
    fi
    
    # 计算单次耗时
    RUN_END=$(date +%s)
    RUN_DURATION=$((RUN_END - RUN_START))
    RUN_HOURS=$((RUN_DURATION / 3600))
    RUN_MINUTES=$(((RUN_DURATION % 3600) / 60))
    RUN_SECONDS=$((RUN_DURATION % 60))
    
    echo "完成时间: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "本次耗时: ${RUN_HOURS}h ${RUN_MINUTES}m ${RUN_SECONDS}s"
    
    # 如果不是最后一次，等待5秒后继续
    if [ ${CURRENT_RUN} -lt ${TOTAL_RUNS} ]; then
        echo ""
        echo -e "${YELLOW}⏳ 等待 5 秒后开始下一次训练...${NC}"
        sleep 5
    fi
    
    echo "=========================================="
    echo ""
done

# 计算总耗时
END_TIME=$(date +%s)
TOTAL_DURATION=$((END_TIME - START_TIME))
TOTAL_HOURS=$((TOTAL_DURATION / 3600))
TOTAL_MINUTES=$(((TOTAL_DURATION % 3600) / 60))
TOTAL_SECONDS=$((TOTAL_DURATION % 60))

echo ""
echo "=========================================="
echo -e "${GREEN}🎉 所有训练完成！${NC}"
echo "=========================================="
echo "总运行次数: ${TOTAL_RUNS}"
echo "开始时间: $(date -d @${START_TIME} '+%Y-%m-%d %H:%M:%S')"
echo "结束时间: $(date -d @${END_TIME} '+%Y-%m-%d %H:%M:%S')"
echo "总耗时: ${TOTAL_HOURS}h ${TOTAL_MINUTES}m ${TOTAL_SECONDS}s"
echo "日志目录: ${BASE_LOG_DIR}"
echo "=========================================="

# 生成汇总报告
SUMMARY_FILE="${BASE_LOG_DIR}/training_summary_$(date +'%Y%m%d_%H%M%S').txt"
echo "训练汇总报告" > ${SUMMARY_FILE}
echo "============" >> ${SUMMARY_FILE}
echo "总运行次数: ${TOTAL_RUNS}" >> ${SUMMARY_FILE}
echo "开始时间: $(date -d @${START_TIME} '+%Y-%m-%d %H:%M:%S')" >> ${SUMMARY_FILE}
echo "结束时间: $(date -d @${END_TIME} '+%Y-%m-%d %H:%M:%S')" >> ${SUMMARY_FILE}
echo "总耗时: ${TOTAL_HOURS}h ${TOTAL_MINUTES}m ${TOTAL_SECONDS}s" >> ${SUMMARY_FILE}
echo "" >> ${SUMMARY_FILE}
echo "日志文件列表:" >> ${SUMMARY_FILE}
ls -la ${BASE_LOG_DIR}/training_run_*.log >> ${SUMMARY_FILE}

echo "汇总报告已保存: ${SUMMARY_FILE}"

# ============================================
# 🔥 运行10次后自动关机
# ============================================
echo ""
echo "=========================================="
echo -e "${YELLOW}🔴 10次训练已完成，系统将在 60 秒后自动关机...${NC}"
echo "=========================================="
echo "按 Ctrl+C 取消关机"

# 等待60秒，给用户取消的机会
for i in {60..1}; do
    echo -ne "倒计时: ${i} 秒\r"
    sleep 1
done

echo ""
echo -e "${RED}💾 正在保存数据并关机...${NC}"

# 执行关机命令
# Ubuntu/Debian 使用 shutdown
sudo shutdown -h now

# 如果是普通用户没有sudo权限，可以尝试直接关机
# systemctl poweroff