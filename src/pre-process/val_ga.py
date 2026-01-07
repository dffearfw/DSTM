import pandas as pd
import numpy as np
from scipy import stats

# 读取数据
df = pd.read_excel('E:/pycharmworkspace/DSTM/output/pr_ex.xlsx')  # 修改为你的文件路径

# 确保有需要的列
if 'zone_id' not in df.columns:
    print("错误: 数据中没有zone_id列")
    exit()

# 按zone_id分组计算
results = []

for zone_id, group in df.groupby('zone_id'):
    # 提取数据
    swe = group['swe'].values
    precip = group['precipitation_mm_day'].values

    # 移除NaN值
    mask = ~np.isnan(swe) & ~np.isnan(precip)
    swe_clean = swe[mask]
    precip_clean = precip[mask]

    if len(swe_clean) < 2:  # 至少需要2个点才能计算
        continue

    # 计算相关系数
    pearson_r, pearson_p = stats.pearsonr(swe_clean, precip_clean)
    spearman_r, spearman_p = stats.spearmanr(swe_clean, precip_clean)

    # 线性回归
    slope, intercept, r_value, p_value, std_err = stats.linregress(precip_clean, swe_clean)

    # 收集结果
    results.append({
        'zone_id': zone_id,
        '数据点数': len(swe_clean),
        'SWE均值': swe_clean.mean(),
        '降水均值': precip_clean.mean(),
        'Pearson_r': pearson_r,
        'Pearson_p': pearson_p,
        'Spearman_r': spearman_r,
        'Spearman_p': spearman_p,
        '回归_截距': intercept,
        '回归_斜率': slope,
        '回归_R2': r_value ** 2,
        '回归_p值': p_value,
        '回归_标准误差': std_err,
        '回归方程': f'swe = {slope:.3f}*precip + {intercept:.3f}'
    })

# 转换为DataFrame
result_df = pd.DataFrame(results)

# 保存结果
result_df.to_excel('zone_分组分析结果.xlsx', index=False)
print(f"分析完成! 共分析 {len(result_df)} 个zone")
print("\n结果预览:")
print(result_df[['zone_id', '数据点数', 'Pearson_r', '回归_R2', '回归方程']].head())

# 输出汇总统计
print(f"\n=== 汇总统计 ===")
print(f"平均Pearson相关系数: {result_df['Pearson_r'].mean():.3f}")
print(f"平均R²: {result_df['回归_R2'].mean():.3f}")
print(f"显著相关比例 (p<0.05): {(result_df['Pearson_p'] < 0.05).mean():.1%}")

# 保存汇总统计
summary = pd.DataFrame({
    '统计项': ['平均数据点数', '平均Pearson_r', '平均R²', '显著相关比例(p<0.05)', '总zone数'],
    '值': [
        result_df['数据点数'].mean(),
        result_df['Pearson_r'].mean(),
        result_df['回归_R2'].mean(),
        (result_df['Pearson_p'] < 0.05).mean(),
        len(result_df)
    ]
})
summary.to_excel('zone_汇总统计.xlsx', index=False)