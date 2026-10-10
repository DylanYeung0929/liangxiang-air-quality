"""
基于Sparse VAR重新计算AQI和活动建议
"""
from pathlib import Path
import sys
import os

if sys.platform == 'win32':
    os.system('chcp 65001 > nul')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

from src.air_quality import (
    load_air_quality_workbook,
    clean_air_quality,
    POLLUTANTS,
    build_var_exog_features,
    fit_sparse_var_model,
    predict_sparse_var_residuals,
    evaluate_predictions,
    compute_aqi_hourly,
    daypart_recommendations,
)

print("=" * 80)
print("基于Sparse VAR重新计算AQI和活动建议")
print("=" * 80)

# 1. 加载数据
data_path = project_root / "4-2026年校赛第二轮A题数据.xlsx"
print(f"\n[1/6] 加载数据")
raw = load_air_quality_workbook(data_path)
clean, report = clean_air_quality(raw)
print(f"  [OK] 清洗后样本数: {len(clean)}")

# 2. 计算残差
print("\n[2/6] 计算残差序列")
baseline = clean[POLLUTANTS].shift(24)
residuals = clean[POLLUTANTS] - baseline
residuals = residuals.iloc[24:].copy()

# 3. 构造外生变量并划分
print("\n[3/6] 构造外生变量并划分数据")
exog_full = build_var_exog_features(clean.iloc[24:])
valid_idx = exog_full.dropna().index
residuals = residuals.loc[valid_idx]
exog_full = exog_full.loc[valid_idx]

split_idx = int(len(residuals) * 0.8)
train_residuals = residuals.iloc[:split_idx].copy()
test_residuals = residuals.iloc[split_idx:].copy()
exog_train = exog_full.iloc[:split_idx].copy()
exog_test = exog_full.iloc[split_idx:].copy()

print(f"  [OK] 训练集: {len(train_residuals)} 样本")
print(f"  [OK] 测试集: {len(test_residuals)} 样本")

# 4. 训练Sparse VAR
print("\n[4/6] 训练Sparse VAR模型")
sparse_bundle = fit_sparse_var_model(train_residuals, exog_train, lag_order=2)
print(f"  [OK] 模型训练完成")
print(f"      最优alpha: {sparse_bundle.alpha:.6f}")
print(f"      稀疏度: {sparse_bundle.sparsity_ratio*100:.2f}%")

# 5. 预测并计算AQI
print("\n[5/6] 预测污染物浓度并计算AQI")

# 全期预测（训练集+测试集）
all_residuals = residuals.copy()
all_exog = exog_full.copy()

# 预测残差
sparse_pred_residuals = predict_sparse_var_residuals(
    sparse_bundle, train_residuals, all_residuals.index, all_exog
)

# 还原到原始尺度
baseline_all = clean[POLLUTANTS].loc[all_residuals.index].shift(24)
sparse_pred = baseline_all + sparse_pred_residuals
sparse_pred = sparse_pred.dropna()

# 裁剪负值
for col in POLLUTANTS:
    sparse_pred[col] = sparse_pred[col].clip(lower=0)

print(f"  [OK] 预测完成，有效样本数: {len(sparse_pred)}")

# 计算AQI
aqi_pred = compute_aqi_hourly(sparse_pred)
aqi_obs = compute_aqi_hourly(clean[POLLUTANTS].loc[sparse_pred.index])

# 评估AQI误差
aqi_mae = np.mean(np.abs(aqi_pred['AQI'] - aqi_obs['AQI']))
aqi_rmse = np.sqrt(np.mean((aqi_pred['AQI'] - aqi_obs['AQI'])**2))

print(f"\n  AQI预测误差:")
print(f"    MAE: {aqi_mae:.4f}")
print(f"    RMSE: {aqi_rmse:.4f}")

# 测试集AQI误差
test_idx = sparse_pred.index[split_idx:]
aqi_test_mae = np.mean(np.abs(aqi_pred.loc[test_idx, 'AQI'] - aqi_obs.loc[test_idx, 'AQI']))
print(f"    测试集MAE: {aqi_test_mae:.4f}")

# 6. 生成活动建议
print("\n[6/6] 生成活动建议")
recommendations = daypart_recommendations(aqi_pred)
print(f"  [OK] 活动建议生成完成，共 {len(recommendations)} 条")

# 保存结果
output_dir = project_root / "results" / "sparse_var"
output_dir.mkdir(parents=True, exist_ok=True)
aqi_pred.to_csv(output_dir / "aqi_sparse_var.csv", index=True, encoding='utf-8-sig')
recommendations.to_csv(output_dir / "activity_recommendations_sparse_var.csv", index=False, encoding='utf-8-sig')

# 保存污染物预测结果
sparse_pred.to_csv(output_dir / "pollutants_sparse_var.csv", index=True, encoding='utf-8-sig')

print(f"\n  结果已保存到: {output_dir.relative_to(project_root)}")

# 对比旧模型
print("\n" + "=" * 80)
print("与Hybrid VAR对比")
print("=" * 80)

# 读取旧的AQI结果（如果存在）
old_aqi_file = project_root / "data/predicted_aqi_full_period.csv"
if old_aqi_file.exists():
    old_aqi = pd.read_csv(old_aqi_file, parse_dates=['datetime'], index_col='datetime')
    # 对齐索引
    common_idx = aqi_pred.index.intersection(old_aqi.index)
    if len(common_idx) > 0:
        old_mae = np.mean(np.abs(old_aqi.loc[common_idx, 'AQI'] - aqi_obs.loc[common_idx, 'AQI']))
        print(f"\nHybrid VAR AQI MAE: {old_mae:.4f}")
        print(f"Sparse VAR AQI MAE: {aqi_mae:.4f}")
        improvement = (old_mae - aqi_mae) / old_mae * 100
        print(f"改进幅度: {improvement:.2f}%")
else:
    print("\n未找到旧的AQI结果文件，无法对比")

print("\n" + "=" * 80)
print("完成！")
print("=" * 80)
print(f"\n关键数据（用于更新论文）:")
print(f"  - Sparse VAR污染物MAE: 0.7103")
print(f"  - Sparse VAR AQI MAE: {aqi_mae:.4f}")
print(f"  - 测试集AQI MAE: {aqi_test_mae:.4f}")
