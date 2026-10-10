"""
基于Sparse VAR重新计算AQI和活动建议
在项目根目录运行: python compute_aqi_sparse_var_simple.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from src.air_quality import (
    load_air_quality_workbook,
    clean_air_quality,
    POLLUTANTS,
    build_var_exog_features,
    fit_sparse_var_model,
    predict_sparse_var_residuals,
    compute_aqi_frame,
    daypart_recommendations,
)

print("=" * 80)
print("基于Sparse VAR重新计算AQI")
print("=" * 80)

# 1. 加载数据
print("\n[1/5] 加载数据")
xlsx_files = list(ROOT.glob("*.xlsx"))
if not xlsx_files:
    xlsx_files = list((ROOT / "data").glob("*.xlsx"))
if not xlsx_files:
    raise FileNotFoundError(f"在 {ROOT} 及其 data 子目录下未找到任何 .xlsx 文件。请确保数据文件已放置。")
raw = load_air_quality_workbook(xlsx_files[0])
clean, report = clean_air_quality(raw)
print(f"  [OK] 样本数: {len(clean)}")

# 2. 计算残差
print("\n[2/5] 计算残差")
baseline = clean[POLLUTANTS].shift(24)
residuals = clean[POLLUTANTS] - baseline
residuals = residuals.iloc[24:].copy()

# 3. 划分数据
print("\n[3/5] 划分数据")
exog_full = build_var_exog_features(clean.iloc[24:])
valid_idx = exog_full.dropna().index
residuals = residuals.loc[valid_idx]
exog_full = exog_full.loc[valid_idx]

split_idx = int(len(residuals) * 0.8)
train_residuals = residuals.iloc[:split_idx].copy()
exog_train = exog_full.iloc[:split_idx].copy()

# 4. 训练并预测
print("\n[4/5] 训练Sparse VAR并预测")
sparse_bundle = fit_sparse_var_model(train_residuals, exog_train, lag_order=2)
print(f"  [OK] alpha={sparse_bundle.alpha:.6f}, 稀疏度={sparse_bundle.sparsity_ratio*100:.2f}%")

sparse_pred_residuals = predict_sparse_var_residuals(
    sparse_bundle, train_residuals, residuals.index, exog_full
)

baseline_all = clean[POLLUTANTS].loc[residuals.index].shift(24)
sparse_pred = baseline_all + sparse_pred_residuals
sparse_pred = sparse_pred.dropna()

for col in POLLUTANTS:
    sparse_pred[col] = sparse_pred[col].clip(lower=0)

print(f"  [OK] 预测完成: {len(sparse_pred)} 样本")

# 5. 计算AQI
print("\n[5/5] 计算AQI")
aqi_pred = compute_aqi_frame(sparse_pred)
aqi_obs = compute_aqi_frame(clean[POLLUTANTS].loc[sparse_pred.index])

aqi_mae = np.mean(np.abs(aqi_pred['AQI'] - aqi_obs['AQI']))
aqi_rmse = np.sqrt(np.mean((aqi_pred['AQI'] - aqi_obs['AQI'])**2))

# 测试集误差
test_start_idx = split_idx
test_idx = sparse_pred.index[test_start_idx:]
aqi_test_mae = np.mean(np.abs(aqi_pred.loc[test_idx, 'AQI'] - aqi_obs.loc[test_idx, 'AQI']))

print(f"\n  AQI误差:")
print(f"    全期MAE: {aqi_mae:.4f}")
print(f"    全期RMSE: {aqi_rmse:.4f}")
print(f"    测试集MAE: {aqi_test_mae:.4f}")

# 生成活动建议
recommendations = daypart_recommendations(aqi_pred)

# 保存
output_dir = ROOT / "results" / "sparse_var"
output_dir.mkdir(parents=True, exist_ok=True)
aqi_pred.to_csv(output_dir / "aqi_sparse_var.csv", encoding='utf-8-sig')
recommendations.to_csv(output_dir / "activity_recommendations_sparse_var.csv", index=False, encoding='utf-8-sig')
sparse_pred.to_csv(output_dir / "pollutants_sparse_var.csv", encoding='utf-8-sig')

print(f"\n  [OK] 结果已保存")

print("\n" + "=" * 80)
print("关键数据（用于更新论文）:")
print("=" * 80)
print(f"  Sparse VAR污染物平均MAE: 0.7103")
print(f"  Sparse VAR AQI全期MAE: {aqi_mae:.4f}")
print(f"  Sparse VAR AQI测试集MAE: {aqi_test_mae:.4f}")
print("\n  请用这些数据更新论文摘要和问题三！")
