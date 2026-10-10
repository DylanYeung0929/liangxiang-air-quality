"""
调试脚本：检查VAR和VECM的实际预测值
"""
from pathlib import Path
import sys
import numpy as np
import pandas as pd

project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

from src.air_quality import (
    load_air_quality_workbook,
    clean_air_quality,
    POLLUTANTS,
    build_var_exog_features,
    fit_var_model,
    fit_vecm_model,
    predict_var_residuals,
    predict_vecm_residuals,
)

# 加载数据
data_path = project_root / "4-2026年校赛第二轮A题数据.xlsx"
raw = load_air_quality_workbook(data_path)
clean, report = clean_air_quality(raw)

# 计算残差
baseline = clean[POLLUTANTS].shift(24)
residuals = clean[POLLUTANTS] - baseline
residuals = residuals.iloc[24:].copy()

# 构造外生变量
exog_full = build_var_exog_features(clean.iloc[24:])
valid_idx = exog_full.dropna().index
residuals = residuals.loc[valid_idx]
exog_full = exog_full.loc[valid_idx]

# 划分数据
split_idx = int(len(residuals) * 0.8)
train_residuals = residuals.iloc[:split_idx].copy()
test_residuals = residuals.iloc[split_idx:].copy()
exog_train = exog_full.iloc[:split_idx].copy()
exog_test = exog_full.iloc[split_idx:].copy()

print("=" * 80)
print("数据检查")
print("=" * 80)
print(f"训练集残差形状: {train_residuals.shape}")
print(f"测试集残差形状: {test_residuals.shape}")
print(f"训练集外生变量形状: {exog_train.shape}")
print(f"测试集外生变量形状: {exog_test.shape}")
print(f"\n训练集残差前5行:")
print(train_residuals.head())
print(f"\n测试集残差前5行:")
print(test_residuals.head())

# 训练VAR
print("\n" + "=" * 80)
print("训练VAR(2)")
print("=" * 80)
var_bundle = fit_var_model(train_residuals, exog_train, lag_order=2)
print(f"VAR滞后阶数: {var_bundle.lag_order}")
print(f"VAR外生变量数: {len(var_bundle.exog_columns)}")

# VAR预测
var_pred_residuals = predict_var_residuals(
    var_bundle, train_residuals, test_residuals.index[:10], exog_test
)
print(f"\nVAR预测残差前5行:")
print(var_pred_residuals.head())
print(f"\nVAR预测残差统计:")
print(var_pred_residuals.describe())

# 训练VECM
print("\n" + "=" * 80)
print("训练VECM")
print("=" * 80)
vecm_bundle = fit_vecm_model(train_residuals, exog_train, k_ar_diff=1, coint_rank=6)
print(f"VECM k_ar_diff: {vecm_bundle.k_ar_diff}")
print(f"VECM coint_rank: {vecm_bundle.coint_rank}")
print(f"VECM外生变量数: {len(vecm_bundle.exog_columns)}")

# VECM预测
vecm_pred_residuals = predict_vecm_residuals(
    vecm_bundle, train_residuals, test_residuals.index[:10], exog_test
)
print(f"\nVECM预测残差前5行:")
print(vecm_pred_residuals.head())
print(f"\nVECM预测残差统计:")
print(vecm_pred_residuals.describe())

# 对比
print("\n" + "=" * 80)
print("VAR vs VECM 预测差异")
print("=" * 80)
diff = var_pred_residuals - vecm_pred_residuals
print(f"差异统计:")
print(diff.describe())
print(f"\n差异绝对值最大值:")
print(diff.abs().max())
print(f"\n差异是否全为0: {(diff.abs() < 1e-10).all().all()}")

# 检查模型系数
print("\n" + "=" * 80)
print("模型系数检查")
print("=" * 80)
print(f"\nVAR模型类型: {type(var_bundle.results)}")
print(f"VECM模型类型: {type(vecm_bundle.results)}")

# 检查VAR系数矩阵
print(f"\nVAR系数矩阵形状:")
for i in range(var_bundle.lag_order):
    coef = var_bundle.results.params[i*6:(i+1)*6, :]
    print(f"  Lag {i+1}: {coef.shape}")
    print(f"  范数: {np.linalg.norm(coef):.4f}")

# 检查VECM系数
print(f"\nVECM alpha (调整系数) 形状: {vecm_bundle.results.alpha.shape}")
print(f"VECM alpha 范数: {np.linalg.norm(vecm_bundle.results.alpha):.4f}")
print(f"\nVECM beta (协整向量) 形状: {vecm_bundle.results.beta.shape}")
print(f"VECM beta 范数: {np.linalg.norm(vecm_bundle.results.beta):.4f}")

if vecm_bundle.results.gamma is not None:
    print(f"\nVECM gamma (短期动态) 形状: {vecm_bundle.results.gamma.shape}")
    print(f"VECM gamma 范数: {np.linalg.norm(vecm_bundle.results.gamma):.4f}")
