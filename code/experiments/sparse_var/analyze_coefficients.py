"""
分析稀疏VAR系数矩阵
"""
from pathlib import Path
import numpy as np
import pandas as pd

project_root = Path(__file__).resolve().parents[3]

# 加载系数矩阵
coef = np.load(project_root / "results" / "sparse_var" / "sparse_var_coefficients.npy")

print("=" * 80)
print("稀疏VAR系数矩阵分析")
print("=" * 80)

print(f"\n系数矩阵形状: {coef.shape}")
print(f"  (6个污染物 × {coef.shape[1]}个特征)")

# 统计系数分布
abs_coef = np.abs(coef)
print(f"\n系数绝对值统计:")
print(f"  最小值: {abs_coef.min():.6f}")
print(f"  最大值: {abs_coef.max():.6f}")
print(f"  平均值: {abs_coef.mean():.6f}")
print(f"  中位数: {np.median(abs_coef):.6f}")

# 不同阈值下的稀疏度
thresholds = [1e-10, 1e-6, 1e-4, 1e-3, 1e-2, 0.05, 0.1]
print(f"\n不同阈值下的稀疏度:")
for thresh in thresholds:
    n_nonzero = np.sum(abs_coef > thresh)
    sparsity = 1 - (n_nonzero / coef.size)
    print(f"  阈值 > {thresh:8.0e}: 非零系数 {n_nonzero:3d}, 稀疏度 {sparsity*100:5.2f}%")

# 分析特征重要性
feature_importance = np.sum(abs_coef, axis=0)
top_10_idx = np.argsort(feature_importance)[-10:][::-1]

print(f"\n前10个最重要的特征（按系数绝对值和排序）:")
for i, idx in enumerate(top_10_idx, 1):
    print(f"  {i:2d}. 特征{idx:3d}: 重要性 {feature_importance[idx]:.4f}")

# 分析每个污染物的系数分布
pollutants = ["PM2.5", "PM10", "O3", "NO2", "SO2", "CO"]
print(f"\n各污染物系数统计:")
for i, pollutant in enumerate(pollutants):
    coef_row = abs_coef[i, :]
    print(f"  {pollutant:8s}: 均值={coef_row.mean():.4f}, "
          f"最大={coef_row.max():.4f}, "
          f">0.01的系数数={np.sum(coef_row > 0.01):3d}")
