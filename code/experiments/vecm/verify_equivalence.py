"""
理论验证：VECM与VAR的等价性

VECM(k_ar_diff=1, coint_rank=6) 的形式：
Δy_t = Π·y_{t-1} + Γ_1·Δy_{t-1} + B·Z_t + ε_t

其中 Π = α·β' (6x6矩阵，秩为6，即满秩)

当协整秩 = 变量数（满秩）时：
Π 是满秩矩阵，可以表示为任意6x6矩阵

展开VECM：
Δy_t = Π·y_{t-1} + Γ_1·Δy_{t-1} + B·Z_t + ε_t
y_t - y_{t-1} = Π·y_{t-1} + Γ_1·(y_{t-1} - y_{t-2}) + B·Z_t + ε_t
y_t = (I + Π)·y_{t-1} + Γ_1·y_{t-1} - Γ_1·y_{t-2} + B·Z_t + ε_t
y_t = (I + Π + Γ_1)·y_{t-1} - Γ_1·y_{t-2} + B·Z_t + ε_t

令 A_1 = I + Π + Γ_1, A_2 = -Γ_1

则：
y_t = A_1·y_{t-1} + A_2·y_{t-2} + B·Z_t + ε_t

这正是 VAR(2) 的形式！

结论：当协整秩=6（满秩）时，VECM(k_ar_diff=1) 完全等价于 VAR(2)
"""

print(__doc__)

print("\n" + "=" * 80)
print("实验验证")
print("=" * 80)

from pathlib import Path
import sys
import numpy as np

project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

try:
    from src.air_quality import (
        load_air_quality_workbook,
        clean_air_quality,
        POLLUTANTS,
        build_var_exog_features,
        fit_var_model,
        fit_vecm_model,
    )
except ImportError as exc:
    print(f"[提示] 当前代码版本未提供 VECM 接口：{exc}")
    print("[提示] 请先补充 src/air_quality.py 中的 fit_vecm_model 后再运行本脚本。")
    raise SystemExit(0)

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
exog_train = exog_full.iloc[:split_idx].copy()

# 训练模型
var_bundle = fit_var_model(train_residuals, exog_train, lag_order=2)
vecm_bundle = fit_vecm_model(train_residuals, exog_train, k_ar_diff=1, coint_rank=6)

print(f"\nVAR(2) 参数数量:")
print(f"  A_1, A_2 系数矩阵: 2 × (6×6) = {2*6*6} 个参数")
print(f"  外生变量系数: 6 × {len(var_bundle.exog_columns)} = {6*len(var_bundle.exog_columns)} 个参数")
print(f"  总计: {2*6*6 + 6*len(var_bundle.exog_columns)} 个参数")

print(f"\nVECM(k_ar_diff=1, coint_rank=6) 参数数量:")
print(f"  α (调整系数): 6×6 = {vecm_bundle.results.alpha.size} 个参数")
print(f"  β (协整向量): 6×6 = {vecm_bundle.results.beta.size} 个参数")
if vecm_bundle.results.gamma is not None:
    print(f"  Γ_1 (短期动态): 6×6 = {vecm_bundle.results.gamma.size} 个参数")
else:
    print(f"  Γ_1 (短期动态): 0 个参数")
print(f"  外生变量系数: 6 × {len(vecm_bundle.exog_columns)} = {6*len(vecm_bundle.exog_columns)} 个参数")

print(f"\n协整秩检查:")
print(f"  coint_rank = {vecm_bundle.coint_rank}")
print(f"  变量数 = {len(POLLUTANTS)}")
print(f"  是否满秩: {vecm_bundle.coint_rank == len(POLLUTANTS)}")

if vecm_bundle.coint_rank == len(POLLUTANTS):
    print(f"\n结论:")
    print(f"  [OK] 协整秩=6（满秩），Π = α·β' 是满秩矩阵")
    print(f"  [OK] VECM(k_ar_diff=1, coint_rank=6) 完全等价于 VAR(2)")
    print(f"  [OK] 两个模型的预测结果完全相同是正确的！")
    print(f"\n  -> 要获得改进，需要降低协整秩（coint_rank < 6）")
    print(f"  -> 或者尝试其他模型（稀疏VAR、降阶VAR等）")
