"""
生成鲁棒性分析汇总表
"""
import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
results_dir = ROOT / "results"
robustness_file = results_dir / "robustness_analysis.csv"
if not robustness_file.exists():
    print(f"未找到输入文件: {robustness_file}")
    print("请先运行: python code/robustness_test.py")
    raise SystemExit(0)

robustness = pd.read_csv(robustness_file)

# 过滤掉baseline_MAE为0或异常值的行
robustness = robustness[robustness["baseline_MAE"] > 0.01]

# 按时段类型汇总
summary = robustness.groupby("period_name").agg({
    "hours": "sum",
    "baseline_MAE": "mean",
    "VAR_MAE": "mean",
    "baseline_R2": "mean",
    "VAR_R2": "mean",
    "MAE_improvement_%": "mean",
    "RMSE_improvement_%": "mean",
}).reset_index()

summary.columns = [
    "Period_Type",
    "Total_Hours",
    "Baseline_MAE",
    "VAR_MAE",
    "Baseline_R2",
    "VAR_R2",
    "MAE_Improvement_%",
    "RMSE_Improvement_%",
]

# 计算实际改进（避免inf）
summary["Actual_MAE_Improvement_%"] = (
    (summary["Baseline_MAE"] - summary["VAR_MAE"]) / summary["Baseline_MAE"] * 100
)

# 保存汇总
summary.to_csv(results_dir / "robustness_summary.csv", index=False)

print("鲁棒性分析汇总表:")
print(summary[["Period_Type", "Total_Hours", "Baseline_MAE", "VAR_MAE",
               "Actual_MAE_Improvement_%"]].to_string(index=False))

# 计算整体结论
print("\n" + "=" * 80)
print("结论:")
print("=" * 80)

for _, row in summary.iterrows():
    period_type = row["Period_Type"]
    mae_improve = row["Actual_MAE_Improvement_%"]
    baseline_mae = row["Baseline_MAE"]
    var_mae = row["VAR_MAE"]

    if mae_improve > 5:
        print(f"[+] {period_type}: VAR相比基线MAE改进 {mae_improve:.2f}%")
        print(f"    (Baseline: {baseline_mae:.4f} -> VAR: {var_mae:.4f})")
    elif mae_improve > 0:
        print(f"[~] {period_type}: VAR相比基线MAE轻微改进 {mae_improve:.2f}%")
        print(f"    (Baseline: {baseline_mae:.4f} -> VAR: {var_mae:.4f})")
    else:
        print(f"[-] {period_type}: VAR相比基线MAE下降 {abs(mae_improve):.2f}%")
        print(f"    (Baseline: {baseline_mae:.4f} -> VAR: {var_mae:.4f})")
        print(f"    说明：在{period_type}时段，VAR的气象外生项未能有效修正预测")

print("\n汇总已保存至 results/robustness_summary.csv")
