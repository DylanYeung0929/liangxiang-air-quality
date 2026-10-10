"""
消融实验：验证VAR协同建模的必要性

对比两种策略：
1. 独立模型：6个独立的Ridge回归，每个污染物单独预测
2. 联合VAR模型：周期残差混合VAR，多变量联合建模

目标：验证"协同"是否带来实质性增益
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.air_quality import (
    POLLUTANTS,
    load_air_quality_workbook,
    clean_air_quality,
    build_supervised_dataset,
    chronological_split,
    regression_metrics,
    seasonal_lag_frame,
    build_var_exog_features,
    fit_var_model,
    predict_var_residuals,
)


def fit_independent_ridge_models(x_train, y_train):
    """为每个污染物训练独立的Ridge模型"""
    models = {}
    for pollutant in POLLUTANTS:
        model = make_pipeline(
            StandardScaler(),
            RidgeCV(alphas=np.logspace(-3, 4, 20)),
        ).fit(x_train, y_train[pollutant])
        models[pollutant] = model
    return models


def predict_independent(models, x_test):
    """使用独立模型预测"""
    predictions = {}
    for pollutant in POLLUTANTS:
        pred = models[pollutant].predict(x_test)
        predictions[pollutant] = np.clip(pred, 0, None)
    return pd.DataFrame(predictions, index=x_test.index)


def main():
    print("=" * 80)
    print("消融实验：独立模型 vs 联合VAR模型")
    print("=" * 80)

    # 加载数据
    xlsx_files = list(ROOT.glob("*.xlsx"))
    if not xlsx_files:
        xlsx_files = list((ROOT / "data").glob("*.xlsx"))
    if not xlsx_files:
        raise FileNotFoundError(f"在 {ROOT} 及其 data 子目录下未找到任何 .xlsx 文件。请确保数据文件已放置。")
    raw_path = xlsx_files[0]

    raw = load_air_quality_workbook(raw_path)
    clean, _ = clean_air_quality(raw)

    # 构建特征
    x_full, y_full = build_supervised_dataset(
        clean,
        include_pollution_lags=True,
    )
    x_train, x_test, y_train, y_test = chronological_split(x_full, y_full, test_fraction=0.2)

    print(f"\n训练集大小: {len(x_train)}, 测试集大小: {len(x_test)}")

    # ========== 策略1：独立Ridge模型 ==========
    print("\n[1/3] 训练独立Ridge模型（每个污染物单独预测）...")
    independent_models = fit_independent_ridge_models(x_train, y_train)
    y_pred_independent = predict_independent(independent_models, x_test)
    metrics_independent = regression_metrics(y_test, y_pred_independent, "Independent Ridge")

    print("\n独立模型测试集性能:")
    print(metrics_independent.to_string(index=False))

    # ========== 策略2：联合VAR模型（周期残差混合） ==========
    print("\n[2/3] 训练联合VAR模型（周期残差混合）...")

    # 计算周期基线
    baseline_train = seasonal_lag_frame(y_train, y_train.index, lag_hours=24)
    baseline_test = seasonal_lag_frame(y_full, y_test.index, lag_hours=24)

    # 计算残差
    residual_train = y_train - baseline_train
    residual_test = y_test - baseline_test

    # 构建VAR外生特征
    exog_full = build_var_exog_features(clean)
    exog_train = exog_full.loc[residual_train.index]
    exog_test = exog_full.loc[residual_test.index]

    # 训练VAR(2)
    var_bundle = fit_var_model(residual_train, exog_train, lag_order=2)

    # 预测残差
    residual_pred = predict_var_residuals(
        var_bundle,
        residual_train,
        y_test.index,
        exog_future=exog_full,
    )

    # 合成最终预测
    y_pred_var = baseline_test + residual_pred
    for col in POLLUTANTS:
        y_pred_var[col] = y_pred_var[col].clip(lower=0)

    metrics_var = regression_metrics(y_test, y_pred_var, "Hybrid VAR")

    print("\n联合VAR模型测试集性能:")
    print(metrics_var.to_string(index=False))

    # ========== 对比分析 ==========
    print("\n[3/3] 生成对比表格...")

    # 合并指标
    comparison = pd.concat([metrics_independent, metrics_var], ignore_index=True)

    # 计算平均性能
    avg_metrics = comparison.groupby("Model")[["MAE", "RMSE", "R2"]].mean().reset_index()
    avg_metrics.insert(1, "Pollutant", "Average")

    # 计算改进幅度
    improvement = []
    for pollutant in POLLUTANTS:
        ind_row = metrics_independent[metrics_independent["Pollutant"] == pollutant].iloc[0]
        var_row = metrics_var[metrics_var["Pollutant"] == pollutant].iloc[0]

        mae_improve = (ind_row["MAE"] - var_row["MAE"]) / ind_row["MAE"] * 100
        rmse_improve = (ind_row["RMSE"] - var_row["RMSE"]) / ind_row["RMSE"] * 100
        r2_improve = (var_row["R2"] - ind_row["R2"]) * 100  # R2是越大越好

        improvement.append({
            "Pollutant": pollutant,
            "MAE_Improvement_%": mae_improve,
            "RMSE_Improvement_%": rmse_improve,
            "R2_Improvement_pp": r2_improve,  # pp = percentage points
        })

    improvement_df = pd.DataFrame(improvement)

    # 保存结果
    results_dir = ROOT / "results"
    results_dir.mkdir(exist_ok=True)

    comparison.to_csv(results_dir / "ablation_comparison.csv", index=False)
    avg_metrics.to_csv(results_dir / "ablation_average.csv", index=False)
    improvement_df.to_csv(results_dir / "ablation_improvement.csv", index=False)

    print("\n" + "=" * 80)
    print("消融实验结果汇总")
    print("=" * 80)
    print("\n平均性能对比:")
    print(avg_metrics.to_string(index=False))

    print("\n\nVAR相对于独立模型的改进幅度:")
    print(improvement_df.to_string(index=False))

    # 解释结果
    avg_mae_improve = improvement_df["MAE_Improvement_%"].mean()
    avg_r2_improve = improvement_df["R2_Improvement_pp"].mean()

    print("\n" + "=" * 80)
    print("结论:")
    print("=" * 80)
    if avg_mae_improve > 5:
        print(f"[+] VAR联合建模相比独立模型，MAE平均改进 {avg_mae_improve:.2f}%")
        print(f"[+] R2平均提升 {avg_r2_improve:.4f} 个百分点")
        print("[+] 协同建模带来了实质性增益，证明多变量联合建模的必要性")
    elif avg_mae_improve > 0:
        print(f"[~] VAR联合建模相比独立模型，MAE平均改进 {avg_mae_improve:.2f}%")
        print(f"[~] R2平均提升 {avg_r2_improve:.4f} 个百分点")
        print("[~] 协同建模带来了轻微增益，但改进幅度有限")
    else:
        print(f"[-] VAR联合建模相比独立模型，MAE平均下降 {abs(avg_mae_improve):.2f}%")
        print("[-] 在当前测试集上，协同建模未带来显著增益")
        print("    可能原因：测试期过于平稳，周期基线已解释绝大部分方差")

    print("\n结果已保存至 results/ 目录")
    print("  - ablation_comparison.csv: 详细对比")
    print("  - ablation_average.csv: 平均性能")
    print("  - ablation_improvement.csv: 改进幅度")


if __name__ == "__main__":
    main()
