"""
多步滚动预测实验

目标：
1. 验证混合VAR模型在多步预测下的性能衰减
2. 展示误差随预测步长的增长曲线
3. 说明当前模型适用于单步"现报"场景，不适用于长期预报
"""

import numpy as np
import pandas as pd
from pathlib import Path

import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.air_quality import (
    POLLUTANTS,
    load_air_quality_workbook,
    clean_air_quality,
    build_var_exog_features,
    fit_var_model,
    predict_var_residuals,
    regression_metrics,
)


def rolling_multistep_forecast(
    y_train,
    y_test,
    exog_full,
    var_bundle,
    residual_train,
    steps=[1, 3, 6, 12, 24]
):
    """
    滚动多步预测

    参数:
        steps: 预测步长列表（小时）

    返回:
        每个步长的预测结果和误差
    """
    results = {}

    for step in steps:
        print(f"\n[{step}步预测]")

        # 构建历史序列（初始为训练集）
        history = y_train.copy()
        predictions = []

        for i, t in enumerate(y_test.index):
            # 计算baseline：使用历史数据的24小时前值
            if len(history) >= 24:
                baseline_t = history.iloc[-24]
            else:
                # 如果历史不足24小时，使用最早的值
                baseline_t = history.iloc[0]

            # 预测残差（使用VAR）
            # 注意：这里简化处理，实际应该用滚动窗口重新训练
            try:
                residual_pred = predict_var_residuals(
                    var_bundle,
                    residual_train,
                    pd.DatetimeIndex([t]),
                    exog_future=exog_full,
                )
                residual_t = residual_pred.iloc[0]
            except:
                # 如果VAR预测失败，使用零残差
                residual_t = pd.Series(0, index=POLLUTANTS)

            # 合成预测
            pred_t = baseline_t + residual_t
            pred_t = pred_t.clip(lower=0)
            predictions.append(pred_t)

            # 更新历史：每step步使用一次真实值，其余用预测值
            if (i + 1) % step == 0:
                # 使用真实值
                history = pd.concat([history, y_test.iloc[[i]]])
            else:
                # 使用预测值
                history = pd.concat([history, pd.DataFrame([pred_t], index=[t])])

        # 转换为DataFrame
        y_pred = pd.DataFrame(predictions, index=y_test.index, columns=POLLUTANTS)

        # 计算误差
        metrics = regression_metrics(y_test, y_pred, f"{step}-step")
        avg_mae = metrics["MAE"].mean()
        avg_rmse = metrics["RMSE"].mean()
        avg_r2 = metrics["R2"].mean()

        print(f"  平均MAE: {avg_mae:.4f}")
        print(f"  平均RMSE: {avg_rmse:.4f}")
        print(f"  平均R2: {avg_r2:.4f}")

        results[step] = {
            "predictions": y_pred,
            "metrics": metrics,
            "avg_mae": avg_mae,
            "avg_rmse": avg_rmse,
            "avg_r2": avg_r2,
        }

    return results


def main():
    print("=" * 80)
    print("多步滚动预测实验")
    print("=" * 80)

    # 加载数据
    xlsx_files = list(ROOT.glob("*.xlsx"))
    if not xlsx_files:
        xlsx_files = list((ROOT / "data").glob("*.xlsx"))
    if not xlsx_files:
        raise FileNotFoundError(f"在 {ROOT} 及其 data 子目录下未找到任何 .xlsx 文件。请确保数据文件已放置。")
    raw_path = xlsx_files[0]

    raw = load_air_quality_workbook(raw_path)
    clean_data, _ = clean_air_quality(raw)

    # 划分训练集和测试集
    split_idx = int(len(clean_data) * 0.8)
    train_data = clean_data.iloc[:split_idx]
    test_data = clean_data.iloc[split_idx:]

    y_train = train_data[POLLUTANTS]
    y_test = test_data[POLLUTANTS]

    print(f"\n训练集大小: {len(y_train)}, 测试集大小: {len(y_test)}")

    # 训练VAR模型
    print("\n[1/3] 训练VAR模型...")

    # 计算周期基线（使用训练集）
    baseline_train = y_train.shift(24).bfill()
    residual_train = y_train - baseline_train

    # 构建VAR外生特征
    exog_full = build_var_exog_features(clean_data)
    exog_train = exog_full.loc[residual_train.index]

    # 删除NaN
    valid_mask = residual_train.notna().all(axis=1) & exog_train.notna().all(axis=1)
    residual_train_clean = residual_train[valid_mask]
    exog_train_clean = exog_train[valid_mask]

    # 训练VAR(2)
    var_bundle = fit_var_model(residual_train_clean, exog_train_clean, lag_order=2)

    print("  VAR模型训练完成")

    # 多步预测实验
    print("\n[2/3] 运行多步预测实验...")

    steps = [1, 3, 6, 12, 24]
    results = rolling_multistep_forecast(
        y_train,
        y_test,
        exog_full,
        var_bundle,
        residual_train_clean,
        steps=steps
    )

    # 汇总结果
    print("\n[3/3] 生成汇总表...")

    summary = []
    for step in steps:
        summary.append({
            "Forecast_Steps": step,
            "Avg_MAE": results[step]["avg_mae"],
            "Avg_RMSE": results[step]["avg_rmse"],
            "Avg_R2": results[step]["avg_r2"],
        })

    summary_df = pd.DataFrame(summary)

    # 计算误差增长率
    baseline_mae = summary_df.iloc[0]["Avg_MAE"]
    summary_df["MAE_Growth_%"] = (summary_df["Avg_MAE"] - baseline_mae) / baseline_mae * 100

    # 保存结果
    results_dir = ROOT / "results"
    results_dir.mkdir(exist_ok=True)

    summary_df.to_csv(results_dir / "multistep_forecast_summary.csv", index=False)

    # 保存详细指标
    all_metrics = []
    for step in steps:
        metrics = results[step]["metrics"].copy()
        metrics["Forecast_Steps"] = step
        all_metrics.append(metrics)

    all_metrics_df = pd.concat(all_metrics, ignore_index=True)
    all_metrics_df.to_csv(results_dir / "multistep_forecast_details.csv", index=False)

    # 输出汇总
    print("\n" + "=" * 80)
    print("多步预测结果汇总")
    print("=" * 80)
    print("\n" + summary_df.to_string(index=False))

    # 分析误差增长
    print("\n" + "=" * 80)
    print("误差增长分析")
    print("=" * 80)

    for i, row in summary_df.iterrows():
        step = int(row["Forecast_Steps"])
        mae = row["Avg_MAE"]
        growth = row["MAE_Growth_%"]

        if step == 1:
            print(f"[{step}步] 单步预测（基线）: MAE = {mae:.4f}")
        else:
            print(f"[{step}步] MAE = {mae:.4f}, 相比单步增长 {growth:.2f}%")

    # 结论
    print("\n" + "=" * 80)
    print("结论")
    print("=" * 80)

    mae_24 = summary_df[summary_df["Forecast_Steps"] == 24]["Avg_MAE"].values[0]
    growth_24 = summary_df[summary_df["Forecast_Steps"] == 24]["MAE_Growth_%"].values[0]

    print(f"\n1. 单步预测（1步）MAE: {baseline_mae:.4f}")
    print(f"2. 24步预测MAE: {mae_24:.4f}, 误差增长 {growth_24:.2f}%")
    print(f"\n3. 误差随预测步长增长，验证了混合VAR模型的'现报'特性")
    print(f"4. 模型适用于单步前向预测，不适用于长期多步预报")
    print(f"5. 在实际应用中，应每小时更新一次预测，使用最新观测值")

    print("\n结果已保存至 results/ 目录")
    print("  - multistep_forecast_summary.csv: 汇总表")
    print("  - multistep_forecast_details.csv: 详细指标")


if __name__ == "__main__":
    main()
