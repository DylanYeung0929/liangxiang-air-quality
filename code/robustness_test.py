"""
鲁棒性分析：气象突变场景压力测试

目标：
1. 识别训练集中的气象突变时段（温度骤降>5°C）
2. 测试模型在突变时段的预测误差
3. 验证VAR的气象外生项是否在突变时提供有效修正
"""

import numpy as np
import pandas as pd
from pathlib import Path

import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.air_quality import (
    POLLUTANTS,
    METEOROLOGY,
    load_air_quality_workbook,
    clean_air_quality,
    regression_metrics,
    seasonal_lag_frame,
    build_var_exog_features,
    fit_var_model,
    predict_var_residuals,
)


def identify_temperature_shocks(clean_df, threshold_drop=5.0, window_hours=6):
    """
    识别温度骤降事件

    参数:
        threshold_drop: 温度下降阈值（°C）
        window_hours: 观察窗口（小时）

    返回:
        包含突变时段信息的DataFrame
    """
    temp = clean_df["temperature"]

    # 计算滚动窗口内的温度变化
    temp_change = temp - temp.shift(window_hours)

    # 识别骤降事件（负值表示下降）
    shock_mask = temp_change < -threshold_drop

    if shock_mask.sum() == 0:
        print(f"未发现温度骤降>{threshold_drop}°C的事件（窗口={window_hours}小时）")
        return pd.DataFrame()

    # 提取突变时段
    shock_periods = []
    for idx in clean_df[shock_mask].index:
        start = idx - pd.Timedelta(hours=window_hours)
        end = idx + pd.Timedelta(hours=window_hours)

        shock_periods.append({
            "shock_time": idx,
            "start": start,
            "end": end,
            "temp_drop": temp_change.loc[idx],
            "temp_before": temp.shift(window_hours).loc[idx],
            "temp_after": temp.loc[idx],
        })

    return pd.DataFrame(shock_periods)


def identify_temperature_spikes(clean_df, threshold_rise=5.0, window_hours=6):
    """识别温度骤升事件"""
    temp = clean_df["temperature"]
    temp_change = temp - temp.shift(window_hours)

    shock_mask = temp_change > threshold_rise

    if shock_mask.sum() == 0:
        print(f"未发现温度骤升>{threshold_rise}°C的事件（窗口={window_hours}小时）")
        return pd.DataFrame()

    shock_periods = []
    for idx in clean_df[shock_mask].index:
        start = idx - pd.Timedelta(hours=window_hours)
        end = idx + pd.Timedelta(hours=window_hours)

        shock_periods.append({
            "shock_time": idx,
            "start": start,
            "end": end,
            "temp_rise": temp_change.loc[idx],
            "temp_before": temp.shift(window_hours).loc[idx],
            "temp_after": temp.loc[idx],
        })

    return pd.DataFrame(shock_periods)


def evaluate_on_periods(y_true, y_pred_baseline, y_pred_var, periods_df, period_name):
    """评估模型在特定时段的性能"""
    if len(periods_df) == 0:
        return pd.DataFrame()

    results = []
    for _, period in periods_df.iterrows():
        mask = (y_true.index >= period["start"]) & (y_true.index <= period["end"])

        if mask.sum() == 0:
            continue

        y_period = y_true[mask]
        baseline_period = y_pred_baseline[mask]
        var_period = y_pred_var[mask]

        metrics_baseline = regression_metrics(y_period, baseline_period, "Baseline")
        metrics_var = regression_metrics(y_period, var_period, "VAR")

        # 计算平均误差
        avg_baseline = metrics_baseline[["MAE", "RMSE", "R2"]].mean()
        avg_var = metrics_var[["MAE", "RMSE", "R2"]].mean()

        results.append({
            "period_name": period_name,
            "shock_time": str(period["shock_time"]),
            "start": str(period["start"]),
            "end": str(period["end"]),
            "hours": mask.sum(),
            "baseline_MAE": avg_baseline["MAE"],
            "baseline_RMSE": avg_baseline["RMSE"],
            "baseline_R2": avg_baseline["R2"],
            "VAR_MAE": avg_var["MAE"],
            "VAR_RMSE": avg_var["RMSE"],
            "VAR_R2": avg_var["R2"],
            "MAE_improvement_%": (avg_baseline["MAE"] - avg_var["MAE"]) / avg_baseline["MAE"] * 100,
            "RMSE_improvement_%": (avg_baseline["RMSE"] - avg_var["RMSE"]) / avg_baseline["RMSE"] * 100,
        })

    return pd.DataFrame(results)


def main():
    print("=" * 80)
    print("鲁棒性分析：气象突变场景压力测试")
    print("=" * 80)

    # 加载数据
    # --- 修改开始：使用自动搜索逻辑 ---
    # ROOT 已经在脚本上方定义为 Path(__file__).resolve().parents[1] (即 ver2 目录)
    xlsx_files = list(ROOT.glob("*.xlsx"))

    # 如果根目录没找到，尝试在 data 文件夹找
    if not xlsx_files:
        xlsx_files = list((ROOT / "data").glob("*.xlsx"))

    if not xlsx_files:
        raise FileNotFoundError(f"在 {ROOT} 及其 data 子目录下未找到任何 .xlsx 文件。请确保数据文件已放置。")

    raw_path = xlsx_files[0]

    raw = load_air_quality_workbook(raw_path)
    clean, _ = clean_air_quality(raw)

    print(f"\n数据时间范围: {clean.index.min()} 至 {clean.index.max()}")
    print(f"总样本数: {len(clean)}")

    # 识别气象突变事件
    print("\n[1/4] 识别气象突变事件...")

    # 温度骤降
    temp_drops = identify_temperature_shocks(clean, threshold_drop=5.0, window_hours=6)
    print(f"  发现温度骤降事件: {len(temp_drops)} 个")

    # 温度骤升
    temp_rises = identify_temperature_spikes(clean, threshold_rise=5.0, window_hours=6)
    print(f"  发现温度骤升事件: {len(temp_rises)} 个")

    if len(temp_drops) == 0 and len(temp_rises) == 0:
        print("\n未发现显著气象突变事件，尝试降低阈值...")
        temp_drops = identify_temperature_shocks(clean, threshold_drop=3.0, window_hours=6)
        temp_rises = identify_temperature_spikes(clean, threshold_rise=3.0, window_hours=6)
        print(f"  (阈值=3°C) 温度骤降: {len(temp_drops)} 个, 温度骤升: {len(temp_rises)} 个")

    # 训练模型
    print("\n[2/4] 训练模型...")

    # 使用前80%作为训练集
    split_idx = int(len(clean) * 0.8)
    train_data = clean.iloc[:split_idx]
    test_data = clean.iloc[split_idx:]

    y_train = train_data[POLLUTANTS]
    y_test = test_data[POLLUTANTS]

    # 周期基线
    baseline_train = seasonal_lag_frame(y_train, y_train.index, lag_hours=24)
    baseline_test = seasonal_lag_frame(clean[POLLUTANTS], y_test.index, lag_hours=24)

    # VAR残差模型
    residual_train = y_train - baseline_train
    exog_full = build_var_exog_features(clean)
    exog_train = exog_full.loc[residual_train.index]

    # 删除包含NaN的行
    valid_mask = residual_train.notna().all(axis=1) & exog_train.notna().all(axis=1)
    residual_train_clean = residual_train[valid_mask]
    exog_train_clean = exog_train[valid_mask]

    var_bundle = fit_var_model(residual_train_clean, exog_train_clean, lag_order=2)

    residual_pred = predict_var_residuals(
        var_bundle,
        residual_train_clean,
        y_test.index,
        exog_future=exog_full,
    )

    y_pred_var = baseline_test + residual_pred
    for col in POLLUTANTS:
        y_pred_var[col] = y_pred_var[col].clip(lower=0)

    print("  模型训练完成")

    # 评估突变时段性能
    print("\n[3/4] 评估突变时段性能...")

    results_drops = evaluate_on_periods(
        y_test, baseline_test, y_pred_var, temp_drops, "Temperature Drop"
    )

    results_rises = evaluate_on_periods(
        y_test, baseline_test, y_pred_var, temp_rises, "Temperature Rise"
    )

    # 评估平稳时段性能（作为对照）
    print("\n[4/4] 评估平稳时段性能（对照组）...")

    # 识别平稳时段：温度变化<2°C
    temp = clean["temperature"]
    temp_change_6h = abs(temp - temp.shift(6))
    stable_mask = temp_change_6h < 2.0

    stable_periods = []
    # 找连续的平稳时段
    stable_blocks = (stable_mask != stable_mask.shift()).cumsum()
    for block_id, group in clean[stable_mask].groupby(stable_blocks[stable_mask]):
        if len(group) >= 12:  # 至少12小时
            stable_periods.append({
                "shock_time": group.index[len(group)//2],
                "start": group.index[0],
                "end": group.index[-1],
            })

    stable_periods_df = pd.DataFrame(stable_periods)
    results_stable = evaluate_on_periods(
        y_test, baseline_test, y_pred_var, stable_periods_df, "Stable Period"
    )

    # 合并结果
    all_results = pd.concat([results_drops, results_rises, results_stable], ignore_index=True)

    # 保存结果
    results_dir = ROOT / "results"
    results_dir.mkdir(exist_ok=True)

    all_results.to_csv(results_dir / "robustness_analysis.csv", index=False)

    if len(temp_drops) > 0:
        temp_drops.to_csv(results_dir / "temperature_drop_events.csv", index=False)
    if len(temp_rises) > 0:
        temp_rises.to_csv(results_dir / "temperature_rise_events.csv", index=False)

    # 输出汇总
    print("\n" + "=" * 80)
    print("鲁棒性分析结果汇总")
    print("=" * 80)

    if len(all_results) > 0:
        summary = all_results.groupby("period_name")[
            ["baseline_MAE", "VAR_MAE", "MAE_improvement_%"]
        ].mean()

        print("\n各类时段平均性能:")
        print(summary.to_string())

        print("\n详细结果:")
        print(all_results[["period_name", "shock_time", "hours",
                          "baseline_MAE", "VAR_MAE", "MAE_improvement_%"]].to_string(index=False))
    else:
        print("\n无有效评估结果")

    print("\n结果已保存至 results/ 目录")
    print("  - robustness_analysis.csv: 完整分析结果")
    if len(temp_drops) > 0:
        print("  - temperature_drop_events.csv: 温度骤降事件")
    if len(temp_rises) > 0:
        print("  - temperature_rise_events.csv: 温度骤升事件")


if __name__ == "__main__":
    main()
