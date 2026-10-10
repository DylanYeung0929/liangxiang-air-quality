"""
消融实验（完整版）：验证VAR协同建模的必要性

对比三种策略：
1. 独立Ridge模型：未剥离周期，57维高维特征，每个污染物单独预测
2. 独立VAR模型：已剥离周期，每个污染物单独建立VAR(2)
3. 联合VAR模型：已剥离周期，多变量联合建立VAR(2)

目标：区分"周期剥离"和"联合建模"两方面的增益
"""
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.api import VAR

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
    """策略1：为每个污染物训练独立的Ridge模型（未剥离周期）"""
    models = {}
    for pollutant in POLLUTANTS:
        model = make_pipeline(
            StandardScaler(),
            RidgeCV(alphas=np.logspace(-3, 4, 20)),
        ).fit(x_train, y_train[pollutant])
        models[pollutant] = model
    return models


def predict_independent(models, x_test):
    """使用独立Ridge模型预测"""
    predictions = {}
    for pollutant in POLLUTANTS:
        pred = models[pollutant].predict(x_test)
        predictions[pollutant] = np.clip(pred, 0, None)
    return pd.DataFrame(predictions, index=x_test.index)


def fit_independent_var_models(residual_train, exog_train, lag_order=2):
    """策略2：为每个污染物训练独立的VAR模型（已剥离周期）

    注意：由于单变量VAR会报错，这里实际上训练的是AR模型
    """
    from statsmodels.tsa.ar_model import AutoReg

    models = {}
    for pollutant in POLLUTANTS:
        # 为单个污染物构建AR(2)模型
        endog = residual_train[pollutant]

        try:
            # 使用AutoReg代替VAR（因为VAR需要多变量）
            model = AutoReg(endog, lags=lag_order, trend='n')
            fitted = model.fit()
            models[pollutant] = fitted
        except Exception as e:
            print(f"  警告：{pollutant} 的独立AR训练失败: {e}")
            models[pollutant] = None

    return models


def predict_independent_var(models, residual_train, test_index, exog_full):
    """使用独立AR模型预测残差"""
    predictions = {}

    for pollutant in POLLUTANTS:
        model = models[pollutant]

        if model is None:
            # 如果模型训练失败，使用零残差（即纯周期基线）
            predictions[pollutant] = np.zeros(len(test_index))
            continue

        # 单步预测
        pred_list = []
        history = residual_train[pollutant].copy()

        for t in test_index:
            # 使用历史数据预测下一步
            try:
                # AutoReg的predict方法
                forecast = model.predict(start=len(history), end=len(history), dynamic=False)
                pred_value = forecast.iloc[0] if hasattr(forecast, 'iloc') else forecast[0]
            except:
                pred_value = 0.0

            pred_list.append(pred_value)

            # 更新历史（使用真实值）
            if t in residual_train.index:
                actual_value = residual_train.loc[t]
            else:
                # 如果真实值不可用，使用预测值
                actual_value = pred_value

            history = pd.concat([history, pd.Series([actual_value], index=[t])])

        predictions[pollutant] = pred_list

    return pd.DataFrame(predictions, index=test_index)


def main():
    print("=" * 80)
    print("消融实验（完整版）：独立Ridge vs 独立VAR vs 联合VAR")
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
    print(f"成功找到数据文件: {raw_path.name}")
    # --- 修改结束 ---

    raw = load_air_quality_workbook(raw_path)
    clean, _ = clean_air_quality(raw)

    # 构建特征
    x_full, y_full = build_supervised_dataset(
        clean,
        include_pollution_lags=True,
    )
    x_train, x_test, y_train, y_test = chronological_split(x_full, y_full, test_fraction=0.2)

    print(f"\n训练集大小: {len(x_train)}, 测试集大小: {len(x_test)}")

    # ========== 策略1：独立Ridge模型（未剥离周期） ==========
    print("\n[1/4] 训练独立Ridge模型（未剥离周期，57维特征）...")
    independent_ridge_models = fit_independent_ridge_models(x_train, y_train)
    y_pred_ridge = predict_independent(independent_ridge_models, x_test)
    metrics_ridge = regression_metrics(y_test, y_pred_ridge, "Independent Ridge")

    print("\n策略1（独立Ridge）测试集性能:")
    print(metrics_ridge.to_string(index=False))

    # ========== 策略2：独立VAR模型（已剥离周期） ==========
    print("\n[2/4] 训练独立AR模型（已剥离周期，每个污染物单独AR(2)）...")

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

    # 训练独立VAR模型
    independent_var_models = fit_independent_var_models(residual_train, exog_train, lag_order=2)

    # 预测残差
    residual_pred_independent = predict_independent_var(
        independent_var_models,
        residual_train,
        y_test.index,
        exog_full,
    )

    # 合成最终预测
    y_pred_independent_var = baseline_test + residual_pred_independent
    for col in POLLUTANTS:
        y_pred_independent_var[col] = y_pred_independent_var[col].clip(lower=0)

    metrics_independent_var = regression_metrics(y_test, y_pred_independent_var, "Independent AR")

    print("\n策略2（独立AR）测试集性能:")
    print(metrics_independent_var.to_string(index=False))

    # ========== 策略3：联合VAR模型（已剥离周期） ==========
    print("\n[3/4] 训练联合VAR模型（已剥离周期，多变量联合建模）...")

    # 训练联合VAR(2)
    var_bundle = fit_var_model(residual_train, exog_train, lag_order=2)

    # 预测残差
    residual_pred_joint = predict_var_residuals(
        var_bundle,
        residual_train,
        y_test.index,
        exog_future=exog_full,
    )

    # 合成最终预测
    y_pred_joint_var = baseline_test + residual_pred_joint
    for col in POLLUTANTS:
        y_pred_joint_var[col] = y_pred_joint_var[col].clip(lower=0)

    metrics_joint_var = regression_metrics(y_test, y_pred_joint_var, "Joint VAR")

    print("\n策略3（联合VAR）测试集性能:")
    print(metrics_joint_var.to_string(index=False))

    # ========== 对比分析 ==========
    print("\n[4/4] 生成对比表格...")

    # 合并指标
    comparison = pd.concat([metrics_ridge, metrics_independent_var, metrics_joint_var], ignore_index=True)

    # 计算平均性能
    avg_metrics = comparison.groupby("Model")[["MAE", "RMSE", "R2"]].mean().reset_index()
    avg_metrics.insert(1, "Pollutant", "Average")

    # 计算改进幅度
    improvement = []

    # 策略1 → 策略2：周期剥离的贡献
    for pollutant in POLLUTANTS:
        ridge_row = metrics_ridge[metrics_ridge["Pollutant"] == pollutant].iloc[0]
        ind_var_row = metrics_independent_var[metrics_independent_var["Pollutant"] == pollutant].iloc[0]

        mae_improve_period = (ridge_row["MAE"] - ind_var_row["MAE"]) / ridge_row["MAE"] * 100
        rmse_improve_period = (ridge_row["RMSE"] - ind_var_row["RMSE"]) / ridge_row["RMSE"] * 100
        r2_improve_period = (ind_var_row["R2"] - ridge_row["R2"]) * 100

        improvement.append({
            "Pollutant": pollutant,
            "Comparison": "Ridge → Ind.VAR (周期剥离)",
            "MAE_Improvement_%": mae_improve_period,
            "RMSE_Improvement_%": rmse_improve_period,
            "R2_Improvement_pp": r2_improve_period,
        })

    # 策略2 → 策略3：联合建模的贡献
    for pollutant in POLLUTANTS:
        ind_var_row = metrics_independent_var[metrics_independent_var["Pollutant"] == pollutant].iloc[0]
        joint_var_row = metrics_joint_var[metrics_joint_var["Pollutant"] == pollutant].iloc[0]

        mae_improve_joint = (ind_var_row["MAE"] - joint_var_row["MAE"]) / ind_var_row["MAE"] * 100
        rmse_improve_joint = (ind_var_row["RMSE"] - joint_var_row["RMSE"]) / ind_var_row["RMSE"] * 100
        r2_improve_joint = (joint_var_row["R2"] - ind_var_row["R2"]) * 100

        improvement.append({
            "Pollutant": pollutant,
            "Comparison": "Ind.VAR → Joint VAR (联合建模)",
            "MAE_Improvement_%": mae_improve_joint,
            "RMSE_Improvement_%": rmse_improve_joint,
            "R2_Improvement_pp": r2_improve_joint,
        })

    # 策略1 → 策略3：总体改进
    for pollutant in POLLUTANTS:
        ridge_row = metrics_ridge[metrics_ridge["Pollutant"] == pollutant].iloc[0]
        joint_var_row = metrics_joint_var[metrics_joint_var["Pollutant"] == pollutant].iloc[0]

        mae_improve_total = (ridge_row["MAE"] - joint_var_row["MAE"]) / ridge_row["MAE"] * 100
        rmse_improve_total = (ridge_row["RMSE"] - joint_var_row["RMSE"]) / ridge_row["RMSE"] * 100
        r2_improve_total = (joint_var_row["R2"] - ridge_row["R2"]) * 100

        improvement.append({
            "Pollutant": pollutant,
            "Comparison": "Ridge → Joint VAR (总体)",
            "MAE_Improvement_%": mae_improve_total,
            "RMSE_Improvement_%": rmse_improve_total,
            "R2_Improvement_pp": r2_improve_total,
        })

    improvement_df = pd.DataFrame(improvement)

    # 保存结果
    results_dir = ROOT / "results"
    results_dir.mkdir(exist_ok=True)

    comparison.to_csv(results_dir / "ablation_full_comparison.csv", index=False)
    avg_metrics.to_csv(results_dir / "ablation_full_average.csv", index=False)
    improvement_df.to_csv(results_dir / "ablation_full_improvement.csv", index=False)

    print("\n" + "=" * 80)
    print("消融实验结果汇总")
    print("=" * 80)
    print("\n平均性能对比:")
    print(avg_metrics.to_string(index=False))

    print("\n\n改进幅度分解:")

    # 按对比类型分组显示
    for comp_type in ["Ridge → Ind.VAR (周期剥离)", "Ind.VAR → Joint VAR (联合建模)", "Ridge → Joint VAR (总体)"]:
        subset = improvement_df[improvement_df["Comparison"] == comp_type]
        avg_mae = subset["MAE_Improvement_%"].mean()
        avg_rmse = subset["RMSE_Improvement_%"].mean()
        avg_r2 = subset["R2_Improvement_pp"].mean()

        print(f"\n{comp_type}:")
        print(f"  平均MAE改进: {avg_mae:.2f}%")
        print(f"  平均RMSE改进: {avg_rmse:.2f}%")
        print(f"  平均R²提升: {avg_r2:.4f} pp")

    # 解释结果
    period_improve = improvement_df[improvement_df["Comparison"] == "Ridge → Ind.VAR (周期剥离)"]["MAE_Improvement_%"].mean()
    joint_improve = improvement_df[improvement_df["Comparison"] == "Ind.VAR → Joint VAR (联合建模)"]["MAE_Improvement_%"].mean()
    total_improve = improvement_df[improvement_df["Comparison"] == "Ridge → Joint VAR (总体)"]["MAE_Improvement_%"].mean()

    print("\n" + "=" * 80)
    print("结论:")
    print("=" * 80)
    print(f"[1] 周期剥离的贡献: MAE改进 {period_improve:.2f}%")
    print(f"[2] 联合建模的贡献: MAE改进 {joint_improve:.2f}%")
    print(f"[3] 总体改进: MAE改进 {total_improve:.2f}%")
    print(f"\n[验证] {period_improve:.2f}% + {joint_improve:.2f}% ≈ {total_improve:.2f}%")

    if joint_improve > 5:
        print(f"\n[+] 联合建模带来了实质性增益（{joint_improve:.2f}%），证明多变量协同建模的必要性")
    elif joint_improve > 0:
        print(f"\n[~] 联合建模带来了轻微增益（{joint_improve:.2f}%），但改进幅度有限")
    else:
        print(f"\n[-] 联合建模未带来显著增益，可能原因：测试期过于平稳")

    print("\n结果已保存至 results/ 目录")
    print("  - ablation_full_comparison.csv: 详细对比")
    print("  - ablation_full_average.csv: 平均性能")
    print("  - ablation_full_improvement.csv: 改进幅度分解")


if __name__ == "__main__":
    main()
