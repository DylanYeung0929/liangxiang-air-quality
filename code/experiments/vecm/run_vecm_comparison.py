"""
VECM vs VAR 对比实验
训练VECM模型并与原VAR模型对比性能
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
    METEOROLOGY,
    build_var_exog_features,
    fit_var_model,
    fit_vecm_model,
    predict_var_residuals,
    predict_vecm_residuals,
    evaluate_predictions,
)


def plot_comparison(
    y_true: pd.DataFrame,
    y_pred_var: pd.DataFrame,
    y_pred_vecm: pd.DataFrame,
    output_path: Path
):
    """绘制VAR vs VECM预测对比图"""
    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    axes = axes.flatten()

    for i, pollutant in enumerate(POLLUTANTS):
        ax = axes[i]
        ax.plot(y_true.index, y_true[pollutant], 'k-', label='Observed', linewidth=1.5, alpha=0.7)
        ax.plot(y_pred_var.index, y_pred_var[pollutant], 'b--', label='VAR(2)', linewidth=1.2, alpha=0.8)
        ax.plot(y_pred_vecm.index, y_pred_vecm[pollutant], 'r-', label='VECM', linewidth=1.2, alpha=0.8)
        ax.set_title(pollutant, fontsize=12, fontweight='bold')
        ax.set_xlabel('Time', fontsize=10)
        ax.set_ylabel('Concentration', fontsize=10)
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  [OK] 对比图已保存: {output_path.relative_to(project_root)}")


def plot_metrics_comparison(
    metrics_var: pd.DataFrame,
    metrics_vecm: pd.DataFrame,
    output_path: Path
):
    """绘制VAR vs VECM指标对比柱状图"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    metrics = ['MAE', 'RMSE', 'R2']
    colors = ['#3498db', '#e74c3c']

    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        x = np.arange(len(POLLUTANTS))
        width = 0.35

        var_values = metrics_var[metric].values
        vecm_values = metrics_vecm[metric].values

        ax.bar(x - width/2, var_values, width, label='VAR(2)', color=colors[0], alpha=0.8)
        ax.bar(x + width/2, vecm_values, width, label='VECM', color=colors[1], alpha=0.8)

        ax.set_xlabel('Pollutant', fontsize=11)
        ax.set_ylabel(metric, fontsize=11)
        ax.set_title(f'{metric} Comparison', fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(POLLUTANTS, rotation=45, ha='right')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  [OK] 指标对比图已保存: {output_path.relative_to(project_root)}")


def main():
    print("=" * 80)
    print("VECM vs VAR 对比实验")
    print("=" * 80)

    # 1. 加载数据
    data_path = project_root / "4-2026年校赛第二轮A题数据.xlsx"
    print(f"\n[1/6] 加载数据: {data_path.name}")
    raw = load_air_quality_workbook(data_path)
    clean, report = clean_air_quality(raw)
    print(f"  [OK] 清洗后样本数: {len(clean)}")

    # 2. 计算残差序列
    print("\n[2/6] 计算残差序列（Y_t - Y_{t-24}）")
    baseline = clean[POLLUTANTS].shift(24)
    residuals = clean[POLLUTANTS] - baseline
    residuals = residuals.iloc[24:].copy()
    print(f"  [OK] 残差序列长度: {len(residuals)}")

    # 3. 构造外生变量并对齐
    print("\n[3/6] 构造外生变量并划分训练集和测试集（80% / 20%）")
    split_idx = int(len(residuals) * 0.8)
    train_residuals = residuals.iloc[:split_idx].copy()
    test_residuals = residuals.iloc[split_idx:].copy()

    # 构造外生变量
    exog_full = build_var_exog_features(clean.iloc[24:])

    # 删除NaN行
    valid_idx = exog_full.dropna().index
    residuals = residuals.loc[valid_idx]
    exog_full = exog_full.loc[valid_idx]

    # 划分训练集和测试集
    split_idx = int(len(residuals) * 0.8)
    train_residuals = residuals.iloc[:split_idx].copy()
    test_residuals = residuals.iloc[split_idx:].copy()
    exog_train = exog_full.iloc[:split_idx].copy()
    exog_test = exog_full.iloc[split_idx:].copy()

    print(f"  [OK] 训练集: {len(train_residuals)} 样本")
    print(f"  [OK] 测试集: {len(test_residuals)} 样本")

    # 4. 训练VAR(2)模型
    print("\n[4/6] 训练VAR(2)模型")
    try:
        var_bundle = fit_var_model(
            residual_train=train_residuals,
            exog_train=exog_train,
            lag_order=2
        )
        print(f"  [OK] VAR模型训练完成，滞后阶数: {var_bundle.lag_order}")

        # 预测
        var_pred_residuals = predict_var_residuals(
            model_bundle=var_bundle,
            residual_train=train_residuals,
            target_index=test_residuals.index,
            exog_future=exog_test
        )

        # 还原到原始尺度
        baseline_test = clean[POLLUTANTS].loc[test_residuals.index].shift(24)
        var_pred = baseline_test + var_pred_residuals
        var_pred = var_pred.dropna()

        # 评估
        y_true_var = clean[POLLUTANTS].loc[var_pred.index]
        var_metrics = evaluate_predictions(y_true_var, var_pred)
        print(f"  [OK] VAR平均MAE: {var_metrics['MAE'].mean():.4f}")
        print(f"  [OK] VAR平均R2: {var_metrics['R2'].mean():.4f}")

    except Exception as e:
        print(f"  [ERROR] VAR训练失败: {e}")
        var_pred = None
        var_metrics = None

    # 5. 训练VECM模型
    print("\n[5/6] 训练VECM模型")
    try:
        vecm_bundle = fit_vecm_model(
            residual_train=train_residuals,
            exog_train=exog_train,
            k_ar_diff=1,
            coint_rank=6,  # 根据协整检验结果
            deterministic='ci'
        )
        print(f"  [OK] VECM模型训练完成")
        print(f"      k_ar_diff: {vecm_bundle.k_ar_diff}")
        print(f"      coint_rank: {vecm_bundle.coint_rank}")
        print(f"      deterministic: {vecm_bundle.deterministic}")

        # 预测
        vecm_pred_residuals = predict_vecm_residuals(
            model_bundle=vecm_bundle,
            residual_train=train_residuals,
            target_index=test_residuals.index,
            exog_future=exog_test
        )

        # 还原到原始尺度
        baseline_test = clean[POLLUTANTS].loc[test_residuals.index].shift(24)
        vecm_pred = baseline_test + vecm_pred_residuals
        vecm_pred = vecm_pred.dropna()

        # 评估
        y_true_vecm = clean[POLLUTANTS].loc[vecm_pred.index]
        vecm_metrics = evaluate_predictions(y_true_vecm, vecm_pred)
        print(f"  [OK] VECM平均MAE: {vecm_metrics['MAE'].mean():.4f}")
        print(f"  [OK] VECM平均R2: {vecm_metrics['R2'].mean():.4f}")

    except Exception as e:
        print(f"  [ERROR] VECM训练失败: {e}")
        import traceback
        traceback.print_exc()
        vecm_pred = None
        vecm_metrics = None

    # 6. 保存结果和绘图
    print("\n[6/6] 保存结果和绘图")

    results_dir = project_root / "results" / "vecm"
    figures_dir = project_root / "figures" / "vecm"
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    if var_metrics is not None:
        var_metrics.to_csv(
            results_dir / "var_metrics.csv",
            index=False,
            encoding='utf-8-sig'
        )
        print(f"  [OK] VAR指标已保存")

    if vecm_metrics is not None:
        vecm_metrics.to_csv(
            results_dir / "vecm_metrics.csv",
            index=False,
            encoding='utf-8-sig'
        )
        print(f"  [OK] VECM指标已保存")

    # 对比表格
    if var_metrics is not None and vecm_metrics is not None:
        comparison = pd.DataFrame({
            'Pollutant': POLLUTANTS,
            'VAR_MAE': var_metrics['MAE'].values,
            'VECM_MAE': vecm_metrics['MAE'].values,
            'MAE_Improvement_%': ((var_metrics['MAE'] - vecm_metrics['MAE']) / var_metrics['MAE'] * 100).values,
            'VAR_RMSE': var_metrics['RMSE'].values,
            'VECM_RMSE': vecm_metrics['RMSE'].values,
            'RMSE_Improvement_%': ((var_metrics['RMSE'] - vecm_metrics['RMSE']) / var_metrics['RMSE'] * 100).values,
            'VAR_R2': var_metrics['R2'].values,
            'VECM_R2': vecm_metrics['R2'].values,
        })
        comparison.to_csv(
            results_dir / "comparison.csv",
            index=False,
            encoding='utf-8-sig'
        )
        print(f"  [OK] 对比表格已保存")

        # 汇总统计
        summary = pd.DataFrame({
            'Model': ['VAR(2)', 'VECM'],
            'Avg_MAE': [var_metrics['MAE'].mean(), vecm_metrics['MAE'].mean()],
            'Avg_RMSE': [var_metrics['RMSE'].mean(), vecm_metrics['RMSE'].mean()],
            'Avg_R2': [var_metrics['R2'].mean(), vecm_metrics['R2'].mean()],
        })
        summary['MAE_Improvement_%'] = [0, (summary.loc[0, 'Avg_MAE'] - summary.loc[1, 'Avg_MAE']) / summary.loc[0, 'Avg_MAE'] * 100]
        summary.to_csv(
            results_dir / "summary.csv",
            index=False,
            encoding='utf-8-sig'
        )
        print(f"  [OK] 汇总统计已保存")

        # 绘图
        if var_pred is not None and vecm_pred is not None:
            # 对齐索引
            common_idx = var_pred.index.intersection(vecm_pred.index)
            plot_comparison(
                y_true=y_true_var.loc[common_idx],
                y_pred_var=var_pred.loc[common_idx],
                y_pred_vecm=vecm_pred.loc[common_idx],
                output_path=figures_dir / "var_vs_vecm_predictions.pdf"
            )

            plot_metrics_comparison(
                metrics_var=var_metrics,
                metrics_vecm=vecm_metrics,
                output_path=figures_dir / "var_vs_vecm_metrics.pdf"
            )

    print("\n" + "=" * 80)
    print("实验完成！")
    print("=" * 80)

    if var_metrics is not None and vecm_metrics is not None:
        print(f"\n模型对比汇总:")
        print(f"  VAR(2)  - 平均MAE: {var_metrics['MAE'].mean():.4f}, 平均R2: {var_metrics['R2'].mean():.4f}")
        print(f"  VECM    - 平均MAE: {vecm_metrics['MAE'].mean():.4f}, 平均R2: {vecm_metrics['R2'].mean():.4f}")
        improvement = (var_metrics['MAE'].mean() - vecm_metrics['MAE'].mean()) / var_metrics['MAE'].mean() * 100
        print(f"  改进幅度: {improvement:.2f}%")


if __name__ == "__main__":
    main()
