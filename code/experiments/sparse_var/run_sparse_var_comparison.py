"""
稀疏VAR对比实验
训练稀疏VAR模型并与VAR(2)、VECM对比性能
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
    fit_sparse_var_model,
    predict_var_residuals,
    predict_vecm_residuals,
    predict_sparse_var_residuals,
    evaluate_predictions,
)


def plot_sparsity_heatmap(coef_matrix: np.ndarray, output_path: Path):
    """绘制稀疏VAR系数矩阵热图"""
    fig, ax = plt.subplots(figsize=(12, 8))

    # 只显示非零系数
    mask = np.abs(coef_matrix) < 1e-10
    coef_display = coef_matrix.copy()
    coef_display[mask] = np.nan

    im = ax.imshow(coef_display, cmap='RdBu_r', aspect='auto', vmin=-0.5, vmax=0.5)

    ax.set_yticks(range(len(POLLUTANTS)))
    ax.set_yticklabels(POLLUTANTS)
    ax.set_xlabel('Feature Index', fontsize=11)
    ax.set_ylabel('Target Pollutant', fontsize=11)
    ax.set_title('Sparse VAR Coefficient Matrix (Non-zero coefficients only)',
                 fontsize=13, fontweight='bold')

    plt.colorbar(im, ax=ax, label='Coefficient Value')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  [OK] 稀疏性热图已保存: {output_path.relative_to(project_root)}")


def plot_three_model_comparison(
    y_true: pd.DataFrame,
    y_pred_var: pd.DataFrame,
    y_pred_vecm: pd.DataFrame,
    y_pred_sparse: pd.DataFrame,
    output_path: Path
):
    """绘制三个模型的预测对比图"""
    fig, axes = plt.subplots(3, 2, figsize=(15, 13))
    axes = axes.flatten()

    for i, pollutant in enumerate(POLLUTANTS):
        ax = axes[i]
        ax.plot(y_true.index, y_true[pollutant], 'k-',
                label='Observed', linewidth=1.5, alpha=0.7)
        ax.plot(y_pred_var.index, y_pred_var[pollutant], 'b--',
                label='VAR(2)', linewidth=1.2, alpha=0.7)
        ax.plot(y_pred_vecm.index, y_pred_vecm[pollutant], 'g:',
                label='VECM', linewidth=1.2, alpha=0.7)
        ax.plot(y_pred_sparse.index, y_pred_sparse[pollutant], 'r-',
                label='Sparse VAR', linewidth=1.3, alpha=0.8)

        ax.set_title(pollutant, fontsize=12, fontweight='bold')
        ax.set_xlabel('Time', fontsize=10)
        ax.set_ylabel('Concentration', fontsize=10)
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  [OK] 三模型对比图已保存: {output_path.relative_to(project_root)}")


def plot_metrics_bar_comparison(
    metrics_var: pd.DataFrame,
    metrics_vecm: pd.DataFrame,
    metrics_sparse: pd.DataFrame,
    output_path: Path
):
    """绘制三个模型的指标对比柱状图"""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    metrics = ['MAE', 'RMSE', 'R2']
    colors = ['#3498db', '#2ecc71', '#e74c3c']

    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        x = np.arange(len(POLLUTANTS))
        width = 0.25

        var_values = metrics_var[metric].values
        vecm_values = metrics_vecm[metric].values
        sparse_values = metrics_sparse[metric].values

        ax.bar(x - width, var_values, width, label='VAR(2)', color=colors[0], alpha=0.8)
        ax.bar(x, vecm_values, width, label='VECM', color=colors[1], alpha=0.8)
        ax.bar(x + width, sparse_values, width, label='Sparse VAR', color=colors[2], alpha=0.8)

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
    print(f"  [OK] 指标对比柱状图已保存: {output_path.relative_to(project_root)}")


def main():
    print("=" * 80)
    print("稀疏VAR对比实验")
    print("=" * 80)

    # 1. 加载数据
    data_path = project_root / "4-2026年校赛第二轮A题数据.xlsx"
    print(f"\n[1/7] 加载数据: {data_path.name}")
    raw = load_air_quality_workbook(data_path)
    clean, report = clean_air_quality(raw)
    print(f"  [OK] 清洗后样本数: {len(clean)}")

    # 2. 计算残差序列
    print("\n[2/7] 计算残差序列（Y_t - Y_{t-24}）")
    baseline = clean[POLLUTANTS].shift(24)
    residuals = clean[POLLUTANTS] - baseline
    residuals = residuals.iloc[24:].copy()
    print(f"  [OK] 残差序列长度: {len(residuals)}")

    # 3. 构造外生变量并划分数据
    print("\n[3/7] 构造外生变量并划分训练集和测试集（80% / 20%）")
    exog_full = build_var_exog_features(clean.iloc[24:])
    valid_idx = exog_full.dropna().index
    residuals = residuals.loc[valid_idx]
    exog_full = exog_full.loc[valid_idx]

    split_idx = int(len(residuals) * 0.8)
    train_residuals = residuals.iloc[:split_idx].copy()
    test_residuals = residuals.iloc[split_idx:].copy()
    exog_train = exog_full.iloc[:split_idx].copy()
    exog_test = exog_full.iloc[split_idx:].copy()

    print(f"  [OK] 训练集: {len(train_residuals)} 样本")
    print(f"  [OK] 测试集: {len(test_residuals)} 样本")

    # 4. 训练VAR(2)
    print("\n[4/7] 训练VAR(2)模型")
    var_bundle = fit_var_model(train_residuals, exog_train, lag_order=2)
    print(f"  [OK] VAR模型训练完成，滞后阶数: {var_bundle.lag_order}")

    var_pred_residuals = predict_var_residuals(
        var_bundle, train_residuals, test_residuals.index, exog_test
    )
    baseline_test = clean[POLLUTANTS].loc[test_residuals.index].shift(24)
    var_pred = baseline_test + var_pred_residuals
    var_pred = var_pred.dropna()

    y_true_var = clean[POLLUTANTS].loc[var_pred.index]
    var_metrics = evaluate_predictions(y_true_var, var_pred)
    print(f"  [OK] VAR平均MAE: {var_metrics['MAE'].mean():.4f}")

    # 5. 训练VECM
    print("\n[5/7] 训练VECM模型")
    vecm_bundle = fit_vecm_model(train_residuals, exog_train, k_ar_diff=1, coint_rank=6)
    print(f"  [OK] VECM模型训练完成")

    vecm_pred_residuals = predict_vecm_residuals(
        vecm_bundle, train_residuals, test_residuals.index, exog_test
    )
    vecm_pred = baseline_test + vecm_pred_residuals
    vecm_pred = vecm_pred.dropna()

    y_true_vecm = clean[POLLUTANTS].loc[vecm_pred.index]
    vecm_metrics = evaluate_predictions(y_true_vecm, vecm_pred)
    print(f"  [OK] VECM平均MAE: {vecm_metrics['MAE'].mean():.4f}")

    # 6. 训练稀疏VAR
    print("\n[6/7] 训练稀疏VAR模型（MultiTaskLassoCV）")
    sparse_bundle = fit_sparse_var_model(train_residuals, exog_train, lag_order=2)
    print(f"  [OK] 稀疏VAR模型训练完成")
    print(f"      最优alpha: {sparse_bundle.alpha:.6f}")
    print(f"      非零系数数: {sparse_bundle.n_nonzero_coefs}")
    print(f"      稀疏度: {sparse_bundle.sparsity_ratio*100:.2f}%")

    sparse_pred_residuals = predict_sparse_var_residuals(
        sparse_bundle, train_residuals, test_residuals.index, exog_test
    )
    sparse_pred = baseline_test + sparse_pred_residuals
    sparse_pred = sparse_pred.dropna()

    y_true_sparse = clean[POLLUTANTS].loc[sparse_pred.index]
    sparse_metrics = evaluate_predictions(y_true_sparse, sparse_pred)
    print(f"  [OK] 稀疏VAR平均MAE: {sparse_metrics['MAE'].mean():.4f}")

    # 7. 保存结果和绘图
    print("\n[7/7] 保存结果和绘图")

    results_dir = project_root / "results" / "sparse_var"
    figures_dir = project_root / "figures" / "sparse_var"
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    # 保存指标
    var_metrics.to_csv(results_dir / "var_metrics.csv", index=False, encoding='utf-8-sig')
    vecm_metrics.to_csv(results_dir / "vecm_metrics.csv", index=False, encoding='utf-8-sig')
    sparse_metrics.to_csv(results_dir / "sparse_var_metrics.csv", index=False, encoding='utf-8-sig')
    print(f"  [OK] 指标已保存")

    # 对比表格
    comparison = pd.DataFrame({
        'Pollutant': POLLUTANTS,
        'VAR_MAE': var_metrics['MAE'].values,
        'VECM_MAE': vecm_metrics['MAE'].values,
        'SparseVAR_MAE': sparse_metrics['MAE'].values,
        'VAR_vs_Sparse_Improvement_%': ((var_metrics['MAE'] - sparse_metrics['MAE']) / var_metrics['MAE'] * 100).values,
        'VAR_R2': var_metrics['R2'].values,
        'VECM_R2': vecm_metrics['R2'].values,
        'SparseVAR_R2': sparse_metrics['R2'].values,
    })
    comparison.to_csv(results_dir / "comparison.csv", index=False, encoding='utf-8-sig')
    print(f"  [OK] 对比表格已保存")

    # 汇总统计
    summary = pd.DataFrame({
        'Model': ['VAR(2)', 'VECM', 'Sparse VAR'],
        'Avg_MAE': [
            var_metrics['MAE'].mean(),
            vecm_metrics['MAE'].mean(),
            sparse_metrics['MAE'].mean()
        ],
        'Avg_RMSE': [
            var_metrics['RMSE'].mean(),
            vecm_metrics['RMSE'].mean(),
            sparse_metrics['RMSE'].mean()
        ],
        'Avg_R2': [
            var_metrics['R2'].mean(),
            vecm_metrics['R2'].mean(),
            sparse_metrics['R2'].mean()
        ],
    })
    summary['MAE_Improvement_vs_VAR_%'] = [
        0,
        (summary.loc[0, 'Avg_MAE'] - summary.loc[1, 'Avg_MAE']) / summary.loc[0, 'Avg_MAE'] * 100,
        (summary.loc[0, 'Avg_MAE'] - summary.loc[2, 'Avg_MAE']) / summary.loc[0, 'Avg_MAE'] * 100
    ]
    summary.to_csv(results_dir / "summary.csv", index=False, encoding='utf-8-sig')
    print(f"  [OK] 汇总统计已保存")

    # 保存稀疏VAR系数矩阵
    coef_matrix = sparse_bundle.model.coef_
    np.save(results_dir / "sparse_var_coefficients.npy", coef_matrix)

    # 绘图
    plot_sparsity_heatmap(coef_matrix, figures_dir / "sparse_var_coefficients_heatmap.pdf")

    # 对齐索引
    common_idx = var_pred.index.intersection(vecm_pred.index).intersection(sparse_pred.index)
    plot_three_model_comparison(
        y_true=y_true_var.loc[common_idx],
        y_pred_var=var_pred.loc[common_idx],
        y_pred_vecm=vecm_pred.loc[common_idx],
        y_pred_sparse=sparse_pred.loc[common_idx],
        output_path=figures_dir / "three_model_predictions.pdf"
    )

    plot_metrics_bar_comparison(
        var_metrics, vecm_metrics, sparse_metrics,
        figures_dir / "three_model_metrics.pdf"
    )

    print("\n" + "=" * 80)
    print("实验完成！")
    print("=" * 80)

    print(f"\n模型对比汇总:")
    print(f"  VAR(2)      - 平均MAE: {var_metrics['MAE'].mean():.4f}, 平均R2: {var_metrics['R2'].mean():.4f}")
    print(f"  VECM        - 平均MAE: {vecm_metrics['MAE'].mean():.4f}, 平均R2: {vecm_metrics['R2'].mean():.4f}")
    print(f"  Sparse VAR  - 平均MAE: {sparse_metrics['MAE'].mean():.4f}, 平均R2: {sparse_metrics['R2'].mean():.4f}")

    improvement = (var_metrics['MAE'].mean() - sparse_metrics['MAE'].mean()) / var_metrics['MAE'].mean() * 100
    print(f"\n  稀疏VAR相比VAR(2)改进: {improvement:.2f}%")
    print(f"  稀疏度: {sparse_bundle.sparsity_ratio*100:.2f}% (非零系数: {sparse_bundle.n_nonzero_coefs})")


if __name__ == "__main__":
    main()
