"""
协整检验脚本
运行Johansen协整检验，确定VECM的协整秩
"""
from pathlib import Path
import sys
import os

# 设置控制台编码为UTF-8
if sys.platform == 'win32':
    os.system('chcp 65001 > nul')

import numpy as np
import pandas as pd
from statsmodels.tsa.vector_ar.vecm import coint_johansen, select_coint_rank
from statsmodels.tsa.stattools import adfuller

# 添加项目根目录到路径
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

from src.air_quality import (
    load_air_quality_workbook,
    clean_air_quality,
    POLLUTANTS,
    METEOROLOGY,
)


def adf_test(series: pd.Series, name: str) -> dict:
    """
    增强迪基-富勒检验（ADF Test）
    检验序列是否平稳
    """
    result = adfuller(series.dropna(), autolag='AIC')
    return {
        'Variable': name,
        'ADF_Statistic': result[0],
        'p_value': result[1],
        'Used_Lag': result[2],
        'Observations': result[3],
        'Critical_1%': result[4]['1%'],
        'Critical_5%': result[4]['5%'],
        'Critical_10%': result[4]['10%'],
        'Is_Stationary': result[1] < 0.05,  # p < 0.05 拒绝原假设，序列平稳
    }


def johansen_test(data: pd.DataFrame, det_order: int = 1, k_ar_diff: int = 1) -> dict:
    """
    Johansen协整检验

    Args:
        data: 包含多个变量的DataFrame
        det_order: 0=无常数, 1=协整关系中有常数
        k_ar_diff: 差分后的滞后阶数

    Returns:
        检验结果字典
    """
    result = coint_johansen(data, det_order=det_order, k_ar_diff=k_ar_diff)

    n_vars = data.shape[1]
    trace_stats = result.lr1  # Trace统计量
    max_eig_stats = result.lr2  # 最大特征值统计量

    # 临界值（90%, 95%, 99%）
    trace_crit = result.cvt
    max_eig_crit = result.cvm

    # 确定协整秩（使用5%显著性水平）
    trace_rank = np.sum(trace_stats > trace_crit[:, 1])  # 第1列是95%临界值
    max_eig_rank = np.sum(max_eig_stats > max_eig_crit[:, 1])

    return {
        'n_variables': n_vars,
        'det_order': det_order,
        'k_ar_diff': k_ar_diff,
        'trace_statistics': trace_stats,
        'trace_critical_90': trace_crit[:, 0],
        'trace_critical_95': trace_crit[:, 1],
        'trace_critical_99': trace_crit[:, 2],
        'max_eig_statistics': max_eig_stats,
        'max_eig_critical_90': max_eig_crit[:, 0],
        'max_eig_critical_95': max_eig_crit[:, 1],
        'max_eig_critical_99': max_eig_crit[:, 2],
        'coint_rank_trace': trace_rank,
        'coint_rank_max_eig': max_eig_rank,
        'eigenvalues': result.eig,
    }


def main():
    print("=" * 80)
    print("协整检验：确定VECM模型的协整秩")
    print("=" * 80)

    # 1. 加载数据
    data_path = project_root / "4-2026年校赛第二轮A题数据.xlsx"
    print(f"\n[1/4] 加载数据: {data_path.name}")
    raw = load_air_quality_workbook(data_path)
    clean, report = clean_air_quality(raw)
    print(f"  [OK] 清洗后样本数: {len(clean)}")

    # 2. 计算残差序列（去除24小时周期基线）
    print("\n[2/4] 计算残差序列（Y_t - Y_{t-24}）")
    residuals = clean[POLLUTANTS].copy()
    for col in POLLUTANTS:
        residuals[col] = clean[col] - clean[col].shift(24)
    residuals = residuals.iloc[24:].copy()  # 去除前24小时
    print(f"  [OK] 残差序列长度: {len(residuals)}")

    # 3. ADF平稳性检验
    print("\n[3/4] ADF平稳性检验（检验残差序列是否平稳）")
    adf_results = []
    for pollutant in POLLUTANTS:
        adf_res = adf_test(residuals[pollutant], pollutant)
        adf_results.append(adf_res)
        status = "[OK] 平稳" if adf_res['Is_Stationary'] else "[X] 非平稳"
        print(f"  {pollutant:8s}: ADF={adf_res['ADF_Statistic']:8.4f}, "
              f"p={adf_res['p_value']:.4f} {status}")

    results_dir = project_root / "results" / "vecm"
    results_dir.mkdir(parents=True, exist_ok=True)

    adf_df = pd.DataFrame(adf_results)
    adf_output = results_dir / "adf_test.csv"
    adf_df.to_csv(adf_output, index=False, encoding='utf-8-sig')
    print(f"\n  → ADF检验结果已保存: {adf_output.relative_to(project_root)}")

    # 4. Johansen协整检验
    print("\n[4/4] Johansen协整检验（检验是否存在长期均衡关系）")

    # 测试不同的k_ar_diff
    johansen_results = []
    for k_ar_diff in [0, 1, 2]:
        print(f"\n  k_ar_diff = {k_ar_diff}:")
        joh_res = johansen_test(residuals[POLLUTANTS], det_order=1, k_ar_diff=k_ar_diff)

        print(f"    Trace统计量检验:")
        for r in range(len(joh_res['trace_statistics'])):
            stat = joh_res['trace_statistics'][r]
            crit_95 = joh_res['trace_critical_95'][r]
            reject = "[OK]" if stat > crit_95 else "[X]"
            print(f"      r ≤ {r}: {stat:8.4f} vs {crit_95:8.4f} (95%) {reject}")

        print(f"    → 协整秩（Trace）: {joh_res['coint_rank_trace']}")
        print(f"    → 协整秩（Max Eig）: {joh_res['coint_rank_max_eig']}")

        # 保存详细结果
        detail_df = pd.DataFrame({
            'Rank_r': range(len(joh_res['trace_statistics'])),
            'Trace_Statistic': joh_res['trace_statistics'],
            'Trace_Critical_95': joh_res['trace_critical_95'],
            'Trace_Reject_H0': joh_res['trace_statistics'] > joh_res['trace_critical_95'],
            'MaxEig_Statistic': joh_res['max_eig_statistics'],
            'MaxEig_Critical_95': joh_res['max_eig_critical_95'],
            'MaxEig_Reject_H0': joh_res['max_eig_statistics'] > joh_res['max_eig_critical_95'],
        })
        detail_output = results_dir / f"johansen_k{k_ar_diff}.csv"
        detail_df.to_csv(detail_output, index=False, encoding='utf-8-sig')

        johansen_results.append({
            'k_ar_diff': k_ar_diff,
            'coint_rank_trace': joh_res['coint_rank_trace'],
            'coint_rank_max_eig': joh_res['coint_rank_max_eig'],
            'eigenvalues': joh_res['eigenvalues'][:3].tolist(),  # 前3个特征值
        })

    # 保存汇总结果
    summary_df = pd.DataFrame(johansen_results)
    summary_output = results_dir / "johansen_summary.csv"
    summary_df.to_csv(summary_output, index=False, encoding='utf-8-sig')
    print(f"\n  → Johansen检验汇总已保存: {summary_output.relative_to(project_root)}")

    # 5. 结论和建议
    print("\n" + "=" * 80)
    print("检验结论与建议")
    print("=" * 80)

    # 统计平稳的变量数
    n_stationary = sum(adf_df['Is_Stationary'])
    print(f"\n1. ADF检验: {n_stationary}/{len(POLLUTANTS)} 个残差序列平稳")

    if n_stationary >= len(POLLUTANTS) * 0.8:
        print("   → 大部分残差序列已平稳，适合VAR或VECM建模")
    else:
        print("   [!] 部分残差序列非平稳，需谨慎选择模型")

    # 推荐协整秩
    recommended_k = 1
    recommended_rank = johansen_results[recommended_k]['coint_rank_trace']
    print(f"\n2. Johansen检验: 推荐使用 k_ar_diff={recommended_k}")
    print(f"   → 协整秩（Trace方法）: {recommended_rank}")
    print(f"   → 协整秩（Max Eig方法）: {johansen_results[recommended_k]['coint_rank_max_eig']}")

    if recommended_rank > 0:
        print(f"\n3. 建议使用VECM模型:")
        print(f"   - k_ar_diff = {recommended_k}")
        print(f"   - coint_rank = {recommended_rank}")
        print(f"   - deterministic = 'ci' (协整关系中包含常数项)")
    else:
        print(f"\n3. [!] 未检测到显著协整关系，建议:")
        print(f"   - 方案A: 使用VAR(1)模型（降低滞后阶数）")
        print(f"   - 方案B: 使用稀疏VAR（Lasso正则化）")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
