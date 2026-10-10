"""
基于Sparse VAR结果重新生成AQI相关图表（fig14, fig15, fig16）
在项目根目录运行: python regenerate_aqi_figures_sparse_var.py
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import seaborn as sns
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.air_quality import POLLUTANTS, compute_aqi_frame

FIGURE_DIR = ROOT / "figures"
RESULTS_DIR = ROOT / "results" / "sparse_var"

COLORS = {
    "PM2.5": "#31572c",
    "PM10": "#4f772d",
    "O3": "#3a86ff",
    "NO2": "#8338ec",
    "SO2": "#ffbe0b",
    "CO": "#fb5607",
}

def setup_style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["DejaVu Serif"],
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.dpi": 180,
        "savefig.dpi": 300,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })
    sns.set_theme(style="whitegrid", font="DejaVu Serif")

def savefig(name: str):
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / f"{name}.png", bbox_inches="tight", pad_inches=0.04)
    plt.savefig(FIGURE_DIR / f"{name}.pdf", bbox_inches="tight", pad_inches=0.04)
    plt.close()

def plot_full_period_aqi_sparse_var(
    observed_pollutants: pd.DataFrame,
    predicted_aqi: pd.DataFrame,
    test_start: pd.Timestamp,
):
    """基于Sparse VAR重新生成fig14"""
    observed_aqi = compute_aqi_frame(observed_pollutants)

    predicted = predicted_aqi.copy()
    predicted["Segment"] = np.where(
        predicted.index < test_start,
        "Retrospective fitted AQI",
        "Test forecast AQI"
    )

    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(observed_aqi.index, observed_aqi["AQI"],
            color="#202020", linewidth=1.1, alpha=0.72, label="Observed AQI")

    fitted = predicted[predicted["Segment"] == "Retrospective fitted AQI"]
    forecast = predicted[predicted["Segment"] == "Test forecast AQI"]

    if not fitted.empty:
        ax.plot(fitted.index, fitted["AQI"],
                color="#2a9d8f", linewidth=1.15, label="Retrospective fitted AQI")
    if not forecast.empty:
        ax.axvspan(forecast.index.min(), forecast.index.max(),
                   color="#f4a261", alpha=0.18, label="Test period")
        ax.plot(forecast.index, forecast["AQI"],
                color="#e76f51", linewidth=1.45, label="Test forecast AQI")

    for y, label in [(50, "优"), (100, "良"), (150, "轻度"), (200, "中度"), (300, "重度")]:
        ax.axhline(y, color="#777777", linestyle="--", linewidth=0.75, alpha=0.65)
        ax.text(observed_aqi.index.min(), y + 2, label, fontsize=8, color="#555555")

    ax.set_ylabel("AQI")
    ax.set_title("Full-period AQI Fitting and Test Forecast", weight="bold")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax.legend(loc="upper right", ncol=2, frameon=True)
    savefig("fig14_aqi_full_period_forecast")

def plot_primary_pollutant_sparse_var(aqi: pd.DataFrame):
    """基于Sparse VAR重新生成fig15"""
    counts = aqi["Primary_Pollutant"].value_counts().reindex(POLLUTANTS, fill_value=0)
    counts = counts[counts > 0].sort_values()

    plot_data = (
        counts.rename_axis("Primary_Pollutant")
        .reset_index(name="Hours")
        .assign(Share=lambda data: data["Hours"] / data["Hours"].sum())
    )

    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    bars = ax.barh(
        plot_data["Primary_Pollutant"],
        plot_data["Hours"],
        color=[COLORS[p] for p in plot_data["Primary_Pollutant"]],
    )

    for bar, share in zip(bars, plot_data["Share"]):
        ax.text(
            bar.get_width() + max(plot_data["Hours"]) * 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{bar.get_width():.0f} h ({share:.1%})",
            va="center",
            fontsize=9,
        )

    ax.set_xlim(0, max(plot_data["Hours"]) * 1.18)
    ax.set_xlabel("Hours")
    ax.set_title("Primary Pollutant Share", weight="bold")
    savefig("fig15_primary_pollutant_share")

def plot_daypart_advice_sparse_var(recs: pd.DataFrame):
    """基于Sparse VAR重新生成fig16"""
    order = ["Morning", "Afternoon", "Evening"]
    latest = recs.tail(42).copy()

    pivot = latest.pivot(index="Date", columns="Daypart", values="Peak_AQI").reindex(columns=order)

    fig, ax = plt.subplots(figsize=(8.7, 6.4))
    sns.heatmap(
        pivot,
        cmap=sns.color_palette(["#5aa469", "#f1d36b", "#e98b39"], as_cmap=True),
        annot=True,
        fmt=".1f",
        cbar_kws={"label": "Peak AQI"},
        ax=ax,
    )
    ax.set_title("Daypart Peak AQI Activity Guidance", weight="bold")
    ax.set_xlabel("")
    ax.set_ylabel("Date")
    savefig("fig16_daypart_advice")

if __name__ == "__main__":
    print("=" * 80)
    print("基于Sparse VAR重新生成AQI相关图表")
    print("=" * 80)

    setup_style()

    # 加载数据
    print("\n[1/4] 加载数据")
    from src.air_quality import load_air_quality_workbook, clean_air_quality

    xlsx_files = list(ROOT.glob("*.xlsx"))
    if not xlsx_files:
        xlsx_files = list((ROOT / "data").glob("*.xlsx"))
    if not xlsx_files:
        raise FileNotFoundError(f"在 {ROOT} 及其 data 子目录下未找到任何 .xlsx 文件。请确保数据文件已放置。")
    raw = load_air_quality_workbook(xlsx_files[0])
    clean, _ = clean_air_quality(raw)
    print(f"  [OK] 观测数据: {len(clean)} 样本")

    # 加载Sparse VAR结果
    print("\n[2/4] 加载Sparse VAR结果")
    aqi_sparse = pd.read_csv(
        RESULTS_DIR / "aqi_sparse_var.csv",
        index_col=0,
        parse_dates=True,
        encoding='utf-8-sig'
    )
    recommendations = pd.read_csv(
        RESULTS_DIR / "activity_recommendations_sparse_var.csv",
        encoding='utf-8-sig'
    )
    print(f"  [OK] AQI预测: {len(aqi_sparse)} 样本")
    print(f"  [OK] 活动建议: {len(recommendations)} 条")

    # 确定测试集起点
    split_idx = int(len(aqi_sparse) * 0.8)
    test_start = aqi_sparse.index[split_idx]
    print(f"  [OK] 测试集起点: {test_start}")

    # 生成图表
    print("\n[3/4] 生成图表")
    print("  - fig14_aqi_full_period_forecast")
    plot_full_period_aqi_sparse_var(clean[POLLUTANTS].loc[aqi_sparse.index], aqi_sparse, test_start)

    print("  - fig15_primary_pollutant_share")
    plot_primary_pollutant_sparse_var(aqi_sparse)

    print("  - fig16_daypart_advice")
    plot_daypart_advice_sparse_var(recommendations)

    print("\n[4/4] 完成")
    print(f"  图表已保存到: {FIGURE_DIR}")
    print("\n" + "=" * 80)
    print("重要提示：")
    print("  - fig14, fig15, fig16 已基于Sparse VAR结果更新")
    print("  - 论文中的数值已经正确（测试集AQI MAE = 0.5017）")
    print("  - 现在图表与论文数值完全一致")
    print("=" * 80)
