from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from scipy import stats
from sklearn.metrics import accuracy_score, confusion_matrix, mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.air_quality import (
    ACTIVITY_ADVICE,
    POLLUTANTS,
    build_var_exog_features,
    build_supervised_dataset,
    chronological_split,
    clean_air_quality,
    compute_aqi_frame,
    daypart_recommendations,
    fit_ridge_model,
    fit_var_model,
    load_air_quality_workbook,
    predict_frame,
    predict_var_residuals,
    regression_metrics,
    seasonal_lag_frame,
    seasonal_naive_predictions,
)


DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
FIGURE_DIR = ROOT / "figures"
PAPER_DIR = ROOT / "paper"

COLORS = {
    "PM2.5": "#31572c",
    "PM10": "#4f772d",
    "O3": "#3a86ff",
    "NO2": "#8338ec",
    "SO2": "#ffbe0b",
    "CO": "#fb5607",
    "temperature": "#d62828",
    "humidity": "#0077b6",
}


def ensure_dirs() -> None:
    for directory in [DATA_DIR, RESULTS_DIR, FIGURE_DIR, PAPER_DIR]:
        directory.mkdir(exist_ok=True)


def clear_generated_figures() -> None:
    for pattern in ("fig*.png", "fig*.pdf"):
        for path in FIGURE_DIR.glob(pattern):
            path.unlink()


def setup_style() -> None:
    plt.rcParams.update(
        {
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
        }
    )
    sns.set_theme(style="whitegrid", font="DejaVu Serif")


def savefig(name: str) -> None:
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / f"{name}.png", bbox_inches="tight", pad_inches=0.04)
    plt.savefig(FIGURE_DIR / f"{name}.pdf", bbox_inches="tight", pad_inches=0.04)
    plt.close()


def describe_numeric(frame: pd.DataFrame) -> pd.DataFrame:
    desc = frame[POLLUTANTS + ["temperature", "humidity"]].agg(
        ["mean", "std", "min", "median", "max", "skew"]
    )
    q = frame[POLLUTANTS + ["temperature", "humidity"]].quantile([0.25, 0.75])
    desc.loc["Q1"] = q.loc[0.25]
    desc.loc["Q3"] = q.loc[0.75]
    return desc.T[["min", "Q1", "median", "Q3", "max", "mean", "std", "skew"]]


def monthly_summary(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.groupby(frame.index.month)[POLLUTANTS + ["temperature", "humidity"]].mean().rename_axis("Month")


def lag_correlation_table(frame: pd.DataFrame, max_lag: int = 48) -> pd.DataFrame:
    rows = []
    for pollutant in POLLUTANTS:
        for driver in ["temperature", "humidity"]:
            for lag in range(max_lag + 1):
                corr = frame[pollutant].corr(frame[driver].shift(lag))
                rows.append(
                    {
                        "Pollutant": pollutant,
                        "Driver": driver,
                        "Lag_h": lag,
                        "Correlation": corr,
                        "AbsCorrelation": abs(corr),
                    }
                )
    return pd.DataFrame(rows)


def summarize_lag_effects(lag_table: pd.DataFrame) -> pd.DataFrame:
    idx = lag_table.groupby(["Pollutant", "Driver"])["AbsCorrelation"].idxmax()
    return (
        lag_table.loc[idx, ["Pollutant", "Driver", "Lag_h", "Correlation"]]
        .sort_values(["Pollutant", "Driver"])
        .reset_index(drop=True)
    )


def create_daily_aqi_forecast(prediction: pd.DataFrame) -> pd.DataFrame:
    predicted_aqi = compute_aqi_frame(prediction)
    recs = daypart_recommendations(predicted_aqi)
    return recs


def plot_workflow() -> None:
    import matplotlib.font_manager as fm
    chinese_fonts = [f.name for f in fm.fontManager.ttflist if
                     'Hei' in f.name or 'Song' in f.name or 'Yuan' in f.name or 'Ming' in f.name or 'SimSun' in f.name]
    cn_font = chinese_fonts[0] if chinese_fonts else 'sans-serif'

    fig, ax = plt.subplots(figsize=(12.5, 3.6))
    ax.axis("off")

    steps = [
        ("原始数据", "Raw hourly data", "Excel导入", "Excel import"),
        ("数据清洗", "Data cleaning", "去重与插补", "Dedup & gap fill"),
        ("规律分析", "Pattern analysis", "时序/气象/滞后", "Time/Meteo/Lag"),
        ("联动预测", "Joint prediction", "Ridge与VAR", "Ridge & VAR"),
        ("AQI分级", "AQI grading", "HJ 633-2026", "HJ 633-2026"),
        ("校园建议", "Campus advice", "三个时段", "Three dayparts"),
    ]

    x_positions = np.linspace(0.09, 0.91, len(steps))
    box_width = 0.125

    for i, (cn_title, en_title, cn_sub, en_sub) in enumerate(steps):
        x = x_positions[i]

        box = plt.Rectangle(
            (x - box_width / 2, 0.28),
            box_width,
            0.34,
            transform=ax.transAxes,
            facecolor="#f3f7ee",
            edgecolor="#31572c",
            linewidth=1.4,
            joinstyle="round",
        )
        ax.add_patch(box)

        ax.text(x, 0.55, cn_title, ha="center", va="center",
                transform=ax.transAxes, weight="bold", fontsize=13, fontfamily=cn_font)
        ax.text(x, 0.49, en_title, ha="center", va="center",
                transform=ax.transAxes, fontsize=8)
        ax.text(x, 0.41, cn_sub, ha="center", va="center",
                transform=ax.transAxes, fontsize=10, fontfamily=cn_font)
        ax.text(x, 0.35, en_sub, ha="center", va="center",
                transform=ax.transAxes, fontsize=7.5, color="#555555")

        if i < len(steps) - 1:
            ax.annotate(
                "",
                xy=(x_positions[i + 1] - box_width / 2 - 0.01, 0.45),
                xytext=(x + box_width / 2 + 0.01, 0.45),
                xycoords=ax.transAxes,
                arrowprops=dict(arrowstyle="->", color="#31572c", lw=1.5),
            )

    ax.set_title("Modeling Workflow", weight="bold", fontsize=14, pad=15)
    savefig("fig01_workflow")


def plot_timeseries(frame: pd.DataFrame) -> None:
    fig, axes = plt.subplots(6, 1, figsize=(11, 9), sharex=True)
    for ax, pollutant in zip(axes, POLLUTANTS):
        daily = frame[pollutant].resample("D").mean()
        ax.plot(frame.index, frame[pollutant], color=COLORS[pollutant], alpha=0.22, linewidth=0.6)
        ax.plot(daily.index, daily, color=COLORS[pollutant], linewidth=1.6)
        ax.set_ylabel(pollutant)
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    fig.suptitle("Hourly Pollutant Concentrations and Daily Means", weight="bold")
    savefig("fig02_pollutant_timeseries")


def plot_meteorology(frame: pd.DataFrame) -> None:
    fig, ax1 = plt.subplots(figsize=(11, 4))
    daily = frame[["temperature", "humidity"]].resample("D").mean()
    ax1.plot(daily.index, daily["temperature"], color=COLORS["temperature"], linewidth=2, label="Temperature")
    ax1.set_ylabel("Temperature (C)", color=COLORS["temperature"])
    ax2 = ax1.twinx()
    ax2.plot(daily.index, daily["humidity"], color=COLORS["humidity"], linewidth=2, label="Humidity")
    ax2.set_ylabel("Humidity (%)", color=COLORS["humidity"])
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax1.set_title("Daily Meteorological Conditions", weight="bold")
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc="upper left")
    savefig("fig03_meteorology")


def plot_distributions(frame: pd.DataFrame) -> None:
    long = frame[POLLUTANTS].reset_index().melt(id_vars="datetime", var_name="Pollutant", value_name="Concentration")
    fig, ax = plt.subplots(figsize=(10, 4.5))
    sns.boxenplot(
        data=long,
        x="Pollutant",
        y="Concentration",
        hue="Pollutant",
        palette=[COLORS[p] for p in POLLUTANTS],
        legend=False,
        ax=ax,
    )
    ax.set_title("Pollutant Concentration Distributions", weight="bold")
    ax.set_xlabel("")
    ax.set_ylabel("Concentration")
    savefig("fig04_pollutant_distribution")


def plot_hourly_heatmap(frame: pd.DataFrame) -> None:
    hourly = frame.groupby(frame.index.hour)[POLLUTANTS].mean()
    normalized = (hourly - hourly.min()) / (hourly.max() - hourly.min())
    fig, ax = plt.subplots(figsize=(9, 4.7))
    sns.heatmap(normalized.T, cmap="YlGnBu", ax=ax, cbar_kws={"label": "Normalized Mean"})
    ax.set_title("Mean Diurnal Pattern by Pollutant", weight="bold")
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("")
    savefig("fig05_diurnal_heatmap")


def plot_correlation(frame: pd.DataFrame) -> None:
    corr = frame[POLLUTANTS + ["temperature", "humidity"]].corr()
    fig, ax = plt.subplots(figsize=(7.2, 6))
    sns.heatmap(corr, cmap="RdBu_r", center=0, annot=True, fmt=".2f", ax=ax, square=True)
    ax.set_title("Correlation Matrix", weight="bold")
    savefig("fig06_correlation_matrix")


def plot_lag_correlation(lag_table: pd.DataFrame) -> None:
    best_temp = lag_table[lag_table["Driver"] == "temperature"].pivot(
        index="Pollutant", columns="Lag_h", values="Correlation"
    )
    best_hum = lag_table[lag_table["Driver"] == "humidity"].pivot(
        index="Pollutant", columns="Lag_h", values="Correlation"
    )
    fig, axes = plt.subplots(2, 1, figsize=(11, 5.8), sharex=True)
    sns.heatmap(best_temp, cmap="RdBu_r", center=0, ax=axes[0], cbar_kws={"label": "Correlation"})
    axes[0].set_title("Lagged Correlation with Temperature", weight="bold")
    axes[0].set_ylabel("")
    sns.heatmap(best_hum, cmap="RdBu_r", center=0, ax=axes[1], cbar_kws={"label": "Correlation"})
    axes[1].set_title("Lagged Correlation with Humidity", weight="bold")
    axes[1].set_xlabel("Lag (hours)")
    axes[1].set_ylabel("")
    savefig("fig07_lag_correlation")


def plot_three_dimensional(frame: pd.DataFrame) -> None:
    daily = frame.resample("D").mean()
    sample = frame.sample(min(1000, len(frame)), random_state=2026)
    response_model = RandomForestRegressor(
        n_estimators=220,
        max_depth=12,
        random_state=2026,
        n_jobs=-1,
    )
    response_model.fit(sample[["temperature", "humidity"]], sample["PM2.5"])
    temp_grid, humidity_grid = np.meshgrid(
        np.linspace(sample["temperature"].min(), sample["temperature"].max(), 45),
        np.linspace(sample["humidity"].min(), sample["humidity"].max(), 45),
    )
    grid = pd.DataFrame({"temperature": temp_grid.ravel(), "humidity": humidity_grid.ravel()})
    pm25_surface = response_model.predict(grid).reshape(temp_grid.shape)

    plot_data = pd.concat(
        [
            pd.DataFrame(
                {
                    "Layer": "daily_observation",
                    "Date": daily.index.strftime("%Y-%m-%d"),
                    "Temperature": daily["temperature"].to_numpy(),
                    "Humidity": daily["humidity"].to_numpy(),
                    "PM2.5": daily["PM2.5"].to_numpy(),
                }
            ),
            pd.DataFrame(
                {
                    "Layer": "hourly_sample",
                    "Date": sample.index.strftime("%Y-%m-%d %H:%M:%S"),
                    "Temperature": sample["temperature"].to_numpy(),
                    "Humidity": sample["humidity"].to_numpy(),
                    "PM2.5": sample["PM2.5"].to_numpy(),
                }
            ),
            pd.DataFrame(
                {
                    "Layer": "fitted_surface",
                    "Date": "",
                    "Temperature": temp_grid.ravel(),
                    "Humidity": humidity_grid.ravel(),
                    "PM2.5": pm25_surface.ravel(),
                }
            ),
        ],
        ignore_index=True,
    )
    plot_data.to_csv(RESULTS_DIR / "pm25_response_surface_plot_data.csv", index=False, encoding="utf-8-sig", float_format="%.5f")

    fig = plt.figure(figsize=(8.2, 6))
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(
        temp_grid,
        humidity_grid,
        pm25_surface,
        cmap="viridis",
        alpha=0.88,
        linewidth=0,
    )
    ax.scatter(
        sample["temperature"],
        sample["humidity"],
        sample["PM2.5"],
        s=8,
        alpha=0.22,
        c="#c44e52",
    )
    ax.set_xlabel(r"Temperature ($^\circ$C)")
    ax.set_ylabel("Humidity (%)")
    ax.set_zlabel(r"PM2.5 ($\mu$g/m$^3$)")
    ax.view_init(elev=28, azim=-58)
    ax.set_title("PM2.5 Response Surface")
    fig.colorbar(surf, ax=ax, shrink=0.62, pad=0.08, label="Predicted PM2.5")
    savefig("fig08_3d_pm25_meteorology")


def plot_metrics(metrics: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, metric in zip(axes, ["MAE", "RMSE", "R2"]):
        sns.barplot(data=metrics, x="Pollutant", y=metric, hue="Model", ax=ax)
        ax.set_title(metric, weight="bold")
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=30)
        if metric != "MAE":
            ax.legend_.remove()
    axes[0].legend(loc="best", frameon=True)
    fig.suptitle("Prediction Performance by Pollutant", weight="bold")
    savefig("fig09_model_metrics")


def plot_prediction_lines(y_test: pd.DataFrame, predictions: pd.DataFrame) -> None:
    subset = y_test.index[: min(168, len(y_test))]
    fig, axes = plt.subplots(3, 2, figsize=(12, 8), sharex=True)
    for ax, pollutant in zip(axes.flat, POLLUTANTS):
        ax.plot(subset, y_test.loc[subset, pollutant], color="#202020", linewidth=1.5, label="Observed")
        ax.plot(subset, predictions.loc[subset, pollutant], color=COLORS[pollutant], linewidth=1.4, label="Predicted")
        ax.set_title(pollutant, weight="bold")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    axes.flat[0].legend(loc="best")
    fig.suptitle("Observed vs Predicted Concentrations in Test Period", weight="bold")
    savefig("fig10_prediction_lines")


def plot_prediction_scatter(y_test: pd.DataFrame, predictions: pd.DataFrame) -> None:
    fig, axes = plt.subplots(3, 2, figsize=(10, 10))
    rng = np.random.default_rng(2026)
    diagnostic_rows = []
    for ax, pollutant in zip(axes.flat, POLLUTANTS):
        observed = y_test[pollutant]
        predicted = predictions[pollutant]
        obs_span = max(float(observed.max() - observed.min()), 1e-9)
        pred_span = max(float(predicted.max() - predicted.min()), 1e-9)
        x_jitter = rng.normal(0, obs_span * 0.004, size=len(observed))
        y_jitter = rng.normal(0, pred_span * 0.004, size=len(predicted))
        ax.scatter(
            observed + x_jitter,
            predicted + y_jitter,
            s=14,
            alpha=0.54,
            color=COLORS[pollutant],
            edgecolors="none",
        )
        low = min(y_test[pollutant].min(), predictions[pollutant].min())
        high = max(y_test[pollutant].max(), predictions[pollutant].max())
        ax.plot([low, high], [low, high], color="#202020", linestyle="--", linewidth=1)
        sample_count = int(len(observed))
        observed_unique = int(observed.nunique())
        predicted_unique = int(predicted.round(4).nunique())
        diagnostic_rows.append(
            {
                "Pollutant": pollutant,
                "Sample_Count": sample_count,
                "Observed_Unique": observed_unique,
                "Predicted_Unique": predicted_unique,
            }
        )
        ax.text(
            0.04,
            0.94,
            f"n={sample_count}, unique={observed_unique}",
            transform=ax.transAxes,
            va="top",
            fontsize=8,
            color="#444444",
        )
        ax.set_title(pollutant, weight="bold")
        ax.set_xlabel("Observed")
        ax.set_ylabel("Predicted")
    fig.suptitle("Prediction Scatter Diagnostics", weight="bold")
    pd.DataFrame(diagnostic_rows).to_csv(
        RESULTS_DIR / "prediction_scatter_diagnostics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    savefig("fig11_prediction_scatter")


def plot_var_residual_lag_heatmap(residuals: pd.DataFrame, max_lag: int = 24) -> None:
    rows = []
    for pollutant in POLLUTANTS:
        series = residuals[pollutant]
        for lag in range(1, max_lag + 1):
            corr = series.corr(series.shift(lag))
            rows.append({"Pollutant": pollutant, "Lag_h": lag, "Correlation": corr})
    lag_df = pd.DataFrame(rows)
    lag_df.to_csv(RESULTS_DIR / "var_residual_lag_plot_data.csv", index=False, encoding="utf-8-sig", float_format="%.5f")

    heat = lag_df.pivot(index="Pollutant", columns="Lag_h", values="Correlation")
    fig, ax = plt.subplots(figsize=(10.2, 4.2))
    sns.heatmap(heat, cmap="RdBu_r", center=0, ax=ax, cbar_kws={"label": "Residual Autocorrelation"})
    ax.set_title("VAR Residual Lag Correlation Heatmap", weight="bold")
    ax.set_xlabel("Lag (hours)")
    ax.set_ylabel("")
    savefig("fig12_var_residual_lag_heatmap")


def plot_hourly_error_heatmap(y_test: pd.DataFrame, predictions: pd.DataFrame) -> None:
    error_abs = (y_test - predictions).abs()
    hourly_mae = error_abs.groupby(error_abs.index.hour).mean()
    hourly_long = hourly_mae.reset_index(names="Hour").melt(id_vars="Hour", var_name="Pollutant", value_name="MAE")
    hourly_long.to_csv(RESULTS_DIR / "hourly_error_heatmap_plot_data.csv", index=False, encoding="utf-8-sig", float_format="%.5f")

    heat = hourly_mae.T
    fig, ax = plt.subplots(figsize=(10.5, 4.5))
    sns.heatmap(heat, cmap="YlOrRd", ax=ax, cbar_kws={"label": "Hourly MAE"})
    ax.set_title("Hourly Prediction Error Heatmap", weight="bold")
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("")
    savefig("fig17_hourly_error_heatmap")


def plot_residual_covariance(residuals: pd.DataFrame) -> None:
    cov = residuals[POLLUTANTS].cov()
    cov.to_csv(RESULTS_DIR / "residual_covariance_matrix.csv", encoding="utf-8-sig", float_format="%.5f")

    fig, ax = plt.subplots(figsize=(7.2, 6))
    sns.heatmap(cov, cmap="PuOr", center=0, annot=True, fmt=".3f", square=True, ax=ax)
    ax.set_title("Residual Covariance Matrix Across Pollutants", weight="bold")
    savefig("fig18_residual_covariance")


def plot_full_period_aqi(
    observed_aqi: pd.DataFrame,
    predicted_aqi: pd.DataFrame,
    test_start: pd.Timestamp,
) -> None:
    predicted = predicted_aqi.copy()
    predicted["Segment"] = np.where(predicted.index < test_start, "Retrospective fitted AQI", "Test forecast AQI")
    predicted.to_csv(
        DATA_DIR / "predicted_aqi_full_period.csv",
        encoding="utf-8-sig",
        float_format="%.5f",
    )

    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(observed_aqi.index, observed_aqi["AQI"], color="#202020", linewidth=1.1, alpha=0.72, label="Observed AQI")
    fitted = predicted[predicted["Segment"] == "Retrospective fitted AQI"]
    forecast = predicted[predicted["Segment"] == "Test forecast AQI"]
    if not fitted.empty:
        ax.plot(fitted.index, fitted["AQI"], color="#2a9d8f", linewidth=1.15, label="Retrospective fitted AQI")
    if not forecast.empty:
        ax.axvspan(forecast.index.min(), forecast.index.max(), color="#f4a261", alpha=0.18, label="Test period")
        ax.plot(forecast.index, forecast["AQI"], color="#e76f51", linewidth=1.45, label="Test forecast AQI")
    for y, label in [(50, "优"), (100, "良"), (150, "轻度"), (200, "中度"), (300, "重度")]:
        ax.axhline(y, color="#777777", linestyle="--", linewidth=0.75, alpha=0.65)
        ax.text(observed_aqi.index.min(), y + 2, label, fontsize=8, color="#555555")
    ax.set_ylabel("AQI")
    ax.set_title("Full-period AQI Fitting and Test Forecast", weight="bold")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax.legend(loc="upper right", ncol=2, frameon=True)
    savefig("fig14_aqi_full_period_forecast")


def plot_aqi(frame: pd.DataFrame, predicted_aqi: pd.DataFrame | None = None) -> None:
    aqi = compute_aqi_frame(frame)
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(aqi.index, aqi["AQI"], color="#31572c", linewidth=1.5, label="Observed AQI")
    if predicted_aqi is not None:
        ax.plot(predicted_aqi.index, predicted_aqi["AQI"], color="#fb5607", linewidth=1.2, label="Predicted AQI")
    for y, label in [(50, "优"), (100, "良"), (150, "轻度"), (200, "中度"), (300, "重度")]:
        ax.axhline(y, color="#777777", linestyle="--", linewidth=0.8, alpha=0.7)
        ax.text(aqi.index.min(), y + 2, label, fontsize=8, color="#555555")
    ax.set_ylabel("AQI")
    ax.set_title("AQI Evolution and Grading Bands", weight="bold")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax.legend(loc="upper right")
    savefig("fig13_aqi_timeseries")


def plot_primary_pollutant(aqi: pd.DataFrame) -> None:
    counts = aqi["Primary_Pollutant"].value_counts().reindex(POLLUTANTS, fill_value=0)
    counts = counts[counts > 0].sort_values()
    plot_data = (
        counts.rename_axis("Primary_Pollutant")
        .reset_index(name="Hours")
        .assign(Share=lambda data: data["Hours"] / data["Hours"].sum())
    )
    plot_data.to_csv(
        RESULTS_DIR / "primary_pollutant_plot_data.csv",
        index=False,
        encoding="utf-8-sig",
        float_format="%.5f",
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


def plot_daypart_advice(recs: pd.DataFrame) -> None:
    order = ["Morning", "Afternoon", "Evening"]
    latest = recs.tail(42).copy()
    latest.to_csv(
        RESULTS_DIR / "daypart_activity_plot_data.csv",
        index=False,
        encoding="utf-8-sig",
        float_format="%.5f",
    )
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


def plot_residuals(y_test: pd.DataFrame, predictions: pd.DataFrame) -> None:
    residuals = y_test - predictions
    long = residuals.reset_index().melt(id_vars="datetime", var_name="Pollutant", value_name="Residual")
    fig, ax = plt.subplots(figsize=(10, 4.8))
    sns.violinplot(
        data=long,
        x="Pollutant",
        y="Residual",
        hue="Pollutant",
        palette=[COLORS[p] for p in POLLUTANTS],
        legend=False,
        inner="quartile",
        ax=ax,
    )
    ax.axhline(0, color="#202020", linestyle="--", linewidth=1)
    ax.set_title("Prediction Residual Distribution", weight="bold")
    ax.set_xlabel("")
    savefig("fig13_residual_distribution")


def plot_advanced_model_comparison() -> None:
    metrics_path = RESULTS_DIR / "advanced_model_metrics.csv"
    if not metrics_path.exists():
        return

    data = pd.read_csv(metrics_path)
    data.to_csv(
        RESULTS_DIR / "advanced_model_comparison_plot_data.csv",
        index=False,
        encoding="utf-8-sig",
        float_format="%.5f",
    )

    long = data.melt(id_vars="Model", value_vars=["MAE", "RMSE", "R2"], var_name="Metric", value_name="Value")
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.2))
    for ax, metric in zip(axes, ["MAE", "RMSE", "R2"]):
        subset = long[long["Metric"] == metric]
        sns.barplot(data=subset, x="Model", y="Value", hue="Model", legend=False, ax=ax)
        ax.set_title(metric, weight="bold")
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=25)
    fig.suptitle("Advanced Model Comparison with Hybrid VAR Baseline", weight="bold")
    savefig("fig19_advanced_model_comparison")


def format_table_csv(path: Path, frame: pd.DataFrame) -> None:
    frame.to_csv(path, index=True, encoding="utf-8-sig", float_format="%.4f")


def build_results() -> dict:
    ensure_dirs()
    clear_generated_figures()
    setup_style()

    workbook = next(ROOT.glob("*.xlsx"))
    raw = load_air_quality_workbook(workbook)
    clean, cleaning_report = clean_air_quality(raw)
    clean.to_csv(DATA_DIR / "clean_air_quality_hourly.csv", encoding="utf-8-sig")

    aqi = compute_aqi_frame(clean)
    aqi.to_csv(DATA_DIR / "aqi_observed_hourly.csv", encoding="utf-8-sig")

    desc = describe_numeric(clean)
    monthly = monthly_summary(clean)
    lag_table = lag_correlation_table(clean)
    lag_summary = summarize_lag_effects(lag_table)
    format_table_csv(RESULTS_DIR / "descriptive_statistics.csv", desc)
    format_table_csv(RESULTS_DIR / "monthly_mean_summary.csv", monthly)
    lag_table.to_csv(RESULTS_DIR / "lag_correlation_full.csv", index=False, encoding="utf-8-sig", float_format="%.5f")
    lag_summary.to_csv(RESULTS_DIR / "lag_correlation_summary.csv", index=False, encoding="utf-8-sig", float_format="%.5f")
    pd.DataFrame([cleaning_report]).to_csv(RESULTS_DIR / "cleaning_report.csv", index=False, encoding="utf-8-sig")

    x_exog, y_exog = build_supervised_dataset(clean, include_pollution_lags=False)
    x_joint, y_joint = build_supervised_dataset(clean, include_pollution_lags=True)
    x_train, x_test, y_train, y_test = chronological_split(x_joint, y_joint)

    var_exog_full = build_var_exog_features(clean).reindex(x_joint.index)
    exog_train_joint = var_exog_full.loc[x_train.index]
    exog_test_joint = var_exog_full.loc[x_test.index]
    train_mask_joint = exog_train_joint.notna().all(axis=1)
    test_mask_joint = exog_test_joint.notna().all(axis=1)

    y_train_joint = y_train.loc[train_mask_joint]
    y_test_joint = y_test.loc[test_mask_joint]
    exog_train_joint = exog_train_joint.loc[train_mask_joint]
    exog_test_joint = exog_test_joint.loc[test_mask_joint]

    baseline = seasonal_naive_predictions(clean[POLLUTANTS], y_test_joint.index)
    ridge_pred = predict_frame(fit_ridge_model(x_train, y_train), x_test).loc[y_test_joint.index]

    var_joint_model = fit_var_model(y_train_joint, exog_train=exog_train_joint)
    var_joint_pred = predict_var_residuals(
        var_joint_model,
        residual_train=y_train_joint,
        target_index=y_test_joint.index,
        exog_future=exog_test_joint,
    ).clip(lower=0)

    train_baseline = seasonal_lag_frame(clean[POLLUTANTS], y_train_joint.index)
    test_baseline = seasonal_lag_frame(clean[POLLUTANTS], y_test_joint.index)
    residual_train = y_train_joint - train_baseline
    hybrid_model = fit_var_model(residual_train, exog_train=exog_train_joint)
    residual_pred = predict_var_residuals(
        hybrid_model,
        residual_train=residual_train,
        target_index=y_test_joint.index,
        exog_future=exog_test_joint,
    )
    hybrid_pred = (test_baseline + residual_pred).clip(lower=0)

    x_train_exog, _, y_train_exog, y_test_exog = chronological_split(x_exog, y_exog)
    exog_train_only = build_var_exog_features(clean).reindex(y_train_exog.index)
    exog_test_only = build_var_exog_features(clean).reindex(y_test_exog.index)
    train_mask_only = exog_train_only.notna().all(axis=1)
    test_mask_only = exog_test_only.notna().all(axis=1)

    y_train_exog_var = y_train_exog.loc[train_mask_only]
    y_test_exog_var = y_test_exog.loc[test_mask_only]
    exog_train_only = exog_train_only.loc[train_mask_only]
    exog_test_only = exog_test_only.loc[test_mask_only]

    var_exog_model = fit_var_model(y_train_exog_var, exog_train=exog_train_only)
    exog_pred = predict_var_residuals(
        var_exog_model,
        residual_train=y_train_exog_var,
        target_index=y_test_exog_var.index,
        exog_future=exog_test_only,
    ).clip(lower=0)

    metrics = pd.concat(
        [
            regression_metrics(y_test_joint, baseline, "Seasonal naive"),
            regression_metrics(y_test_joint, ridge_pred, "Ridge lag model"),
            regression_metrics(y_test_joint, var_joint_pred, "VAR joint model"),
            regression_metrics(y_test_joint, hybrid_pred, "Hybrid periodic VAR"),
            regression_metrics(y_test_exog_var, exog_pred, "VAR time-meteorology"),
        ],
        ignore_index=True,
    )
    metrics.to_csv(RESULTS_DIR / "model_metrics.csv", index=False, encoding="utf-8-sig", float_format="%.5f")
    hybrid_pred.to_csv(DATA_DIR / "predicted_pollutants_test.csv", encoding="utf-8-sig", float_format="%.5f")
    y_test_joint.to_csv(DATA_DIR / "observed_pollutants_test.csv", encoding="utf-8-sig", float_format="%.5f")

    full_hybrid_pred = pd.concat([y_train_joint, hybrid_pred]).sort_index()
    full_hybrid_pred = full_hybrid_pred[~full_hybrid_pred.index.duplicated(keep="last")]

    observed_test_aqi = compute_aqi_frame(y_test_joint)
    predicted_test_aqi = compute_aqi_frame(hybrid_pred)
    predicted_full_aqi = compute_aqi_frame(full_hybrid_pred)
    level_order = ["Excellent", "Good", "Lightly polluted", "Moderately polluted", "Heavily polluted", "Severely polluted"]
    common_idx = observed_test_aqi.index.intersection(predicted_test_aqi.index)
    cm = pd.DataFrame(
        confusion_matrix(
            observed_test_aqi.loc[common_idx, "AQI_Level"],
            predicted_test_aqi.loc[common_idx, "AQI_Level"],
            labels=level_order,
        ),
        index=[f"Observed {x}" for x in level_order],
        columns=[f"Predicted {x}" for x in level_order],
    )
    cm.to_csv(RESULTS_DIR / "aqi_confusion_matrix.csv", encoding="utf-8-sig")
    aqi_accuracy = accuracy_score(
        observed_test_aqi.loc[common_idx, "AQI_Level"],
        predicted_test_aqi.loc[common_idx, "AQI_Level"],
    )
    aqi_eval = pd.DataFrame(
        [
            {
                "MAE": mean_absolute_error(
                    observed_test_aqi.loc[common_idx, "AQI"], predicted_test_aqi.loc[common_idx, "AQI"]
                ),
                "RMSE": mean_squared_error(
                    observed_test_aqi.loc[common_idx, "AQI"], predicted_test_aqi.loc[common_idx, "AQI"]
                )
                ** 0.5,
                "Level_Accuracy": aqi_accuracy,
                "Test_Hours": len(common_idx),
            }
        ]
    )
    aqi_eval.to_csv(RESULTS_DIR / "aqi_prediction_metrics.csv", index=False, encoding="utf-8-sig", float_format="%.5f")

    future_recs = create_daily_aqi_forecast(full_hybrid_pred)
    future_recs.to_csv(RESULTS_DIR / "daypart_activity_recommendations.csv", index=False, encoding="utf-8-sig", float_format="%.2f")

    plot_workflow()
    plot_timeseries(clean)
    plot_meteorology(clean)
    plot_distributions(clean)
    plot_hourly_heatmap(clean)
    plot_correlation(clean)
    plot_lag_correlation(lag_table)
    plot_three_dimensional(clean)
    plot_metrics(metrics)
    plot_prediction_lines(y_test_joint, hybrid_pred)
    plot_prediction_scatter(y_test_joint, hybrid_pred)
    plot_var_residual_lag_heatmap(residual_train)
    plot_residuals(y_test_joint, hybrid_pred)
    plot_hourly_error_heatmap(y_test_joint, hybrid_pred)
    plot_residual_covariance(y_test_joint - hybrid_pred)
    plot_full_period_aqi(aqi, predicted_full_aqi, y_test_joint.index.min())
    plot_primary_pollutant(aqi)
    plot_daypart_advice(future_recs)
    plot_advanced_model_comparison()

    summary = {
        "cleaning_report": cleaning_report,
        "aqi_accuracy": float(aqi_accuracy),
        "best_metrics": metrics[metrics["Model"] == "Hybrid periodic VAR"].round(4).to_dict(orient="records"),
        "best_lag_effects": lag_summary.round(4).to_dict(orient="records"),
        "aqi_level_counts": aqi["AQI_Level"].value_counts().to_dict(),
        "primary_pollutant_counts": aqi["Primary_Pollutant"].value_counts().to_dict(),
        "figure_count_png": len(list(FIGURE_DIR.glob("*.png"))),
        "activity_advice": ACTIVITY_ADVICE,
        "advanced_comparison_available": bool((RESULTS_DIR / "advanced_model_metrics.csv").exists()),
    }
    (RESULTS_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


if __name__ == "__main__":
    result = build_results()
    print(json.dumps(result, ensure_ascii=False, indent=2))
