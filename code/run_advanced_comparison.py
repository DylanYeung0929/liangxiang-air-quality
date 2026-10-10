from __future__ import annotations

from pathlib import Path
import warnings

import sys
import warnings

# --- 新增这段代码 ---
# 获取当前脚本的父目录的父目录（即 ver2 目录）
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch

warnings.filterwarnings("ignore", category=RuntimeWarning)

from src.advanced_models import (
    fit_esn_model,
    fit_lstnet_model,
    predict_esn_model,
    predict_lstnet_model,
)
from src.air_quality import (
    build_supervised_dataset,
    chronological_split,
    clean_air_quality,
    load_air_quality_workbook,
    regression_metrics,
)


def mean_metrics(df: pd.DataFrame) -> tuple[float, float, float]:
    return float(df["MAE"].mean()), float(df["RMSE"].mean()), float(df["R2"].mean())


def main() -> None:
    np.random.seed(2026)
    torch.manual_seed(2026)

    root = Path(__file__).resolve().parents[1]
    results_dir = root / "results"
    results_dir.mkdir(exist_ok=True)
    workbook = next(root.glob("*.xlsx"))

    raw = load_air_quality_workbook(workbook)
    clean, _ = clean_air_quality(raw)

    x, y = build_supervised_dataset(clean, include_pollution_lags=True)
    x_train, x_test, y_train, y_test = chronological_split(x, y, test_fraction=0.2)

    esn_model = fit_esn_model(x_train, y_train)
    pred_esn = predict_esn_model(esn_model, x_test, y_test.index)
    m_esn = regression_metrics(y_test, pred_esn, "ESN")

    lstnet_model = fit_lstnet_model(y_train)
    pred_lstnet = predict_lstnet_model(lstnet_model, y_train, y_test.index, y_test_true=y_test)
    m_lstnet = regression_metrics(y_test, pred_lstnet, "LSTNet")

    esn_mae, esn_rmse, esn_r2 = mean_metrics(m_esn)
    lst_mae, lst_rmse, lst_r2 = mean_metrics(m_lstnet)

    rows = [
        ("Hybrid VAR (Baseline)", 0.7309, 0.9141, 0.9940),
        ("ESN (Non-linear)", esn_mae, esn_rmse, esn_r2),
        ("LSTNet (Deep Learning)", lst_mae, lst_rmse, lst_r2),
    ]

    summary_df = pd.DataFrame(rows, columns=["Model", "MAE", "RMSE", "R2"])
    summary_df.to_csv(results_dir / "advanced_model_metrics.csv", index=False, encoding="utf-8-sig", float_format="%.5f")

    long_df = pd.concat([m_esn, m_lstnet], ignore_index=True)
    baseline_long = pd.DataFrame(
        [
            {"Model": "Hybrid VAR (Baseline)", "Pollutant": "Mean", "MAE": 0.7309, "RMSE": 0.9141, "R2": 0.9940}
        ]
    )
    pd.concat([long_df, baseline_long], ignore_index=True).to_csv(
        results_dir / "advanced_model_metrics_long.csv",
        index=False,
        encoding="utf-8-sig",
        float_format="%.5f",
    )

    print("| Model | MAE | RMSE | R2 |")
    print("|---|---:|---:|---:|")
    for name, mae, rmse, r2 in rows:
        print(f"| {name} | {mae:.4f} | {rmse:.4f} | {r2:.4f} |")


if __name__ == "__main__":
    main()
