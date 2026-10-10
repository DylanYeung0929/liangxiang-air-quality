from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


POLLUTANTS = ["PM2.5", "PM10", "O3", "NO2", "SO2", "CO"]
METEOROLOGY = ["temperature", "humidity"]
ALL_COLUMNS = ["datetime", *POLLUTANTS, *METEOROLOGY]

IAQI_POINTS = np.array([0, 50, 100, 150, 200, 300, 400, 500], dtype=float)

# HJ 633-2026 breakpoint limits. For realtime campus guidance, particulates
# use 24-hour moving average concentration, and gaseous pollutants use hourly
# concentration. Units follow the source data.
AQI_BREAKPOINTS = {
    "PM2.5": np.array([0, 35, 60, 115, 150, 250, 350, 500], dtype=float),
    "PM10": np.array([0, 50, 120, 250, 350, 420, 500, 600], dtype=float),
    "O3": np.array([0, 160, 200, 300, 400, 800, 1000, 1200], dtype=float),
    "NO2": np.array([0, 100, 200, 700, 1200, 2340, 3090, 3840], dtype=float),
    "SO2": np.array([0, 150, 500, 650, 800, 1600, 2100, 2620], dtype=float),
    "CO": np.array([0, 5, 10, 35, 60, 90, 120, 150], dtype=float),
}

CATEGORY_BOUNDS = [
    (0, 50, "Excellent"),
    (50, 100, "Good"),
    (100, 150, "Lightly polluted"),
    (150, 200, "Moderately polluted"),
    (200, 300, "Heavily polluted"),
    (300, 500, "Severely polluted"),
 ]


ACTIVITY_ADVICE = {
    "Excellent": "适宜户外活动；建议正常通风与锻炼。",
    "Good": "适宜户外活动；敏感人群可以减少极长时间运动。",
    "Lightly polluted": "减少长时间高强度户外活动，敏感人群建议室内。",
    "Moderately polluted": "室内锻炼，缩短户外停留时间。",
    "Heavily polluted": "避免户外锻炼，活动尽量安排在室内。",
    "Severely polluted": "避免一切户外活动，室内采取防护措施。",
}


@dataclass(frozen=True)
class ModelBundle:
    name: str
    model: object
    feature_names: list[str]
    train_index: pd.DatetimeIndex
    test_index: pd.DatetimeIndex
    predictions: pd.DataFrame
    metrics: pd.DataFrame


def load_air_quality_workbook(path: str | Path) -> pd.DataFrame:
    raw = pd.read_excel(path, header=None)
    header_candidates = raw.index[raw.iloc[:, 0].astype(str).str.strip().str.lower().eq("datetime")]
    if len(header_candidates) == 0:
        raise ValueError("The workbook does not contain a datetime header row.")

    header_row = int(header_candidates[0])
    data = raw.iloc[header_row + 1 :].copy()
    data.columns = [str(v).strip() for v in raw.iloc[header_row].tolist()]
    return data.loc[:, ALL_COLUMNS]


def clean_air_quality(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int | str]]:
    frame = raw.copy()
    frame["datetime"] = pd.to_datetime(frame["datetime"], errors="coerce")
    invalid_datetime_rows = int(frame["datetime"].isna().sum())
    frame = frame[frame["datetime"].notna()].copy()

    for col in POLLUTANTS + METEOROLOGY:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")

    duplicate_timestamps = int(frame.duplicated("datetime").sum())
    grouped = (
        frame.groupby("datetime", as_index=True)[POLLUTANTS + METEOROLOGY]
        .mean(numeric_only=True)
        .sort_index()
    )

    full_index = pd.date_range(grouped.index.min(), grouped.index.max(), freq="h")
    inserted_missing_hours = int((~full_index.isin(grouped.index)).sum())
    clean = grouped.reindex(full_index)
    clean.index.name = "datetime"
    clean = clean.interpolate(method="time").ffill().bfill()

    report = {
        "start": str(clean.index.min()),
        "end": str(clean.index.max()),
        "raw_rows_after_header": int(len(raw)),
        "invalid_datetime_rows": invalid_datetime_rows,
        "duplicate_timestamps": duplicate_timestamps,
        "inserted_missing_hours": inserted_missing_hours,
        "clean_rows": int(len(clean)),
        "remaining_missing_values": int(clean.isna().sum().sum()),
    }
    return clean, report


def calculate_iaqi(pollutant: str, concentration: float) -> float:
    if pollutant not in AQI_BREAKPOINTS:
        raise KeyError(f"Unsupported pollutant: {pollutant}")
    if pd.isna(concentration):
        return np.nan

    breakpoints = AQI_BREAKPOINTS[pollutant]
    value = float(max(concentration, 0))
    if value >= breakpoints[-1]:
        return 500.0
    return float(np.interp(value, breakpoints, IAQI_POINTS))


def aqi_category(aqi: float) -> str:
    value = float(aqi)
    for lower, upper, label in CATEGORY_BOUNDS:
        if lower <= value <= upper:
            return label
    return "Severely polluted"


def effective_aqi_concentrations(frame: pd.DataFrame) -> pd.DataFrame:
    effective = frame[POLLUTANTS].copy()
    for particle in ["PM2.5", "PM10"]:
        effective[particle] = frame[particle].rolling(24, min_periods=1).mean()
    return effective


def compute_aqi_frame(frame: pd.DataFrame) -> pd.DataFrame:
    effective = effective_aqi_concentrations(frame)
    iaqi_cols: dict[str, pd.Series] = {}
    for pollutant in POLLUTANTS:
        iaqi_cols[f"IAQI_{pollutant}"] = effective[pollutant].apply(
            lambda value, p=pollutant: calculate_iaqi(p, value)
        )

    aqi = pd.DataFrame(iaqi_cols, index=frame.index)
    aqi_values = aqi.max(axis=1)
    primary = aqi.idxmax(axis=1).str.replace("IAQI_", "", regex=False)
    aqi["AQI"] = aqi_values
    aqi["Primary_Pollutant"] = primary
    aqi["AQI_Level"] = aqi_values.apply(aqi_category)
    return aqi


def add_time_features(index: pd.DatetimeIndex) -> pd.DataFrame:
    features = pd.DataFrame(index=index)
    hour = index.hour.to_numpy()
    dayofyear = index.dayofyear.to_numpy()
    weekday = index.dayofweek.to_numpy()
    features["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    features["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    features["doy_sin"] = np.sin(2 * np.pi * dayofyear / 366)
    features["doy_cos"] = np.cos(2 * np.pi * dayofyear / 366)
    features["weekday_sin"] = np.sin(2 * np.pi * weekday / 7)
    features["weekday_cos"] = np.cos(2 * np.pi * weekday / 7)
    features["elapsed_days"] = (index - index.min()).total_seconds() / 86400
    return features


def lagged_features(
    frame: pd.DataFrame,
    columns: Iterable[str],
    lags: Iterable[int],
    rolling_windows: Iterable[int] = (),
    include_current: bool = True,
) -> pd.DataFrame:
    features = pd.DataFrame(index=frame.index)
    for col in columns:
        if include_current:
            features[col] = frame[col]
        for lag in lags:
            features[f"{col}_lag_{lag}h"] = frame[col].shift(lag)
        for window in rolling_windows:
            rolled = frame[col].shift(1).rolling(window, min_periods=1).mean()
            features[f"{col}_roll_{window}h"] = rolled
    return features


def build_supervised_dataset(
    frame: pd.DataFrame,
    include_pollution_lags: bool,
    meteorology_lags: Iterable[int] = (1, 3, 6, 12, 24, 48),
    pollution_lags: Iterable[int] = (1, 3, 6, 12, 24, 48),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    features = [add_time_features(frame.index)]
    features.append(
        lagged_features(
            frame,
            METEOROLOGY,
            meteorology_lags,
            rolling_windows=(6, 12, 24),
        )
    )
    if include_pollution_lags:
        features.append(
            lagged_features(
                frame,
                POLLUTANTS,
                pollution_lags,
                rolling_windows=(6, 24),
                include_current=False,
            )
        )

    x = pd.concat(features, axis=1)
    y = frame[POLLUTANTS].copy()
    valid = x.join(y, how="inner").dropna()
    return valid[x.columns], valid[POLLUTANTS]


def chronological_split(
    x: pd.DataFrame, y: pd.DataFrame, test_fraction: float = 0.2
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction must be between 0 and 1.")
    split = int(len(x) * (1 - test_fraction))
    return x.iloc[:split], x.iloc[split:], y.iloc[:split], y.iloc[split:]


def regression_metrics(y_true: pd.DataFrame, y_pred: pd.DataFrame, model_name: str) -> pd.DataFrame:
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    rows = []
    for pollutant in POLLUTANTS:
        truth = y_true[pollutant]
        pred = y_pred[pollutant]
        rows.append(
            {
                "Model": model_name,
                "Pollutant": pollutant,
                "MAE": mean_absolute_error(truth, pred),
                "RMSE": mean_squared_error(truth, pred) ** 0.5,
                "R2": r2_score(truth, pred),
            }
        )
    return pd.DataFrame(rows)


def seasonal_naive_predictions(y: pd.DataFrame, test_index: pd.DatetimeIndex, lag_hours: int = 24) -> pd.DataFrame:
    return seasonal_lag_frame(y, test_index, lag_hours=lag_hours)


def seasonal_lag_frame(frame: pd.DataFrame, target_index: pd.DatetimeIndex, lag_hours: int = 24) -> pd.DataFrame:
    shifted = frame[POLLUTANTS].shift(lag_hours)
    return shifted.loc[target_index].bfill().ffill()


def fit_ridge_model(x_train: pd.DataFrame, y_train: pd.DataFrame):
    from sklearn.linear_model import RidgeCV
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    return make_pipeline(
        StandardScaler(),
        RidgeCV(alphas=np.logspace(-3, 4, 20)),
    ).fit(x_train, y_train)


@dataclass(frozen=True)
class VarModelBundle:
    results: object
    lag_order: int
    exog_columns: list[str]


def build_var_exog_features(
    frame: pd.DataFrame,
    lags: Iterable[int] = (1, 3, 6, 12, 24),
) -> pd.DataFrame:
    return lagged_features(
        frame,
        METEOROLOGY,
        lags=lags,
        rolling_windows=(),
        include_current=True,
    )


def fit_var_model(
    residual_train: pd.DataFrame,
    exog_train: pd.DataFrame | None = None,
    lag_order: int = 2,
) -> VarModelBundle:
    from statsmodels.tsa.api import VAR

    y = residual_train.loc[:, POLLUTANTS].copy()
    exog = None
    exog_columns: list[str] = []
    if exog_train is not None:
        exog = exog_train.copy()
        exog_columns = list(exog.columns)

    model = VAR(endog=y, exog=exog)
    lag_order = max(1, int(lag_order))
    max_feasible_lag = max(1, min(24, len(y) // 4))
    lag_order = min(lag_order, max_feasible_lag)

    try:
        results = model.fit(lag_order)
        used_lag = lag_order
    except Exception:
        results = model.fit(1)
        used_lag = 1

    return VarModelBundle(results=results, lag_order=used_lag, exog_columns=exog_columns)


def predict_var_residuals(
    model_bundle: VarModelBundle,
    residual_train: pd.DataFrame,
    target_index: pd.DatetimeIndex,
    exog_future: pd.DataFrame | None = None,
) -> pd.DataFrame:
    history = residual_train.loc[:, POLLUTANTS].iloc[-model_bundle.lag_order :].to_numpy()

    exog_future_np = None
    if model_bundle.exog_columns:
        if exog_future is None:
            raise ValueError("exog_future is required for VAR model with exogenous variables.")
        exog_aligned = exog_future.loc[target_index, model_bundle.exog_columns]
        exog_future_np = exog_aligned.to_numpy()

    pred = model_bundle.results.forecast(
        y=history,
        steps=len(target_index),
        exog_future=exog_future_np,
    )
    return pd.DataFrame(pred, index=target_index, columns=POLLUTANTS)


@dataclass(frozen=True)
class VecmModelBundle:
    results: object
    k_ar_diff: int
    coint_rank: int
    deterministic: str
    exog_columns: list[str]


@dataclass(frozen=True)
class SparseVarBundle:
    model: object
    lag_order: int
    exog_columns: list[str]
    n_targets: int
    alpha: float
    n_nonzero_coefs: int
    sparsity_ratio: float


def _to_numpy_matrix(frame: pd.DataFrame, columns: list[str]) -> np.ndarray:
    return frame.loc[:, columns].to_numpy(dtype=float)


def _build_sparse_design(
    residual: pd.DataFrame,
    exog: pd.DataFrame | None,
    lag_order: int,
) -> tuple[np.ndarray, np.ndarray]:
    y = _to_numpy_matrix(residual, POLLUTANTS)
    exog_values = exog.to_numpy(dtype=float) if exog is not None else None

    rows_x: list[np.ndarray] = []
    rows_y: list[np.ndarray] = []
    for t in range(lag_order, len(residual)):
        lagged = [y[t - lag] for lag in range(1, lag_order + 1)]
        features = np.concatenate(lagged)
        if exog_values is not None:
            features = np.concatenate([features, exog_values[t]])
        rows_x.append(features)
        rows_y.append(y[t])

    if not rows_x:
        raise ValueError("Not enough samples for sparse VAR design matrix.")

    return np.asarray(rows_x, dtype=float), np.asarray(rows_y, dtype=float)


def fit_vecm_model(
    residual_train: pd.DataFrame,
    exog_train: pd.DataFrame | None = None,
    k_ar_diff: int = 1,
    coint_rank: int = 1,
    deterministic: str = "ci",
) -> VecmModelBundle:
    from statsmodels.tsa.vector_ar.vecm import VECM

    y = residual_train.loc[:, POLLUTANTS].copy()
    exog = None
    exog_columns: list[str] = []
    if exog_train is not None:
        exog = exog_train.copy()
        exog_columns = list(exog.columns)

    k_ar_diff = max(1, int(k_ar_diff))
    coint_rank = max(1, min(int(coint_rank), len(POLLUTANTS)))

    try:
        model = VECM(
            endog=y,
            exog=exog,
            k_ar_diff=k_ar_diff,
            coint_rank=coint_rank,
            deterministic=deterministic,
        )
        results = model.fit()
    except Exception:
        model = VECM(
            endog=y,
            exog=exog,
            k_ar_diff=1,
            coint_rank=min(1, len(POLLUTANTS)),
            deterministic="ci",
        )
        results = model.fit()
        k_ar_diff = 1
        coint_rank = min(1, len(POLLUTANTS))
        deterministic = "ci"

    return VecmModelBundle(
        results=results,
        k_ar_diff=k_ar_diff,
        coint_rank=coint_rank,
        deterministic=deterministic,
        exog_columns=exog_columns,
    )


def predict_vecm_residuals(
    model_bundle: VecmModelBundle,
    residual_train: pd.DataFrame,
    target_index: pd.DatetimeIndex,
    exog_future: pd.DataFrame | None = None,
) -> pd.DataFrame:
    history = residual_train.loc[:, POLLUTANTS].to_numpy(dtype=float)
    k_ar = int(getattr(model_bundle.results, "k_ar", model_bundle.k_ar_diff + 1))
    if len(history) < k_ar:
        raise ValueError("Not enough history for VECM prediction.")

    exog_future_np = None
    if model_bundle.exog_columns:
        if exog_future is None:
            raise ValueError("exog_future is required for VECM model with exogenous variables.")
        exog_future_np = exog_future.loc[target_index, model_bundle.exog_columns].to_numpy(dtype=float)

    pred = model_bundle.results.predict(
        steps=len(target_index),
        exog_fc=exog_future_np,
    )
    return pd.DataFrame(pred, index=target_index, columns=POLLUTANTS)


def fit_sparse_var_model(
    residual_train: pd.DataFrame,
    exog_train: pd.DataFrame | None = None,
    lag_order: int = 2,
) -> SparseVarBundle:
    from sklearn.linear_model import MultiTaskLassoCV

    lag_order = max(1, int(lag_order))
    exog_columns = list(exog_train.columns) if exog_train is not None else []

    x, y = _build_sparse_design(residual_train, exog_train, lag_order=lag_order)
    model = MultiTaskLassoCV(cv=5, random_state=2026, max_iter=20000)
    model.fit(x, y)

    coef = np.asarray(model.coef_, dtype=float)
    n_nonzero = int(np.sum(np.abs(coef) > 1e-10))
    sparsity = 1.0 - (n_nonzero / float(coef.size)) if coef.size else 0.0

    return SparseVarBundle(
        model=model,
        lag_order=lag_order,
        exog_columns=exog_columns,
        n_targets=len(POLLUTANTS),
        alpha=float(getattr(model, "alpha_", 0.0)),
        n_nonzero_coefs=n_nonzero,
        sparsity_ratio=float(sparsity),
    )


def predict_sparse_var_residuals(
    model_bundle: SparseVarBundle,
    residual_train: pd.DataFrame,
    target_index: pd.DatetimeIndex,
    exog_future: pd.DataFrame | None = None,
) -> pd.DataFrame:
    history = residual_train.loc[:, POLLUTANTS].copy()

    preds: list[np.ndarray] = []
    for ts in target_index:
        lagged = [history.iloc[-lag].to_numpy(dtype=float) for lag in range(1, model_bundle.lag_order + 1)]
        features = np.concatenate(lagged)
        if model_bundle.exog_columns:
            if exog_future is None:
                raise ValueError("exog_future is required for sparse VAR model with exogenous variables.")
            exog_row = exog_future.loc[ts, model_bundle.exog_columns].to_numpy(dtype=float)
            features = np.concatenate([features, exog_row])

        pred = np.asarray(model_bundle.model.predict(features.reshape(1, -1))[0], dtype=float)
        preds.append(pred)
        history.loc[ts] = pred

    return pd.DataFrame(preds, index=target_index, columns=POLLUTANTS)


def evaluate_predictions(y_true: pd.DataFrame, y_pred: pd.DataFrame) -> pd.DataFrame:
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    common_idx = y_true.index.intersection(y_pred.index)
    truth = y_true.loc[common_idx, POLLUTANTS]
    pred = y_pred.loc[common_idx, POLLUTANTS]

    rows = []
    for pollutant in POLLUTANTS:
        y_t = truth[pollutant]
        y_p = pred[pollutant]
        rows.append(
            {
                "Pollutant": pollutant,
                "MAE": mean_absolute_error(y_t, y_p),
                "RMSE": mean_squared_error(y_t, y_p) ** 0.5,
                "R2": r2_score(y_t, y_p),
            }
        )
    return pd.DataFrame(rows)


def compute_aqi_hourly(frame: pd.DataFrame) -> pd.DataFrame:
    return compute_aqi_frame(frame)


def predict_frame(model, x_test: pd.DataFrame) -> pd.DataFrame:
    pred = pd.DataFrame(model.predict(x_test), index=x_test.index, columns=POLLUTANTS)
    for pollutant in POLLUTANTS:
        pred[pollutant] = pred[pollutant].clip(lower=0)
    return pred


def daypart_label(hour: int) -> str | None:
    if 6 <= hour <= 11:
        return "Morning"
    if 12 <= hour <= 17:
        return "Afternoon"
    if 18 <= hour <= 23:
        return "Evening"
    return None


def daypart_recommendations(aqi: pd.DataFrame) -> pd.DataFrame:
    frame = aqi.copy()
    frame["Date"] = frame.index.date
    frame["Daypart"] = [daypart_label(h) for h in frame.index.hour]
    frame = frame[frame["Daypart"].notna()].copy()

    rows = []
    for (date, daypart), group in frame.groupby(["Date", "Daypart"], sort=True):
        peak_row = group.loc[group["AQI"].idxmax()]
        level = str(peak_row["AQI_Level"])
        rows.append(
            {
                "Date": str(date),
                "Daypart": daypart,
                "Peak_AQI": float(peak_row["AQI"]),
                "AQI_Level": level,
                "Primary_Pollutant": str(peak_row.get("Primary_Pollutant", "")),
                "Advice": ACTIVITY_ADVICE[level],
            }
        )
    return pd.DataFrame(rows)
