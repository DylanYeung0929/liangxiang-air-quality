from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import torch.nn as nn
from sklearn.linear_model import RidgeCV

from src.air_quality import POLLUTANTS


@dataclass
class ESNModel:
    win: np.ndarray
    wres: sp.csr_matrix
    readout: RidgeCV
    leak_rate: float
    noise_scale: float
    washout: int
    last_state: np.ndarray
    x_mean: np.ndarray
    x_std: np.ndarray


class LSTNet(nn.Module):
    def __init__(
        self,
        n_features: int,
        hidden_dim: int = 16,
        kernel_size: int = 3,
        dropout: float = 0.4,
        ar_window: int = 24,
    ):
        super().__init__()
        self.ar_window = ar_window
        self.conv = nn.Conv1d(
            in_channels=n_features,
            out_channels=hidden_dim,
            kernel_size=kernel_size,
        )
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(hidden_dim, n_features)
        self.ar = nn.Linear(ar_window, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, seq_len, features]
        z = x.transpose(1, 2)  # [batch, features, seq_len]
        z = torch.relu(self.conv(z))
        z = z.transpose(1, 2)  # [batch, seq_len', hidden]
        _, h = self.gru(z)
        h = self.dropout(h[-1])
        nonlinear = self.proj(h)

        ar_input = x[:, -self.ar_window :, :].transpose(1, 2)  # [batch, features, ar_window]
        ar_part = self.ar(ar_input).squeeze(-1)
        return nonlinear + ar_part


@dataclass
class LSTNetBundle:
    model: LSTNet
    lookback: int
    device: torch.device
    y_mean: np.ndarray
    y_std: np.ndarray


def _build_sparse_reservoir(
    reservoir_size: int,
    sparsity: float,
    spectral_radius: float,
) -> sp.csr_matrix:
    rng = np.random.default_rng(2026)
    wres = sp.random(
        reservoir_size,
        reservoir_size,
        density=sparsity,
        format="csr",
        data_rvs=lambda n: rng.uniform(-0.5, 0.5, size=n).astype(np.float32),
    ).astype(np.float32).tocsr()

    v = rng.normal(size=reservoir_size).astype(np.float32)
    v = v / (np.linalg.norm(v) + 1e-12)
    radius = 0.0
    for _ in range(60):
        v = wres.dot(v)
        radius = float(np.linalg.norm(v))
        if not np.isfinite(radius) or radius < 1e-12:
            radius = 1.0
            break
        v = v / radius

    scale = spectral_radius / max(radius, 1e-6)
    wres = (wres * np.float32(scale)).tocsr()
    wres.data = np.clip(wres.data, -2.0, 2.0).astype(np.float32)
    return wres


def fit_esn_model(
    x_train: pd.DataFrame,
    y_train: pd.DataFrame,
    reservoir_size: int = 500,
    sparsity: float = 0.1,
    spectral_radius: float = 0.85,
    leak_rate: float = 0.2,
    washout: int = 50,
    noise_scale: float = 1e-4,
) -> ESNModel:
    x_raw = x_train.to_numpy(dtype=np.float32)
    y = y_train[POLLUTANTS].to_numpy(dtype=np.float32)

    x_mean = x_raw.mean(axis=0)
    x_std = x_raw.std(axis=0)
    x_std = np.where(x_std < 1e-3, 1.0, x_std)
    x = np.clip((x_raw - x_mean) / x_std, -10.0, 10.0)

    in_dim = x.shape[1]
    rng = np.random.default_rng(2026)
    win = rng.uniform(-0.05, 0.05, size=(reservoir_size, in_dim + 1)).astype(np.float32)
    wres = _build_sparse_reservoir(reservoir_size, sparsity, spectral_radius)

    state = np.zeros(reservoir_size, dtype=np.float32)
    states = []
    targets = []

    for t in range(len(x)):
        u = np.concatenate(([1.0], x[t])).astype(np.float32)
        u = np.nan_to_num(u, nan=0.0, posinf=10.0, neginf=-10.0)
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            pre_raw = win @ u + wres.dot(state) + rng.normal(0.0, noise_scale, size=reservoir_size).astype(np.float32)
        pre = np.nan_to_num(pre_raw, nan=0.0, posinf=10.0, neginf=-10.0)
        state = (1.0 - leak_rate) * state + leak_rate * np.tanh(np.clip(pre, -10.0, 10.0))
        state = np.nan_to_num(state, nan=0.0, posinf=1.0, neginf=-1.0)
        if t >= washout:
            states.append(np.concatenate([state, x[t]]).astype(np.float32))
            targets.append(y[t].copy())

    state_matrix = np.asarray(states, dtype=np.float64)
    target_matrix = np.asarray(targets, dtype=np.float64)

    readout = RidgeCV(alphas=np.logspace(-3, 4, 20))
    readout.fit(state_matrix, target_matrix)

    return ESNModel(
        win=win,
        wres=wres,
        readout=readout,
        leak_rate=leak_rate,
        noise_scale=noise_scale,
        washout=washout,
        last_state=state.copy(),
        x_mean=x_mean,
        x_std=x_std,
    )


def predict_esn_model(
    model: ESNModel,
    x_test: pd.DataFrame,
    index: pd.Index,
) -> pd.DataFrame:
    x_raw = x_test.to_numpy(dtype=np.float32)
    x = np.clip((x_raw - model.x_mean) / model.x_std, -10.0, 10.0)
    state = model.last_state.astype(np.float32).copy()
    rng = np.random.default_rng(2027)

    preds = []
    for t in range(len(x)):
        u = np.concatenate(([1.0], x[t])).astype(np.float32)
        u = np.nan_to_num(u, nan=0.0, posinf=10.0, neginf=-10.0)
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            pre_raw = model.win @ u + model.wres.dot(state) + rng.normal(0.0, model.noise_scale, size=state.shape[0]).astype(np.float32)
        pre = np.nan_to_num(pre_raw, nan=0.0, posinf=10.0, neginf=-10.0)
        state = (1.0 - model.leak_rate) * state + model.leak_rate * np.tanh(np.clip(pre, -10.0, 10.0))
        state = np.nan_to_num(state, nan=0.0, posinf=1.0, neginf=-1.0)
        yhat = model.readout.predict(np.concatenate([state, x[t]]).reshape(1, -1))[0]
        preds.append(yhat)

    pred = pd.DataFrame(preds, index=index, columns=POLLUTANTS)
    return pred.clip(lower=0)


def _make_sequences(values: np.ndarray, lookback: int) -> tuple[np.ndarray, np.ndarray]:
    xs, ys = [], []
    for t in range(lookback, len(values)):
        xs.append(values[t - lookback : t])
        ys.append(values[t])
    return np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=np.float32)


def fit_lstnet_model(
    y_train: pd.DataFrame,
    hidden_dim: int = 16,
    kernel_size: int = 3,
    dropout: float = 0.4,
    weight_decay: float = 1e-3,
    epochs: int = 50,
    patience: int = 5,
    lookback: int = 48,
    ar_window: int = 24,
) -> LSTNetBundle:
    device = torch.device("cpu")

    values_raw = y_train[POLLUTANTS].to_numpy(dtype=np.float32)
    y_mean = values_raw.mean(axis=0)
    y_std = values_raw.std(axis=0)
    y_std = np.where(y_std < 1e-3, 1.0, y_std)
    values = (values_raw - y_mean) / y_std

    x_all, y_all = _make_sequences(values, lookback=lookback)

    split = int(len(x_all) * 0.85)
    x_tr, y_tr = x_all[:split], y_all[:split]
    x_va, y_va = x_all[split:], y_all[split:]

    x_tr_t = torch.from_numpy(x_tr).to(device)
    y_tr_t = torch.from_numpy(y_tr).to(device)
    x_va_t = torch.from_numpy(x_va).to(device)
    y_va_t = torch.from_numpy(y_va).to(device)

    model = LSTNet(
        n_features=len(POLLUTANTS),
        hidden_dim=hidden_dim,
        kernel_size=kernel_size,
        dropout=dropout,
        ar_window=ar_window,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=weight_decay)
    criterion = nn.MSELoss()

    best_state = None
    best_val = float("inf")
    bad_epochs = 0

    for _ in range(epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        pred = model(x_tr_t)
        loss = criterion(pred, y_tr_t)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_pred = model(x_va_t)
            val_loss = float(criterion(val_pred, y_va_t).item())

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return LSTNetBundle(model=model, lookback=lookback, device=device, y_mean=y_mean, y_std=y_std)


def predict_lstnet_model(
    bundle: LSTNetBundle,
    y_train: pd.DataFrame,
    test_index: pd.Index,
    y_test_true: pd.DataFrame | None = None,
) -> pd.DataFrame:
    model = bundle.model
    model.eval()

    history_raw = y_train[POLLUTANTS].to_numpy(dtype=np.float32)
    history = ((history_raw - bundle.y_mean) / bundle.y_std).tolist()
    preds = []

    with torch.no_grad():
        for step in range(len(test_index)):
            seq = np.asarray(history[-bundle.lookback :], dtype=np.float32)
            x_t = torch.from_numpy(seq).unsqueeze(0).to(bundle.device)
            yhat_scaled = np.clip(model(x_t).squeeze(0).cpu().numpy(), -3.0, 3.0)
            yhat = yhat_scaled * bundle.y_std + bundle.y_mean
            yhat = np.clip(yhat, a_min=0.0, a_max=None)
            preds.append(yhat)

            if y_test_true is not None:
                next_true = y_test_true.iloc[step][POLLUTANTS].to_numpy(dtype=np.float32)
                history.append(((next_true - bundle.y_mean) / bundle.y_std).astype(np.float32))
            else:
                history.append(((yhat - bundle.y_mean) / bundle.y_std).astype(np.float32))

    return pd.DataFrame(preds, index=test_index, columns=POLLUTANTS).clip(lower=0)
