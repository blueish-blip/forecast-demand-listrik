"""
build_hourly.py
===============
Pipeline data PER JAM untuk dashboard beban listrik Panama.

Menghasilkan ``docs/data/hourly_results.json`` berisi 7 model yang sama
dengan mode harian, dihitung pada resolusi per jam:

- Model_A  Prophet (daily + weekly + yearly seasonality)
- Model_B  Seasonal Naive periode 168 jam (7 hari)
- Model_C  Holt-Winters (trend aditif + seasonal aditif, 168 jam)
- Model_D  OLS trend linear + harmonik harian/mingguan/tahunan
- Model_E  GRU (lookback 24 jam)
- Model_F  LSTM (lookback 24 jam)
- Model_G  CNN-1D (lookback 24 jam)

Lingkup data:
- hanya hari lengkap 24 jam (2.001 hari = 48.024 jam),
- split: 47.304 jam train / 720 jam test / 720 jam future,
- tampilan: seluruh data historis (47.304 jam) + test + future = 49.464 baris,
  sama seperti mode harian yang menampilkan seluruh rentang data,
- interval 95%: Prophet memakai intervalnya sendiri; model lain memakai
  kuantil residu training (2,5% - 97,5%),
- deep learning dijalankan ulang dengan arsitektur identik notebook
  (batch 32, epoch 20, early stopping patience 10).

Output artefak:
- prophet_hourly_model.pkl, gru_hourly.keras, lstm_hourly.keras,
  cnn1d_hourly.keras, hourly_scalers.pkl, hourly_predictions.pkl
- docs/data/hourly_results.json

Jalankan:  python build_hourly.py
"""

from __future__ import annotations

import json
import pickle
import warnings

import numpy as np
import pandas as pd
from prophet import Prophet
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.holtwinters import ExponentialSmoothing

warnings.filterwarnings("ignore")

from build_data import MODEL_META as META_STAT
from build_dl_models import MODELS as DL_MODELS, build_cnn1d, build_lstm, fit_one
from build_gru import (
    BATCH_SIZE,
    BINARY,
    CONTINUOUS,
    EPOCHS,
    EXCEL_PATH,
    LOOKBACK,
    PATIENCE,
    ROOT,
    build_matrix,
    build_model as build_gru_base,
    clean_number,
    metric_block,
    recursive_forecast,
    residual_bounds,
    set_seed,
)

# ---------------------------------------------------------------------------
# Konfigurasi
# ---------------------------------------------------------------------------
OUT_JSON = ROOT / "docs" / "data" / "hourly_results.json"
PROPHET_PATH = ROOT / "prophet_hourly_model.pkl"
SCALER_PATH = ROOT / "hourly_scalers.pkl"
PRED_PATH = ROOT / "hourly_predictions.pkl"

MODEL_META = {
    **META_STAT,
    "Model_E": {"label": "GRU", "short": "GRU"},
    "Model_F": {"label": "LSTM", "short": "LSTM"},
    "Model_G": {"label": "CNN-1D", "short": "CNN-1D"},
}

SUBTITLE = (
    "Evaluasi dan proyeksi 30 hari ke depan (per jam) · Prophet vs Seasonal "
    "Naive vs Holt-Winters vs OLS Harmonik vs GRU vs LSTM vs CNN-1D"
)

HOUR_PERIOD = 168    # siklus mingguan dalam jam
TEST_HOURS = 720     # 30 hari
FUTURE_HOURS = 720   # 30 hari
HIST_HOURS = None     # None = seluruh data historis (47.304 jam) yang ditampilkan
VALID_HOURS = 720    # window validasi terakhir saat training
SEED = 42
GRANULARITY = "hourly"

# builder + konfigurasi untuk tiga model deep learning
DL_SPECS = {
    "Model_E": (lambda shape, cfg: build_gru_base(shape),
                {"prefix": "gru", "reduce_lr": False, "shuffle": True}),
    "Model_F": (build_lstm, DL_MODELS["Model_F"]),
    "Model_G": (build_cnn1d, DL_MODELS["Model_G"]),
}


# ---------------------------------------------------------------------------
# Utilitas model klasik
# ---------------------------------------------------------------------------
def fit_prophet_hourly(df: pd.DataFrame) -> Prophet:
    model = Prophet(
        daily_seasonality=True,
        weekly_seasonality=True,
        yearly_seasonality=True,
        seasonality_mode="additive",
        interval_width=0.95,
    )
    model.fit(df[["ds", "y"]])
    return model


def harmonic_hourly(t: np.ndarray) -> np.ndarray:
    """Matriks desain per jam: intersep, tren, harmonik harian/mingguan/tahunan."""
    t = np.asarray(t, dtype=float)
    cols = [np.ones_like(t), t]
    for period in (24.0, HOUR_PERIOD, 365.25 * 24.0):
        for k in range(1, 4):
            cols.append(np.sin(2 * np.pi * k * t / period))
            cols.append(np.cos(2 * np.pi * k * t / period))
    return np.column_stack(cols)


def seasonal_naive_fitted(values, period=HOUR_PERIOD):
    s = pd.Series(np.asarray(values, dtype=float))
    return s.shift(period).to_numpy()


def seasonal_naive_forecast(history, horizon, period=HOUR_PERIOD):
    history = list(np.asarray(history, dtype=float))
    preds = []
    for h in range(horizon):
        idx = len(history) + h - period
        if idx < len(history):
            preds.append(history[idx])
        else:
            preds.append(preds[idx - len(history)])
    return np.asarray(preds, dtype=float)


def interval_stack(mean, lo, hi):
    """Kolom [mean, lower, upper] dengan offset residu yang sama."""
    mean = np.asarray(mean, dtype=float)
    return np.column_stack([mean, mean + lo, mean + hi])


def triple(vals):
    return {
        "mean": clean_number(vals[0]),
        "lower_95": clean_number(vals[1]),
        "upper_95": clean_number(vals[2]),
    }


# ---------------------------------------------------------------------------
# 1. Muat data per jam (hanya hari lengkap 24 jam)
# ---------------------------------------------------------------------------
def load_hourly() -> pd.DataFrame:
    if not EXCEL_PATH.exists():
        raise FileNotFoundError(f"Dataset tidak ditemukan: {EXCEL_PATH}")

    usecols = ["datetime", "nat_demand"] + CONTINUOUS + BINARY
    raw = pd.read_excel(EXCEL_PATH, sheet_name="data", usecols=usecols)
    raw["datetime"] = pd.to_datetime(raw["datetime"])
    raw["date"] = raw["datetime"].dt.normalize()

    grouped = (
        raw.groupby("date")["nat_demand"]
        .agg(total="sum", hours="count")
        .reset_index()
    )
    partial = grouped[grouped["hours"] != 24]
    if not partial.empty:
        print("Hari tidak lengkap (dibuang):")
        for _, row in partial.iterrows():
            print(f"  - {row['date'].date()} ({int(row['hours'])} jam)")

    complete = set(grouped[grouped["hours"] == 24]["date"])
    hourly = (
        raw[raw["date"].isin(complete)]
        .sort_values("datetime")
        .drop(columns=["date"])
        .rename(columns={"datetime": "ds", "nat_demand": "y"})
        .reset_index(drop=True)
    )
    return hourly


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    set_seed(SEED)

    hourly = load_hourly()
    n = len(hourly)
    n_train = n - TEST_HOURS
    y = hourly["y"].to_numpy(dtype=float)
    y_train = y[:n_train]
    y_test = y[n_train:]

    future_ts = pd.date_range(
        hourly["ds"].iloc[-1] + pd.Timedelta(hours=1),
        periods=FUTURE_HOURS,
        freq="h",
    )
    print(f"\nJam bersih            : {n} ({n // 24} hari)")
    print(f"Train                 : {hourly['ds'].iloc[0]} -> "
          f"{hourly['ds'].iloc[n_train - 1]} ({n_train} jam)")
    print(f"Test                  : {hourly['ds'].iloc[n_train]} -> "
          f"{hourly['ds'].iloc[-1]} ({TEST_HOURS} jam)")
    print(f"Future                : {future_ts[0]} -> {future_ts[-1]} "
          f"({FUTURE_HOURS} jam)")
    hist_hours = n_train if HIST_HOURS is None else min(HIST_HOURS, n_train)
    print(f"Jendela tampilan      : {hist_hours} jam historis + test + future")

    train_df = hourly.iloc[:n_train][["ds", "y"]]
    test_df = hourly.iloc[n_train:][["ds", "y"]]

    forecasts: dict = {}
    metrics: dict = {}

    # ---------------------------- Model A: Prophet -------------------------
    print("\n[Model A] Prophet per jam (daily+weekly+yearly) ...")
    prophet_train = fit_prophet_hourly(train_df)
    fc_train = prophet_train.predict(train_df[["ds"]])
    fc_test = prophet_train.predict(test_df[["ds"]])

    print("[Model A] Refit Prophet per jam pada seluruh data ...")
    prophet_full = fit_prophet_hourly(hourly[["ds", "y"]])
    fc_future = prophet_full.predict(pd.DataFrame({"ds": future_ts}))
    with open(PROPHET_PATH, "wb") as fh:
        pickle.dump(prophet_full, fh)
    print(f"[Model A] Model disimpan -> {PROPHET_PATH.name}")

    forecasts["Model_A"] = {
        "train": fc_train[["yhat", "yhat_lower", "yhat_upper"]].to_numpy(),
        "test": fc_test[["yhat", "yhat_lower", "yhat_upper"]].to_numpy(),
        "future": fc_future[["yhat", "yhat_lower", "yhat_upper"]].to_numpy(),
    }
    metrics["Model_A"] = metric_block(
        y_test,
        fc_test["yhat"].to_numpy(),
        y_train - fc_train["yhat"].to_numpy(),
        n_params=25,
    )

    # ------------------- Model B: Seasonal Naive 168 jam ------------------
    print("[Model B] Seasonal Naive 168 jam (7 hari) ...")
    naive_fitted = seasonal_naive_fitted(y_train)
    naive_test = seasonal_naive_forecast(y_train, TEST_HOURS)
    naive_future = seasonal_naive_forecast(y, FUTURE_HOURS)
    lo_b, hi_b = residual_bounds(y_train - naive_fitted)
    forecasts["Model_B"] = {
        "train": interval_stack(naive_fitted, lo_b, hi_b),
        "test": interval_stack(naive_test, lo_b, hi_b),
        "future": interval_stack(naive_future, lo_b, hi_b),
    }
    metrics["Model_B"] = metric_block(
        y_test, naive_test, y_train - naive_fitted, n_params=1
    )

    # --------------------------- Model C: Holt-Winters ---------------------
    print("[Model C] Holt-Winters (trend + musiman aditif, 168 jam) ...")
    hw_train = ExponentialSmoothing(
        y_train,
        trend="add",
        damped_trend=False,
        seasonal="add",
        seasonal_periods=HOUR_PERIOD,
        initialization_method="estimated",
    ).fit(optimized=True)
    hw_fitted = np.asarray(hw_train.fittedvalues, dtype=float)
    hw_test = np.asarray(hw_train.forecast(TEST_HOURS), dtype=float)

    hw_full = ExponentialSmoothing(
        y,
        trend="add",
        damped_trend=False,
        seasonal="add",
        seasonal_periods=HOUR_PERIOD,
        initialization_method="estimated",
    ).fit(optimized=True)
    hw_future = np.asarray(hw_full.forecast(FUTURE_HOURS), dtype=float)

    lo_c, hi_c = residual_bounds(y_train - hw_fitted)
    forecasts["Model_C"] = {
        "train": interval_stack(hw_fitted, lo_c, hi_c),
        "test": interval_stack(hw_test, lo_c, hi_c),
        "future": interval_stack(hw_future, lo_c, hi_c),
    }
    metrics["Model_C"] = metric_block(
        y_test, hw_test, y_train - hw_fitted, n_params=9
    )

    # -------------------- Model D: OLS trend + harmonik --------------------
    print("[Model D] OLS trend + harmonik harian/mingguan/tahunan ...")
    X_train = harmonic_hourly(np.arange(n_train, dtype=float))
    beta_tr, *_ = np.linalg.lstsq(X_train, y_train, rcond=None)
    ols_fitted = X_train @ beta_tr
    ols_test = harmonic_hourly(np.arange(n_train, n, dtype=float)) @ beta_tr

    X_all = harmonic_hourly(np.arange(n, dtype=float))
    beta_full, *_ = np.linalg.lstsq(X_all, y, rcond=None)
    ols_future = (
        harmonic_hourly(np.arange(n, n + FUTURE_HOURS, dtype=float)) @ beta_full
    )

    lo_d, hi_d = residual_bounds(y_train - ols_fitted)
    forecasts["Model_D"] = {
        "train": interval_stack(ols_fitted, lo_d, hi_d),
        "test": interval_stack(ols_test, lo_d, hi_d),
        "future": interval_stack(ols_future, lo_d, hi_d),
    }
    metrics["Model_D"] = metric_block(
        y_test, ols_test, y_train - ols_fitted, n_params=X_train.shape[1]
    )

    # --------------- Model E/F/G: GRU, LSTM, CNN-1D (jam) ------------------
    scaler_store: dict = {}
    pred_store: dict = {}
    for key, (builder, cfg) in DL_SPECS.items():
        label = MODEL_META[key]["label"]
        prefix = cfg["prefix"]

        set_seed(SEED)
        print(f"\n[{label}] Melatih model evaluasi per jam ...")
        fs_eval, ts_eval = StandardScaler(), StandardScaler()
        X_eval = build_matrix(hourly, fs_eval, ts_eval, fit_on=True)

        eval_model = fit_one(
            builder, cfg, X_eval, X_eval[:, 0],
            n_train - VALID_HOURS, n_train, f"{label} eval",
        )

        fitted = np.full(n_train, np.nan, dtype=float)
        fit_idx = np.arange(LOOKBACK, n_train)
        fit_seq = np.asarray([X_eval[i - LOOKBACK:i] for i in fit_idx])
        fit_pred = eval_model.predict(fit_seq, batch_size=512, verbose=0).ravel()
        fitted[fit_idx] = ts_eval.inverse_transform(
            fit_pred.reshape(-1, 1)
        ).ravel()

        pred_test = recursive_forecast(
            eval_model,
            X_eval[n_train - LOOKBACK:n_train],
            X_eval[n_train:n, 1:],
            ts_eval,
            TEST_HOURS,
        )
        lo, hi = residual_bounds(y_train - fitted)
        metrics[key] = metric_block(
            y_test, pred_test, y_train - fitted,
            n_params=int(eval_model.count_params()),
        )
        print(f"  [{label}] RMSE test: {metrics[key]['rmse']:,.4f}")

        set_seed(SEED)
        print(f"[{label}] Refit model per jam pada seluruh data ...")
        fs_full, ts_full = StandardScaler(), StandardScaler()
        X_full = build_matrix(hourly, fs_full, ts_full, fit_on=True)
        final_model = fit_one(
            builder, cfg, X_full, X_full[:, 0],
            n - VALID_HOURS, n, f"{label} final",
        )

        exog_cycle = X_full[n - HOUR_PERIOD:, 1:]
        exog_future = [exog_cycle[h % HOUR_PERIOD] for h in range(FUTURE_HOURS)]
        pred_future = recursive_forecast(
            final_model, X_full, exog_future, ts_full, FUTURE_HOURS
        )

        model_path = ROOT / f"{prefix}_hourly.keras"
        final_model.save(model_path)
        scaler_store[key] = {"feature_scaler": fs_full, "target_scaler": ts_full}
        pred_store[key] = {
            "fitted_train": fitted,
            "pred_test": pred_test,
            "pred_future": pred_future,
            "metrics": metrics[key],
            "lower_offset": lo,
            "upper_offset": hi,
        }
        print(f"  Model disimpan -> {model_path.name}")

        forecasts[key] = {
            "train": interval_stack(fitted, lo, hi),
            "test": interval_stack(pred_test, lo, hi),
            "future": interval_stack(pred_future, lo, hi),
        }

    with open(SCALER_PATH, "wb") as fh:
        pickle.dump({"scalers": scaler_store, "lookback": LOOKBACK}, fh)
    with open(PRED_PATH, "wb") as fh:
        pickle.dump(pred_store, fh)
    print(f"\nScaler disimpan  -> {SCALER_PATH.name}")
    print(f"Prediksi disimpan -> {PRED_PATH.name}")

    # ---------------------------------------------------------------------
    # Susun payload JSON (30 hari historis + test + future)
    # ---------------------------------------------------------------------
    best_model = min(metrics, key=lambda key: metrics[key]["rmse"])
    print("\nRMSE test per jam:")
    for key in MODEL_META:
        print(f"  {key} {MODEL_META[key]['label']:<24} {metrics[key]['rmse']:>12,.2f}")
    print(f"Model terbaik: {best_model} ({MODEL_META[best_model]['label']})")

    series = []
    hist_start = n_train - hist_hours
    for i in range(hist_start, n_train):
        series.append({
            "date": hourly["ds"].iloc[i].strftime("%Y-%m-%d %H:%M"),
            "split": "train",
            "actual": clean_number(y[i]),
            "forecasts": {k: triple(forecasts[k]["train"][i]) for k in MODEL_META},
        })
    for i in range(n_train, n):
        series.append({
            "date": hourly["ds"].iloc[i].strftime("%Y-%m-%d %H:%M"),
            "split": "test",
            "actual": clean_number(y[i]),
            "forecasts": {
                k: triple(forecasts[k]["test"][i - n_train]) for k in MODEL_META
            },
        })
    for j, ts in enumerate(future_ts):
        series.append({
            "date": ts.strftime("%Y-%m-%d %H:%M"),
            "split": "future",
            "actual": None,
            "forecasts": {k: triple(forecasts[k]["future"][j]) for k in MODEL_META},
        })

    payload = {
        "metadata": {
            "title": "Dashboard Peramalan Beban Listrik Harian Panama",
            "subtitle": SUBTITLE,
            "unit": "MWh",
            "granularity": GRANULARITY,
            "resolution_hours": 1,
            "last_updated": pd.Timestamp.today().strftime("%Y-%m-%d"),
            "forecast_start_date": future_ts[0].strftime("%Y-%m-%d"),
            "data_start_date": hourly["ds"].min().strftime("%Y-%m-%d"),
            "data_end_date": hourly["ds"].max().strftime("%Y-%m-%d"),
            "test_start_date": hourly["ds"].iloc[n_train].strftime("%Y-%m-%d"),
            "test_end_date": hourly["ds"].iloc[-1].strftime("%Y-%m-%d"),
            "train_days": int(n_train // 24),
            "test_days": int(TEST_HOURS // 24),
            "test_hours": int(TEST_HOURS),
            "future_days": int(FUTURE_HOURS // 24),
            "future_hours": int(FUTURE_HOURS),
            "history_days": int(hist_hours // 24),
            "history_hours": int(hist_hours),
            "total_days": int(n // 24),
            "best_model": best_model,
            "models": MODEL_META,
        },
        "metrics": {
            key: {
                "label": MODEL_META[key]["label"],
                "short": MODEL_META[key]["short"],
                **{k: clean_number(v) for k, v in values.items()},
            }
            for key, values in metrics.items()
        },
        "series": series,
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, allow_nan=False,
                  separators=(",", ":"))

    print(f"\nJSON -> {OUT_JSON.relative_to(ROOT)} "
          f"({OUT_JSON.stat().st_size:,} bytes, {len(series)} titik)")
    print("Selesai.")


if __name__ == "__main__":
    main()
