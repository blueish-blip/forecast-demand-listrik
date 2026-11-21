"""
build_data.py
=============
Pipeline data untuk dashboard peramalan beban listrik harian Panama.

Langkah:
1. Baca data per jam dari Excel, agregasi ke harian (hanya hari lengkap 24 jam).
2. Split: 30 hari terakhir sebagai test, sisanya train.
3. Latih 4 model: Prophet, Seasonal Naive 7 hari, Holt-Winters, OLS harmonik.
4. Hitung metrik evaluasi (RMSE, MAE, MAPE, sMAPE, R2, AIC).
5. Refit Prophet + model baseline pada seluruh data bersih, ramalkan 30 hari.
6. Simpan model Prophet dan ekspor docs/data/forecast_results.json.

Jalankan:  python build_data.py
"""

from __future__ import annotations

import json
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from prophet import Prophet
from statsmodels.tsa.holtwinters import ExponentialSmoothing

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Konfigurasi
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
EXCEL_PATH = Path(
    r"C:\Users\hhfzsn\Downloads\Panama_beban_listrik_per_jam_2015-2020_.xlsx"
)
OUT_JSON = ROOT / "docs" / "data" / "forecast_results.json"
MODEL_PATH = ROOT / "prophet_model.pkl"
RESULTS_PATH = ROOT / "forecast_results.pkl"

TEST_DAYS = 30
FUTURE_DAYS = 30
SEASONAL_PERIOD = 7
INTERVAL_WIDTH = 0.95

MODEL_META = {
    "Model_A": {"label": "Prophet", "short": "Prophet"},
    "Model_B": {"label": "Seasonal Naive 7 Hari", "short": "Seasonal Naive"},
    "Model_C": {"label": "Holt-Winters", "short": "Holt-Winters"},
    "Model_D": {"label": "OLS Trend + Harmonik", "short": "OLS Harmonik"},
}


# ---------------------------------------------------------------------------
# Utilitas
# ---------------------------------------------------------------------------
def clean_number(value):
    """Ubah numpy/NaN menjadi float/None yang aman untuk JSON."""
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return None if not np.isfinite(value) else value


def metric_block(y_true, y_pred, train_resid, n_params):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    err = y_true - y_pred
    abs_err = np.abs(err)

    rmse = float(np.sqrt(np.mean(err**2)))
    mae = float(np.mean(abs_err))
    mape = float(np.mean(abs_err / np.abs(y_true)) * 100.0)
    smape = float(
        np.mean(2.0 * abs_err / (np.abs(y_true) + np.abs(y_pred))) * 100.0
    )

    ss_res = float(np.sum(err**2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else None

    resid = np.asarray(train_resid, dtype=float)
    resid = resid[np.isfinite(resid)]
    if resid.size and np.sum(resid**2) > 0:
        aic = float(
            resid.size * np.log(np.sum(resid**2) / resid.size) + 2 * n_params
        )
    else:
        aic = None

    return {
        "rmse": rmse,
        "mae": mae,
        "mape": mape,
        "smape": smape,
        "r2": r2,
        "aic": aic,
    }


def residual_bounds(resid, level=INTERVAL_WIDTH):
    """Kuantil empiris residual (two-sided) untuk interval baseline."""
    resid = np.asarray(resid, dtype=float)
    resid = resid[np.isfinite(resid)]
    alpha = (1.0 - level) / 2.0
    lo, hi = np.quantile(resid, [alpha, 1.0 - alpha])
    return float(lo), float(hi)


def harmonic_design(t: np.ndarray) -> np.ndarray:
    """Matriks desain: intersep, tren linear, harmonik mingguan & tahunan."""
    t = np.asarray(t, dtype=float)
    cols = [np.ones_like(t), t]
    for k in range(1, 4):
        cols.append(np.sin(2 * np.pi * k * t / 7.0))
        cols.append(np.cos(2 * np.pi * k * t / 7.0))
    for k in range(1, 4):
        cols.append(np.sin(2 * np.pi * k * t / 365.25))
        cols.append(np.cos(2 * np.pi * k * t / 365.25))
    return np.column_stack(cols)


def fit_prophet(df: pd.DataFrame) -> Prophet:
    model = Prophet(
        daily_seasonality=False,
        weekly_seasonality=True,
        yearly_seasonality=True,
        seasonality_mode="additive",
        interval_width=INTERVAL_WIDTH,
    )
    model.fit(df[["ds", "y"]])
    return model


def seasonal_naive_forecast(history, horizon):
    """Ramalan rekursif seasonal naive periode 7."""
    history = list(np.asarray(history, dtype=float))
    preds = []
    for h in range(horizon):
        idx = len(history) + h - SEASONAL_PERIOD
        if idx < len(history):
            preds.append(history[idx])
        else:
            preds.append(preds[idx - len(history)])
    return np.asarray(preds)


def seasonal_naive_fitted(values):
    s = pd.Series(np.asarray(values, dtype=float))
    return s.shift(SEASONAL_PERIOD).to_numpy()


# ---------------------------------------------------------------------------
# 1. Muat & agregasi harian
# ---------------------------------------------------------------------------
def load_daily() -> pd.DataFrame:
    if not EXCEL_PATH.exists():
        raise FileNotFoundError(f"Dataset tidak ditemukan: {EXCEL_PATH}")

    raw = pd.read_excel(
        EXCEL_PATH, sheet_name="data", usecols=["datetime", "nat_demand"]
    )
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

    daily = (
        grouped[grouped["hours"] == 24]
        .rename(columns={"date": "ds", "total": "y"})
        .sort_values("ds")
        .reset_index(drop=True)
    )
    daily["ds"] = pd.to_datetime(daily["ds"])
    return daily


# ---------------------------------------------------------------------------
# 2. Latih & evaluasi
# ---------------------------------------------------------------------------
def main() -> None:
    daily = load_daily()
    n = len(daily)
    if n <= TEST_DAYS + SEASONAL_PERIOD:
        raise ValueError("Data terlalu pendek untuk split train/test.")

    train = daily.iloc[:-TEST_DAYS].copy().reset_index(drop=True)
    test = daily.iloc[-TEST_DAYS:].copy().reset_index(drop=True)
    n_train = len(train)

    print(f"Total hari bersih : {n}")
    print(f"Train             : {train['ds'].min().date()} -> {train['ds'].max().date()} ({n_train})")
    print(f"Test              : {test['ds'].min().date()} -> {test['ds'].max().date()} ({len(test)})")

    future_dates = pd.date_range(
        daily["ds"].max() + pd.Timedelta(days=1), periods=FUTURE_DAYS, freq="D"
    )
    print(f"Future            : {future_dates[0].date()} -> {future_dates[-1].date()} ({len(future_dates)})")

    y_train = train["y"].to_numpy(dtype=float)
    y_test = test["y"].to_numpy(dtype=float)
    y_all = daily["y"].to_numpy(dtype=float)

    forecasts = {}
    fitted = {}
    metrics = {}

    # ---------------------------- Model A: Prophet -------------------------
    print("\n[Model A] Melatih Prophet pada train ...")
    prophet_train = fit_prophet(train)
    fc_train = prophet_train.predict(train[["ds"]])
    fc_test = prophet_train.predict(test[["ds"]])

    print("[Model A] Refit Prophet pada seluruh data ...")
    prophet_full = fit_prophet(daily)
    fc_future = prophet_full.predict(pd.DataFrame({"ds": future_dates}))

    with open(MODEL_PATH, "wb") as fh:
        pickle.dump(prophet_full, fh)
    print(f"[Model A] Model disimpan -> {MODEL_PATH.name}")

    forecasts["Model_A"] = {
        "test": fc_test[["yhat", "yhat_lower", "yhat_upper"]].to_numpy(),
        "future": fc_future[["yhat", "yhat_lower", "yhat_upper"]].to_numpy(),
    }
    fitted["Model_A"] = {
        "train": fc_train[["yhat", "yhat_lower", "yhat_upper"]].to_numpy(),
    }
    metrics["Model_A"] = metric_block(
        y_test,
        fc_test["yhat"].to_numpy(),
        (train["y"].to_numpy() - fc_train["yhat"].to_numpy()),
        n_params=25,
    )

    # ------------------- Model B: Seasonal Naive 7 hari -------------------
    print("\n[Model B] Seasonal Naive 7 hari ...")
    naive_fit_train = seasonal_naive_fitted(y_train)
    # Evaluasi out-of-sample: ulangi siklus 7 hari terakhir dari train.
    naive_test = seasonal_naive_forecast(y_train, TEST_DAYS)
    naive_future = seasonal_naive_forecast(y_all, FUTURE_DAYS)

    lo_b, hi_b = residual_bounds(y_train - naive_fit_train)
    forecasts["Model_B"] = {
        "test": np.column_stack([naive_test, naive_test + lo_b, naive_test + hi_b]),
        "future": np.column_stack([naive_future, naive_future + lo_b, naive_future + hi_b]),
    }
    fitted["Model_B"] = {
        "train": np.column_stack(
            [naive_fit_train, naive_fit_train + lo_b, naive_fit_train + hi_b]
        ),
    }
    metrics["Model_B"] = metric_block(
        y_test, naive_test, y_train - naive_fit_train, n_params=1
    )

    # --------------------------- Model C: Holt-Winters --------------------
    print("\n[Model C] Holt-Winters (trend + musiman aditif, 7 hari) ...")
    hw_train = ExponentialSmoothing(
        y_train,
        trend="add",
        damped_trend=False,
        seasonal="add",
        seasonal_periods=SEASONAL_PERIOD,
        initialization_method="estimated",
    ).fit(optimized=True)
    hw_fit_train = np.asarray(hw_train.fittedvalues, dtype=float)
    hw_test = np.asarray(hw_train.forecast(TEST_DAYS), dtype=float)

    hw_full = ExponentialSmoothing(
        y_all,
        trend="add",
        damped_trend=False,
        seasonal="add",
        seasonal_periods=SEASONAL_PERIOD,
        initialization_method="estimated",
    ).fit(optimized=True)
    hw_future = np.asarray(hw_full.forecast(FUTURE_DAYS), dtype=float)

    lo_c, hi_c = residual_bounds(y_train - hw_fit_train)
    forecasts["Model_C"] = {
        "test": np.column_stack([hw_test, hw_test + lo_c, hw_test + hi_c]),
        "future": np.column_stack([hw_future, hw_future + lo_c, hw_future + hi_c]),
    }
    fitted["Model_C"] = {
        "train": np.column_stack([hw_fit_train, hw_fit_train + lo_c, hw_fit_train + hi_c]),
    }
    metrics["Model_C"] = metric_block(
        y_test, hw_test, y_train - hw_fit_train, n_params=9
    )

    # -------------------- Model D: OLS trend + harmonik -------------------
    print("\n[Model D] OLS trend linear + harmonik mingguan/tahunan ...")
    t_train = np.arange(n_train, dtype=float)
    X_train = harmonic_design(t_train)
    beta, *_ = np.linalg.lstsq(X_train, y_train, rcond=None)
    ols_fit_train = X_train @ beta

    t_test = np.arange(n_train, n_train + TEST_DAYS, dtype=float)
    ols_test = harmonic_design(t_test) @ beta

    t_all = np.arange(n, dtype=float)
    X_all = harmonic_design(t_all)
    beta_full, *_ = np.linalg.lstsq(X_all, y_all, rcond=None)
    t_future = np.arange(n, n + FUTURE_DAYS, dtype=float)
    ols_future = harmonic_design(t_future) @ beta_full

    lo_d, hi_d = residual_bounds(y_train - ols_fit_train)
    forecasts["Model_D"] = {
        "test": np.column_stack([ols_test, ols_test + lo_d, ols_test + hi_d]),
        "future": np.column_stack([ols_future, ols_future + lo_d, ols_future + hi_d]),
    }
    fitted["Model_D"] = {
        "train": np.column_stack([ols_fit_train, ols_fit_train + lo_d, ols_fit_train + hi_d]),
    }
    metrics["Model_D"] = metric_block(
        y_test, ols_test, y_train - ols_fit_train, n_params=X_train.shape[1]
    )

    # ---------------------------------------------------------------------
    # 3. Susun payload JSON
    # ---------------------------------------------------------------------
    best_model = min(metrics, key=lambda key: metrics[key]["rmse"])
    print("\nRMSE test:")
    for key in metrics:
        print(f"  {key} {MODEL_META[key]['label']:<24} {metrics[key]['rmse']:,.2f}")
    print(f"Model terbaik: {best_model} ({MODEL_META[best_model]['label']})")

    series = []
    for i, row in daily.iterrows():
        split = "train" if i < n_train else "test"
        item = {
            "date": row["ds"].strftime("%Y-%m-%d"),
            "split": split,
            "actual": clean_number(row["y"]),
            "forecasts": {},
        }
        for key in MODEL_META:
            if split == "train":
                vals = fitted[key]["train"][i]
            else:
                vals = forecasts[key]["test"][i - n_train]
            item["forecasts"][key] = {
                "mean": clean_number(vals[0]),
                "lower_95": clean_number(vals[1]),
                "upper_95": clean_number(vals[2]),
            }
        series.append(item)

    for j, ds in enumerate(future_dates):
        item = {
            "date": ds.strftime("%Y-%m-%d"),
            "split": "future",
            "actual": None,
            "forecasts": {},
        }
        for key in MODEL_META:
            vals = forecasts[key]["future"][j]
            item["forecasts"][key] = {
                "mean": clean_number(vals[0]),
                "lower_95": clean_number(vals[1]),
                "upper_95": clean_number(vals[2]),
            }
        series.append(item)

    payload = {
        "metadata": {
            "title": "Dashboard Peramalan Beban Listrik Harian Panama",
            "subtitle": "Evaluasi dan proyeksi 30 hari ke depan · Prophet vs Seasonal Naive vs Holt-Winters vs OLS Harmonik",
            "unit": "MWh",
            "last_updated": pd.Timestamp.today().strftime("%Y-%m-%d"),
            "forecast_start_date": future_dates[0].strftime("%Y-%m-%d"),
            "data_start_date": daily["ds"].min().strftime("%Y-%m-%d"),
            "data_end_date": daily["ds"].max().strftime("%Y-%m-%d"),
            "test_start_date": test["ds"].min().strftime("%Y-%m-%d"),
            "test_end_date": test["ds"].max().strftime("%Y-%m-%d"),
            "train_days": int(n_train),
            "test_days": int(len(test)),
            "future_days": int(len(future_dates)),
            "total_days": int(n),
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
        json.dump(payload, fh, ensure_ascii=False, allow_nan=False, separators=(",", ":"))

    with open(RESULTS_PATH, "wb") as fh:
        pickle.dump(
            {
                "test_df": test.assign(
                    yhat=fc_test["yhat"].to_numpy(),
                    yhat_lower=fc_test["yhat_lower"].to_numpy(),
                    yhat_upper=fc_test["yhat_upper"].to_numpy(),
                ),
                "metrics": metrics,
                "payload": payload,
                "best_model": best_model,
            },
            fh,
        )

    size_kb = OUT_JSON.stat().st_size / 1024
    print(f"\nJSON  -> {OUT_JSON} ({size_kb:,.1f} KB, {len(series)} titik)")
    print(f"PKL   -> {RESULTS_PATH.name}")
    print("Selesai.")


if __name__ == "__main__":
    main()
