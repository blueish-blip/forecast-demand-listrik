"""
rebuild_hourly_json.py
======================
Tulis ulang ``docs/data/hourly_results.json`` dengan SELURUH data historis
(47.304 jam) tanpa melatih ulang model deep learning.

Sumber nilai per model:
- Model_E/F/G : ``hourly_predictions.pkl`` (fitted, test, future, metrik, CI)
                 -> tidak ada training ulang sama sekali.
- Model_A     : refit Prophet pada train (MAP deterministik) + model final
                 yang sudah disimpan di ``prophet_hourly_model.pkl``.
- Model_B/D   : dihitung ulang (deterministik, instan).
- Model_C     : Holt-Winters dihitung ulang (deterministik).

Sebelum file ditimpa, seluruh metrik DAN seluruh prediksi test/future
diverifikasi identik (toleransi numerik sangat ketat) dengan JSON lama,
sehingga hasilnya konsisten dengan yang sudah terpasang di dashboard.

Jalankan:  python rebuild_hourly_json.py
"""

from __future__ import annotations

import json
import pickle

import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from build_gru import clean_number, metric_block, residual_bounds, set_seed
from build_hourly import (
    FUTURE_HOURS,
    HOUR_PERIOD,
    MODEL_META,
    OUT_JSON,
    PRED_PATH,
    PROPHET_PATH,
    SEED,
    SUBTITLE,
    TEST_HOURS,
    fit_prophet_hourly,
    harmonic_hourly,
    interval_stack,
    load_hourly,
    seasonal_naive_fitted,
    seasonal_naive_forecast,
    triple,
)

DL_KEYS = ("Model_E", "Model_F", "Model_G")


def same_number(a, b, rtol=1e-9, atol=1e-9):
    """Bandingkan dua angka (boleh None/NaN) secara numerik ketat."""
    if a is None and b is None:
        return True
    if a is None or b is None:
        return False
    a, b = float(a), float(b)
    if not (np.isfinite(a) and np.isfinite(b)):
        return bool(np.isnan(a) and np.isnan(b))
    return bool(np.isclose(a, b, rtol=rtol, atol=atol))


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

    with open(OUT_JSON, encoding="utf-8") as fh:
        old = json.load(fh)

    forecasts: dict = {}
    metrics: dict = {}

    # ---------------------------- Model A: Prophet -------------------------
    print("[Model A] Refit Prophet train (tanpa sampling ulang) ...")
    train_df = hourly.iloc[:n_train][["ds", "y"]]
    test_df = hourly.iloc[n_train:][["ds", "y"]]
    prophet_train = fit_prophet_hourly(train_df)
    fc_train = prophet_train.predict(train_df[["ds"]])
    fc_test = prophet_train.predict(test_df[["ds"]])

    with open(PROPHET_PATH, "rb") as fh:
        prophet_full = pickle.load(fh)
    fc_future = prophet_full.predict(pd.DataFrame({"ds": future_ts}))

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
    print("[Model B] Seasonal Naive 168 jam ...")
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
    print("[Model D] OLS trend + harmonik ...")
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

    # --------------- Model E/F/G: dari artefak (tanpa training) ------------
    with open(PRED_PATH, "rb") as fh:
        pred_store = pickle.load(fh)
    for key in DL_KEYS:
        print(f"[{MODEL_META[key]['label']}] Memuat prediksi tersimpan ...")
        p = pred_store[key]
        lo, hi = p["lower_offset"], p["upper_offset"]
        forecasts[key] = {
            "train": interval_stack(p["fitted_train"], lo, hi),
            "test": interval_stack(p["pred_test"], lo, hi),
            "future": interval_stack(p["pred_future"], lo, hi),
        }
        metrics[key] = p["metrics"]

    # ----------------------- Verifikasi vs JSON lama -----------------------
    print("\nVerifikasi terhadap JSON sebelumnya ...")
    worst = 0.0
    for key in MODEL_META:
        for field in ("rmse", "mae", "mape", "smape", "r2", "aic"):
            a = float(metrics[key][field])
            b = float(old["metrics"][key][field])
            if not np.isclose(a, b, rtol=1e-7, atol=1e-7):
                raise ValueError(
                    f"Metrik berbeda {key}.{field}: baru={a!r} lama={b!r}"
                )
            if np.isfinite(a) and np.isfinite(b) and b:
                worst = max(worst, abs(a - b) / max(abs(b), 1.0))

        # Prediksi test & future harus identik baris demi baris.
        old_rows = [r for r in old["series"] if r["split"] in ("test", "future")]
        for split in ("test", "future"):
            arr = forecasts[key][split]
            rows = [r for r in old_rows if r["split"] == split]
            if len(rows) != len(arr):
                raise ValueError(f"Jumlah baris {key}/{split} tidak cocok")
            for j, row in enumerate(rows):
                for idx, field in enumerate(("mean", "lower_95", "upper_95")):
                    if not same_number(arr[j][idx], row["forecasts"][key][field]):
                        raise ValueError(
                            f"Prediksi berbeda {key}/{split} "
                            f"{row['date']} {field}: "
                            f"{arr[j][idx]!r} vs {row['forecasts'][key][field]!r}"
                        )
    print(f"  Metrik & prediksi test/future identik (selisih relatif "
          f"maks {worst:.2e})")

    # --------------------- Susun payload (seluruh data) --------------------
    best_model = min(metrics, key=lambda key: metrics[key]["rmse"])
    print("\nRMSE test per jam:")
    for key in MODEL_META:
        print(f"  {key} {MODEL_META[key]['label']:<24} "
              f"{metrics[key]['rmse']:>12,.2f}")
    print(f"Model terbaik: {best_model} ({MODEL_META[best_model]['label']})")

    series = []
    for i in range(n_train):
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
            "forecasts": {
                k: triple(forecasts[k]["future"][j]) for k in MODEL_META
            },
        })

    if len(series) != n + FUTURE_HOURS:
        raise ValueError(f"Jumlah baris salah: {len(series)}")

    payload = {
        "metadata": {
            "title": "Dashboard Peramalan Beban Listrik Harian Panama",
            "subtitle": SUBTITLE,
            "unit": "MWh",
            "granularity": "hourly",
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
            "history_days": int(n_train // 24),
            "history_hours": int(n_train),
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

    with open(OUT_JSON, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, allow_nan=False,
                  separators=(",", ":"))

    print(f"\nJSON -> {OUT_JSON.relative_to(OUT_JSON.parents[2])} "
          f"({OUT_JSON.stat().st_size:,} bytes, {len(series)} titik)")
    print("Selesai (tanpa training ulang).")


if __name__ == "__main__":
    main()
