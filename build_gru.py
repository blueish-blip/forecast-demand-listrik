"""
build_gru.py
============
Model GRU tambahan untuk dashboard peramalan beban listrik harian Panama.

Diadaptasi dari ``GRU_for_Panama_National_Demand.ipynb``:
arsitektur GRU(64) -> Dropout(0.2) -> Dense(1), Adam(1e-3), MSE,
lookback 24, early stopping patience 10.

Perbedaan adaptasi (agar sejajar dengan 4 model lain di dashboard):
- data per jam diagregasi menjadi harian (hanya hari lengkap 24 jam),
- split train/test mengikuti build_data.py (1971 train / 30 test / 30 future),
- forecast 30 hari dilakukan rekursif (prediksi sebelumnya diumpankan kembali),
- interval 95% memakai kuantil residual empiris training.

Output:
- gru_model.keras        model final (refit pada seluruh data bersih)
- gru_scalers.pkl        scaler + konfigurasi fitur
- gru_predictions.pkl    prediksi & metrik GRU
- docs/data/forecast_results.json  ditambah entri Model_E

Jalankan:  python build_gru.py
"""

from __future__ import annotations

import json
import pickle
import random
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Dropout, GRU
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

# ---------------------------------------------------------------------------
# Konfigurasi
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
EXCEL_PATH = Path(
    r"C:\Users\hhfzsn\Downloads\Panama_beban_listrik_per_jam_2015-2020_.xlsx"
)
OUT_JSON = ROOT / "docs" / "data" / "forecast_results.json"
MODEL_PATH = ROOT / "gru_model.keras"
SCALER_PATH = ROOT / "gru_scalers.pkl"
PRED_PATH = ROOT / "gru_predictions.pkl"

MODEL_KEY = "Model_E"
MODEL_META = {"label": "GRU", "short": "GRU"}

CONTINUOUS = [
    "T2M_toc", "QV2M_toc", "TQL_toc", "W2M_toc",
    "T2M_san", "QV2M_san", "TQL_san", "W2M_san",
    "T2M_dav", "QV2M_dav", "TQL_dav", "W2M_dav",
]
BINARY = ["holiday", "school"]
LOOKBACK = 24
UNITS = 64
DROPOUT = 0.2
LEARNING_RATE = 0.001
EPOCHS = 20
BATCH_SIZE = 32
PATIENCE = 10
SEED = 42
TEST_DAYS = 30
FUTURE_DAYS = 30
VALID_DAYS = 30
INTERVAL_LEVEL = 0.95


# ---------------------------------------------------------------------------
# Utilitas
# ---------------------------------------------------------------------------
def clean_number(value):
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
        "rmse": rmse, "mae": mae, "mape": mape,
        "smape": smape, "r2": r2, "aic": aic,
    }


def residual_bounds(resid, level=INTERVAL_LEVEL):
    resid = np.asarray(resid, dtype=float)
    resid = resid[np.isfinite(resid)]
    alpha = (1.0 - level) / 2.0
    lo, hi = np.quantile(resid, [alpha, 1.0 - alpha])
    return float(lo), float(hi)


def set_seed(seed=SEED):
    np.random.seed(seed)
    random.seed(seed)
    tf.random.set_seed(seed)


def load_daily():
    """Agregasi harian identik dengan build_data.py, lengkap dengan fitur GRU."""
    if not EXCEL_PATH.exists():
        raise FileNotFoundError(f"Dataset tidak ditemukan: {EXCEL_PATH}")

    usecols = ["datetime", "nat_demand"] + CONTINUOUS + BINARY
    raw = pd.read_excel(EXCEL_PATH, sheet_name="data", usecols=usecols)
    raw["datetime"] = pd.to_datetime(raw["datetime"])
    raw["date"] = raw["datetime"].dt.normalize()

    hours = raw.groupby("date")["nat_demand"].count()
    complete = hours[hours == 24].index

    agg = {c: "mean" for c in CONTINUOUS}
    agg.update({c: "max" for c in BINARY})
    agg["nat_demand"] = "sum"

    daily = (
        raw[raw["date"].isin(complete)]
        .groupby("date")
        .agg(agg)
        .rename(columns={"nat_demand": "y", "date": "ds"})
        .reset_index()
        .sort_values("date")
        .reset_index(drop=True)
    )
    daily["ds"] = pd.to_datetime(daily["date"])
    return daily


def build_matrix(daily, feature_scaler, target_scaler, fit_on=False):
    cont = daily[CONTINUOUS].to_numpy(dtype=float)
    if fit_on:
        cont_scaled = feature_scaler.fit_transform(cont)
    else:
        cont_scaled = feature_scaler.transform(cont)

    y = daily[["y"]].to_numpy(dtype=float)
    if fit_on:
        y_scaled = target_scaler.fit_transform(y)
    else:
        y_scaled = target_scaler.transform(y)

    binary = daily[BINARY].to_numpy(dtype=float)
    return np.concatenate([y_scaled, cont_scaled, binary], axis=1)


def create_sequences(X, y, lookback):
    X_seq, y_seq = [], []
    for i in range(lookback, len(X)):
        X_seq.append(X[i - lookback:i])
        y_seq.append(y[i])
    return np.asarray(X_seq), np.asarray(y_seq)


def create_sequences_with_history(X_hist, y_hist, X_cur, y_cur, lookback):
    X_all = np.concatenate([X_hist, X_cur], axis=0)
    y_all = np.concatenate([y_hist, y_cur], axis=0)
    X_seq, y_seq = [], []
    for i in range(len(X_hist), len(X_all)):
        X_seq.append(X_all[i - lookback:i])
        y_seq.append(y_all[i])
    return np.asarray(X_seq), np.asarray(y_seq)


def build_model(input_shape):
    model = Sequential([
        GRU(UNITS, input_shape=input_shape, return_sequences=False),
        Dropout(DROPOUT),
        Dense(1),
    ])
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss="mse",
        metrics=["mae"],
    )
    return model


def fit_model(X, y_scaled, valid_from, valid_to, tag):
    """Latih satu model; window validasi adalah [valid_from, valid_to)."""
    X_fit, y_fit = X[:valid_from], y_scaled[:valid_from]
    X_val, y_val = X[valid_from:valid_to], y_scaled[valid_from:valid_to]

    X_tr_seq, y_tr_seq = create_sequences(X_fit, y_fit, LOOKBACK)
    X_va_seq, y_va_seq = create_sequences_with_history(
        X_fit, y_fit, X_val, y_val, LOOKBACK
    )

    print(f"  [{tag}] train sequences {X_tr_seq.shape}, val {X_va_seq.shape}")
    model = build_model((LOOKBACK, X.shape[1]))
    early_stop = EarlyStopping(
        monitor="val_loss", patience=PATIENCE, restore_best_weights=True
    )
    history = model.fit(
        X_tr_seq, y_tr_seq,
        validation_data=(X_va_seq, y_va_seq),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[early_stop],
        verbose=0,
    )
    print(f"  [{tag}] epochs dijalankan: {len(history.history['loss'])}")
    return model


def inverse_y(scaler, value):
    return float(
        scaler.inverse_transform(np.asarray([[value]], dtype=float))[0, 0]
    )


def recursive_forecast(model, X_history, exog_rows, target_scaler, horizon):
    """Forecast rekursif: prediksi y diumpankan kembali sebagai history."""
    history = [row.copy() for row in X_history]
    preds = []
    for h in range(horizon):
        seq = np.asarray(history[-LOOKBACK:], dtype=float)[None, :, :]
        pred_scaled = float(model.predict(seq, verbose=0)[0, 0])
        preds.append(inverse_y(target_scaler, pred_scaled))

        row = np.concatenate(
            [np.asarray([pred_scaled], dtype=float),
             np.asarray(exog_rows[h], dtype=float)]
        )
        history.append(row)
    return np.asarray(preds, dtype=float)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    set_seed()
    daily = load_daily()
    dates = daily["ds"].dt.strftime("%Y-%m-%d").tolist()
    n = len(daily)
    n_train = n - TEST_DAYS
    print(f"Total hari bersih : {n}")
    print(f"Train             : {dates[0]} -> {dates[n_train - 1]} ({n_train})")
    print(f"Test              : {dates[n_train]} -> {dates[-1]} ({TEST_DAYS})")
    print(f"Future            : {FUTURE_DAYS} hari")

    with open(OUT_JSON, encoding="utf-8") as fh:
        payload = json.load(fh)
    json_dates = [row["date"] for row in payload["series"]]
    if json_dates[:n] != dates:
        raise ValueError("Tanggal JSON tidak cocok dengan agregasi harian.")

    # ---------------------- Model evaluasi (train-only) --------------------
    print("\n[GRU] Melatih model evaluasi ...")
    fs_eval, ts_eval = StandardScaler(), StandardScaler()
    X_eval = build_matrix(daily, fs_eval, ts_eval, fit_on=True)
    y_scaled_eval = X_eval[:, 0]

    eval_model = fit_model(
        X_eval, y_scaled_eval, n_train - VALID_DAYS, n_train, "eval"
    )

    # Fitted in-sample pada train (teacher forcing), diprediksi sekaligus.
    fitted_train = np.full(n_train, np.nan, dtype=float)
    fit_idx = np.arange(LOOKBACK, n_train)
    fit_seq = np.asarray([X_eval[i - LOOKBACK:i] for i in fit_idx])
    fit_pred = eval_model.predict(fit_seq, verbose=0).ravel()
    fitted_train[fit_idx] = ts_eval.inverse_transform(
        fit_pred.reshape(-1, 1)
    ).ravel()

    y_train = daily["y"].to_numpy(dtype=float)[:n_train]
    train_resid = y_train - fitted_train

    # Test: 30 hari ke depan, fitur eksogen test diketahui, y rekursif.
    test_start = n_train - LOOKBACK
    X_hist_test = X_eval[test_start:n_train]
    exog_test = X_eval[n_train:n, 1:]
    pred_test = recursive_forecast(
        eval_model, X_hist_test, exog_test, ts_eval, TEST_DAYS
    )
    y_test = daily["y"].to_numpy(dtype=float)[n_train:]

    lo, hi = residual_bounds(train_resid)

    metrics = metric_block(
        y_test, pred_test, train_resid, n_params=int(eval_model.count_params())
    )
    print("  [GRU] metrik test:")
    for key in ("rmse", "mae", "mape", "smape", "r2", "aic"):
        print(f"    {key:5s}: {metrics[key]:,.4f}" if metrics[key] is not None
              else f"    {key:5s}: —")
    print(f"  [GRU] interval residual 95%: {lo:,.1f} .. {hi:,.1f}")

    # ---------------- Model final (refit seluruh data) ---------------------
    print("\n[GRU] Refit model pada seluruh data ...")
    fs_full, ts_full = StandardScaler(), StandardScaler()
    X_full = build_matrix(daily, fs_full, ts_full, fit_on=True)
    y_scaled_full = X_full[:, 0]

    final_model = fit_model(X_full, y_scaled_full, n - VALID_DAYS, n, "final")

    # Fitur eksogen future tidak tersedia: ulang siklus 7 hari terakhir.
    exog_cycle = X_full[n - 7:, 1:]
    exog_future = [exog_cycle[h % 7] for h in range(FUTURE_DAYS)]
    pred_future = recursive_forecast(
        final_model, X_full, exog_future, ts_full, FUTURE_DAYS
    )

    final_model.save(MODEL_PATH)
    with open(SCALER_PATH, "wb") as fh:
        pickle.dump(
            {
                "feature_scaler": fs_full,
                "target_scaler": ts_full,
                "continuous": CONTINUOUS,
                "binary": BINARY,
                "lookback": LOOKBACK,
                "units": UNITS,
                "dropout": DROPOUT,
            },
            fh,
        )
    print(f"  Model disimpan -> {MODEL_PATH.name}")
    print(f"  Scaler disimpan -> {SCALER_PATH.name}")

    # --------------------------- Update JSON -------------------------------
    md = payload["metadata"]
    md.setdefault("models", {})[MODEL_KEY] = MODEL_META
    md["subtitle"] = (
        "Evaluasi dan proyeksi 30 hari ke depan · Prophet vs Seasonal Naive vs "
        "Holt-Winters vs OLS Harmonik vs GRU"
    )

    payload["metrics"][MODEL_KEY] = {
        "label": MODEL_META["label"],
        "short": MODEL_META["short"],
        **{k: clean_number(v) for k, v in metrics.items()},
    }

    def block(mean, interval=True):
        if mean is None or not np.isfinite(mean):
            return {"mean": None, "lower_95": None, "upper_95": None}
        if not interval:
            return {"mean": clean_number(mean), "lower_95": None, "upper_95": None}
        return {
            "mean": clean_number(mean),
            "lower_95": clean_number(mean + lo),
            "upper_95": clean_number(mean + hi),
        }

    date_index = {d: i for i, d in enumerate(dates)}
    fut_i, test_i = 0, 0
    for row in payload["series"]:
        if row["split"] == "future":
            value = pred_future[fut_i]
            row["forecasts"][MODEL_KEY] = block(value)
            fut_i += 1
        elif row["split"] == "test":
            value = pred_test[test_i]
            row["forecasts"][MODEL_KEY] = block(value)
            test_i += 1
        else:
            i = date_index[row["date"]]
            row["forecasts"][MODEL_KEY] = block(
                fitted_train[i] if np.isfinite(fitted_train[i]) else None
            )
    if fut_i != FUTURE_DAYS or test_i != TEST_DAYS:
        raise ValueError(
            f"Jumlah prediksi tidak sesuai: future={fut_i}, test={test_i}"
        )

    # Model terbaik dihitung ulang dari semua metrik yang tersedia.
    md["best_model"] = min(
        payload["metrics"], key=lambda k: payload["metrics"][k]["rmse"]
    )

    with open(OUT_JSON, "w", encoding="utf-8") as fh:
        json.dump(
            payload, fh, ensure_ascii=False, allow_nan=False,
            separators=(",", ":"),
        )

    with open(PRED_PATH, "wb") as fh:
        pickle.dump(
            {
                "fitted_train": fitted_train,
                "pred_test": pred_test,
                "pred_future": pred_future,
                "metrics": metrics,
                "lower_offset": lo,
                "upper_offset": hi,
            },
            fh,
        )

    print(f"\nJSON diperbarui -> {OUT_JSON.relative_to(ROOT)}")
    print(f"Model terbaik   : {md['best_model']}")
    print(f"Ukuran JSON     : {OUT_JSON.stat().st_size:,} bytes")


if __name__ == "__main__":
    main()
