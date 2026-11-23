"""
build_dl_models.py
==================
Model LSTM dan CNN-1D tambahan untuk dashboard peramalan beban listrik
harian Panama.

Diadaptasi dari ``GRU_for_Panama_National_Demand.ipynb`` bagian LSTM
(cell 39-69) dan CNN-1D (cell 70-110):

- LSTM : LSTM(64) -> Dropout(0.2) -> Dense(32, relu) -> Dense(1)
- CNN-1D: Conv1D(64, k=3) -> MaxPooling1D(2) -> Conv1D(32, k=3)
          -> Flatten -> Dense(64, relu) -> Dropout(0.2) -> Dense(1)

Keduanya pakai Adam(1e-3), loss MSE, metric MAE, EarlyStopping
patience 10 restore_best_weights, 20 epoch, batch 32. CNN-1D juga
memakai ReduceLROnPlateau(factor=0.5, patience=5, min_lr=1e-6)
dan shuffle=False sesuai notebook.

Adaptasi ke dashboard mengikuti build_gru.py (fungsi dipakai ulang):
- agregasi harian 2.001 hari lengkap, split 1971 / 30 / 30,
- forecast 30 hari rekursif,
- interval 95% dari kuantil residual empiris training.

Output:
- lstm_model.keras / lstm_scalers.pkl / lstm_predictions.pkl  -> Model_F
- cnn1d_model.keras / cnn1d_scalers.pkl / cnn1d_predictions.pkl -> Model_G
- docs/data/forecast_results.json ditambah entri Model_F & Model_G

Jalankan:  python build_dl_models.py
"""

from __future__ import annotations

import json
import pickle

import numpy as np
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import (
    Conv1D,
    Dense,
    Dropout,
    Flatten,
    Input,
    LSTM,
    MaxPooling1D,
)
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

from build_gru import (
    BATCH_SIZE,
    CONTINUOUS,
    BINARY,
    EPOCHS,
    FUTURE_DAYS,
    LOOKBACK,
    OUT_JSON,
    PATIENCE,
    ROOT,
    SEED,
    TEST_DAYS,
    VALID_DAYS,
    build_matrix,
    clean_number,
    create_sequences,
    create_sequences_with_history,
    load_daily,
    metric_block,
    recursive_forecast,
    residual_bounds,
    set_seed,
)

# ---------------------------------------------------------------------------
# Konfigurasi model
# ---------------------------------------------------------------------------
MODELS = {
    "Model_F": {
        "label": "LSTM",
        "short": "LSTM",
        "prefix": "lstm",
        "units": 64,
        "dropout": 0.2,
        "dense": 32,
        "lr": 0.001,
        "shuffle": True,
        "reduce_lr": False,
        "builder": "lstm",
    },
    "Model_G": {
        "label": "CNN-1D",
        "short": "CNN-1D",
        "prefix": "cnn1d",
        "filters1": 64,
        "filters2": 32,
        "kernel": 3,
        "pool": 2,
        "dense": 64,
        "dropout": 0.2,
        "lr": 0.001,
        "shuffle": False,
        "reduce_lr": True,
        "builder": "cnn1d",
    },
}

SUBTITLE = (
    "Evaluasi dan proyeksi 30 hari ke depan · Prophet vs Seasonal Naive vs "
    "Holt-Winters vs OLS Harmonik vs GRU vs LSTM vs CNN-1D"
)


# ---------------------------------------------------------------------------
# Arsitektur
# ---------------------------------------------------------------------------
def build_lstm(input_shape, cfg):
    model = Sequential([
        Input(shape=input_shape),
        LSTM(cfg["units"]),
        Dropout(cfg["dropout"]),
        Dense(cfg["dense"], activation="relu"),
        Dense(1),
    ])
    model.compile(
        optimizer=Adam(learning_rate=cfg["lr"]),
        loss="mse",
        metrics=["mae"],
    )
    return model


def build_cnn1d(input_shape, cfg):
    model = Sequential([
        Input(shape=input_shape),
        Conv1D(cfg["filters1"], cfg["kernel"], activation="relu"),
        MaxPooling1D(pool_size=cfg["pool"]),
        Conv1D(cfg["filters2"], cfg["kernel"], activation="relu"),
        Flatten(),
        Dense(cfg["dense"], activation="relu"),
        Dropout(cfg["dropout"]),
        Dense(1),
    ])
    model.compile(
        optimizer=Adam(learning_rate=cfg["lr"]),
        loss="mse",
        metrics=["mae"],
    )
    return model


BUILDERS = {"lstm": build_lstm, "cnn1d": build_cnn1d}


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def fit_one(builder, cfg, X, y_scaled, valid_from, valid_to, tag):
    X_fit, y_fit = X[:valid_from], y_scaled[:valid_from]
    X_val, y_val = X[valid_from:valid_to], y_scaled[valid_from:valid_to]

    X_tr_seq, y_tr_seq = create_sequences(X_fit, y_fit, LOOKBACK)
    X_va_seq, y_va_seq = create_sequences_with_history(
        X_fit, y_fit, X_val, y_val, LOOKBACK
    )

    print(f"  [{tag}] train sequences {X_tr_seq.shape}, val {X_va_seq.shape}")
    model = builder((LOOKBACK, X.shape[1]), cfg)

    callbacks = [
        EarlyStopping(
            monitor="val_loss", patience=PATIENCE, restore_best_weights=True
        )
    ]
    if cfg["reduce_lr"]:
        callbacks.append(
            ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6
            )
        )

    history = model.fit(
        X_tr_seq, y_tr_seq,
        validation_data=(X_va_seq, y_va_seq),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        shuffle=cfg["shuffle"],
        verbose=0,
    )
    print(f"  [{tag}] epochs dijalankan: {len(history.history['loss'])}")
    return model


def train_one(cfg, daily, n_train, n):
    label = cfg["label"]
    builder = BUILDERS[cfg["builder"]]
    y_all = daily["y"].to_numpy(dtype=float)

    set_seed(SEED)
    print(f"\n[{label}] Melatih model evaluasi ...")
    fs_eval, ts_eval = StandardScaler(), StandardScaler()
    X_eval = build_matrix(daily, fs_eval, ts_eval, fit_on=True)

    eval_model = fit_one(
        builder, cfg, X_eval, X_eval[:, 0],
        n_train - VALID_DAYS, n_train, f"{label} eval",
    )

    # Fitted in-sample pada train (teacher forcing), prediksi sekaligus.
    fitted_train = np.full(n_train, np.nan, dtype=float)
    fit_idx = np.arange(LOOKBACK, n_train)
    fit_seq = np.asarray([X_eval[i - LOOKBACK:i] for i in fit_idx])
    fit_pred = eval_model.predict(fit_seq, verbose=0).ravel()
    fitted_train[fit_idx] = ts_eval.inverse_transform(
        fit_pred.reshape(-1, 1)
    ).ravel()

    train_resid = y_all[:n_train] - fitted_train

    # Test: 30 hari ke depan, eksogen diketahui, y direkursif.
    X_hist_test = X_eval[n_train - LOOKBACK:n_train]
    exog_test = X_eval[n_train:n, 1:]
    pred_test = recursive_forecast(
        eval_model, X_hist_test, exog_test, ts_eval, TEST_DAYS
    )
    y_test = y_all[n_train:]

    lo, hi = residual_bounds(train_resid)
    metrics = metric_block(
        y_test, pred_test, train_resid, n_params=int(eval_model.count_params())
    )
    print(f"  [{label}] metrik test:")
    for key in ("rmse", "mae", "mape", "smape", "r2", "aic"):
        print(f"    {key:5s}: {metrics[key]:,.4f}" if metrics[key] is not None
              else f"    {key:5s}: —")
    print(f"  [{label}] interval residual 95%: {lo:,.1f} .. {hi:,.1f}")

    # Refit pada seluruh data bersih.
    set_seed(SEED)
    print(f"\n[{label}] Refit model pada seluruh data ...")
    fs_full, ts_full = StandardScaler(), StandardScaler()
    X_full = build_matrix(daily, fs_full, ts_full, fit_on=True)

    final_model = fit_one(
        builder, cfg, X_full, X_full[:, 0],
        n - VALID_DAYS, n, f"{label} final",
    )

    exog_cycle = X_full[n - 7:, 1:]
    exog_future = [exog_cycle[h % 7] for h in range(FUTURE_DAYS)]
    pred_future = recursive_forecast(
        final_model, X_full, exog_future, ts_full, FUTURE_DAYS
    )

    prefix = cfg["prefix"]
    model_path = ROOT / f"{prefix}_model.keras"
    scaler_path = ROOT / f"{prefix}_scalers.pkl"
    pred_path = ROOT / f"{prefix}_predictions.pkl"

    final_model.save(model_path)
    with open(scaler_path, "wb") as fh:
        pickle.dump(
            {
                "feature_scaler": fs_full,
                "target_scaler": ts_full,
                "continuous": CONTINUOUS,
                "binary": BINARY,
                "lookback": LOOKBACK,
                "architecture": cfg["builder"],
                "config": cfg,
            },
            fh,
        )
    with open(pred_path, "wb") as fh:
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
    print(f"  Model disimpan -> {model_path.name}")
    print(f"  Scaler disimpan -> {scaler_path.name}")

    return {
        "fitted_train": fitted_train,
        "pred_test": pred_test,
        "pred_future": pred_future,
        "metrics": metrics,
        "lower_offset": lo,
        "upper_offset": hi,
    }


def forecast_block(mean, lo, hi):
    if mean is None or not np.isfinite(mean):
        return {"mean": None, "lower_95": None, "upper_95": None}
    return {
        "mean": clean_number(mean),
        "lower_95": clean_number(mean + lo),
        "upper_95": clean_number(mean + hi),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
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

    results = {}
    for key, cfg in MODELS.items():
        results[key] = train_one(cfg, daily, n_train, n)

    # --------------------------- Update JSON -------------------------------
    md = payload["metadata"]
    for key, cfg in MODELS.items():
        md.setdefault("models", {})[key] = {
            "label": cfg["label"], "short": cfg["short"]
        }
    md["subtitle"] = SUBTITLE

    date_index = {d: i for i, d in enumerate(dates)}
    for key, cfg in MODELS.items():
        res = results[key]
        payload["metrics"][key] = {
            "label": cfg["label"],
            "short": cfg["short"],
            **{k: clean_number(v) for k, v in res["metrics"].items()},
        }

        lo, hi = res["lower_offset"], res["upper_offset"]
        fut_i, test_i = 0, 0
        for row in payload["series"]:
            if row["split"] == "future":
                value = res["pred_future"][fut_i]
                row["forecasts"][key] = forecast_block(value, lo, hi)
                fut_i += 1
            elif row["split"] == "test":
                value = res["pred_test"][test_i]
                row["forecasts"][key] = forecast_block(value, lo, hi)
                test_i += 1
            else:
                i = date_index[row["date"]]
                fitted = res["fitted_train"][i]
                row["forecasts"][key] = forecast_block(
                    fitted if np.isfinite(fitted) else None, lo, hi
                )
        if fut_i != FUTURE_DAYS or test_i != TEST_DAYS:
            raise ValueError(
                f"Jumlah prediksi {key} tidak sesuai: "
                f"future={fut_i}, test={test_i}"
            )

    md["best_model"] = min(
        payload["metrics"], key=lambda k: payload["metrics"][k]["rmse"]
    )

    with open(OUT_JSON, "w", encoding="utf-8") as fh:
        json.dump(
            payload, fh, ensure_ascii=False, allow_nan=False,
            separators=(",", ":"),
        )

    print(f"\nJSON diperbarui -> {OUT_JSON.relative_to(ROOT)}")
    print("Metrik semua model:")
    for key, m in payload["metrics"].items():
        print(f"  {key} {m['short']:16s} RMSE {m['rmse']:>12,.2f}")
    print(f"Model terbaik   : {md['best_model']}")
    print(f"Ukuran JSON     : {OUT_JSON.stat().st_size:,} bytes")


if __name__ == "__main__":
    main()
