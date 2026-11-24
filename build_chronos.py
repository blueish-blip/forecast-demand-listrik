"""
Tambahkan Chronos-2 + Kalender sebagai Model_H ke dashboard.

Sumber:
  - choronos/Chronos2_Panama_Demand.ipynb   (metodologi & setup evaluasi)
  - choronos/data_harian_train_test.csv     (konteks + aktual test harian)
  - choronos/data_per_jam_train_test.csv    (konteks + aktual test per jam)
  - choronos/chronos2_forecast_harian.csv   (forecast 30 hari ke depan)
  - choronos/chronos2_forecast_per_jam.csv  (forecast 720 jam ke depan)

Model dijalankan zero-shot (tanpa training). Evaluasi memakai setup yang sama
dengan notebook: test 30 hari terakhir / 720 jam terakhir, forecast multi-step,
varian "+ kalender" (holiday, school, Holiday_ID sebagai known covariates).

Jalankan dengan venv yang berisi torch + chronos-forecasting:
    python build_chronos.py

Hasil inference test disimpan ke cache `chronos_test_cache.json` supaya
inkorporasi ulang tidak perlu menjalankan model lagi.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "choronos"
DATA = ROOT / "docs" / "data"
CACHE = ROOT / "chronos_test_cache.json"

MODEL_KEY = "Model_H"
MODEL_LABEL = "Chronos-2 + Kalender"
MODEL_SHORT = "Chronos-2"
MODEL_NAME = "amazon/chronos-2"
CONTEXT_LEN = 8192
QUANTILES = [0.025, 0.5, 0.975]
CALENDAR = ["holiday", "school", "Holiday_ID"]

INPUTS = {
    "daily": {
        "data": "data_harian_train_test.csv",
        "time": "tanggal",
        "future": "chronos2_forecast_harian.csv",
        "future_time": "datetime",
        "horizon": 30,
    },
    "hourly": {
        "data": "data_per_jam_train_test.csv",
        "time": "datetime",
        "future": "chronos2_forecast_per_jam.csv",
        "future_time": "datetime",
        "horizon": 720,
    },
}

SUBTITLES = {
    "forecast_results.json":
        "Evaluasi dan proyeksi 30 hari ke depan · Prophet vs Seasonal Naive vs Holt-Winters vs OLS Harmonik vs GRU vs LSTM vs CNN-1D vs Chronos-2",
    "hourly_results.json":
        "Evaluasi per jam (metrik) · tampilan agregat harian: rata-rata, minimum, maksimum · proyeksi 30 hari · Prophet vs Seasonal Naive vs Holt-Winters vs OLS Harmonik vs GRU vs LSTM vs CNN-1D vs Chronos-2",
}


def read_sep_csv(path: Path, **kwargs) -> pd.DataFrame:
    """CSV sumber memakai ';' sebagai delimiter dan ',' sebagai desimal."""
    return pd.read_csv(path, sep=";", decimal=",", **kwargs)


def find_quantile(pred: pd.DataFrame, q: float) -> str:
    for col in pred.columns:
        try:
            if abs(float(col) - q) < 1e-9:
                return col
        except (TypeError, ValueError):
            continue
    raise KeyError(f"Kolom kuantil {q} tidak ditemukan: {list(pred.columns)}")


def metrics_block(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    err = y_true - y_pred
    abs_err = np.abs(err)
    ss_res = float(np.sum(err**2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    return {
        "rmse": float(np.sqrt(np.mean(err**2))),
        "mae": float(np.mean(abs_err)),
        "mape": float(np.mean(abs_err / np.abs(y_true)) * 100.0),
        "smape": float(np.mean(2.0 * abs_err / (np.abs(y_true) + np.abs(y_pred))) * 100.0),
        "r2": 1.0 - ss_res / ss_tot if ss_tot > 0 else None,
        "aic": None,  # Chronos-2 zero-shot: tidak ada residual training untuk AIC.
    }


def run_inference() -> dict:
    """Jalankan Chronos-2 pada kedua resolusi dan kembalikan prediksi test."""
    import torch
    from chronos import Chronos2Pipeline

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Memuat {MODEL_NAME} pada {device} …")
    pipe = Chronos2Pipeline.from_pretrained(MODEL_NAME, device_map=device)

    out = {}
    for resolution, cfg in INPUTS.items():
        time_col = cfg["time"]
        df = read_sep_csv(SRC / cfg["data"], parse_dates=[time_col])
        df[time_col] = pd.to_datetime(df[time_col])
        df = df.sort_values(time_col).reset_index(drop=True)

        horizon = cfg["horizon"]
        ctx = df.iloc[:-horizon].copy()
        test = df.iloc[-horizon:].copy()
        ctx["y"] = ctx["nat_demand_bersih"].astype(float)
        ctx["Holiday_ID"] = ctx["Holiday_ID"].astype(str)

        context = ctx[["datetime" if time_col == "datetime" else "tanggal", "y", *CALENDAR]].copy()
        context = context.rename(columns={time_col: "datetime"}).iloc[-CONTEXT_LEN:].copy()
        context["item_id"] = "panama"
        future = test[[time_col, *CALENDAR]].copy().rename(columns={time_col: "datetime"})
        future["Holiday_ID"] = future["Holiday_ID"].astype(str)
        future["item_id"] = "panama"

        print(f"Prediksi {resolution}: {len(context)} titik konteks → {horizon} langkah …")
        pred = pipe.predict_df(
            context,
            future_df=future,
            prediction_length=horizon,
            quantile_levels=QUANTILES,
            id_column="item_id",
            timestamp_column="datetime",
            target="y",
            freq="D" if resolution == "daily" else "h",
        )
        med = pred[find_quantile(pred, 0.5)].to_numpy(dtype=float)
        lo = pred[find_quantile(pred, QUANTILES[0])].to_numpy(dtype=float)
        hi = pred[find_quantile(pred, QUANTILES[-1])].to_numpy(dtype=float)

        y_true = test["nat_demand"].to_numpy(dtype=float)
        out[resolution] = {
            "timestamps": test[time_col].dt.strftime("%Y-%m-%d %H:%M:%S").tolist(),
            "actual": y_true.tolist(),
            "mean": med.tolist(),
            "lower_95": lo.tolist(),
            "upper_95": hi.tolist(),
            "metrics": metrics_block(y_true, med),
        }
        print(f"  RMSE {out[resolution]['metrics']['rmse']:,.2f} · MAPE {out[resolution]['metrics']['mape']:.2f}%")

    return out


def load_test_predictions() -> dict:
    if CACHE.exists():
        cached = json.loads(CACHE.read_text(encoding="utf-8"))
        if all(k in cached for k in INPUTS):
            print(f"Menggunakan cache {CACHE.name}")
            return cached
    preds = run_inference()
    CACHE.write_text(json.dumps(preds, ensure_ascii=False), encoding="utf-8")
    print(f"Cache disimpan ke {CACHE.name}")
    return preds


def daily_map(timestamps: list[str], mean: list[float], lo: list[float], hi: list[float]) -> dict:
    """Rata-rata predicted mean/PI per tanggal (untuk JSON agregat harian)."""
    df = pd.DataFrame({"date": [str(t)[:10] for t in timestamps], "mean": mean, "lo": lo, "hi": hi})
    grouped = df.groupby("date", sort=True).mean()
    return {
        day: {
            "mean": float(row["mean"]),
            "lower_95": float(row["lo"]),
            "upper_95": float(row["hi"]),
        }
        for day, row in grouped.iterrows()
    }


def future_map(resolution: str) -> dict:
    cfg = INPUTS[resolution]
    df = pd.read_csv(SRC / cfg["future"])
    df["date"] = pd.to_datetime(df[cfg["future_time"]]).dt.strftime("%Y-%m-%d")
    return {
        row["date"]: {
            "mean": float(row["forecast"]),
            "lower_95": float(row["lower_95"]),
            "upper_95": float(row["upper_95"]),
        }
        for _, row in df.iterrows()
    }


def inject(filename: str, resolution: str, preds: dict) -> None:
    path = DATA / filename
    payload = json.loads(path.read_text(encoding="utf-8"))
    metadata = payload["metadata"]

    metadata.setdefault("models", {})[MODEL_KEY] = {"label": MODEL_LABEL, "short": MODEL_SHORT}
    metadata["subtitle"] = SUBTITLES[filename]
    metadata["last_updated"] = date.today().isoformat()

    test = preds[resolution]
    test_by_day = daily_map(test["timestamps"], test["mean"], test["lower_95"], test["upper_95"])
    fut_by_day = future_map(resolution)
    payload["metrics"][MODEL_KEY] = {
        "label": MODEL_LABEL,
        "short": MODEL_SHORT,
        **test["metrics"],
    }

    filled = 0
    for row in payload["series"]:
        day = row["date"]
        if day in test_by_day:
            forecast = test_by_day[day]
        elif day in fut_by_day:
            forecast = fut_by_day[day]
        else:
            forecast = {"mean": None, "lower_95": None, "upper_95": None}
        row["forecasts"][MODEL_KEY] = forecast
        filled += forecast["mean"] is not None

    best = min(payload["metrics"].items(), key=lambda kv: kv[1]["rmse"])[0]
    metadata["best_model"] = best

    path.write_text(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )
    print(f"{filename}: Model_H ditambahkan ({filled} titik), best_model = {best}")


def main() -> None:
    preds = load_test_predictions()
    inject("forecast_results.json", "daily", preds)
    inject("hourly_results.json", "hourly", preds)
    print("\nSelesai. Periksa kembali README.md dan docs/index.html untuk narasi 8 model.")


if __name__ == "__main__":
    main()
