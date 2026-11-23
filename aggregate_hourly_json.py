"""
aggregate_hourly_json.py
========================
Ubah ``docs/data/hourly_results.json`` dari data PER JAM penuh (48.744 titik)
menjadi tampilan AGREGAT HARIAN (2.031 titik) supaya ringan dimuat.

Per hari, untuk setiap kolom dihitung:
- ``actual``       = rata-rata nilai per jam (min/maks disimpan di
                     ``actual_min`` / ``actual_max`` untuk pita grafik),
- ``forecasts[*]`` = rata-rata prediksi, batas CI bawah, dan batas CI atas
                     per jam (model tetap sama, metrik tidak diubah).

Data latih dan artefak model TIDAK berubah — ini hanya mengubah isi JSON
yang dikonsumsi dashboard. Schema tetap kompatibel dengan mode harian.

Jalankan:  python aggregate_hourly_json.py
(untuk regenerasi penuh: jalankan ``rebuild_hourly_json.py`` lebih dulu)
"""

from __future__ import annotations

import json
import math
from pathlib import Path

from build_hourly import OUT_JSON

SUBTITLE = (
    "Evaluasi per jam (metrik) · tampilan agregat harian: rata-rata, minimum, "
    "maksimum · proyeksi 30 hari · Prophet vs Seasonal Naive vs Holt-Winters "
    "vs OLS Harmonik vs GRU vs LSTM vs CNN-1D"
)


def finite_mean(values):
    vals = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    return sum(vals) / len(vals) if vals else None


def aggregate_series(series):
    """Kelompokkan baris per jam menjadi satu baris per tanggal."""
    groups = {}
    for row in series:
        day = row["date"][:10]
        groups.setdefault(day, []).append(row)

    out = []
    for day in sorted(groups):
        rows = groups[day]
        splits = {r["split"] for r in rows}
        if len(splits) != 1:
            raise ValueError(f"Tanggal {day} memiliki lebih dari satu split: {splits}")
        split = next(iter(splits))

        actuals = [r.get("actual") for r in rows]
        finite_actual = [
            float(v) for v in actuals if v is not None and math.isfinite(float(v))
        ]
        forecasts = {}
        for model in rows[0]["forecasts"]:
            block = [r["forecasts"][model] for r in rows]
            forecasts[model] = {
                field: finite_mean([f.get(field) for f in block])
                for field in ("mean", "lower_95", "upper_95")
            }

        item = {
            "date": day,
            "split": split,
            "actual": finite_mean(actuals),
            "actual_min": min(finite_actual) if finite_actual else None,
            "actual_max": max(finite_actual) if finite_actual else None,
            "forecasts": forecasts,
        }
        out.append(item)
    return out


def main() -> None:
    with open(OUT_JSON, encoding="utf-8") as fh:
        payload = json.load(fh)

    md = payload["metadata"]
    if md.get("aggregated_from_hourly"):
        print("JSON sudah berupa agregat harian — tidak ada yang diubah.")
        return
    if md.get("granularity") != "hourly":
        raise ValueError("Sumber bukan JSON per jam.")

    dates = [r["date"] for r in payload["series"]]
    if not dates or " " not in dates[0]:
        raise ValueError(
            "JSON per jam penuh tidak ditemukan. "
            "Jalankan `python rebuild_hourly_json.py` lebih dulu."
        )

    n_before = len(dates)
    payload["series"] = aggregate_series(payload["series"])

    md["granularity"] = "daily"
    md["source_granularity"] = "hourly"
    md["aggregated_from_hourly"] = True
    md["aggregation"] = "mean/min/max per day"
    md["source_hours"] = n_before
    md["subtitle"] = SUBTITLE

    with open(OUT_JSON, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, allow_nan=False,
                  separators=(",", ":"))

    print(f"Agregasi selesai: {n_before:,} titik per jam -> "
          f"{len(payload['series']):,} titik harian")
    print(f"JSON -> {OUT_JSON.name} ({OUT_JSON.stat().st_size:,} bytes)")
    print("Model, metrik, dan artefak tetap memakai seluruh data per jam.")


if __name__ == "__main__":
    main()
