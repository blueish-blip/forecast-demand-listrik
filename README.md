# Panama Electricity Load Forecasting

Dashboard SPA statis untuk evaluasi dan peramalan beban listrik Panama (`nat_demand`). Proyek ini membandingkan **Prophet**, tiga baseline time series, dan tiga model deep learning (**GRU**, **LSTM**, **CNN-1D**), lalu menyajikan hasilnya dalam dashboard yang siap di-deploy ke GitHub Pages. Tersedia dua resolusi: **Harian** (default) dan **Per Jam**, dipilih lewat kontrol **Resolusi**.

- **Target:** beban listrik nasional harian (`nat_demand`), satuan **MWh**
- **Periode data:** 2015-01-04 s/d 2020-06-26 (2.001 hari bersih)
- **Test set:** 30 hari terakhir (2020-05-28 s/d 2020-06-26)
- **Future forecast:** 30 hari (2020-06-27 s/d 2020-07-26)
- **Model terbaik (RMSE test):** Seasonal Naive 7 Hari

---

## Fitur Dashboard

- Header dengan judul, subjudul metodologi, periode data, dan `Last Updated`.
- Kontrol **Resolusi** (**Harian / Per Jam**), model (termasuk **Bandingkan Semua Model**), filter rentang (**Semua Data / Test Set / Future Forecast**), dan **Export CSV**.
- KPI grid 4 kolom: **RMSE, MAE, MAPE, sMAPE**, dengan subteks **R2** dan **AIC**.
- Grafik Plotly interaktif: data aktual, fitted values, forecast, pita **95% CI**, garis awal forecast, dan range slider.
- Panel residual (`Actual - Forecast`) dengan garis nol untuk melihat bias/heteroskedastisitas.
- Tabel proyeksi dengan pencarian tanggal, paginasi, dan format angka `id-ID`.
- Tema **Swiss Minimalist Dark + Pastel**, font Inter + JetBrains Mono, angka `tabular-nums`.
- **Fallback demo data** realistis bila JSON belum tersedia atau halaman dibuka langsung via `file://`.

---

## Hasil Evaluasi

Metrik dihitung pada 30 hari test (out-of-sample):

| Model | RMSE | MAE | MAPE | sMAPE | R2 | AIC (approx.) |
|---|---:|---:|---:|---:|---:|---:|
| **Prophet** (`Model_A`) | 2.527,55 | 2.310,93 | 8,84% | 8,40% | -3,3142 | 28.685,8 |
| **Seasonal Naive 7 Hari** (`Model_B`) | **1.148,72** | **913,16** | **3,52%** | **3,47%** | **0,1089** | 29.238,6 |
| **Holt-Winters** (`Model_C`) | 1.250,35 | 996,40 | 3,75% | 3,84% | -0,0558 | 27.717,0 |
| **OLS Trend + Harmonik** (`Model_D`) | 4.058,26 | 3.899,86 | 14,89% | 13,77% | -10,1220 | 29.006,6 |
| **GRU** (`Model_E`) | 1.305,95 | 1.142,23 | 4,39% | 4,33% | -0,1517 | 58.152,7 |
| **LSTM** (`Model_F`) | 1.417,96 | 1.122,66 | 4,26% | 4,27% | -0,3578 | 72.176,9 |
| **CNN-1D** (`Model_G`) | 1.529,22 | 1.337,43 | 5,12% | 4,99% | -0,5792 | 82.852,3 |

Catatan:

- **Seasonal Naive 7 Hari** adalah model terbaik pada jendela test COVID-era ini. Baseline ini mengulang siklus 7 hari terakhir secara rekursif.
- **Prophet** tetap menjadi model utama untuk interval prediksi 95% dan disimpan ke `prophet_model.pkl`.
- **GRU, LSTM, dan CNN-1D** diadaptasi dari `GRU_for_Panama_National_Demand.ipynb` (ketiga bagian: cell 0–38, 39–69, dan 70–110). Semua memakai lookback 24, 15 fitur, Adam 1e-3, dan early stopping patience 10. Model final disimpan berturut-turut ke `gru_model.keras`, `lstm_model.keras`, dan `cnn1d_model.keras` beserta scaler masing-masing.
- **AIC** dihitung dari residual training dengan pendekatan Gaussian `n * ln(RSS/n) + 2k`. Nilai AIC antar-kelas model tidak sepenuhnya sebanding; gunakan sebagai konteks tambahan, bukan satu-satunya dasar pemilihan.
- Baseline dan ketiga model deep learning memakai interval empiris dari kuantil residual training (2,5% - 97,5%).

### Hasil Evaluasi Mode Per Jam

Metrik dihitung pada 720 jam test (30 hari, 2020-05-28 s/d 2020-06-26):

| Model | RMSE | MAE | MAPE | R2 |
|---|---:|---:|---:|---:|
| **Prophet** (`Model_A`) | 109,10 | 86,81 | 8,09% | 0,1041 |
| **Seasonal Naive 7 Hari** (`Model_B`) | **72,33** | **56,91** | **5,21%** | **0,6063** |
| **Holt-Winters** (`Model_C`) | 114,81 | 96,45 | 8,98% | 0,0080 |
| **OLS Trend + Harmonik** (`Model_D`) | 194,34 | 170,50 | 15,60% | -1,8426 |
| **GRU** (`Model_E`) | 115,81 | 86,72 | 8,14% | -0,0095 |
| **LSTM** (`Model_F`) | 119,92 | 85,97 | 8,10% | -0,0824 |
| **CNN-1D** (`Model_G`) | 101,41 | 76,51 | 6,91% | 0,2259 |

Catatan mode per jam:

- Urutan peringkat mirip mode harian: **Seasonal Naive 168 jam** tetap terbaik, diikuti Prophet dan CNN-1D.
- Pada resolusi per jam, pola harian (24 jam) sangat dominan, sehingga CNN-1D unggul dibanding GRU/LSTM.
- Semua model memakai jendela test yang sama (720 jam) dan horizon 720 jam.

---

## Metodologi

1. **Agregasi harian**
   - Data per jam dijumlahkan menjadi total harian.
   - Hanya hari dengan **24 jam lengkap** yang dipakai.
   - Dua hari parsial dibuang: `2015-01-03` (23 jam) dan `2020-06-27` (1 jam).
   - Hasil: **2.001 hari bersih**.

2. **Split**
   - 1.971 hari pertama untuk training (2015-01-04 s/d 2020-05-27).
   - 30 hari terakhir untuk test (2020-05-28 s/d 2020-06-26).
   - 30 hari setelah data terakhir untuk future forecast (2020-06-27 s/d 2020-07-26).

3. **Model**
   - `Model_A`: Prophet (weekly + yearly seasonality, additive, interval 95%).
   - `Model_B`: Seasonal Naive 7 hari (siklus mingguan terakhir diulang).
   - `Model_C`: Holt-Winters (trend aditif + seasonal aditif, periode 7).
   - `Model_D`: OLS dengan trend linear + harmonik mingguan dan tahunan.
   - `Model_E`: GRU 64 unit + dropout 0,2.
   - `Model_F`: LSTM(64) -> Dropout(0,2) -> Dense(32, relu) -> Dense(1), early stopping patience 10.
   - `Model_G`: Conv1D(64, k=3) -> MaxPooling1D(2) -> Conv1D(32, k=3) -> Flatten -> Dense(64, relu) -> Dropout(0,2) -> Dense(1), dengan ReduceLROnPlateau dan `shuffle=False`.
   - Semua model deep learning memakai lookback 24 hari dan 15 fitur (`nat_demand`, 12 fitur cuaca, `holiday`, `school`). Fitur diskalakan dengan `StandardScaler`. Forecast 30 hari dilakukan rekursif; fitur eksogen future mengulang siklus 7 hari terakhir.

4. **Refit & simpan**
   - Setelah evaluasi, Prophet di-refit pada seluruh 2.001 hari bersih.
   - Model Prophet final disimpan ke `prophet_model.pkl`. Model deep learning final disimpan ke `gru_model.keras`, `lstm_model.keras`, dan `cnn1d_model.keras` + scaler masing-masing.
   - Dashboard tidak perlu retraining. Prediksi test, metrik, dan payload dashboard disimpan ke `forecast_results.pkl`, `gru_predictions.pkl`, `lstm_predictions.pkl`, dan `cnn1d_predictions.pkl`.

5. **Mode per jam (opsional)**
   - Pipeline terpisah `build_hourly.py` menghitung ulang ketujuh model pada resolusi 1 jam.
   - Hanya hari lengkap 24 jam yang dipakai → **48.024 jam** (2.001 hari).
   - Split: 47.304 jam train / 720 jam test (30 hari) / 720 jam future (30 hari).
   - Tampilan dashboard dibatasi 30 hari historis (720 jam) + test + future = **2.160 titik** agar JSON dan grafik tetap ringan.
   - Seasonal Naive memakai periode 168 jam, Holt-Winters `seasonal_periods=168`, Prophet memakai `daily_seasonality`, dan OLS menambah harmonik harian (periode 24 jam).
   - Deep learning memakai lookback 24 jam dengan arsitektur identik mode harian; fitur eksogen future mengulang siklus 168 jam terakhir.
   - Keluaran: `docs/data/hourly_results.json` + artefak `*_hourly.keras`, `hourly_scalers.pkl`, `hourly_predictions.pkl`.

---

## Struktur Proyek

```text
.
|-- docs/
|   |-- index.html                  # Dashboard SPA (HTML + Tailwind + Plotly)
|   `-- data/
|       |-- forecast_results.json   # Data mode harian (metadata, metrics, series)
|       `-- hourly_results.json     # Data mode per jam (2.160 titik)
|-- build_data.py                   # Pipeline utama: 4 model + ekspor JSON
|-- build_gru.py                    # Pipeline GRU (Model_E) + update JSON
|-- build_dl_models.py              # Pipeline LSTM (Model_F) + CNN-1D (Model_G)
|-- build_hourly.py                 # Pipeline mode per jam: 7 model + JSON
|-- gru_model.keras                 # Model GRU final (TensorFlow/Keras)
|-- gru_scalers.pkl                 # Scaler & konfigurasi fitur GRU
|-- gru_predictions.pkl             # Prediksi & metrik GRU
|-- lstm_model.keras                # Model LSTM final
|-- lstm_scalers.pkl                # Scaler & konfigurasi fitur LSTM
|-- lstm_predictions.pkl            # Prediksi & metrik LSTM
|-- cnn1d_model.keras               # Model CNN-1D final
|-- cnn1d_scalers.pkl               # Scaler & konfigurasi fitur CNN-1D
|-- cnn1d_predictions.pkl           # Prediksi & metrik CNN-1D
|-- prophet_hourly_model.pkl        # Model Prophet final mode per jam
|-- gru_hourly.keras / lstm_hourly.keras / cnn1d_hourly.keras  # Model DL per jam
|-- hourly_scalers.pkl              # Scaler ketiga model DL per jam
|-- hourly_predictions.pkl          # Prediksi & metrik per jam
|-- prophet_forecasting.ipynb       # Notebook EDA + Prophet
|-- prophet_model.pkl               # Model Prophet final (refit seluruh data)
|-- forecast_results.pkl            # Prediksi test, metrik, payload
|-- buat_dashboard.py               # Legacy: generator HTML standalone lama
`-- README.md
```

---

## Menjalankan Secara Lokal

Dashboard memuat data dengan `fetch('./data/forecast_results.json')` (harian) dan `fetch('./data/hourly_results.json')` (per jam). Karena `fetch` diblokir pada protokol `file://`, jalankan server statis:

```bash
python -m http.server 8000 --directory docs
```

Buka:

```text
http://localhost:8000
```

Jika JSON tidak ditemukan (misalnya 404), dashboard otomatis memakai **fallback demo data** dan menampilkan badge `Mode demo - data fallback`.

---

## Deploy ke GitHub Pages

1. Commit dan push folder `docs/` ke repository.
2. Buka **Settings -> Pages**.
3. Pilih **Source: Deploy from a branch**.
4. Pilih branch `main` dan folder **`/docs`**.
5. Simpan. Dashboard tersedia di:

```text
https://<username>.github.io/<nama-repo>/
```

Semua path data memakai relative path (`./data/forecast_results.json`), sehingga tetap bekerja pada sub-path repository. Library Tailwind dan Plotly dimuat dari CDN.

---

## Regenerasi Data & Model

Jalankan berurutan:

```bash
python build_data.py        # Model_A - Model_D + JSON awal
python build_gru.py         # Model_E (GRU)
python build_dl_models.py   # Model_F (LSTM) + Model_G (CNN-1D)
python build_hourly.py      # Mode per jam: ketujuh model + hourly_results.json
```

Keempat script akan:

- membaca dataset Excel dan membuang hari parsial,
- melatih model dengan split train/test yang identik,
- refit model final pada seluruh 2.001 hari bersih,
- menulis/memperbarui `docs/data/forecast_results.json` beserta artefak model dan prediksinya.

Jika lokasi dataset berbeda, ubah konstanta `EXCEL_PATH` di bagian atas `build_data.py` dan `build_gru.py` (`build_dl_models.py` memakai ulang konstanta dari `build_gru.py`).

---

## Skema JSON

```json
{
  "metadata": {
    "title": "Dashboard Peramalan Beban Listrik Harian Panama",
    "unit": "MWh",
    "last_updated": "2026-11-21",
    "forecast_start_date": "2020-06-27"
  },
  "metrics": {
    "Model_A": {
      "rmse": 2527.55,
      "mae": 2310.93,
      "mape": 8.84,
      "smape": 8.40,
      "r2": -3.3142,
      "aic": 28685.84
    }
  },
  "series": [
    {
      "date": "2020-06-27",
      "split": "future",
      "actual": null,
      "forecasts": {
        "Model_A": {
          "mean": 26688.23,
          "lower_95": 24043.97,
          "upper_95": 29454.71
        }
      }
    }
  ]
}
```

Nilai `split` adalah `train`, `test`, atau `future`. `actual` bernilai `null` untuk periode future.

Untuk mode per jam, skema identik dengan pengecualian:

- `metadata.granularity: "hourly"` (harian: `"daily"` atau tanpa field),
- `metadata` juga memuat `test_hours`, `future_hours`, `history_hours`,
- field `date` memakai format `YYYY-MM-DD HH:MM` (mis. `2020-05-28 00:00`),
- file: `docs/data/hourly_results.json`.

---

## Sumber Data

- File: `C:\Users\hhfzsn\Downloads\Panama_beban_listrik_per_jam_2015-2020_.xlsx`
- Sheet: `data`, `deskripsi`
- 48.048 baris per jam, 17 kolom
- Rentang: 2015-01-03 01:00 s/d 2020-06-27 00:00
- Kolom target: `nat_demand` (MWh)

---

## Tech Stack

- **Python 3.10**, pandas, numpy
- **Prophet 1.5**, statsmodels 0.15
- **HTML5**, Tailwind CSS (CDN)
- **Plotly.js 2.35.2** + locale `id`
- **Inter** dan **JetBrains Mono** (Google Fonts)

---

## Catatan Legacy

`buat_dashboard.py` adalah generator HTML standalone lama (export notebook). Pipeline yang dipakai saat ini adalah **`build_data.py`** + `docs/index.html`.
