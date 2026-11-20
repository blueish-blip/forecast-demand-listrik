# 🇵🇦 Panama Electricity Load Forecasting with Facebook Prophet

## 📌 Deskripsi

Proyek ini merupakan implementasi **time series forecasting** untuk **beban listrik nasional (`nat_demand`)** di **Panama** menggunakan model **Facebook Prophet**, dengan data kinerja beban listrik pada periode **2015–2020**.

Alur kerja dalam notebook ini:
1. **Load data** dari file Excel (`Panama_beban_listrik_per_jam_2015-2020_.xlsx`)
2. **Exploratory Data Analysis (EDA)** & **pengecekan data** (shape, columns, dtypes, missing value, statistik dasar)
3. **Persiapan time series** — agregasi harian beban listrik (`nat_demand` → `ds` + `y`)
4. **Training model** Prophet (`daily_seasonality`, `weekly_seasonality`, `yearly_seasonality`)
5. **Forecast** ke depan (±30 hari)
6. **Simpan model** ke file `.pkl` agar tidak perlu retraining

---

## 📊 Dataset

| Item | Keterangan |
|---|---|
| File | `C:\Users\hhfzsn\Downloads\Panama_beban_listrik_per_jam_2015-2020_.xlsx` |
| Sheet | `data`, `deskripsi` |
| Shape | 48.048 baris × 17 kolom |
| Rentang Waktu | 2015-01-03 01:00 → 2020-06-27 00:00 (per jam) |
| Satu Kolom Penting | `nat_demand` — Beban listrik nasional (**MWh**) |

**Kolom Lain:**
- Variabel cuaca (suhu, kelembaban, presipitasi, kecepatan angin) di 3 lokasi: **Tocumen** (Kota Panama), **Santiago**, **David**
- `Holiday_ID`, `holiday`, `school` — indikator hari libur & periode sekolah

---

## 🔁 Alur Notebook (`prophet_forecasting.ipynb`)

```
EDA → Pengecekan Data → Persiapan TS → Training Prophet → Forecast → Simpan Model
```

---

## 💾 Model

- Model Prophet berhasil **di-run dan disave** ke `prophet_model.pkl`
- Model dapat **di-load kembali** tanpa perlu retrain → siap dipakai untuk dashboard

---

## 📈 Dashboard

- Dihasilkan oleh `buat_dashboard.py`
- Menggunakan **Plotly** (standalone HTML dengan inline Plotly.js)
- Menampilkan: **beban listrik aktual + forecast + batas CI 95%**
- Prediksi: **60 hari ke depan** dari model tersimpan

---

## 🚀 Deploy ke GitHub Pages

1. Push folder ke GitHub (struktur minimal):
   ```
   docs/
   └── index.html
   ```
2. **Settings → Pages** → Source: `main` branch → Folder: `/docs`
3. Save → akses: `https://<username>.github.io/<repo>/`

---

## ▶️ Cara Menjalankan

```bash
# 1. Jalankan notebook
jupyter notebook / buka di VS Code: prophet_forecasting.ipynb

# 2. Generate dashboard (membaca model .pkl)
python buat_dashboard.py

# 3. Hasil dashboard: docs/index.html
```

---

## 🛠️ Tech Stack

- Python 3.10
- pandas, numpy, matplotlib, plotly
- Facebook Prophet (CmdStanPy backend)
- Jupyter Notebook / nbformat
- Plotly → GitHub Pages (static HTML)
