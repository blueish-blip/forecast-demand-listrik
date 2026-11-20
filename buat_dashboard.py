import pickle
import plotly.graph_objects as go

MODEL_PATH = "D:/FreeLLM/Open Code/Project/prophet_model.pkl"
RESULTS_PATH = "D:/FreeLLM/Open Code/Project/forecast_results.pkl"
OUT_PATH = "D:/FreeLLM/Open Code/Project/docs/index.html"

with open(RESULTS_PATH, "rb") as f:
    results = pickle.load(f)
merged = results["test_df"]
metrics = results["metrics"]

mae = float(metrics["MAE"])
rmse = float(metrics["RMSE"])
mape = float(metrics["MAPE"])

# ---------- Figure: Forecast ----------
fig_fc = go.Figure()
fig_fc.add_trace(
    go.Scatter(x=merged["ds"], y=merged["y"], mode="lines", name="Beban Listrik Aktual"),
)
fig_fc.add_trace(go.Scatter(x=merged["ds"], y=merged["yhat"], mode="lines", name="Forecast"))
fig_fc.add_trace(
    go.Scatter(
        x=merged["ds"],
        y=merged["yhat_upper"],
        mode="lines",
        name="CI Atas (95%)",
        line=dict(dash="dash"),
    ),
)
fig_fc.add_trace(
    go.Scatter(
        x=merged["ds"],
        y=merged["yhat_lower"],
        mode="lines",
        name="CI Bawah (95%)",
        line=dict(dash="dash"),
    ),
)
fig_fc.update_layout(
    title="Forecast vs Aktual",
    xaxis_title="Tanggal",
    yaxis_title="Beban Listrik (MWh)",
    hovermode="x unified",
)
plot1 = fig_fc.to_html(full_html=False, include_plotlyjs=True)

# ---------- Figure: Metrics ----------
fig_m = go.Figure()
fig_m.add_trace(
    go.Bar(
        x=["MAE", "RMSE", "MAPE"],
        y=[mae, rmse, mape],
        text=[f"{mae:.2f}", f"{rmse:.2f}", f"{mape:.2f}"],
        textposition="outside",
        name="Metrik Evaluasi",
    ),
)
fig_m.update_layout(
    title="Metrik Evaluasi",
    yaxis_title="Beban Listrik (MWh) / %",
    showlegend=False,
)
plot2 = fig_m.to_html(full_html=False, include_plotlyjs=True)

html = f"""<!doctype html>
<html lang="id">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width,initial-scale=1" />
<title>Forecasting Beban Listrik Panama - Prophet</title>
<style>
:root{{--ink:#E8ECF4;--mut:#9AA5B8;--line:#263043;--bg:#0D1220;--blue:#7B9CC8;--apr:#E6B48F}}
*{{box-sizing:border-box}}
body{{margin:0;background:var(--bg);color:var(--ink);font:15px/1.6 'Instrument Sans',system-ui,sans-serif}}
.wrap{{display:grid;grid-template-columns:190px minmax(0,1fr);max-width:1260px;margin:0 auto;gap:44px;padding:0 28px}}
nav{{position:sticky;top:0;align-self:start;padding:38px 0}}
nav b{{font:500 1.15rem 'Newsreader',Georgia,serif;display:block;margin-bottom:20px;line-height:1.25}}
nav a{{display:block;padding:6px 0 6px 14px;color:var(--mut);text-decoration:none;border-left:2px solid var(--line);font-size:.9rem}}
nav a:hover{{color:var(--ink)}}nav a.on{{color:var(--ink);border-color:var(--ink);font-weight:600}}
main{{padding:44px 0 90px;min-width:0}}
section{{padding-top:54px}}
h1,h2,h3{{font-family:'Newsreader',Georgia,serif;font-weight:500;letter-spacing:-.012em;margin:0}}
h1{{font-size:clamp(2rem,3.8vw,2.9rem);line-height:1.1;max-width:17ch}}
h2{{font-size:1.65rem;margin-bottom:4px}}
h3{{font-size:1.05rem;margin:0 0 8px}}
.lede{{color:var(--mut);max-width:62ch;margin:0 0 22px}}
.hero{{display:grid;grid-template-columns:minmax(0,.9fr) minmax(0,1.1fr);gap:36px;align-items:start}}
.hero p{{max-width:46ch;color:#C3CCE0}}
dl{{margin:22px 0 0;border-top:1px solid var(--line)}}dl div{{display:flex;justify-content:space-between;gap:14px;padding:9px 0;border-bottom:1px solid var(--line)}}
dt{{color:var(--mut)}}dd{{margin:0;font-weight:600;text-align:right}}
.panel{{background:#141A2B;border:1px solid var(--line);border-radius:10px;padding:18px 20px}}
.lede2{{color:var(--mut);max-width:62ch;margin:0 0 22px}}
.plot{{width:100%;height:520px}}
@media(max-width:900px){{.wrap{{grid-template-columns:1fr;padding:0 16px}}nav{{display:none}}.hero{{grid-template-columns:1fr}}}}
</style>
</head>
<body>
<div class="wrap">
<nav>
<b>Beban Listrik</b>
<a href="#ringkasan">Ringkasan</a>
<a href="#forecast">Hasil Forecast</a>
<a href="#evaluasi">Evaluasi</a>
</nav>
<main>
<section id="ringkasan">
<div class="hero">
<div>
<h1>Forecasting Beban Listrik Panama</h1>
<p class="lede">Prediksi beban listrik nasional (<code>nat_demand</code>) di Panama selama ±5 tahun (2015&ndash;2020) menggunakan Facebook Prophet, dengan data uji terakhir (30 hari) untuk evaluasi.</p>
<dl>
<div><dt>MAE</dt><dd>{mae:.2f} MWh</dd></div>
<div><dt>RMSE</dt><dd>{rmse:.2f} MWh</dd></div>
<div><dt>MAPE</dt><dd>{mape:.2f} %</dd></div>
<div><dt>Data uji</dt><dd>30 hari</dd></div>
</dl>
</div>
<div class="panel">
<h3>Key Findings</h3>
<ul style="padding-left:18px;max-width:78ch;margin:0">
<li style="margin:7px 0">Model Prophet mampu memodelkan tren dan pola musiman beban listrik nasional.</li>
<li style="margin:7px 0">Hasil forecasting ditinjau pada data uji terakhir untuk menilai performa model.</li>
<li style="margin:7px 0">Metrik evaluasi (MAE, RMSE, MAPE) dianalisis untuk mengukur akurasi prediksi.</li>
</ul>
</div>
</div>
</section>

<section id="forecast">
<h2>Hasil Forecast</h2>
<p class="lede2">Perbandingan beban listrik aktual dengan hasil forecasting dari model Prophet beserta batas interval kepercayaan (95%).</p>
<div class="panel">
{plot1}
</div>
</section>

<section id="evaluasi">
<h2>Evaluasi Model</h2>
<p class="lede2">Ringkasan metrik evaluasi pada data uji (30 hari terakhir).</p>
<div class="panel">
{plot2}
</div>
</section>
</main>
<footer>Menyusun dengan Facebook Prophet - Forecasting Beban Listrik Panama (nat_demand, 2015-2020).</footer>
</div>
</body>
</html>"""

with open(OUT_PATH, "w", encoding="utf-8") as f:
    f.write(html)

print("DASHBOARD HTML TERSIMPAN:", OUT_PATH)
print("METRIK:", metrics)
