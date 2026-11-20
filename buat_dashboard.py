import pickle
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

MODEL_PATH = "D:/FreeLLM/Open Code/Project/prophet_model.pkl"
RESULTS_PATH = "D:/FreeLLM/Open Code/Project/forecast_results.pkl"
OUT_PATH = "D:/FreeLLM/Open Code/Project/docs/index.html"

with open(RESULTS_PATH, "rb") as f:
    results = pickle.load(f)
merged = results["test_df"]
metrics = results["metrics"]

fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.15)

fig.add_trace(
    go.Scatter(x=merged["ds"], y=merged["y"], mode="lines", name="Beban Listrik Aktual"),
    row=1,
    col=1,
)
fig.add_trace(
    go.Scatter(x=merged["ds"], y=merged["yhat"], mode="lines", name="Forecast"),
    row=1,
    col=1,
)
fig.add_trace(
    go.Scatter(
        x=merged["ds"],
        y=merged["yhat_upper"],
        mode="lines",
        name="CI Atas (95%)",
        line=dict(dash="dash"),
    ),
    row=1,
    col=1,
)
fig.add_trace(
    go.Scatter(
        x=merged["ds"],
        y=merged["yhat_lower"],
        mode="lines",
        name="CI Bawah (95%)",
        line=dict(dash="dash"),
    ),
    row=1,
    col=1,
)

fig.add_trace(
    go.Bar(
        x=["MAE", "RMSE", "MAPE"],
        y=[metrics["MAE"], metrics["RMSE"], metrics["MAPE"]],
        text=[
            f"{metrics['MAE']:.2f}",
            f"{metrics['RMSE']:.2f}",
            f"{metrics['MAPE']:.2f}",
        ],
        textposition="outside",
        name="Metrik Evaluasi",
    ),
    row=2,
    col=1,
)

fig.update_layout(
    title="Forecasting Beban Listrik Panama - Prophet (Hasil Evaluasi)",
    xaxis_title="Tanggal",
    yaxis_title="Beban Listrik (MWh)",
    hovermode="x unified",
)

fig.write_html(OUT_PATH, include_plotlyjs=True)
print("DASHBOARD HTML TERSIMPAN:", OUT_PATH)
print("METRIK:", metrics)
