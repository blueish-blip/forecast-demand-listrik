import pickle
import pandas as pd
import plotly.graph_objects as go
from prophet import Prophet

MODEL_PATH = "D:/FreeLLM/Open Code/Project/prophet_model.pkl"
DATA_PATH = r"C:\Users\hhfzsn\Downloads\Panama_beban_listrik_per_jam_2015-2020_.xlsx"
OUT_PATH = "D:/FreeLLM/Open Code/Project/docs/index.html"

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

future = model.make_future_dataframe(periods=60)
forecast = model.predict(future)

df = pd.read_excel(DATA_PATH, sheet_name="data")
df["datetime"] = pd.to_datetime(df["datetime"])
daily = (
    df.groupby(df["datetime"].dt.date)["nat_demand"]
    .sum()
    .reset_index()
    .rename(columns={"datetime": "ds", "nat_demand": "y"})
)
daily["ds"] = pd.to_datetime(daily["ds"])

fig = go.Figure()

fig.add_trace(
    go.Scatter(x=daily["ds"], y=daily["y"], mode="lines", name="Beban Listrik Aktual")
)

fig.add_trace(
    go.Scatter(x=forecast["ds"], y=forecast["yhat"], mode="lines", name="Forecast")
)

fig.add_trace(
    go.Scatter(
        x=forecast["ds"],
        y=forecast["yhat_upper"],
        mode="lines",
        name="Batas Atas 95%",
        line=dict(dash="dash"),
    )
)

fig.add_trace(
    go.Scatter(
        x=forecast["ds"],
        y=forecast["yhat_lower"],
        mode="lines",
        name="Batas Bawah 95%",
        line=dict(dash="dash"),
    )
)

fig.update_layout(
    title="Forecasting Beban Listrik Panama - Prophet",
    xaxis_title="Tanggal",
    yaxis_title="Beban Listrik (MWh)",
    hovermode="x unified",
)

fig.write_html(OUT_PATH, include_plotlyjs=True)
print("DASHBOARD HTML TERSIMPAN:", OUT_PATH)
print("PREDIKSI 60 HARI KE DEPAN:")
print(forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].head(10).to_string())
