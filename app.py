from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from stock_predictor.data import fetch_stock_data
from stock_predictor.indicators import add_indicators
from stock_predictor.predictor import train_and_forecast


st.set_page_config(page_title="Stock Price Analysis & Prediction", layout="wide")
st.title("Stock Price Analysis & Prediction")
st.caption("Analyze historical prices and generate a short-term predictive forecast.")

with st.sidebar:
    st.header("Inputs")
    ticker = st.text_input("Ticker", value="AAPL").upper().strip()
    period = st.selectbox("History Range", ["1y", "2y", "5y", "10y"], index=2)
    forecast_days = st.slider("Forecast Days", min_value=7, max_value=60, value=30)
    run_clicked = st.button("Run Analysis")

if run_clicked:
    try:
        raw = fetch_stock_data(ticker=ticker, period=period)
        data = add_indicators(raw)
        result = train_and_forecast(data, forecast_days=forecast_days)

        col1, col2, col3 = st.columns(3)
        col1.metric("MAE", f"{result.metrics['MAE']:.2f}")
        col2.metric("RMSE", f"{result.metrics['RMSE']:.2f}")
        col3.metric("R²", f"{result.metrics['R2']:.3f}")

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data.index, y=data["Close"], mode="lines", name="Close"))
        fig.add_trace(go.Scatter(x=data.index, y=data["SMA_20"], mode="lines", name="SMA 20"))
        fig.add_trace(go.Scatter(x=data.index, y=data["SMA_50"], mode="lines", name="SMA 50"))
        fig.add_trace(
            go.Scatter(
                x=result.future_predictions.index,
                y=result.future_predictions["Predicted_Close"],
                mode="lines",
                name="Predicted",
            )
        )
        fig.update_layout(title=f"{ticker} Price History + Forecast", xaxis_title="Date", yaxis_title="Price")
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Latest Data")
        latest = data[["Close", "SMA_20", "SMA_50", "Rsi_14"]].tail(10).copy()
        latest.index = pd.to_datetime(latest.index).date
        st.dataframe(latest)

        st.subheader("Forecast Values")
        st.dataframe(result.future_predictions)

    except Exception as exc:
        st.error(f"Unable to run analysis: {exc}")
else:
    st.info("Select inputs and click 'Run Analysis' to start.")
