from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from stock_predictor.data import fetch_stock_data
from stock_predictor.indicators import add_indicators
from stock_predictor.predictor import train_and_forecast


@st.cache_data(ttl=3600, show_spinner=False)
def load_data(ticker: str, period: str) -> pd.DataFrame:
    raw = fetch_stock_data(ticker=ticker, period=period)
    return add_indicators(raw)


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
    if not ticker:
        st.warning("Please enter a ticker symbol.")
    else:
        with st.spinner(f"Fetching data and running model for {ticker}…"):
            try:
                data = load_data(ticker=ticker, period=period)
                result = train_and_forecast(data, forecast_days=forecast_days)

                current_price = float(data["Close"].iloc[-1])
                prev_price = float(data["Close"].iloc[-2])
                pct_change = (current_price - prev_price) / prev_price * 100

                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Current Price", f"${current_price:.2f}", f"{pct_change:+.2f}%")
                col2.metric("MAE", f"{result.metrics['MAE']:.2f}")
                col3.metric("RMSE", f"{result.metrics['RMSE']:.2f}")
                col4.metric("R²", f"{result.metrics['R2']:.3f}")

                fig = make_subplots(
                    rows=4, cols=1,
                    shared_xaxes=True,
                    row_heights=[0.50, 0.15, 0.175, 0.175],
                    vertical_spacing=0.03,
                    subplot_titles=("Price & Forecast", "Volume", "RSI (14)", "MACD"),
                )

                fig.add_trace(go.Scatter(x=data.index, y=data["Close"], mode="lines", name="Close", line=dict(color="#1f77b4", width=1.5)), row=1, col=1)
                fig.add_trace(go.Scatter(x=data.index, y=data["SMA_20"], mode="lines", name="SMA 20", line=dict(color="#ff7f0e", width=1)), row=1, col=1)
                fig.add_trace(go.Scatter(x=data.index, y=data["SMA_50"], mode="lines", name="SMA 50", line=dict(color="#2ca02c", width=1)), row=1, col=1)
                fig.add_trace(go.Scatter(x=data.index, y=data["BB_Upper"], mode="lines", name="BB Upper", line=dict(color="rgba(150,150,150,0.4)", width=1), showlegend=False), row=1, col=1)
                fig.add_trace(go.Scatter(x=data.index, y=data["BB_Lower"], mode="lines", name="BB Lower", fill="tonexty", fillcolor="rgba(150,150,150,0.1)", line=dict(color="rgba(150,150,150,0.4)", width=1), showlegend=False), row=1, col=1)
                fig.add_trace(go.Scatter(x=result.future_predictions.index, y=result.future_predictions["Predicted_Close"], mode="lines", name="Forecast", line=dict(color="#d62728", width=1.5, dash="dash")), row=1, col=1)

                fig.add_trace(go.Bar(x=data.index, y=data["Volume"], name="Volume", marker_color="rgba(100,100,200,0.4)", showlegend=False), row=2, col=1)

                fig.add_trace(go.Scatter(x=data.index, y=data["Rsi_14"], mode="lines", name="RSI", line=dict(color="#9467bd", width=1), showlegend=False), row=3, col=1)
                fig.add_hline(y=70, line_dash="dot", line_color="red", opacity=0.5, row=3, col=1)
                fig.add_hline(y=30, line_dash="dot", line_color="green", opacity=0.5, row=3, col=1)

                fig.add_trace(go.Scatter(x=data.index, y=data["MACD_Line"], mode="lines", name="MACD", line=dict(color="#1f77b4", width=1), showlegend=False), row=4, col=1)
                fig.add_trace(go.Scatter(x=data.index, y=data["MACD_Signal"], mode="lines", name="Signal", line=dict(color="#ff7f0e", width=1), showlegend=False), row=4, col=1)
                hist_colors = ["#2ca02c" if v >= 0 else "#d62728" for v in data["MACD_Hist"].fillna(0)]
                fig.add_trace(go.Bar(x=data.index, y=data["MACD_Hist"], name="Histogram", marker_color=hist_colors, showlegend=False), row=4, col=1)

                fig.update_layout(
                    title=f"{ticker} — Price History & {forecast_days}-Day Forecast",
                    height=820,
                    hovermode="x unified",
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                )
                fig.update_yaxes(title_text="Price (USD)", row=1, col=1)
                fig.update_yaxes(title_text="Volume", row=2, col=1)
                fig.update_yaxes(title_text="RSI", row=3, col=1, range=[0, 100])
                fig.update_yaxes(title_text="MACD", row=4, col=1)
                st.plotly_chart(fig, use_container_width=True)

                col_a, col_b = st.columns(2)
                with col_a:
                    st.subheader("Latest 10 Rows")
                    latest = data[["Close", "SMA_20", "SMA_50", "Rsi_14", "BB_Upper", "BB_Lower"]].tail(10).copy()
                    latest.index = pd.to_datetime(latest.index).date
                    latest.columns = ["Close", "SMA 20", "SMA 50", "RSI 14", "BB Upper", "BB Lower"]
                    st.dataframe(latest.style.format("{:.2f}"))

                with col_b:
                    st.subheader(f"{forecast_days}-Day Forecast")
                    fcast = result.future_predictions.copy()
                    fcast.index = pd.to_datetime(fcast.index).date
                    st.dataframe(fcast.style.format("{:.2f}"))

            except Exception as exc:
                st.error(f"Unable to run analysis: {exc}")
else:
    st.info("Select inputs and click 'Run Analysis' to start.")
