from __future__ import annotations

import pandas as pd
import yfinance as yf


def fetch_stock_data(ticker: str, period: str = "5y", interval: str = "1d") -> pd.DataFrame:
    """Fetch historical OHLCV data for a ticker from Yahoo Finance."""
    data = yf.download(ticker, period=period, interval=interval, auto_adjust=False, progress=False)
    if data.empty:
        raise ValueError(f"No data returned for ticker '{ticker}'.")

    data = data.rename(columns=str.title)
    data.index = pd.to_datetime(data.index)
    return data
