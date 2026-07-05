from __future__ import annotations

import pandas as pd
import yfinance as yf


def fetch_stock_data(ticker: str, period: str = "5y", interval: str = "1d") -> pd.DataFrame:
    """Fetch historical OHLCV data for a ticker from Yahoo Finance."""
    data = yf.download(ticker, period=period, interval=interval, auto_adjust=False, progress=False)
    if data.empty:
        raise ValueError(f"No data returned for ticker '{ticker}'.")

    # yfinance may return MultiIndex columns (Price, Ticker) even for one ticker.
    if isinstance(data.columns, pd.MultiIndex):
        ticker_level = data.columns.get_level_values(-1)
        if ticker_level.nunique() == 1:
            data.columns = data.columns.get_level_values(0)
        else:
            data = data.xs(str(ticker).upper(), axis=1, level=-1, drop_level=True)

    data = data.rename(columns=str.title)
    data.index = pd.to_datetime(data.index)
    return data
