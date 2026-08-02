from __future__ import annotations

import pandas as pd


def add_indicators(data: pd.DataFrame) -> pd.DataFrame:
    """Add technical indicators used for analysis and modeling."""
    df = data.copy()

    df["SMA_20"] = df["Close"].rolling(window=20).mean()
    df["SMA_50"] = df["Close"].rolling(window=50).mean()

    delta = df["Close"].diff()
    gains = delta.where(delta > 0, 0.0)
    losses = -delta.where(delta < 0, 0.0)
    avg_gain = gains.rolling(window=14).mean()
    avg_loss = losses.rolling(window=14).mean()
    rs = avg_gain / avg_loss.replace(0, pd.NA)
    df["Rsi_14"] = 100 - (100 / (1 + rs))

    bb_std = df["Close"].rolling(window=20).std()
    df["BB_Upper"] = df["SMA_20"] + 2 * bb_std
    df["BB_Lower"] = df["SMA_20"] - 2 * bb_std

    ema12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema26 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD_Line"] = ema12 - ema26
    df["MACD_Signal"] = df["MACD_Line"].ewm(span=9, adjust=False).mean()
    df["MACD_Hist"] = df["MACD_Line"] - df["MACD_Signal"]

    return df
