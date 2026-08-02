"""Shared pytest fixtures for all test modules."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def _make_price_series(n: int = 300, start: str = "2020-01-01", seed: int = 42) -> pd.Series:
    rng = np.random.default_rng(seed)
    prices = 100 + np.cumsum(rng.normal(0, 1, n))
    idx = pd.bdate_range(start=start, periods=n)
    return pd.Series(prices, index=idx, name="Close")


@pytest.fixture
def raw_ohlcv() -> pd.DataFrame:
    """Minimal single-level OHLCV DataFrame (no network needed)."""
    close = _make_price_series()
    df = pd.DataFrame(
        {
            "Open": close * 0.99,
            "High": close * 1.01,
            "Low": close * 0.98,
            "Close": close,
            "Adj Close": close,
            "Volume": np.ones(len(close), dtype=int) * 1_000_000,
        }
    )
    df.index.name = "Date"
    return df


@pytest.fixture
def multiindex_ohlcv(raw_ohlcv) -> pd.DataFrame:
    """DataFrame with yfinance-style MultiIndex (Price, Ticker) columns."""
    ticker = "AAPL"
    mi = pd.MultiIndex.from_tuples(
        [(col, ticker) for col in raw_ohlcv.columns], names=["Price", "Ticker"]
    )
    df = raw_ohlcv.copy()
    df.columns = mi
    return df


@pytest.fixture
def indicator_df(raw_ohlcv):
    """OHLCV data with SMA and RSI columns already added."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

    from stock_predictor.indicators import add_indicators
    return add_indicators(raw_ohlcv)
