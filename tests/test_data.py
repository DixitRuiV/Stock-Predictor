"""Unit tests for stock_predictor.data module."""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from stock_predictor.data import fetch_stock_data


class TestFetchStockData:
    def test_single_level_columns_returned_as_is(self, raw_ohlcv):
        """When yfinance returns flat columns they should be title-cased and returned."""
        with patch("stock_predictor.data.yf.download", return_value=raw_ohlcv):
            result = fetch_stock_data("AAPL")

        assert isinstance(result.columns, pd.Index)
        assert not isinstance(result.columns, pd.MultiIndex)
        assert "Close" in result.columns

    def test_multiindex_columns_flattened(self, multiindex_ohlcv):
        """MultiIndex columns from yfinance should be reduced to a single level."""
        with patch("stock_predictor.data.yf.download", return_value=multiindex_ohlcv):
            result = fetch_stock_data("AAPL")

        assert not isinstance(result.columns, pd.MultiIndex)
        assert "Close" in result.columns

    def test_close_column_is_series(self, multiindex_ohlcv):
        """Close column must be a 1-D Series, never a DataFrame."""
        with patch("stock_predictor.data.yf.download", return_value=multiindex_ohlcv):
            result = fetch_stock_data("AAPL")

        assert isinstance(result["Close"], pd.Series)

    def test_index_is_datetime(self, raw_ohlcv):
        """DataFrame index must be DatetimeIndex."""
        with patch("stock_predictor.data.yf.download", return_value=raw_ohlcv):
            result = fetch_stock_data("AAPL")

        assert isinstance(result.index, pd.DatetimeIndex)

    def test_empty_download_raises_value_error(self):
        """Empty response from yfinance should raise a descriptive ValueError."""
        empty = pd.DataFrame()
        with patch("stock_predictor.data.yf.download", return_value=empty):
            with pytest.raises(ValueError, match="No data returned"):
                fetch_stock_data("INVALID_TICKER_XYZ")

    def test_required_columns_present(self, raw_ohlcv):
        """Result must contain the standard OHLCV columns."""
        with patch("stock_predictor.data.yf.download", return_value=raw_ohlcv):
            result = fetch_stock_data("AAPL")

        for col in ("Open", "High", "Low", "Close", "Volume"):
            assert col in result.columns, f"Missing column: {col}"

    def test_row_count_preserved(self, raw_ohlcv):
        """Row count must not change during column normalisation."""
        with patch("stock_predictor.data.yf.download", return_value=raw_ohlcv):
            result = fetch_stock_data("AAPL")

        assert len(result) == len(raw_ohlcv)
