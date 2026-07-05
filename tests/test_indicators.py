"""Unit tests for stock_predictor.indicators module."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from stock_predictor.indicators import add_indicators


class TestAddIndicators:
    def test_sma20_column_added(self, raw_ohlcv):
        result = add_indicators(raw_ohlcv)
        assert "SMA_20" in result.columns

    def test_sma50_column_added(self, raw_ohlcv):
        result = add_indicators(raw_ohlcv)
        assert "SMA_50" in result.columns

    def test_rsi_column_added(self, raw_ohlcv):
        result = add_indicators(raw_ohlcv)
        assert "Rsi_14" in result.columns

    def test_original_columns_preserved(self, raw_ohlcv):
        result = add_indicators(raw_ohlcv)
        for col in ("Open", "High", "Low", "Close", "Volume"):
            assert col in result.columns

    def test_input_not_mutated(self, raw_ohlcv):
        """add_indicators must not modify the original DataFrame."""
        before = raw_ohlcv.copy()
        add_indicators(raw_ohlcv)
        pd.testing.assert_frame_equal(raw_ohlcv, before)

    def test_sma20_first_19_rows_are_nan(self, raw_ohlcv):
        result = add_indicators(raw_ohlcv)
        assert result["SMA_20"].iloc[:19].isna().all()

    def test_sma20_values_correct(self, raw_ohlcv):
        result = add_indicators(raw_ohlcv)
        expected = raw_ohlcv["Close"].rolling(20).mean()
        pd.testing.assert_series_equal(result["SMA_20"], expected, check_names=False)

    def test_sma50_values_correct(self, raw_ohlcv):
        result = add_indicators(raw_ohlcv)
        expected = raw_ohlcv["Close"].rolling(50).mean()
        pd.testing.assert_series_equal(result["SMA_50"], expected, check_names=False)

    def test_rsi_bounded_0_to_100(self, raw_ohlcv):
        result = add_indicators(raw_ohlcv)
        rsi = result["Rsi_14"].dropna()
        assert (rsi >= 0).all() and (rsi <= 100).all()

    def test_rsi_first_14_rows_are_nan(self, raw_ohlcv):
        # rolling(14) computes the first value at position 13 (0-based);
        # positions 0-12 (first 13 rows) must all be NaN.
        result = add_indicators(raw_ohlcv)
        assert result["Rsi_14"].iloc[:13].isna().all()

    def test_row_count_unchanged(self, raw_ohlcv):
        result = add_indicators(raw_ohlcv)
        assert len(result) == len(raw_ohlcv)

    def test_mostly_increasing_yields_high_rsi(self):
        """Prices that trend strongly upward should produce RSI well above 50."""
        idx = pd.bdate_range("2020-01-01", periods=100)
        # Explicit pattern: 9 up-days of +2 followed by 1 down-day of -1, repeated.
        # This guarantees avg_gain >> avg_loss, so RSI is defined and high.
        steps = np.array(([2.0] * 9 + [-1.0]) * 10)
        close = pd.Series(100.0 + np.cumsum(steps), index=idx)
        df = pd.DataFrame({"Open": close, "High": close, "Low": close, "Close": close,
                           "Adj Close": close, "Volume": 1})
        result = add_indicators(df)
        last_rsi = result["Rsi_14"].dropna().iloc[-1]
        assert last_rsi > 70

    def test_monotone_decrease_yields_rsi_near_0(self):
        """Purely falling prices should give RSI close to 0."""
        idx = pd.bdate_range("2020-01-01", periods=100)
        close = pd.Series(range(200, 100, -1), index=idx, dtype=float)
        df = pd.DataFrame({"Open": close, "High": close, "Low": close, "Close": close,
                           "Adj Close": close, "Volume": 1})
        result = add_indicators(df)
        assert result["Rsi_14"].dropna().iloc[-1] < 5
