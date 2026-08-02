"""Unit tests for stock_predictor.predictor module."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from stock_predictor.predictor import PredictionResult, _build_feature_frame, train_and_forecast


class TestBuildFeatureFrame:
    def test_returns_dataframe(self, indicator_df):
        result = _build_feature_frame(indicator_df)
        assert isinstance(result, pd.DataFrame)

    def test_no_nan_rows(self, indicator_df):
        result = _build_feature_frame(indicator_df)
        assert not result.isna().any().any()

    def test_target_column_present(self, indicator_df):
        result = _build_feature_frame(indicator_df)
        assert "Target" in result.columns

    def test_lag_columns_present(self, indicator_df):
        result = _build_feature_frame(indicator_df, lags=5)
        for i in range(1, 6):
            assert f"Lag_{i}" in result.columns

    def test_column_count(self, indicator_df):
        lags = 7
        result = _build_feature_frame(indicator_df, lags=lags)
        # Target + lags + SMA_20 + SMA_50 + Rsi_14
        assert len(result.columns) == 1 + lags + 3


class TestTrainAndForecast:
    def test_returns_prediction_result(self, indicator_df):
        result = train_and_forecast(indicator_df, forecast_days=5)
        assert isinstance(result, PredictionResult)

    def test_model_is_linear_regression(self, indicator_df):
        result = train_and_forecast(indicator_df, forecast_days=5)
        assert isinstance(result.model, LinearRegression)

    def test_metrics_keys_present(self, indicator_df):
        result = train_and_forecast(indicator_df, forecast_days=5)
        for key in ("MAE", "RMSE", "R2"):
            assert key in result.metrics

    def test_mae_is_non_negative(self, indicator_df):
        result = train_and_forecast(indicator_df, forecast_days=5)
        assert result.metrics["MAE"] >= 0

    def test_rmse_is_non_negative(self, indicator_df):
        result = train_and_forecast(indicator_df, forecast_days=5)
        assert result.metrics["RMSE"] >= 0

    def test_r2_in_reasonable_range(self, indicator_df):
        result = train_and_forecast(indicator_df, forecast_days=5)
        assert -10 <= result.metrics["R2"] <= 1.0

    def test_forecast_row_count(self, indicator_df):
        days = 14
        result = train_and_forecast(indicator_df, forecast_days=days)
        assert len(result.future_predictions) == days

    def test_forecast_dates_after_history(self, indicator_df):
        result = train_and_forecast(indicator_df, forecast_days=7)
        last_history_date = indicator_df.index.max()
        assert result.future_predictions.index.min() > last_history_date

    def test_forecast_dates_are_sequential(self, indicator_df):
        result = train_and_forecast(indicator_df, forecast_days=10)
        diffs = result.future_predictions.index.to_series().diff().dropna()
        assert (diffs > pd.Timedelta(0)).all()

    def test_forecast_column_name(self, indicator_df):
        result = train_and_forecast(indicator_df, forecast_days=5)
        assert "Predicted_Close" in result.future_predictions.columns

    def test_forecast_values_are_numeric(self, indicator_df):
        result = train_and_forecast(indicator_df, forecast_days=5)
        assert result.future_predictions["Predicted_Close"].dtype.kind == "f"

    def test_custom_lags(self, indicator_df):
        """Non-default lag count must still produce a valid result."""
        result = train_and_forecast(indicator_df, forecast_days=5, lags=5)
        assert len(result.future_predictions) == 5

    def test_single_day_forecast(self, indicator_df):
        result = train_and_forecast(indicator_df, forecast_days=1)
        assert len(result.future_predictions) == 1
