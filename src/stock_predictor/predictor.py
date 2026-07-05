from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split


@dataclass
class PredictionResult:
    model: LinearRegression
    metrics: dict
    future_predictions: pd.DataFrame


def _build_feature_frame(df: pd.DataFrame, lags: int = 10) -> pd.DataFrame:
    frame = pd.DataFrame(index=df.index)
    frame["Target"] = df["Close"]

    for i in range(1, lags + 1):
        frame[f"Lag_{i}"] = df["Close"].shift(i)

    frame["SMA_20"] = df["SMA_20"]
    frame["SMA_50"] = df["SMA_50"]
    frame["Rsi_14"] = df["Rsi_14"]
    return frame.dropna()


def train_and_forecast(df: pd.DataFrame, forecast_days: int = 30, lags: int = 10) -> PredictionResult:
    """Train a regression model and produce iterative forecasts."""
    features = _build_feature_frame(df, lags=lags)

    X = features.drop(columns=["Target"])
    y = features["Target"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    model = LinearRegression()
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    metrics = {
        "MAE": float(mean_absolute_error(y_test, preds)),
        "RMSE": float(np.sqrt(mean_squared_error(y_test, preds))),
        "R2": float(model.score(X_test, y_test)),
    }

    # Recursive forecast using latest known lags and static last-known indicators.
    close_history = list(df["Close"].dropna().iloc[-lags:].values)
    sma20 = float(df["SMA_20"].dropna().iloc[-1])
    sma50 = float(df["SMA_50"].dropna().iloc[-1])
    rsi14 = float(df["Rsi_14"].dropna().iloc[-1])

    future_values = []
    last_date = df.index.max()

    for day in range(1, forecast_days + 1):
        lag_vector = list(reversed(close_history[-lags:]))
        row = np.array(lag_vector + [sma20, sma50, rsi14], dtype=float).reshape(1, -1)
        next_price = float(model.predict(row)[0])

        close_history.append(next_price)
        future_values.append((last_date + pd.Timedelta(days=day), next_price))

    future_df = pd.DataFrame(future_values, columns=["Date", "Predicted_Close"]).set_index("Date")

    return PredictionResult(model=model, metrics=metrics, future_predictions=future_df)
