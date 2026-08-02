# Stock Price Analysis and Prediction

A Python Streamlit application that fetches stock market data, computes technical indicators, and predicts future stock prices using a machine learning baseline model.

## Features

- Historical stock data fetch from Yahoo Finance
- Visual analysis of Close price with SMA 20 and SMA 50
- RSI(14) indicator calculation
- Baseline regression model for short-term price forecasting
- Forecast table and model metrics (MAE, RMSE, R2)

## Project Structure

- `app.py`: Streamlit entrypoint
- `src/stock_predictor/data.py`: Data retrieval
- `src/stock_predictor/indicators.py`: Technical indicators
- `src/stock_predictor/predictor.py`: Model training and forecasting

## Setup (Windows PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Run

```powershell
streamlit run app.py
```

## Notes

- This project is for educational analysis purposes and is not financial advice.
- Model predictions are baseline estimates and can be improved with richer features and models.
