# Time Series Forecasting – US Consumer Credit Card Loans

## Overview
This project analyzes and forecasts US consumer credit card loans from 2000 to 2025 using classical time series models. The dataset is sourced from the Federal Reserve Economic Data (FRED) and contains monthly observations in billions of dollars.

## Objective
To compare the forecasting performance of ARIMA, SARIMA, and SARIMAX models on a real-world macroeconomic time series, using a rolling window one-step-ahead evaluation strategy.

## Dataset
- **Target variable:** Consumer Credit Card Loans (`CCLACBM027NBOG`) – Monthly, in billions of USD
- **Exogenous variables (SARIMAX only):**
  - Federal Funds Rate (`FEDFUNDS`)
  - Unemployment Rate (`UNRATE`)
- **Source:** Federal Reserve Economic Data (FRED)
- **Period:** January 2000 – 2025

## Methods

### Exploratory Data Analysis
- Trend and seasonality visualization
- Year-over-year and month-over-month growth rates
- Seasonal decomposition (additive model)
- Stationarity testing via Augmented Dickey-Fuller (ADF) test
- ACF and PACF analysis
- Structural break identification (2008 financial crisis, COVID-19, Fed rate hikes)

### Model Selection
Grid search over candidate orders using AIC/BIC criteria:
- **ARIMA** – Best order selected: ARIMA(0,1,1)
- **SARIMA** – Seasonal component with period m=12
- **SARIMAX** – SARIMA extended with Federal Funds Rate and Unemployment Rate as exogenous regressors

### Evaluation
All models evaluated using rolling window one-step-ahead forecasting on a held-out test set (20% of data):
- RMSE (Root Mean Squared Error)
- MAE (Mean Absolute Error)
- MAPE (Mean Absolute Percentage Error)
- BIC (Bayesian Information Criterion)

## Results
SARIMAX incorporating macroeconomic variables achieved the best overall performance across evaluation metrics, demonstrating that Federal Funds Rate and Unemployment Rate carry meaningful predictive signal for consumer credit dynamics.

## Technologies
- Python 3
- `pandas`, `numpy` – data handling
- `statsmodels` – ARIMA, SARIMA, SARIMAX modeling
- `scikit-learn` – evaluation metrics
- `matplotlib`, `seaborn` – visualization
- `scipy` – statistical testing

## Files
| File | Description |
|------|-------------|
| `forecasting.py` | Main analysis script |
| `CCLACBM027NBOG.csv` | Consumer credit card loans dataset |
| `FEDFUNDS.csv` | Federal Funds Rate dataset |
| `UNRATE.csv` | Unemployment Rate dataset |
| `arima_forecast_results.csv` | ARIMA forecast output |
| `sarima_comparison_results.csv` | SARIMA vs ARIMA comparison |
| `sarimax_full_comparison_results.csv` | Full model comparison |

## Report
A detailed project report is available in `Forecasting_Presentation.pdf`.