# Bitcoin Price Prediction: Performance vs. Environmental Cost

A time-series forecasting project comparing neural and non-neural approaches to Bitcoin (BTC) price prediction — not just on accuracy, but on **environmental cost** (energy consumption and emissions), using [CodeCarbon](https://codecarbon.io/) to track every model.

## Why this project is different

Most Bitcoin prediction projects optimise for accuracy alone. This one asks a second question most don't: **is the performance gain from a neural model worth its environmental cost?** The univariate LSTM outperformed every other model, but consumed roughly **3x the energy** of its non-neural counterparts — a trade-off worth naming explicitly, especially as neural/LLM-based approaches become the default choice across the industry regardless of problem complexity.

## Methodology overview

1. **Data collection** — Bitcoin OHLC data (Kaggle), supplementary financial market data (Alpha Vantage: Apple, Nvidia, Tesla, Coinbase, PayPal, S&P 500 and NASDAQ ETFs), and Twitter sentiment data, each investigated for suitability, licensing, and ethical use (GDPR Article 5, responsible data reuse checklist).
2. **Cleaning & feature engineering** — missing data handled per-source based on its likely mechanism (MCAR vs. MAR); VADER sentiment scoring on tokenised/anonymised tweets; multicollinearity addressed via VIF (dropped predictors with VIF > 10); RSI and lag features engineered; PCA explored as an alternative to dropping correlated predictors.
3. **Modelling** — four approaches benchmarked head-to-head on two datasets (a long-term dataset and a shorter-term dataset including Twitter sentiment):
   - **Linear Regression** (Optuna-tuned Ridge/Lasso hybrid)
   - **ARIMA & ARIMAX** (auto-selected (p,d,q), stationarity via ADF testing, residual diagnostics via Q-Q plots and Ljung-Box)
   - **LSTM** (univariate and multivariate, Optuna-tuned dropout/learning rate/epochs)
   - **AutoGluon** (as an automated ML benchmark alternative)
4. **Ensemble** — a meta-learner combining predictions from all models via simple averaging.
5. **Evaluation** — MAE, RMSE, and sMAPE via a custom, reusable time-series cross-validation module (`eval_metrics_cv.py`), plus environmental cost (energy consumption, emissions) tracked for every model.

## Key results

| Model | Dataset 1 — MAE | RMSE | sMAPE | Dataset 2 — MAE | RMSE | sMAPE |
|---|---|---|---|---|---|---|
| Linear Regression | 0.0652 | 0.0797 | 8.67% | 0.0961 | 0.0971 | 200.0% |
| ARIMA | 0.3327 | 0.3371 | 36.43% | 0.1392 | 0.1393 | 93.11% |
| ARIMAX | 0.3742 | 0.3858 | 39.80% | 0.0481 | 0.0492 | 46.18% |
| **LSTM (Univariate)** | **0.0433** | **0.0545** | **5.75%** | **0.0115** | **0.0127** | **15.27%** |
| LSTM (Multivariate) | 0.0549 | 0.0665 | 7.14% | 0.0231 | 0.0235 | 33.44% |
| Ensemble | 0.1597 | 0.1652 | 19.37% | 0.0282 | 0.0289 | 30.09% |

**Univariate LSTM performed best on both datasets** — but at roughly 3x the energy consumption and emissions of the non-neural models. Full environmental comparison figures are in `reports/Report_2_Modelling_Results.pdf`.

## Repository structure

```
notebooks/    — numbered in pipeline order: data collection → feature engineering → four modelling approaches → ensemble comparison
src/          — eval_metrics_cv.py: reusable MAE/RMSE/sMAPE + time-series cross-validation utilities
reports/      — full written reports (methodology, literature review, results, AI usage disclosure)
```

## Tech stack

Python · pandas, NumPy, statsmodels (ARIMA/ARIMAX) · scikit-learn (regularised regression, time-series CV) · Optuna (hyperparameter tuning) · NLTK + VADER (sentiment analysis) · AutoGluon (AutoML benchmark) · CodeCarbon (energy/emissions tracking)

## Reproducibility notes

- Notebooks were developed in Google Colab and expect a mounted Google Drive; file paths will need updating to run locally.
- Requires an Alpha Vantage API key for the financial market data.
- `02_tweet_sentiment_processing.ipynb` is RAM-intensive and is kept separate from the main pipeline for this reason.

## Reports

Two full written reports are included in `reports/`, covering data collection & ethics ([Report 1](reports/Report_1_Data_Collection.pdf)) and modelling methodology & results ([Report 2](reports/Report_2_Modelling_Results.pdf)), including a full literature review and an AI usage disclosure statement.
