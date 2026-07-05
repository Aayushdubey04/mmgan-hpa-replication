# Replicating MMGAN-HPA: Hybrid Stock Price Prediction

A course-project replication of:

> Polamuri, S.R., Srinivas, K., Krishna Mohan, A. (2022). *Multi-Model Generative
> Adversarial Network Hybrid Prediction Algorithm (MMGAN-HPA) for stock market
> prices prediction.* Journal of King Saud University – Computer and Information
> Sciences, 34, 7433–7444.

## What this does

- Loads daily OHLCV data for six NSE-listed stocks used in the original paper:
  **TCS, BHEL, WIPRO, AXISBANK, MARUTI, TATASTEEL**.
- Runs EDA: rolling averages, cross-stock return correlation heatmap,
  volume-vs-volatility scatter.
- Trains a **hybrid model** per ticker — Linear Regression (linear component)
  + Random Forest Regressor (non-linear component), averaged together — in
  the spirit of the paper's MM-HPA / MMGAN-HPA linear+non-linear fusion,
  without the full LSTM-generator / CNN-discriminator GAN.
- Evaluates with MAE, MSE, and correlation between actual and predicted
  closing prices on a chronological 85/15 train/test split.

## Setup

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

Place per-ticker CSVs (`Date, Close, High, Low, Open, Volume`) in `./data/`,
named `TCS.csv`, `BHEL.csv`, etc.

## Run

```bash
python stock_prediction_replication.py
```

Outputs:
- `charts/rolling_avg_tcs.png`
- `charts/correlation_heatmap.png`
- `charts/scatter_volume_volatility.png`
- `charts/hybrid_predictions_grid.png`
- `model_results.csv` (MAE / MSE / correlation per ticker)

## Results summary

| Ticker    | MAE    | MSE       | Correlation |
|-----------|--------|-----------|-------------|
| TCS       | 26.52  | 1267.48   | 0.976       |
| BHEL      | 10.67  | 302.11    | 0.985       |
| WIPRO     | 1.69   | 5.56      | 0.973       |
| AXISBANK  | 22.65  | 1117.74   | 0.976       |
| MARUTI    | 342.29 | 194209.00 | 0.982       |
| TATASTEEL | 1.12   | 2.24      | 0.989       |

## Known limitations

- Substitutes a Linear Regression + Random Forest hybrid for the paper's
  full Stock-GAN (LSTM generator / CNN discriminator).
- Simplifies the paper's preprocessing (Fourier transforms, ARIMA features,
  stacked autoencoders, XGBoost feature selection, Bayesian/RL tuning) to
  lagged prices, moving averages, and volatility.
- Single chronological split rather than walk-forward validation.
- Data window (2018–2023) does not overlap the original paper's dataset,
  so this is a generalization check rather than an exact replication.
