# Retail Demand Forecasting

**Live demo:** https://retail-demand-forecasting-yoseph.streamlit.app

This project started from a simple question:

> If I manage dozens of stores, how do I know how much each one will sell in the next few weeks so I do not over-stock or run out?

I took the public Rossmann store sales dataset and turned it into a small end-to-end product:
data cleaning, feature engineering, model experiments, and a Streamlit app that lets someone
play with forecasts and see simple inventory suggestions.



## Live demo

**Streamlit app:**  https://retail-demand-forecasting-yoseph.streamlit.app
This will be a public link once the app is deployed to Streamlit Community Cloud.



## What this project actually does

In simple words:

- Reads daily sales data for many stores.
- Builds features that describe **when** a sale happened (calendar, promos) and **what just happened recently** (lags, rolling averages).
- Trains several models and compares them against a very honest baseline.
- Picks the best model and uses it inside a dashboard that shows:
  - how good the forecasts are,
  - which features matter,
  - and a simple idea of how much stock might be needed for the next days.

The focus is more on telling the story of the modelling process than on a single metric.


## Data and problem

The data comes from the Rossmann Kaggle competition: multiple stores, daily sales, promotions, holidays, and more.

I framed the problem as:

- **Task:** forecast daily sales per store for a short horizon.
- **Goal:** beat a naïve baseline that simply says  
  “today’s sales = sales from 7 days ago”.



## How I approached the modelling

I kept the process clear:

1. **Baseline first**  
   I used the 7-day lag as a direct prediction.  
   This gave a test MAPE of about **36%**.

2. **Feature engineering**  
   I added:
   - Calendar features: year, month, day of week, promo flag.
   - Lag features: sales from 1, 7, 14 and 28 days ago (per store).
   - Rolling windows: 7-day and 28-day mean sales.
   These features let the models see trend, seasonality and short-term momentum.

3. **Models I tried**
   - RandomForest with basic features.
   - XGBoost with basic features.
   - RandomForest with the full feature set.
   - XGBoost with the full feature set.
   - A small ensemble that blends RandomForest and XGBoost predictions.

4. **What worked best**
   The advanced XGBoost model clearly won:
   - Baseline MAPE ≈ **36%**
   - Advanced RandomForest MAPE ≈ **10.1%**
   - **Advanced XGBoost MAPE ≈ 9.6%**

   Roughly, the final model cuts the error by about **72%** compared to the baseline.

I also checked an ensemble of RandomForest + XGBoost with different weights,
but the tuned XGBoost model alone was already the best, so I kept it as the final model.


## The dashboard

I did not want this to stay as just a notebook, so I wrapped the results in a Streamlit app.

The app shows:

- **Model performance overview**  
  Baseline vs all models, with MAPE and RMSE. The best model is highlighted, and the
  improvement over baseline is displayed in percentage points.

- **Feature importance**  
  A bar chart for the advanced RandomForest model so it is easy to see that lag and rolling
  features carry a lot of signal, together with calendar and promo information.

- **Store-level view**  
  Choose a store ID and see actual vs predicted daily sales on a line chart.  
  This is the “does the line actually follow reality?” sanity check.

- **Inventory suggestion (demo)**  
  A simple calculation that takes:
  - a planning horizon (for example 14 days),
  - a chosen service level (90%, 95%, 99%),
  - the current stock,
  and returns a suggested order quantity using forecast demand + safety stock.
  It is intentionally simple, more as a conversation starter than a full optimisation model.

- **Error distribution**  
  Histogram and summary statistics for prediction errors (Predicted − Actual)
  to see whether the model systematically over- or under-predicts.



## Files in this repo

Simple layout:

- `app.py` – the Streamlit dashboard.
- `01_eda.ipynb` – data exploration, feature engineering, modelling and evaluation.
- `model_metrics.csv` – comparison table of baseline, RandomForest and XGBoost variants.
- `feature_importances.csv` – feature importance for the advanced RandomForest model.
- `test_results_with_predictions.csv` – test set with final model predictions and errors.



## How to run it locally

```bash
# clone the repo
git clone https://github.com/yosephlijalem/Retail-Demand-Forecasting-.git
cd "Retail Demand Forecasting"

# install basic dependencies
pip install pandas numpy scikit-learn xgboost streamlit

# start the dashboard
streamlit run app.py
