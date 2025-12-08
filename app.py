import streamlit as st
import pandas as pd
import numpy as np
import altair as alt

# ---------------------------------------------------------
# Page config
# ---------------------------------------------------------
st.set_page_config(
    page_title="Retail Demand Forecasting",
    layout="wide",
)

# ---------------------------------------------------------
# Load data
# ---------------------------------------------------------
@st.cache_data
def load_data():
    metrics = pd.read_csv("model_metrics.csv")
    feature_importances = pd.read_csv("feature_importances.csv")
    test_results = pd.read_csv("test_results_with_predictions.csv")
    # Ensure Date column is datetime if present
    if "Date" in test_results.columns:
        test_results["Date"] = pd.to_datetime(test_results["Date"])
    return metrics, feature_importances, test_results


metrics, feature_importances, test_results = load_data()

# ---------------------------------------------------------
# Header
# ---------------------------------------------------------
st.title("Retail Demand Forecasting Dashboard")

st.write(
    "Daily sales forecasting for a chain of retail stores using gradient-boosted trees "
    "and engineered time-series features."
)

st.caption(
    "All machine learning modeling and dashboard implementation by **Yoseph Lijalem**. "
    "Data source: Rossmann Store Sales (Kaggle)."
)

# ---------------------------------------------------------
# Helper for Altair charts
# ---------------------------------------------------------
def default_chart_base(chart):
    return chart.configure_axis(
        labelFontSize=11,
        titleFontSize=12,
    ).configure_legend(
        labelFontSize=11,
        titleFontSize=12,
    )

# ---------------------------------------------------------
# Model Performance Overview
# ---------------------------------------------------------
st.header("Model Performance Overview")

# Identify baseline vs advanced models
baseline_row = metrics[metrics["model"].str.contains("Baseline")].iloc[0]
model_rows = metrics[~metrics["model"].str.contains("Baseline")]

# Best model by MAPE
best_row = model_rows.loc[model_rows["mape"].idxmin()]

abs_improvement = baseline_row["mape"] - best_row["mape"]
rel_improvement = abs_improvement / baseline_row["mape"] * 100

col1, col2, col3 = st.columns(3)
col1.metric("Baseline MAPE", f"{baseline_row['mape']:.2f}%")
col2.metric(f"{best_row['model']} MAPE", f"{best_row['mape']:.2f}%")
col3.metric(
    "MAPE improvement",
    f"{abs_improvement:.2f} pts",
    f"{rel_improvement:.2f}%",
)

st.caption("MAPE = Mean Absolute Percentage Error. Lower values indicate better forecasts.")

description_text = (
    f"**Best model on the test period:** `{best_row['model']}` "
    f"with **MAPE {best_row['mape']:.2f}%** and **RMSE {best_row['rmse']:,.0f}**, "
    f"compared to baseline MAPE **{baseline_row['mape']:.2f}%**."
)
st.markdown(description_text)

with st.expander("Show full metrics table"):
    st.dataframe(metrics, use_container_width=True)

# ---------------------------------------------------------
# Feature Importance
# ---------------------------------------------------------
st.header("Feature Importance")

fi_plot_data = feature_importances.sort_values("importance", ascending=False)

fi_chart = alt.Chart(fi_plot_data).mark_bar().encode(
    x=alt.X("importance:Q", title="Importance"),
    y=alt.Y("feature:N", sort="-x", title="Feature"),
    tooltip=["feature", "importance"],
).properties(
    height=400,
    width="container",
)

st.altair_chart(default_chart_base(fi_chart), use_container_width=True)

# ---------------------------------------------------------
# Store-Level Actual vs Predicted
# ---------------------------------------------------------
st.header("Store-Level Actual vs Predicted")

store_ids = sorted(test_results["Store"].unique())
selected_store = st.selectbox("Select store ID", store_ids)

store_df = (
    test_results[test_results["Store"] == selected_store]
    .sort_values("Date")
    .copy()
)

long_df = store_df.melt(
    id_vars="Date",
    value_vars=["Sales", "PredictedSales"],
    var_name="Type",
    value_name="Value",
)

color_scale = alt.Scale(
    domain=["Sales", "PredictedSales"],
    range=["#1f77b4", "#ff7f0e"],  # blue and orange
)

store_chart = alt.Chart(long_df).mark_line(point=False).encode(
    x=alt.X("Date:T", title="Date"),
    y=alt.Y("Value:Q", title="Sales"),
    color=alt.Color("Type:N", scale=color_scale, title="Series"),
    tooltip=["Date:T", "Type:N", "Value:Q"],
).properties(
    height=400,
    width="container",
)

st.altair_chart(default_chart_base(store_chart), use_container_width=True)

# ---------------------------------------------------------
# Inventory Suggestion (Demo)
# ---------------------------------------------------------
st.header("Inventory Suggestion (Demo)")

st.write(
    "Simple translation of demand forecasts into an order suggestion for the selected store. "
    "Inputs are the planning horizon, target service level, and current on-hand stock."
)

lead_time = st.slider(
    "Planning horizon (days)",
    min_value=7,
    max_value=30,
    value=14,
    step=1,
)

service_level_label = st.selectbox(
    "Target service level", ["90%", "95%", "99%"]
)
service_level_map = {"90%": 1.28, "95%": 1.65, "99%": 2.33}
z = service_level_map[service_level_label]

if store_df.empty:
    current_stock_default = 0.0
else:
    # Use average of last 7 days actual sales as a proxy default
    current_stock_default = float(store_df["Sales"].tail(7).mean())

current_stock = st.number_input(
    "Current stock level for this store (units)",
    min_value=0.0,
    value=current_stock_default,
    step=10.0,
)

if store_df.empty:
    st.warning("No test data available for this store.")
else:
    horizon_df = store_df.sort_values("Date").head(lead_time)
    forecast_demand = horizon_df["PredictedSales"].sum()

    # Daily error standard deviation for safety stock
    sigma_daily = store_df["Error"].std()
    if np.isnan(sigma_daily):
        sigma_daily = 0.0

    safety_stock = z * sigma_daily * np.sqrt(lead_time)
    recommended_order = max(
        0.0, forecast_demand + safety_stock - current_stock
    )

    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Forecast demand (next days)", f"{forecast_demand:,.0f}")
    col_b.metric("Safety stock", f"{safety_stock:,.0f}")
    col_c.metric("Recommended order", f"{recommended_order:,.0f}")

    st.caption(
        "Order quantity = forecast over the planning horizon + safety stock − current stock. "
        "This is a simple illustrative policy, not a fully optimized inventory model."
    )

# ---------------------------------------------------------
# Prediction Error Distribution
# ---------------------------------------------------------
st.header("Prediction Error Distribution (Test Set)")

error_data = test_results["Error"]
hist_values, bin_edges = np.histogram(error_data, bins=40)
hist_df = pd.DataFrame(
    {
        "bin_center": (bin_edges[:-1] + bin_edges[1:]) / 2,
        "count": hist_values,
    }
)

hist_chart = alt.Chart(hist_df).mark_bar().encode(
    x=alt.X("bin_center:Q", title="Prediction error (Predicted − Actual)"),
    y=alt.Y("count:Q", title="Count"),
    tooltip=["bin_center", "count"],
).properties(
    height=350,
    width="container",
)

st.altair_chart(default_chart_base(hist_chart), use_container_width=True)

st.caption(
    "Error distribution summarizes how predictions deviate from actual sales on the test period."
)
