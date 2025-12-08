import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="Retail Demand Forecasting", layout="wide")


def apply_custom_style():
    st.markdown(
        """
        <style>
        /* Global page background */
        body {
            background-color: #f3f4f6;
        }
        .main .block-container {
            padding-top: 1.2rem;
            padding-bottom: 3rem;
            max-width: 1160px;
        }

        /* Global font */
        html, body, [class*="css"] {
            font-family: system-ui, -apple-system, BlinkMacSystemFont,
                         "Segoe UI", "Roboto", sans-serif;
            color: #111827;
        }

        /* Hide default menu/footer */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}

        /* Headings */
        h1 {
            font-size: 2.3rem;
            font-weight: 700;
            letter-spacing: -0.03em;
        }
        h2, h3 {
            font-weight: 600;
        }

        /* Turn each vertical block into a card */
        section[data-testid="stVerticalBlock"] > div:first-child {
            background-color: #ffffff;
            padding: 1.3rem 1.6rem;
            border-radius: 1.0rem;
            box-shadow: 0 14px 30px rgba(15, 23, 42, 0.06);
            margin-bottom: 1.4rem;
        }

        /* Metrics styling */
        [data-testid="stMetric"] {
            background: #f9fafb;
            padding: 1rem 1.25rem;
            border-radius: 0.9rem;
            border: 1px solid #e5e7eb;
        }
        [data-testid="stMetricLabel"] {
            font-size: 0.8rem;
            color: #6b7280;
            text-transform: uppercase;
            letter-spacing: 0.04em;
        }
        [data-testid="stMetricValue"] {
            font-size: 1.6rem;
            font-weight: 650;
            color: #111827;
        }
        [data-testid="stMetricDelta"] {
            font-size: 0.9rem;
        }

        /* Captions */
        .element-container p {
            font-size: 0.9rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


apply_custom_style()

# -------------------------------------------------------------------
# Load data
# -------------------------------------------------------------------
metrics = pd.read_csv("model_metrics.csv")
feature_importances = pd.read_csv("feature_importances.csv")
test_results = pd.read_csv("test_results_with_predictions.csv")

# -------------------------------------------------------------------
# Title / intro card
# -------------------------------------------------------------------
st.title("Retail Demand Forecasting Dashboard")
st.write(
    "End-to-end daily sales forecasting for a network of retail stores. "
    "The dashboard highlights model accuracy, feature importance, "
    "store-level predictions, and a simple inventory planning demo."
)

st.markdown(
    "[📄 View code and README on GitHub]"
    "(https://github.com/yosephlijalem/Retail-Demand-Forecasting-)",
)

# -------------------------------------------------------------------
# Model performance overview
# -------------------------------------------------------------------
st.subheader("Model Performance Overview")

baseline_row = metrics[metrics["model"].str.contains("Baseline")].iloc[0]
model_rows = metrics[~metrics["model"].str.contains("Baseline")]

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

st.markdown(
    f"**Best model on the test period:** `{best_row['model']}` "
    f"with **MAPE {best_row['mape']:.2f}%** and **RMSE {best_row['rmse']:,.0f}**, "
    f"compared to baseline MAPE **{baseline_row['mape']:.2f}%**."
)

with st.expander("Show full metrics table"):
    st.dataframe(metrics, use_container_width=True)

# -------------------------------------------------------------------
# Feature importance
# -------------------------------------------------------------------
st.subheader("Feature Importance")

fi_plot_data = feature_importances.sort_values("importance", ascending=True)
st.bar_chart(fi_plot_data.set_index("feature"), use_container_width=True)

st.caption(
    "Relative importance of input features in the RandomForest (advanced) model. "
    "Calendar and lag features capture trends, seasonality, and short-term momentum."
)

# -------------------------------------------------------------------
# Store-level actual vs predicted
# -------------------------------------------------------------------
st.subheader("Store-Level Actual vs Predicted")

store_ids = sorted(test_results["Store"].unique())
selected_store = st.selectbox("Select store ID", store_ids)

store_df = (
    test_results[test_results["Store"] == selected_store]
    .sort_values("Date")
    .copy()
)
store_df["Date"] = pd.to_datetime(store_df["Date"])

store_chart_data = store_df.set_index("Date")[["Sales", "PredictedSales"]]
st.line_chart(store_chart_data, use_container_width=True)

st.caption(
    "Blue line shows historical actual sales; orange line shows model predictions "
    "for the same period for the selected store."
)

# -------------------------------------------------------------------
# Inventory suggestion demo
# -------------------------------------------------------------------
st.subheader("Inventory Suggestion (Demo)")

st.write(
    "This demo converts demand forecasts into a simple order suggestion for the selected store. "
    "Inputs are the planning horizon, service level, and current on-hand stock."
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

    sigma_daily = store_df["Error"].std()
    if np.isnan(sigma_daily):
        sigma_daily = 0.0

    safety_stock = z * sigma_daily * np.sqrt(lead_time)
    recommended_order = max(
        0.0, forecast_demand + safety_stock - current_stock
    )

    c_inv1, c_inv2, c_inv3 = st.columns(3)
    c_inv1.metric("Forecast demand (next days)", f"{forecast_demand:,.0f}")
    c_inv2.metric("Safety stock", f"{safety_stock:,.0f}")
    c_inv3.metric("Recommended order", f"{recommended_order:,.0f}")

    st.caption(
        "Order quantity = forecast for the horizon + safety stock − current stock. "
        "This is a simplified illustration, not an optimized policy."
    )

# -------------------------------------------------------------------
# Error distribution
# -------------------------------------------------------------------
st.subheader("Prediction Error Distribution (Test Set)")

error_data = test_results["Error"]
hist_values, bin_edges = np.histogram(error_data, bins=50)
hist_df = pd.DataFrame(
    {
        "bin_center": (bin_edges[:-1] + bin_edges[1:]) / 2,
        "count": hist_values,
    }
)

c1, c2 = st.columns([2, 1])

with c1:
    st.bar_chart(hist_df.set_index("bin_center"), use_container_width=True)

with c2:
    st.write("Error summary (Predicted − Actual):")
    st.dataframe(
        error_data.describe()[["mean", "std", "min", "max"]].to_frame("value")
    )
