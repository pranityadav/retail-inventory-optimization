import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import streamlit as st
import pandas as pd
import numpy as np

from src.preprocessing import validate_input, preprocess
from src.forecasting import load_model, predict
from src.inventory import inventory_decision

st.set_page_config(page_title="Retail Inventory Optimizer", layout="wide")

st.title("📦 Retail Inventory Optimization System")

st.markdown(
    """
    **Turn historical sales data into smart inventory decisions.**

    Upload your sales data and this system will:
    - Forecast product demand
    - Recommend how much stock to keep
    - Estimate inventory-related cost impact
    """
)
st.markdown("---")
st.subheader("📝 How to Use")

st.markdown(
    """
    **Step 1:** Prepare your sales data in the required CSV format  
    **Step 2:** Upload the CSV file  
    **Step 3:** Adjust cost assumptions (optional)  
    **Step 4:** View and download inventory recommendations  
    """
)
st.markdown("---")
# Sidebar controls (business assumptions)
st.sidebar.header("Business Assumptions")

understock_cost = st.sidebar.number_input(
    "Understock cost (per unit)",
    min_value=0.0,
    value=10.0
)

overstock_cost = st.sidebar.number_input(
    "Overstock cost (per unit)",
    min_value=0.0,
    value=2.0
)

st.subheader("📂 Required CSV Format")

st.markdown(
    """
    Your CSV file **must** contain the following columns:

    - `date` – Date of sale (YYYY-MM-DD)
    - `store_id` – Store identifier (integer)
    - `item_id` – Product identifier (integer)
    - `sales` – Units sold (numeric)
    """
)

sample_df = pd.DataFrame({
    "date": ["2024-01-01", "2024-01-02", "2024-01-03"],
    "store_id": [1, 1, 1],
    "item_id": [101, 101, 101],
    "sales": [20, 18, 22]
})

st.dataframe(sample_df)
sample_csv = sample_df.to_csv(index=False)

st.download_button(
    label="⬇️ Download Sample CSV",
    data=sample_csv,
    file_name="sample_sales_data.csv",
    mime="text/csv"
)
st.markdown("---")
st.subheader("📤 Upload Your Sales Data")
uploaded_file = st.file_uploader(
    "Upload sales CSV",
    type=["csv"]
)
if uploaded_file is None:
    st.info("Please upload a CSV file to generate inventory recommendations.")
if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)

        validate_input(df)

        st.subheader("📄 Uploaded Data Preview")
        st.dataframe(df.head())

        with st.spinner("Processing data and generating recommendations..."):
            processed = preprocess(df)

            model = load_model("models/xgboost_demand_forecaster.pkl")

            forecasted = predict(model, processed)

            safety_stock = model.predict(
                processed[
                    ["lag_1", "lag_7", "lag_14",
                     "rolling_mean_7", "rolling_std_7",
                     "dayofweek", "week", "month"]
                ]
            ).std()

            final = inventory_decision(
                forecasted,
                safety_stock=safety_stock,
                understock_cost=understock_cost,
                overstock_cost=overstock_cost
            )

        st.subheader("✅ Inventory Recommendations")

        display_cols = [
            "store_id",
            "item_id",
            "sales",
            "predicted_sales",
            "recommended_inventory",
            "estimated_cost"
        ]

        st.dataframe(final[display_cols].head(50))

        total_cost = final["estimated_cost"].sum()

        st.metric(
            label="Estimated Total Inventory Cost",
            value=f"{total_cost:,.2f}"
        )
        st.markdown("---")
        # Downloadable output
        csv = final[display_cols].to_csv(index=False)

        st.download_button(
            label="📥 Download Recommendations CSV",
            data=csv,
            file_name="inventory_recommendations.csv",
            mime="text/csv"
        )

    except Exception as e:
        st.error(f"Error: {e}")
        