import sys
import os
import pandas as pd
pd.options.mode.string_storage = "python"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import numpy as np

from src.preprocessing import validate_input, preprocess
from src.forecasting import load_model, predict
from src.inventory import inventory_decision
from src.agent import run_agent, chat_with_agent

st.set_page_config(page_title="Retail Inventory Optimizer", layout="wide")

# ---------------------------------------------------------------------------
# Sidebar — shared controls
# ---------------------------------------------------------------------------
st.sidebar.header("Business Assumptions")
understock_cost = st.sidebar.number_input("Understock cost (per unit)", min_value=0.0, value=10.0)
overstock_cost  = st.sidebar.number_input("Overstock cost (per unit)",  min_value=0.0, value=2.0)

st.sidebar.markdown("---")
st.sidebar.header("🤖 Agent Settings")
api_key = st.sidebar.text_input(
    "Anthropic API Key",
    type="password",
    help="Required for the AI Agent tab. Get yours at console.anthropic.com",
)

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab1, tab2 = st.tabs(["📦 Inventory Optimizer", "🤖 AI Inventory Agent"])


# ===========================================================================
# TAB 1 — Original forecasting tool (unchanged)
# ===========================================================================
with tab1:
    st.title("📦 Retail Inventory Optimization System")
    st.markdown("""
    **Turn historical sales data into smart inventory decisions.**

    Upload your sales data and this system will:
    - Forecast product demand
    - Recommend how much stock to keep
    - Estimate inventory-related cost impact
    """)
    st.markdown("---")
    st.subheader("📝 How to Use")
    st.markdown("""
    **Step 1:** Prepare your sales data in the required CSV format
    **Step 2:** Upload the CSV file
    **Step 3:** Adjust cost assumptions (optional)
    **Step 4:** View and download inventory recommendations
    """)
    st.markdown("---")

    st.subheader("📂 Required CSV Format")
    st.markdown("""
    Your CSV file **must** contain the following columns:

    - `date` – Date of sale (YYYY-MM-DD)
    - `store_id` – Store identifier
    - `item_id` – Product identifier
    - `sales` – Units sold (numeric)
    """)

    sample_df = pd.DataFrame({
        "date": ["2024-01-01", "2024-01-02", "2024-01-03"],
        "store_id": [1, 1, 1],
        "item_id": [101, 101, 101],
        "sales": [20, 18, 22],
    })
    st.table(sample_df)
    st.download_button(
        label="⬇️ Download Sample CSV",
        data=sample_df.to_csv(index=False),
        file_name="sample_sales_data.csv",
        mime="text/csv",
    )

    st.markdown("---")
    st.subheader("📤 Upload Your Sales Data")
    uploaded_file = st.file_uploader("Upload sales CSV", type=["csv"], key="tab1_upload")

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
                model     = load_model("models/xgboost_demand_forecaster.pkl")
                forecasted = predict(model, processed)
                safety_stock = model.predict(
                    processed[["lag_1","lag_7","lag_14",
                               "rolling_mean_7","rolling_std_7",
                               "dayofweek","week","month"]]
                ).std()
                final = inventory_decision(
                    forecasted,
                    safety_stock=safety_stock,
                    understock_cost=understock_cost,
                    overstock_cost=overstock_cost,
                )

            st.subheader("✅ Inventory Recommendations")
            display_cols = ["store_id","item_id","sales",
                            "predicted_sales","recommended_inventory","estimated_cost"]
            st.dataframe(final[display_cols].head(50))

            total_cost = final["estimated_cost"].sum()
            st.metric(label="Estimated Total Inventory Cost", value=f"{total_cost:,.2f}")
            st.markdown("---")
            st.download_button(
                label="📥 Download Recommendations CSV",
                data=final[display_cols].to_csv(index=False),
                file_name="inventory_recommendations.csv",
                mime="text/csv",
            )

            # Pass to Agent tab
            st.session_state["forecast_df"] = final

        except Exception as e:
            st.error(f"Error: {e}")


# ===========================================================================
# TAB 2 — AI Inventory Agent
# ===========================================================================
with tab2:
    st.title("🤖 AI Inventory Management Agent")
    st.markdown("""
    The agent autonomously analyses your forecast, decides which items need
    action, raises purchase orders, flags anomalies, and explains its reasoning
    in plain English — all without you clicking through a table.
    """)
    st.markdown("---")

    if not api_key:
        st.warning("Enter your **Anthropic API Key** in the sidebar to use the agent.")
        st.stop()

    # --- Data source ---
    st.subheader("1. Load forecast data")
    data_source = st.radio(
        "Where should the agent get its data?",
        ["Use forecast from Tab 1 (if already run)", "Upload a CSV here"],
        horizontal=True,
    )

    agent_df = None

    if data_source == "Use forecast from Tab 1 (if already run)":
        if "forecast_df" in st.session_state:
            agent_df = st.session_state["forecast_df"]
            st.success(
                f"Using existing forecast: {len(agent_df)} rows, "
                f"{agent_df['store_id'].nunique()} store(s), "
                f"{agent_df['item_id'].nunique()} item(s)."
            )
        else:
            st.info("No forecast found. Run Tab 1 first, or upload a CSV below.")
    else:
        agent_file = st.file_uploader("Upload sales CSV for the agent", type=["csv"], key="agent_upload")
        if agent_file:
            try:
                raw = pd.read_csv(agent_file)
                validate_input(raw)
                processed = preprocess(raw)
                model = load_model("models/xgboost_demand_forecaster.pkl")
                forecasted = predict(model, processed)
                safety_stock = model.predict(
                    processed[["lag_1","lag_7","lag_14",
                               "rolling_mean_7","rolling_std_7",
                               "dayofweek","week","month"]]
                ).std()
                agent_df = inventory_decision(
                    forecasted,
                    safety_stock=safety_stock,
                    understock_cost=understock_cost,
                    overstock_cost=overstock_cost,
                )
                st.session_state["forecast_df"] = agent_df
                st.success(f"Forecast ready: {len(agent_df)} rows processed.")
            except Exception as e:
                st.error(f"Error processing file: {e}")

    if agent_df is None:
        st.stop()

    st.markdown("---")

    # -----------------------------------------------------------------------
    # Section A — Autonomous Run
    # -----------------------------------------------------------------------
    st.subheader("2. Run the agent autonomously")
    st.markdown(
        "The agent will loop through the forecast, call its tools, "
        "take actions, and return a full report — no input needed from you."
    )

    if st.button("▶ Run autonomous agent", type="primary"):
        with st.spinner("Agent is analysing, reasoning, and taking actions..."):
            try:
                result = run_agent(agent_df, api_key=api_key)
                st.session_state["agent_result"] = result
            except Exception as e:
                st.error(f"Agent error: {e}")

    if "agent_result" in st.session_state:
        result = st.session_state["agent_result"]

        st.success(
            f"Agent completed in {result['iterations']} reasoning loop(s) "
            f"with {len(result['action_log'])} tool call(s)."
        )

        st.subheader("📋 Agent report")
        st.markdown(result["final_response"])

        if result["action_log"]:
            st.subheader("🔧 Tool calls made")
            for entry in result["action_log"]:
                with st.expander(f"[{entry['timestamp']}] {entry['tool']}", expanded=False):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.caption("Inputs")
                        st.json(entry["inputs"])
                    with col2:
                        st.caption("Result")
                        st.json(entry["result"])

        pos = [e["result"] for e in result["action_log"] if e["tool"] == "generate_purchase_order"]
        if pos:
            st.subheader("📄 Purchase orders drafted")
            po_df = pd.DataFrame(pos)
            st.dataframe(po_df, use_container_width=True)
            st.download_button(
                label="📥 Download purchase orders CSV",
                data=po_df.to_csv(index=False),
                file_name="purchase_orders.csv",
                mime="text/csv",
            )

        flags = [e["result"] for e in result["action_log"] if e["tool"] == "flag_for_review"]
        if flags:
            st.subheader("🚩 Items flagged for review")
            st.dataframe(pd.DataFrame(flags), use_container_width=True)

    st.markdown("---")

    # -----------------------------------------------------------------------
    # Section B — Conversational chat
    # -----------------------------------------------------------------------
    st.subheader("3. Chat with the agent")
    st.markdown(
        "Ask follow-up questions in plain English. "
        "The agent remembers the full conversation and can call its tools again."
    )

    if "chat_history"      not in st.session_state: st.session_state["chat_history"]      = []
    if "display_messages"  not in st.session_state: st.session_state["display_messages"]  = []

    for msg in st.session_state["display_messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_input = st.chat_input("Ask the agent anything about your inventory...")

    if user_input:
        st.session_state["display_messages"].append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Agent thinking..."):
                try:
                    reply, updated_history = chat_with_agent(
                        user_message=user_input,
                        conversation_history=st.session_state["chat_history"],
                        forecast_df=agent_df,
                        api_key=api_key,
                    )
                    st.session_state["chat_history"] = updated_history
                    st.session_state["display_messages"].append({"role": "assistant", "content": reply})
                    st.markdown(reply)
                except Exception as e:
                    st.error(f"Chat error: {e}")

    if st.session_state["display_messages"]:
        if st.button("🗑 Clear chat history"):
            st.session_state["chat_history"]     = []
            st.session_state["display_messages"] = []
            st.rerun()
