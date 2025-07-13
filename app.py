import streamlit as st
import pandas as pd
import gdown
import os
import gc
from datetime import datetime
import matplotlib.pyplot as plt
import plotly.graph_objects as go

st.set_page_config(page_title="Dubai Real Estate Pattern Recommender", layout="wide")
st.title("🏙️ Dubai Real Estate Pattern Recommender")

# =======================
# 1. LOAD FILTER OPTIONS
# =======================
@st.cache_data
def get_filter_metadata():
    file_path = "transactions.parquet"
    if not os.path.exists(file_path):
        gdown.download("https://drive.google.com/uc?id=15kO9WvSnWbY4l9lpHwPYRhDmrwuiDjoI", file_path, quiet=False)
    df = pd.read_parquet(file_path, columns=[
        "area_name_en", "property_type_en", "rooms_en", "actual_worth", "instance_date", "reg_type_en", "transaction_id"
    ])
    df["instance_date"] = pd.to_datetime(df["instance_date"], errors="coerce")
    return {
        "areas": sorted(df["area_name_en"].dropna().unique()),
        "types": sorted(df["property_type_en"].dropna().unique()),
        "rooms": sorted(df["rooms_en"].dropna().unique()),
        "min_price": int(df["actual_worth"].min()),
        "max_price": int(df["actual_worth"].max())
    }

filters = get_filter_metadata()

# =======================
# 2. SIDEBAR FILTERS
# =======================
st.sidebar.header("🔍 Property Filters")
with st.sidebar.form("filters_form"):
    selected_areas = st.multiselect("Area", filters["areas"])
    selected_types = st.multiselect("Property Type", filters["types"])
    selected_rooms = st.multiselect("Bedrooms", filters["rooms"])
    budget = st.number_input("Max Budget (AED)", value=filters["max_price"], step=100000)
    view_mode = st.radio("View Insights for", ["Investor", "EndUser"])
    submit = st.form_submit_button("Run Analysis")

# =======================
# 3. DATA FILTERING
# =======================
@st.cache_data
def load_and_filter_data(areas, types, rooms, max_price):
    df = pd.read_parquet("transactions.parquet")
    df["instance_date"] = pd.to_datetime(df["instance_date"], errors="coerce")
    if areas:
        df = df[df["area_name_en"].isin(areas)]
    if types:
        df = df[df["property_type_en"].isin(types)]
    if rooms:
        df = df[df["rooms_en"].isin(rooms)]
    df = df[df["actual_worth"] <= max_price]
    return df

# =======================
# 4. INSIGHT CLASSIFIERS
# =======================
def classify_change(val):
    if val > 5:
        return "Up"
    elif val < -5:
        return "Down"
    else:
        return "Flat"

def classify_offplan(pct):
    if pct > 0.5:
        return "High"
    elif pct > 0.2:
        return "Medium"
    else:
        return "Low"

@st.cache_data
def load_pattern_matrix():
    url = "https://raw.githubusercontent.com/Tariq-qazi/Insights/refs/heads/main/PatternMatrix.csv"
    df = pd.read_csv(url, encoding="utf-8")
    for col in ["Insight_Investor", "Recommendation_Investor", "Insight_EndUser", "Recommendation_EndUser"]:
        if col in df.columns:
            df[col] = df[col].astype(str).apply(lambda x: x.replace('\\n', '\n'))
    return df

def get_pattern_insight(qoq_price, yoy_price, qoq_volume, yoy_volume, offplan_pct):
    pattern_matrix = load_pattern_matrix()
    pattern = {
        "QoQ_Price": classify_change(qoq_price),
        "YoY_Price": classify_change(yoy_price),
        "QoQ_Volume": classify_change(qoq_volume),
        "YoY_Vol": classify_change(yoy_volume),
        "Offplan_Level": classify_offplan(offplan_pct),
    }
    match = pattern_matrix[
        (pattern_matrix["QoQ_Price"] == pattern["QoQ_Price"]) &
        (pattern_matrix["YoY_Price"] == pattern["YoY_Price"]) &
        (pattern_matrix["QoQ_Volume"] == pattern["QoQ_Volume"]) &
        (pattern_matrix["YoY_Vol"] == pattern["YoY_Vol"]) &
        (pattern_matrix["Offplan_Level"] == pattern["Offplan_Level"])
    ]
    return match.iloc[0] if not match.empty else None

# =======================
# 5. MAIN PROCESS
# =======================
if submit:
    with st.spinner("⏳ Running analysis..."):
        gc.collect()
        try:
            df_filtered = load_and_filter_data(selected_areas, selected_types, selected_rooms, budget)
        except Exception as e:
            st.error(f"Error filtering data: {e}")
            st.stop()

        latest_qtr_count = grouped.iloc[-1]["volume"]
        st.success(f"✅ {int(latest_qtr_count)} transactions in the latest quarter.")

        grouped = df_filtered.groupby(pd.Grouper(key="instance_date", freq="Q")).agg({
            "actual_worth": "mean",
            "transaction_id": "count"
        }).rename(columns={"actual_worth": "avg_price", "transaction_id": "volume"}).dropna()

        if len(grouped) >= 5:
            latest, previous = grouped.iloc[-1], grouped.iloc[-2]
            year_ago = grouped.iloc[-5]

            qoq_price = ((latest["avg_price"] - previous["avg_price"]) / previous["avg_price"]) * 100
            yoy_price = ((latest["avg_price"] - year_ago["avg_price"]) / year_ago["avg_price"]) * 100
            qoq_volume = ((latest["volume"] - previous["volume"]) / previous["volume"]) * 100
            yoy_volume = ((latest["volume"] - year_ago["volume"]) / year_ago["volume"]) * 100

            latest_q_start = grouped.index[-1]
            latest_q_df = df_filtered[df_filtered["instance_date"].dt.to_period("Q") == latest_q_start.to_period("Q")]
            offplan_pct = latest_q_df["reg_type_en"].eq("Off-Plan Properties").mean()

            tag_qoq_price = classify_change(qoq_price)
            tag_yoy_price = classify_change(yoy_price)
            tag_qoq_vol = classify_change(qoq_volume)
            tag_yoy_vol = classify_change(yoy_volume)
            tag_offplan = classify_offplan(offplan_pct)

            st.subheader("📊 Market Summary Trends")
            col1, col2, col3 = st.columns(3)
            col1.metric("🏷️ Price QoQ", tag_qoq_price)
            col1.metric("🏷️ Price YoY", tag_yoy_price)
            col2.metric("📈 Volume QoQ", tag_qoq_vol)
            col2.metric("📈 Volume YoY", tag_yoy_vol)
            col3.metric("🧱 Offplan Level", tag_offplan)

            pattern = get_pattern_insight(qoq_price, yoy_price, qoq_volume, yoy_volume, offplan_pct)

            if pattern is not None:
                st.subheader("📌 Recommendation")
                st.markdown(f"**Insight ({view_mode}):** {pattern[f'Insight_{view_mode}']}")
                st.markdown(f"**Recommendation ({view_mode}):** {pattern[f'Recommendation_{view_mode}']}")
            else:
                st.warning("❌ No matching pattern found for current market tags.")

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=grouped.index,
                y=grouped["avg_price"],
                mode='lines+markers',
                name='Avg Price',
                line=dict(width=3)
            ))

            fig.update_layout(
                title="Quarterly Avg Price (AED)",
                xaxis_title="Quarter",
                yaxis_title="AED",
                yaxis=dict(range=[
                    grouped["avg_price"].min() * 0.98,
                    grouped["avg_price"].max() * 1.02
                ]),
                template="plotly_white",
                height=400
            )

            st.plotly_chart(fig, use_container_width=True)

        else:
            st.warning("Not enough quarterly data to calculate changes.")
else:
    st.info("🌟 Use the sidebar filters and click 'Run Analysis' to begin.")
