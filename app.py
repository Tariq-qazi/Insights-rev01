import streamlit as st
import pandas as pd
import gdown
import os
import gc
from datetime import datetime
import matplotlib.pyplot as plt

st.set_page_config(page_title="Dubai Real Estate Pattern Recommender", layout="wide")
st.title("🏙️ Dubai Real Estate Pattern Recommender")

# 1. LOAD FILTER OPTIONS
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

# 2. SIDEBAR
st.sidebar.header("🔍 Property Filters")
with st.sidebar.form("filters_form"):
    selected_areas = st.multiselect("Area", filters["areas"])
    selected_types = st.multiselect("Property Type", filters["types"])
    selected_rooms = st.multiselect("Bedrooms", filters["rooms"])
    budget = st.number_input("Max Budget (AED)", value=filters["max_price"], step=100000)
    view_mode = st.radio("View Insights for", ["Investor", "EndUser"])
    submit = st.form_submit_button("Run Analysis")

# 3. FILTERED DATA
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

# 4. CLASSIFIERS
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

# 5. MAIN
if submit:
    with st.spinner("⏳ Running analysis..."):
        gc.collect()
        df_filtered = load_and_filter_data(selected_areas, selected_types, selected_rooms, budget)

        st.success(f"✅ {len(df_filtered)} transactions matched.")

        if len(df_filtered) < 10:
            st.warning("📉 Not enough data to calculate trends.")
            st.stop()

        grouped = df_filtered.groupby(pd.Grouper(key="instance_date", freq="Q")).agg({
            "actual_worth": "mean",
            "transaction_id": "count"
        }).rename(columns={"actual_worth": "avg_price", "transaction_id": "volume"}).dropna()

        if len(grouped) >= 2:
            latest = grouped.iloc[-1]
            previous = grouped.iloc[-2]
            qoq_price = ((latest["avg_price"] - previous["avg_price"]) / previous["avg_price"]) * 100
            qoq_volume = ((latest["volume"] - previous["volume"]) / previous["volume"]) * 100

            year_ago = grouped.iloc[-5] if len(grouped) >= 5 else previous
            yoy_price = ((latest["avg_price"] - year_ago["avg_price"]) / year_ago["avg_price"]) * 100
            yoy_volume = ((latest["volume"] - year_ago["volume"]) / year_ago["volume"]) * 100

            latest_qtr = grouped.index[-1]
            df_latest_qtr = df_filtered[df_filtered["instance_date"].dt.to_period("Q") == latest_qtr.to_period("Q")]
            offplan_pct = df_latest_qtr["reg_type_en"].eq("Off-Plan Properties").mean()

            # Diagnostic: Offplan composition in latest quarter
            total_latest = len(df_latest_qtr)
            offplan_latest = df_latest_qtr["reg_type_en"].eq("Off-Plan Properties").sum()
            offplan_ratio = offplan_latest / total_latest if total_latest > 0 else 0
            st.info(f"📦 Latest quarter: {offplan_latest} off-plan out of {total_latest} transactions ({offplan_ratio:.1%})")

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

            import plotly.graph_objects as go
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
            st.warning("📉 Not enough quarterly data.")
else:
    st.info("🎯 Use the sidebar filters and click 'Run Analysis' to begin.")
