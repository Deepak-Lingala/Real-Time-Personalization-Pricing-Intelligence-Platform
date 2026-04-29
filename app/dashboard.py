from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st


ROOT = Path(__file__).resolve().parents[1]
PROCESSED_SUMMARY = ROOT / "data" / "processed" / "dashboard_summary.json"
SAMPLE_SUMMARY = ROOT / "data" / "sample" / "dashboard_snapshot.json"


@st.cache_data
def load_snapshot() -> dict:
    path = PROCESSED_SUMMARY if PROCESSED_SUMMARY.exists() else SAMPLE_SUMMARY
    return json.loads(path.read_text(encoding="utf-8"))


snapshot = load_snapshot()
kpis = snapshot["kpis"]

st.set_page_config(
    page_title="Personalization & Pricing Intelligence",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      .block-container { padding-top: 1.4rem; padding-bottom: 2rem; }
      [data-testid="stMetricValue"] { font-size: 1.55rem; }
      [data-testid="stSidebar"] { background: #0f172a; }
      [data-testid="stSidebar"] * { color: #f8fafc; }
      .section-title { font-size: 1.25rem; font-weight: 700; margin: 0.8rem 0 0.3rem; }
      .caption { color: #64748b; font-size: 0.88rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Real-Time Personalization & Pricing Intelligence Platform")
st.caption("Synthetic e-commerce ML platform. All data and metrics are simulated portfolio metrics.")

page = st.sidebar.radio(
    "Dashboard",
    [
        "Executive Overview",
        "Recommendation Engine",
        "Retrieval and Ranking Performance",
        "Dynamic Pricing",
        "Demand Forecasting",
        "Customer Segmentation",
        "Product Analytics",
        "Feature Store",
        "Model Performance",
        "MLOps Monitoring",
    ],
)


def metric_grid() -> None:
    cols = st.columns(5)
    cols[0].metric("Users", f"{kpis['total_users']:,}")
    cols[1].metric("Products", f"{kpis['total_products']:,}")
    cols[2].metric("Sessions", f"{kpis.get('total_sessions', 0):,}")
    cols[3].metric("CTR", f"{kpis['ctr']:.1%}")
    cols[4].metric("Conversion", f"{kpis['conversion_rate']:.1%}")
    cols = st.columns(5)
    cols[0].metric("Revenue", f"${kpis['total_revenue']:,.0f}")
    cols[1].metric("Revenue Uplift", f"{kpis['estimated_revenue_uplift']:.1%}")
    cols[2].metric("Margin Improvement", f"{kpis['estimated_margin_improvement']:.1%}")
    cols[3].metric("Precision@K", f"{kpis['recommendation_precision_at_k']:.3f}")
    cols[4].metric("Forecast WAPE", f"{kpis.get('forecast_wape', 0):.1%}")


if page == "Executive Overview":
    metric_grid()
    left, right = st.columns([1.35, 1])
    revenue = pd.DataFrame(snapshot["revenue_trend"])
    categories = pd.DataFrame(snapshot["category_performance"])
    left.plotly_chart(px.line(revenue, x="date", y="revenue", markers=True, title="Revenue Trend"), use_container_width=True)
    right.plotly_chart(px.bar(categories, x="product_category", y="revenue", color="conversion_rate", title="Category Performance"), use_container_width=True)
    st.dataframe(categories, use_container_width=True, hide_index=True)

elif page == "Recommendation Engine":
    st.markdown('<div class="section-title">Top-N Recommendations</div>', unsafe_allow_html=True)
    recommendations = pd.DataFrame(snapshot["top_recommended_products"])
    st.dataframe(recommendations, use_container_width=True, hide_index=True)
    st.plotly_chart(
        px.bar(recommendations, x="product_id", y="product_score" if "product_score" in recommendations else "score", color="product_category", title="Final Product Scores"),
        use_container_width=True,
    )

elif page == "Retrieval and Ranking Performance":
    metrics = pd.DataFrame([row for row in snapshot["model_metrics"] if "recommender" in row["model"]])
    if not metrics.empty:
        st.dataframe(metrics, use_container_width=True, hide_index=True)
        melted = metrics.melt(id_vars="model")
        st.plotly_chart(px.bar(melted, x="variable", y="value", color="model", title="Retrieval and Ranking Metrics"), use_container_width=True)

elif page == "Dynamic Pricing":
    pricing = pd.DataFrame(snapshot["pricing_optimization_comparison"])
    st.plotly_chart(
        px.bar(
            pricing.melt(id_vars="product_id", value_vars=["current_price", "optimal_price"]),
            x="product_id",
            y="value",
            color="variable",
            barmode="group",
            title="Current vs Optimized Price",
        ),
        use_container_width=True,
    )
    st.dataframe(pricing, use_container_width=True, hide_index=True)

elif page == "Demand Forecasting":
    forecast = pd.DataFrame(snapshot["forecasted_demand"]["forecast"])
    st.plotly_chart(px.line(forecast, x="date", y="predicted_demand", markers=True, title="30-Day Product Demand Forecast"), use_container_width=True)
    if "category_forecasted_demand" in snapshot:
        category_forecast = pd.DataFrame(snapshot["category_forecasted_demand"]["forecast"])
        st.plotly_chart(px.line(category_forecast, x="date", y="predicted_demand", markers=True, title="30-Day Category Demand Forecast"), use_container_width=True)
    demand = pd.DataFrame(snapshot["demand_summary"])
    st.plotly_chart(px.bar(demand, x="category", y="daily_sales", color="promo_days", title="Demand by Category"), use_container_width=True)

elif page == "Customer Segmentation":
    segments = pd.DataFrame(snapshot["customer_segments"])
    st.plotly_chart(px.pie(segments, names="customer_segment", values="users", title="Customer Segment Mix"), use_container_width=True)
    st.dataframe(segments, use_container_width=True, hide_index=True)

elif page == "Product Analytics":
    categories = pd.DataFrame(snapshot["category_performance"])
    inventory = pd.DataFrame(snapshot.get("inventory_risk", []))
    left, right = st.columns(2)
    left.plotly_chart(px.bar(categories, x="product_category", y="revenue", color="conversion_rate", title="Category Revenue and Conversion"), use_container_width=True)
    if not inventory.empty:
        right.plotly_chart(px.bar(inventory, x="stockout_risk_bucket", y="products", color="stockout_risk_bucket", title="Inventory Risk"), use_container_width=True)
    st.dataframe(categories, use_container_width=True, hide_index=True)

elif page == "Feature Store":
    feature_store = pd.DataFrame(snapshot.get("feature_store", []))
    data_quality = snapshot.get("data_quality", {})
    cols = st.columns(4)
    cols[0].metric("Raw Events", f"{int(data_quality.get('raw_event_count', 0)):,}")
    cols[1].metric("Clean Events", f"{int(data_quality.get('clean_event_count', 0)):,}")
    cols[2].metric("Duplicate Rate", f"{data_quality.get('duplicate_event_rate', 0):.1%}")
    cols[3].metric("Missing Rate", f"{data_quality.get('tracked_missing_value_rate', 0):.1%}")
    st.dataframe(feature_store, use_container_width=True, hide_index=True)

elif page == "Model Performance":
    metrics = pd.DataFrame(snapshot["model_metrics"])
    importance = pd.DataFrame(snapshot["feature_importance"])
    st.dataframe(metrics, use_container_width=True, hide_index=True)
    st.plotly_chart(px.bar(importance, x="importance", y="feature", orientation="h", title="Pricing Model Feature Importance"), use_container_width=True)

elif page == "MLOps Monitoring":
    monitoring = pd.DataFrame(snapshot["monitoring"])
    st.dataframe(monitoring, use_container_width=True, hide_index=True)
    st.plotly_chart(px.bar(monitoring, x="model_name", y="prediction_volume", color="drift_status", title="Prediction Volume and Drift Status"), use_container_width=True)
    if "feature_drift" in monitoring:
        st.plotly_chart(px.line(monitoring, x="model_name", y=["drift_score", "feature_drift"], markers=True, title="Prediction and Feature Drift"), use_container_width=True)
