from __future__ import annotations

import json
from dataclasses import asdict
from typing import Any

import joblib
import numpy as np
import pandas as pd

from ecommerce_intelligence.config import (
    FEATURE_STORE_DIR,
    MODEL_DIR,
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
    SYNTHETIC_DATA_DIR,
    PipelineConfig,
    ensure_project_directories,
)
from ecommerce_intelligence.customer_analytics import CustomerSegmentation
from ecommerce_intelligence.data_generator import SyntheticDataBundle, SyntheticEcommerceGenerator
from ecommerce_intelligence.feature_store import FeastStyleFeatureStore
from ecommerce_intelligence.features import FeatureEngineer
from ecommerce_intelligence.forecasting import DemandForecaster
from ecommerce_intelligence.ingestion import BatchStreamingIngestionSimulator
from ecommerce_intelligence.mlops import ExperimentTracker, MonitoringSimulator
from ecommerce_intelligence.preprocessing import EventPreprocessor
from ecommerce_intelligence.pricing import DynamicPricingOptimizer
from ecommerce_intelligence.recommenders import TwoStageRecommendationSystem
from ecommerce_intelligence.validation import DataValidator, ValidationReport


def run_pipeline(config: PipelineConfig | None = None) -> dict[str, Any]:
    """Run the synthetic company-style ML platform pipeline end to end."""

    config = config or PipelineConfig()
    ensure_project_directories()

    generator = SyntheticEcommerceGenerator(seed=config.seed)
    bundle = generator.generate_all(
        n_users=config.n_users,
        n_products=config.n_products,
        n_events=config.n_events,
        start_date=config.start_date,
        days=config.training_window_days,
    )

    validation_reports = _validate(bundle.events, bundle.product_catalog, bundle.demand)
    failed = [report for report in validation_reports if not report.passed]
    if failed:
        details = "; ".join(f"{report.table_name}: {report.errors}" for report in failed)
        raise ValueError(f"Validation failed: {details}")

    _save_synthetic_sources(bundle)
    ingested = BatchStreamingIngestionSimulator().ingest(bundle)
    ingested.event_stream.to_csv(PROCESSED_DATA_DIR / "streaming_event_microbatches.csv", index=False)

    preprocessor = EventPreprocessor()
    clean_events = preprocessor.clean_events(ingested.event_stream)
    sessionized_events = preprocessor.sessionize(clean_events)
    data_quality_summary = preprocessor.data_quality_summary(ingested.event_stream, clean_events)
    clean_events.to_csv(PROCESSED_DATA_DIR / "clean_events.csv", index=False)
    sessionized_events.to_csv(PROCESSED_DATA_DIR / "sessionized_events.csv", index=False)

    feature_engineer = FeatureEngineer()
    user_features = feature_engineer.build_user_features(clean_events)
    product_features = feature_engineer.build_product_features(clean_events, bundle.product_catalog)
    product_features = product_features.merge(bundle.pricing, on="product_id", how="left", suffixes=("", "_pricing"))
    product_features = product_features.merge(bundle.inventory, on="product_id", how="left", suffixes=("", "_inventory"))
    product_features["product_category"] = product_features.get("product_category", product_features["category"])
    session_features = feature_engineer.build_session_features(sessionized_events)
    pricing_frame = feature_engineer.build_pricing_frame(clean_events, bundle.demand)
    pricing_frame = pricing_frame.drop(columns=["margin", "price_elasticity_score"], errors="ignore").merge(
        bundle.pricing[["product_id", "margin", "price_elasticity_score"]],
        on="product_id",
        how="left",
    )
    pricing_frame["margin"] = pricing_frame["margin"].fillna(0.35)
    pricing_frame["price_elasticity_score"] = pricing_frame["price_elasticity_score"].fillna(1.0)
    inventory_features = feature_engineer.build_inventory_features(bundle.inventory, product_features)

    feature_engineer.save_feature_store(
        {
            "user_features": user_features,
            "product_features": product_features,
            "session_features": session_features,
            "pricing_features": pricing_frame,
            "inventory_features": inventory_features,
        },
        FEATURE_STORE_DIR,
    )
    feature_store = FeastStyleFeatureStore(FEATURE_STORE_DIR, version="v1")
    feature_store_metadata = feature_store.materialize(
        {
            "user_features": (user_features, "user_id"),
            "product_features": (product_features, "product_id"),
            "session_features": (session_features, "session_id"),
            "pricing_features": (pricing_frame, "product_id"),
            "inventory_features": (inventory_features, "product_id"),
        }
    )

    sorted_events = clean_events.sort_values("timestamp")
    cutoff = sorted_events["timestamp"].quantile(0.82)
    train_events = sorted_events.loc[sorted_events["timestamp"] <= cutoff]
    test_events = sorted_events.loc[sorted_events["timestamp"] > cutoff]

    recommender = TwoStageRecommendationSystem(
        random_state=config.seed,
        retrieval_backend=config.retrieval_backend,
        ranking_backend=config.ranking_backend,
        retrieval_epochs=config.retrieval_epochs,
        retrieval_batch_size=config.retrieval_batch_size,
    )
    recommender_metrics = recommender.evaluate(
        train_events=train_events,
        test_events=test_events,
        catalog=bundle.product_catalog,
        user_features=user_features,
        product_features=product_features,
        k=config.recommendation_k,
    )

    pricing_optimizer = DynamicPricingOptimizer(
        random_state=config.seed,
        backend=config.pricing_backend,
    ).fit(pricing_frame)
    pricing_impact = pricing_optimizer.simulate_business_impact(pricing_frame, bundle.product_catalog)

    forecaster_for_eval = DemandForecaster(
        random_state=config.seed,
        backend=config.forecasting_backend,
    )
    forecast_metrics = forecaster_for_eval.evaluate(bundle.demand, holdout_days=21)
    forecast_metrics.update(forecaster_for_eval.seasonal_naive_baseline(bundle.demand, holdout_days=21))
    forecaster = DemandForecaster(random_state=config.seed, backend=config.forecasting_backend).fit(
        bundle.demand
    )

    segmentation = CustomerSegmentation(random_state=config.seed)
    customer_segments = segmentation.fit_predict(user_features)

    joblib.dump(recommender, MODEL_DIR / "two_stage_recommender.joblib")
    joblib.dump(pricing_optimizer, MODEL_DIR / "dynamic_pricing.joblib")
    joblib.dump(forecaster, MODEL_DIR / "demand_forecaster.joblib")
    customer_segments.to_csv(PROCESSED_DATA_DIR / "customer_segments.csv", index=False)
    batch_recommendations = _build_batch_recommendations(
        recommender=recommender,
        user_features=user_features,
        top_k=config.recommendation_k,
    )
    batch_recommendations.to_csv(PROCESSED_DATA_DIR / "recommendation_outputs.csv", index=False)

    registry_path = MODEL_DIR / "model_registry.json"
    if registry_path.exists():
        registry_path.unlink()
    tracker = ExperimentTracker(registry_path)
    registry_entries = [
        tracker.log_model(
            model_name="two_stage_recommender",
            version="v1.0.0",
            metrics=recommender_metrics,
            artifact_path="models/two_stage_recommender.joblib",
            deployment_status="staging",
            parameters={
                "retrieval_top_k": 100,
                "retrieval_backend_requested": config.retrieval_backend,
                "retrieval_backend_used": recommender.retrieval_model.backend_used or "unknown",
                "retrieval_training_device": recommender.retrieval_model.training_device or "cpu",
                "retrieval_epochs": config.retrieval_epochs,
                "ranking_backend_requested": config.ranking_backend,
                "ranking_backend_used": recommender.reranker.backend_used or "unknown",
            },
        ),
        tracker.log_model(
            model_name="dynamic_pricing",
            version="v1.0.0",
            metrics=pricing_impact,
            artifact_path="models/dynamic_pricing.joblib",
            deployment_status="staging",
            parameters={
                "optimizer": "constrained_expected_margin",
                "pricing_backend_requested": config.pricing_backend,
                "pricing_backend_used": pricing_optimizer.backend_used or "unknown",
                "guardrails": "margin_inventory_competitor",
            },
        ),
        tracker.log_model(
            model_name="demand_forecaster",
            version="v1.0.0",
            metrics=forecast_metrics,
            artifact_path="models/demand_forecaster.joblib",
            deployment_status="staging",
            parameters={
                "forecasting_backend_requested": config.forecasting_backend,
                "forecasting_backend_used": forecaster.backend_used or "unknown",
                "benchmark": "seasonal_naive",
            },
        ),
    ]
    monitoring = MonitoringSimulator(seed=config.seed).generate_monitoring_snapshot(
        registry=[asdict(entry) for entry in registry_entries],
        days=30,
    )
    monitoring.to_csv(PROCESSED_DATA_DIR / "monitoring_snapshot.csv", index=False)

    dashboard_summary = build_dashboard_summary(
        events=clean_events,
        catalog=bundle.product_catalog,
        demand=bundle.demand,
        user_features=user_features,
        product_features=product_features,
        customer_segments=customer_segments,
        recommender=recommender,
        pricing_optimizer=pricing_optimizer,
        forecaster=forecaster,
        recommender_metrics=recommender_metrics,
        pricing_impact=pricing_impact,
        forecast_metrics=forecast_metrics,
        monitoring=monitoring,
        feature_store_metadata=[asdict(item) for item in feature_store_metadata],
        data_quality_summary=data_quality_summary,
        inventory_features=inventory_features,
        model_version="v1.0.0",
    )
    (PROCESSED_DATA_DIR / "dashboard_summary.json").write_text(
        json.dumps(_json_ready(dashboard_summary), indent=2),
        encoding="utf-8",
    )
    return dashboard_summary


def build_dashboard_summary(
    events: pd.DataFrame,
    catalog: pd.DataFrame,
    demand: pd.DataFrame,
    user_features: pd.DataFrame,
    product_features: pd.DataFrame,
    customer_segments: pd.DataFrame,
    recommender: TwoStageRecommendationSystem,
    pricing_optimizer: DynamicPricingOptimizer,
    forecaster: DemandForecaster,
    recommender_metrics: dict[str, float],
    pricing_impact: dict[str, float],
    forecast_metrics: dict[str, float],
    monitoring: pd.DataFrame,
    feature_store_metadata: list[dict[str, Any]],
    data_quality_summary: dict[str, float],
    inventory_features: pd.DataFrame,
    model_version: str,
) -> dict[str, Any]:
    events = events.copy()
    events["date"] = pd.to_datetime(events["timestamp"]).dt.date.astype(str)
    events["net_revenue"] = events["product_price"] * (1 - events["discount_percentage"]) * events["purchase_label"]
    events = events.merge(catalog[["product_id", "margin"]], on="product_id", how="left")

    total_revenue = float(events["net_revenue"].sum())
    total_margin = float((events["net_revenue"] * events["margin"].fillna(0.35)).sum())
    purchases = int(events["purchase_label"].sum())
    views = int((events["event_type"] == "view").sum())
    clicks = int((events["event_type"] == "click").sum())
    total_sessions = int(events["session_id"].nunique())
    repeat_purchase_rate = (
        float((events.loc[events["purchase_label"] == 1].groupby("user_id").size() > 1).mean())
        if purchases
        else 0.0
    )

    revenue_trend = (
        events.groupby("date")
        .agg(
            revenue=("net_revenue", "sum"),
            margin=("margin", lambda values: float((events.loc[values.index, "net_revenue"] * values.fillna(0.35)).sum())),
            purchases=("purchase_label", "sum"),
            ctr=("event_type", lambda values: int((values == "click").sum()) / max(int((values == "view").sum()), 1)),
            conversion_rate=("purchase_label", "mean"),
        )
        .reset_index()
        .tail(45)
        .to_dict("records")
    )
    category_performance = (
        events.groupby("product_category")
        .agg(
            revenue=("net_revenue", "sum"),
            purchases=("purchase_label", "sum"),
            views=("event_type", lambda values: int((values == "view").sum())),
            conversion_rate=("purchase_label", "mean"),
        )
        .reset_index()
        .sort_values("revenue", ascending=False)
        .to_dict("records")
    )

    sample_user = str(user_features.sort_values("ltv_estimate", ascending=False)["user_id"].iloc[0])
    recommendations = [asdict(item) for item in recommender.recommend(sample_user, k=10)]

    pricing_contexts = (
        product_features.sort_values("demand_score", ascending=False)
        .head(8)
        .assign(product_category=lambda frame: frame["category"])
        .to_dict("records")
    )
    pricing_comparison = [asdict(pricing_optimizer.optimize_price(context)) for context in pricing_contexts]

    forecast_product = str(product_features.sort_values("demand_score", ascending=False)["product_id"].iloc[0])
    forecast_30 = asdict(forecaster.forecast(forecast_product, horizon_days=30))
    forecast_category = str(product_features.sort_values("demand_score", ascending=False)["category"].iloc[0])
    category_forecast_30 = forecaster.forecast_category(forecast_category, horizon_days=30)
    segment_summary = CustomerSegmentation.segment_summary(customer_segments).to_dict("records")

    model_metrics = [
        {"model": "two_stage_recommender", **recommender_metrics},
        {"model": "dynamic_pricing", **pricing_impact},
        {"model": "demand_forecaster", **forecast_metrics},
    ]
    feature_importance = _pricing_feature_importance(pricing_optimizer)
    monitoring_latest = (
        monitoring.sort_values("date")
        .groupby("model_name")
        .tail(1)
        .sort_values("model_name")
        .to_dict("records")
    )

    return {
        "kpis": {
            "total_users": int(events["user_id"].nunique()),
            "total_products": int(catalog["product_id"].nunique()),
            "total_sessions": total_sessions,
            "total_revenue": round(total_revenue, 2),
            "total_margin": round(total_margin, 2),
            "ctr": round(clicks / max(views, 1), 4),
            "conversion_rate": round(purchases / max(views, 1), 4),
            "average_order_value": round(total_revenue / max(purchases, 1), 2),
            "stockout_rate": round(float(inventory_features["stockout_flag"].mean()), 4),
            "return_rate": round(float(events.get("return_label", pd.Series(dtype=float)).sum() / max(purchases, 1)), 4),
            "repeat_purchase_rate": round(repeat_purchase_rate, 4),
            "customer_lifetime_value_estimate": round(float(user_features["ltv_estimate"].mean()), 2),
            "estimated_revenue_uplift": pricing_impact["estimated_revenue_uplift"],
            "estimated_margin_improvement": pricing_impact["estimated_margin_improvement"],
            "recommendation_recall_at_k": recommender_metrics.get("recall_at_k", 0.0),
            "recommendation_precision_at_k": recommender_metrics.get("precision_at_k", 0.0),
            "recommendation_ndcg_at_k": recommender_metrics.get("ndcg_at_k", 0.0),
            "forecast_wape": forecast_metrics.get("wape", 0.0),
            "inference_latency_ms": round(float(monitoring["average_latency_ms"].mean()), 2),
            "prediction_volume": int(monitoring["prediction_volume"].sum()),
            "drift_status": "watch" if (monitoring["drift_status"] == "watch").any() else "stable",
            "model_version": model_version,
        },
        "sample_user": sample_user,
        "revenue_trend": revenue_trend,
        "category_performance": category_performance,
        "top_recommended_products": recommendations,
        "pricing_optimization_comparison": pricing_comparison,
        "forecasted_demand": forecast_30,
        "category_forecasted_demand": category_forecast_30,
        "customer_segments": segment_summary,
        "model_metrics": model_metrics,
        "feature_importance": feature_importance,
        "monitoring": monitoring_latest,
        "feature_store": feature_store_metadata,
        "data_quality": data_quality_summary,
        "inventory_risk": (
            inventory_features.groupby("stockout_risk_bucket")
            .agg(products=("product_id", "count"))
            .reset_index()
            .to_dict("records")
        ),
        "demand_summary": (
            demand.groupby("category")
            .agg(daily_sales=("daily_sales", "sum"), promo_days=("promotion_flag", "sum"))
            .reset_index()
            .sort_values("daily_sales", ascending=False)
            .to_dict("records")
        ),
    }


def _validate(events: pd.DataFrame, catalog: pd.DataFrame, demand: pd.DataFrame) -> list[ValidationReport]:
    validator = DataValidator()
    reports = [
        validator.validate_events(events),
        validator.validate_catalog(catalog),
        validator.validate_demand(demand),
    ]
    (PROCESSED_DATA_DIR / "validation_report.json").write_text(
        json.dumps([asdict(report) for report in reports], indent=2),
        encoding="utf-8",
    )
    return reports


def _save_synthetic_sources(bundle: SyntheticDataBundle) -> None:
    tables = {
        "user_profiles": bundle.users,
        "product_catalog": bundle.product_catalog,
        "session_events": bundle.events,
        "demand": bundle.demand,
        "product_reviews": bundle.reviews,
        "pricing": bundle.pricing,
        "inventory": bundle.inventory,
        "recommendation_features": bundle.recommendation_features,
    }
    for name, frame in tables.items():
        frame.to_csv(RAW_DATA_DIR / f"{name}.csv", index=False)
        frame.to_csv(SYNTHETIC_DATA_DIR / f"{name}.csv", index=False)


def _build_batch_recommendations(
    recommender: TwoStageRecommendationSystem,
    user_features: pd.DataFrame,
    top_k: int,
    max_users: int = 500,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if user_features.empty:
        return pd.DataFrame(rows)
    batch_users = (
        user_features.sort_values("ltv_estimate", ascending=False)["user_id"]
        .astype(str)
        .head(max_users)
    )
    for user_id in batch_users:
        for rank, item in enumerate(recommender.recommend(user_id, k=top_k), start=1):
            rows.append({"user_id": user_id, "rank": rank, **asdict(item)})
    return pd.DataFrame(rows)


def _pricing_feature_importance(pricing_optimizer: DynamicPricingOptimizer) -> list[dict[str, float | str]]:
    if pricing_optimizer.pipeline is None:
        return []
    preprocessor = pricing_optimizer.pipeline.named_steps["preprocessor"]
    model = pricing_optimizer.pipeline.named_steps["model"]
    try:
        names = preprocessor.get_feature_names_out()
    except Exception:
        names = np.array(DynamicPricingOptimizer.FEATURES)
    importances = getattr(model, "feature_importances_", np.zeros(len(names)))
    rows = sorted(zip(names, importances), key=lambda row: row[1], reverse=True)[:10]
    return [{"feature": str(name), "importance": round(float(value), 4)} for name, value in rows]


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value
