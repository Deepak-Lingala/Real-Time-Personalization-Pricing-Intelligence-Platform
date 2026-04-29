from __future__ import annotations

import json
import sys
import time
from dataclasses import asdict
from functools import lru_cache
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ecommerce_intelligence.config import MODEL_DIR, PROCESSED_DATA_DIR, SAMPLE_DATA_DIR  # noqa: E402


class RecommendationResponse(BaseModel):
    user_id: str
    recommendations: list[dict[str, Any]]
    latency_ms: float


class PricingResponse(BaseModel):
    product_id: str
    current_price: float
    optimal_price: float
    demand_probability: float
    expected_margin: float
    decision_rule: str
    competitor_price: float | None = None
    competitor_price_delta: float | None = None
    revenue_uplift_estimate: float | None = None
    guardrail_flags: list[str] | None = None
    latency_ms: float


class PricingOptimizeRequest(BaseModel):
    product_id: str
    current_price: float | None = None
    competitor_price: float | None = None
    inventory_level: float | None = None
    demand_score: float | None = None
    product_category: str | None = None
    seasonality_factor: float | None = None
    discount_percentage: float | None = None
    historical_conversion_rate: float | None = None
    margin: float | None = None
    price_elasticity_score: float | None = None


class ForecastResponse(BaseModel):
    product_id: str
    horizon_days: int = Field(..., ge=1, le=30)
    forecast: list[dict[str, Any]]
    latency_ms: float


class SegmentResponse(BaseModel):
    user_id: str
    customer_segment: str
    churn_risk: float
    ltv_estimate: float
    conversion_rate: float | None = None
    retention_signal: str
    latency_ms: float


app = FastAPI(
    title="Real-Time Personalization & Pricing Intelligence API",
    version="0.1.0",
    description=(
        "Synthetic portfolio API for recommendation, pricing optimization, demand forecasting, "
        "customer segmentation, and MLOps metrics. Metrics are simulated using synthetic data only."
    ),
)


def _load_snapshot() -> dict[str, Any]:
    processed = PROCESSED_DATA_DIR / "dashboard_summary.json"
    sample = SAMPLE_DATA_DIR / "dashboard_snapshot.json"
    path = processed if processed.exists() else sample
    if not path.exists():
        raise HTTPException(status_code=503, detail="No dashboard artifact found. Run scripts/run_pipeline.py first.")
    return json.loads(path.read_text(encoding="utf-8"))


@lru_cache(maxsize=4)
def _load_model(filename: str) -> Any | None:
    path = MODEL_DIR / filename
    if not path.exists():
        return None
    import joblib

    return joblib.load(path)


def _load_product_context(product_id: str) -> dict[str, Any] | None:
    feature_path = ROOT / "data" / "feature_store" / "product_features.csv"
    if not feature_path.exists():
        return None
    import csv

    with feature_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("product_id") == product_id:
                return _coerce_numeric(row)
    return None


def _load_customer_segment(user_id: str) -> dict[str, Any] | None:
    segment_path = PROCESSED_DATA_DIR / "customer_segments.csv"
    if not segment_path.exists():
        return None
    import csv

    with segment_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("user_id") == user_id:
                return _coerce_numeric(row)
    return None


def _coerce_numeric(row: dict[str, Any]) -> dict[str, Any]:
    coerced: dict[str, Any] = {}
    for key, value in row.items():
        if value is None:
            coerced[key] = value
            continue
        try:
            coerced[key] = float(value)
        except (TypeError, ValueError):
            coerced[key] = value
    return coerced


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "artifact_mode": "trained" if (PROCESSED_DATA_DIR / "dashboard_summary.json").exists() else "sample"}


@app.get("/recommend/{user_id}", response_model=RecommendationResponse)
def recommend(user_id: str, k: int = Query(default=10, ge=1, le=25)) -> RecommendationResponse:
    start = time.perf_counter()
    model = _load_model("two_stage_recommender.joblib")
    if model is not None:
        recommendations = [asdict(item) for item in model.recommend(user_id, k=k)]
    else:
        snapshot = _load_snapshot()
        recommendations = snapshot.get("top_recommended_products", [])[:k]
    return RecommendationResponse(
        user_id=user_id,
        recommendations=recommendations,
        latency_ms=round((time.perf_counter() - start) * 1000, 2),
    )


@app.post("/pricing/optimize", response_model=PricingResponse)
def pricing_optimize(payload: PricingOptimizeRequest) -> PricingResponse:
    start = time.perf_counter()
    model = _load_model("dynamic_pricing.joblib")
    context = _load_product_context(payload.product_id) or {"product_id": payload.product_id}
    overrides = payload.model_dump(exclude_none=True)
    if "current_price" in overrides and "price" not in overrides:
        overrides["price"] = overrides["current_price"]
    context.update(overrides)
    context.setdefault("price", context.get("current_price", 100.0))
    context.setdefault("product_category", context.get("category", "Unknown"))
    if model is not None:
        decision = asdict(model.optimize_price(context))
    else:
        snapshot = _load_snapshot()
        decisions = snapshot.get("pricing_optimization_comparison", [])
        decision = next((item for item in decisions if item["product_id"] == payload.product_id), decisions[0])
    return PricingResponse(**decision, latency_ms=round((time.perf_counter() - start) * 1000, 2))


@app.get("/forecast/{product_id}", response_model=ForecastResponse)
def forecast(product_id: str, horizon_days: int = Query(default=7, ge=1, le=30)) -> ForecastResponse:
    start = time.perf_counter()
    model = _load_model("demand_forecaster.joblib")
    if model is not None:
        try:
            payload = asdict(model.forecast(product_id, horizon_days=horizon_days))
            forecast_rows = payload.get("forecast", [])
        except ValueError:
            payload = {}
            forecast_rows = []
    else:
        snapshot = _load_snapshot()
        payload = snapshot.get("forecasted_demand", {})
        forecast_rows = payload.get("forecast", [])[:horizon_days]
    return ForecastResponse(
        product_id=product_id,
        horizon_days=horizon_days,
        forecast=forecast_rows,
        latency_ms=round((time.perf_counter() - start) * 1000, 2),
    )


@app.get("/customer/{user_id}/segment", response_model=SegmentResponse)
def customer_segment(user_id: str) -> SegmentResponse:
    start = time.perf_counter()
    trained_segment = _load_customer_segment(user_id)
    if trained_segment is not None:
        return SegmentResponse(
            user_id=user_id,
            customer_segment=str(trained_segment.get("customer_segment", "growth customers")),
            churn_risk=round(float(trained_segment.get("churn_risk", 0.0)), 4),
            ltv_estimate=round(float(trained_segment.get("ltv_estimate", 0.0)), 2),
            conversion_rate=round(float(trained_segment.get("conversion_rate", 0.0)), 4),
            retention_signal=str(trained_segment.get("retention_signal", "healthy")),
            latency_ms=round((time.perf_counter() - start) * 1000, 2),
        )
    snapshot = _load_snapshot()
    segments = snapshot.get("customer_segments", [])
    match = max(segments, key=lambda item: item.get("users", 0)) if segments else {}
    return SegmentResponse(
        user_id=user_id,
        customer_segment=match.get("customer_segment", "growth customers"),
        churn_risk=round(float(match.get("churn_risk", 0.31)), 4),
        ltv_estimate=round(float(match.get("ltv_estimate", 0.0)), 2),
        conversion_rate=round(float(match.get("conversion_rate", 0.0)), 4),
        retention_signal="healthy" if float(match.get("churn_risk", 0.31)) < 0.35 else "save_offer",
        latency_ms=round((time.perf_counter() - start) * 1000, 2),
    )
@app.get("/product/{product_id}/insights")
def product_insights(product_id: str) -> dict[str, Any]:
    snapshot = _load_snapshot()
    pricing = next(
        (item for item in snapshot.get("pricing_optimization_comparison", []) if item.get("product_id") == product_id),
        None,
    )
    context = _load_product_context(product_id) or {}
    return {
        "product_id": product_id,
        "features": context,
        "pricing_decision": pricing,
        "recommendation_presence": [
            item for item in snapshot.get("top_recommended_products", []) if item.get("product_id") == product_id
        ],
        "disclaimer": "Synthetic product insight generated for portfolio demonstration.",
    }


@app.get("/model/metrics")
def model_metrics() -> dict[str, Any]:
    snapshot = _load_snapshot()
    registry_path = MODEL_DIR / "model_registry.json"
    sample_registry = SAMPLE_DATA_DIR / "model_registry.json"
    if registry_path.exists():
        registry = json.loads(registry_path.read_text(encoding="utf-8"))
    elif sample_registry.exists():
        registry = json.loads(sample_registry.read_text(encoding="utf-8"))
    else:
        registry = []
    return {
        "kpis": snapshot.get("kpis", {}),
        "model_metrics": snapshot.get("model_metrics", []),
        "monitoring": snapshot.get("monitoring", []),
        "registry": registry,
        "disclaimer": "All data and metrics are synthetic simulated portfolio metrics.",
    }
@app.get("/monitoring/drift")
def monitoring_drift() -> dict[str, Any]:
    snapshot = _load_snapshot()
    monitoring = snapshot.get("monitoring", [])
    return {
        "drift": [
            {
                "model_name": item.get("model_name"),
                "drift_score": item.get("drift_score"),
                "recommendation_drift": item.get("recommendation_drift"),
                "feature_drift": item.get("feature_drift"),
                "drift_status": item.get("drift_status"),
            }
            for item in monitoring
        ],
        "data_quality": snapshot.get("data_quality", {}),
    }


@app.get("/dashboard/summary")
def dashboard_summary() -> dict[str, Any]:
    return _load_snapshot()


@app.get("/feature-store/features")
def feature_store_features() -> dict[str, Any]:
    snapshot = _load_snapshot()
    registry_path = ROOT / "data" / "feature_store" / "feature_registry.json"
    registry = json.loads(registry_path.read_text(encoding="utf-8")) if registry_path.exists() else snapshot.get("feature_store", [])
    return {"feature_store": registry, "freshness": snapshot.get("feature_store", [])}
