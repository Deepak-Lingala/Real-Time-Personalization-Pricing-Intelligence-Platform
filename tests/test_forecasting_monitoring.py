from ecommerce_intelligence.data_generator import SyntheticEcommerceGenerator
from ecommerce_intelligence.forecasting import DemandForecaster
from ecommerce_intelligence.mlops import MonitoringSimulator


def test_forecaster_outputs_product_and_category_horizons() -> None:
    bundle = SyntheticEcommerceGenerator(seed=41).generate_all(
        n_users=25,
        n_products=18,
        n_events=260,
        start_date="2025-01-01",
        days=35,
    )
    product_id = str(bundle.product_catalog["product_id"].iloc[0])
    category = str(bundle.product_catalog["category"].iloc[0])

    forecaster = DemandForecaster(random_state=41, backend="sklearn").fit(bundle.demand)
    product_forecast = forecaster.forecast(product_id, horizon_days=7)
    category_forecast = forecaster.forecast_category(category, horizon_days=7)
    metrics = DemandForecaster(random_state=41, backend="sklearn").evaluate(
        bundle.demand,
        holdout_days=7,
    )
    baseline = forecaster.seasonal_naive_baseline(bundle.demand, holdout_days=7)

    assert len(product_forecast.forecast) == 7
    assert len(category_forecast["forecast"]) == 7
    assert {"mae", "rmse", "mape", "wape"}.issubset(metrics)
    assert {"baseline_mae", "baseline_rmse", "baseline_wape"}.issubset(baseline)


def test_monitoring_snapshot_contains_required_mlops_signals() -> None:
    monitoring = MonitoringSimulator(seed=5).generate_monitoring_snapshot(
        registry=[{"model_name": "two_stage_recommender"}],
        days=5,
    )

    assert len(monitoring) == 5
    assert {
        "prediction_volume",
        "clickstream_volume",
        "average_latency_ms",
        "recommendation_drift",
        "feature_drift",
        "data_quality_issues",
        "pricing_guardrail_violations",
        "feedback_clicks",
        "feedback_purchases",
        "feature_freshness_minutes",
    }.issubset(monitoring.columns)
