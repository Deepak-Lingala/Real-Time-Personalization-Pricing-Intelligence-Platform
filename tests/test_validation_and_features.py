from ecommerce_intelligence.data_generator import SyntheticEcommerceGenerator
from ecommerce_intelligence.features import FeatureEngineer
from ecommerce_intelligence.ingestion import BatchStreamingIngestionSimulator
from ecommerce_intelligence.preprocessing import EventPreprocessor
from ecommerce_intelligence.validation import DataValidator


def test_validation_cleaning_and_feature_engineering_outputs() -> None:
    bundle = SyntheticEcommerceGenerator(seed=31).generate_all(
        n_users=40,
        n_products=30,
        n_events=450,
        start_date="2025-01-01",
        days=21,
    )
    validator = DataValidator()

    assert validator.validate_events(bundle.events).passed
    assert validator.validate_catalog(bundle.product_catalog).passed
    assert validator.validate_demand(bundle.demand).passed

    preprocessor = EventPreprocessor()
    clean_events = preprocessor.clean_events(bundle.events)
    sessionized = preprocessor.sessionize(clean_events)
    quality = preprocessor.data_quality_summary(bundle.events, clean_events)

    feature_engineer = FeatureEngineer()
    user_features = feature_engineer.build_user_features(clean_events)
    product_features = feature_engineer.build_product_features(clean_events, bundle.product_catalog)
    session_features = feature_engineer.build_session_features(sessionized)
    pricing_features = feature_engineer.build_pricing_frame(clean_events, bundle.demand)
    inventory_features = feature_engineer.build_inventory_features(bundle.inventory, product_features)

    assert len(clean_events) <= len(bundle.events)
    assert "derived_session_id" in sessionized.columns
    assert {"ctr", "conversion_rate", "ltv_estimate", "dominant_category"}.issubset(
        user_features.columns
    )
    assert {"demand_score", "competitor_price_index"}.issubset(product_features.columns)
    assert {"session_conversion_rate", "avg_dwell_time"}.issubset(session_features.columns)
    assert {"historical_conversion_rate", "price_elasticity_score"}.issubset(
        pricing_features.columns
    )
    assert {"stockout_risk_score", "stockout_risk_bucket"}.issubset(inventory_features.columns)
    assert quality["clean_event_count"] > 0


def test_batch_and_streaming_ingestion_simulator_outputs_microbatches() -> None:
    bundle = SyntheticEcommerceGenerator(seed=37).generate_all(
        n_users=12,
        n_products=10,
        n_events=55,
        start_date="2025-01-01",
        days=7,
    )
    ingested = BatchStreamingIngestionSimulator(microbatch_size=20).ingest(bundle)

    assert {"user_profiles", "product_catalog", "pricing", "inventory", "demand"}.issubset(
        ingested.batch_tables
    )
    assert len(ingested.event_stream) >= len(bundle.events)
    assert {"stream_offset", "microbatch_id", "ingestion_timestamp"}.issubset(
        ingested.event_stream.columns
    )
    assert ingested.event_stream["microbatch_id"].nunique() >= 3
