from ecommerce_intelligence.data_generator import SyntheticEcommerceGenerator


def test_enterprise_synthetic_sources_are_generated() -> None:
    bundle = SyntheticEcommerceGenerator(seed=123).generate_all(
        n_users=30,
        n_products=24,
        n_events=250,
        start_date="2025-01-01",
        days=14,
    )

    assert {"user_id", "user_location", "loyalty_tier"}.issubset(bundle.users.columns)
    assert {"product_id", "product_category", "image_feature_vector"}.issubset(bundle.product_catalog.columns)
    assert {"event_type", "dwell_time_seconds", "page_position"}.issubset(bundle.events.columns)
    assert {"current_price", "price_elasticity_score"}.issubset(bundle.pricing.columns)
    assert {"inventory_level", "stockout_flag"}.issubset(bundle.inventory.columns)
    assert not bundle.reviews.empty
    assert not bundle.recommendation_features.empty

