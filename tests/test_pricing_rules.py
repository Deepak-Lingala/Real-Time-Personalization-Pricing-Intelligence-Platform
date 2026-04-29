import pandas as pd

from ecommerce_intelligence.pricing import DynamicPricingOptimizer


def _training_frame() -> pd.DataFrame:
    rows = []
    for category in ["Electronics", "Home"]:
        for index in range(80):
            rows.append(
                {
                    "product_category": category,
                    "competitor_price": 100 + index % 9,
                    "inventory_level": 40 + index * 3,
                    "demand_score": 0.8 + (index % 11) / 6,
                    "seasonality_factor": 0.9 + (index % 5) / 20,
                    "discount_percentage": (index % 7) / 100,
                    "historical_conversion_rate": 0.02 + (index % 13) / 200,
                    "purchase_label": int(index % 5 in {0, 1}),
                }
            )
    return pd.DataFrame(rows)


def test_pricing_optimizer_respects_margin_floor() -> None:
    optimizer = DynamicPricingOptimizer(random_state=7).fit(_training_frame())
    decision = optimizer.optimize_price(
        {
            "product_id": "P-test",
            "price": 100.0,
            "margin": 0.35,
            "product_category": "Electronics",
            "competitor_price": 104.0,
            "inventory_level": 25,
            "demand_score": 1.8,
            "seasonality_factor": 1.05,
            "discount_percentage": 0.04,
            "historical_conversion_rate": 0.08,
        }
    )

    cost = 100.0 * (1 - 0.35)
    assert decision.optimal_price >= cost * 1.08
    assert decision.expected_margin >= 0

