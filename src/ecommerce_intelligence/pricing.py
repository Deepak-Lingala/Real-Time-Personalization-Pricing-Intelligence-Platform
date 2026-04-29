from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


@dataclass(frozen=True)
class PricingDecision:
    product_id: str
    current_price: float
    optimal_price: float
    demand_probability: float
    expected_margin: float
    decision_rule: str
    competitor_price: float
    competitor_price_delta: float
    revenue_uplift_estimate: float
    guardrail_flags: list[str]


class DynamicPricingOptimizer:
    """Constrained pricing model that balances conversion probability and unit economics.

    `backend="auto"` uses XGBoost when available and falls back to sklearn for laptop/CI runs.
    """

    FEATURES = [
        "competitor_price",
        "inventory_level",
        "demand_score",
        "seasonality_factor",
        "discount_percentage",
        "historical_conversion_rate",
        "margin",
        "price_elasticity_score",
        "product_category",
    ]

    def __init__(self, random_state: int = 42, backend: str = "auto") -> None:
        self.random_state = random_state
        self.backend = backend
        self.backend_used: str | None = None
        self.pipeline: Pipeline | None = None

    def fit(self, pricing_frame: pd.DataFrame) -> "DynamicPricingOptimizer":
        frame = pricing_frame.copy()
        if "margin" not in frame:
            frame["margin"] = 0.35
        if "price_elasticity_score" not in frame:
            frame["price_elasticity_score"] = 1.0
        frame = frame.dropna(subset=self.FEATURES + ["purchase_label"]).copy()
        if frame["purchase_label"].nunique() < 2 and len(frame) > 1:
            frame.loc[frame.index[0], "purchase_label"] = 1 - int(frame["purchase_label"].iloc[0])
        numeric = [column for column in self.FEATURES if column != "product_category"]
        categorical = ["product_category"]
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), numeric),
                ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
            ]
        )
        model, backend_used = self._select_model()
        self.pipeline = Pipeline([("preprocessor", preprocessor), ("model", model)])
        self.pipeline.fit(frame[self.FEATURES], frame["purchase_label"].astype(int))
        self.backend_used = backend_used
        return self

    def optimize_price(self, product_context: dict) -> PricingDecision:
        if self.pipeline is None:
            raise RuntimeError("Pricing optimizer has not been fitted")

        current_price = float(product_context["price"])
        competitor_price = float(product_context.get("competitor_price", current_price))
        margin_rate = float(product_context.get("margin", 0.35))
        cost = current_price * (1 - margin_rate)
        price_floor = cost * 1.08
        price_ceiling = current_price * 1.28

        demand_score = float(product_context.get("demand_score", 1.0))
        inventory = float(product_context.get("inventory_level", product_context.get("inventory", 100)))
        lower_bound = max(price_floor, current_price * 0.78)
        upper_bound = price_ceiling
        rule = "balanced optimization"
        guardrail_flags: list[str] = []

        if inventory > 600 and demand_score < 1.2:
            lower_bound = max(price_floor, current_price * 0.70)
            upper_bound = min(upper_bound, current_price * 1.02)
            rule = "increase discount for high inventory and low demand"
            guardrail_flags.append("high_inventory_low_demand_discount")
        elif inventory < 75 and demand_score > 1.5:
            lower_bound = max(lower_bound, current_price * 1.03)
            upper_bound = max(upper_bound, current_price * 1.35)
            rule = "increase price for low inventory and high demand"
            guardrail_flags.append("low_inventory_high_demand_margin_protection")
        elif current_price < price_floor:
            lower_bound = price_floor
            rule = "avoid price below margin threshold"
            guardrail_flags.append("minimum_margin_floor")

        candidates = np.linspace(lower_bound, upper_bound, 31)
        scored = []
        for price in candidates:
            candidate = self._candidate_frame(product_context, current_price=current_price, candidate_price=float(price))
            demand_probability = float(self.pipeline.predict_proba(candidate[self.FEATURES])[:, 1][0])
            unit_margin = max(float(price) - cost, 0)
            expected_margin = unit_margin * demand_probability
            scored.append((float(price), demand_probability, expected_margin))
        optimal_price, demand_probability, expected_margin = max(scored, key=lambda row: row[2])
        if optimal_price < price_floor:
            guardrail_flags.append("margin_floor_violation")
        if competitor_price and optimal_price > competitor_price * 1.2:
            guardrail_flags.append("competitor_gap_watch")
        revenue_uplift = ((optimal_price * demand_probability) - (current_price * float(product_context.get("historical_conversion_rate", demand_probability)))) / max(current_price * max(float(product_context.get("historical_conversion_rate", demand_probability)), 1e-6), 1e-6)
        return PricingDecision(
            product_id=str(product_context["product_id"]),
            current_price=round(current_price, 2),
            optimal_price=round(optimal_price, 2),
            demand_probability=round(demand_probability, 4),
            expected_margin=round(expected_margin, 2),
            decision_rule=rule,
            competitor_price=round(competitor_price, 2),
            competitor_price_delta=round(optimal_price - competitor_price, 2),
            revenue_uplift_estimate=round(float(revenue_uplift), 4),
            guardrail_flags=guardrail_flags or ["passed"],
        )

    def simulate_business_impact(self, pricing_frame: pd.DataFrame, catalog: pd.DataFrame, sample_size: int = 250) -> dict[str, float]:
        if self.pipeline is None:
            raise RuntimeError("Pricing optimizer has not been fitted")
        product_stats = (
            pricing_frame.groupby("product_id")
            .agg(
                product_category=("product_category", "first"),
                competitor_price=("competitor_price", "mean"),
                inventory_level=("inventory_level", "mean"),
                demand_score=("demand_score", "mean"),
                seasonality_factor=("seasonality_factor", "mean"),
                discount_percentage=("discount_percentage", "mean"),
                historical_conversion_rate=("historical_conversion_rate", "mean"),
                observed_units=("purchase_label", "sum"),
            )
            .reset_index()
        )
        contexts = catalog.merge(product_stats, on="product_id", how="inner").head(sample_size)
        current_revenue = 0.0
        optimized_revenue = 0.0
        current_margin = 0.0
        optimized_margin = 0.0
        current_conversion_sum = 0.0
        optimized_conversion_sum = 0.0

        for context in contexts.to_dict("records"):
            decision = self.optimize_price(context)
            units = max(float(context.get("observed_units", 1)), 1.0)
            current_price = float(context["price"])
            cost = current_price * (1 - float(context.get("margin", 0.35)))
            current_conversion = float(context.get("historical_conversion_rate", 0.03))
            optimized_conversion = decision.demand_probability
            current_revenue += current_price * units * current_conversion
            optimized_revenue += decision.optimal_price * units * optimized_conversion
            current_margin += (current_price - cost) * units * current_conversion
            optimized_margin += (decision.optimal_price - cost) * units * optimized_conversion
            current_conversion_sum += current_conversion
            optimized_conversion_sum += optimized_conversion

        revenue_uplift = (optimized_revenue - current_revenue) / max(current_revenue, 1)
        margin_improvement = (optimized_margin - current_margin) / max(current_margin, 1)
        conversion_improvement = (optimized_conversion_sum - current_conversion_sum) / max(current_conversion_sum, 1e-6)
        return {
            "estimated_revenue_uplift": round(float(revenue_uplift), 4),
            "estimated_margin_improvement": round(float(margin_improvement), 4),
            "estimated_conversion_improvement": round(float(conversion_improvement), 4),
        }

    @staticmethod
    def _candidate_frame(product_context: dict, current_price: float, candidate_price: float) -> pd.DataFrame:
        discount = max(0.0, 1 - candidate_price / max(current_price, 1))
        return pd.DataFrame(
            [
                {
                    "competitor_price": float(product_context.get("competitor_price", current_price)),
                    "inventory_level": float(product_context.get("inventory_level", product_context.get("inventory", 100))),
                    "demand_score": float(product_context.get("demand_score", 1.0)),
                    "seasonality_factor": float(product_context.get("seasonality_factor", 1.0)),
                    "discount_percentage": discount,
                    "historical_conversion_rate": float(product_context.get("historical_conversion_rate", 0.04)),
                    "margin": float(product_context.get("margin", 0.35)),
                    "price_elasticity_score": float(product_context.get("price_elasticity_score", 1.0)),
                    "product_category": str(product_context.get("product_category", product_context.get("category", "Unknown"))),
                }
            ]
        )

    def _select_model(self):
        if self.backend in {"auto", "xgboost"}:
            try:
                from xgboost import XGBClassifier

                return (
                    XGBClassifier(
                        n_estimators=220,
                        learning_rate=0.045,
                        max_depth=4,
                        subsample=0.9,
                        colsample_bytree=0.85,
                        objective="binary:logistic",
                        eval_metric="logloss",
                        tree_method="hist",
                        random_state=self.random_state,
                    ),
                    "xgboost_classifier",
                )
            except ImportError:
                if self.backend == "xgboost":
                    raise

        return GradientBoostingClassifier(random_state=self.random_state), "sklearn_gradient_boosting"
