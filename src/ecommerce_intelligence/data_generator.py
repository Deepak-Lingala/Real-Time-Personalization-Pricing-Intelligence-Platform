from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta

import numpy as np
import pandas as pd


CATEGORY_PROFILES = {
    "Electronics": {
        "base_price": 340,
        "terms": ["wireless", "smart", "portable", "high resolution", "gaming"],
        "brands": ["Northstar", "VoltEdge", "Auralux", "PixelForge"],
    },
    "Home": {
        "base_price": 95,
        "terms": ["minimal", "durable", "energy saving", "compact", "modern"],
        "brands": ["Hearthly", "NestForm", "LumaHome", "Civica"],
    },
    "Fashion": {
        "base_price": 58,
        "terms": ["premium cotton", "tailored", "seasonal", "streetwear", "classic"],
        "brands": ["ModeLab", "UrbanThread", "LinenLane", "Arden"],
    },
    "Beauty": {
        "base_price": 42,
        "terms": ["hydrating", "clean formula", "brightening", "daily care", "refillable"],
        "brands": ["Glowry", "Sero", "Bloomskin", "Aster"],
    },
    "Sports": {
        "base_price": 78,
        "terms": ["breathable", "training", "lightweight", "recovery", "outdoor"],
        "brands": ["StrideIQ", "PeakLab", "MotionWorks", "Fieldstone"],
    },
    "Grocery": {
        "base_price": 18,
        "terms": ["organic", "family size", "high protein", "fresh", "pantry"],
        "brands": ["HarvestHill", "DailyCrop", "NorthPantry", "KindTable"],
    },
}

LOCATIONS = ["CA", "TX", "NY", "FL", "IL", "WA", "AZ", "GA", "NC", "CO"]
DEVICES = ["mobile", "desktop", "tablet"]
EVENT_WEIGHTS = {"view": 0.50, "click": 0.22, "add_to_cart": 0.10, "purchase": 0.07, "search": 0.11}


@dataclass(frozen=True)
class SyntheticDataBundle:
    users: pd.DataFrame
    product_catalog: pd.DataFrame
    events: pd.DataFrame
    demand: pd.DataFrame
    reviews: pd.DataFrame
    pricing: pd.DataFrame
    inventory: pd.DataFrame
    recommendation_features: pd.DataFrame


class SyntheticEcommerceGenerator:
    """Generate synthetic e-commerce data with realistic behavioral correlations."""

    def __init__(self, seed: int = 42) -> None:
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def generate_product_catalog(self, n_products: int) -> pd.DataFrame:
        categories = list(CATEGORY_PROFILES)
        category_weights = np.array([0.21, 0.18, 0.19, 0.12, 0.14, 0.16])
        product_categories = self.rng.choice(categories, size=n_products, p=category_weights)

        rows = []
        for index, category in enumerate(product_categories, start=1):
            profile = CATEGORY_PROFILES[category]
            price = float(
                np.clip(
                    self.rng.lognormal(mean=np.log(profile["base_price"]), sigma=0.34),
                    5,
                    1800,
                )
            )
            brand = str(self.rng.choice(profile["brands"]))
            adjective = str(self.rng.choice(profile["terms"]))
            margin = float(np.clip(self.rng.normal(0.38, 0.08), 0.16, 0.64))
            review_count = int(self.rng.gamma(shape=3.2, scale=95))
            rating = float(np.clip(self.rng.normal(4.15, 0.43), 2.7, 5.0))

            rows.append(
                {
                    "product_id": f"P{index:06d}",
                    "category": category,
                    "product_category": category,
                    "brand": brand,
                    "price": round(price, 2),
                    "product_price": round(price, 2),
                    "margin": round(margin, 3),
                    "product_margin": round(margin, 3),
                    "inventory": int(self.rng.integers(12, 900)),
                    "rating": round(rating, 2),
                    "review_count": review_count,
                    "price_elasticity_score": round(float(np.clip(self.rng.normal(1.15, 0.34), 0.25, 2.3)), 3),
                    "image_feature_vector": self._vector_json(8),
                    "text_description": (
                        f"{brand} {category.lower()} item with {adjective} design, "
                        f"{review_count} reviews, and {rating:.1f} average rating."
                    ),
                }
            )

        return pd.DataFrame(rows)

    def generate_users(self, n_users: int) -> pd.DataFrame:
        categories = list(CATEGORY_PROFILES)
        rows = []
        for index in range(1, n_users + 1):
            affinity = self.rng.dirichlet(alpha=np.array([1.4, 1.1, 1.2, 0.8, 0.9, 1.0]))
            favorite_category = categories[int(np.argmax(affinity))]
            discount_sensitivity = float(np.clip(self.rng.beta(2.2, 3.0), 0.03, 0.95))
            value_tier = str(self.rng.choice(["budget", "mid_market", "premium"], p=[0.38, 0.45, 0.17]))
            rows.append(
                {
                    "user_id": f"U{index:06d}",
                    "user_location": str(self.rng.choice(LOCATIONS)),
                    "device_type": str(self.rng.choice(DEVICES, p=[0.63, 0.29, 0.08])),
                    "user_segment": value_tier,
                    "signup_date": (pd.Timestamp.today().normalize() - pd.Timedelta(days=int(self.rng.integers(5, 1440)))).date().isoformat(),
                    "loyalty_tier": str(self.rng.choice(["bronze", "silver", "gold", "platinum"], p=[0.42, 0.31, 0.19, 0.08])),
                    "home_location": str(self.rng.choice(LOCATIONS)),
                    "preferred_device": str(self.rng.choice(DEVICES, p=[0.63, 0.29, 0.08])),
                    "favorite_category": favorite_category,
                    "category_affinity": dict(zip(categories, np.round(affinity, 4))),
                    "discount_sensitivity": round(discount_sensitivity, 3),
                    "value_tier": value_tier,
                    "signup_age_days": int(self.rng.integers(5, 1440)),
                }
            )
        return pd.DataFrame(rows)

    def generate_events(
        self,
        users: pd.DataFrame,
        product_catalog: pd.DataFrame,
        n_events: int,
        start_date: str,
        days: int,
    ) -> pd.DataFrame:
        start = pd.Timestamp(start_date)
        products_by_category = {
            category: frame.reset_index(drop=True)
            for category, frame in product_catalog.groupby("category")
        }
        user_lookup = users.set_index("user_id")
        categories = [category for category in CATEGORY_PROFILES if category in products_by_category]
        event_types = list(EVENT_WEIGHTS)
        event_probs = np.array(list(EVENT_WEIGHTS.values()))
        rows = []

        user_ids = users["user_id"].to_numpy()
        for event_index in range(n_events):
            user_id = str(self.rng.choice(user_ids))
            user = user_lookup.loc[user_id]
            affinity = np.array([user["category_affinity"][category] for category in categories])
            category = str(self.rng.choice(categories, p=affinity / affinity.sum()))
            product_frame = products_by_category[category]
            popularity = (
                product_frame["rating"].to_numpy()
                * np.log1p(product_frame["review_count"].to_numpy())
                * np.sqrt(product_frame["inventory"].to_numpy())
            )
            product = product_frame.iloc[int(self.rng.choice(len(product_frame), p=popularity / popularity.sum()))]

            timestamp = start + timedelta(
                days=int(self.rng.integers(0, days)),
                minutes=int(self.rng.integers(0, 24 * 60)),
            )
            holiday_flag = int(timestamp.month == 12 or timestamp.strftime("%m-%d") in {"11-28", "11-29"})
            promotion_flag = int(self.rng.random() < (0.18 + 0.16 * holiday_flag))
            discount = float(np.clip(self.rng.beta(2.0, 8.0) * 0.55, 0, 0.55))
            if promotion_flag:
                discount = float(np.clip(discount + self.rng.uniform(0.06, 0.18), 0, 0.65))

            competitor_price = float(product["price"] * self.rng.normal(1.0, 0.07))
            inventory_level = int(max(0, product["inventory"] + self.rng.normal(0, 35)))
            seasonality = self._seasonality_factor(timestamp)
            base_event = str(self.rng.choice(event_types, p=event_probs))
            purchase_probability = self._purchase_probability(
                event_type=base_event,
                discount=discount,
                discount_sensitivity=float(user["discount_sensitivity"]),
                price=float(product["price"]),
                competitor_price=competitor_price,
                inventory_level=inventory_level,
                rating=float(product["rating"]),
                seasonality=seasonality,
                promotion_flag=promotion_flag,
            )
            event_type = "purchase" if self.rng.random() < purchase_probability else base_event
            search_query = self._search_query(category)

            rows.append(
                {
                    "event_id": f"E{event_index:09d}",
                    "user_id": user_id,
                    "session_id": f"S{user_id[-4:]}-{timestamp.strftime('%Y%m%d')}-{event_index % 97:02d}",
                    "timestamp": timestamp,
                    "event_type": event_type,
                    "product_id": product["product_id"],
                    "product_category": category,
                    "product_price": round(float(product["price"]), 2),
                    "discount_percentage": round(discount, 3),
                    "competitor_price": round(competitor_price, 2),
                    "inventory_level": inventory_level,
                    "rating": float(product["rating"]),
                    "review_count": int(product["review_count"]),
                    "user_location": user["home_location"],
                    "device_type": user["preferred_device"] if self.rng.random() < 0.82 else str(self.rng.choice(DEVICES)),
                    "search_query": search_query,
                    "dwell_time_seconds": int(np.clip(self.rng.gamma(shape=2.4, scale=18), 2, 420)),
                    "page_position": int(self.rng.zipf(1.7)),
                    "purchase_label": int(event_type == "purchase"),
                    "return_label": int(event_type == "purchase" and self.rng.random() < 0.065),
                    "promotion_flag": promotion_flag,
                    "holiday_flag": holiday_flag,
                    "seasonality_factor": round(seasonality, 3),
                }
            )

        frame = pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)
        frame.loc[self.rng.random(len(frame)) < 0.012, "search_query"] = np.nan
        frame.loc[self.rng.random(len(frame)) < 0.008, "competitor_price"] = np.nan
        duplicate_count = max(1, int(len(frame) * 0.006))
        duplicates = frame.sample(n=duplicate_count, random_state=self.seed).copy()
        duplicates["event_id"] = duplicates["event_id"] + "-DUP"
        return pd.concat([frame, duplicates], ignore_index=True).sort_values("timestamp").reset_index(drop=True)

    def generate_demand(
        self,
        events: pd.DataFrame,
        product_catalog: pd.DataFrame,
        start_date: str,
        days: int,
    ) -> pd.DataFrame:
        dates = pd.date_range(start=start_date, periods=days, freq="D")
        purchases = events.loc[events["purchase_label"] == 1].copy()
        purchases["date"] = pd.to_datetime(purchases["timestamp"]).dt.date
        actual_sales = purchases.groupby(["product_id", "date"]).size().rename("observed_sales")

        rows = []
        for product in product_catalog.itertuples(index=False):
            category_bias = {
                "Electronics": 1.25,
                "Home": 0.92,
                "Fashion": 1.08,
                "Beauty": 0.98,
                "Sports": 0.9,
                "Grocery": 1.45,
            }[product.category]
            base_rate = max(0.1, np.log1p(product.review_count) * product.rating * category_bias / 10)
            for date in dates:
                holiday_flag = int(date.month == 12 or date.strftime("%m-%d") in {"11-28", "11-29"})
                promotion_flag = int(self.rng.random() < (0.13 + 0.25 * holiday_flag))
                seasonality = self._seasonality_factor(date)
                expected = base_rate * seasonality * (1.0 + 0.35 * promotion_flag + 0.28 * holiday_flag)
                observed = int(actual_sales.get((product.product_id, date.date()), 0))
                synthetic_noise = int(self.rng.poisson(expected))
                daily_sales = max(observed, synthetic_noise)
                rows.append(
                    {
                        "date": date.date().isoformat(),
                        "product_id": product.product_id,
                        "category": product.category,
                        "daily_sales": int(daily_sales),
                        "seasonality_factor": round(seasonality, 3),
                        "holiday_flag": holiday_flag,
                        "promotion_flag": promotion_flag,
                    }
                )

        return pd.DataFrame(rows)

    def generate_pricing(self, product_catalog: pd.DataFrame) -> pd.DataFrame:
        rows = []
        for product in product_catalog.itertuples(index=False):
            current_price = float(product.price)
            competitor_price = current_price * float(self.rng.normal(1.0, 0.08))
            discount = float(np.clip(self.rng.beta(2.0, 7.5) * 0.55, 0, 0.6))
            rows.append(
                {
                    "product_id": product.product_id,
                    "current_price": round(current_price, 2),
                    "competitor_price": round(competitor_price, 2),
                    "discount_percentage": round(discount, 3),
                    "margin": float(product.margin),
                    "price_elasticity_score": float(product.price_elasticity_score),
                    "minimum_margin_threshold": round(max(0.12, float(product.margin) - 0.08), 3),
                }
            )
        return pd.DataFrame(rows)

    def generate_inventory(self, product_catalog: pd.DataFrame, demand: pd.DataFrame) -> pd.DataFrame:
        demand_lookup = demand.groupby("product_id")["daily_sales"].mean().to_dict()
        rows = []
        for product in product_catalog.itertuples(index=False):
            forecast = float(demand_lookup.get(product.product_id, 1.0) * self.rng.uniform(5.8, 8.8))
            reorder_point = int(max(10, forecast * self.rng.uniform(0.7, 1.4)))
            inventory_level = int(product.inventory)
            rows.append(
                {
                    "product_id": product.product_id,
                    "inventory_level": inventory_level,
                    "reorder_point": reorder_point,
                    "supplier_lead_time": int(self.rng.integers(2, 21)),
                    "stockout_flag": int(inventory_level <= reorder_point * 0.35),
                    "demand_forecast": round(forecast, 2),
                    "inventory_risk": "high" if inventory_level <= reorder_point else "normal",
                }
            )
        return pd.DataFrame(rows)

    def generate_reviews(self, product_catalog: pd.DataFrame, max_reviews_per_product: int = 4) -> pd.DataFrame:
        sentiments = ["positive", "neutral", "negative"]
        rows = []
        for product in product_catalog.itertuples(index=False):
            review_count = int(self.rng.integers(1, max_reviews_per_product + 1))
            for review_index in range(review_count):
                sentiment = str(self.rng.choice(sentiments, p=[0.72, 0.19, 0.09]))
                rows.append(
                    {
                        "review_id": f"R{product.product_id[-6:]}-{review_index:02d}",
                        "product_id": product.product_id,
                        "rating": round(float(np.clip(self.rng.normal(float(product.rating), 0.55), 1, 5)), 1),
                        "review_sentiment": sentiment,
                        "review_text": f"{sentiment} synthetic review for {product.brand} {product.category}.",
                    }
                )
        return pd.DataFrame(rows)

    def generate_recommendation_features(self, users: pd.DataFrame, product_catalog: pd.DataFrame, sample_rows: int = 5000) -> pd.DataFrame:
        rows = []
        user_ids = users["user_id"].to_numpy()
        product_ids = product_catalog["product_id"].to_numpy()
        product_categories = product_catalog.set_index("product_id")["category"].to_dict()
        user_affinity = users.set_index("user_id")["category_affinity"].to_dict()
        for index in range(min(sample_rows, len(user_ids) * 4)):
            user_id = str(self.rng.choice(user_ids))
            product_id = str(self.rng.choice(product_ids))
            category = str(product_categories[product_id])
            affinity = float(user_affinity[user_id].get(category, 0.1))
            similarity = float(np.clip(self.rng.normal(affinity, 0.12), 0, 1))
            rows.append(
                {
                    "user_id": user_id,
                    "product_id": product_id,
                    "user_embedding": self._vector_json(8),
                    "product_embedding": self._vector_json(8),
                    "similarity_score": round(similarity, 4),
                    "category_affinity": round(affinity, 4),
                    "purchase_probability": round(float(np.clip(0.02 + similarity * 0.28 + self.rng.normal(0, 0.03), 0.001, 0.9)), 4),
                }
            )
        return pd.DataFrame(rows)

    def generate_all(
        self,
        n_users: int,
        n_products: int,
        n_events: int,
        start_date: str,
        days: int,
    ) -> SyntheticDataBundle:
        product_catalog = self.generate_product_catalog(n_products)
        users = self.generate_users(n_users)
        events = self.generate_events(users, product_catalog, n_events, start_date, days)
        demand = self.generate_demand(events, product_catalog, start_date, days)
        pricing = self.generate_pricing(product_catalog)
        inventory = self.generate_inventory(product_catalog, demand)
        reviews = self.generate_reviews(product_catalog)
        recommendation_features = self.generate_recommendation_features(users, product_catalog)
        return SyntheticDataBundle(
            users=users,
            product_catalog=product_catalog,
            events=events,
            demand=demand,
            reviews=reviews,
            pricing=pricing,
            inventory=inventory,
            recommendation_features=recommendation_features,
        )

    def _search_query(self, category: str) -> str:
        terms = CATEGORY_PROFILES[category]["terms"]
        modifiers = ["best", "near me", "premium", "discount", "new", "top rated", "fast shipping"]
        return f"{self.rng.choice(modifiers)} {self.rng.choice(terms)} {category.lower()}"

    def _vector_json(self, dimensions: int) -> str:
        vector = self.rng.normal(0, 1, dimensions)
        vector = vector / max(np.linalg.norm(vector), 1e-6)
        return "[" + ",".join(f"{value:.4f}" for value in vector) + "]"

    @staticmethod
    def _seasonality_factor(timestamp: pd.Timestamp) -> float:
        week = timestamp.isocalendar().week
        yearly = 1.0 + 0.18 * np.sin(2 * np.pi * week / 52)
        weekend = 1.08 if timestamp.dayofweek >= 5 else 0.97
        q4 = 1.18 if timestamp.month in {11, 12} else 1.0
        return float(yearly * weekend * q4)

    @staticmethod
    def _purchase_probability(
        event_type: str,
        discount: float,
        discount_sensitivity: float,
        price: float,
        competitor_price: float,
        inventory_level: int,
        rating: float,
        seasonality: float,
        promotion_flag: int,
    ) -> float:
        funnel_boost = {"view": 0.01, "click": 0.045, "add_to_cart": 0.18, "purchase": 0.65, "search": 0.025}[event_type]
        relative_value = np.clip((competitor_price - price * (1 - discount)) / max(competitor_price, 1), -0.4, 0.5)
        scarcity = 0.04 if inventory_level < 30 else 0.0
        score = (
            funnel_boost
            + 0.19 * discount * discount_sensitivity
            + 0.11 * relative_value
            + 0.035 * (rating - 4.0)
            + 0.025 * (seasonality - 1.0)
            + 0.035 * promotion_flag
            + scarcity
        )
        return float(np.clip(score, 0.002, 0.82))
