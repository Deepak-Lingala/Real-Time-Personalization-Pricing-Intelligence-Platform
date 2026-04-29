from __future__ import annotations

import json

import numpy as np
import pandas as pd


EVENT_SCORE = {"view": 1.0, "search": 1.5, "click": 2.0, "add_to_cart": 4.0, "purchase": 8.0}


class FeatureEngineer:
    """Build reusable feature tables that mimic a compact offline feature store."""

    def build_user_features(self, events: pd.DataFrame) -> pd.DataFrame:
        frame = events.copy()
        frame["event_score"] = frame["event_type"].map(EVENT_SCORE).fillna(0)
        frame["net_price"] = frame["product_price"] * (1 - frame["discount_percentage"])
        frame["purchase_revenue"] = frame["net_price"] * frame["purchase_label"]

        counts = frame.pivot_table(index="user_id", columns="event_type", values="product_id", aggfunc="count", fill_value=0)
        for column in EVENT_SCORE:
            if column not in counts:
                counts[column] = 0
        counts = counts.rename(
            columns={
                "view": "views",
                "click": "clicks",
                "add_to_cart": "cart_additions",
                "purchase": "purchases",
            }
        )

        aggregates = frame.groupby("user_id").agg(
            total_events=("event_type", "size"),
            total_revenue=("purchase_revenue", "sum"),
            avg_discount_seen=("discount_percentage", "mean"),
            avg_price_seen=("product_price", "mean"),
            unique_products=("product_id", "nunique"),
            last_event_ts=("timestamp", "max"),
        )
        category_counts = (
            frame.groupby(["user_id", "product_category"])["event_score"].sum().rename("score").reset_index()
        )
        category_total = category_counts.groupby("user_id")["score"].transform("sum")
        category_counts["affinity"] = category_counts["score"] / category_total
        affinity = category_counts.groupby("user_id").apply(
            lambda rows: json.dumps(dict(zip(rows["product_category"], rows["affinity"].round(4))))
        )
        dominant_category = category_counts.sort_values(["user_id", "score"], ascending=[True, False]).drop_duplicates("user_id")

        user_features = aggregates.join(counts, how="left")
        user_features["ctr"] = user_features["clicks"] / user_features["views"].clip(lower=1)
        user_features["conversion_rate"] = user_features["purchases"] / user_features["views"].clip(lower=1)
        user_features["aov"] = user_features["total_revenue"] / user_features["purchases"].clip(lower=1)
        user_features["ltv_estimate"] = user_features["aov"] * np.maximum(1, user_features["purchases"]) * 1.8
        user_features["category_affinity"] = affinity
        user_features["dominant_category"] = dominant_category.set_index("user_id")["product_category"]
        max_ts = pd.to_datetime(frame["timestamp"]).max()
        user_features["days_since_last_event"] = (
            max_ts - pd.to_datetime(user_features["last_event_ts"])
        ).dt.days.clip(lower=0)
        user_features["churn_risk"] = np.clip(
            0.15
            + 0.018 * user_features["days_since_last_event"]
            - 0.08 * np.log1p(user_features["purchases"])
            - 0.05 * user_features["ctr"],
            0,
            0.98,
        )
        return user_features.reset_index()

    def build_product_features(self, events: pd.DataFrame, catalog: pd.DataFrame) -> pd.DataFrame:
        frame = events.copy()
        frame["net_price"] = frame["product_price"] * (1 - frame["discount_percentage"])
        product_events = frame.groupby("product_id").agg(
            views=("event_type", lambda values: int((values == "view").sum())),
            clicks=("event_type", lambda values: int((values == "click").sum())),
            cart_additions=("event_type", lambda values: int((values == "add_to_cart").sum())),
            purchases=("purchase_label", "sum"),
            revenue=("net_price", lambda values: float(values[frame.loc[values.index, "purchase_label"] == 1].sum())),
            avg_discount=("discount_percentage", "mean"),
            competitor_price=("competitor_price", "mean"),
            inventory_level=("inventory_level", "mean"),
        )
        product_features = catalog.merge(product_events, on="product_id", how="left").fillna(0)
        product_features["ctr"] = product_features["clicks"] / product_features["views"].clip(lower=1)
        product_features["conversion_rate"] = product_features["purchases"] / product_features["views"].clip(lower=1)
        product_features["demand_score"] = (
            0.45 * np.log1p(product_features["purchases"])
            + 0.2 * product_features["ctr"]
            + 0.2 * product_features["conversion_rate"]
            + 0.15 * product_features["rating"] / 5
        )
        product_features["competitor_price_index"] = (
            product_features["competitor_price"] / product_features["price"].clip(lower=1)
        )
        return product_features

    def build_session_features(self, events: pd.DataFrame) -> pd.DataFrame:
        frame = events.copy()
        frame["timestamp"] = pd.to_datetime(frame["timestamp"])
        frame["event_score"] = frame["event_type"].map(EVENT_SCORE).fillna(0)
        session_features = frame.groupby("session_id").agg(
            user_id=("user_id", "first"),
            session_start=("timestamp", "min"),
            session_end=("timestamp", "max"),
            events=("event_type", "size"),
            searches=("event_type", lambda values: int((values == "search").sum())),
            views=("event_type", lambda values: int((values == "view").sum())),
            clicks=("event_type", lambda values: int((values == "click").sum())),
            cart_additions=("event_type", lambda values: int((values == "add_to_cart").sum())),
            purchases=("purchase_label", "sum"),
            avg_dwell_time=("dwell_time_seconds", "mean"),
            max_event_score=("event_score", "max"),
            unique_products=("product_id", "nunique"),
        )
        session_features["duration_seconds"] = (
            session_features["session_end"] - session_features["session_start"]
        ).dt.total_seconds().clip(lower=0)
        session_features["session_conversion_rate"] = session_features["purchases"] / session_features["views"].clip(lower=1)
        return session_features.reset_index()

    def build_inventory_features(self, inventory: pd.DataFrame, product_features: pd.DataFrame) -> pd.DataFrame:
        frame = inventory.merge(
            product_features[["product_id", "category", "demand_score", "conversion_rate", "purchases"]],
            on="product_id",
            how="left",
        )
        frame["stockout_risk_score"] = (
            (frame["reorder_point"] / frame["inventory_level"].clip(lower=1))
            * (1 + frame["demand_score"].fillna(0))
        ).clip(0, 10)
        frame["stockout_risk_bucket"] = pd.cut(
            frame["stockout_risk_score"],
            bins=[-0.1, 0.8, 1.5, 10],
            labels=["low", "medium", "high"],
        ).astype(str)
        return frame

    def build_interaction_matrix(self, events: pd.DataFrame) -> pd.DataFrame:
        frame = events.copy()
        frame["interaction_score"] = frame["event_type"].map(EVENT_SCORE).fillna(0)
        return frame.pivot_table(
            index="user_id",
            columns="product_id",
            values="interaction_score",
            aggfunc="sum",
            fill_value=0,
        )

    def build_pricing_frame(self, events: pd.DataFrame, demand: pd.DataFrame) -> pd.DataFrame:
        frame = events.copy()
        frame["date"] = pd.to_datetime(frame["timestamp"]).dt.date.astype(str)
        demand_features = demand[["date", "product_id", "daily_sales"]]
        merged = frame.merge(demand_features, on=["date", "product_id"], how="left")
        merged["historical_conversion_rate"] = (
            merged.groupby("product_id")["purchase_label"].transform("mean").fillna(0)
        )
        merged["demand_score"] = np.log1p(merged["daily_sales"].fillna(0)) * merged["seasonality_factor"].fillna(1)
        merged["net_price"] = merged["product_price"] * (1 - merged["discount_percentage"])
        merged["margin"] = 0.35
        merged["price_elasticity_score"] = 1.0
        return merged[
            [
                "product_id",
                "product_category",
                "competitor_price",
                "inventory_level",
                "demand_score",
                "seasonality_factor",
                "discount_percentage",
                "historical_conversion_rate",
                "margin",
                "price_elasticity_score",
                "net_price",
                "purchase_label",
                "daily_sales",
                "holiday_flag",
                "promotion_flag",
            ]
        ].copy()

    @staticmethod
    def save_feature_store(feature_tables: dict[str, pd.DataFrame], output_dir) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        for name, frame in feature_tables.items():
            frame.to_csv(output_dir / f"{name}.csv", index=False)
