from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


class CustomerSegmentation:
    """Customer analytics layer for lifecycle, value, and churn-oriented segmentation."""

    FEATURES = [
        "views",
        "clicks",
        "cart_additions",
        "purchases",
        "total_revenue",
        "ctr",
        "conversion_rate",
        "aov",
        "ltv_estimate",
        "avg_discount_seen",
        "days_since_last_event",
        "churn_risk",
    ]

    def __init__(self, random_state: int = 42) -> None:
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.clusterer = KMeans(n_clusters=5, n_init="auto", random_state=random_state)
        self.segment_table: pd.DataFrame | None = None

    def fit_predict(self, user_features: pd.DataFrame) -> pd.DataFrame:
        frame = user_features.copy()
        numeric = frame[self.FEATURES].replace([np.inf, -np.inf], 0).fillna(0)
        scaled = self.scaler.fit_transform(numeric)
        frame["behavior_cluster"] = self.clusterer.fit_predict(scaled)
        frame["customer_segment"] = frame.apply(self._segment_rule, axis=1)
        frame["retention_signal"] = np.where(
            frame["churn_risk"] < 0.35,
            "healthy",
            np.where(frame["purchases"] > 0, "save_offer", "nurture"),
        )
        self.segment_table = frame
        return frame

    def get_user_segment(self, user_id: str) -> dict:
        if self.segment_table is None:
            raise RuntimeError("Customer segmentation has not been fitted")
        row = self.segment_table.loc[self.segment_table["user_id"] == user_id]
        if row.empty:
            return {
                "user_id": user_id,
                "customer_segment": "unknown",
                "churn_risk": 0.0,
                "ltv_estimate": 0.0,
                "retention_signal": "insufficient_history",
            }
        record = row.iloc[0].to_dict()
        return {
            "user_id": user_id,
            "customer_segment": record["customer_segment"],
            "churn_risk": round(float(record["churn_risk"]), 4),
            "ltv_estimate": round(float(record["ltv_estimate"]), 2),
            "conversion_rate": round(float(record["conversion_rate"]), 4),
            "retention_signal": record["retention_signal"],
        }

    @staticmethod
    def segment_summary(segment_table: pd.DataFrame) -> pd.DataFrame:
        return (
            segment_table.groupby("customer_segment")
            .agg(
                users=("user_id", "nunique"),
                conversion_rate=("conversion_rate", "mean"),
                aov=("aov", "mean"),
                ltv_estimate=("ltv_estimate", "mean"),
                churn_risk=("churn_risk", "mean"),
            )
            .reset_index()
            .sort_values("users", ascending=False)
        )

    @staticmethod
    def _segment_rule(row: pd.Series) -> str:
        if row["total_events"] <= 3:
            return "new users"
        if row["purchases"] >= 5 and row["churn_risk"] < 0.28:
            return "loyal buyers"
        if row["ltv_estimate"] >= 300 and row["purchases"] >= 3:
            return "high-value customers"
        if row["avg_discount_seen"] >= 0.18 and row["conversion_rate"] >= 0.05:
            return "discount-sensitive customers"
        if row["views"] >= 12 and row["purchases"] <= 1:
            return "frequent browsers"
        if row["churn_risk"] >= 0.62:
            return "likely-to-churn customers"
        return "growth customers"
