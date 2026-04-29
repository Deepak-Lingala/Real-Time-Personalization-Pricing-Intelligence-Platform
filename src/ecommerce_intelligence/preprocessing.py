from __future__ import annotations

import pandas as pd


class EventPreprocessor:
    """Spark-style preprocessing steps expressed as pandas transforms for local portability."""

    def clean_events(self, events: pd.DataFrame) -> pd.DataFrame:
        frame = events.copy()
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce")
        frame = frame.dropna(subset=["user_id", "session_id", "timestamp", "event_type", "product_id"])
        frame = frame.drop_duplicates(subset=["user_id", "session_id", "timestamp", "event_type", "product_id"])
        frame["discount_percentage"] = frame["discount_percentage"].fillna(0).clip(0, 0.9)
        frame["competitor_price"] = frame["competitor_price"].fillna(frame["product_price"])
        frame["dwell_time_seconds"] = frame.get("dwell_time_seconds", 0).fillna(0).clip(0, 900)
        frame["page_position"] = frame.get("page_position", 1).fillna(1).clip(1, 200)
        frame["purchase_label"] = frame["purchase_label"].astype(int)
        return frame.sort_values(["user_id", "timestamp"]).reset_index(drop=True)

    def sessionize(self, events: pd.DataFrame, timeout_minutes: int = 30) -> pd.DataFrame:
        frame = events.copy().sort_values(["user_id", "timestamp"])
        frame["timestamp"] = pd.to_datetime(frame["timestamp"])
        gap = frame.groupby("user_id")["timestamp"].diff().dt.total_seconds().div(60)
        new_session = gap.isna() | (gap > timeout_minutes)
        frame["derived_session_number"] = new_session.groupby(frame["user_id"]).cumsum().astype(int)
        frame["derived_session_id"] = (
            "DS-"
            + frame["user_id"].astype(str)
            + "-"
            + frame["derived_session_number"].astype(str).str.zfill(4)
        )
        return frame

    def data_quality_summary(self, raw_events: pd.DataFrame, cleaned_events: pd.DataFrame) -> dict[str, float]:
        duplicate_rate = 1 - len(raw_events.drop_duplicates(subset=["user_id", "session_id", "timestamp", "event_type", "product_id"])) / max(len(raw_events), 1)
        missing_rate = raw_events[["search_query", "competitor_price"]].isna().mean().mean()
        return {
            "raw_event_count": float(len(raw_events)),
            "clean_event_count": float(len(cleaned_events)),
            "duplicate_event_rate": round(float(duplicate_rate), 4),
            "tracked_missing_value_rate": round(float(missing_rate), 4),
            "records_removed": float(max(len(raw_events) - len(cleaned_events), 0)),
        }

