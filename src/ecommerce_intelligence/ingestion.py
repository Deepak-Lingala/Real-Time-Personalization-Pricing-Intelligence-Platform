from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import pandas as pd

from ecommerce_intelligence.data_generator import SyntheticDataBundle


@dataclass(frozen=True)
class IngestedSources:
    batch_tables: dict[str, pd.DataFrame]
    event_stream: pd.DataFrame


class BatchStreamingIngestionSimulator:
    """Simulate warehouse-style batch loads plus event-stream microbatches."""

    def __init__(self, microbatch_size: int = 5000) -> None:
        self.microbatch_size = microbatch_size

    def ingest(self, bundle: SyntheticDataBundle) -> IngestedSources:
        return IngestedSources(
            batch_tables=self.batch_tables(bundle),
            event_stream=self.collect_event_stream(bundle.events),
        )

    @staticmethod
    def batch_tables(bundle: SyntheticDataBundle) -> dict[str, pd.DataFrame]:
        return {
            "user_profiles": bundle.users.copy(),
            "product_catalog": bundle.product_catalog.copy(),
            "pricing": bundle.pricing.copy(),
            "inventory": bundle.inventory.copy(),
            "demand": bundle.demand.copy(),
            "product_reviews": bundle.reviews.copy(),
            "recommendation_features": bundle.recommendation_features.copy(),
        }

    def event_microbatches(self, events: pd.DataFrame) -> Iterator[pd.DataFrame]:
        frame = events.copy()
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce")
        frame = frame.sort_values("timestamp").reset_index(drop=True)
        for microbatch_id, start in enumerate(range(0, len(frame), self.microbatch_size), start=1):
            microbatch = frame.iloc[start : start + self.microbatch_size].copy()
            microbatch["stream_offset"] = range(start, start + len(microbatch))
            microbatch["microbatch_id"] = microbatch_id
            microbatch["ingestion_timestamp"] = pd.Timestamp.utcnow().isoformat()
            yield microbatch

    def collect_event_stream(self, events: pd.DataFrame) -> pd.DataFrame:
        microbatches = list(self.event_microbatches(events))
        if not microbatches:
            return pd.DataFrame(columns=list(events.columns) + ["stream_offset", "microbatch_id"])
        return pd.concat(microbatches, ignore_index=True)
