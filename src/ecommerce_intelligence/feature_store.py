from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class FeatureTableMetadata:
    name: str
    version: str
    entity: str
    row_count: int
    feature_count: int
    freshness_timestamp: str
    offline_path: str
    online_path: str


class FeastStyleFeatureStore:
    """A compact Feast-style offline/online feature store simulation."""

    def __init__(self, root_dir: Path, version: str = "v1") -> None:
        self.root_dir = root_dir
        self.version = version
        self.offline_dir = root_dir / "offline" / version
        self.online_dir = root_dir / "online" / version
        self.metadata_path = root_dir / "feature_registry.json"
        self.offline_dir.mkdir(parents=True, exist_ok=True)
        self.online_dir.mkdir(parents=True, exist_ok=True)

    def materialize(self, tables: dict[str, tuple[pd.DataFrame, str]]) -> list[FeatureTableMetadata]:
        metadata = []
        now = datetime.now(timezone.utc).isoformat()
        for name, (frame, entity) in tables.items():
            offline_path = self.offline_dir / f"{name}.csv"
            online_path = self.online_dir / f"{name}.json"
            frame.to_csv(offline_path, index=False)
            records = frame.head(5000).to_dict("records")
            online_path.write_text(json.dumps(records, default=str), encoding="utf-8")
            metadata.append(
                FeatureTableMetadata(
                    name=name,
                    version=self.version,
                    entity=entity,
                    row_count=int(len(frame)),
                    feature_count=int(len(frame.columns)),
                    freshness_timestamp=now,
                    offline_path=str(offline_path.as_posix()),
                    online_path=str(online_path.as_posix()),
                )
            )
        self.metadata_path.write_text(
            json.dumps([asdict(item) for item in metadata], indent=2),
            encoding="utf-8",
        )
        return metadata

    def load_registry(self) -> list[dict]:
        if not self.metadata_path.exists():
            return []
        return json.loads(self.metadata_path.read_text(encoding="utf-8"))

    def freshness_summary(self) -> list[dict]:
        rows = []
        for table in self.load_registry():
            rows.append(
                {
                    "feature_table": table["name"],
                    "version": table["version"],
                    "entity": table["entity"],
                    "row_count": table["row_count"],
                    "freshness_timestamp": table["freshness_timestamp"],
                    "status": "fresh",
                }
            )
        return rows

