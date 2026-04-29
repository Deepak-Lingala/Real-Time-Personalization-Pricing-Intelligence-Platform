import pandas as pd

from ecommerce_intelligence.feature_store import FeastStyleFeatureStore


def test_feature_store_materializes_offline_online_and_registry(tmp_path) -> None:
    store = FeastStyleFeatureStore(tmp_path, version="test")
    metadata = store.materialize(
        {
            "user_features": (
                pd.DataFrame([{"user_id": "U1", "ctr": 0.2, "ltv_estimate": 120.0}]),
                "user_id",
            )
        }
    )

    assert metadata[0].name == "user_features"
    assert (tmp_path / "offline" / "test" / "user_features.csv").exists()
    assert (tmp_path / "online" / "test" / "user_features.json").exists()
    assert store.load_registry()[0]["entity"] == "user_id"

