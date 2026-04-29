from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
FEATURE_STORE_DIR = DATA_DIR / "feature_store"
SAMPLE_DATA_DIR = DATA_DIR / "sample"
SYNTHETIC_DATA_DIR = DATA_DIR / "synthetic"
MODEL_DIR = ROOT_DIR / "models"
REPORTS_DIR = ROOT_DIR / "reports"


@dataclass(frozen=True)
class PipelineConfig:
    """Runtime configuration for local synthetic training runs."""

    seed: int = int(os.getenv("PIPELINE_SEED", "42"))
    n_users: int = int(os.getenv("N_USERS", "5000"))
    n_products: int = int(os.getenv("N_PRODUCTS", "800"))
    n_events: int = int(os.getenv("N_EVENTS", "120000"))
    training_window_days: int = int(os.getenv("TRAINING_WINDOW_DAYS", "180"))
    recommendation_k: int = int(os.getenv("RECOMMENDATION_K", "10"))
    retrieval_backend: str = os.getenv("RETRIEVAL_BACKEND", "auto")
    ranking_backend: str = os.getenv("RANKING_BACKEND", "auto")
    pricing_backend: str = os.getenv("PRICING_BACKEND", "auto")
    forecasting_backend: str = os.getenv("FORECASTING_BACKEND", "auto")
    retrieval_epochs: int = int(os.getenv("RETRIEVAL_EPOCHS", "4"))
    retrieval_batch_size: int = int(os.getenv("RETRIEVAL_BATCH_SIZE", "1024"))
    start_date: str = os.getenv("START_DATE", "2025-01-01")


def ensure_project_directories() -> None:
    """Create the runtime directories used by the pipeline."""

    for directory in [
        RAW_DATA_DIR,
        PROCESSED_DATA_DIR,
        FEATURE_STORE_DIR,
        SAMPLE_DATA_DIR,
        SYNTHETIC_DATA_DIR,
        MODEL_DIR,
        REPORTS_DIR,
    ]:
        directory.mkdir(parents=True, exist_ok=True)
