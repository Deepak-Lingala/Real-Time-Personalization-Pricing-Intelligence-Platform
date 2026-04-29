from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ecommerce_intelligence.config import PipelineConfig  # noqa: E402
from ecommerce_intelligence.pipeline import run_pipeline  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the synthetic e-commerce ML pipeline.")
    parser.add_argument("--users", type=int, default=PipelineConfig.n_users)
    parser.add_argument("--products", type=int, default=PipelineConfig.n_products)
    parser.add_argument("--events", type=int, default=PipelineConfig.n_events)
    parser.add_argument("--days", type=int, default=PipelineConfig.training_window_days)
    parser.add_argument("--seed", type=int, default=PipelineConfig.seed)
    parser.add_argument("--retrieval-backend", choices=["auto", "torch", "sklearn"], default=PipelineConfig.retrieval_backend)
    parser.add_argument(
        "--ranking-backend",
        choices=["auto", "xgboost", "lightgbm", "sklearn"],
        default=PipelineConfig.ranking_backend,
    )
    parser.add_argument(
        "--pricing-backend",
        choices=["auto", "xgboost", "sklearn"],
        default=PipelineConfig.pricing_backend,
    )
    parser.add_argument(
        "--forecasting-backend",
        choices=["auto", "lightgbm", "sklearn"],
        default=PipelineConfig.forecasting_backend,
    )
    parser.add_argument("--retrieval-epochs", type=int, default=PipelineConfig.retrieval_epochs)
    parser.add_argument("--retrieval-batch-size", type=int, default=PipelineConfig.retrieval_batch_size)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = PipelineConfig(
        seed=args.seed,
        n_users=args.users,
        n_products=args.products,
        n_events=args.events,
        training_window_days=args.days,
        retrieval_backend=args.retrieval_backend,
        ranking_backend=args.ranking_backend,
        pricing_backend=args.pricing_backend,
        forecasting_backend=args.forecasting_backend,
        retrieval_epochs=args.retrieval_epochs,
        retrieval_batch_size=args.retrieval_batch_size,
    )
    summary = run_pipeline(config)
    print(
        json.dumps(
            {
                "status": "complete",
                "kpis": summary["kpis"],
                "sample_user": summary["sample_user"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
