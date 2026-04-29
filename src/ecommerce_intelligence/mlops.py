from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ModelRegistryEntry:
    model_name: str
    version: str
    training_date: str
    metrics: dict[str, float]
    deployment_status: str
    artifact_path: str
    parameters: dict[str, str | int | float] | None = None
    artifacts: list[str] | None = None
    registry_stage: str = "Staging"
    rollback_version: str | None = None
    tracking_backend: str = "json"
    mlflow_run_id: str | None = None


class ExperimentTracker:
    """MLflow-compatible tracker with a JSON registry fallback for portable demos."""

    def __init__(self, registry_path: Path) -> None:
        self.registry_path = registry_path
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)

    def log_model(
        self,
        model_name: str,
        version: str,
        metrics: dict[str, float],
        artifact_path: str,
        deployment_status: str = "staging",
        parameters: dict[str, str | int | float] | None = None,
        artifacts: list[str] | None = None,
        registry_stage: str = "Staging",
        rollback_version: str | None = None,
    ) -> ModelRegistryEntry:
        resolved_artifacts = artifacts or [artifact_path]
        mlflow_run_id = self._log_to_mlflow(
            model_name=model_name,
            version=version,
            metrics=metrics,
            artifact_path=artifact_path,
            deployment_status=deployment_status,
            parameters=parameters or {},
            artifacts=resolved_artifacts,
            registry_stage=registry_stage,
            rollback_version=rollback_version,
        )
        entry = ModelRegistryEntry(
            model_name=model_name,
            version=version,
            training_date=datetime.now(timezone.utc).isoformat(),
            metrics=metrics,
            deployment_status=deployment_status,
            artifact_path=artifact_path,
            parameters=parameters or {},
            artifacts=resolved_artifacts,
            registry_stage=registry_stage,
            rollback_version=rollback_version,
            tracking_backend="mlflow+json" if mlflow_run_id else "json",
            mlflow_run_id=mlflow_run_id,
        )
        registry = self.load_registry()
        registry.append(asdict(entry))
        self.registry_path.write_text(json.dumps(registry, indent=2), encoding="utf-8")
        return entry

    def load_registry(self) -> list[dict]:
        if not self.registry_path.exists():
            return []
        return json.loads(self.registry_path.read_text(encoding="utf-8"))

    def _log_to_mlflow(
        self,
        model_name: str,
        version: str,
        metrics: dict[str, float],
        artifact_path: str,
        deployment_status: str,
        parameters: dict[str, str | int | float],
        artifacts: list[str],
        registry_stage: str,
        rollback_version: str | None,
    ) -> str | None:
        try:
            import mlflow
        except ImportError:
            return None

        try:
            tracking_dir = (self.registry_path.parent / "mlruns").resolve()
            tracking_dir.mkdir(parents=True, exist_ok=True)
            mlflow.set_tracking_uri(tracking_dir.as_uri())
            mlflow.set_experiment("real_time_personalization_pricing_intelligence")
            project_root = self.registry_path.parent.parent
            with mlflow.start_run(run_name=f"{model_name}-{version}") as run:
                mlflow.log_params({key: value for key, value in parameters.items()})
                numeric_metrics = {
                    key: float(value)
                    for key, value in metrics.items()
                    if isinstance(value, (int, float, np.integer, np.floating))
                }
                mlflow.log_metrics(numeric_metrics)
                mlflow.set_tags(
                    {
                        "model_name": model_name,
                        "version": version,
                        "deployment_status": deployment_status,
                        "registry_stage": registry_stage,
                        "rollback_version": rollback_version or "",
                        "artifact_path": artifact_path,
                    }
                )
                for artifact in artifacts:
                    path = project_root / artifact
                    if path.exists():
                        mlflow.log_artifact(str(path))
                return str(run.info.run_id)
        except Exception:
            return None


class MonitoringSimulator:
    """Generate serving and drift telemetry for portfolio MLOps dashboards."""

    def __init__(self, seed: int = 42) -> None:
        self.rng = np.random.default_rng(seed)

    def generate_monitoring_snapshot(self, registry: list[dict], days: int = 30) -> pd.DataFrame:
        end = pd.Timestamp.utcnow().normalize()
        dates = pd.date_range(end=end, periods=days, freq="D")
        rows = []
        model_names = [entry["model_name"] for entry in registry] or [
            "two_stage_recommender",
            "dynamic_pricing",
            "demand_forecaster",
        ]
        for model_name in model_names:
            base_volume = self.rng.integers(1800, 6800)
            for day_index, date in enumerate(dates):
                drift = max(0.0, self.rng.normal(0.18 + day_index * 0.003, 0.045))
                latency = max(8.0, self.rng.normal(42, 9))
                accuracy = np.clip(0.86 - drift * 0.16 + self.rng.normal(0, 0.012), 0.55, 0.97)
                rows.append(
                    {
                        "date": date.date().isoformat(),
                        "model_name": model_name,
                        "prediction_volume": int(base_volume + self.rng.normal(0, 420)),
                        "clickstream_volume": int(base_volume * self.rng.uniform(2.5, 5.2)),
                        "average_latency_ms": round(float(latency), 2),
                        "p95_latency_ms": round(float(latency * self.rng.uniform(1.6, 2.4)), 2),
                        "drift_score": round(float(drift), 4),
                        "recommendation_drift": round(float(drift * self.rng.uniform(0.8, 1.4)), 4),
                        "feature_drift": round(float(drift * self.rng.uniform(0.7, 1.6)), 4),
                        "drift_status": "watch" if drift > 0.28 else "stable",
                        "accuracy_trend": round(float(accuracy), 4),
                        "conversion_trend": round(float(np.clip(0.065 + self.rng.normal(0, 0.01) - drift * 0.02, 0.01, 0.2)), 4),
                        "data_quality_issues": int(max(0, self.rng.poisson(1 + drift * 3))),
                        "pricing_guardrail_violations": int(max(0, self.rng.poisson(0.5 + drift))),
                        "feedback_clicks": int(base_volume * self.rng.uniform(0.12, 0.28)),
                        "feedback_purchases": int(base_volume * self.rng.uniform(0.015, 0.055)),
                        "feature_freshness_minutes": int(self.rng.integers(4, 75)),
                    }
                )
        return pd.DataFrame(rows)
