# Architecture Notes

This repository simulates the shape of a production e-commerce machine learning platform while keeping every dataset synthetic.

## Data Flow

1. Synthetic event, product, user, inventory, competitor price, and demand data are generated in `src/ecommerce_intelligence/data_generator.py`.
2. Batch tables and clickstream microbatches are created by `BatchStreamingIngestionSimulator`.
3. Validation checks enforce expected schemas, ranges, and uniqueness constraints.
4. Feature engineering builds user, product, session, pricing, inventory, and interaction features in `data/feature_store/`.
5. Model training produces:
   - PyTorch two-tower retrieval and XGBoost/LightGBM ranking recommendation model
   - XGBoost-backed dynamic pricing optimizer when available
   - LightGBM-backed demand forecasting model when available
   - Customer segmentation table
6. Feature store offline/online artifacts, model registry entries, and monitoring snapshots are written as JSON/CSV artifacts.
7. FastAPI serves inference-like responses.
8. Streamlit dashboard pages expose executive and ML views.
9. Kubernetes manifests deploy the serving layer, persistent artifact storage, autoscaling, and pipeline jobs.

## Batch and Real-Time Simulation

- Batch layer: `scripts/run_pipeline.py` regenerates synthetic data, retrains models, and refreshes artifacts.
- Streaming layer: event microbatches add offsets, batch ids, and ingestion timestamps before cleaning/sessionization.
- Feature store simulation: offline CSV tables, online JSON snapshots, registry metadata, freshness tracking, and versioning under `data/feature_store/`.
- Real-time serving simulation: FastAPI endpoints load the latest dashboard artifact and return low-latency JSON responses.
- Monitoring simulation: clickstream volume, prediction volume, latency, recommendation drift, feature drift, conversion trend, data quality issues, pricing guardrail violations, feedback clicks/purchases, and feature freshness are generated in `MonitoringSimulator`.
- Kubernetes deployment simulation: `k8s/` contains Deployments, Services, PVCs, HPA, and CPU/GPU training Jobs.

## Production Extensions

The same boundaries could be extended with Kafka/Kinesis for streaming events, Feast for feature serving, a managed MLflow registry, Airflow for orchestration, and a warehouse such as BigQuery, Snowflake, or Redshift.
