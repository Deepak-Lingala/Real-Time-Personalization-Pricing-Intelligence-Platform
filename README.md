# Real-Time Personalization & Pricing Intelligence Platform

Company-style, end-to-end e-commerce machine learning system built with **synthetic data only**. The project mirrors how an e-commerce ML platform can connect personalization, dynamic pricing, demand forecasting, customer analytics, model serving, monitoring, and business dashboards.

> Portfolio disclaimer: all users, products, events, pricing signals, metrics, business impact numbers, and monitoring outputs are synthetic simulations. This project does not claim real company deployment.

## Business Problem

E-commerce teams need to personalize product discovery, optimize prices, forecast demand, reduce stockouts, understand customers, and serve low-latency predictions at scale. This repo demonstrates the architecture and applied data science workflow behind that system:

- two-stage recommendations with retrieval plus ranking
- constrained price optimization with inventory, margin, and competitor guardrails
- hierarchical demand forecasting with benchmark metrics
- Feast-style feature store simulation
- FastAPI serving for real-time inference
- executive and ML operations dashboards
- MLflow-compatible experiment tracking, model registry, and monitoring feedback loop

## Architecture

```mermaid
flowchart LR
    A[Synthetic Events] --> B[Batch + Streaming Ingestion]
    C[Product Catalog] --> B
    D[Pricing Signals] --> B
    E[Inventory Data] --> B
    F[Competitor Signals] --> B
    B --> G[Data Validation]
    G --> H[Spark-Style Processing]
    H --> I[Feature Store]
    I --> J[Two-Tower Retrieval]
    J --> K[LTR Ranking]
    I --> L[Pricing Optimization]
    I --> M[Demand Forecasting]
    I --> N[Customer Segmentation]
    K --> O[MLflow Registry]
    L --> O
    M --> O
    O --> P[FastAPI Serving]
    P --> Q[Enterprise Dashboard]
    Q --> R[Monitoring]
    R --> S[Feedback Loop]
    S --> B
```

## Repository Structure

```text
real-time-personalization-pricing-platform/
├── app/                         # Streamlit dashboard
├── api/                         # FastAPI inference layer
├── data/
│   ├── raw/                     # generated source tables
│   ├── processed/               # pipeline outputs
│   ├── synthetic/               # synthetic source snapshots
│   └── feature_store/           # offline/online feature tables
├── notebooks/                   # optional exploration and Colab workflow
├── src/
│   ├── ecommerce_intelligence/  # core package
│   ├── data_generation/         # production-style facade packages
│   ├── data_validation/
│   ├── preprocessing/
│   ├── feature_engineering/
│   ├── feature_store/
│   ├── retrieval/
│   ├── ranking/
│   ├── pricing/
│   ├── forecasting/
│   ├── segmentation/
│   ├── evaluation/
│   ├── inference/
│   └── monitoring/
├── models/
├── reports/
├── tests/
├── docs/
├── Dockerfile
├── Dockerfile.gpu
├── k8s/
├── docker-compose.yml
├── requirements.txt
├── requirements-colab-gpu.txt
├── Makefile
└── .github/workflows/ci.yml
```

## Synthetic Data Sources

The generator creates realistic, intentionally imperfect e-commerce data:

- user profiles: `user_id`, location, device, segment, signup date, loyalty tier
- clickstream/session events: views, clicks, carts, purchases, searches, dwell time, page position
- product catalog: category, brand, price, margin, rating, reviews, text, image feature vector
- product reviews
- search queries
- pricing and discount history
- competitor pricing signals
- inventory and supply-chain data
- product/category demand forecasting data
- recommendation feature samples

Realism features include seasonality, promotions, noisy events, missing values, duplicate events, skewed product popularity, cold-start users/products, price elasticity, stockout risk, category affinities, and repeat-purchase behavior.

## Pipeline

`scripts/run_pipeline.py` orchestrates:

1. synthetic data generation
2. batch table and streaming microbatch ingestion simulation
3. schema validation
4. event cleaning
5. sessionization
6. feature engineering
7. feature store creation
8. PyTorch two-tower retrieval training
9. XGBoost/LightGBM learning-to-rank reranker training when available
10. XGBoost pricing demand model training when available
11. constrained price optimization
12. hierarchical demand forecasting
13. RFM and behavioral segmentation
14. model evaluation
15. MLflow-compatible tracking and registry
16. batch prediction artifact creation
17. FastAPI-ready inference artifacts
18. monitoring and feedback-loop simulation

## Feature Store

`FeastStyleFeatureStore` materializes:

- user features
- product features
- session features
- pricing features
- inventory features
- offline CSV tables
- online JSON serving snapshots
- freshness timestamps
- feature version metadata

## Recommendation System

The project implements a company-style two-stage recommendation system.

**Stage 1: Candidate Retrieval**

- `TwoTowerRetrievalModel` trains user and product towers in PyTorch with negative sampling.
- The item tower combines product ID embeddings with category, brand, price, margin, inventory, rating, review, and image-vector features.
- `backend="auto"` uses PyTorch when installed and trains on CUDA when a Colab GPU is available.
- `backend="sklearn"` is kept only as a local/CI fallback for environments without Torch.
- Retrieval returns top 100 candidate products.

**Stage 2: Ranking**

- `LearningToRankReranker` reranks candidates using user, product, context, inventory, margin, and behavior features.
- `backend="auto"` uses XGBoost Ranker or LightGBM Ranker when installed, with a sklearn fallback for local/CI runs.
- Final output returns top 10 products with retrieval score, ranking score, product score, recommendation reason, category match, and predicted purchase probability.

Business filters remove out-of-stock products, suppress low-margin products, prioritize available inventory, respect category affinity, and fall back to content/embedding similarity for cold-start cases.

Metrics include retrieval Recall@100, Precision@K, Recall@K, NDCG@K, MAP@K, catalog coverage, diversity, cold-start performance, and simulated CTR lift.

## Dynamic Pricing

The pricing system combines XGBoost demand prediction when available, sklearn fallback behavior for CI, and constrained optimization.

Features include competitor price, inventory, demand score, historical conversion, category, seasonality, discount, margin, and price elasticity. Outputs include optimal price, conversion probability, expected margin, revenue uplift estimate, competitor price delta, and guardrail flags.

Guardrails:

- do not price below minimum margin threshold
- increase discount when inventory is high and demand is low
- protect margin when inventory is low and demand is high
- compare optimized price with competitor price
- flag competitor and margin violations

## Demand Forecasting

The forecasting layer uses LightGBM regression when available, sklearn fallback behavior for CI, and hierarchical feature-based demand modeling:

- seasonal naive benchmark
- product-level forecasts
- category-level forecasts
- 7-day and 30-day horizons
- lag features, rolling averages, promotions, holidays, category, seasonality, and price context

Metrics include MAE, RMSE, MAPE, and WAPE.

## Customer Analytics

Customer segmentation combines RFM and behavioral features:

- high-value customers
- discount-sensitive customers
- frequent browsers
- likely-to-churn customers
- new users
- loyal buyers / growth customers

KPIs include CTR, conversion rate, AOV, revenue, margin, stockout rate, return rate, repeat purchase rate, and customer lifetime value estimate.

## API

FastAPI endpoints:

| Endpoint | Purpose |
| --- | --- |
| `GET /recommend/{user_id}` | top-N recommendations |
| `POST /pricing/optimize` | constrained pricing optimization |
| `GET /forecast/{product_id}` | product-level forecast |
| `GET /customer/{user_id}/segment` | customer segment and churn signal |
| `GET /product/{product_id}/insights` | product features and pricing/recommendation context |
| `GET /model/metrics` | model metrics and registry |
| `GET /monitoring/drift` | drift and data quality status |
| `GET /dashboard/summary` | dashboard artifact |
| `GET /feature-store/features` | feature registry and freshness |

Examples are in [docs/api_examples.md](docs/api_examples.md).

## Dashboard

The Streamlit dashboard includes:

1. Executive Overview
2. Recommendation Engine
3. Retrieval and Ranking Performance
4. Dynamic Pricing
5. Demand Forecasting
6. Customer Segmentation
7. Product Analytics
8. Feature Store
9. Model Performance
10. MLOps Monitoring

Charts include revenue trend, CTR and conversion trend, top recommended products, retrieval vs ranking metrics, price comparison, demand forecast, inventory risk, customer segments, category performance, feature importance, latency, drift, and feature freshness.

## MLOps and Monitoring

The repo writes a portable JSON registry and logs real MLflow experiment runs, parameters, metrics, artifacts, model version, registry stage, deployment status, and rollback version when `mlflow` is installed.

Monitoring simulates clickstream volume, prediction volume, recommendation drift, feature drift, model latency, conversion trend, data quality issues, pricing guardrail violations, feature freshness, and feedback from clicks and purchases.

Kubernetes manifests in `k8s/` provide a deployable serving layout with API and dashboard deployments, services, persistent artifact volumes, a pipeline Job, an optional GPU pipeline Job, health probes, resource limits, and API autoscaling.

## Run Locally

Create and activate a venv:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Run a smoke pipeline:

```powershell
python scripts/run_pipeline.py --users 500 --products 120 --events 8000 --days 60 --retrieval-backend sklearn --ranking-backend sklearn --pricing-backend sklearn --forecasting-backend sklearn
```

Run tests:

```powershell
pytest
```

Start the API:

```powershell
uvicorn api.main:app --reload --port 8000
```

Start the dashboard:

```powershell
streamlit run app/dashboard.py
```

## Colab GPU

Use the PyTorch backend explicitly in Colab. It will train the two-tower retrieval model on CUDA when a GPU runtime is enabled:

```bash
git clone https://github.com/YOUR_USERNAME/real-time-personalization-pricing-platform.git
cd real-time-personalization-pricing-platform
pip install -r requirements-colab-gpu.txt
python scripts/run_pipeline.py \
  --users 5000 \
  --products 800 \
  --events 120000 \
  --days 180 \
  --retrieval-backend torch \
  --ranking-backend xgboost \
  --pricing-backend xgboost \
  --forecasting-backend lightgbm \
  --retrieval-epochs 5 \
  --retrieval-batch-size 2048
```

Check the backend after training:

```bash
cat models/model_registry.json
```

The registry includes `retrieval_backend_used`, `retrieval_training_device`, `ranking_backend_used`, `pricing_backend_used`, and `forecasting_backend_used`.

## Docker

```bash
docker compose up --build
```

Services:

- API: `http://localhost:8000`
- Dashboard: `http://localhost:8501`

## Kubernetes

Build and deploy the local serving image:

```bash
docker build -t real-time-personalization-pricing-platform:latest .
kubectl apply -k k8s
kubectl apply -f k8s/jobs/pipeline-job.yaml
```

Port-forward services:

```bash
kubectl -n personalization-pricing port-forward svc/pricing-intelligence-api 8000:8000
kubectl -n personalization-pricing port-forward svc/pricing-intelligence-dashboard 8501:8501
```

For GPU training on a Kubernetes cluster with NVIDIA support:

```bash
docker build -f Dockerfile.gpu -t real-time-personalization-pricing-platform:gpu .
kubectl apply -f k8s/jobs/pipeline-gpu-job.yaml
```

See [docs/kubernetes.md](docs/kubernetes.md).

## Recruiter Summary

This project demonstrates production-style applied ML engineering for e-commerce: synthetic data generation, data quality, feature engineering, feature store simulation, two-stage recommendations, dynamic pricing, forecasting, segmentation, FastAPI serving, dashboards, monitoring, CI/CD, and business-facing documentation.

## Limitations and Future Improvements

- Add transformer text embeddings and richer session context to the two-tower model.
- Replace local MLflow runs and JSON/CSV registry with a managed MLflow tracking server.
- Add Kafka/Kinesis streaming ingestion.
- Add Feast, Redis, and warehouse-backed online/offline feature serving.
- Add A/B testing, fairness review, and pricing governance workflows.
