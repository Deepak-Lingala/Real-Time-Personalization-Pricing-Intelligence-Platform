# Model Card

## Intended Use

This project is a synthetic portfolio simulation for e-commerce personalization, pricing intelligence, forecasting, customer analytics, and MLOps workflows. It is not trained on real customer data and is not deployed for a real company.

## Models

| Model | Purpose | Approach |
| --- | --- | --- |
| Two-stage recommender | Generate top-N products per user | PyTorch two-tower retrieval plus optional XGBoost/LightGBM learning-to-rank reranking |
| Dynamic pricing optimizer | Recommend constrained product prices | Optional XGBoost conversion model plus margin, inventory, and competitor guardrails |
| Demand forecaster | Forecast product/category demand | Optional LightGBM regression with lag, calendar, seasonality, promotion, and holiday features |
| Customer segmentation | Classify user behavior segments | KMeans behavioral clusters with business-readable rule labels |

## Evaluation Metrics

Recommendation metrics include retrieval `recall@100`, `precision@k`, `recall@k`, `NDCG@k`, `MAP@k`, catalog coverage, diversity, cold-start performance, and simulated CTR lift.

Pricing metrics include simulated revenue uplift, margin improvement, and conversion improvement.

Forecasting metrics include `MAE`, `RMSE`, `MAPE`, and `WAPE`, with a seasonal naive benchmark.

All metrics are simulated portfolio metrics generated from synthetic data.

## Limitations

- Synthetic data cannot represent all real-world customer behavior.
- Pricing optimization does not include legal, brand, fairness, marketplace, or long-term elasticity constraints.
- Forecasting does not model supply-chain disruptions or external macroeconomic signals.
- API responses use artifact-backed simulation rather than a production online feature store.

## Responsible Use

This repo is meant to demonstrate applied data science engineering patterns. A real pricing or personalization system should include privacy review, experimentation design, guardrails, model governance, fairness checks, monitoring, and human approval workflows.
