# API Examples

The API serves trained artifacts from `models/` and `data/processed/` when available. Before training, it falls back to synthetic sample artifacts in `data/sample/`.

Run:

```bash
uvicorn api.main:app --reload --port 8000
```

## Recommendations

```bash
curl "http://localhost:8000/recommend/U003812?k=10"
```

Returns top products with retrieval score, ranking score, product score, recommendation reason, category match, and predicted purchase probability.

## Pricing Optimization

```bash
curl -X POST "http://localhost:8000/pricing/optimize" \
  -H "Content-Type: application/json" \
  -d '{
    "product_id": "P000144",
    "current_price": 319.99,
    "competitor_price": 331.20,
    "inventory_level": 42,
    "demand_score": 1.8,
    "product_category": "Electronics",
    "seasonality_factor": 1.05,
    "discount_percentage": 0.04,
    "historical_conversion_rate": 0.08,
    "margin": 0.36,
    "price_elasticity_score": 1.12
  }'
```

## Forecast

```bash
curl "http://localhost:8000/forecast/P000144?horizon_days=7"
```

## Customer Segment

```bash
curl http://localhost:8000/customer/U003812/segment
```

## Product Insights

```bash
curl http://localhost:8000/product/P000144/insights
```

## Model Metrics

```bash
curl http://localhost:8000/model/metrics
```

## Monitoring Drift

```bash
curl http://localhost:8000/monitoring/drift
```

## Dashboard Summary

```bash
curl http://localhost:8000/dashboard/summary
```

## Feature Store

```bash
curl http://localhost:8000/feature-store/features
```

All responses are synthetic portfolio outputs.

