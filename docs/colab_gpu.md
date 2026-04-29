# Colab GPU Runbook

Use this after pushing the repository to GitHub.

## 1. Enable GPU

In Colab, select `Runtime` -> `Change runtime type` -> `T4 GPU` or better.

```python
!nvidia-smi
```

## 2. Clone The Repo

```python
!git clone https://github.com/YOUR_USERNAME/real-time-personalization-pricing-platform.git
%cd real-time-personalization-pricing-platform
```

## 3. Install GPU Dependencies

```python
!pip install -r requirements-colab-gpu.txt
```

## 4. Run The PyTorch Two-Tower Pipeline

```python
!python scripts/run_pipeline.py \
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

## 5. Confirm GPU Backend

```python
import json

registry = json.load(open("models/model_registry.json"))
two_tower = [row for row in registry if row["model_name"] == "two_stage_recommender"][0]
two_tower["parameters"]
```

Expected fields:

- `retrieval_backend_requested`: `torch`
- `retrieval_backend_used`: `torch`
- `retrieval_training_device`: `cuda`
- `ranking_backend_used`: `xgboost_ranker`

The pricing and forecasting registry entries should show `xgboost_classifier` and `lightgbm_regressor`.

## 6. Download Outputs

```python
!zip -r pipeline_outputs.zip data/processed data/feature_store models reports
```
