# Kubernetes Deployment Guide

This project includes Kubernetes manifests for a production-style ML serving layout:

- FastAPI inference deployment and service
- Streamlit executive dashboard deployment and service
- shared persistent volumes for generated data and model artifacts
- batch pipeline job for synthetic training and artifact refresh
- optional GPU pipeline job for PyTorch retrieval plus XGBoost/LightGBM backends
- API horizontal pod autoscaling

All data remains synthetic.

## Build Images

CPU/local serving image:

```bash
docker build -t real-time-personalization-pricing-platform:latest .
```

Optional GPU training image:

```bash
docker build -f Dockerfile.gpu -t real-time-personalization-pricing-platform:gpu .
```

For kind:

```bash
kind load docker-image real-time-personalization-pricing-platform:latest
kind load docker-image real-time-personalization-pricing-platform:gpu
```

For Minikube:

```bash
minikube image load real-time-personalization-pricing-platform:latest
minikube image load real-time-personalization-pricing-platform:gpu
```

## Deploy Serving Layer

```bash
kubectl apply -k k8s
kubectl -n personalization-pricing get pods
```

The API and dashboard can start from sample artifacts baked into the image. Run the pipeline job to populate the persistent volumes with fresh synthetic artifacts.

## Run The Pipeline Job

CPU-compatible job:

```bash
kubectl apply -f k8s/jobs/pipeline-job.yaml
kubectl -n personalization-pricing logs job/personalization-pricing-pipeline -f
kubectl -n personalization-pricing rollout restart deployment/pricing-intelligence-api
kubectl -n personalization-pricing rollout restart deployment/pricing-intelligence-dashboard
```

GPU job for a cluster with NVIDIA GPU support:

```bash
kubectl apply -f k8s/jobs/pipeline-gpu-job.yaml
kubectl -n personalization-pricing logs job/personalization-pricing-gpu-pipeline -f
```

The GPU job requests `nvidia.com/gpu: "1"` and uses:

- `--retrieval-backend torch`
- `--ranking-backend xgboost`
- `--pricing-backend xgboost`
- `--forecasting-backend lightgbm`

## Access Services

API:

```bash
kubectl -n personalization-pricing port-forward svc/pricing-intelligence-api 8000:8000
curl http://localhost:8000/health
curl http://localhost:8000/model/metrics
```

Dashboard:

```bash
kubectl -n personalization-pricing port-forward svc/pricing-intelligence-dashboard 8501:8501
```

Open `http://localhost:8501`.

## Scale And Observe

```bash
kubectl -n personalization-pricing get hpa
kubectl -n personalization-pricing get deploy,svc,pvc
kubectl -n personalization-pricing logs deploy/pricing-intelligence-api
```

## Clean Up

```bash
kubectl delete -f k8s/jobs/pipeline-job.yaml --ignore-not-found
kubectl delete -f k8s/jobs/pipeline-gpu-job.yaml --ignore-not-found
kubectl delete -k k8s
```
