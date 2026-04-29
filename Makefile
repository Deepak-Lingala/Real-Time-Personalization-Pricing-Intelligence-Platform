.PHONY: setup test lint pipeline api dashboard docker-build docker-build-gpu k8s-apply k8s-pipeline k8s-gpu-pipeline k8s-api k8s-dashboard k8s-delete

setup:
	python -m venv .venv
	.venv/Scripts/python.exe -m pip install --upgrade pip
	.venv/Scripts/python.exe -m pip install -r requirements.txt

test:
	.venv/Scripts/python.exe -m pytest

lint:
	.venv/Scripts/python.exe -m ruff check src api tests

pipeline:
	.venv/Scripts/python.exe scripts/run_pipeline.py --users 500 --products 120 --events 8000 --days 60

api:
	.venv/Scripts/python.exe -m uvicorn api.main:app --reload --port 8000

dashboard:
	.venv/Scripts/streamlit.exe run app/dashboard.py

docker-build:
	docker build -t real-time-personalization-pricing-platform:latest .

docker-build-gpu:
	docker build -f Dockerfile.gpu -t real-time-personalization-pricing-platform:gpu .

k8s-apply:
	kubectl apply -k k8s

k8s-pipeline:
	kubectl apply -f k8s/jobs/pipeline-job.yaml

k8s-gpu-pipeline:
	kubectl apply -f k8s/jobs/pipeline-gpu-job.yaml

k8s-api:
	kubectl -n personalization-pricing port-forward svc/pricing-intelligence-api 8000:8000

k8s-dashboard:
	kubectl -n personalization-pricing port-forward svc/pricing-intelligence-dashboard 8501:8501

k8s-delete:
	kubectl delete -f k8s/jobs/pipeline-job.yaml --ignore-not-found
	kubectl delete -f k8s/jobs/pipeline-gpu-job.yaml --ignore-not-found
	kubectl delete -k k8s
