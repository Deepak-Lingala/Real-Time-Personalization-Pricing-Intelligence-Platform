from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_kubernetes_manifests_cover_serving_training_and_scaling() -> None:
    required = [
        "k8s/namespace.yaml",
        "k8s/configmap.yaml",
        "k8s/data-pvc.yaml",
        "k8s/models-pvc.yaml",
        "k8s/api-deployment.yaml",
        "k8s/api-service.yaml",
        "k8s/dashboard-deployment.yaml",
        "k8s/dashboard-service.yaml",
        "k8s/api-hpa.yaml",
        "k8s/jobs/pipeline-job.yaml",
        "k8s/jobs/pipeline-gpu-job.yaml",
    ]

    for manifest in required:
        assert (ROOT / manifest).exists()

    api_deployment = (ROOT / "k8s/api-deployment.yaml").read_text(encoding="utf-8")
    dashboard_deployment = (ROOT / "k8s/dashboard-deployment.yaml").read_text(encoding="utf-8")
    pipeline_job = (ROOT / "k8s/jobs/pipeline-job.yaml").read_text(encoding="utf-8")
    gpu_job = (ROOT / "k8s/jobs/pipeline-gpu-job.yaml").read_text(encoding="utf-8")

    assert "readinessProbe" in api_deployment
    assert "livenessProbe" in api_deployment
    assert "pricing-intelligence-api" in api_deployment
    assert "pricing-intelligence-dashboard" in dashboard_deployment
    assert "python scripts/run_pipeline.py" in pipeline_job
    assert "nvidia.com/gpu" in gpu_job
    assert "--retrieval-backend torch" in gpu_job
