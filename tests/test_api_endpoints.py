from fastapi.testclient import TestClient

from api.main import app


def test_enterprise_api_endpoints_return_sample_artifacts() -> None:
    client = TestClient(app)

    assert client.get("/health").status_code == 200
    assert client.get("/recommend/U003812").status_code == 200
    assert client.post("/pricing/optimize", json={"product_id": "P000001"}).status_code == 200
    assert client.get("/forecast/P000001").status_code == 200
    assert client.get("/customer/U003812/segment").status_code == 200
    assert client.get("/product/P000001/insights").status_code == 200
    assert client.get("/model/metrics").status_code == 200
    assert client.get("/monitoring/drift").status_code == 200
    assert client.get("/dashboard/summary").status_code == 200
    assert client.get("/feature-store/features").status_code == 200
