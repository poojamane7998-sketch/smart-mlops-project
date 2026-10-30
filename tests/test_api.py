from fastapi.testclient import TestClient

from src.api import app


client = TestClient(app)


def test_health_endpoint():

    response = client.get("/health")

    assert response.status_code == 200

    assert response.json()["status"] == "ok"


def test_prediction_endpoint():

    from sklearn.datasets import load_breast_cancer

    dataset = load_breast_cancer()

    response = client.post(
        "/predict",
        json={
            "features": dataset.data[0].tolist()
        }
    )

    assert response.status_code == 200

    result = response.json()

    assert result["predicted_label"] in [
        "malignant",
        "benign"
    ]

    assert "prediction_code" in result


def test_prediction_rejects_invalid_features():

    response = client.post(
        "/predict",
        json={
            "features": [1.0, 2.0]
        }
    )

    assert response.status_code == 422