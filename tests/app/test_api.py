# tests/app/test_api.py
# Test app/api.py components.

from http import HTTPStatus

from fastapi.testclient import TestClient

from app import api
from app.api import app

client = TestClient(app)


def test_load_best_artifacts():
    api.load_best_artifacts()
    assert len(api.run_ids)


def test_index():
    response = client.get("/")
    assert response.json()["status-code"] == HTTPStatus.OK
    assert response.json()["message"] == HTTPStatus.OK.phrase


def test_construct_response():
    response = client.get("/")
    assert response.json()["status-code"] == HTTPStatus.OK
    assert response.json()["method"] == "GET"


def test_best_predict():
    data = {
        "run_id": "",
        "texts": [
            {"text": "Transfer learning with transformers for self-supervised learning."},
            {"text": "Generative adversarial networks in both PyTorch and TensorFlow."},
        ],
    }
    response = client.post("/predict", json=data)
    assert response.json()["status-code"] == HTTPStatus.OK
    assert response.json()["method"] == "POST"
    assert len(response.json()["data"]["predictions"]) == len(data["texts"])


def test_runs():
    # User with statements to get startup and shutdown events
    # to work when using TestClient
    with TestClient(app) as client:
        response = client.get("/runs?top=1")
        assert response.json()["status-code"] == HTTPStatus.OK
        assert response.json()["method"] == "GET"
        assert len(response.json()["data"]["runs"]) == 1


def test_validate_run_id():
    with TestClient(app) as client:
        run_id = "invalid_run_id"
        response = client.get(f"/runs/{run_id}")
        assert response.json()["status-code"] == HTTPStatus.BAD_REQUEST
        assert response.json()["method"] == "GET"
        assert response.json()["message"] == "Invalid run ID"


def test_run():
    with TestClient(app) as client:
        response = client.get("/runs?top=1")
        run_id = response.json()["data"]["runs"][0]["run_id"]
        response = client.get(f"/runs/{run_id}")
        assert response.json()["status-code"] == HTTPStatus.OK
        assert response.json()["method"] == "GET"
        assert response.json()["data"]["run_id"] == run_id


def test_predict():
    with TestClient(app) as client:
        # Normal data
        response = client.get("/runs?top=1")
        run_id = response.json()["data"]["runs"][0]["run_id"]
        data = {
            "run_id": "",
            "texts": [
                {"text": "Transfer learning with transformers for self-supervised learning."},
                {"text": "Generative adversarial networks in both PyTorch and TensorFlow."},
            ],
        }
        response = client.post(f"/runs/{run_id}/predict", json=data)
        assert response.json()["status-code"] == HTTPStatus.OK
        assert response.json()["method"] == "POST"
        assert len(response.json()["data"]["predictions"]) == len(data["texts"])

        # Empty texts
        data = {
            "run_id": "",
            "texts": [],
        }
        response = client.post(f"/runs/{run_id}/predict", json=data)
        assert response.status_code == HTTPStatus.UNPROCESSABLE_ENTITY
        assert response.json()["detail"][0]["type"] == "value_error"

        # Empty text
        data = {
            "run_id": "",
            "texts": [{"text": ""}],
        }
        response = client.post(f"/runs/{run_id}/predict", json=data)
        assert response.status_code == HTTPStatus.UNPROCESSABLE_ENTITY
        assert response.json()["detail"][0]["type"] == "value_error.any_str.min_length"
