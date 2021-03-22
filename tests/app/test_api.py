# tests/app/test_api.py
# Test app/api.py components.

from http import HTTPStatus

from fastapi.testclient import TestClient

from app import api
from app.api import app

client = TestClient(app)


def test_load_artifacts():
    api.load_artifacts()
    assert len(api.artifacts)


def test_index():
    response = client.get("/")
    assert response.status_code == HTTPStatus.OK
    assert response.json()["message"] == HTTPStatus.OK.phrase


def test_construct_response():
    response = client.get("/")
    assert response.status_code == HTTPStatus.OK
    assert response.request.method == "GET"


def test_predict():
    data = {
        "texts": [
            {"text": "Transfer learning with transformers for self-supervised learning."},
            {"text": "Generative adversarial networks in both PyTorch and TensorFlow."},
        ],
    }
    response = client.post("/predict", json=data)
    assert response.status_code == HTTPStatus.OK
    assert response.request.method == "POST"
    assert len(response.json()["data"]["predictions"]) == len(data["texts"])


def test_empty_predict():
    data = {
        "texts": [],
    }
    response = client.post("/predict", json=data)
    assert response.status_code == HTTPStatus.UNPROCESSABLE_ENTITY
    assert response.request.method == "POST"
    assert response.json()["detail"][0]["type"] == "value_error"


def test_params():
    with TestClient(app) as client:
        response = client.get("/params")
        assert response.status_code == HTTPStatus.OK
        assert response.request.method == "GET"
        print(response.json())
        assert isinstance(response.json()["data"]["params"]["seed"], int)


def test_param():
    with TestClient(app) as client:
        response = client.get("/params/seed")
        assert response.status_code == HTTPStatus.OK
        assert response.request.method == "GET"
        assert isinstance(response.json()["data"]["params"]["seed"], int)


def test_performance():
    with TestClient(app) as client:
        response = client.get("/performance")
        assert response.status_code == HTTPStatus.OK
        assert response.request.method == "GET"
        assert isinstance(response.json()["data"]["performance"]["overall"], dict)


def test_filtered_performance():
    with TestClient(app) as client:
        response = client.get("/performance?filter=overall")
        assert response.status_code == HTTPStatus.OK
        assert response.request.method == "GET"
        assert isinstance(response.json()["data"]["performance"]["overall"], dict)
