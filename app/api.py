# app/api.py
# FastAPI application endpoints.


from datetime import datetime
from functools import wraps
from http import HTTPStatus
from typing import Dict, Optional

from fastapi import FastAPI, Request

from app.schemas import PredictPayload
from tagifai import config, main, predict
from tagifai.config import logger

# Define application
app = FastAPI(
    title="TagIfAI - Made With ML",
    description="Predict relevant tags given a text input.",
    version="0.1",
)


@app.on_event("startup")
def load_artifacts():
    global artifacts
    artifacts = main.load_artifacts(model_dir=config.MODEL_DIR)
    logger.info("Ready for inference!")


def construct_response(f):
    """Construct a JSON response for an endpoint's results."""

    @wraps(f)
    def wrap(request: Request, *args, **kwargs):
        results = f(request, *args, **kwargs)

        # Construct response
        response = {
            "message": results["message"],
            "method": request.method,
            "status-code": results["status-code"],
            "timestamp": datetime.now().isoformat(),
            "url": request.url._url,
        }

        # Add data
        if "data" in results:
            response["data"] = results["data"]

        return response

    return wrap


@app.get("/", tags=["General"])
@construct_response
def _index(request: Request):
    """Health check."""
    response = {
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
        "data": {},
    }
    return response


@app.post("/predict", tags=["Prediction"])
@construct_response
def _predict(request: Request, payload: PredictPayload) -> Dict:
    """Predict tags for a list of texts using the best run. """
    # Predict
    texts = [item.text for item in payload.texts]
    predictions = predict.predict(texts=texts, artifacts=artifacts)
    response = {
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
        "data": {"predictions": predictions},
    }
    return response


@app.get("/params", tags=["Parameters"])
@construct_response
def _params(request: Request) -> Dict:
    """Get parameter values used for a run."""
    response = {
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
        "data": {
            "params": vars(artifacts["params"]),
        },
    }
    return response


@app.get("/params/{param}", tags=["Parameters"])
@construct_response
def _param(request: Request, param: str) -> Dict:
    """Get a specific parameter's value used for a run."""
    response = {
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
        "data": {
            "params": {
                param: vars(artifacts["params"]).get(param, ""),
            }
        },
    }
    return response


@app.get("/performance", tags=["Performance"])
@construct_response
def _performance(request: Request, filter: Optional[str] = None) -> Dict:
    """Get the performance metrics for a run."""
    performance = artifacts["performance"]
    if filter:
        for key in filter.split("."):
            performance = performance.get(key, {})
        data = {"performance": {filter: performance}}
    else:
        data = {"performance": performance}
    response = {
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
        "data": data,
    }
    return response
