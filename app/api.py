from datetime import datetime
from functools import wraps
from http import HTTPStatus
from pathlib import Path
from typing import Dict

from fastapi import FastAPI, Request

from app.schemas import PredictPayload
from config import config
from config.config import logger
from tagifai import main, predict

# Define application
app = FastAPI(
    title="TagIfAI - Made With ML",
    description="Classify machine learning projects.",
    version="0.1",
)


@app.on_event("startup")
def load_artifacts():
    global artifacts
    run_id = open(Path(config.CONFIG_DIR, "run_id.txt")).read()
    artifacts = main.load_artifacts(run_id=run_id)
    logger.info("Ready for inference!")


def construct_response(f):
    """Construct a JSON response for an endpoint."""

    @wraps(f)
    def wrap(request: Request, *args, **kwargs) -> Dict:
        results = f(request, *args, **kwargs)
        response = {
            "message": results["message"],
            "method": request.method,
            "status-code": results["status-code"],
            "timestamp": datetime.now().isoformat(),
            "url": request.url._url,
        }
        if "data" in results:
            response["data"] = results["data"]
        return response

    return wrap


@app.get("/", tags=["General"])
@construct_response
def _index(request: Request) -> Dict:
    """Health check."""
    response = {
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
        "data": {},
    }
    return response


@app.get("/performance", tags=["Performance"])
@construct_response
def _performance(request: Request, filter: str = None) -> Dict:
    """Get the performance metrics."""
    performance = artifacts["performance"]
    data = {"performance": performance.get(filter, performance)}
    response = {
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
        "data": data,
    }
    return response


@app.get("/args", tags=["Arguments"])
@construct_response
def _args(request: Request) -> Dict:
    """Get all arguments used for the run."""
    response = {
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
        "data": {
            "args": vars(artifacts["args"]),
        },
    }
    return response


@app.get("/args/{arg}", tags=["Arguments"])
@construct_response
def _arg(request: Request, arg: str) -> Dict:
    """Get a specific parameter's value used for the run."""
    response = {
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
        "data": {
            arg: vars(artifacts["args"]).get(arg, ""),
        },
    }
    return response


@app.post("/predict", tags=["Prediction"])
@construct_response
def _predict(request: Request, payload: PredictPayload) -> Dict:
    """Predict tags for a list of texts."""
    texts = [item.text for item in payload.texts]
    predictions = predict.predict(texts=texts, artifacts=artifacts)
    response = {
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
        "data": {"predictions": predictions},
    }
    return response
