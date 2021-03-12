# api.py
# FastAPI application endpoints.


from datetime import datetime
from functools import wraps
from http import HTTPStatus
from typing import Dict, Optional

import torch
from fastapi import FastAPI, Request

from app.schemas import PredictPayload
from tagifai import main, predict, utils
from tagifai.config import logger

# Define application
app = FastAPI(
    title="TagIfAI - Made With ML",
    description="Predict relevant tags given a text input.",
    version="0.1",
)


@app.on_event("startup")
def load_best_artifacts():
    global runs, run_ids, best_artifacts, best_run_id
    runs = utils.get_sorted_runs(experiment_name="best", order_by=["metrics.f1 DESC"])
    run_ids = [run["run_id"] for run in runs]
    best_run_id = run_ids[0]
    best_artifacts = main.load_artifacts(run_id=best_run_id, device=torch.device("cpu"))
    logger.info("Loaded trained model and other required artifacts for inference!")


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


def validate_run_id(f):
    """Validates a `run_id`."""

    @wraps(f)
    def wrap(request: Request, *args, **kwargs):
        # Retrieve run_id
        run_id = kwargs["run_id"]

        # Invalid run_id
        if run_id not in run_ids:
            return {
                "message": "Invalid run ID",
                "method": request.method,
                "status-code": HTTPStatus.BAD_REQUEST,
                "timestamp": datetime.now().isoformat(),
                "url": request.url._url,
            }

        results = f(request, *args, **kwargs)
        return results

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


@app.post("/predict", tags=["General"])
@construct_response
def _best_predict(request: Request, payload: PredictPayload) -> Dict:
    """Predict tags for a list of texts using the best run. """
    # Predict
    texts = [item.text for item in payload.texts]
    predictions = predict.predict(texts=texts, artifacts=best_artifacts)
    response = {
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
        "data": {"run_id": best_run_id, "predictions": predictions},
    }
    return response


@app.get("/runs", tags=["Runs"])
@construct_response
def _runs(request: Request, top: Optional[int] = None) -> Dict:
    """Get all runs sorted by f1 score."""
    response = {
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
        "data": {"runs": runs[:top]},
    }
    return response


@app.get("/runs/{run_id}", tags=["Runs"])
@construct_response
@validate_run_id
def _run(request: Request, run_id: str) -> Dict:
    """Get details about a specific run."""
    artifacts = main.load_artifacts(run_id=run_id)
    response = {
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
        "data": {
            "run_id": run_id,
            "performance": artifacts["performance"],
            "behavioral_report": artifacts["behavioral_report"],
        },
    }
    return response


@app.post("/runs/{run_id}/predict", tags=["Runs"])
@construct_response
@validate_run_id
def _predict(request: Request, run_id: str, payload: PredictPayload) -> Dict:
    """Predict tags for a list of texts using artifacts from run `run_id`."""
    artifacts = main.load_artifacts(run_id=run_id)
    texts = [item.text for item in payload.texts]
    predictions = predict.predict(texts=texts, artifacts=artifacts)
    response = {
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
        "data": {"run_id": run_id, "predictions": predictions},
    }
    return response
