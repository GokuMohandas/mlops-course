# main.py
# Check API endpoints.

import json
import os

import requests

from tagifai.config import logger

# Headers
headers = {
    "accept": "application/json",
    "Content-Type": "application/json",
}


def health_check():
    response = requests.get(
        f"http://0.0.0.0:{os.environ.get('PORT', 5000)}/",
        headers=headers,
    )
    results = json.loads(response.text)
    logger.info(json.dumps(results, indent=4))


def predict():
    data = {
        "run_id": "",
        "texts": [
            {
                "text": "Transfer learning with transformers for self-supervised learning."
            },
            {
                "text": "Generative adversarial networks in both PyTorch and TensorFlow."
            },
        ],
    }
    response = requests.post(
        f"http://0.0.0.0:{os.environ.get('PORT', 5000)}/predict",
        headers=headers,
        data=json.dumps(data),
    )
    results = json.loads(response.text)
    logger.info(json.dumps(results, indent=4))


if __name__ == "__main__":
    predict()
