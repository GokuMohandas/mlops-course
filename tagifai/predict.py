from typing import Dict, List

import numpy as np


def custom_predict(y_prob: np.ndarray, threshold: float, index: int) -> np.ndarray:
    """Custom predict function that defaults
    to an index if conditions are not met.

    Args:
        y_prob (np.ndarray): predicted probabilities
        threshold (float): minimum softmax score to predict majority class
        index (int): label index to use if custom conditions is not met.

    Returns:
        np.ndarray: predicted label indices.
    """
    y_pred = [np.argmax(p) if max(p) > threshold else index for p in y_prob]
    return np.array(y_pred)


def predict(texts: List, artifacts: Dict) -> List:
    """Predict tags for given texts.

    Args:
        texts (List): raw input texts to classify.
        artifacts (Dict): artifacts from a run.

    Returns:
        List: predictions for input texts.
    """
    x = artifacts["vectorizer"].transform(texts)
    y_pred = custom_predict(
        y_prob=artifacts["model"].predict_proba(x),
        threshold=artifacts["args"].threshold,
        index=artifacts["label_encoder"].class_to_index["other"],
    )
    tags = artifacts["label_encoder"].decode(y_pred)
    predictions = [
        {
            "input_text": texts[i],
            "predicted_tag": tags[i],
        }
        for i in range(len(tags))
    ]
    return predictions
