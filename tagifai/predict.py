# predict.py
# Prediction operations.

import tempfile
from argparse import Namespace
from distutils.util import strtobool
from pathlib import Path
from typing import Dict, List

import mlflow
import numpy as np
import torch

from tagifai import data, models, train, utils


def load_artifacts(
    run_id: str,
    device: torch.device = torch.device("cpu"),
) -> Dict:
    """Load artifacts for a particular `run_id`.

    Args:
        run_id (str): ID of the run to load model artifacts from.
        device (torch.device): Device to run model on. Defaults to CPU.

    Returns:
        Artifacts needed for inference.
    """
    # Load model
    client = mlflow.tracking.MlflowClient()
    device = torch.device("cpu")
    with tempfile.TemporaryDirectory() as fp:
        client.download_artifacts(run_id=run_id, path="", dst_path=fp)
        label_encoder = data.LabelEncoder.load(
            fp=Path(fp, "label_encoder.json")
        )
        tokenizer = data.Tokenizer.load(fp=Path(fp, "tokenizer.json"))
        model_state = torch.load(Path(fp, "model.pt"), map_location=device)
        performance = utils.load_dict(filepath=Path(fp, "performance.json"))

    # Load model
    run = mlflow.get_run(run_id=run_id)
    args = Namespace(**run.data.params)
    model = models.initialize_model(
        args=args, vocab_size=len(tokenizer), num_classes=len(label_encoder)
    )
    model.load_state_dict(model_state)

    return {
        "args": args,
        "label_encoder": label_encoder,
        "tokenizer": tokenizer,
        "model": model,
        "performance": performance,
    }


def predict(
    texts: List, artifacts: Dict, device: torch.device = torch.device("cpu")
) -> Dict:
    """Predict tags for an input text using the
    best model from the `best` experiment.

    Usage:

    ```python
    texts = ["Transfer learning with BERT."]
    artifacts = load_artifacts(run_id="264ac530b78c42608e5dea1086bc2c73")
    predict(texts=texts, artifacts=artifacts)
    ```
    <pre>
    [
      {
          "input_text": "Transfer learning with BERT.",
          "preprocessed_text": "transfer learning bert",
          "predicted_tags": [
            "attention",
            "language-modeling",
            "natural-language-processing",
            "transfer-learning",
            "transformers"
          ]
      }
    ]
    </pre>

    Note:
        The input argument `texts` can hold multiple input texts and so the resulting prediction dictionary will have `len(texts)` items.

    Args:
        texts (List): List of input texts to predict tags for.
        artifacts (Dict): Artifacts needed for inference.
        device (torch.device): Device to run model on. Defaults to CPU.

    Returns:
        Predicted tags for each of the input texts.

    """
    # Retrieve artifacts
    args = artifacts["args"]
    label_encoder = artifacts["label_encoder"]
    tokenizer = artifacts["tokenizer"]
    model = artifacts["model"]

    # Prepare data
    preprocessed_texts = [
        data.preprocess(
            text,
            lower=bool(strtobool(args.lower)),
            stem=bool(strtobool(args.stem)),
        )
        for text in texts
    ]
    X = np.array(
        tokenizer.texts_to_sequences(preprocessed_texts), dtype=object
    )
    y_filler = np.zeros((len(X), len(label_encoder)))
    dataset = data.CNNTextDataset(
        X=X, y=y_filler, max_filter_size=int(args.max_filter_size)
    )
    dataloader = dataset.create_dataloader(batch_size=int(args.batch_size))

    # Get predictions
    trainer = train.Trainer(model=model, device=device)
    _, y_prob = trainer.predict_step(dataloader)
    y_pred = np.array(
        [np.where(prob >= float(args.threshold), 1, 0) for prob in y_prob]
    )
    tags = label_encoder.decode(y_pred)
    predictions = [
        {
            "input_text": texts[i],
            "preprocessed_text": preprocessed_texts[i],
            "predicted_tags": tags[i],
        }
        for i in range(len(tags))
    ]

    return predictions
