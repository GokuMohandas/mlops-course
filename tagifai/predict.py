# predict.py
# Prediction operations.

from distutils.util import strtobool
from typing import Dict, List

import numpy as np
import torch

from tagifai import data, train


def predict(texts: List, artifacts: Dict, device: torch.device = torch.device("cpu")) -> Dict:
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
            lower=bool(strtobool(str(args.lower))),  # args.lower could be str/bool
            stem=bool(strtobool(str(args.stem))),
        )
        for text in texts
    ]
    X = np.array(tokenizer.texts_to_sequences(preprocessed_texts), dtype="object")
    y_filler = np.zeros((len(X), len(label_encoder)))
    dataset = data.CNNTextDataset(X=X, y=y_filler, max_filter_size=int(args.max_filter_size))
    dataloader = dataset.create_dataloader(batch_size=int(args.batch_size))

    # Get predictions
    trainer = train.Trainer(model=model, device=device)
    _, y_prob = trainer.predict_step(dataloader)
    y_pred = [np.where(prob >= float(args.threshold), 1, 0) for prob in y_prob]
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
