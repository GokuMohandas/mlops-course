# predict.py
# Prediction operations.

import tempfile
from argparse import Namespace
from pathlib import Path
from typing import Dict, List

import mlflow
import numpy as np
import torch

from tagifai import config, data, train, utils


def predict(texts: List, run_id: str) -> Dict:
    """Predict tags for an input text using the
    best model from the `best` experiment.

    Args:
        texts (List): List of input text to predict tags for.
        run_id (str): ID of the run to load model artifacts from.

    Returns:
        Predicted tags for input texts.
    """
    # Load artifacts from run
    client = mlflow.tracking.MlflowClient()
    run = mlflow.get_run(run_id=run_id)
    with tempfile.TemporaryDirectory() as fp:
        client.download_artifacts(run_id=run_id, path="", dst_path=fp)
        args = Namespace(
            **utils.load_dict(filepath=Path(config.CONFIG_DIR, "args.json"))
        )
        label_encoder = data.LabelEncoder.load(
            fp=Path(fp, "label_encoder.json")
        )
        tokenizer = data.Tokenizer.load(fp=Path(fp, "tokenizer.json"))
        model_state = torch.load(
            Path(fp, "model.pt"), map_location=torch.device("cpu")
        )
        # performance = utils.load_dict(filepath=Path(fp, "performance.json"))

    # Load model
    args = Namespace(**run.data.params)
    model = train.initialize_model(
        args=args, vocab_size=len(tokenizer), num_classes=len(label_encoder)
    )
    model.load_state_dict(model_state)

    # Prepare data
    preprocessed_texts = [data.preprocess(text) for text in texts]
    X = np.array(tokenizer.texts_to_sequences(preprocessed_texts))
    y_filler = label_encoder.encode(
        [np.array([label_encoder.classes[0]] * len(X))]
    )
    dataset = data.CNNTextDataset(
        X=X, y=y_filler, max_filter_size=int(args.max_filter_size)
    )
    dataloader = dataset.create_dataloader(batch_size=int(args.batch_size))

    # Get predictions
    trainer = train.Trainer(model=model)
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
