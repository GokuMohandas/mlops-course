# tagifai/eval.py
# Evaluation components.

import itertools
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import precision_recall_fscore_support
from snorkel.slicing import PandasSFApplier, slice_dataframe, slicing_function

from tagifai import predict, train


@slicing_function()
def cv_transformers(x):
    """Projects with the `computer-vision` and `transformers` tags."""
    return all(tag in x.tags for tag in ["computer-vision", "transformers"])


@slicing_function()
def short_text(x):
    """Projects with short titles and descriptions."""
    return len(x.text.split()) < 7  # less than 7 words


def get_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, classes: List, df: pd.DataFrame = None
) -> Dict:
    """Per-class performance metrics.

    Args:
        y_true (np.ndarray): True class labels.
        y_pred (np.ndarray): Predicted class labels.
        classes (List): List of all unique classes.
        df (pd.DataFrame, optional): dataframe used for slicing.

    Returns:
        Dictionary of overall and per-class performance metrics.
    """
    # Performance
    metrics = {"overall": {}, "class": {}}

    # Overall metrics
    overall_metrics = precision_recall_fscore_support(y_true, y_pred, average="weighted")
    metrics["overall"]["precision"] = overall_metrics[0]
    metrics["overall"]["recall"] = overall_metrics[1]
    metrics["overall"]["f1"] = overall_metrics[2]
    metrics["overall"]["num_samples"] = np.float64(len(y_true))

    # Per-class metrics
    class_metrics = precision_recall_fscore_support(y_true, y_pred, average=None)
    for i in range(len(classes)):
        metrics["class"][classes[i]] = {
            "precision": class_metrics[0][i],
            "recall": class_metrics[1][i],
            "f1": class_metrics[2][i],
            "num_samples": np.float64(class_metrics[3][i]),
        }

    # Slicing metrics
    if df is not None:
        # Slices
        slicing_functions = [cv_transformers, short_text]
        applier = PandasSFApplier(slicing_functions)
        slices = applier.apply(df)

        # Score slices
        # Use snorkel.analysis.Scorer for multiclass tasks
        # Naive implementation for our multilabel task
        # based on snorkel.analysis.Scorer
        metrics["slices"] = {}
        metrics["slices"]["class"] = {}
        for slice_name in slices.dtype.names:
            mask = slices[slice_name].astype(bool)
            if sum(mask):
                slice_metrics = precision_recall_fscore_support(
                    y_true[mask], y_pred[mask], average="micro"
                )
                metrics["slices"]["class"][slice_name] = {}
                metrics["slices"]["class"][slice_name]["precision"] = slice_metrics[0]
                metrics["slices"]["class"][slice_name]["recall"] = slice_metrics[1]
                metrics["slices"]["class"][slice_name]["f1"] = slice_metrics[2]
                metrics["slices"]["class"][slice_name]["num_samples"] = len(y_true[mask])

        # Weighted overall slice metrics
        metrics["slices"]["overall"] = {}
        for metric in ["precision", "recall", "f1"]:
            metrics["slices"]["overall"][metric] = np.mean(
                list(
                    itertools.chain.from_iterable(
                        [
                            [metrics["slices"]["class"][slice_name][metric]]
                            * metrics["slices"]["class"][slice_name]["num_samples"]
                            for slice_name in metrics["slices"]["class"]
                        ]
                    )
                )
            )

    return metrics


def compare_tags(texts: str, tags: List, artifacts: Dict, test_type: str) -> List:
    """Compare ground truth with predicted tags.

    Args:
        texts (List): List of input texts to predict on.
        tags (Dict): List of ground truth tags for each input.
        artifacts (Dict): Artifacts needed for inference.
        test_type (str): Type of test (INV, DIR, MFT, etc.)

    Returns:
        List: Results with inputs, predictions and success status.
    """
    # Predict
    predictions = predict.predict(texts=texts, artifacts=artifacts)

    # Evaluate
    results = {"passed": [], "failed": []}
    for i, prediction in enumerate(predictions):
        result = {
            "input": {"text": texts[i], "tags": tags[i]},
            "prediction": predictions[i],
            "type": test_type,
        }
        if all(tag in prediction["predicted_tags"] for tag in tags[i]):
            results["passed"].append(result)
        else:
            results["failed"].append(result)
    return results


def get_behavioral_report(artifacts: Dict) -> Dict:
    """Assess failure rate by performing
    behavioral tests on our trained model.

    Args:
        artifacts (Dict): Artifacts needed for inference.

    Returns:
        Dict: Results of behavioral tests.
    """
    results = {"passed": [], "failed": []}

    # INVariance via verb injection (changes should not affect outputs)
    tokens = ["revolutionized", "disrupted", "accelerated"]
    tags = [["transformers"], ["transformers"], ["transformers"]]
    texts = [f"Transformers have {token} the ML field." for token in tokens]
    for status, items in compare_tags(
        texts=texts, tags=tags, artifacts=artifacts, test_type="INV"
    ).items():
        results[status].extend(items)

    # INVariance via misspelling
    tokens = ["generative adverseril network", "generated adversarial networks"]
    tags = [["generative-adversarial-networks"], ["generative-adversarial-networks"]]
    texts = [f"{token} are very popular in machine learning projects." for token in tokens]
    for status, items in compare_tags(
        texts=texts, tags=tags, artifacts=artifacts, test_type="INV"
    ).items():
        results[status].extend(items)

    # DIRectional expectations (changes with known outputs)
    tokens = ["TensorFlow", "Huggingface"]
    tags = [
        ["tensorflow", "transformers"],
        ["huggingface", "transformers"],
    ]
    texts = [f"A {token} implementation of transformers." for token in tokens]
    for status, items in compare_tags(
        texts=texts, tags=tags, artifacts=artifacts, test_type="DIR"
    ).items():
        results[status].extend(items)

    # Minimum Functionality Tests (simple input/output pairs)
    tokens = ["transformers", "graph neural networks"]
    tags = [["transformers"], ["graph-neural-networks"]]
    texts = [f"{token} have revolutionized machine learning." for token in tokens]
    for status, items in compare_tags(
        texts=texts, tags=tags, artifacts=artifacts, test_type="MFT"
    ).items():
        results[status].extend(items)

    # Behavioral score
    score = len(results["passed"]) / float(len(results["passed"]) + len(results["failed"]))

    return {"score": score, "results": results}


def evaluate(
    artifacts: Dict,
    dataloader: torch.utils.data.DataLoader,
    df: pd.DataFrame,
    device: torch.device,
) -> Dict:
    """Evaluate performance on data.

    Args:
        artifacts (Dict): Artifacts needed for inference.
        dataloader (torch.utils.data.DataLoader): Dataloader with the data your want to evaluate.
        df (pd.DataFrame): dataframe (used for slicing).
        device (torch.device): Device to run model on. Defaults to CPU.

    Returns:
        Evaluation report.
    """
    # Artifacts
    args = artifacts["args"]
    model = artifacts["model"]
    label_encoder = artifacts["label_encoder"]
    model = model.to(device)
    classes = label_encoder.classes

    # Determine predictions using threshold
    trainer = train.Trainer(model=model, device=device)
    y_true, y_prob = trainer.predict_step(dataloader=dataloader)
    y_pred = np.array([np.where(prob >= float(args.threshold), 1, 0) for prob in y_prob])

    # Evaluate performance
    performance = {}
    performance = get_metrics(df=df, y_true=y_true, y_pred=y_pred, classes=classes)
    performance["behavioral_report"] = get_behavioral_report(artifacts=artifacts)

    return performance


if __name__ == "__main__":  # pragma: no cover, playground for eval components
    import json
    from argparse import Namespace
    from pathlib import Path

    import numpy as np
    import pandas as pd

    from tagifai import config, data, main, utils
    from tagifai.config import logger

    # Set experiment and start run
    args_fp = Path(config.CONFIG_DIR, "args.json")
    args = Namespace(**utils.load_dict(filepath=args_fp))

    # 1. Set seed
    utils.set_seed(seed=args.seed)

    # 2. Set device
    device = utils.set_device(cuda=args.cuda)

    # 3. Load data
    projects_fp = Path(config.DATA_DIR, "projects.json")
    tags_fp = Path(config.DATA_DIR, "tags.json")
    projects = utils.load_dict(filepath=projects_fp)
    tags_dict = utils.list_to_dict(utils.load_dict(filepath=tags_fp), key="tag")
    df = pd.DataFrame(projects)
    if args.shuffle:
        df = df.sample(frac=1).reset_index(drop=True)
    df = df[: args.num_samples]  # None = all samples

    # 4. Clean data
    df, tags_above_frequency = data.clean(
        df=df,
        include=list(tags_dict.keys()),
        exclude=config.EXCLUDE,
        min_tag_freq=args.min_tag_freq,
    )

    # 5. Preprocess data
    df.text = df.text.apply(data.preprocess, lower=args.lower, stem=args.stem)

    # 6. Encode labels
    labels = df.tags
    label_encoder = data.MultiLabelLabelEncoder()
    label_encoder.fit(labels)
    y = label_encoder.encode(labels)

    # Class weights
    all_tags = list(itertools.chain.from_iterable(labels.values))
    counts = np.bincount([label_encoder.class_to_index[class_] for class_ in all_tags])
    class_weights = {i: 1.0 / count for i, count in enumerate(counts)}

    # 7. Split data
    utils.set_seed(seed=args.seed)  # needed for skmultilearn
    X = df.text.to_numpy()
    X_train, X_, y_train, y_ = data.iterative_train_test_split(X=X, y=y, train_size=args.train_size)
    X_val, X_test, y_val, y_test = data.iterative_train_test_split(X=X_, y=y_, train_size=0.5)

    # View slices
    test_df = pd.DataFrame({"text": X_test, "tags": label_encoder.decode(y_test)})
    cv_transformers_df = slice_dataframe(test_df, cv_transformers)
    print(f"{len(cv_transformers_df)} projects")
    print(cv_transformers_df[["text", "tags"]].head())
    short_text_df = slice_dataframe(test_df, short_text)
    print(f"{len(short_text_df)} projects")
    print(short_text_df[["text", "tags"]].head())

    # 8. Tokenize inputs
    tokenizer = data.Tokenizer(char_level=args.char_level)
    tokenizer.fit_on_texts(texts=X_train)
    X_train = np.array(tokenizer.texts_to_sequences(X_train), dtype=object)
    X_val = np.array(tokenizer.texts_to_sequences(X_val), dtype=object)
    X_test = np.array(tokenizer.texts_to_sequences(X_test), dtype=object)

    # 9. Create dataloaders
    train_dataset = data.CNNTextDataset(X=X_train, y=y_train, max_filter_size=args.max_filter_size)
    val_dataset = data.CNNTextDataset(X=X_val, y=y_val, max_filter_size=args.max_filter_size)
    test_dataset = data.CNNTextDataset(X=X_test, y=y_test, max_filter_size=args.max_filter_size)
    train_dataloader = train_dataset.create_dataloader(batch_size=args.batch_size)
    val_dataloader = val_dataset.create_dataloader(batch_size=args.batch_size)
    test_dataloader = test_dataset.create_dataloader(batch_size=args.batch_size)

    # Load artifacts
    runs = utils.get_sorted_runs(experiment_name="best", order_by=["metrics.f1 DESC"])
    run_ids = [run["run_id"] for run in runs]
    artifacts = main.load_artifacts(run_id=run_ids[0], device=torch.device("cpu"))

    # Evaluation
    device = torch.device("cpu")
    performance = evaluate(
        artifacts=artifacts,
        dataloader=test_dataloader,
        df=test_df,
        device=device,
    )
    logger.info(json.dumps(performance, indent=2))
