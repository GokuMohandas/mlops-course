## Lessons
All the Applied ML course lessons can be found [here](https://madewithml.com/#applied-ml){:target="_blank"}.

- [Getting started](workflows.md): Overview of common workflows and examples.
- [Reference](tagifai/main.md): Detailed documentation of each script and it's respective functions and classes.
- [API](api.md): Documentation of the different API calls with information on payloads, responses, etc.

## Set up
```bash
python3 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
make install-dev
```

## Start Jupyterlab
```bash
python -m ipykernel install --user --name=tagifai
jupyter labextension install @jupyter-widgets/jupyterlab-manager
jupyter labextension install @jupyterlab/toc
jupyter lab
```
> You can also run all notebooks on [Google Colab](https://colab.research.google.com/github/GokuMohandas/applied-ml/blob/main/notebooks/tagifai.ipynb){:target="_blank"}.

## Directory structure
```
tagifai/
├── config.py     - configuration setup
├── data.py       - data processing utilities
├── main.py       - main operations (CLI)
├── models.py     - model architectures
├── predict.py    - inference utilities
├── train.py      - training utilities
└── utils.py      - supplementary utilities
```
> Documentation can be found [here](https://gokumohandas.github.io/applied-ml/){:target="_blank"}.

## MLFlow
```
mlflow server -h 0.0.0.0 -p 5000 --backend-store-uri assets/experiments/
```

## Mkdocs
```
python -m mkdocs serve
```