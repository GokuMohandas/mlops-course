# Applied ML in Production

## Set up
```bash
virtualenv -p python3.8 venv
source venv/bin/activate
pip install -r requirements.txt
```

## Start Jupyterlab
```bash
python -m ipykernel install --user --name=tagifai
jupyter labextension install @jupyter-widgets/jupyterlab-manager
jupyter labextension install @jupyterlab/toc
jupyter lab
```