## Use existing model

1. Set up environment.
```bash
export venv_name="venv"
make venv name=${venv_name} env="prod"
source ${venv_name}/bin/activate
```

2. Pull latest model.
```bash
dvc pull experiments
tagifai fix-artifact-metadata
```

3. Run Application
```bash
make app env="dev"
```
You can interact with the API directly or explore via the generated documentation at [http://0.0.0.0:5000/docs](http://0.0.0.0:5000/docs).

## Update model (CI/CD)
Coming soon after CI/CD lesson where the entire application will be retrained and deployed when we push new data (or trigger manual reoptimization/training). The deployed model, with performance comparisons to previously deployed versions, will be ready on a PR to push to the main branch.

## Update model (manual)

1. Set up the development environment.
```bash
export venv_name="venv"
make venv name=${venv_name} env="dev"
source ${venv_name}/bin/activate
```

2. Pull versioned data.
```bash
dvc pull data/tags.json
dvc pull data/projects.json
```

3. Optimize using distributions specified in `tagifai.main.objective`. This also writes the best model's params to [config/params.json](https://github.com/GokuMohandas/applied-ml/blob/main/config/params.json)
```bash
tagifai optimize \
    --params-fp config/params.json \
    --study-name optimization \
    --num-trials 100
```
> We'll cover how to train using compute instances on the cloud from Amazon Web Services (AWS) or Google Cloud Platforms (GCP) in later lessons. But in the meantime, if you don't have access to GPUs, check out the [optimize.ipynb](https://colab.research.google.com/github/GokuMohandas/applied-ml/blob/main/notebooks/optimize.ipynb) notebook for how to train on Colab and transfer to local. We essentially run optimization, then train the best model to download and transfer it's artifacts.

4. Train a model (and save all it's artifacts) using params from [config/params.json](https://github.com/GokuMohandas/applied-ml/blob/main/config/params.json) and publish metrics to [metrics/performance.json](https://github.com/GokuMohandas/applied-ml/blob/main/metrics/performance.json). You can view the entire run's details inside `experiments/{experiment_id}/{run_id}` or via the API (`GET` /runs/{run_id}).
```bash
tagifai train-model \
    --params-fp config/params.json \
    --experiment-name best \
    --run-name model \
    --publish-metrics  # save to metrics/performance.json
```

5. Predict tags for an input sentence. It'll use the best model saved from `train-model` but you can also specify a `run-id` to choose a specific model.
```bash
tagifai predict-tags --text "Transfer learning with BERT"  # test with CLI app
make app env="dev"  # run API and test if you want
```

6. View improvements
Once you're done training the best model using the current data version, best hyperparameters, etc., we can view performance difference.
```bash
tagifai diff --commit-a workspace --commit-b HEAD
```

7. Commit to git
This will clean and update versioned assets (data, experiments), run tests, styling, etc.
```bash
git add .
git commit -m ""
<precommit (dvc, tests, style, clean, etc.) will execute>
git push origin main
```