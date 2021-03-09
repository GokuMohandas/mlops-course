## Environment
Preparing the development environment.
```bash
export venv_name="venv"
make venv name=${venv_name}
source ${venv_name}/bin/activate
make assets
```

## Assets
If you've already created a virtual environment previously, be sure to update your assets such as data, previous runs, etc.
```bash
make assets
```

## CLI app
View all available options using the CLI application.
```bash
# View all CLI commands
$ tagifai --help
```
<pre>
Usage: tagifai [OPTIONS] COMMAND [ARGS]
👉  Commands:
    download-data  Download data from online to local drive.
    optimize       Optimize a subset of hyperparameters towards ...
    train-model    Predict tags for a give input text using a ...
    predict-tags   Train a model using the specified parameters.
</pre>
```bash
# Help for a specific command
$ tagifai train-model --help
```
<pre>
Usage: tagifai train-model [OPTIONS]
Options:
    --args-fp  PATH [default: config/args.json]
    --help     Show this message and exit.
</pre>
```bash
# Train a model
$ tagifai train-model --args-fp $PATH
```
<pre>
🚀 Training...
</pre>

## Optimize
Optimize using distributions specified in `tagifai.main.objective`. This also writes the best model's args to `config/args.json`.
```bash
tagifai optimize --args-fp config/args.json --study-name optimization --num-trials 100
```
> We'll cover how to train using compute instances on the cloud from Amazon Web Services (AWS) or Google Cloud Platforms (GCP) in later lessons. But in the meantime, if you don't have access to GPUs, check out the [optimize.ipynb](https://colab.research.google.com/github/GokuMohandas/applied-ml/blob/main/notebooks/optimize.ipynb){:target="_blank"} notebook for how to train on Colab and transfer to local. We essentially run optimization, then train the best model to download and transfer it's arguments and artifacts. Once we have them in our local machine, we can run `tagifai set-artifact-metadata` to match all metadata as if it were run from your machine.

## Train
Train a model (and save all it's artifacts) using args from `config/args.json`.
```bash
tagifai train-model --args-fp config/args.json --experiment-name best --run-name model
```

## Predict
Predict tags for an input sentence. It'll use the best model saved from `train-model` but you can also specify a `run-id` to choose a specific model.
```bash
tagifai predict-tags --text "Transfer learning with BERT"
```
