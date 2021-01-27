All the functions here can be used as a CLI command thanks to your Typer application.
```bash
# View all Typer commands
$ tagifai --help
Usage: tagifai [OPTIONS] COMMAND [ARGS]
👉  Commands:
    download-data  Download data from online to local drive.
    optimize       Optimize a subset of hyperparameters towards ...
    train-model    Predict tags for a give input text using a ...
    predict-tags   Train a model using the specified parameters.
```

View individual commands and their arguments:
```bash
# Help for a specific command
$ tagifai train-model --help
Usage: tagifai train-model [OPTIONS]
Options:
    --args-fp  PATH [default: config/args.json]
    --help     Show this message and exit.

# Train a model
$ tagifai train-model --args-fp $PATH
🚀 Training...
```

::: tagifai.main