In this file we're setting up the configuration needed for all our workflows.

First up is creating required directories so we can save and load from them:
```python
# Directories
BASE_DIR = Path(__file__).parent.parent.absolute()
CONFIG_DIR = Path(BASE_DIR, "config")
LOGS_DIR = Path(BASE_DIR, "logs")
DATA_DIR = Path(BASE_DIR, "data")
MODEL_REGISTRY = Path(BASE_DIR, "experiments")
DVC_REMOTE_STORAGE = Path(BASE_DIR, "tmp/dvcstore")

# Create dirs
LOGS_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)
MODEL_REGISTRY.mkdir(parents=True, exist_ok=True)
DVC_REMOTE_STORAGE.mkdir(parents=True, exist_ok=True)
```

Then, we'll set the tracking URI for all MLFlow experiments:
```python
# MLFlow
mlflow.set_tracking_uri("file://" + str(MODEL_REGISTRY.absolute()))
```

Finally, we'll establish our logger using the `logging_config` dictionary:
```python
# Logger
logging.config.dictConfig(logging_config)
logger = logging.getLogger("root")
logger.handlers[0] = RichHandler(markup=True)
```

::: tagifai.config