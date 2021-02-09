In this file we're setting up the configuration needed for all our workflows.

First up is creating required directories so we can save and load from them:
```python
# Directories
BASE_DIR = Path(__file__).parent.parent.absolute()
CONFIG_DIR = Path(BASE_DIR, "config")
LOGS_DIR = Path(BASE_DIR, "logs")
ASSETS_DIR = Path(BASE_DIR, "assets")
DATA_DIR = Path(ASSETS_DIR, "data")
EXPERIMENTS_DIR = Path(ASSETS_DIR, "experiments")

# Create dirs
utils.create_dirs(dirpath=LOGS_DIR)
utils.create_dirs(dirpath=ASSETS_DIR)
utils.create_dirs(dirpath=DATA_DIR)
utils.create_dirs(dirpath=EXPERIMENTS_DIR)
```

Then, we'll set the tracking URI for all MLFlow experiments:
```python
# MLFlow
mlflow.set_tracking_uri("file://" + str(EXPERIMENTS_DIR.absolute()))
```

Finally, we'll establish our logger using the `logging_config` dictionary:
```python
# Logger
logging.config.dictConfig(logging_config)
logger = logging.getLogger("root")
logger.handlers[0] = RichHandler(markup=True)
```

::: tagifai.config