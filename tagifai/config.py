# config.py
# Configurations.

import logging
import sys
from pathlib import Path

import mlflow
import pretty_errors  # NOQA: F401 (imported but unused)
from rich.logging import RichHandler

from tagifai import utils

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

# MLFlow
mlflow.set_tracking_uri("file://" + str(EXPERIMENTS_DIR.absolute()))

# Logger
logging_config = {
    "version": 1,
    "formatters": {
        "minimal": {"format": "%(message)s"},
        "detailed": {
            "format": "%(levelname)s %(asctime)s [%(filename)s:%(funcName)s:%(lineno)d]\n%(message)s\n"
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "stream": sys.stdout,
            "formatter": "minimal",
            "level": logging.DEBUG,
        },
        "info": {
            "class": "logging.handlers.RotatingFileHandler",
            "filename": Path(LOGS_DIR, "info.log"),
            "maxBytes": 10485760,  # 1 MB
            "backupCount": 10,
            "formatter": "detailed",
            "level": logging.INFO,
        },
        "error": {
            "class": "logging.handlers.RotatingFileHandler",
            "filename": Path(LOGS_DIR, "error.log"),
            "maxBytes": 10485760,  # 1 MB
            "backupCount": 10,
            "formatter": "detailed",
            "level": logging.ERROR,
        },
    },
    "root": {"handlers": ["console", "info", "error"], "level": logging.DEBUG},
}
logging.config.dictConfig(logging_config)
logger = logging.getLogger()
logger.handlers[0] = RichHandler(markup=True)
