# Makefile
.PHONY: help
help:
	@echo "Commands:"
	@echo "install            : installs requirements."
	@echo "install-dev        : installs development requirements."
	@echo "install-test       : installs test requirements."
	@echo "venv               : set up the virtual environment for development"
	@echo "app                : launches FastAPI app with uvicorn worker"
	@echo "app-prod           : launches FastAPI app with uvicorn workers managed by guincorn"
	@echo "test               : runs all tests."
	@echo "test-non-training  : runs tests that don't involve training."
	@echo "style              : runs style formatting."
	@echo "clean              : cleans all unecessary files."
	@echo "docs               : serve generated documentation."

# Installation
.PHONY: install
install:
	python -m pip install -e . --no-cache-dir

.PHONY: install-dev
install-dev:
	python -m pip install -e ".[dev]" --no-cache-dir
	pre-commit install
	pre-commit autoupdate

.PHONY: install-test
install-test:
	python -m pip install -e ".[test]" --no-cache-dir

venv:
	python3 -m venv ${name}
	source ${name}/bin/activate && \
	python -m pip install --upgrade pip setuptools wheel && \
	make install-$(env)

# Docker
.PHONY: docker
docker:
	docker build -t tagifai:latest -f Dockerfile .
	docker run -p 5000:5000 --name tagifai tagifai:latest

# Application
.PHONY: app
SHELL ::= /bin/bash
app:
ifeq (${env}, prod)
	gunicorn -c config/gunicorn.py -k uvicorn.workers.UvicornWorker app.api:app
else
	uvicorn app.api:app --host 0.0.0.0 --port 5000 --reload --reload-dir tagifai --reload-dir app
endif

# MLFlow
.PHONY: mlflow
mlflow:
	mlflow server -h 0.0.0.0 -p 5000 --backend-store-uri experiments/

# Streamlit dashboard
.PHONY: streamlit
streamlit:
	streamlit run streamlit/st_app.py

# DVC
.PHONY: dvc
dvc:
	dvc add data/projects.json
	dvc add data/tags.json
	dvc add model/label_encoder.json
	dvc add model/tokenizer.json
	dvc add model/model.pt
	dvc push

# Tests
.PHONY: great-expectations
great-expectations:
	great_expectations checkpoint run projects
	great_expectations checkpoint run tags

.PHONY: test
test: great-expectations
	pytest --cov tagifai --cov app --cov-report html

.PHONY: test-non-training
test-non-training: great-expectations
	pytest -m "not training"

# Styling
.PHONY: style
style:
	black .
	flake8
	isort .

# Cleaning
.PHONY: clean
clean:
	find . -type f -name "*.DS_Store" -ls -delete
	find . | grep -E "(__pycache__|\.pyc|\.pyo)" | xargs rm -rf
	find . | grep -E ".pytest_cache" | xargs rm -rf
	find . | grep -E ".ipynb_checkpoints" | xargs rm -rf
	rm -f .coverage

# Documentation
.PHONY: docs
docs:
	python -m mkdocs serve
