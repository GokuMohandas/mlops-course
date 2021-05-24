# Makefile
.PHONY: help
help:
	@echo "Commands:"
	@echo "venv   : creates development environment."
	@echo "style  : runs style formatting."
	@echo "clean  : cleans all unecessary files."
	@echo "dvc    : pushes versioned artifacts to blob storage."
	@echo "test   : run non-training tests."

# Environment
.ONESHELL:
venv:
	python3 -m venv venv
	source venv/bin/activate && \
	python -m pip install --upgrade pip setuptools wheel && \
	python -m pip install -e ".[dev]" --no-cache-dir && \
	pre-commit install && \
	pre-commit autoupdate && \
	pip uninstall dataclasses -y

# Styling
.PHONY: style
style:
	black .
	flake8
	isort .

# Cleaning
.PHONY: clean
clean: style
	find . -type f -name "*.DS_Store" -ls -delete
	find . | grep -E "(__pycache__|\.pyc|\.pyo)" | xargs rm -rf
	find . | grep -E ".pytest_cache" | xargs rm -rf
	find . | grep -E ".ipynb_checkpoints" | xargs rm -rf
	rm -f .coverage

# DVC
.PHONY: dvc
dvc:
	dvc add data/projects.json
	dvc add data/tags.json
	dvc add data/features.json
	dvc add data/projects.parquet
	dvc push

# Test
.PHONY: test
test:
	great_expectations checkpoint run projects
	great_expectations checkpoint run tags
	pytest -m "not training"
