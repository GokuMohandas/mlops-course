help:
	@echo "Commands:"
	@echo "install         : installs required packages."
	@echo "install-dev     : installs development requirements."
	@echo "install-test    : installs test requirements."

install:
	python -m pip install -e .

install-dev:
	python -m pip install -e ".[dev]"

install-test:
	python -m pip install -e ".[test]"