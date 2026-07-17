.PHONY: help setup format lint type-check test test-python

PYTHON ?= python3

help:
	@echo "TensorWright development targets"
	@echo "  setup       Install TensorWright and development tools"
	@echo "  format      Format Python source"
	@echo "  lint        Check Python formatting and lint rules"
	@echo "  type-check  Run static type checking"
	@echo "  test        Run the Python test suite"

setup:
	$(PYTHON) -m pip install -e ".[dev]"

format:
	$(PYTHON) -m ruff format tensorwright tests
	$(PYTHON) -m ruff check --fix tensorwright tests

lint:
	$(PYTHON) -m ruff format --check tensorwright tests
	$(PYTHON) -m ruff check tensorwright tests

type-check:
	$(PYTHON) -m mypy

test:
	$(PYTHON) -m unittest discover -s tests -v

test-python: test
