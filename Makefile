.PHONY: help setup format lint lint-rtl type-check test test-python test-rtl test-cocotb

PYTHON ?= python3

help:
	@echo "TensorWright development targets"
	@echo "  setup       Install TensorWright and development tools"
	@echo "  format      Format Python source"
	@echo "  lint        Check Python formatting and lint rules"
	@echo "  type-check  Run static type checking"
	@echo "  test        Run the Python test suite"
	@echo "  lint-rtl    Lint the arithmetic RTL with Verilator"
	@echo "  test-rtl    Run vector-driven RTL tests with Verilator"
	@echo "  test-cocotb Run cocotb RTL tests when supported"

setup:
	$(PYTHON) -m pip install -e ".[dev]"

format:
	$(PYTHON) -m ruff format tensorwright tests scripts verification/cocotb
	$(PYTHON) -m ruff check --fix tensorwright tests scripts verification/cocotb

lint:
	$(PYTHON) -m ruff format --check tensorwright tests scripts verification/cocotb
	$(PYTHON) -m ruff check tensorwright tests scripts verification/cocotb

type-check:
	$(PYTHON) -m mypy

test:
	$(PYTHON) -m unittest discover -s tests -v

test-python: test

lint-rtl:
	verilator --lint-only --timing -Wall -Wno-fatal -Wno-MULTITOP \
		rtl/compute/tensorwright_multiplier.sv \
		rtl/compute/tensorwright_mac.sv \
		rtl/compute/tensorwright_adder_tree.sv \
		rtl/compute/tensorwright_channel_accumulator.sv \
		rtl/postprocess/tensorwright_postprocess.sv \
		rtl/compute/tensorwright_arithmetic_core.sv

test-rtl:
	$(PYTHON) -m scripts.run_verilator_tests

test-cocotb:
	$(PYTHON) -m scripts.run_rtl_tests
