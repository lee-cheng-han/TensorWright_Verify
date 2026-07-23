.PHONY: help setup format lint lint-rtl type-check test test-python test-rtl test-cocotb demo demo-clean demo-numerical-fault demo-protocol-fault demo-model demo-bundle-rtl synth implement benchmark release-check

PYTHON := $(if $(wildcard .venv/bin/python),.venv/bin/python,python3)

help:
	@echo "TensorWright development targets"
	@echo "  setup       Install TensorWright and development tools"
	@echo "  format      Format Python source"
	@echo "  lint        Check Python formatting and lint rules"
	@echo "  type-check  Run static type checking"
	@echo "  test        Run the Python test suite"
	@echo "  lint-rtl    Lint arithmetic, streaming, and control RTL"
	@echo "  test-rtl    Run vector-driven RTL tests with Verilator"
	@echo "  test-cocotb Run cocotb RTL tests when supported"
	@echo "  demo        Run the video-friendly reference-versus-RTL demo"
	@echo "  demo-clean  Run the demo and highlight the known-good baseline"
	@echo "  demo-numerical-fault  Highlight arithmetic diagnosis"
	@echo "  demo-protocol-fault   Highlight stream protocol diagnosis"
	@echo "  demo-model  Compile and simulate a recognizable digit classifier"
	@echo "  demo-bundle-rtl  Compile a .twmodel and execute it on real Verilator RTL"
	@echo "  synth       Run board-independent Zybo Z7-20 Vivado synthesis"
	@echo "  implement   Place and route the accelerator IP at 100 MHz"
	@echo "  benchmark   Benchmark the recognizable demo bundle"
	@echo "  release-check  Run every board-independent open-source check"

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
		rtl/compute/tensorwright_arithmetic_core.sv \
		rtl/interfaces/tensorwright_stream_if.sv \
		rtl/memory/tensorwright_stream_fifo.sv \
		rtl/memory/tensorwright_activation_buffer.sv \
		rtl/memory/tensorwright_weight_buffer.sv \
		rtl/memory/tensorwright_window_generator_3x3.sv \
		rtl/control/tensorwright_registers_pkg.sv \
		rtl/interfaces/tensorwright_axil_if.sv \
		rtl/control/tensorwright_control.sv \
		rtl/engine/tensorwright_convolution_engine.sv \
		rtl/tensorwright_top.sv

test-rtl:
	$(PYTHON) -m scripts.run_verilator_tests

test-cocotb:
	$(PYTHON) -m scripts.run_rtl_tests

demo:
	$(PYTHON) -m scripts.bootstrap_demo $(DEMO_ARGS)

demo-clean:
	$(PYTHON) -m scripts.bootstrap_demo --focus clean $(DEMO_ARGS)

demo-numerical-fault:
	$(PYTHON) -m scripts.bootstrap_demo --focus numerical $(DEMO_ARGS)

demo-protocol-fault:
	$(PYTHON) -m scripts.bootstrap_demo --focus protocol $(DEMO_ARGS)

demo-model:
	$(PYTHON) -m scripts.run_model_demo

demo-bundle-rtl:
	$(PYTHON) -m scripts.run_bundle_rtl_demo

synth:
	$(PYTHON) -m scripts.run_synthesis

implement:
	$(PYTHON) -m scripts.run_implementation

benchmark: demo-model
	$(PYTHON) -m tensorwright.cli benchmark build/model_demo/seven_segment_digits.twmodel

release-check: lint type-check test lint-rtl test-rtl demo-model demo-bundle-rtl
	$(PYTHON) -m scripts.bootstrap_demo --pace 0
