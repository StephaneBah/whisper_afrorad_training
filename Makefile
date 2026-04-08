ifeq ($(OS),Windows_NT)
VENV_PYTHON := .venv/Scripts/python.exe
else
VENV_PYTHON := .venv/bin/python
endif

PYTHON ?= python

ifneq ("$(wildcard $(VENV_PYTHON))","")
PYTHON := $(VENV_PYTHON)
endif

.PHONY: setup doctor train eval run test lint

setup:
	$(PYTHON) -m pip install -U pip
	$(PYTHON) -m pip install -e .[dev]

doctor:
	$(PYTHON) -m afrorad_pipeline.doctor

train:
	$(PYTHON) -m accelerate.commands.launch -m afrorad_pipeline.train

eval:
	$(PYTHON) -m accelerate.commands.launch -m afrorad_pipeline.eval

run:
	$(MAKE) doctor
	$(MAKE) train
	$(MAKE) eval

test:
	pytest -q

lint:
	ruff check src tests
