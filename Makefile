PYTHON ?= python
EXP ?= bench_01_delta_regr_illus_synth
OUT ?= runs
SUBMIT_ARGS ?=
DRY ?= 0

DRY_FLAG := $(if $(filter 1 true yes,$(DRY)),--dry-run,)

.DEFAULT_GOAL := help
.PHONY: help setup check list info smoke send plot

help:
	@printf '%s\n' 'make setup   Install the benchmark and test tools.' 'make check   Run tests and lint.' 'make smoke   Compute and plot a small synthetic example locally.' 'make list    List experiments.' 'make info    Describe EXP.' 'make send    Submit configured experiments to SLURM (DRY=1 previews).' 'make plot    Plot results from OUT.'

setup:
	$(PYTHON) -m pip install -e ".[test]"
	bash dyntr/datasets/extract_data.sh

check:
	$(PYTHON) -m pytest -q
	flake8 --ignore E501 dyntr benchmark benchmark_tasks.py scripts

list:
	$(PYTHON) -m benchmark_tasks list

info:
	$(PYTHON) -m benchmark_tasks info $(EXP)

smoke:
	$(PYTHON) -m benchmark_tasks run $(EXP) --config benchmark/configs/config-blank.yml --output-dir $(OUT)
	$(PYTHON) -m benchmark_tasks plot $(EXP) --config benchmark/configs/config-blank.yml --output-dir $(OUT)

send:
	$(PYTHON) -m benchmark_tasks submit --output-dir $(OUT) $(SUBMIT_ARGS) $(DRY_FLAG)

plot:
	$(PYTHON) -m benchmark_tasks plot --output-dir $(OUT)
