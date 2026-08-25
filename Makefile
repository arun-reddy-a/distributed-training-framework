.PHONY: help install test lint fmt bench bench-cpu smoke clean

NPROC ?= 4

help:
	@echo "make install     editable install plus dev dependencies"
	@echo "make test        full correctness suite (Gloo/CPU, no GPU needed)"
	@echo "make lint        ruff"
	@echo "make smoke       3D-parallel training smoke test under torchrun"
	@echo "make bench       full benchmark suite (NPROC=$(NPROC))"
	@echo "make bench-cpu   benchmark suite on Gloo/CPU (exercises code, not perf)"

install:
	pip install -e ".[dev]"

test:
	pytest -v

lint:
	ruff check minidist tests benchmarks examples

fmt:
	ruff check --fix minidist tests benchmarks examples

smoke:
	torchrun --nproc_per_node=4 examples/train_gpt.py \
		--tp 2 --pp 2 --strategy none --microbatches 4 \
		--layers 4 --embd 128 --heads 4 --vocab 1024 --seq-len 64 \
		--batch-size 8 --steps 4 --backend gloo

bench:
	NPROC=$(NPROC) scripts/run_benchmarks.sh

bench-cpu:
	BACKEND=gloo NPROC=$(NPROC) scripts/run_benchmarks.sh

clean:
	rm -rf .pytest_cache .ruff_cache **/__pycache__ *.egg-info build dist
