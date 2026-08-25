#!/usr/bin/env bash
# Full correctness suite on the Gloo/CPU backend.
#
# Every test spawns a real multi-process group, so no GPU is needed and none is
# used. What this does *not* cover is NCCL's own kernels and anything about
# bandwidth -- see scripts/run_benchmarks.sh for that.
set -euo pipefail

cd "$(dirname "$0")/.."

echo "==> ruff"
ruff check minidist tests benchmarks examples

echo
echo "==> pytest"
pytest -v "$@"

echo
echo "==> torchrun smoke test (3D parallel: tp=2 x pp=2)"
torchrun --nproc_per_node=4 examples/train_gpt.py \
    --tp 2 --pp 2 --strategy none --microbatches 4 \
    --layers 4 --embd 128 --heads 4 --vocab 1024 --seq-len 64 \
    --batch-size 8 --steps 4 --backend gloo

echo
echo "all green"
