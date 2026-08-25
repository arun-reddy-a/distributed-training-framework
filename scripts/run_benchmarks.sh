#!/usr/bin/env bash
# Run the full benchmark suite and save every artifact under results/<timestamp>/.
#
#   scripts/run_benchmarks.sh              # use all visible GPUs
#   NPROC=4 scripts/run_benchmarks.sh      # or pick a world size
#   BACKEND=gloo NPROC=4 scripts/run_benchmarks.sh    # CPU dry run
#
# On CPU the numbers exercise the code but say nothing about performance:
# Gloo has no device kernels, so there is no communication/computation overlap
# to measure and the collectives are orders of magnitude slower than NCCL.
set -euo pipefail

cd "$(dirname "$0")/.."

NPROC="${NPROC:-$(python -c 'import torch; print(max(1, torch.cuda.device_count()))')}"
BACKEND="${BACKEND:-}"
BACKEND_ARG=""
[ -n "$BACKEND" ] && BACKEND_ARG="--backend $BACKEND"

STAMP="$(date +%Y%m%d_%H%M%S)"
OUT="results/${STAMP}"
mkdir -p "$OUT/traces"

echo "==> world size ${NPROC}, results -> ${OUT}"
{
    echo "host:    $(uname -a)"
    echo "date:    $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "nproc:   ${NPROC}"
    python -c 'import torch; print("torch:  ", torch.__version__)'
    python -c 'import torch; print("cuda:   ", torch.version.cuda)'
    python -c 'import torch; [print(f"gpu {i}:  ", torch.cuda.get_device_name(i)) for i in range(torch.cuda.device_count())]'
    python -c 'import torch; print("nccl:   ", ".".join(map(str, torch.cuda.nccl.version())) if torch.cuda.is_available() else "n/a")'
} | tee "$OUT/environment.txt"

echo
echo "==> 1/3  collective microbenchmark"
torchrun --nproc_per_node="$NPROC" benchmarks/bench_collectives.py \
    --max-mb 256 --csv "$OUT/collectives.csv" $BACKEND_ARG \
    2>&1 | tee "$OUT/collectives.txt"

echo
echo "==> 2/3  parallelism strategies (with profiler traces)"
torchrun --nproc_per_node="$NPROC" benchmarks/bench_parallelism.py \
    --profile-dir "$OUT/traces" $BACKEND_ARG \
    2>&1 | tee "$OUT/parallelism.txt"

if [ "$NPROC" -ge 2 ]; then
    echo
    echo "==> 3/3  pipeline bubble sweep"
    torchrun --nproc_per_node="$NPROC" benchmarks/bench_pipeline.py \
        --microbatches 1,2,4,8,16 --gantt $BACKEND_ARG \
        2>&1 | tee "$OUT/pipeline.txt"
fi

echo
echo "==> overlap analysis"
python benchmarks/analyze_trace.py "$OUT"/traces/*rank0.json \
    --csv "$OUT/overlap.csv" 2>&1 | tee "$OUT/overlap.txt" || true

echo
echo "done -> ${OUT}"
