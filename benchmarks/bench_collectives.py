#!/usr/bin/env python3
"""Microbenchmark the four collectives the framework depends on.

.. code-block:: bash

    torchrun --nproc_per_node=8 benchmarks/bench_collectives.py --max-mb 256

Reports two bandwidths per size, and the difference matters:

**algorithm bandwidth** = ``payload_bytes / time``.  What the caller
experiences.  It keeps rising with rank count for a fixed payload, which makes
it useless for comparing across world sizes.

**bus bandwidth** = the volume actually crossing the interconnect / time.  For a
ring all-reduce each rank sends ``2(W-1)/W`` times its payload (a
reduce-scatter around the ring, then an all-gather back), so bus bandwidth
stays roughly flat as ``W`` grows and can be compared directly against the
link's specified peak.  This is the number ``nccl-tests`` reports and the one
worth quoting.

============  =====================================
collective    bus bytes for payload ``S``
============  =====================================
all-reduce    ``2 * (W-1)/W * S``
all-gather    ``(W-1)/W * S`` (S = output size)
reduce-scat.  ``(W-1)/W * S`` (S = input size)
broadcast     ``S``
============  =====================================

Note the identity worth internalising: **all-reduce = reduce-scatter +
all-gather**, and it costs exactly twice either one.  That is the entire reason
FSDP's communication bill is 1.5x DDP's rather than 3x — FSDP does one
all-gather forward, one all-gather backward and one reduce-scatter, i.e. three
half-price operations against DDP's one full-price one.
"""

from __future__ import annotations

import argparse
import csv
import time
from dataclasses import dataclass

import torch
import torch.distributed as dist

from minidist import init_distributed, print_rank0, shutdown_distributed


@dataclass
class Result:
    collective: str
    payload_bytes: int
    seconds: float
    bus_bytes: float

    @property
    def alg_gbps(self) -> float:
        return self.payload_bytes / self.seconds / 1e9

    @property
    def bus_gbps(self) -> float:
        return self.bus_bytes / self.seconds / 1e9


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def time_collective(fn, device, iters: int, warmup: int) -> float:
    """Median-ish timing, reported as the slowest rank's average.

    A collective is only complete when every rank has finished, so the
    meaningful number is the maximum across ranks, not this rank's own view.
    """
    for _ in range(warmup):
        fn()
    _sync(device)
    dist.barrier()

    start = time.perf_counter()
    for _ in range(iters):
        fn()
    _sync(device)
    elapsed = torch.tensor([(time.perf_counter() - start) / iters], device=device)

    dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
    return float(elapsed.item())


def run(args) -> list[Result]:
    rank, world, device = init_distributed(backend=args.backend)
    dtype = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}[args.dtype]
    itemsize = torch.empty((), dtype=dtype).element_size()

    print_rank0(f"\n{'=' * 78}")
    print_rank0(f"  collective microbenchmark   world={world}  backend={dist.get_backend()}  "
                f"dtype={args.dtype}  device={device.type}")
    print_rank0("=" * 78)
    print_rank0(f"  {'collective':<16}{'size':>12}{'time':>12}{'algbw':>12}{'busbw':>12}")
    print_rank0(f"  {'':<16}{'(MB)':>12}{'(ms)':>12}{'(GB/s)':>12}{'(GB/s)':>12}")
    print_rank0("  " + "-" * 62)

    results: list[Result] = []
    size = args.min_kb * 1024
    while size <= args.max_mb * 1024 * 1024:
        numel = size // itemsize
        # Every collective needs a payload divisible by the world size.
        numel = (numel // world) * world
        if numel == 0:
            size *= 2
            continue
        payload = numel * itemsize

        buf = torch.ones(numel, dtype=dtype, device=device)
        shard = torch.ones(numel // world, dtype=dtype, device=device)

        specs = [
            # Bind the buffers as defaults: these lambdas outlive the loop
            # iteration that built them, and late binding would silently
            # benchmark the largest size every time.
            ("all_reduce", lambda b=buf: dist.all_reduce(b),
             2 * (world - 1) / world * payload),
            ("all_gather", lambda b=buf, s=shard: dist.all_gather_into_tensor(b, s),
             (world - 1) / world * payload),
            ("reduce_scatter", lambda b=buf, s=shard: dist.reduce_scatter_tensor(s, b),
             (world - 1) / world * payload),
            ("broadcast", lambda b=buf: dist.broadcast(b, src=0), float(payload)),
        ]
        for name, fn, bus in specs:
            if name not in args.collectives:
                continue
            seconds = time_collective(fn, device, args.iters, args.warmup)
            result = Result(name, payload, seconds, bus)
            results.append(result)
            print_rank0(
                f"  {name:<16}{payload / 1024**2:>12.2f}{seconds * 1e3:>12.3f}"
                f"{result.alg_gbps:>12.2f}{result.bus_gbps:>12.2f}"
            )
        print_rank0("  " + "-" * 62)
        size *= 2

    if args.csv and rank == 0:
        with open(args.csv, "w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(
                ["collective", "world_size", "payload_bytes", "seconds", "alg_gbps", "bus_gbps"]
            )
            for r in results:
                writer.writerow(
                    [r.collective, world, r.payload_bytes, f"{r.seconds:.9f}",
                     f"{r.alg_gbps:.4f}", f"{r.bus_gbps:.4f}"]
                )
        print_rank0(f"\n  wrote {args.csv}")

    shutdown_distributed()
    return results


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--min-kb", type=int, default=64)
    p.add_argument("--max-mb", type=int, default=128)
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--dtype", default="fp32", choices=["fp32", "bf16", "fp16"])
    p.add_argument("--backend", default=None, choices=[None, "nccl", "gloo"])
    p.add_argument("--csv", default=None, help="write results here")
    p.add_argument(
        "--collectives",
        default="all_reduce,all_gather,reduce_scatter,broadcast",
        type=lambda s: s.split(","),
    )
    run(p.parse_args())


if __name__ == "__main__":
    main()
