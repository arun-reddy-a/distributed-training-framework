#!/usr/bin/env python3
"""Measure the pipeline bubble against its analytical prediction.

.. code-block:: bash

    torchrun --nproc_per_node=4 benchmarks/bench_pipeline.py --microbatches 1,2,4,8,16

The theory says the bubble fraction is ``(P-1)/(M+P-1)`` for both GPipe and
1F1B.  This sweeps ``M``, measures what each stage actually spent outside
forward and backward, and prints the two side by side.

Expect **measured >= theoretical**, and the gap is the interesting part.  The
analytical formula counts only the structural fill and drain; a real pipeline
also pays for

* **stage imbalance** — uniform layer partitioning puts the embedding on stage 0
  and the LM head on stage P-1, so those stages do more work per microbatch and
  everyone else waits on them.  Per-stage busy times below make this visible.
* **transfer time** — the activation send/recv is not free, and on a slow
  interconnect it is not hidden.
* **launch overhead** — smaller microbatches mean more, smaller kernels.

The peak-activations column is the other half of the story: GPipe holds ``M``
microbatches of activations at stage 0 while 1F1B holds about ``P``.  Both have
the same bubble; only one of them lets you raise ``M`` far enough to shrink it.
"""

from __future__ import annotations

import argparse
import time

import torch
import torch.distributed as dist
from torch import nn

from minidist import (
    ParallelMesh,
    PipelineStage,
    build_schedule,
    init_distributed,
    partition_layers,
    print_rank0,
    shutdown_distributed,
    theoretical_bubble_fraction,
)
from minidist.models import GPTConfig, build_pipeline_layers
from minidist.tensor_parallel import vocab_parallel_cross_entropy


def build_stage(config, mesh, device):
    layers = build_pipeline_layers(config, tp_group=mesh.tp_group, device=device)
    mine = partition_layers(layers, mesh.pp_size, mesh.pp_rank)
    vocab_start = mesh.tp_rank * (config.vocab_size // mesh.tp_size)

    def loss_fn(logits, targets):
        return vocab_parallel_cross_entropy(
            logits, targets, group=mesh.tp_group, vocab_start=vocab_start
        )

    return PipelineStage(
        nn.Sequential(*mine), mesh, loss_fn=loss_fn if mesh.is_last_stage else None
    ), len(mine)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--microbatches", default="1,2,4,8", type=lambda s: [int(x) for x in s.split(",")])
    p.add_argument("--schedules", default="gpipe,1f1b", type=lambda s: s.split(","))
    p.add_argument("--tp", type=int, default=1)
    p.add_argument("--layers", type=int, default=8)
    p.add_argument("--heads", type=int, default=8)
    p.add_argument("--embd", type=int, default=512)
    p.add_argument("--vocab", type=int, default=8192)
    p.add_argument("--seq-len", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--iters", type=int, default=5)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--backend", default=None, choices=[None, "nccl", "gloo"])
    p.add_argument("--gantt", action="store_true", help="print an ASCII schedule timeline")
    args = p.parse_args()

    rank, world, device = init_distributed(backend=args.backend)
    mesh = ParallelMesh(tp_size=args.tp, pp_size=world // args.tp)
    if mesh.pp_size < 2:
        print_rank0("  pipeline benchmark needs pp_size >= 2; launch more ranks")
        shutdown_distributed()
        return

    config = GPTConfig(
        vocab_size=args.vocab, block_size=args.seq_len, n_layer=args.layers,
        n_head=args.heads, n_embd=args.embd, seed=args.seed,
    )
    torch.manual_seed(args.seed)
    g = torch.Generator().manual_seed(args.seed)
    ids = torch.randint(0, config.vocab_size, (args.batch_size, config.block_size), generator=g).to(device)
    targets = torch.randint(0, config.vocab_size, (args.batch_size, config.block_size), generator=g).to(device)

    print_rank0(f"\n{'=' * 96}")
    print_rank0(f"  pipeline bubble sweep   P={mesh.pp_size} stages   "
                f"batch={args.batch_size}   {config.n_layer}L x {config.n_embd}d   device={device.type}")
    print_rank0("=" * 96)
    print_rank0(f"  {'schedule':<10}{'M':>4}{'ms/step':>10}{'measured':>11}{'theory':>9}"
                f"{'gap':>8}{'live acts':>11}{'comm ms':>10}   per-stage busy ms")
    print_rank0("  " + "-" * 92)

    for schedule_name in args.schedules:
        for num_microbatches in args.microbatches:
            if args.batch_size % num_microbatches:
                continue
            stage, n_layers = build_stage(config, mesh, device)
            optimizer = torch.optim.SGD(stage.parameters(), lr=0.0)
            schedule = build_schedule(schedule_name, stage, mesh, num_microbatches)

            schedule.step(inputs=ids, targets=targets)  # warm up
            optimizer.zero_grad(set_to_none=True)

            start = time.perf_counter()
            for _ in range(args.iters):
                optimizer.zero_grad(set_to_none=True)
                schedule.step(inputs=ids, targets=targets)
            elapsed = (time.perf_counter() - start) / args.iters

            report = schedule.report()
            # Gather every stage's view; the bubble is a property of the whole
            # pipeline, and stage 0 sees a very different one from stage P-1.
            gathered: list[dict] = [None] * mesh.pp_size  # type: ignore[list-item]
            dist.all_gather_object(gathered, report, group=mesh.pp_group)

            if mesh.rank == 0:
                measured = max(r["measured_bubble"] for r in gathered)
                theory = theoretical_bubble_fraction(mesh.pp_size, num_microbatches)
                live = max(r["peak_activations_in_flight"] for r in gathered)
                comm_ms = max(r["comm_s"] for r in gathered) * 1e3
                busy = "  ".join(f"{r['busy_s'] * 1e3:.1f}" for r in gathered)
                print_rank0(
                    f"  {schedule_name:<10}{num_microbatches:>4}{elapsed * 1e3:>10.2f}"
                    f"{measured * 100:>10.1f}%{theory * 100:>8.1f}%"
                    f"{(measured - theory) * 100:>+7.1f}%{live:>11.0f}{comm_ms:>10.2f}   {busy}"
                )

            if args.gantt and num_microbatches == args.microbatches[-1]:
                rows: list[str] = [None] * mesh.pp_size  # type: ignore[list-item]
                dist.all_gather_object(
                    rows, schedule.timeline.ascii_gantt(), group=mesh.pp_group
                )
                if mesh.rank == 0:
                    print_rank0(f"\n  {schedule_name} timeline, M={num_microbatches} "
                                "(F=forward B=backward >=send <=recv .=idle)")
                    for i, row in enumerate(rows):
                        print_rank0(f"    stage {i}  {row}")
                    print_rank0("")
            del stage, schedule, optimizer

    print_rank0("")
    shutdown_distributed()


if __name__ == "__main__":
    main()
