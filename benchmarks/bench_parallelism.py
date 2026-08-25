#!/usr/bin/env python3
"""Compare data-parallel strategies on one mesh: time, memory, and bytes moved.

.. code-block:: bash

    torchrun --nproc_per_node=8 benchmarks/bench_parallelism.py
    torchrun --nproc_per_node=8 benchmarks/bench_parallelism.py --tp 2 --embd 1024

Runs the same model and batch under each strategy along the data-parallel axis
and reports four things per strategy:

* **ms/step** and tokens/s — the headline.
* **comm bytes/step** — counted by instrumenting the collectives themselves
  (:func:`minidist.comm.record_comm`), then checked against the analytical
  prediction.  When the two disagree, one of the two is wrong, and it is
  usually the mental model.
* **peak memory** — the reason FSDP exists.
* **exposed communication** — from the profiler trace: the fraction of the step
  spent in collectives with no compute overlapping them.  This is the number
  that explains a disappointing scaling curve; total comm time does not.

``ddp-no-overlap`` is included deliberately as a control.  Comparing it against
``ddp`` isolates what backward/all-reduce overlap is actually worth on the
hardware in front of you, rather than assuming it is worth something.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass, field

import torch

from minidist import (
    DistributedDataParallel,
    FullyShardedDataParallel,
    ParallelMesh,
    analyze_trace,
    init_distributed,
    print_rank0,
    profile_steps,
    record_comm,
    shutdown_distributed,
)
from minidist.models import GPT, Block, GPTConfig

STRATEGIES = ["none", "ddp", "ddp-no-overlap", "fsdp", "fsdp-bf16"]


@dataclass
class Row:
    strategy: str
    ms_per_step: float = 0.0
    tokens_per_s: float = 0.0
    comm_bus_mb: float = 0.0
    peak_mem_mb: float = 0.0
    state_mb: float = 0.0
    exposed_comm_pct: float = float("nan")
    per_collective_mb: dict[str, float] = field(default_factory=dict)


def build(strategy: str, mesh: ParallelMesh, config: GPTConfig, device, bucket_cap_mb: float):
    model = GPT(config, tp_group=mesh.tp_group, device=device)
    if mesh.dp_size == 1 or strategy == "none":
        return model
    if strategy.startswith("ddp"):
        return DistributedDataParallel(
            model,
            process_group=mesh.dp_group,
            bucket_cap_mb=bucket_cap_mb,
            overlap_with_backward=(strategy == "ddp"),
        )
    from minidist.fsdp import MixedPrecisionPolicy

    param_dtype = torch.bfloat16 if strategy.endswith("bf16") else torch.float32
    return FullyShardedDataParallel(
        model,
        unit_types=(Block,),
        process_group=mesh.dp_group,
        mixed_precision=MixedPrecisionPolicy(
            param_dtype=param_dtype, reduce_dtype=torch.float32
        ),
    )


def analytical_comm_mb(strategy: str, config: GPTConfig, mesh: ParallelMesh) -> float:
    """What the strategy *should* move per step, per rank, in MB.

    DDP:  one all-reduce over all parameters      -> 2(W-1)/W * P * 4 bytes
    FSDP: all-gather fwd + all-gather bwd + reduce-scatter, each (W-1)/W * P
          -> 3 * (W-1)/W * P * 4 bytes = 1.5x DDP
    """
    world = mesh.dp_size
    if world == 1 or strategy == "none":
        return 0.0
    params = config.param_count() / max(1, mesh.tp_size)
    factor = (world - 1) / world
    bytes_per_param = 4.0
    if strategy.startswith("ddp"):
        volume = 2 * factor * params * bytes_per_param
    else:
        gather_bytes = 2.0 if strategy.endswith("bf16") else 4.0
        volume = factor * params * (2 * gather_bytes + 4.0)
    return volume / 1024**2


def measure(strategy: str, args, mesh: ParallelMesh, config: GPTConfig, device) -> Row:
    torch.manual_seed(args.seed)
    model = build(strategy, mesh, config, device, args.bucket_cap_mb)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    g = torch.Generator().manual_seed(args.seed + mesh.dp_rank)
    ids = torch.randint(0, config.vocab_size, (args.batch_size, config.block_size), generator=g).to(device)
    targets = torch.randint(0, config.vocab_size, (args.batch_size, config.block_size), generator=g).to(device)

    def step(_=0):
        optimizer.zero_grad(set_to_none=True)
        _, loss = model(ids, targets)
        loss.backward()
        optimizer.step()

    for _ in range(args.warmup):
        step()
    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    # One instrumented step for byte accounting; the counters add a few
    # microseconds each, so they stay out of the timed loop.
    with record_comm() as stats:
        step()
    per_collective = {
        name: bus / 1024**2 for name, _, _, bus in stats.as_rows()
    }
    comm_mb = stats.total_bus_bytes() / 1024**2

    start = time.perf_counter()
    for _ in range(args.iters):
        step()
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    row = Row(
        strategy=strategy,
        ms_per_step=elapsed / args.iters * 1e3,
        tokens_per_s=args.batch_size * config.block_size * mesh.dp_size * args.iters / elapsed,
        comm_bus_mb=comm_mb,
        per_collective_mb=per_collective,
    )
    if device.type == "cuda":
        row.peak_mem_mb = torch.cuda.max_memory_allocated() / 1024**2
    if isinstance(model, FullyShardedDataParallel):
        row.state_mb = model.memory_summary()["fsdp_state_MB"]
    else:
        params = sum(p.numel() for p in model.parameters())
        row.state_mb = params * 4 * 4 / 1024**2  # params + grads + Adam m,v

    if args.profile_dir:
        trace = f"{args.profile_dir}/trace_{strategy}_rank{mesh.rank}.json"
        profile_steps(step, num_steps=6, trace_path=trace)
        if mesh.rank == 0:
            report = analyze_trace(trace)
            row.exposed_comm_pct = report.exposed_fraction_of_step * 100

    return row


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tp", type=int, default=1)
    p.add_argument("--strategies", default=",".join(STRATEGIES), type=lambda s: s.split(","))
    p.add_argument("--layers", type=int, default=8)
    p.add_argument("--heads", type=int, default=8)
    p.add_argument("--embd", type=int, default=512)
    p.add_argument("--vocab", type=int, default=8192)
    p.add_argument("--seq-len", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--iters", type=int, default=10)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--bucket-cap-mb", type=float, default=25.0)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--backend", default=None, choices=[None, "nccl", "gloo"])
    p.add_argument("--profile-dir", default=None,
                   help="export per-strategy traces here and report exposed comm")
    args = p.parse_args()

    rank, world, device = init_distributed(backend=args.backend)
    mesh = ParallelMesh(tp_size=args.tp)
    config = GPTConfig(
        vocab_size=args.vocab, block_size=args.seq_len, n_layer=args.layers,
        n_head=args.heads, n_embd=args.embd, seed=args.seed,
    )

    print_rank0(f"\n{'=' * 92}")
    print_rank0(f"  {mesh.describe()}   device={device.type}")
    print_rank0(f"  model {config.n_layer}L x {config.n_embd}d, vocab {config.vocab_size}, "
                f"seq {config.block_size}, batch {args.batch_size}/replica "
                f"({config.param_count() / 1e6:.1f}M params)")
    print_rank0("=" * 92)
    header = (f"  {'strategy':<16}{'ms/step':>10}{'tok/s':>12}{'comm MB':>10}"
              f"{'predicted':>11}{'state MB':>10}{'peak MB':>10}{'exposed':>10}")
    print_rank0(header)
    print_rank0("  " + "-" * 88)

    rows = []
    for strategy in args.strategies:
        row = measure(strategy, args, mesh, config, device)
        rows.append(row)
        predicted = analytical_comm_mb(strategy, config, mesh)
        exposed = "-" if row.exposed_comm_pct != row.exposed_comm_pct else f"{row.exposed_comm_pct:.1f}%"
        print_rank0(
            f"  {strategy:<16}{row.ms_per_step:>10.2f}{row.tokens_per_s:>12.0f}"
            f"{row.comm_bus_mb:>10.1f}{predicted:>11.1f}{row.state_mb:>10.1f}"
            f"{row.peak_mem_mb:>10.1f}{exposed:>10}"
        )

    print_rank0("\n  per-collective bus traffic (MB/step/rank):")
    for row in rows:
        if row.per_collective_mb:
            detail = "  ".join(f"{k}={v:.1f}" for k, v in sorted(row.per_collective_mb.items()))
            print_rank0(f"    {row.strategy:<16}{detail}")

    baseline = next((r for r in rows if r.strategy == "ddp"), None)
    if baseline and len(rows) > 1:
        print_rank0("\n  relative to ddp:")
        for row in rows:
            if row is baseline:
                continue
            speed = baseline.ms_per_step / row.ms_per_step
            print_rank0(
                f"    {row.strategy:<16}{speed:>6.2f}x speed   "
                f"{row.comm_bus_mb / max(baseline.comm_bus_mb, 1e-9):>5.2f}x comm   "
                f"{row.state_mb / max(baseline.state_mb, 1e-9):>5.2f}x state"
            )

    shutdown_distributed()


if __name__ == "__main__":
    main()
