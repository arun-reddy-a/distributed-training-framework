#!/usr/bin/env python3
"""Train a GPT under any combination of the four parallelism strategies.

One entry point, because the point of a mesh is that the strategies are
independent axes rather than mutually exclusive modes:

.. code-block:: bash

    # 8 GPUs, pure data parallel
    torchrun --nproc_per_node=8 examples/train_gpt.py --strategy ddp

    # 8 GPUs, ZeRO-3 with bf16 parameters
    torchrun --nproc_per_node=8 examples/train_gpt.py --strategy fsdp --precision bf16

    # 8 GPUs as 2-way tensor x 2-way pipeline x 2-way data parallel
    torchrun --nproc_per_node=8 examples/train_gpt.py \\
        --tp 2 --pp 2 --strategy ddp --microbatches 8 --schedule 1f1b

    # No GPUs at all — Gloo on CPU, for checking the wiring
    torchrun --nproc_per_node=4 examples/train_gpt.py --tp 2 --backend gloo

The data is synthetic random tokens.  This trains nothing useful; it exists to
exercise and measure the parallelism, and the loss should hover near
``ln(vocab_size)`` because random tokens are unpredictable by construction.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from torch import nn

from minidist import (
    DistributedDataParallel,
    DistributedGradScaler,
    FullyShardedDataParallel,
    ParallelMesh,
    PipelineStage,
    analyze_trace,
    barrier,
    build_schedule,
    format_collective_table,
    init_distributed,
    partition_layers,
    print_rank0,
    profile_steps,
    resolve_precision,
    shutdown_distributed,
    summarize_collectives,
)
from minidist.fsdp import MixedPrecisionPolicy
from minidist.models import GPT, Block, GPTConfig, build_pipeline_layers
from minidist.tensor_parallel import vocab_parallel_cross_entropy


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mesh = p.add_argument_group("mesh")
    mesh.add_argument("--tp", type=int, default=1, help="tensor-parallel degree")
    mesh.add_argument("--pp", type=int, default=1, help="pipeline-parallel degree")
    mesh.add_argument("--backend", default=None, choices=[None, "nccl", "gloo"])

    strat = p.add_argument_group("strategy")
    strat.add_argument("--strategy", default="ddp", choices=["none", "ddp", "fsdp"],
                       help="what to do along the data-parallel axis")
    strat.add_argument("--schedule", default="1f1b", choices=["gpipe", "1f1b"])
    strat.add_argument("--microbatches", type=int, default=4)
    strat.add_argument("--precision", default="fp32", choices=["fp32", "fp16", "bf16", "auto"])
    strat.add_argument("--fsdp-param-dtype", default=None, choices=[None, "bf16", "fp16"],
                       help="gather FSDP parameters in this dtype (independent of --precision)")
    strat.add_argument("--bucket-cap-mb", type=float, default=25.0)
    strat.add_argument("--no-overlap", action="store_true",
                       help="disable DDP backward/all-reduce overlap (for comparison)")

    model = p.add_argument_group("model")
    model.add_argument("--layers", type=int, default=8)
    model.add_argument("--heads", type=int, default=8)
    model.add_argument("--embd", type=int, default=512)
    model.add_argument("--vocab", type=int, default=8192)
    model.add_argument("--seq-len", type=int, default=256)
    model.add_argument("--batch-size", type=int, default=16, help="per data-parallel replica")

    run = p.add_argument_group("run")
    run.add_argument("--steps", type=int, default=20)
    run.add_argument("--lr", type=float, default=3e-4)
    run.add_argument("--seed", type=int, default=1234)
    run.add_argument("--profile", metavar="PATH", default=None,
                     help="export a Chrome trace here and print an overlap report")
    return p.parse_args()


def make_batch(config: GPTConfig, batch_size: int, dp_rank: int, step: int, seed: int, device):
    """Synthetic tokens.

    The generator is seeded from ``dp_rank`` and the step, and deliberately not
    from the tensor- or pipeline-parallel rank: TP and PP peers are different
    pieces of *one* model replica and must see identical data, while DP
    replicas must see different data or the run is just a smaller batch
    computed several times.
    """
    g = torch.Generator().manual_seed(seed + step * 10_000 + dp_rank)
    ids = torch.randint(0, config.vocab_size, (batch_size, config.block_size), generator=g)
    targets = torch.randint(0, config.vocab_size, (batch_size, config.block_size), generator=g)
    return ids.to(device), targets.to(device)


def _broadcast_loss(loss, mesh: ParallelMesh, device) -> float:
    """Share the last stage's loss with every stage, for logging only."""
    value = torch.zeros(1, device=device)
    if loss is not None:
        value[0] = loss.sum()
    if mesh.pp_size > 1:
        torch.distributed.broadcast(value, src=mesh.pp_ranks[-1], group=mesh.pp_group)
    return float(value.item())


def build(args, mesh: ParallelMesh, config: GPTConfig, device):
    """Assemble the model for this rank's position in the mesh."""
    loss_fn = None
    if mesh.pp_size > 1:
        layers = build_pipeline_layers(config, tp_group=mesh.tp_group, device=device)
        mine = partition_layers(layers, mesh.pp_size, mesh.pp_rank)
        core: nn.Module = nn.Sequential(*mine)
        vocab_start = mesh.tp_rank * (config.vocab_size // mesh.tp_size)

        def loss_fn(logits, targets):
            return vocab_parallel_cross_entropy(
                logits, targets, group=mesh.tp_group, vocab_start=vocab_start
            )
    else:
        core = GPT(config, tp_group=mesh.tp_group, device=device)

    wrapped = core
    if mesh.dp_size > 1 and args.strategy == "ddp":
        wrapped = DistributedDataParallel(
            core,
            process_group=mesh.dp_group,
            bucket_cap_mb=args.bucket_cap_mb,
            overlap_with_backward=not args.no_overlap,
        )
    elif mesh.dp_size > 1 and args.strategy == "fsdp":
        param_dtype = {
            None: torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16
        }[args.fsdp_param_dtype]
        wrapped = FullyShardedDataParallel(
            core,
            unit_types=(Block,),
            process_group=mesh.dp_group,
            mixed_precision=MixedPrecisionPolicy(
                param_dtype=param_dtype, reduce_dtype=torch.float32
            ),
        )
    return wrapped, core, loss_fn


def main() -> None:
    args = parse_args()
    rank, world, device = init_distributed(backend=args.backend)
    torch.manual_seed(args.seed)

    mesh = ParallelMesh(tp_size=args.tp, pp_size=args.pp)
    config = GPTConfig(
        vocab_size=args.vocab, block_size=args.seq_len, n_layer=args.layers,
        n_head=args.heads, n_embd=args.embd, seed=args.seed,
    )
    precision = resolve_precision(args.precision, device)

    print_rank0(f"\n{'=' * 74}")
    print_rank0(f"  {mesh.describe()}")
    print_rank0(
        f"  model: {config.n_layer}L x {config.n_embd}d x {config.n_head}h, "
        f"vocab {config.vocab_size}, seq {config.block_size} "
        f"({config.param_count() / 1e6:.1f}M params)"
    )
    print_rank0(
        f"  strategy: {args.strategy}  precision: {precision}  "
        + (f"pipeline: {args.schedule} x{args.microbatches}" if mesh.pp_size > 1 else "")
    )
    print_rank0("=" * 74)

    model, core, loss_fn = build(args, mesh, config, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scaler = DistributedGradScaler(enabled=precision.needs_grad_scaler)

    schedule = None
    if mesh.pp_size > 1:
        stage = PipelineStage(model, mesh, loss_fn=loss_fn if mesh.is_last_stage else None)
        schedule = build_schedule(args.schedule, stage, mesh, args.microbatches)

    if isinstance(model, FullyShardedDataParallel):
        summary = model.memory_summary()
        print_rank0(
            f"  FSDP: {summary['num_units']} units, "
            f"{summary['shard_params_MB']:.1f}MB/rank sharded vs "
            f"{summary['replicated_params_MB']:.1f}MB replicated; "
            f"transient gather {summary['transient_gather_MB']:.1f}MB"
        )

    def train_step(step: int) -> float:
        ids, targets = make_batch(config, args.batch_size, mesh.dp_rank, step, args.seed, device)
        optimizer.zero_grad(set_to_none=True)

        if schedule is not None:
            schedule.loss_scale = scaler.scale
            # Every microbatch triggers a backward; without `no_sync` each one
            # would launch its own gradient all-reduce instead of one
            # reduction over the whole accumulated gradient.
            sync_ctx = model.no_sync() if hasattr(model, "no_sync") else _null()
            with sync_ctx, precision.autocast():
                loss = schedule.step(inputs=ids, targets=targets)
            if hasattr(model, "finish_gradient_sync"):
                model.finish_gradient_sync()
            # Only the last stage computes the loss, but every stage wants to
            # log it, so hand it back down the pipeline.
            value = _broadcast_loss(loss, mesh, device)
        else:
            with precision.autocast():
                _, loss = model(ids, targets)
            scaler.scale_loss(loss).backward()
            value = float(loss.detach())

        scaler.step(optimizer)
        scaler.update()
        return value

    # Warm up outside the timing loop: the first step pays for NCCL
    # communicator setup, allocator growth and cuDNN autotuning.
    train_step(0)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    tokens_per_step = args.batch_size * config.block_size * mesh.dp_size
    start = time.perf_counter()
    for step in range(1, args.steps + 1):
        loss = train_step(step)
        if step % max(1, args.steps // 10) == 0:
            elapsed = time.perf_counter() - start
            print_rank0(
                f"  step {step:>4}/{args.steps}  loss {loss:>7.4f}  "
                f"{elapsed / step * 1e3:>7.1f} ms/step  "
                f"{tokens_per_step * step / elapsed:>9.0f} tok/s"
            )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    total = time.perf_counter() - start

    print_rank0(f"\n  {args.steps} steps in {total:.2f}s "
                f"({total / args.steps * 1e3:.1f} ms/step, "
                f"{tokens_per_step * args.steps / total:.0f} tok/s)")
    if torch.cuda.is_available():
        print_rank0(f"  peak memory: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GiB/rank")
    if scaler.enabled:
        print_rank0(f"  {scaler.summary()}")
    if schedule is not None:
        report = schedule.report()
        print_rank0(
            f"  pipeline bubble: measured {report['measured_bubble'] * 100:.1f}%  "
            f"theoretical {report['theoretical_bubble'] * 100:.1f}%  "
            f"(stage {mesh.pp_rank})"
        )

    if args.profile:
        # One trace per rank. A trace is a per-process artifact, and every rank
        # writing the same path races on the temp-file rename that
        # `export_chrome_trace` does internally — leaving a truncated file, or
        # none at all.
        trace = Path(args.profile)
        trace = trace.with_name(f"{trace.stem}_rank{mesh.rank}{trace.suffix or '.json'}")
        print_rank0(f"\n  profiling -> {trace.parent}/{trace.stem[:-6]}_rank*.json")
        prof = profile_steps(lambda s: train_step(s + 1000), num_steps=6, trace_path=trace)
        print_rank0("\n  collectives:")
        print_rank0(format_collective_table(summarize_collectives(prof)))
        barrier()
        if mesh.rank == 0:
            print_rank0("\n  overlap:")
            print_rank0(analyze_trace(trace).format())

    shutdown_distributed()


class _null:
    def __enter__(self):
        return None

    def __exit__(self, *exc):
        return False


if __name__ == "__main__":
    main()
