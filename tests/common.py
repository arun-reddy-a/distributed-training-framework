"""Multi-process test harness.

Every correctness claim in this repo is checked the same way: run the
distributed implementation across ``N`` spawned processes on the Gloo/CPU
backend, and compare against a reference computed in a single process.  Gloo on
CPU exercises the identical code path as NCCL on GPU — ``torch.distributed``
collectives, autograd interaction, hook ordering, and every place a rank could
disagree with its peers — so the tests run in CI on machines with no GPU.

What CPU testing does *not* cover: NCCL's own kernels, stream/event ordering
between the compute and communication streams, and anything about achieved
bandwidth.  Those need real hardware, and the benchmarks are where they live.
"""

from __future__ import annotations

import os
import tempfile
from collections.abc import Callable

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def _entrypoint(rank: int, world_size: int, store_file: str, fn: Callable, args: tuple):
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["LOCAL_RANK"] = str(rank)
    # A file store avoids the flaky-port problem that TCP init has when several
    # test cases spawn process groups back to back.
    dist.init_process_group(
        backend="gloo",
        init_method=f"file://{store_file}",
        rank=rank,
        world_size=world_size,
    )
    torch.manual_seed(0)
    try:
        fn(rank, world_size, *args)
    finally:
        dist.barrier()
        dist.destroy_process_group()


def run_distributed(fn: Callable, world_size: int = 2, *args) -> None:
    """Spawn ``world_size`` processes running ``fn(rank, world_size, *args)``.

    ``fn`` must be importable at module level (the spawn start method pickles
    it by qualified name).  Any exception in a worker — including a failed
    assertion — propagates out of ``mp.spawn`` and fails the test.
    """
    with tempfile.TemporaryDirectory() as tmp:
        store_file = os.path.join(tmp, "store")
        mp.spawn(
            _entrypoint,
            args=(world_size, store_file, fn, args),
            nprocs=world_size,
            join=True,
        )


def solo_group(rank: int, world_size: int):
    """A process group containing only this rank.

    Lets a worker build an unsharded reference model (``tp_size == 1``) while
    still inside a multi-rank job.  ``new_group`` is collective, so every rank
    walks the full loop and keeps only its own.
    """
    mine = None
    for r in range(world_size):
        group = dist.new_group([r])
        if r == rank:
            mine = group
    return mine


def assert_close(actual: torch.Tensor, expected: torch.Tensor, msg: str, tol: float = 1e-5):
    actual, expected = actual.detach().float(), expected.detach().float()
    if actual.shape != expected.shape:
        raise AssertionError(f"{msg}: shape {tuple(actual.shape)} != {tuple(expected.shape)}")
    diff = (actual - expected).abs()
    denom = expected.abs().max().clamp(min=1.0)
    rel = (diff.max() / denom).item()
    if rel > tol:
        raise AssertionError(
            f"{msg}: max abs diff {diff.max().item():.3e}, relative {rel:.3e} > {tol:.1e}"
        )


def all_ranks_agree(tensor: torch.Tensor, group=None) -> bool:
    """True if every rank in ``group`` holds a bit-comparable tensor."""
    gathered = [torch.empty_like(tensor) for _ in range(dist.get_world_size(group))]
    dist.all_gather(gathered, tensor.contiguous(), group=group)
    return all(torch.equal(gathered[0], g) for g in gathered[1:])
