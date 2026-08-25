"""Process-group bootstrap and instrumented collective wrappers.

Everything in ``minidist`` talks to NCCL/Gloo through this module rather than
calling ``torch.distributed`` directly.  The indirection buys two things:

1. **A single place where backend/device selection happens.**  The rest of the
   framework never asks "am I on CUDA?"; it asks the mesh for a device.
2. **Byte accounting.**  Every collective records how many bytes it moved and
   how long it took, so the benchmarks can report *measured* communication
   volume next to the analytical prediction instead of only trusting the model.
   The counters are a few adds per collective and are disabled by default in
   the training loop.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import timedelta

import torch
import torch.distributed as dist

__all__ = [
    "CommStats",
    "comm_stats",
    "record_comm",
    "init_distributed",
    "shutdown_distributed",
    "is_distributed",
    "get_rank",
    "get_world_size",
    "is_primary",
    "barrier",
    "all_reduce",
    "all_gather_into_tensor",
    "reduce_scatter_tensor",
    "broadcast",
    "print_rank0",
]


# --------------------------------------------------------------------------
# Byte accounting
# --------------------------------------------------------------------------


@dataclass
class _OpStat:
    count: int = 0
    payload_bytes: int = 0
    bus_bytes: int = 0


@dataclass
class CommStats:
    """Per-collective byte counters.

    ``payload_bytes`` is the size of the caller's tensor.  ``bus_bytes`` is the
    volume that actually crosses the interconnect, which is what you divide by
    time to get a bandwidth number comparable to ``nccl-tests``:

    ==================  ====================================
    collective          bus bytes for an N-element payload
    ==================  ====================================
    all-reduce          ``2 * (W - 1) / W * N``
    all-gather          ``(W - 1) / W * N_out``
    reduce-scatter      ``(W - 1) / W * N_in``
    broadcast           ``N``
    ==================  ====================================

    The ring-algorithm factors come from each rank sending its data around the
    ring once (all-gather / reduce-scatter) or twice (all-reduce = a
    reduce-scatter followed by an all-gather).
    """

    enabled: bool = False
    ops: dict[str, _OpStat] = field(default_factory=dict)

    def add(self, op: str, payload_bytes: int, bus_bytes: int) -> None:
        if not self.enabled:
            return
        stat = self.ops.setdefault(op, _OpStat())
        stat.count += 1
        stat.payload_bytes += payload_bytes
        stat.bus_bytes += bus_bytes

    def reset(self) -> None:
        self.ops.clear()

    def total_bus_bytes(self) -> int:
        return sum(s.bus_bytes for s in self.ops.values())

    def as_rows(self) -> list[tuple[str, int, int, int]]:
        return sorted(
            (name, s.count, s.payload_bytes, s.bus_bytes) for name, s in self.ops.items()
        )


comm_stats = CommStats()


@contextmanager
def record_comm() -> Iterator[CommStats]:
    """Enable byte accounting for the duration of the block."""
    previous, comm_stats.enabled = comm_stats.enabled, True
    comm_stats.reset()
    try:
        yield comm_stats
    finally:
        comm_stats.enabled = previous


# --------------------------------------------------------------------------
# Bootstrap
# --------------------------------------------------------------------------


def _default_backend() -> str:
    if torch.cuda.is_available() and dist.is_nccl_available():
        return "nccl"
    return "gloo"


def init_distributed(
    backend: str | None = None,
    timeout_seconds: int = 1800,
) -> tuple[int, int, torch.device]:
    """Initialise the default process group from ``torchrun``'s environment.

    Returns ``(rank, world_size, device)``.  Safe to call when launched without
    ``torchrun``: it falls back to a single-rank world so every script in this
    repo also runs as a plain ``python foo.py``.
    """
    if dist.is_initialized():
        return get_rank(), get_world_size(), current_device()

    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    backend = backend or _default_backend()

    # Bind the process to its GPU *before* initialising NCCL.  NCCL derives its
    # communicator topology from the current device; leaving every rank on
    # cuda:0 is the single most common cause of a hang at the first collective.
    if backend == "nccl":
        if not torch.cuda.is_available():
            raise RuntimeError("backend='nccl' requested but CUDA is unavailable")
        torch.cuda.set_device(local_rank)

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")

    dist.init_process_group(
        backend=backend,
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=timeout_seconds),
    )
    return rank, world_size, current_device()


def shutdown_distributed() -> None:
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def current_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda", torch.cuda.current_device())
    return torch.device("cpu")


def is_distributed() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_rank(group: dist.ProcessGroup | None = None) -> int:
    return dist.get_rank(group) if is_distributed() else 0


def get_world_size(group: dist.ProcessGroup | None = None) -> int:
    return dist.get_world_size(group) if is_distributed() else 1


def is_primary() -> bool:
    return get_rank() == 0


def barrier(group: dist.ProcessGroup | None = None) -> None:
    if is_distributed():
        dist.barrier(group)


def print_rank0(*args, **kwargs) -> None:
    if is_primary():
        print(*args, **kwargs, flush=True)


# --------------------------------------------------------------------------
# Instrumented collectives
# --------------------------------------------------------------------------


def _nbytes(t: torch.Tensor) -> int:
    return t.numel() * t.element_size()


# torch 2.9 renamed the single-tensor variants; keep working on both. These are
# the fused forms that take one contiguous buffer instead of a Python list of
# tensors — worth insisting on, because the list forms copy every shard into a
# temporary before the collective even starts.
_all_gather_single = getattr(dist, "all_gather_single", None) or dist.all_gather_into_tensor
_reduce_scatter_single = getattr(dist, "reduce_scatter_single", None) or dist.reduce_scatter_tensor


def all_reduce(
    tensor: torch.Tensor,
    op: dist.ReduceOp = dist.ReduceOp.SUM,
    group: dist.ProcessGroup | None = None,
    async_op: bool = False,
):
    world = get_world_size(group)
    if world == 1:
        return None
    payload = _nbytes(tensor)
    comm_stats.add("all_reduce", payload, int(2 * (world - 1) / world * payload))
    return dist.all_reduce(tensor, op=op, group=group, async_op=async_op)


def all_gather_into_tensor(
    output: torch.Tensor,
    input_: torch.Tensor,
    group: dist.ProcessGroup | None = None,
    async_op: bool = False,
):
    world = get_world_size(group)
    if world == 1:
        output.copy_(input_)
        return None
    payload = _nbytes(output)
    comm_stats.add("all_gather", payload, int((world - 1) / world * payload))
    return _all_gather_single(output, input_, group=group, async_op=async_op)


def reduce_scatter_tensor(
    output: torch.Tensor,
    input_: torch.Tensor,
    op: dist.ReduceOp = dist.ReduceOp.SUM,
    group: dist.ProcessGroup | None = None,
    async_op: bool = False,
):
    world = get_world_size(group)
    if world == 1:
        output.copy_(input_)
        return None
    payload = _nbytes(input_)
    comm_stats.add("reduce_scatter", payload, int((world - 1) / world * payload))
    return _reduce_scatter_single(output, input_, op=op, group=group, async_op=async_op)


def broadcast(
    tensor: torch.Tensor,
    src: int,
    group: dist.ProcessGroup | None = None,
    async_op: bool = False,
):
    if get_world_size(group) == 1:
        return None
    comm_stats.add("broadcast", _nbytes(tensor), _nbytes(tensor))
    return dist.broadcast(tensor, src=src, group=group, async_op=async_op)
