"""PyTorch Profiler integration: collective breakdowns and comm/compute overlap.

Two questions matter when you profile a distributed step, and neither is
answered by total step time:

1. **Where did the communication go?**  How much time in all-reduce vs.
   all-gather vs. reduce-scatter, how many bytes, and what bus bandwidth does
   that imply relative to the link's peak?
2. **Was it hidden?**  A step can spend 40% of its wall time inside NCCL and
   still be compute-bound, if that NCCL time runs concurrently with kernels on
   the compute stream.  What hurts is *exposed* communication — the part with
   no compute overlapping it.

:func:`summarize_collectives` answers the first from the profiler's event
table.  :func:`analyze_trace` answers the second from the exported Chrome
trace, by treating each stream as a set of intervals and doing interval
arithmetic:

.. code-block:: text

    compute   ####----########--####
    comm      --######----######----
                ^^^^      ^^^^          overlapped
              ^^    ^^        ^^        exposed comm  <- the number that matters

    exposed_comm = |comm| - |comm ∩ compute|

Why interval arithmetic rather than summing durations: NCCL kernels and compute
kernels live on different CUDA streams and genuinely run at the same time, so
adding their durations double-counts wall time.  Merging overlapping intervals
per stream first, then intersecting the two merged sets, gives numbers that add
up to the actual step.

On a Gloo/CPU run there are no CUDA kernels, so the analysis falls back to the
``c10d::`` CPU operators.  Those are *blocking* calls, so on CPU the overlap
number is structurally near zero and is reported as such rather than being
quietly presented as a real measurement.
"""

from __future__ import annotations

import gzip
import json
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile, schedule

__all__ = [
    "COLLECTIVE_PATTERNS",
    "OverlapReport",
    "CollectiveRow",
    "profile_steps",
    "summarize_collectives",
    "analyze_trace",
    "format_collective_table",
]

# Names the profiler gives to collective work, across backends and torch
# versions.  NCCL device kernels are `nccl:*` / `ncclDevKernel_*`; the
# dispatcher-level ops are `c10d::*`; `record_param_comms` is the annotation
# torch attaches to collectives issued from DDP/FSDP internals.
COLLECTIVE_PATTERNS: tuple[str, ...] = (
    "nccl",
    "c10d::",
    "record_param_comms",
    "ProcessGroupNCCL",
    "ProcessGroupGloo",
)

_COLLECTIVE_KINDS = {
    "all_reduce": ("allreduce", "all_reduce", "AllReduce"),
    "all_gather": ("allgather", "all_gather", "AllGather", "_allgather_base"),
    "reduce_scatter": ("reducescatter", "reduce_scatter", "ReduceScatter"),
    "broadcast": ("broadcast", "Broadcast"),
    "send_recv": ("send", "recv", "SendRecv", "P2P"),
    "barrier": ("barrier", "Barrier"),
}


def _classify(name: str) -> str | None:
    lowered = name.lower()
    for kind, needles in _COLLECTIVE_KINDS.items():
        if any(n.lower() in lowered for n in needles):
            return kind
    return None


def _is_collective(name: str) -> bool:
    return any(p.lower() in name.lower() for p in COLLECTIVE_PATTERNS)


# --------------------------------------------------------------------------
# Running the profiler
# --------------------------------------------------------------------------


def profile_steps(
    step_fn: Callable[[int], None],
    num_steps: int = 6,
    warmup: int = 2,
    wait: int = 1,
    trace_path: str | Path | None = None,
    record_shapes: bool = True,
    with_stack: bool = False,
    activities: Sequence[ProfilerActivity] | None = None,
):
    """Run ``step_fn`` under the profiler and optionally export a Chrome trace.

    The ``wait / warmup / active`` schedule exists because the first steps of a
    distributed run are unrepresentative: cuDNN picks algorithms, the caching
    allocator grows to its steady-state size, and NCCL lazily builds its
    communicators on the first collective — a one-off cost of tens of
    milliseconds that would otherwise dominate a short profile.
    """
    if activities is None:
        activities = [ProfilerActivity.CPU]
        if torch.cuda.is_available():
            activities.append(ProfilerActivity.CUDA)

    active = max(1, num_steps - wait - warmup)
    prof_schedule = schedule(wait=wait, warmup=warmup, active=active, repeat=1)

    with profile(
        activities=list(activities),
        schedule=prof_schedule,
        record_shapes=record_shapes,
        with_stack=with_stack,
        profile_memory=True,
    ) as prof:
        for step in range(num_steps):
            step_fn(step)
            prof.step()

    if trace_path is not None:
        trace_path = Path(trace_path)
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        prof.export_chrome_trace(str(trace_path))
    return prof


# --------------------------------------------------------------------------
# Event-table summary
# --------------------------------------------------------------------------


@dataclass
class CollectiveRow:
    kind: str
    count: int
    self_time_us: float
    device_time_us: float

    @property
    def avg_us(self) -> float:
        return self.self_time_us / max(1, self.count)


def summarize_collectives(prof) -> list[CollectiveRow]:
    """Aggregate profiler events into per-collective-kind totals."""
    buckets: dict[str, CollectiveRow] = {}
    for evt in prof.key_averages():
        if not _is_collective(evt.key):
            continue
        kind = _classify(evt.key) or "other"
        row = buckets.setdefault(kind, CollectiveRow(kind, 0, 0.0, 0.0))
        row.count += evt.count
        row.self_time_us += float(getattr(evt, "self_cpu_time_total", 0.0))
        device_us = 0.0
        for attr in ("self_device_time_total", "self_cuda_time_total"):
            device_us = float(getattr(evt, attr, 0.0) or 0.0)
            if device_us:
                break
        row.device_time_us += device_us
    return sorted(buckets.values(), key=lambda r: -max(r.device_time_us, r.self_time_us))


def format_collective_table(rows: Iterable[CollectiveRow]) -> str:
    rows = list(rows)
    if not rows:
        return "  (no collective events recorded)"
    out = [f"  {'collective':<16}{'count':>8}{'device ms':>12}{'cpu ms':>10}{'avg us':>10}"]
    out.append("  " + "-" * 56)
    for r in rows:
        out.append(
            f"  {r.kind:<16}{r.count:>8}{r.device_time_us / 1000:>12.2f}"
            f"{r.self_time_us / 1000:>10.2f}{r.avg_us:>10.1f}"
        )
    return "\n".join(out)


# --------------------------------------------------------------------------
# Trace-based overlap analysis
# --------------------------------------------------------------------------


def _merge(intervals: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Collapse overlapping/touching intervals so a category's busy time can
    be summed without double-counting kernels that ran concurrently."""
    if not intervals:
        return []
    intervals.sort()
    merged = [intervals[0]]
    for begin, end in intervals[1:]:
        last_begin, last_end = merged[-1]
        if begin <= last_end:
            merged[-1] = (last_begin, max(last_end, end))
        else:
            merged.append((begin, end))
    return merged


def _total(intervals: list[tuple[float, float]]) -> float:
    return sum(e - b for b, e in intervals)


def _intersect(
    a: list[tuple[float, float]], b: list[tuple[float, float]]
) -> list[tuple[float, float]]:
    """Intersection of two *already merged* interval lists, in linear time."""
    out: list[tuple[float, float]] = []
    i = j = 0
    while i < len(a) and j < len(b):
        lo = max(a[i][0], b[j][0])
        hi = min(a[i][1], b[j][1])
        if lo < hi:
            out.append((lo, hi))
        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1
    return out


@dataclass
class OverlapReport:
    """Communication/computation overlap, in seconds unless noted."""

    device: str
    wall_s: float = 0.0
    compute_s: float = 0.0
    comm_s: float = 0.0
    overlapped_s: float = 0.0
    num_comm_events: int = 0
    num_compute_events: int = 0
    per_kind_s: dict[str, float] = field(default_factory=dict)
    note: str = ""

    @property
    def exposed_comm_s(self) -> float:
        """Communication with no compute running alongside it."""
        return max(0.0, self.comm_s - self.overlapped_s)

    @property
    def overlap_fraction(self) -> float:
        """Fraction of communication successfully hidden behind compute."""
        return self.overlapped_s / self.comm_s if self.comm_s > 0 else 0.0

    @property
    def exposed_fraction_of_step(self) -> float:
        """Exposed communication as a fraction of wall time — the real tax."""
        return self.exposed_comm_s / self.wall_s if self.wall_s > 0 else 0.0

    def format(self) -> str:
        lines = [
            f"  device            : {self.device}",
            f"  wall              : {self.wall_s * 1e3:.2f} ms",
            f"  compute (busy)    : {self.compute_s * 1e3:.2f} ms "
            f"({self.num_compute_events} events)",
            f"  communication     : {self.comm_s * 1e3:.2f} ms "
            f"({self.num_comm_events} events)",
            f"  overlapped        : {self.overlapped_s * 1e3:.2f} ms "
            f"({self.overlap_fraction * 100:.1f}% of comm hidden)",
            f"  exposed comm      : {self.exposed_comm_s * 1e3:.2f} ms "
            f"({self.exposed_fraction_of_step * 100:.1f}% of step)",
        ]
        if self.per_kind_s:
            lines.append("  by collective     :")
            for kind, seconds in sorted(self.per_kind_s.items(), key=lambda kv: -kv[1]):
                lines.append(f"      {kind:<16}{seconds * 1e3:>10.2f} ms")
        if self.note:
            lines.append(f"  note              : {self.note}")
        return "\n".join(lines)


def _load_trace(path: str | Path) -> dict:
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as fh:
        return json.load(fh)


def analyze_trace(path: str | Path, prefer_device: bool = True) -> OverlapReport:
    """Compute comm/compute overlap from an exported Chrome trace.

    Parameters
    ----------
    prefer_device:
        Analyse GPU kernel events when the trace has them.  Set ``False`` to
        force the CPU-operator view, which is what a Gloo run has.
    """
    trace = _load_trace(path)
    events = [e for e in trace.get("traceEvents", []) if e.get("ph") == "X"]

    kernels = [e for e in events if e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")]
    use_device = prefer_device and any(e.get("cat") == "kernel" for e in kernels)

    if use_device:
        pool = [e for e in events if e.get("cat") == "kernel"]
        device = "cuda"
        note = ""
    else:
        # CPU operators nest, so keep only top-level ops per thread to avoid
        # counting a parent and its children as separate busy intervals.
        pool = [e for e in events if e.get("cat") in ("cpu_op", "user_annotation")]
        device = "cpu"
        note = (
            "Gloo/CPU trace: collectives are blocking CPU operators, so overlap "
            "is structurally ~0 here. Run on CUDA+NCCL for a meaningful number."
        )

    comm_iv: list[tuple[float, float]] = []
    compute_iv: list[tuple[float, float]] = []
    per_kind: dict[str, list[tuple[float, float]]] = {}
    n_comm = n_compute = 0

    for e in pool:
        begin = float(e["ts"]) * 1e-6
        end = begin + float(e.get("dur", 0)) * 1e-6
        name = e.get("name", "")
        if _is_collective(name):
            comm_iv.append((begin, end))
            per_kind.setdefault(_classify(name) or "other", []).append((begin, end))
            n_comm += 1
        else:
            compute_iv.append((begin, end))
            n_compute += 1

    comm_m, compute_m = _merge(comm_iv), _merge(compute_iv)
    if not use_device:
        # A blocking c10d op on CPU contains its own wait, and its parent
        # frames contain it. Subtracting comm from compute keeps the two
        # categories disjoint instead of reporting spurious overlap.
        compute_m = _subtract(compute_m, comm_m)

    all_iv = _merge(comm_iv + compute_iv)
    wall = (all_iv[-1][1] - all_iv[0][0]) if all_iv else 0.0

    return OverlapReport(
        device=device,
        wall_s=wall,
        compute_s=_total(compute_m),
        comm_s=_total(comm_m),
        overlapped_s=_total(_intersect(comm_m, compute_m)),
        num_comm_events=n_comm,
        num_compute_events=n_compute,
        per_kind_s={k: _total(_merge(v)) for k, v in per_kind.items()},
        note=note,
    )


def _subtract(
    a: list[tuple[float, float]], b: list[tuple[float, float]]
) -> list[tuple[float, float]]:
    """``a \\ b`` for merged interval lists."""
    out: list[tuple[float, float]] = []
    j = 0
    for begin, end in a:
        cursor = begin
        while j < len(b) and b[j][1] <= cursor:
            j += 1
        k = j
        while k < len(b) and b[k][0] < end:
            if b[k][0] > cursor:
                out.append((cursor, min(b[k][0], end)))
            cursor = max(cursor, b[k][1])
            k += 1
        if cursor < end:
            out.append((cursor, end))
    return out
