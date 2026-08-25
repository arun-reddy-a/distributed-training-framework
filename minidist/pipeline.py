r"""Pipeline (inter-layer) parallelism: GPipe and 1F1B schedules.

Pipeline parallelism splits the model *depth-wise*: stage 0 owns layers 0..k,
stage 1 owns k+1..2k, and activations flow forward while gradients flow back.
Communication is point-to-point and tiny — one ``[batch, seq, hidden]`` tensor
per stage boundary — which is why pipelining is the strategy that survives slow
inter-node links.

The cost is the **bubble**: while stage 0 computes microbatch 0, stages 1..P-1
have nothing to do.  Splitting the batch into ``M`` microbatches amortises it,
because stage 0 can start microbatch 1 as soon as it has handed off microbatch 0:

.. code-block:: text

    P = 4 stages, M = 4 microbatches (GPipe)

    stage 0  F0 F1 F2 F3  ·  ·  ·  B3 B2 B1 B0
    stage 1   · F0 F1 F2 F3  ·  B3 B2 B1 B0  ·
    stage 2   ·  · F0 F1 F2 F3 B3 B2 B1 B0 ·  ·
    stage 3   ·  ·  · F0 F1 F2 F3 B3 B2 B1 ·  ·
              \___ fill ___/       \_ drain _/

    bubble fraction = (P - 1) / (M + P - 1) = 3/7 ≈ 43%

Raising ``M`` is the only lever that shrinks the bubble, and it has a floor:
microbatches must stay large enough to keep the GPU's matmuls efficient.

GPipe vs. 1F1B — what actually differs
--------------------------------------
A widespread misreading is that 1F1B removes the bubble.  It does not: both
schedules execute ``M`` forwards and ``M`` backwards per stage with the same
``P-1`` steps of fill and drain, so **both have bubble fraction
``(P-1)/(M+P-1)``**.  What 1F1B changes is *peak activation memory*.

GPipe runs all ``M`` forwards before the first backward, so stage 0 holds all
``M`` microbatches' activations at once.  1F1B starts backpropagating microbatch
0 as soon as it can, so a stage holds at most ``P - stage_rank`` in flight:

.. code-block:: text

    activations in flight, stage 0:   GPipe  M      1F1B  P

Since ``M`` must be *large* to shrink the bubble and ``P`` is fixed and small,
that is the difference between "raise M until memory explodes" and "raise M
freely".  1F1B is what makes a low bubble affordable, not what makes it low.
(Driving the bubble itself below ``(P-1)/(M+P-1)`` needs *interleaved* 1F1B,
where each rank owns ``v`` non-contiguous stages — out of scope here, but the
measured timelines this module emits are what you would use to evaluate it.)

Every stage records a timeline of forward/backward/communication intervals, so
the benchmarks report a *measured* bubble fraction next to the analytical one
instead of assuming the model holds.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import torch
import torch.distributed as dist
from torch import nn

from .mesh import ParallelMesh

__all__ = [
    "PipelineStage",
    "GPipeSchedule",
    "OneForwardOneBackwardSchedule",
    "build_schedule",
    "partition_layers",
    "StageTimeline",
    "theoretical_bubble_fraction",
]

_DTYPES = [torch.float32, torch.float16, torch.bfloat16, torch.float64]
_HEADER_LEN = 10  # [ndim, d0..d7, dtype_index]


def theoretical_bubble_fraction(num_stages: int, num_microbatches: int) -> float:
    """``(P-1)/(M+P-1)`` — identical for GPipe and (non-interleaved) 1F1B."""
    if num_stages <= 1:
        return 0.0
    return (num_stages - 1) / (num_microbatches + num_stages - 1)


def partition_layers(layers: Sequence[nn.Module], num_stages: int, stage: int) -> list[nn.Module]:
    """Contiguous, as-even-as-possible split of ``layers`` across stages.

    Uniform partitioning assumes uniform layer cost, which is true for a stack
    of identical transformer blocks and false once embeddings and the LM head
    enter the picture — those are attached to the first and last stage
    respectively and make them heavier.  Real systems profile per-layer time
    and partition to balance it; the imbalance shows up directly in the
    per-stage busy times this module records.
    """
    n = len(layers)
    if num_stages > n:
        raise ValueError(f"{num_stages} stages requested for only {n} layers")
    base, extra = divmod(n, num_stages)
    start = stage * base + min(stage, extra)
    end = start + base + (1 if stage < extra else 0)
    return list(layers[start:end])


# --------------------------------------------------------------------------
# Timeline instrumentation
# --------------------------------------------------------------------------


@dataclass
class StageTimeline:
    """Per-stage record of what happened when, in seconds since ``start()``."""

    events: list[tuple[str, int, float, float]] = field(default_factory=list)
    t0: float = 0.0
    t_end: float = 0.0

    def start(self) -> None:
        self.events.clear()
        self.t0 = time.perf_counter()

    def stop(self) -> None:
        self.t_end = time.perf_counter()

    def record(self, kind: str, microbatch: int, begin: float, end: float) -> None:
        self.events.append((kind, microbatch, begin - self.t0, end - self.t0))

    @property
    def wall_time(self) -> float:
        return self.t_end - self.t0

    def busy_time(self, kinds: tuple[str, ...] = ("forward", "backward")) -> float:
        return sum(e - b for kind, _, b, e in self.events if kind in kinds)

    def comm_time(self) -> float:
        return sum(e - b for kind, _, b, e in self.events if kind in ("send", "recv"))

    def bubble_fraction(self) -> float:
        """Fraction of wall time this stage spent neither in forward nor backward.

        Communication counts as bubble: a stage blocked in ``recv`` is idle
        compute, which is exactly what the bubble measures.
        """
        if self.wall_time <= 0:
            return 0.0
        return max(0.0, 1.0 - self.busy_time() / self.wall_time)

    def ascii_gantt(self, width: int = 72) -> str:
        """One-line-per-stage visualisation of the schedule actually executed."""
        if self.wall_time <= 0:
            return ""
        row = ["."] * width
        glyph = {"forward": "F", "backward": "B", "send": ">", "recv": "<"}
        for kind, _, b, e in self.events:
            lo = int(b / self.wall_time * (width - 1))
            hi = max(lo, int(e / self.wall_time * (width - 1)))
            for i in range(lo, min(hi + 1, width)):
                if kind in ("forward", "backward") or row[i] == ".":
                    row[i] = glyph[kind]
        return "".join(row)


# --------------------------------------------------------------------------
# Point-to-point transport
# --------------------------------------------------------------------------


class _P2P:
    """Shape-self-describing send/recv between adjacent pipeline stages.

    The receiver has to allocate before it can receive, so it needs the shape
    in advance.  Rather than making the caller declare activation shapes, the
    first transfer of each kind carries a small integer header; both sides
    cache it and every subsequent transfer is a single message.  Sender and
    receiver stay in lockstep because they consult the same cache key.

    All sends are non-blocking and their handles are drained at the end of the
    step.  Blocking sends would deadlock the 1F1B steady state, where a stage
    wants to send an activation forward and receive a gradient backward at the
    same moment.
    """

    def __init__(self, group: dist.ProcessGroup | None, device: torch.device):
        self.group = group
        self.device = device
        # Send and receive caches are deliberately separate.  A middle stage
        # both receives activations from p-1 and sends activations to p+1; a
        # single cache would let the incoming header satisfy the outgoing
        # channel's "already described" check, so p+1 would sit waiting for a
        # header that was never sent and then mis-parse the payload as one.
        self._send_known: dict[str, tuple[torch.Size, torch.dtype]] = {}
        self._recv_known: dict[str, tuple[torch.Size, torch.dtype]] = {}
        self._pending: list[tuple[dist.Work, torch.Tensor]] = []

    def _header_for(self, t: torch.Tensor) -> torch.Tensor:
        h = torch.zeros(_HEADER_LEN, dtype=torch.int64, device=self.device)
        h[0] = t.dim()
        for i, d in enumerate(t.shape):
            h[1 + i] = d
        h[_HEADER_LEN - 1] = _DTYPES.index(t.dtype)
        return h

    def send(self, tensor: torch.Tensor, dst: int, key: str) -> None:
        # Detach before sending: the receiving stage starts a fresh autograd
        # graph rooted at the tensor it receives, and gradients cross the stage
        # boundary explicitly as a separate message, not through the graph.
        tensor = tensor.detach().contiguous()
        if key not in self._send_known:
            header = self._header_for(tensor)
            self._pending.append((dist.isend(header, dst=dst, group=self.group), header))
            self._send_known[key] = (tensor.shape, tensor.dtype)
        self._pending.append((dist.isend(tensor, dst=dst, group=self.group), tensor))

    def recv(self, src: int, key: str) -> torch.Tensor:
        if key not in self._recv_known:
            header = torch.zeros(_HEADER_LEN, dtype=torch.int64, device=self.device)
            dist.recv(header, src=src, group=self.group)
            ndim = int(header[0])
            shape = torch.Size([int(header[1 + i]) for i in range(ndim)])
            self._recv_known[key] = (shape, _DTYPES[int(header[_HEADER_LEN - 1])])
        shape, dtype = self._recv_known[key]
        buf = torch.empty(shape, dtype=dtype, device=self.device)
        dist.recv(buf, src=src, group=self.group)
        return buf

    def drain(self) -> None:
        for work, _ in self._pending:
            work.wait()
        self._pending.clear()


# --------------------------------------------------------------------------
# Stage
# --------------------------------------------------------------------------


class PipelineStage(nn.Module):
    """This rank's slice of the model, plus the loss on the final stage.

    ``module`` maps the incoming activation to the outgoing one.  On the last
    stage it produces whatever ``loss_fn(output, target)`` consumes.
    """

    def __init__(
        self,
        module: nn.Module,
        mesh: ParallelMesh,
        loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
    ):
        super().__init__()
        self.module = module
        self.mesh = mesh
        self.loss_fn = loss_fn
        if mesh.is_last_stage and loss_fn is None:
            raise ValueError("the last pipeline stage needs a loss_fn")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.module(x)


# --------------------------------------------------------------------------
# Schedules
# --------------------------------------------------------------------------


class _BaseSchedule:
    def __init__(self, stage: PipelineStage, mesh: ParallelMesh, num_microbatches: int):
        if num_microbatches < 1:
            raise ValueError("num_microbatches must be >= 1")
        self.stage = stage
        self.mesh = mesh
        self.num_microbatches = num_microbatches
        self.timeline = StageTimeline()
        self.max_activations_in_flight = 0
        # fp16 loss scaling has to be applied before the schedule calls
        # backward, since the caller never sees the per-microbatch loss.
        self.loss_scale = 1.0
        self._p2p = _P2P(mesh.pp_group, mesh.device)
        # (input, output) per in-flight microbatch, consumed by backward.
        self._in_flight: list[tuple[torch.Tensor | None, torch.Tensor]] = []

    # -- primitives ---------------------------------------------------------

    def _forward_one(self, index: int, inputs, targets) -> torch.Tensor | None:
        mesh = self.mesh
        if mesh.is_first_stage:
            x = inputs[index]
        else:
            t = time.perf_counter()
            x = self._p2p.recv(mesh.prev_stage_rank, "act")
            self.timeline.record("recv", index, t, time.perf_counter())
            x.requires_grad_(True)

        t = time.perf_counter()
        y = self.stage(x)
        reported = None
        if mesh.is_last_stage:
            # Scale by 1/M so the accumulated gradient equals the gradient of
            # the mean loss over the whole batch, matching a non-pipelined step.
            loss = self.stage.loss_fn(y, targets[index]) / self.num_microbatches
            reported = loss.detach()
            y = loss * self.loss_scale if self.loss_scale != 1.0 else loss
        self.timeline.record("forward", index, t, time.perf_counter())

        if not mesh.is_last_stage:
            t = time.perf_counter()
            self._p2p.send(y, mesh.next_stage_rank, "act")
            self.timeline.record("send", index, t, time.perf_counter())

        self._in_flight.append((x if not mesh.is_first_stage else None, y))
        self.max_activations_in_flight = max(
            self.max_activations_in_flight, len(self._in_flight)
        )
        # The reported loss is always unscaled, so a caller reading the loss
        # curve never has to know whether fp16 scaling was active.
        return reported

    def _backward_one(self, index: int) -> None:
        mesh = self.mesh
        x, y = self._in_flight.pop(0)

        if mesh.is_last_stage:
            t = time.perf_counter()
            torch.autograd.backward(y)
            self.timeline.record("backward", index, t, time.perf_counter())
        else:
            t = time.perf_counter()
            grad = self._p2p.recv(mesh.next_stage_rank, "grad")
            self.timeline.record("recv", index, t, time.perf_counter())
            t = time.perf_counter()
            torch.autograd.backward(y, grad_tensors=grad)
            self.timeline.record("backward", index, t, time.perf_counter())

        if not mesh.is_first_stage:
            assert x is not None and x.grad is not None
            t = time.perf_counter()
            self._p2p.send(x.grad, mesh.prev_stage_rank, "grad")
            self.timeline.record("send", index, t, time.perf_counter())

    def _split(self, batch: torch.Tensor | None) -> list[torch.Tensor] | None:
        if batch is None:
            return None
        if batch.size(0) % self.num_microbatches:
            raise ValueError(
                f"batch of {batch.size(0)} is not divisible by "
                f"{self.num_microbatches} microbatches"
            )
        return list(batch.chunk(self.num_microbatches, dim=0))

    def step(self, inputs: torch.Tensor | None = None, targets: torch.Tensor | None = None):
        raise NotImplementedError

    # -- reporting ----------------------------------------------------------

    def report(self) -> dict[str, float]:
        return {
            "wall_s": self.timeline.wall_time,
            "busy_s": self.timeline.busy_time(),
            "comm_s": self.timeline.comm_time(),
            "measured_bubble": self.timeline.bubble_fraction(),
            "theoretical_bubble": theoretical_bubble_fraction(
                self.mesh.pp_size, self.num_microbatches
            ),
            "peak_activations_in_flight": float(self.max_activations_in_flight),
        }


class GPipeSchedule(_BaseSchedule):
    """All forwards, then all backwards.

    Simple and the easiest to reason about, but every microbatch's activations
    stay alive from its forward until its backward, so peak activation memory
    scales with ``M`` — the very knob you need to raise to shrink the bubble.
    """

    name = "gpipe"

    def step(self, inputs=None, targets=None):
        mb_inputs, mb_targets = self._split(inputs), self._split(targets)
        self.timeline.start()
        self._in_flight.clear()
        self.max_activations_in_flight = 0
        losses = []

        for i in range(self.num_microbatches):
            out = self._forward_one(i, mb_inputs, mb_targets)
            if out is not None:
                losses.append(out.detach())
        for i in range(self.num_microbatches):
            self._backward_one(i)

        self._p2p.drain()
        self.timeline.stop()
        return torch.stack(losses).sum() if losses else None


class OneForwardOneBackwardSchedule(_BaseSchedule):
    """1F1B: fill the pipe, then alternate one forward with one backward.

    Warm-up length is ``P - 1 - stage_rank`` forwards, which is exactly the
    number of microbatches that must be in flight for this stage to reach the
    steady state.  Earlier stages warm up more, and consequently hold more
    activations — the memory profile is triangular across stages, heaviest at
    stage 0.
    """

    name = "1f1b"

    def step(self, inputs=None, targets=None):
        mb_inputs, mb_targets = self._split(inputs), self._split(targets)
        self.timeline.start()
        self._in_flight.clear()
        self.max_activations_in_flight = 0
        losses = []

        M, P, p = self.num_microbatches, self.mesh.pp_size, self.mesh.pp_rank
        num_warmup = min(P - 1 - p, M)
        num_steady = M - num_warmup

        fwd = bwd = 0
        for _ in range(num_warmup):
            out = self._forward_one(fwd, mb_inputs, mb_targets)
            if out is not None:
                losses.append(out.detach())
            fwd += 1

        for _ in range(num_steady):
            out = self._forward_one(fwd, mb_inputs, mb_targets)
            if out is not None:
                losses.append(out.detach())
            fwd += 1
            self._backward_one(bwd)
            bwd += 1

        while bwd < M:
            self._backward_one(bwd)
            bwd += 1

        self._p2p.drain()
        self.timeline.stop()
        return torch.stack(losses).sum() if losses else None


_SCHEDULES = {
    "gpipe": GPipeSchedule,
    "1f1b": OneForwardOneBackwardSchedule,
}


def build_schedule(
    name: str,
    stage: PipelineStage,
    mesh: ParallelMesh,
    num_microbatches: int,
) -> _BaseSchedule:
    if name not in _SCHEDULES:
        raise ValueError(f"unknown schedule {name!r}; choose from {sorted(_SCHEDULES)}")
    return _SCHEDULES[name](stage, mesh, num_microbatches)
