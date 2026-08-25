"""Distributed Data Parallel: bucketed gradient all-reduce overlapped with backward.

The naive implementation of data parallelism is "run backward, then all-reduce
every gradient".  That is correct and slow, for two reasons:

* **Latency.**  A transformer has hundreds of parameter tensors, many of them
  small (biases, LayerNorm weights).  A collective on a 1024-element tensor is
  pure launch overhead; NCCL's fixed per-op cost dominates.  Fix: *bucketing* —
  pack gradients into ~25 MB flat buffers and issue one collective per bucket.

* **Serialisation.**  Waiting for backward to finish leaves the network idle
  for the whole backward pass and the GPU idle for the whole all-reduce.  Fix:
  *overlap* — a bucket's gradients are final as soon as its parameters'
  ``AccumulateGrad`` nodes have run, which happens progressively through
  backward.  Launch each bucket's all-reduce asynchronously the moment it fills
  and let it ride alongside the remaining backward compute.

Because backward visits layers last-to-first, gradients become ready in roughly
the reverse of ``model.parameters()`` order.  Buckets are therefore built in
reverse parameter order, which makes "bucket 0 fills first" the common case.

Deterministic collective ordering
---------------------------------
NCCL matches collectives *by call order*, not by name or tag: if rank 0 issues
all-reduce(bucket 2) while rank 1 issues all-reduce(bucket 5), the two get
paired and you silently reduce mismatched buffers, or hang.  Gradient-ready
order is not guaranteed identical across ranks in general.

This implementation removes the hazard structurally: bucket ``k`` is only
launched once buckets ``0..k-1`` have launched.  Readiness is still what
*triggers* a launch, so overlap is preserved, but the observable sequence of
collectives is bucket 0, 1, 2, ... on every rank, always.  Reverse-order
bucketing is what keeps that in-order constraint from actually costing
anything.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

import torch
import torch.distributed as dist
from torch import nn

from . import comm

__all__ = ["DistributedDataParallel", "DDP"]

_DEFAULT_BUCKET_CAP_MB = 25.0


class _Bucket:
    """A flat buffer holding the gradients of a contiguous run of parameters."""

    __slots__ = ("params", "offsets", "numel", "buffer", "pending", "work", "index")

    def __init__(self, index: int, params: list[nn.Parameter]):
        self.index = index
        self.params = params
        self.offsets: list[int] = []
        offset = 0
        for p in params:
            self.offsets.append(offset)
            offset += p.numel()
        self.numel = offset
        self.buffer: torch.Tensor | None = None
        self.pending = len(params)
        self.work: dist.Work | None = None

    def reset(self) -> None:
        self.pending = len(self.params)
        self.work = None

    def gather_into_buffer(self, scale: float) -> torch.Tensor:
        """Flatten parameter grads into the bucket, pre-scaled by ``1/world``.

        Scaling *before* the reduction rather than after keeps fp16 gradients
        away from the top of the exponent range: summing ``W`` copies of a
        gradient can overflow where the mean cannot.
        """
        p0 = self.params[0]
        if self.buffer is None:
            self.buffer = torch.zeros(self.numel, dtype=p0.dtype, device=p0.device)
        for p, off in zip(self.params, self.offsets, strict=True):
            flat = self.buffer[off : off + p.numel()]
            if p.grad is None:
                flat.zero_()
            else:
                flat.copy_(p.grad.detach().reshape(-1))
        self.buffer.mul_(scale)
        return self.buffer

    def scatter_from_buffer(self) -> None:
        assert self.buffer is not None
        for p, off in zip(self.params, self.offsets, strict=True):
            reduced = self.buffer[off : off + p.numel()].view_as(p)
            if p.grad is None:
                p.grad = reduced.clone()
            else:
                p.grad.copy_(reduced)


class DistributedDataParallel(nn.Module):
    """Replicate a module across a data-parallel group and average gradients.

    Parameters
    ----------
    module:
        The model to wrap.  Its parameters are broadcast from ``src`` at
        construction so every replica starts identical — a requirement people
        usually satisfy by accident (same seed) and then break the first time
        they add a rank-dependent init.
    process_group:
        The data-parallel group (from :class:`~minidist.mesh.ParallelMesh`).
        ``None`` means the default group.
    bucket_cap_mb:
        Target bucket size.  Smaller buckets start communicating earlier (more
        overlap) but issue more collectives (more launch overhead); 25 MB is
        the value PyTorch's own DDP defaults to.
    overlap_with_backward:
        Set ``False`` to disable overlap and reduce everything after backward
        completes.  Only useful for measuring what overlap is worth — the
        benchmarks toggle it to produce that comparison.
    """

    def __init__(
        self,
        module: nn.Module,
        process_group: dist.ProcessGroup | None = None,
        bucket_cap_mb: float = _DEFAULT_BUCKET_CAP_MB,
        broadcast_parameters: bool = True,
        overlap_with_backward: bool = True,
        src: int = 0,
    ):
        super().__init__()
        self.module = module
        self.process_group = process_group
        self.world_size = comm.get_world_size(process_group)
        self.overlap_with_backward = overlap_with_backward
        self.require_backward_grad_sync = True

        if broadcast_parameters and self.world_size > 1:
            self._sync_module_states(src)

        params = [p for p in module.parameters() if p.requires_grad]
        self._buckets = self._build_buckets(params, bucket_cap_mb)
        self._param_to_bucket = {
            id(p): b for b in self._buckets for p in b.params
        }
        self._next_to_launch = 0
        self._callback_queued = False
        self._handles: list[torch.utils.hooks.RemovableHandle] = []

        if self.world_size > 1:
            self._register_hooks()

    # -- setup --------------------------------------------------------------

    def _sync_module_states(self, src: int) -> None:
        group_src = 0 if self.process_group is None else src
        with torch.no_grad():
            for p in self.module.parameters():
                dist.broadcast(p.data, src=self._global_src(group_src), group=self.process_group)
            for b in self.module.buffers():
                if b.is_floating_point() or b.dtype in (torch.int64, torch.int32, torch.bool):
                    dist.broadcast(b.data, src=self._global_src(group_src), group=self.process_group)

    def _global_src(self, group_rank: int) -> int:
        """``dist.broadcast`` wants a *global* rank even for a subgroup."""
        if self.process_group is None:
            return group_rank
        return dist.get_global_rank(self.process_group, group_rank)

    def _build_buckets(self, params: list[nn.Parameter], cap_mb: float) -> list[_Bucket]:
        """Group parameters into buckets in reverse (≈ backward) order."""
        cap_bytes = int(cap_mb * 1024 * 1024)
        buckets: list[_Bucket] = []
        current: list[nn.Parameter] = []
        current_bytes = 0
        for p in reversed(params):
            pbytes = p.numel() * p.element_size()
            if current and current_bytes + pbytes > cap_bytes:
                buckets.append(_Bucket(len(buckets), current))
                current, current_bytes = [], 0
            current.append(p)
            current_bytes += pbytes
        if current:
            buckets.append(_Bucket(len(buckets), current))
        return buckets

    def _register_hooks(self) -> None:
        for bucket in self._buckets:
            for p in bucket.params:
                self._handles.append(
                    p.register_post_accumulate_grad_hook(self._on_grad_ready)
                )

    # -- backward-time machinery -------------------------------------------

    def _on_grad_ready(self, param: torch.Tensor) -> None:
        if not self.require_backward_grad_sync:
            return

        # Queue the end-of-backward callback from inside the engine, which is
        # the only place `queue_callback` is legal.  This is what lets callers
        # write a plain `loss.backward(); opt.step()` without an explicit sync.
        if not self._callback_queued:
            self._callback_queued = True
            try:
                torch.autograd.Variable._execution_engine.queue_callback(
                    self._finalize_backward
                )
            except Exception:  # pragma: no cover - private API safety net
                self._callback_queued = False

        bucket = self._param_to_bucket[id(param)]
        bucket.pending -= 1
        if self.overlap_with_backward:
            self._launch_ready_buckets()

    def _launch_ready_buckets(self) -> None:
        """Launch every bucket that is both full and next in line."""
        while self._next_to_launch < len(self._buckets):
            bucket = self._buckets[self._next_to_launch]
            if bucket.pending > 0:
                break
            self._launch(bucket)
            self._next_to_launch += 1

    def _launch(self, bucket: _Bucket) -> None:
        flat = bucket.gather_into_buffer(1.0 / self.world_size)
        bucket.work = comm.all_reduce(
            flat, group=self.process_group, async_op=True
        )

    def _finalize_backward(self) -> None:
        self._callback_queued = False
        self.finish_gradient_sync()

    def finish_gradient_sync(self) -> None:
        """Flush and wait for all outstanding gradient all-reduces.

        Idempotent, and normally invoked automatically at the end of backward;
        call it explicitly if you drive the autograd engine yourself (the
        pipeline schedules do).
        """
        if self.world_size == 1 or not self.require_backward_grad_sync:
            return
        # Any bucket still unlaunched (parameters that never received a
        # gradient this step, or overlap disabled) goes out now, in index
        # order, so the global collective sequence stays identical everywhere.
        while self._next_to_launch < len(self._buckets):
            self._launch(self._buckets[self._next_to_launch])
            self._next_to_launch += 1

        for bucket in self._buckets:
            if bucket.work is not None:
                bucket.work.wait()
                bucket.scatter_from_buffer()
            bucket.reset()
        self._next_to_launch = 0

    # -- public API ---------------------------------------------------------

    @contextmanager
    def no_sync(self) -> Iterator[None]:
        """Accumulate gradients locally without communicating.

        Used for gradient accumulation and by the pipeline schedules, where
        every microbatch's backward would otherwise trigger its own all-reduce
        instead of one reduction over the accumulated gradient.
        """
        previous = self.require_backward_grad_sync
        self.require_backward_grad_sync = False
        try:
            yield
        finally:
            self.require_backward_grad_sync = previous

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def bucket_summary(self) -> list[tuple[int, int, float]]:
        """``(bucket_index, n_params, megabytes)`` — used by the benchmarks."""
        out = []
        for b in self._buckets:
            mb = sum(p.numel() * p.element_size() for p in b.params) / 1024**2
            out.append((b.index, len(b.params), mb))
        return out

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)


DDP = DistributedDataParallel
