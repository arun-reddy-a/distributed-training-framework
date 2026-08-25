"""Mixed-precision training, including the part that only breaks in a distributed run.

Choosing a precision
--------------------
====================  =========  =========  ================================
format                exp bits   mantissa   dynamic range
====================  =========  =========  ================================
fp32                  8          23         ~1e-38 .. 3e38
fp16                  5          10         ~6e-8 .. 65504
bf16                  8          7          ~1e-38 .. 3e38  (fp32's range)
====================  =========  =========  ================================

bf16 keeps fp32's exponent field and spends the savings out of the mantissa.
Gradients that would flush to zero in fp16 (activation gradients in deep stacks
routinely land at 1e-9) are representable in bf16, which is why bf16 needs no
loss scaling at all.  Its cost is precision, not range — and gradient
*accumulation* in 7 mantissa bits does lose information, which is why the
reductions in :mod:`minidist.fsdp` default to fp32 even when parameters are bf16.

fp16 has the mantissa but not the range, so it needs **loss scaling**: multiply
the loss by ``S`` before backward, which scales every gradient by ``S`` and
lifts small values off the denormal floor; divide by ``S`` before the optimizer
sees them.  ``S`` is adapted at runtime — doubled after a run of clean steps,
halved whenever a gradient overflows.

Why the scaler has to be collective
-----------------------------------
This is the part that has no single-GPU analogue, and it is a genuinely nasty
bug when you get it wrong.

``inf``/``nan`` detection is inherently *local*: each rank inspects the
gradients it holds.  Under DDP the gradients are already all-reduced so every
rank sees the same values — but under **FSDP each rank holds a different shard**,
and under **tensor parallelism each rank holds a different slice of the weight
matrix**.  An overflow in one shard is invisible to every other rank.

If rank 2 detects the overflow and skips its optimizer step while ranks 0, 1, 3
apply theirs, the replicas silently diverge — and because the loss still goes
down, nothing looks wrong until an evaluation much later disagrees with
training.  Worse, the ranks now disagree about the scale factor, so their
scaling histories drift apart permanently.

The fix is one line of communication and is not optional: **all-reduce the
``found_inf`` flag with MAX across every rank that holds a piece of the
gradient, and let all ranks act on the same answer.**
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass

import torch
import torch.distributed as dist

from . import comm

__all__ = ["PrecisionConfig", "DistributedGradScaler", "resolve_precision"]


@dataclass
class PrecisionConfig:
    """The precision decisions for one training run."""

    dtype: torch.dtype
    device_type: str
    enabled: bool

    @property
    def needs_grad_scaler(self) -> bool:
        return self.enabled and self.dtype == torch.float16

    def autocast(self):
        if not self.enabled:
            return nullcontext()
        return torch.autocast(device_type=self.device_type, dtype=self.dtype)

    def __repr__(self) -> str:
        if not self.enabled:
            return "PrecisionConfig(fp32)"
        return f"PrecisionConfig({str(self.dtype).replace('torch.', '')} autocast)"


def resolve_precision(name: str, device: torch.device | None = None) -> PrecisionConfig:
    """Turn ``'fp32' | 'fp16' | 'bf16' | 'auto'`` into a concrete configuration.

    ``auto`` prefers bf16 wherever the hardware supports it natively (Ampere
    and later, or CPU), because it avoids the loss-scaling machinery entirely,
    and falls back to fp16 on older CUDA devices.
    """
    device = device or comm.current_device()
    device_type = device.type

    if name == "auto":
        if device_type == "cuda":
            name = "bf16" if torch.cuda.is_bf16_supported() else "fp16"
        else:
            name = "bf16"

    if name == "fp32":
        return PrecisionConfig(torch.float32, device_type, enabled=False)
    if name == "bf16":
        return PrecisionConfig(torch.bfloat16, device_type, enabled=True)
    if name == "fp16":
        if device_type == "cpu":
            # CPU fp16 autocast exists but is emulated and slower than fp32;
            # more importantly the overflow behaviour under test would not
            # match a GPU run, so refuse rather than mislead.
            raise ValueError("fp16 autocast is not useful on CPU; use bf16 or fp32")
        return PrecisionConfig(torch.float16, device_type, enabled=True)
    raise ValueError(f"unknown precision {name!r}")


class DistributedGradScaler:
    """Adaptive loss scaling whose skip/scale decisions are agreed across ranks.

    A drop-in replacement for ``torch.amp.GradScaler`` that adds the collective
    ``found_inf`` reduction described in the module docstring, and works on CPU
    so the behaviour is testable without a GPU.

    Parameters
    ----------
    process_group:
        The group across which the overflow decision is agreed.  This must span
        **every rank holding a piece of the gradient** — for a 3D-parallel run
        that is the whole world, not just the data-parallel group, because TP
        ranks hold disjoint slices of the same weight matrix.  ``None`` (the
        default) means the default group, i.e. the whole world.
    growth_interval:
        Consecutive non-overflowing steps before the scale is doubled.  Too
        small and the scale oscillates, wasting a step on every overflow; 2000
        is the usual compromise.
    """

    def __init__(
        self,
        init_scale: float = 2.0**16,
        growth_factor: float = 2.0,
        backoff_factor: float = 0.5,
        growth_interval: int = 2000,
        enabled: bool = True,
        process_group: dist.ProcessGroup | None = None,
        min_scale: float = 1.0,
    ):
        self.enabled = enabled
        self.process_group = process_group
        self._scale = float(init_scale)
        self._growth_factor = growth_factor
        self._backoff_factor = backoff_factor
        self._growth_interval = growth_interval
        self._min_scale = min_scale
        self._good_steps = 0
        self._found_inf_this_step = False
        self._unscaled = False
        self.num_skipped = 0
        self.num_steps = 0

    # -- state --------------------------------------------------------------

    @property
    def scale(self) -> float:
        return self._scale if self.enabled else 1.0

    def state_dict(self) -> dict:
        return {
            "scale": self._scale,
            "good_steps": self._good_steps,
            "num_skipped": self.num_skipped,
            "num_steps": self.num_steps,
        }

    def load_state_dict(self, state: dict) -> None:
        self._scale = state["scale"]
        self._good_steps = state["good_steps"]
        self.num_skipped = state.get("num_skipped", 0)
        self.num_steps = state.get("num_steps", 0)

    # -- the training-loop surface -----------------------------------------

    def scale_loss(self, loss: torch.Tensor) -> torch.Tensor:
        if not self.enabled:
            return loss
        return loss * self._scale

    # Alias matching torch.amp.GradScaler's spelling.
    scale_tensor = scale_loss

    def unscale_(self, optimizer: torch.optim.Optimizer) -> bool:
        """Divide gradients by the current scale and detect overflow.

        Returns whether an overflow was found anywhere in the group.  Call this
        directly if you need real (unscaled) gradients before ``step`` — for
        gradient clipping, which is otherwise applied to scaled values and
        clips to the wrong threshold.
        """
        if not self.enabled:
            return False
        if self._unscaled:
            return self._found_inf_this_step

        inv_scale = 1.0 / self._scale
        device = comm.current_device()
        # Accumulate on-device and reduce once.  Checking each gradient with
        # `.item()` would force a host synchronisation per parameter tensor and
        # serialise the whole backward pipeline behind the CPU.
        found_inf = torch.zeros(1, device=device)

        for group in optimizer.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                p.grad.mul_(inv_scale)
                found_inf += (~torch.isfinite(p.grad)).any().to(found_inf.dtype)

        # The collective that makes this correct under FSDP / TP.
        if comm.is_distributed() and comm.get_world_size(self.process_group) > 1:
            dist.all_reduce(found_inf, op=dist.ReduceOp.MAX, group=self.process_group)

        self._found_inf_this_step = bool(found_inf.item() > 0)
        self._unscaled = True
        return self._found_inf_this_step

    def step(self, optimizer: torch.optim.Optimizer) -> bool:
        """Run ``optimizer.step()`` unless the group agreed an overflow occurred.

        Returns ``True`` if the step was applied.
        """
        self.num_steps += 1
        if not self.enabled:
            optimizer.step()
            return True

        found_inf = self.unscale_(optimizer)
        if found_inf:
            self.num_skipped += 1
            return False
        optimizer.step()
        return True

    def update(self) -> None:
        """Adapt the scale, then reset per-step state.

        Every rank runs identical arithmetic on an identical ``found_inf``, so
        the scale stays bit-identical across the group without communicating
        the scale itself.
        """
        if not self.enabled:
            return
        if self._found_inf_this_step:
            self._scale = max(self._min_scale, self._scale * self._backoff_factor)
            self._good_steps = 0
        else:
            self._good_steps += 1
            if self._good_steps >= self._growth_interval:
                self._scale *= self._growth_factor
                self._good_steps = 0
        self._found_inf_this_step = False
        self._unscaled = False

    def summary(self) -> str:
        if not self.enabled:
            return "grad scaler disabled (bf16 or fp32)"
        pct = 100.0 * self.num_skipped / max(1, self.num_steps)
        return (
            f"scale={self._scale:g}  skipped={self.num_skipped}/{self.num_steps} "
            f"({pct:.1f}%)"
        )
