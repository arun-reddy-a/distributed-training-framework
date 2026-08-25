"""Fully Sharded Data Parallel (ZeRO-3): shard parameters, gradients and optimizer state.

DDP replicates the full model on every rank, so the memory floor is
``P * (2 + 2 + 4 + 4 + 4)`` bytes for a bf16 model with fp32 Adam — parameters,
gradients, the fp32 master copy, and Adam's two moments — regardless of how
many GPUs you own.  Adding ranks buys throughput and nothing else.

FSDP shards all three along the data-parallel group, trading memory for
communication.  Per rank:

.. code-block:: text

    DDP    :  2P + 2P + 12P                    = 16P bytes
    FSDP   : (2P + 2P + 12P) / W + 2P_unit     ≈ 16P/W bytes

where ``P_unit`` is the largest wrapped unit — the transient full-precision
parameters that exist only while that unit is executing.  That last term is why
FSDP is wrapped per *block* rather than once around the whole model: the peak
is set by the biggest thing you ever gather at once.

The lifecycle of one unit
-------------------------
.. code-block:: text

    forward pre-hook   all-gather shard -> full parameters      [comm]
    forward            run the module                           [compute]
    forward post-hook  free full parameters                     [memory]
    backward pre-hook  all-gather shard -> full parameters      [comm]
    backward           compute full gradients                   [compute]
    grads ready        reduce-scatter gradients -> shard, free  [comm]

Communication cost: one all-gather forward, one all-gather backward, one
reduce-scatter backward — ``3/2 x`` DDP's single all-reduce (which is itself a
reduce-scatter plus an all-gather).  That 50% surcharge buys a ``W``-fold
reduction in memory, and it is the whole trade.

Freeing parameters that autograd still needs
--------------------------------------------
The subtle part is step three.  Forward saves parameters for backward, so we
cannot simply drop them.  What we *can* do is free their **storage** and
re-fill it later at the same address-independent offsets:

.. code-block:: python

    full.untyped_storage().resize_(0)          # free — views survive, data does not
    full.untyped_storage().resize_(nbytes)     # realloc, then all-gather back into it

Each parameter's ``.data`` is a view into ``full``, assigned once at
construction and never reassigned.  A view resolves its address as
``storage.data_ptr() + offset`` at *access* time, so re-allocating the storage
silently re-points every view.  Autograd is undisturbed because it saved the
``Parameter`` objects, whose version counters are independent of the buffer we
write into.  This is the same mechanism ``torch.distributed.fsdp`` uses, and it
is the reason parameters must not be touched outside forward/backward — see
:meth:`FullyShardedDataParallel.summon_full_params` for the supported way in.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from contextlib import contextmanager

import torch
import torch.distributed as dist
from torch import nn

from . import comm

__all__ = ["FullyShardedDataParallel", "FSDP", "MixedPrecisionPolicy"]


class MixedPrecisionPolicy:
    """Dtypes for the three places FSDP can independently choose precision.

    Parameters
    ----------
    param_dtype:
        Dtype of the *gathered* parameters, i.e. what the matmuls actually see.
        ``bfloat16`` here halves both all-gather traffic and activation memory.
    reduce_dtype:
        Dtype of the gradient reduce-scatter.  Keeping this at ``float32`` even
        when ``param_dtype`` is bf16 is the usual choice: gradients are summed
        across ``W`` ranks, and bf16's 8-bit mantissa loses small contributions
        in that sum.  It costs 2x the reduction traffic.
    keep_low_precision_shard:
        If ``False`` (default) the sharded master weights stay fp32, so the
        optimizer update happens in full precision.  This is the "master
        weights" half of mixed-precision training, done shard-wise.
    """

    def __init__(
        self,
        param_dtype: torch.dtype = torch.float32,
        reduce_dtype: torch.dtype = torch.float32,
        keep_low_precision_shard: bool = False,
    ):
        self.param_dtype = param_dtype
        self.reduce_dtype = reduce_dtype
        self.keep_low_precision_shard = keep_low_precision_shard

    def __repr__(self) -> str:
        return (
            f"MixedPrecisionPolicy(param={self.param_dtype}, "
            f"reduce={self.reduce_dtype})"
        )


class _FSDPUnit:
    """One shard-gather-free granule: a module plus the parameters it owns."""

    def __init__(
        self,
        module: nn.Module,
        params: list[nn.Parameter],
        group,
        policy: MixedPrecisionPolicy,
        reshard_after_forward: bool,
        device: torch.device,
        name: str,
    ):
        self.module = module
        self.params = params
        self.group = group
        self.policy = policy
        self.reshard_after_forward = reshard_after_forward
        self.name = name
        self.world = comm.get_world_size(group)
        self.rank = comm.get_rank(group)

        self.shapes = [p.shape for p in params]
        self.numels = [p.numel() for p in params]
        self.total_numel = sum(self.numels)
        # Pad so the flat parameter divides evenly; all-gather and
        # reduce-scatter both require equal-sized contributions per rank.
        self.shard_numel = (self.total_numel + self.world - 1) // self.world
        self.padded_numel = self.shard_numel * self.world

        shard_dtype = (
            policy.param_dtype if policy.keep_low_precision_shard else params[0].dtype
        )

        # Build the full flat parameter once, keep this rank's slice, and hand
        # the rest back to the allocator.
        flat = torch.zeros(self.padded_numel, dtype=params[0].dtype, device=device)
        offset = 0
        for p, n in zip(params, self.numels, strict=True):
            flat[offset : offset + n].copy_(p.detach().reshape(-1))
            offset += n
        shard = flat[self.rank * self.shard_numel : (self.rank + 1) * self.shard_numel]
        self.flat_shard = nn.Parameter(shard.clone().to(shard_dtype))
        del flat

        # The gathered buffer.  Views into it are assigned to the original
        # parameters exactly once, here, and are never reassigned again.
        self._full = torch.empty(self.padded_numel, dtype=policy.param_dtype, device=device)
        offset = 0
        for p, n, shape in zip(params, self.numels, self.shapes, strict=True):
            p.data = self._full[offset : offset + n].view(shape)
            offset += n
        self._unsharded = True
        self.reshard()

        self._grads_pending = 0
        self._hook_handles: list[torch.utils.hooks.RemovableHandle] = []

    # -- shard / unshard ----------------------------------------------------

    @property
    def is_unsharded(self) -> bool:
        return self._unsharded

    def unshard(self) -> None:
        if self._unsharded:
            return
        self._full.untyped_storage().resize_(
            self.padded_numel * self._full.element_size()
        )
        src = self.flat_shard.detach()
        if src.dtype != self.policy.param_dtype:
            src = src.to(self.policy.param_dtype)
        comm.all_gather_into_tensor(self._full, src.contiguous(), group=self.group)
        self._unsharded = True

    def reshard(self) -> None:
        """Free the gathered parameters.

        Deliberately does *not* touch ``p.data``: the parameter objects are the
        tensors autograd saved, and reassigning their data would change their
        shape out from under the backward graph.  Only the bytes go away.
        """
        if not self._unsharded:
            return
        self._full.untyped_storage().resize_(0)
        self._unsharded = False

    # -- gradients ----------------------------------------------------------

    def arm_gradient_reduction(self) -> None:
        self._grads_pending = sum(1 for p in self.params if p.requires_grad)

    def on_param_grad_ready(self, sync: bool) -> None:
        self._grads_pending -= 1
        if self._grads_pending > 0:
            return
        if sync:
            self.reduce_scatter_grads()
        # Always reshard once backward is done with this unit, even when
        # `reshard_after_forward` is False.  That flag only governs whether the
        # parameters survive *between* forward and backward; holding them past
        # the end of backward would leave the gathered copy stale the moment
        # the optimizer updates `flat_shard`, and the next forward would
        # silently train on pre-update weights.
        self.reshard()

    def reduce_scatter_grads(self) -> None:
        """Reduce full gradients across the group, keeping only our shard."""
        reduce_dtype = self.policy.reduce_dtype
        buf = torch.zeros(self.padded_numel, dtype=reduce_dtype, device=self.flat_shard.device)
        offset = 0
        for p, n in zip(self.params, self.numels, strict=True):
            if p.grad is not None:
                buf[offset : offset + n].copy_(p.grad.detach().reshape(-1))
            offset += n
        # Pre-scale rather than post-scale: the reduction is a sum over W ranks
        # and a low-precision reduce_dtype can overflow before the divide.
        if self.world > 1:
            buf.div_(self.world)
            out = torch.empty(self.shard_numel, dtype=reduce_dtype, device=buf.device)
            comm.reduce_scatter_tensor(out, buf, group=self.group)
        else:
            out = buf

        out = out.to(self.flat_shard.dtype)
        if self.flat_shard.grad is None:
            self.flat_shard.grad = out
        else:
            self.flat_shard.grad.add_(out)

        # Drop the full-size gradients — this is the memory FSDP exists to save.
        for p in self.params:
            p.grad = None

    def numel_report(self) -> tuple[int, int]:
        return self.total_numel, self.shard_numel


class FullyShardedDataParallel(nn.Module):
    """Shard a module's parameters, gradients and optimizer state across a group.

    Parameters
    ----------
    module:
        The model.  Parameters are broadcast from rank 0 before sharding, so
        replicas need not have been initialised identically.
    unit_types:
        Module classes that become sharding units — typically the transformer
        block.  Any parameter not inside a unit is collected into a root unit.
        Unit granularity is the main tuning knob: more units means lower peak
        memory and more, smaller all-gathers.
    process_group:
        The data-parallel group.
    mixed_precision:
        See :class:`MixedPrecisionPolicy`.
    reshard_after_forward:
        Free a unit's parameters between its forward and backward, re-gathering
        them in the backward pre-hook (ZeRO-3).  ``False`` keeps them resident,
        which removes the backward all-gather at the cost of holding every
        unit's full parameters at once (closer to ZeRO-2).
    """

    def __init__(
        self,
        module: nn.Module,
        unit_types: Sequence[type] = (),
        process_group: dist.ProcessGroup | None = None,
        mixed_precision: MixedPrecisionPolicy | None = None,
        reshard_after_forward: bool = True,
        reshard_root_after_forward: bool = False,
        broadcast_parameters: bool = True,
        device: torch.device | None = None,
    ):
        super().__init__()
        self.module = module
        self.process_group = process_group
        self.world_size = comm.get_world_size(process_group)
        self.policy = mixed_precision or MixedPrecisionPolicy()
        self.require_backward_grad_sync = True
        device = device or comm.current_device()

        if broadcast_parameters and self.world_size > 1:
            self._broadcast_module_states()

        self._units = self._build_units(
            module,
            tuple(unit_types),
            reshard_after_forward,
            reshard_root_after_forward,
            device,
        )
        # Only the shards are trainable state; the original parameters are
        # views that exist solely to run forward.
        self._shards = nn.ParameterList([u.flat_shard for u in self._units])
        self._register_hooks()

    # -- construction -------------------------------------------------------

    def _broadcast_module_states(self) -> None:
        src = 0 if self.process_group is None else dist.get_global_rank(self.process_group, 0)
        with torch.no_grad():
            for p in self.module.parameters():
                dist.broadcast(p.data, src=src, group=self.process_group)
            for b in self.module.buffers():
                dist.broadcast(b.data, src=src, group=self.process_group)

    def _build_units(
        self,
        root: nn.Module,
        unit_types: tuple[type, ...],
        reshard_after_forward: bool,
        reshard_root: bool,
        device: torch.device,
    ) -> list[_FSDPUnit]:
        units: list[_FSDPUnit] = []
        claimed: set[int] = set()

        # `named_modules` is a pre-order walk, so an outer matching module
        # claims its parameters before any nested match sees them — nested
        # units would otherwise double-shard the same tensors.
        for name, m in root.named_modules():
            if m is root or not isinstance(m, unit_types):
                continue
            params = [p for p in m.parameters() if id(p) not in claimed]
            if not params:
                continue
            claimed.update(id(p) for p in params)
            units.append(
                _FSDPUnit(
                    m, params, self.process_group, self.policy,
                    reshard_after_forward, device, name or "root",
                )
            )

        leftovers = [p for p in root.parameters() if id(p) not in claimed]
        if leftovers:
            # The root unit holds embeddings / final norm / head.  It is not
            # resharded after forward by default: its parameters are needed at
            # the very start *and* the very end of backward, so freeing them
            # buys one unit's worth of memory for two extra all-gathers.
            units.append(
                _FSDPUnit(
                    root, leftovers, self.process_group, self.policy,
                    reshard_root, device, "<root>",
                )
            )
        if not units:
            raise ValueError("FSDP wrapped a module with no parameters")
        return units

    def _register_hooks(self) -> None:
        for unit in self._units:
            unit._hook_handles.append(
                unit.module.register_forward_pre_hook(self._make_pre_forward(unit))
            )
            unit._hook_handles.append(
                unit.module.register_forward_hook(self._make_post_forward(unit))
            )
            if unit.reshard_after_forward:
                unit._hook_handles.append(
                    unit.module.register_full_backward_pre_hook(
                        self._make_pre_backward(unit)
                    )
                )
            for p in unit.params:
                if p.requires_grad:
                    unit._hook_handles.append(
                        p.register_post_accumulate_grad_hook(
                            self._make_grad_ready(unit)
                        )
                    )

    def _make_pre_forward(self, unit: _FSDPUnit):
        def hook(module, args):
            unit.unshard()
            if torch.is_grad_enabled():
                unit.arm_gradient_reduction()
        return hook

    def _make_post_forward(self, unit: _FSDPUnit):
        def hook(module, args, output):
            # Under `no_grad` there is no backward to reshard this unit later,
            # so free it here regardless of the flag.
            if unit.reshard_after_forward or not torch.is_grad_enabled():
                unit.reshard()
        return hook

    def _make_pre_backward(self, unit: _FSDPUnit):
        def hook(module, grad_output):
            unit.unshard()
        return hook

    def _make_grad_ready(self, unit: _FSDPUnit):
        def hook(param):
            unit.on_param_grad_ready(sync=self.require_backward_grad_sync)
        return hook

    # -- public API ---------------------------------------------------------

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def named_parameters(self, prefix: str = "", recurse: bool = True, remove_duplicate: bool = True):
        """Yield only the flat shards.

        The optimizer must see the sharded master weights, never the gathered
        views — those are transient and, most of the time, not even backed by
        memory.
        """
        for i, unit in enumerate(self._units):
            name = f"{prefix}{'.' if prefix else ''}_flat_shard_{i}"
            yield name, unit.flat_shard

    def parameters(self, recurse: bool = True):
        for _, p in self.named_parameters(recurse=recurse):
            yield p

    @contextmanager
    def no_sync(self) -> Iterator[None]:
        """Accumulate unsharded gradients locally without reduce-scatter.

        Note the memory cost: skipping the reduce-scatter also skips the
        ``p.grad = None`` that follows it, so every unit's *full* gradients stay
        resident until the next synchronised backward.  Gradient accumulation
        under FSDP is therefore much more expensive than under DDP.
        """
        previous = self.require_backward_grad_sync
        self.require_backward_grad_sync = False
        try:
            yield
        finally:
            self.require_backward_grad_sync = previous

    @contextmanager
    def summon_full_params(self) -> Iterator[None]:
        """Temporarily gather every unit so parameters can be read.

        The supported way to checkpoint or inspect weights.  Peak memory during
        the block is the full unsharded model on every rank.
        """
        was_unsharded = [u.is_unsharded for u in self._units]
        for unit in self._units:
            unit.unshard()
        try:
            yield
        finally:
            # Restore the exact prior state rather than applying the reshard
            # policy: summoning inside a forward pass must not free parameters
            # the surrounding computation is still using.
            for unit, was in zip(self._units, was_unsharded, strict=True):
                if not was:
                    unit.reshard()

    def finish_gradient_sync(self) -> None:
        """Flush any unit whose gradients were accumulated under ``no_sync``."""
        if self.world_size == 1:
            return
        for unit in self._units:
            if any(p.grad is not None for p in unit.params):
                unit.reduce_scatter_grads()
                if unit.reshard_after_forward:
                    unit.reshard()

    def memory_summary(self) -> dict[str, float]:
        """Per-rank state size in MB, next to what DDP would have cost."""
        total = sum(u.total_numel for u in self._units)
        sharded = sum(u.shard_numel for u in self._units)
        shard_bytes = self._units[0].flat_shard.element_size()
        largest_unit = max(u.total_numel for u in self._units)
        param_bytes = torch.empty((), dtype=self.policy.param_dtype).element_size()
        mb = 1024**2
        return {
            "params_total_M": total / 1e6,
            "shard_params_MB": sharded * shard_bytes / mb,
            "replicated_params_MB": total * shard_bytes / mb,
            # params + grads + Adam(m, v), all sharded
            "fsdp_state_MB": sharded * shard_bytes * 4 / mb,
            "ddp_state_MB": total * shard_bytes * 4 / mb,
            "transient_gather_MB": largest_unit * param_bytes / mb,
            "num_units": len(self._units),
        }

    def unit_summary(self) -> list[tuple[str, int, int]]:
        return [(u.name, u.total_numel, u.shard_numel) for u in self._units]

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)


FSDP = FullyShardedDataParallel
