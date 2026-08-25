"""The grad scaler's skip decision must be unanimous across ranks."""

from __future__ import annotations

import pytest
import torch

from minidist.amp import DistributedGradScaler, resolve_precision

from .common import run_distributed


def _model_and_optimizer():
    torch.manual_seed(0)
    model = torch.nn.Linear(4, 4)
    return model, torch.optim.SGD(model.parameters(), lr=0.1)


def _one_rank_overflow_body(rank, world):
    """An overflow on a single rank must stop *every* rank from stepping.

    This is the failure mode that has no single-GPU analogue: under FSDP or
    tensor parallelism each rank holds a different slice of the gradient, so an
    inf is visible to exactly one of them.  Without the collective, that rank
    skips while the others step, and the replicas diverge silently.
    """
    model, optimizer = _model_and_optimizer()
    scaler = DistributedGradScaler(init_scale=1024.0)

    for p in model.parameters():
        p.grad = torch.ones_like(p)
    if rank == 0:
        next(iter(model.parameters())).grad[0, 0] = float("inf")

    before = [p.detach().clone() for p in model.parameters()]
    stepped = scaler.step(optimizer)
    scaler.update()

    assert not stepped, f"[rank {rank}] stepped despite an overflow on rank 0"
    for p, old in zip(model.parameters(), before, strict=True):
        assert torch.equal(p.detach(), old), f"[rank {rank}] weights moved on a skipped step"
    assert scaler.scale == 512.0, f"[rank {rank}] scale is {scaler.scale}, expected 512"
    assert scaler.num_skipped == 1


def _clean_step_body(rank, world):
    """With finite gradients everywhere, every rank steps and agrees on the scale."""
    model, optimizer = _model_and_optimizer()
    scaler = DistributedGradScaler(init_scale=1024.0, growth_interval=2)

    for step in range(2):
        for p in model.parameters():
            # Gradients arrive scaled, as they would after `scale_loss`.
            p.grad = torch.full_like(p, 2.0) * scaler.scale
        assert scaler.step(optimizer), f"[rank {rank}] step {step} was skipped unexpectedly"
        scaler.update()

    assert scaler.scale == 2048.0, f"[rank {rank}] scale {scaler.scale} did not grow"
    assert scaler.num_skipped == 0


def _unscale_body(rank, world):
    """`unscale_` must restore true gradient magnitudes before clipping."""
    model, optimizer = _model_and_optimizer()
    scaler = DistributedGradScaler(init_scale=256.0)
    for p in model.parameters():
        p.grad = torch.full_like(p, 3.0) * scaler.scale

    assert not scaler.unscale_(optimizer)
    for p in model.parameters():
        assert torch.allclose(p.grad, torch.full_like(p.grad, 3.0)), (
            f"[rank {rank}] gradients were not unscaled"
        )

    # Calling it twice must be a no-op, not a second division.
    scaler.unscale_(optimizer)
    for p in model.parameters():
        assert torch.allclose(p.grad, torch.full_like(p.grad, 3.0)), (
            f"[rank {rank}] unscale_ is not idempotent"
        )


def _scale_agreement_body(rank, world):
    """After a mixed history of overflows, all ranks hold the same scale."""
    import torch.distributed as dist

    model, optimizer = _model_and_optimizer()
    scaler = DistributedGradScaler(init_scale=4096.0, growth_interval=2)

    overflow_on = [0, 2, 2, 1, 1]  # which rank sees the inf each step (2 = nobody)
    for bad_rank in overflow_on:
        for p in model.parameters():
            p.grad = torch.ones_like(p)
        if rank == bad_rank:
            next(iter(model.parameters())).grad[0, 0] = float("nan")
        scaler.step(optimizer)
        scaler.update()

    scale = torch.tensor([scaler.scale])
    gathered = [torch.empty_like(scale) for _ in range(world)]
    dist.all_gather(gathered, scale)
    assert all(torch.equal(gathered[0], g) for g in gathered), (
        f"[rank {rank}] ranks disagree on the loss scale: {[g.item() for g in gathered]}"
    )


def test_overflow_on_one_rank_skips_everywhere():
    run_distributed(_one_rank_overflow_body, 2)


def test_clean_steps_grow_the_scale():
    run_distributed(_clean_step_body, 2)


def test_unscale_is_idempotent():
    run_distributed(_unscale_body, 2)


def test_scale_stays_in_sync_across_ranks():
    run_distributed(_scale_agreement_body, 3)


def test_resolve_precision():
    assert not resolve_precision("fp32", torch.device("cpu")).enabled
    bf16 = resolve_precision("bf16", torch.device("cpu"))
    assert bf16.dtype == torch.bfloat16
    assert not bf16.needs_grad_scaler, "bf16 has fp32's exponent range; no scaler needed"
    assert resolve_precision("auto", torch.device("cpu")).dtype == torch.bfloat16
    with pytest.raises(ValueError):
        resolve_precision("fp16", torch.device("cpu"))
    with pytest.raises(ValueError):
        resolve_precision("int4", torch.device("cpu"))


def test_disabled_scaler_is_transparent():
    model, optimizer = _model_and_optimizer()
    scaler = DistributedGradScaler(enabled=False)
    loss = torch.tensor(2.0)
    assert scaler.scale_loss(loss) is loss
    assert scaler.scale == 1.0
    for p in model.parameters():
        p.grad = torch.ones_like(p)
    assert scaler.step(optimizer)
