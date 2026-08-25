"""FSDP must train to the same weights as DDP, while holding a fraction of the state.

Sharding changes *where* values live, never *what* they are.  The strongest
statement of that is an end-to-end one: run the same data through FSDP and DDP
with the same seeds and optimizer, and require the weights to still agree after
several optimizer steps.  A single forward would not catch a stale-gather or a
misordered reduce-scatter; accumulated drift over steps does.
"""

from __future__ import annotations

import pytest
import torch

from minidist.ddp import DistributedDataParallel
from minidist.fsdp import FullyShardedDataParallel, MixedPrecisionPolicy
from minidist.models import GPT, Block, GPTConfig

from .common import assert_close, run_distributed, solo_group

CONFIG = GPTConfig(vocab_size=64, block_size=16, n_layer=3, n_head=4, n_embd=32)
BATCH, STEPS = 8, 4


def _batches(world: int, steps: int):
    g = torch.Generator().manual_seed(1234)
    for _ in range(steps):
        ids = torch.randint(0, CONFIG.vocab_size, (BATCH, CONFIG.block_size), generator=g)
        targets = torch.randint(0, CONFIG.vocab_size, (BATCH, CONFIG.block_size), generator=g)
        yield ids, targets


def _gather_full_grads(fsdp_model, world: int) -> dict[str, torch.Tensor]:
    """Reassemble each unit's sharded gradient into the full parameter shapes."""
    import torch.distributed as dist

    id_to_name = {id(p): n for n, p in fsdp_model.module.named_parameters()}
    out: dict[str, torch.Tensor] = {}
    for unit in fsdp_model._units:
        shard = unit.flat_shard.grad
        parts = [torch.empty_like(shard) for _ in range(world)]
        dist.all_gather(parts, shard.contiguous())
        flat = torch.cat(parts)[: unit.total_numel]
        offset = 0
        for p, numel, shape in zip(unit.params, unit.numels, unit.shapes, strict=True):
            out[id_to_name[id(p)]] = flat[offset : offset + numel].view(shape).clone()
            offset += numel
    return out


def _train(model, rank, world, steps=STEPS, lr=1e-2, optimizer_cls=torch.optim.SGD):
    optimizer = optimizer_cls(model.parameters(), lr=lr)
    losses = []
    for ids, targets in _batches(world, steps):
        my_ids = ids.chunk(world, dim=0)[rank]
        my_targets = targets.chunk(world, dim=0)[rank]
        optimizer.zero_grad(set_to_none=True)
        _, loss = model(my_ids, my_targets)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return losses


def _fsdp_grads_match_ddp_body(rank, world):
    """The sharded gradient, reassembled, must equal DDP's gradient.

    This is the sharpest available statement of FSDP correctness: it compares
    the quantity FSDP actually computes, before any optimizer can blur it.
    """
    solo = solo_group(rank, world)
    ids, targets = next(_batches(world, 1))
    my_ids, my_targets = ids.chunk(world, 0)[rank], targets.chunk(world, 0)[rank]

    torch.manual_seed(0)
    ddp_model = DistributedDataParallel(GPT(CONFIG, tp_group=solo))
    _, ddp_loss = ddp_model(my_ids, my_targets)
    ddp_loss.backward()

    torch.manual_seed(0)
    fsdp_model = FullyShardedDataParallel(GPT(CONFIG, tp_group=solo), unit_types=(Block,))
    _, fsdp_loss = fsdp_model(my_ids, my_targets)
    fsdp_loss.backward()

    assert_close(fsdp_loss, ddp_loss, f"[rank {rank}] loss", tol=1e-6)

    got = _gather_full_grads(fsdp_model, world)
    for name, expected in ddp_model.module.named_parameters():
        assert_close(got[name], expected.grad, f"[rank {rank}] gradient for {name}", tol=1e-6)


def _fsdp_matches_ddp_body(rank, world):
    """End-to-end: identical weights after several optimizer steps.

    Uses SGD rather than Adam deliberately.  FSDP reduce-scatters where DDP
    all-reduces, so the two sum the same values in a different order and their
    gradients differ by ~5e-8 relative — pure fp32 rounding.  Adam's update is
    ``m/(sqrt(v) + eps)``, which is scale-invariant and therefore *amplifies*
    that relative noise instead of shrinking it; after a few steps it reaches
    1e-3 on parameters with small gradients, which says nothing about
    correctness.  SGD's update is linear in the gradient, so the comparison
    stays meaningful.  ``test_fsdp_gradients_match_ddp`` is the tight check.
    """
    solo = solo_group(rank, world)

    torch.manual_seed(0)
    ddp_model = DistributedDataParallel(GPT(CONFIG, tp_group=solo))
    ddp_losses = _train(ddp_model, rank, world)

    torch.manual_seed(0)
    fsdp_model = FullyShardedDataParallel(
        GPT(CONFIG, tp_group=solo), unit_types=(Block,)
    )
    fsdp_losses = _train(fsdp_model, rank, world)

    for step, (a, b) in enumerate(zip(ddp_losses, fsdp_losses, strict=True)):
        assert abs(a - b) < 1e-5, f"[rank {rank}] step {step} loss {b:.6f} != DDP {a:.6f}"

    # Compare the actual weights, not just the loss: a bug that leaves one unit
    # stale can still produce a plausible loss curve.
    with fsdp_model.summon_full_params():
        fsdp_params = dict(fsdp_model.module.named_parameters())
        for name, expected in ddp_model.module.named_parameters():
            assert_close(
                fsdp_params[name], expected, f"[rank {rank}] weight {name} after {STEPS} steps",
                tol=1e-5,
            )


def _sharding_is_real_body(rank, world):
    """Each rank must hold roughly 1/world of the parameters, not all of them."""
    solo = solo_group(rank, world)
    model = FullyShardedDataParallel(GPT(CONFIG, tp_group=solo), unit_types=(Block,))

    reference = GPT(CONFIG, tp_group=solo)
    full_numel = sum(p.numel() for p in reference.parameters())
    local_numel = sum(p.numel() for p in model.parameters())

    assert local_numel < full_numel, "FSDP is not sharding anything"
    # Padding to a multiple of `world` per unit adds a little overhead.
    upper = full_numel / world * 1.25
    assert local_numel <= upper, (
        f"[rank {rank}] holds {local_numel} params, expected about "
        f"{full_numel / world:.0f} (<= {upper:.0f})"
    )

    summary = model.memory_summary()
    assert summary["num_units"] == CONFIG.n_layer + 1, "expected one unit per block plus root"
    assert summary["fsdp_state_MB"] < summary["ddp_state_MB"]


def _gathered_params_are_freed_body(rank, world):
    """Between steps, the gathered buffers must actually be released."""
    solo = solo_group(rank, world)
    model = FullyShardedDataParallel(GPT(CONFIG, tp_group=solo), unit_types=(Block,))
    ids, targets = next(_batches(world, 1))
    _, loss = model(ids.chunk(world, dim=0)[rank], targets.chunk(world, dim=0)[rank])
    loss.backward()

    for unit in model._units:
        assert not unit.is_unsharded, f"[rank {rank}] unit {unit.name} still unsharded"
        assert unit._full.untyped_storage().nbytes() == 0, (
            f"[rank {rank}] unit {unit.name} still holds gathered parameter memory"
        )
        for p in unit.params:
            assert p.grad is None, (
                f"[rank {rank}] unit {unit.name} still holds a full-size gradient"
            )


def _no_reshard_after_forward_body(rank, world):
    """ZeRO-2 mode (keep parameters through backward) must give the same answer."""
    solo = solo_group(rank, world)

    torch.manual_seed(0)
    zero3 = FullyShardedDataParallel(GPT(CONFIG, tp_group=solo), unit_types=(Block,))
    losses3 = _train(zero3, rank, world)

    torch.manual_seed(0)
    zero2 = FullyShardedDataParallel(
        GPT(CONFIG, tp_group=solo), unit_types=(Block,), reshard_after_forward=False
    )
    losses2 = _train(zero2, rank, world)

    for step, (a, b) in enumerate(zip(losses3, losses2, strict=True)):
        assert abs(a - b) < 1e-6, f"[rank {rank}] step {step}: ZeRO-2 {b} != ZeRO-3 {a}"


def _mixed_precision_body(rank, world):
    """bf16 parameters with fp32 reductions should track the fp32 run closely."""
    solo = solo_group(rank, world)

    torch.manual_seed(0)
    fp32 = FullyShardedDataParallel(GPT(CONFIG, tp_group=solo), unit_types=(Block,))
    fp32_losses = _train(fp32, rank, world, steps=2)

    torch.manual_seed(0)
    bf16 = FullyShardedDataParallel(
        GPT(CONFIG, tp_group=solo),
        unit_types=(Block,),
        mixed_precision=MixedPrecisionPolicy(
            param_dtype=torch.bfloat16, reduce_dtype=torch.float32
        ),
    )
    bf16_losses = _train(bf16, rank, world, steps=2)

    # The master shard stays fp32, so the optimizer still updates in full
    # precision; only the matmuls are bf16.
    assert bf16._units[0].flat_shard.dtype == torch.float32
    for step, (a, b) in enumerate(zip(fp32_losses, bf16_losses, strict=True)):
        assert abs(a - b) < 0.05, f"[rank {rank}] step {step}: bf16 {b:.4f} vs fp32 {a:.4f}"


def _no_sync_body(rank, world):
    """Accumulating under `no_sync` then flushing equals one full-batch step."""
    solo = solo_group(rank, world)
    ids, targets = next(_batches(world, 1))
    my_ids = ids.chunk(world, dim=0)[rank]
    my_targets = targets.chunk(world, dim=0)[rank]

    torch.manual_seed(0)
    single = FullyShardedDataParallel(GPT(CONFIG, tp_group=solo), unit_types=(Block,))
    _, loss = single(my_ids, my_targets)
    loss.backward()
    expected = [p.grad.clone() for p in single.parameters()]

    torch.manual_seed(0)
    accum = FullyShardedDataParallel(GPT(CONFIG, tp_group=solo), unit_types=(Block,))
    halves = list(zip(my_ids.chunk(2, dim=0), my_targets.chunk(2, dim=0), strict=True))
    with accum.no_sync():
        _, loss = accum(*halves[0])
        (loss / 2).backward()
        _, loss = accum(*halves[1])
        (loss / 2).backward()
    accum.finish_gradient_sync()

    for i, (got, want) in enumerate(zip(accum.parameters(), expected, strict=True)):
        assert_close(got.grad, want, f"[rank {rank}] accumulated shard grad {i}", tol=2e-5)


@pytest.mark.parametrize("world", [2, 4])
def test_fsdp_gradients_match_ddp(world):
    run_distributed(_fsdp_grads_match_ddp_body, world)


@pytest.mark.parametrize("world", [2, 4])
def test_fsdp_matches_ddp(world):
    run_distributed(_fsdp_matches_ddp_body, world)


@pytest.mark.parametrize("world", [2, 4])
def test_fsdp_actually_shards(world):
    run_distributed(_sharding_is_real_body, world)


def test_fsdp_frees_gathered_parameters():
    run_distributed(_gathered_params_are_freed_body, 2)


def test_fsdp_reshard_after_forward_is_transparent():
    run_distributed(_no_reshard_after_forward_body, 2)


def test_fsdp_mixed_precision():
    run_distributed(_mixed_precision_body, 2)


def test_fsdp_gradient_accumulation():
    run_distributed(_no_sync_body, 2)
