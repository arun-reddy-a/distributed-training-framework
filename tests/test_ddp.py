"""DDP must produce exactly the gradient of the full batch, however it buckets."""

from __future__ import annotations

import pytest
import torch

from minidist.ddp import DistributedDataParallel
from minidist.models import GPT, GPTConfig

from .common import assert_close, run_distributed, solo_group

CONFIG = GPTConfig(vocab_size=64, block_size=16, n_layer=2, n_head=4, n_embd=32)
BATCH = 8


def _make_batch(world: int):
    g = torch.Generator().manual_seed(42)
    ids = torch.randint(0, CONFIG.vocab_size, (BATCH, CONFIG.block_size), generator=g)
    targets = torch.randint(0, CONFIG.vocab_size, (BATCH, CONFIG.block_size), generator=g)
    return ids, targets


def _reference_grads(solo, ids, targets) -> dict[str, torch.Tensor]:
    """Gradient of the mean loss over the whole batch, computed in one process."""
    model = GPT(CONFIG, tp_group=solo)
    _, loss = model(ids, targets)
    loss.backward()
    return {n: p.grad.clone() for n, p in model.named_parameters()}


def _compare_to_reference(rank, world, bucket_cap_mb, overlap):
    solo = solo_group(rank, world)
    ids, targets = _make_batch(world)
    expected = _reference_grads(solo, ids, targets)

    model = DistributedDataParallel(
        GPT(CONFIG, tp_group=solo),
        bucket_cap_mb=bucket_cap_mb,
        overlap_with_backward=overlap,
    )
    my_ids = ids.chunk(world, dim=0)[rank]
    my_targets = targets.chunk(world, dim=0)[rank]

    _, loss = model(my_ids, my_targets)
    loss.backward()

    for name, param in model.module.named_parameters():
        assert param.grad is not None, f"{name} received no gradient"
        assert_close(
            param.grad,
            expected[name],
            f"[rank {rank}] cap={bucket_cap_mb}MB overlap={overlap} grad for {name}",
            tol=2e-5,
        )


def _default_body(rank, world):
    _compare_to_reference(rank, world, bucket_cap_mb=25.0, overlap=True)


def _tiny_buckets_body(rank, world):
    """A bucket cap below one parameter's size forces one bucket per tensor.

    This is the configuration most likely to expose an ordering bug, because
    every parameter now triggers its own collective and any divergence in
    launch order between ranks would mismatch immediately.
    """
    _compare_to_reference(rank, world, bucket_cap_mb=1e-6, overlap=True)


def _one_bucket_body(rank, world):
    _compare_to_reference(rank, world, bucket_cap_mb=1e6, overlap=True)


def _no_overlap_body(rank, world):
    _compare_to_reference(rank, world, bucket_cap_mb=25.0, overlap=False)


def _no_sync_accumulation_body(rank, world):
    """Two microbatches under `no_sync` then one synced step == one full step."""
    solo = solo_group(rank, world)
    ids, targets = _make_batch(world)
    expected = _reference_grads(solo, ids, targets)

    model = DistributedDataParallel(GPT(CONFIG, tp_group=solo))
    my_ids = ids.chunk(world, dim=0)[rank]
    my_targets = targets.chunk(world, dim=0)[rank]
    halves = list(zip(my_ids.chunk(2, dim=0), my_targets.chunk(2, dim=0), strict=True))

    with model.no_sync():
        _, loss = model(*halves[0])
        (loss / 2).backward()
    _, loss = model(*halves[1])
    (loss / 2).backward()

    for name, param in model.module.named_parameters():
        assert_close(
            param.grad, expected[name], f"[rank {rank}] accumulated grad for {name}", tol=2e-5
        )


def _bucket_layout_body(rank, world):
    """Buckets are built in reverse parameter order and respect the cap."""
    solo = solo_group(rank, world)
    cap_mb = 0.05
    model = DistributedDataParallel(GPT(CONFIG, tp_group=solo), bucket_cap_mb=cap_mb)
    summary = model.bucket_summary()

    assert len(summary) > 1, "test config should produce several buckets"
    assert [b[0] for b in summary] == list(range(len(summary))), "buckets must be indexed in order"
    # Every bucket except one holding a single oversized tensor stays under cap.
    for index, n_params, mb in summary:
        assert n_params >= 1
        assert mb <= cap_mb or n_params == 1, f"bucket {index} is {mb:.3f}MB over a {cap_mb}MB cap"

    first_bucket_params = model._buckets[0].params
    all_params = [p for p in model.module.parameters() if p.requires_grad]
    assert first_bucket_params[0] is all_params[-1], (
        "bucket 0 must start from the last parameter so it fills first during backward"
    )


@pytest.mark.parametrize("world", [2, 4])
def test_ddp_matches_full_batch_gradient(world):
    run_distributed(_default_body, world)


def test_ddp_with_one_bucket_per_parameter():
    run_distributed(_tiny_buckets_body, 2)


def test_ddp_with_a_single_bucket():
    run_distributed(_one_bucket_body, 2)


def test_ddp_without_backward_overlap():
    run_distributed(_no_overlap_body, 2)


def test_ddp_gradient_accumulation_under_no_sync():
    run_distributed(_no_sync_accumulation_body, 2)


def test_ddp_bucket_layout():
    run_distributed(_bucket_layout_body, 2)
