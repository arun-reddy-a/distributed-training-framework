"""Tensor parallelism must reproduce the unsharded computation exactly."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from minidist.models import GPT, GPTConfig
from minidist.tensor_parallel import (
    ColumnParallelLinear,
    RowParallelLinear,
    VocabParallelEmbedding,
    full_embedding_weight,
    full_linear_weight,
    gather_from_tensor_parallel,
    vocab_parallel_cross_entropy,
)

from .common import all_ranks_agree, assert_close, run_distributed, solo_group

CONFIG = GPTConfig(vocab_size=64, block_size=16, n_layer=2, n_head=4, n_embd=32)


# -- individual layers ------------------------------------------------------


def _column_parallel_body(rank, world):
    torch.manual_seed(0)
    in_f, out_f, seed = 16, 32, 7
    layer = ColumnParallelLinear(in_f, out_f, bias=True, seed=seed)
    x = torch.randn(4, in_f, requires_grad=True)

    y_shard = layer(x)
    y = gather_from_tensor_parallel(y_shard, None)

    full_w = full_linear_weight(out_f, in_f, seed)
    expected = F.linear(x, full_w)  # bias initialised to zero
    assert_close(y, expected, f"[rank {rank}] column-parallel forward")

    # Backward: the gradient w.r.t. the replicated input must be the full
    # gradient, which only happens if `f` all-reduces the partials.
    y.sum().backward()
    expected_grad = full_w.sum(dim=0).expand(4, in_f)
    assert_close(x.grad, expected_grad, f"[rank {rank}] column-parallel input grad", tol=1e-4)


def _row_parallel_body(rank, world):
    torch.manual_seed(0)
    in_f, out_f, seed = 32, 16, 11
    layer = RowParallelLinear(in_f, out_f, bias=True, input_is_parallel=False, seed=seed)
    x = torch.randn(4, in_f)
    y = layer(x)

    full_w = full_linear_weight(out_f, in_f, seed)
    assert_close(y, F.linear(x, full_w), f"[rank {rank}] row-parallel forward")


def _vocab_embedding_body(rank, world):
    torch.manual_seed(0)
    vocab, dim, seed = 32, 8, 3
    emb = VocabParallelEmbedding(vocab, dim, seed=seed)
    ids = torch.randint(0, vocab, (3, 5))
    assert_close(
        emb(ids),
        F.embedding(ids, full_embedding_weight(vocab, dim, seed)),
        f"[rank {rank}] vocab-parallel embedding",
    )


def _vocab_cross_entropy_body(rank, world):
    """The memory-saving loss must match `F.cross_entropy` in value and gradient."""
    torch.manual_seed(0)
    n_tokens, vocab = 12, 32
    shard = vocab // world
    vocab_start = rank * shard

    full_logits = torch.randn(n_tokens, vocab, generator=torch.Generator().manual_seed(5))
    targets = torch.randint(0, vocab, (n_tokens,), generator=torch.Generator().manual_seed(6))

    local = full_logits[:, vocab_start : vocab_start + shard].clone().requires_grad_(True)
    loss = vocab_parallel_cross_entropy(local, targets, group=None, vocab_start=vocab_start)

    reference_input = full_logits.clone().requires_grad_(True)
    reference = F.cross_entropy(reference_input, targets)
    assert_close(loss, reference, f"[rank {rank}] vocab-parallel CE value")

    loss.backward()
    reference.backward()
    assert_close(
        local.grad,
        reference_input.grad[:, vocab_start : vocab_start + shard],
        f"[rank {rank}] vocab-parallel CE gradient",
    )


# -- whole model ------------------------------------------------------------


def _gpt_matches_reference_body(rank, world):
    """A tp=world model must equal a tp=1 model built from the same seeds."""
    solo = solo_group(rank, world)
    torch.manual_seed(0)

    sharded = GPT(CONFIG, tp_group=None)
    reference = GPT(CONFIG, tp_group=solo)

    ids = torch.randint(0, CONFIG.vocab_size, (2, CONFIG.block_size))
    targets = torch.randint(0, CONFIG.vocab_size, (2, CONFIG.block_size))

    _, loss = sharded(ids, targets)
    _, ref_loss = reference(ids, targets)
    assert_close(loss, ref_loss, f"[rank {rank}] tp={world} loss vs tp=1", tol=1e-5)

    loss.backward()
    ref_loss.backward()

    # Spot-check a sharded weight: this rank's slice of the reference's
    # gradient must equal the gradient it computed locally.
    ref_qkv = reference.blocks[0].attn.qkv.weight.grad
    got_qkv = sharded.blocks[0].attn.qkv.weight.grad
    expected = ref_qkv.chunk(world, dim=0)[rank]
    assert_close(got_qkv, expected, f"[rank {rank}] column-parallel qkv grad shard", tol=1e-4)

    ref_proj = reference.blocks[0].attn.proj.weight.grad
    got_proj = sharded.blocks[0].attn.proj.weight.grad
    assert_close(
        got_proj,
        ref_proj.chunk(world, dim=1)[rank],
        f"[rank {rank}] row-parallel proj grad shard",
        tol=1e-4,
    )


def _replicated_grads_agree_body(rank, world):
    """Replicated parameters must receive identical gradients on every rank.

    LayerNorm weights and row-parallel biases are not sharded, so if their
    gradients differed the replicas would silently drift apart.  They agree
    without any extra collective, purely because of where `f` and `g` sit — a
    property that is easy to break by moving a collective, hence this test.
    """
    torch.manual_seed(0)
    model = GPT(CONFIG, tp_group=None)
    ids = torch.randint(0, CONFIG.vocab_size, (2, CONFIG.block_size))
    targets = torch.randint(0, CONFIG.vocab_size, (2, CONFIG.block_size))
    _, loss = model(ids, targets)
    loss.backward()

    replicated = {
        "blocks.0.ln1.weight": model.blocks[0].ln1.weight,
        "blocks.0.ln2.bias": model.blocks[0].ln2.bias,
        "lm_head.ln_f.weight": model.lm_head.ln_f.weight,
        "blocks.0.attn.proj.bias": model.blocks[0].attn.proj.bias,
        "blocks.0.mlp.proj.bias": model.blocks[0].mlp.proj.bias,
    }
    for name, param in replicated.items():
        assert param.grad is not None, f"{name} has no gradient"
        assert all_ranks_agree(param.grad), (
            f"[rank {rank}] replicated parameter {name} has a rank-dependent "
            "gradient; tensor-parallel replicas would diverge"
        )


# -- test entry points ------------------------------------------------------


@pytest.mark.parametrize("world", [2, 4])
def test_column_parallel_linear(world):
    run_distributed(_column_parallel_body, world)


@pytest.mark.parametrize("world", [2, 4])
def test_row_parallel_linear(world):
    run_distributed(_row_parallel_body, world)


def test_vocab_parallel_embedding():
    run_distributed(_vocab_embedding_body, 2)


@pytest.mark.parametrize("world", [2, 4])
def test_vocab_parallel_cross_entropy(world):
    run_distributed(_vocab_cross_entropy_body, world)


@pytest.mark.parametrize("world", [2, 4])
def test_gpt_tensor_parallel_matches_single_device(world):
    run_distributed(_gpt_matches_reference_body, world)


def test_replicated_parameters_get_identical_gradients():
    run_distributed(_replicated_grads_agree_body, 2)
