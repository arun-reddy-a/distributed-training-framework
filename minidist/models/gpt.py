"""A small GPT that can be sharded by tensor parallelism and split by pipeline parallelism.

The model is deliberately plain — this repo is about the parallelism, not the
architecture — but two properties are load-bearing and worth stating:

**Tensor-parallel invariance.**  Every parallel layer materialises its full
weight from a fixed seed and slices out its shard
(:mod:`minidist.tensor_parallel`), so a model built with ``tp_size=4`` has
bit-identical full weights to one built with ``tp_size=1``.  That is what makes
"tensor parallelism produces the same loss as the single-device reference" a
testable claim rather than a hope.

**Replicated parameters need no extra all-reduce.**  LayerNorm weights and the
row-parallel biases are replicated across the TP group, so their gradients must
agree on every rank or the replicas silently diverge.  They do agree, for free,
as a consequence of where ``f`` and ``g`` sit: the gradient arriving at any
replicated parameter has already passed backward through an ``f`` (which
all-reduces), so it is identical everywhere before it is ever used.  This is an
easy property to break by moving a collective, so ``tests/test_tensor_parallel.py``
asserts it explicitly.

Dropout is 0 by default.  Dropout *inside* a tensor-parallel region needs
rank-divergent RNG (each rank drops different elements of its own shard) while
dropout outside needs rank-identical RNG, and getting that wrong is invisible
in the loss curve.  Rather than ship a half-solution, it is off, and the
correctness tests stay meaningful.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from .. import comm
from ..tensor_parallel import (
    ColumnParallelLinear,
    RowParallelLinear,
    VocabParallelEmbedding,
    vocab_parallel_cross_entropy,
)

__all__ = ["GPTConfig", "GPT", "Block", "build_pipeline_layers"]


@dataclass
class GPTConfig:
    vocab_size: int = 1024
    block_size: int = 128
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 128
    dropout: float = 0.0
    bias: bool = True
    seed: int = 1234

    @property
    def head_dim(self) -> int:
        return self.n_embd // self.n_head

    def param_count(self) -> int:
        """Analytical parameter count, used by the benchmarks' memory model."""
        n = self.n_embd
        emb = self.vocab_size * n + self.block_size * n
        # attn: qkv 3n^2 + proj n^2 ; mlp: fc 4n^2 + proj 4n^2
        per_block = 12 * n * n
        per_block += 4 * n  # two LayerNorms (weight + bias)
        if self.bias:
            per_block += 9 * n  # qkv 3n, attn-proj n, fc 4n, mlp-proj n
        head = n * self.vocab_size + 2 * n  # projection + final LayerNorm
        return emb + self.n_layer * per_block + head


def _seed_for(config: GPTConfig, layer: int, role: int) -> int:
    return (config.seed * 1_000_003 + layer * 97 + role) % (2**31 - 1)


class CausalSelfAttention(nn.Module):
    """Multi-head attention with heads distributed across the tensor-parallel group.

    Heads are the natural sharding unit: attention mixes information *within* a
    head (across sequence positions) but never *between* heads until the output
    projection.  Giving each rank whole heads therefore keeps softmax entirely
    local — no collective inside the attention computation itself.

    QKV is column-parallel (output features = heads) and the output projection
    is row-parallel (input features = heads), so the pair composes into exactly
    one all-reduce, at the very end.

    The QKV weight is laid out as ``[n_head, 3, head_dim]`` — q, k and v
    *interleaved per head* — rather than the more familiar ``[q | k | v]``
    concatenation.  This matters: a column-parallel split takes a contiguous
    slice of the output dimension, and with the ``[q | k | v]`` layout at
    ``tp=2`` rank 0 would receive all of q plus half of k while rank 1 received
    half of k plus all of v.  Interleaving per head makes any contiguous slice
    a set of *whole heads*, each with its own q, k and v.
    """

    def __init__(self, config: GPTConfig, layer: int, tp_group=None, device=None, dtype=None):
        super().__init__()
        tp = comm.get_world_size(tp_group)
        if config.n_head % tp:
            raise ValueError(f"n_head={config.n_head} not divisible by tp_size={tp}")
        self.n_head_local = config.n_head // tp
        self.head_dim = config.head_dim

        self.qkv = ColumnParallelLinear(
            config.n_embd, 3 * config.n_embd, bias=config.bias,
            group=tp_group, seed=_seed_for(config, layer, 0), device=device, dtype=dtype,
        )
        self.proj = RowParallelLinear(
            config.n_embd, config.n_embd, bias=config.bias, input_is_parallel=True,
            group=tp_group, seed=_seed_for(config, layer, 1), device=device, dtype=dtype,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, _ = x.shape
        qkv = self.qkv(x)  # [B, S, 3 * n_embd / tp]
        # Per-head interleaved layout: [local_heads, 3, head_dim].  See the
        # class docstring for why the q/k/v axis is *inside* the head axis.
        qkv = qkv.view(B, S, self.n_head_local, 3, self.head_dim)
        q, k, v = qkv.unbind(dim=3)
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))  # [B, h, S, d]
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, S, self.n_head_local * self.head_dim)
        return self.proj(y)


class MLP(nn.Module):
    """Column-parallel expand, elementwise GELU, row-parallel contract.

    See :mod:`minidist.tensor_parallel` for why this particular pairing costs
    one all-reduce instead of two.
    """

    def __init__(self, config: GPTConfig, layer: int, tp_group=None, device=None, dtype=None):
        super().__init__()
        self.fc = ColumnParallelLinear(
            config.n_embd, 4 * config.n_embd, bias=config.bias,
            group=tp_group, seed=_seed_for(config, layer, 2), device=device, dtype=dtype,
        )
        self.proj = RowParallelLinear(
            4 * config.n_embd, config.n_embd, bias=config.bias, input_is_parallel=True,
            group=tp_group, seed=_seed_for(config, layer, 3), device=device, dtype=dtype,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(F.gelu(self.fc(x), approximate="tanh"))


class Block(nn.Module):
    """Pre-norm transformer block — also the default FSDP wrapping unit."""

    def __init__(self, config: GPTConfig, layer: int, tp_group=None, device=None, dtype=None):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd, device=device, dtype=dtype)
        self.attn = CausalSelfAttention(config, layer, tp_group, device, dtype)
        self.ln2 = nn.LayerNorm(config.n_embd, device=device, dtype=dtype)
        self.mlp = MLP(config, layer, tp_group, device, dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class Embedding(nn.Module):
    """Token + learned positional embedding; the pipeline's first stage."""

    def __init__(self, config: GPTConfig, tp_group=None, device=None, dtype=None):
        super().__init__()
        self.wte = VocabParallelEmbedding(
            config.vocab_size, config.n_embd, group=tp_group,
            seed=_seed_for(config, -1, 0), device=device, dtype=dtype,
        )
        generator = torch.Generator(device="cpu").manual_seed(_seed_for(config, -1, 1))
        pos = torch.empty(config.block_size, config.n_embd)
        pos.normal_(0.0, 0.02, generator=generator)
        self.wpe = nn.Parameter(pos.to(device=device, dtype=dtype))

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        S = input_ids.size(1)
        return self.wte(input_ids) + self.wpe[:S]


class LMHead(nn.Module):
    """Final norm + column-parallel projection to the (sharded) vocabulary.

    ``gather_output=False`` keeps the logits sharded so they can be consumed by
    :func:`~minidist.tensor_parallel.vocab_parallel_cross_entropy`, which never
    materialises the full ``[batch, seq, vocab]`` tensor.  For a 128k vocab
    that tensor is usually the single largest allocation in the step.
    """

    def __init__(self, config: GPTConfig, tp_group=None, device=None, dtype=None):
        super().__init__()
        self.ln_f = nn.LayerNorm(config.n_embd, device=device, dtype=dtype)
        self.head = ColumnParallelLinear(
            config.n_embd, config.vocab_size, bias=False, gather_output=False,
            group=tp_group, seed=_seed_for(config, -1, 2), device=device, dtype=dtype,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.ln_f(x))


class GPT(nn.Module):
    """The whole model, optionally tensor-parallel.

    ``tp_group=None`` (or a group of size 1) gives an ordinary single-device
    model — the same class serves as its own reference implementation.
    """

    def __init__(self, config: GPTConfig, tp_group=None, device=None, dtype=None):
        super().__init__()
        self.config = config
        self.tp_group = tp_group
        self.embedding = Embedding(config, tp_group, device, dtype)
        self.blocks = nn.ModuleList(
            [Block(config, i, tp_group, device, dtype) for i in range(config.n_layer)]
        )
        self.lm_head = LMHead(config, tp_group, device, dtype)

    def forward(self, input_ids: torch.Tensor, targets: torch.Tensor | None = None):
        x = self.embedding(input_ids)
        for block in self.blocks:
            x = block(x)
        logits = self.lm_head(x)
        if targets is None:
            return logits
        return logits, self.loss(logits, targets)

    def loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return vocab_parallel_cross_entropy(
            logits, targets, group=self.tp_group, vocab_start=self.vocab_start
        )

    @property
    def vocab_start(self) -> int:
        tp = comm.get_world_size(self.tp_group)
        return comm.get_rank(self.tp_group) * (self.config.vocab_size // tp)

    def num_parameters(self, local: bool = True) -> int:
        n = sum(p.numel() for p in self.parameters())
        return n if local else n * comm.get_world_size(self.tp_group)


def build_pipeline_layers(
    config: GPTConfig, tp_group=None, device=None, dtype=None
) -> list[nn.Module]:
    """Flatten the model into the sequence of modules a pipeline splits over.

    ``[Embedding, Block x n_layer, LMHead]``.  Uniform partitioning of this list
    gives an unbalanced pipeline whenever the embedding or head is expensive
    relative to a block (a large vocabulary makes both heavy), which the
    per-stage busy times in the pipeline benchmark make visible.
    """
    layers: list[nn.Module] = [Embedding(config, tp_group, device, dtype)]
    layers += [Block(config, i, tp_group, device, dtype) for i in range(config.n_layer)]
    layers.append(LMHead(config, tp_group, device, dtype))
    return layers
