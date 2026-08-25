r"""Tensor (intra-layer) parallelism — Megatron-style row/column sharded layers.

Tensor parallelism splits individual weight matrices across ranks so that a
layer too large for one device still fits.  The art is choosing *which* axis to
split so that consecutive layers compose without a collective between them.

The MLP block, ``y = W2 · GELU(W1 · x)``, is the canonical example.  Shard
``W1`` by **columns** (output features):

.. code-block:: text

    W1 = [W1_0 | W1_1]      =>   GELU(x·W1) = [GELU(x·W1_0) | GELU(x·W1_1)]

GELU is elementwise, so it commutes with the column split — no communication
needed.  Now shard ``W2`` by **rows** (input features), matching the way its
input arrives already split:

.. code-block:: text

    W2 = [W2_0 ]            =>   y = (h_0 · W2_0) + (h_1 · W2_1)
         [W2_1 ]                      \_ local _/   \_ all-reduce _/

The whole block costs exactly one all-reduce in forward.  Had we sharded
``W1`` by rows instead, we would need a collective *between* the two matmuls
and another after — twice the traffic on the critical path.

The two conjugate operators
---------------------------
``f`` (:class:`_CopyToTensorParallel`) is identity forward, all-reduce
backward.  ``g`` (:class:`_ReduceFromTensorParallel`) is all-reduce forward,
identity backward.  They bracket every parallel region, and their asymmetry is
the reason a TP block costs one all-reduce each way rather than two.

Attention shards the same way with heads as the natural unit: QKV projections
are column-parallel (each rank owns whole heads, so softmax stays local), the
output projection is row-parallel.

Initialisation
--------------
Every parallel layer materialises the *full* weight under a fixed seed and then
slices out its shard.  This costs one transient full-size allocation at startup
and buys an important property: a model built with ``tp_size=4`` is
bit-identical to the same model built with ``tp_size=1``.  Without it, "does TP
match the single-device reference?" is not even a well-posed question.
"""

from __future__ import annotations

import math

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn

from . import comm

__all__ = [
    "ColumnParallelLinear",
    "RowParallelLinear",
    "VocabParallelEmbedding",
    "vocab_parallel_cross_entropy",
    "copy_to_tensor_parallel",
    "reduce_from_tensor_parallel",
    "gather_from_tensor_parallel",
    "scatter_to_tensor_parallel",
    "full_linear_weight",
    "full_embedding_weight",
]


# --------------------------------------------------------------------------
# Deterministic weight construction
# --------------------------------------------------------------------------
#
# Both the parallel layers and the correctness tests build the *full* weight
# through these two functions.  Having exactly one definition is what lets a
# test say "the tp=4 model must equal this reference" without restating the
# initialisation scheme and risking the two drifting apart.


def full_linear_weight(
    out_features: int, in_features: int, seed: int, dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """Kaiming-uniform over the *full* fan-in, matching ``nn.Linear``'s default."""
    generator = torch.Generator(device="cpu").manual_seed(seed)
    full = torch.empty((out_features, in_features), dtype=torch.float32)
    bound = 1.0 / math.sqrt(in_features)
    full.uniform_(-bound, bound, generator=generator)
    return full.to(dtype)


def full_embedding_weight(
    num_embeddings: int, dim: int, seed: int, dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    full = torch.empty((num_embeddings, dim), dtype=torch.float32)
    full.normal_(mean=0.0, std=0.02, generator=generator)
    return full.to(dtype)


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def _split_last_dim(t: torch.Tensor, world: int, rank: int) -> torch.Tensor:
    last = t.size(-1)
    if last % world:
        raise ValueError(f"cannot split dim of size {last} across {world} ranks")
    return t.split(last // world, dim=-1)[rank].contiguous()


def _gather_last_dim(t: torch.Tensor, group, world: int) -> torch.Tensor:
    if world == 1:
        return t
    parts = [torch.empty_like(t) for _ in range(world)]
    dist.all_gather(parts, t.contiguous(), group=group)
    comm.comm_stats.add(
        "all_gather",
        t.numel() * t.element_size() * world,
        int((world - 1) / world * t.numel() * t.element_size() * world),
    )
    return torch.cat(parts, dim=-1)


# --------------------------------------------------------------------------
# The f / g conjugate pair
# --------------------------------------------------------------------------


class _CopyToTensorParallel(torch.autograd.Function):
    """``f``: identity forward, all-reduce backward.

    Marks the entry to a parallel region.  The input is replicated, so forward
    does nothing; in backward each rank holds a partial gradient w.r.t. that
    replicated input and the partials must be summed.
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, group) -> torch.Tensor:
        ctx.group = group
        return x

    @staticmethod
    def backward(ctx, grad: torch.Tensor):
        if comm.get_world_size(ctx.group) > 1:
            grad = grad.clone()
            comm.all_reduce(grad, group=ctx.group)
        return grad, None


class _ReduceFromTensorParallel(torch.autograd.Function):
    """``g``: all-reduce forward, identity backward.

    Marks the exit from a parallel region.  Each rank holds a partial sum of
    the output; backward needs no communication because the gradient w.r.t. a
    summand is just the incoming gradient.
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, group) -> torch.Tensor:
        if comm.get_world_size(group) > 1:
            x = x.clone()
            comm.all_reduce(x, group=group)
        return x

    @staticmethod
    def backward(ctx, grad: torch.Tensor):
        return grad, None


class _GatherFromTensorParallel(torch.autograd.Function):
    """All-gather along the last dim forward, slice backward."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, group) -> torch.Tensor:
        ctx.group = group
        ctx.world = comm.get_world_size(group)
        ctx.rank = comm.get_rank(group)
        return _gather_last_dim(x, group, ctx.world)

    @staticmethod
    def backward(ctx, grad: torch.Tensor):
        return _split_last_dim(grad, ctx.world, ctx.rank), None


class _ScatterToTensorParallel(torch.autograd.Function):
    """Slice along the last dim forward, all-gather backward."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, group) -> torch.Tensor:
        ctx.group = group
        ctx.world = comm.get_world_size(group)
        return _split_last_dim(x, ctx.world, comm.get_rank(group))

    @staticmethod
    def backward(ctx, grad: torch.Tensor):
        return _gather_last_dim(grad.contiguous(), ctx.group, ctx.world), None


def copy_to_tensor_parallel(x, group=None):
    return _CopyToTensorParallel.apply(x, group)


def reduce_from_tensor_parallel(x, group=None):
    return _ReduceFromTensorParallel.apply(x, group)


def gather_from_tensor_parallel(x, group=None):
    return _GatherFromTensorParallel.apply(x, group)


def scatter_to_tensor_parallel(x, group=None):
    return _ScatterToTensorParallel.apply(x, group)


# --------------------------------------------------------------------------
# Parallel layers
# --------------------------------------------------------------------------


class _ParallelLinearBase(nn.Module):
    def __init__(self, group):
        super().__init__()
        self.group = group
        self.world = comm.get_world_size(group)
        self.rank = comm.get_rank(group)

    def _init_full_then_slice(
        self,
        full_shape: tuple[int, int],
        shard_dim: int,
        seed: int,
        device,
        dtype,
    ) -> torch.Tensor:
        """Build the full weight deterministically, keep only this rank's slice."""
        full = full_linear_weight(full_shape[0], full_shape[1], seed)
        shard = full.chunk(self.world, dim=shard_dim)[self.rank].contiguous()
        return shard.to(device=device, dtype=dtype)


class ColumnParallelLinear(_ParallelLinearBase):
    """``y = x A + b`` with ``A`` split column-wise: ``A = [A_0 | ... | A_{p-1}]``.

    The input is expected to be replicated across the group; the output is
    sharded along its last dimension unless ``gather_output=True``.

    Leaving the output sharded is almost always what you want — the following
    :class:`RowParallelLinear` consumes it in exactly that layout, which is how
    an MLP or attention block gets away with a single collective.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        gather_output: bool = False,
        group=None,
        seed: int = 0,
        device=None,
        dtype=None,
    ):
        super().__init__(group)
        if out_features % self.world:
            raise ValueError(
                f"out_features={out_features} not divisible by tp_size={self.world}"
            )
        self.in_features = in_features
        self.out_features = out_features
        self.out_features_per_partition = out_features // self.world
        self.gather_output = gather_output

        weight = self._init_full_then_slice(
            (out_features, in_features), shard_dim=0, seed=seed, device=device, dtype=dtype
        )
        self.weight = nn.Parameter(weight)
        if bias:
            self.bias = nn.Parameter(
                torch.zeros(self.out_features_per_partition, device=device, dtype=dtype)
            )
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = copy_to_tensor_parallel(x, self.group)  # f
        y = F.linear(x, self.weight, self.bias)
        if self.gather_output:
            y = gather_from_tensor_parallel(y, self.group)
        return y

    @torch.no_grad()
    def set_full_weight(self, weight: torch.Tensor, bias: torch.Tensor | None = None) -> None:
        """Load an unsharded ``[out, in]`` weight — used by the correctness tests."""
        self.weight.copy_(weight.chunk(self.world, dim=0)[self.rank])
        if bias is not None and self.bias is not None:
            self.bias.copy_(bias.chunk(self.world, dim=0)[self.rank])

    def extra_repr(self) -> str:
        return (
            f"in={self.in_features}, out={self.out_features} "
            f"(local {self.out_features_per_partition}), tp={self.world}"
        )


class RowParallelLinear(_ParallelLinearBase):
    """``y = x A + b`` with ``A`` split row-wise (along the input dimension).

    The input must already be sharded along its last dimension (the normal case
    when the previous layer was column-parallel); pass
    ``input_is_parallel=False`` to have a replicated input scattered first.

    The bias is replicated and added *after* the all-reduce.  Adding it before
    would sum it ``world_size`` times.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        input_is_parallel: bool = True,
        group=None,
        seed: int = 0,
        device=None,
        dtype=None,
    ):
        super().__init__(group)
        if in_features % self.world:
            raise ValueError(
                f"in_features={in_features} not divisible by tp_size={self.world}"
            )
        self.in_features = in_features
        self.out_features = out_features
        self.in_features_per_partition = in_features // self.world
        self.input_is_parallel = input_is_parallel

        weight = self._init_full_then_slice(
            (out_features, in_features), shard_dim=1, seed=seed, device=device, dtype=dtype
        )
        self.weight = nn.Parameter(weight)
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, device=device, dtype=dtype))
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.input_is_parallel:
            x = scatter_to_tensor_parallel(x, self.group)
        y = F.linear(x, self.weight)  # partial sum
        y = reduce_from_tensor_parallel(y, self.group)  # g
        if self.bias is not None:
            y = y + self.bias
        return y

    @torch.no_grad()
    def set_full_weight(self, weight: torch.Tensor, bias: torch.Tensor | None = None) -> None:
        self.weight.copy_(weight.chunk(self.world, dim=1)[self.rank])
        if bias is not None and self.bias is not None:
            self.bias.copy_(bias)

    def extra_repr(self) -> str:
        return (
            f"in={self.in_features} (local {self.in_features_per_partition}), "
            f"out={self.out_features}, tp={self.world}"
        )


class VocabParallelEmbedding(nn.Module):
    """Embedding with the vocabulary dimension sharded across the TP group.

    Vocabulary tables are the largest single tensor in most LLMs
    (``vocab x hidden``), so they are worth sharding even though the embedding
    lookup itself is cheap.  Each rank owns a contiguous id range, zeroes the
    rows for ids it does not own, and one all-reduce assembles the result.
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        group=None,
        seed: int = 0,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.group = group
        self.world = comm.get_world_size(group)
        self.rank = comm.get_rank(group)
        if num_embeddings % self.world:
            raise ValueError(
                f"vocab size {num_embeddings} not divisible by tp_size={self.world}"
            )
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.num_embeddings_per_partition = num_embeddings // self.world
        self.vocab_start = self.rank * self.num_embeddings_per_partition
        self.vocab_end = self.vocab_start + self.num_embeddings_per_partition

        full = full_embedding_weight(num_embeddings, embedding_dim, seed)
        shard = full[self.vocab_start : self.vocab_end].contiguous()
        self.weight = nn.Parameter(shard.to(device=device, dtype=dtype))

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        if self.world == 1:
            return F.embedding(input_ids, self.weight)
        mask = (input_ids < self.vocab_start) | (input_ids >= self.vocab_end)
        local_ids = (input_ids - self.vocab_start).clamp_(0, self.num_embeddings_per_partition - 1)
        out = F.embedding(local_ids, self.weight)
        out = out.masked_fill(mask.unsqueeze(-1), 0.0)
        return reduce_from_tensor_parallel(out, self.group)

    @torch.no_grad()
    def set_full_weight(self, weight: torch.Tensor) -> None:
        self.weight.copy_(weight[self.vocab_start : self.vocab_end])


# --------------------------------------------------------------------------
# Vocab-parallel cross entropy
# --------------------------------------------------------------------------


class _VocabParallelCrossEntropy(torch.autograd.Function):
    """Cross entropy over logits sharded along the vocabulary dimension.

    The obvious implementation all-gathers the logits and calls the usual
    softmax.  For a 128k vocabulary with ``b*s = 16k`` tokens that materialises
    an 8 GB fp32 tensor on every rank — routinely the peak-memory event of the
    whole step, and it moves ``(W-1)/W * b*s*V`` bytes.

    Instead, note that log-sum-exp decomposes over a partition of the vocabulary
    into three scalar-per-token reductions:

    * ``max`` over the shard, then all-reduce MAX  (numerical stabilisation)
    * ``sum(exp(.))`` over the shard, then all-reduce SUM
    * the target's own logit, contributed by whichever rank owns that id, then
      all-reduce SUM

    Traffic drops from ``O(b*s*V)`` to ``O(b*s)`` — a factor of the vocabulary
    size — and the full logit tensor never exists.  The backward is the standard
    ``softmax - onehot``, which is computable entirely from shard-local values
    once ``sum_exp`` is known.
    """

    @staticmethod
    def forward(ctx, logits: torch.Tensor, target: torch.Tensor, group, vocab_start: int):
        # logits: [N, V/tp] (already flattened), target: [N] global ids
        logits = logits.float()
        vocab_per_partition = logits.size(-1)
        vocab_end = vocab_start + vocab_per_partition

        logits_max = logits.max(dim=-1).values
        if comm.get_world_size(group) > 1:
            dist.all_reduce(logits_max, op=dist.ReduceOp.MAX, group=group)
        logits = logits - logits_max.unsqueeze(-1)

        exp_logits = logits.exp()
        sum_exp = exp_logits.sum(dim=-1)
        if comm.get_world_size(group) > 1:
            dist.all_reduce(sum_exp, op=dist.ReduceOp.SUM, group=group)

        # Each rank contributes the target logit only for ids it owns.
        target_mask = (target >= vocab_start) & (target < vocab_end)
        local_target = (target - vocab_start).clamp_(0, vocab_per_partition - 1)
        predicted = logits.gather(-1, local_target.unsqueeze(-1)).squeeze(-1)
        predicted = predicted * target_mask
        if comm.get_world_size(group) > 1:
            dist.all_reduce(predicted, op=dist.ReduceOp.SUM, group=group)

        loss = sum_exp.log() - predicted

        softmax = exp_logits / sum_exp.unsqueeze(-1)
        ctx.save_for_backward(softmax, target_mask, local_target)
        return loss

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        softmax, target_mask, local_target = ctx.saved_tensors
        grad = softmax
        # d/dz_i [logsumexp(z) - z_t] = softmax_i - 1{i == t}
        grad.scatter_add_(
            -1,
            local_target.unsqueeze(-1),
            -target_mask.to(grad.dtype).unsqueeze(-1),
        )
        grad = grad * grad_output.unsqueeze(-1)
        return grad, None, None, None


def vocab_parallel_cross_entropy(
    logits: torch.Tensor,
    target: torch.Tensor,
    group=None,
    vocab_start: int = 0,
    reduction: str = "mean",
) -> torch.Tensor:
    """Cross entropy for vocabulary-sharded logits.

    ``logits`` is ``[..., V/tp]``; ``target`` holds *global* token ids.
    """
    flat_logits = logits.reshape(-1, logits.size(-1))
    flat_target = target.reshape(-1)
    loss = _VocabParallelCrossEntropy.apply(flat_logits, flat_target, group, vocab_start)
    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    return loss.view(target.shape)
