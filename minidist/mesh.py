r"""A 3D (pipeline x data x tensor) process mesh.

A rank's position in the mesh determines which collectives it participates in.
The mapping from a flat rank to its ``(pp, dp, tp)`` coordinate is not
arbitrary — it decides which parallelism dimension gets the fast links:

.. code-block:: text

    rank = pp_rank * (dp_size * tp_size) + dp_rank * tp_size + tp_rank
           \_____ slowest varying _____/               \_ fastest varying _/

Tensor parallelism varies fastest, so a TP group is always a block of
*consecutive* ranks — which, under the usual ``torchrun`` launch, means ranks
on the same node, wired by NVLink/NVSwitch.  Pipeline parallelism varies
slowest, so PP groups span nodes.

That ordering follows from the communication volume of each strategy.  Per
transformer layer, per microbatch, with hidden size ``h``, sequence ``s``,
batch ``b``:

==================  =====================================  ==================
strategy            traffic                                when
==================  =====================================  ==================
tensor parallel     ``4 * b*s*h`` (2 all-reduce fwd + 2 bwd) every layer
pipeline parallel   ``2 * b*s*h`` (1 send fwd + 1 recv bwd)  every stage
data parallel       ``2 * |params| * (W-1)/W``               once per step
==================  =====================================  ==================

TP pays per layer and cannot overlap (its all-reduce sits on the critical path
between two matmuls), so it gets the fastest link and must stay intra-node.  DP
pays once per step and overlaps with backward, so it tolerates the slow link.
PP moves the least data but is latency-sensitive, so it lands in between.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.distributed as dist

from .comm import current_device, get_rank, get_world_size

__all__ = ["ParallelMesh"]


@dataclass(frozen=True)
class _Coord:
    pp: int
    dp: int
    tp: int


class ParallelMesh:
    """Factorises a flat process group into pipeline / data / tensor axes.

    Parameters
    ----------
    tp_size, pp_size:
        Tensor- and pipeline-parallel degrees.  The data-parallel degree is
        inferred as ``world_size // (tp_size * pp_size)`` so that the three
        always multiply to the world size.

    Notes
    -----
    Every rank constructs *every* subgroup in the same deterministic order,
    because ``dist.new_group`` is a collective: all ranks must call it the same
    number of times in the same sequence, even for groups they are not a member
    of.  Getting this wrong produces a hang at the first collective rather than
    an error, which is why the loops below are written over the full mesh
    instead of only the caller's slice.
    """

    def __init__(self, tp_size: int = 1, pp_size: int = 1, dp_size: int | None = None):
        world = get_world_size()
        if dp_size is None:
            dp_size = world // (tp_size * pp_size)
        if tp_size * pp_size * dp_size != world:
            raise ValueError(
                f"tp({tp_size}) * pp({pp_size}) * dp({dp_size}) = "
                f"{tp_size * pp_size * dp_size} != world_size({world})"
            )
        if min(tp_size, pp_size, dp_size) < 1:
            raise ValueError("parallel degrees must be >= 1")

        self.tp_size, self.pp_size, self.dp_size = tp_size, pp_size, dp_size
        self.world_size = world
        self.rank = get_rank()
        self.device = current_device()

        c = self.coord_of(self.rank)
        self.tp_rank, self.dp_rank, self.pp_rank = c.tp, c.dp, c.pp

        self._tp_group: dist.ProcessGroup | None = None
        self._dp_group: dist.ProcessGroup | None = None
        self._pp_group: dist.ProcessGroup | None = None
        self.pp_ranks: list[int] = [self.rank]

        if dist.is_available() and dist.is_initialized() and world > 1:
            self._build_groups()

    # -- coordinate algebra -------------------------------------------------

    def coord_of(self, rank: int) -> _Coord:
        # Inverse of rank_of below: peel off tp (fastest-varying) first, then
        # dp, leaving pp. Matches the mixed-radix layout in the module
        # docstring, so a rank's TP group is always a block of consecutive ranks.
        tp = rank % self.tp_size
        dp = (rank // self.tp_size) % self.dp_size
        pp = rank // (self.tp_size * self.dp_size)
        return _Coord(pp=pp, dp=dp, tp=tp)

    def rank_of(self, pp: int, dp: int, tp: int) -> int:
        # rank = pp * (dp_size * tp_size) + dp * tp_size + tp -- see module
        # docstring for why this ordering (tp fastest, pp slowest) matters.
        return pp * (self.dp_size * self.tp_size) + dp * self.tp_size + tp

    # -- group construction -------------------------------------------------

    def _build_groups(self) -> None:
        # Tensor-parallel groups: consecutive ranks, one group per (pp, dp).
        for pp in range(self.pp_size):
            for dp in range(self.dp_size):
                ranks = [self.rank_of(pp, dp, tp) for tp in range(self.tp_size)]
                group = dist.new_group(ranks)
                if self.rank in ranks:
                    self._tp_group = group

        # Data-parallel groups: one per (pp, tp).
        for pp in range(self.pp_size):
            for tp in range(self.tp_size):
                ranks = [self.rank_of(pp, dp, tp) for dp in range(self.dp_size)]
                group = dist.new_group(ranks)
                if self.rank in ranks:
                    self._dp_group = group

        # Pipeline groups: one per (dp, tp).  Also remember the ordered rank
        # list, since send/recv needs the *global* rank of the neighbouring
        # stage, not its index within the group.
        for dp in range(self.dp_size):
            for tp in range(self.tp_size):
                ranks = [self.rank_of(pp, dp, tp) for pp in range(self.pp_size)]
                group = dist.new_group(ranks)
                if self.rank in ranks:
                    self._pp_group = group
                    self.pp_ranks = ranks

    # -- accessors ----------------------------------------------------------

    @property
    def tp_group(self) -> dist.ProcessGroup | None:
        return self._tp_group

    @property
    def dp_group(self) -> dist.ProcessGroup | None:
        return self._dp_group

    @property
    def pp_group(self) -> dist.ProcessGroup | None:
        return self._pp_group

    @property
    def is_first_stage(self) -> bool:
        return self.pp_rank == 0

    @property
    def is_last_stage(self) -> bool:
        return self.pp_rank == self.pp_size - 1

    @property
    def prev_stage_rank(self) -> int | None:
        return None if self.is_first_stage else self.pp_ranks[self.pp_rank - 1]

    @property
    def next_stage_rank(self) -> int | None:
        return None if self.is_last_stage else self.pp_ranks[self.pp_rank + 1]

    def seed_offset(self) -> int:
        """Per-rank RNG offset.

        Data-parallel replicas must see *different* data but identical
        parameters; tensor-parallel ranks must see identical data but hold
        different parameter shards.  Deriving the data seed from ``dp_rank``
        alone (and never from ``tp_rank``) is what keeps both true.
        """
        return self.dp_rank + self.pp_rank * self.dp_size

    def describe(self) -> str:
        return (
            f"world={self.world_size}  tp={self.tp_size} dp={self.dp_size} "
            f"pp={self.pp_size}  |  rank {self.rank} -> "
            f"(pp={self.pp_rank}, dp={self.dp_rank}, tp={self.tp_rank})"
        )

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return f"ParallelMesh({self.describe()})"


def torch_dtype_bytes(dtype: torch.dtype) -> int:
    """Bytes per element for `dtype`, without allocating a real tensor."""
    return torch.empty((), dtype=dtype).element_size()
