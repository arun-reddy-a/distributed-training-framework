"""Mesh coordinates decide which ranks share a fast link — get them wrong and
everything still runs, just far slower than it should."""

from __future__ import annotations

from itertools import pairwise

import pytest
import torch.distributed as dist

from minidist.mesh import ParallelMesh

from .common import run_distributed


def _roundtrip_body(rank, world, tp, pp):
    mesh = ParallelMesh(tp_size=tp, pp_size=pp)
    assert mesh.tp_size * mesh.dp_size * mesh.pp_size == world

    for r in range(world):
        c = mesh.coord_of(r)
        assert mesh.rank_of(c.pp, c.dp, c.tp) == r, f"coordinate round-trip failed for rank {r}"

    coords = {(mesh.coord_of(r).pp, mesh.coord_of(r).dp, mesh.coord_of(r).tp) for r in range(world)}
    assert len(coords) == world, "mesh coordinates are not unique"

    assert dist.get_world_size(mesh.tp_group) == tp
    assert dist.get_world_size(mesh.pp_group) == pp
    assert dist.get_world_size(mesh.dp_group) == world // (tp * pp)


def _tp_ranks_are_contiguous_body(rank, world, tp, pp):
    """Tensor-parallel groups must be consecutive ranks.

    TP all-reduces sit on the critical path between two matmuls and cannot be
    overlapped, so they have to land on the fastest link available. Under the
    usual launcher, consecutive global ranks are the ranks sharing a node, i.e.
    NVLink rather than the network.
    """
    mesh = ParallelMesh(tp_size=tp, pp_size=pp)
    if tp > 1:
        members = [
            mesh.rank_of(mesh.pp_rank, mesh.dp_rank, t) for t in range(tp)
        ]
        assert members == list(range(members[0], members[0] + tp)), (
            f"[rank {rank}] tensor-parallel group {members} is not contiguous"
        )

    if pp > 1:
        # Pipeline neighbours are the *slowest*-varying axis, i.e. furthest apart.
        assert mesh.pp_ranks == sorted(mesh.pp_ranks)
        stride = mesh.dp_size * mesh.tp_size
        assert all(
            b - a == stride for a, b in pairwise(mesh.pp_ranks)
        ), f"[rank {rank}] pipeline ranks {mesh.pp_ranks} are not stride-{stride}"


def _stage_flags_body(rank, world, tp, pp):
    mesh = ParallelMesh(tp_size=tp, pp_size=pp)
    assert mesh.is_first_stage == (mesh.pp_rank == 0)
    assert mesh.is_last_stage == (mesh.pp_rank == pp - 1)
    assert (mesh.prev_stage_rank is None) == mesh.is_first_stage
    assert (mesh.next_stage_rank is None) == mesh.is_last_stage
    if not mesh.is_last_stage:
        assert mesh.coord_of(mesh.next_stage_rank).pp == mesh.pp_rank + 1


@pytest.mark.parametrize("tp,pp", [(1, 1), (2, 1), (1, 2), (2, 2), (4, 1), (1, 4)])
def test_mesh_coordinate_roundtrip(tp, pp):
    run_distributed(_roundtrip_body, 4, tp, pp)


@pytest.mark.parametrize("tp,pp", [(2, 2), (2, 1), (1, 2)])
def test_tensor_parallel_groups_are_contiguous(tp, pp):
    run_distributed(_tp_ranks_are_contiguous_body, 4, tp, pp)


@pytest.mark.parametrize("tp,pp", [(2, 2), (1, 4)])
def test_pipeline_stage_flags(tp, pp):
    run_distributed(_stage_flags_body, 4, tp, pp)


def _invalid_factorisation_body(rank, world):
    # `spawn` pickles the target by qualified name, so worker bodies must live
    # at module level — a closure here would fail to start rather than fail the
    # assertion it was written for.
    with pytest.raises(ValueError):
        ParallelMesh(tp_size=3, pp_size=1)


def test_invalid_factorisation_is_rejected():
    run_distributed(_invalid_factorisation_body, 2)
