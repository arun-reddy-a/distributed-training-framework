"""Pipeline schedules must reproduce the non-pipelined loss and gradients."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from minidist.mesh import ParallelMesh
from minidist.models import GPTConfig, build_pipeline_layers
from minidist.pipeline import (
    PipelineStage,
    build_schedule,
    partition_layers,
    theoretical_bubble_fraction,
)
from minidist.tensor_parallel import vocab_parallel_cross_entropy

from .common import assert_close, run_distributed

CONFIG = GPTConfig(vocab_size=64, block_size=16, n_layer=4, n_head=4, n_embd=32)
BATCH = 8


def _batch():
    g = torch.Generator().manual_seed(99)
    ids = torch.randint(0, CONFIG.vocab_size, (BATCH, CONFIG.block_size), generator=g)
    targets = torch.randint(0, CONFIG.vocab_size, (BATCH, CONFIG.block_size), generator=g)
    return ids, targets


def _loss_fn(tp_group):
    def fn(logits, targets):
        return vocab_parallel_cross_entropy(logits, targets, group=tp_group, vocab_start=0)
    return fn


def _reference(mesh, ids, targets):
    """Run the whole model in one process and return (loss, per-layer grads)."""
    layers = build_pipeline_layers(CONFIG, tp_group=mesh.tp_group)
    x = ids
    for layer in layers:
        x = layer(x)
    loss = _loss_fn(mesh.tp_group)(x, targets)
    loss.backward()
    grads = [
        [p.grad.clone() if p.grad is not None else None for p in layer.parameters()]
        for layer in layers
    ]
    return loss.detach(), grads


def _pipeline_body(rank, world, schedule_name, num_microbatches):
    mesh = ParallelMesh(tp_size=1, pp_size=world)
    ids, targets = _batch()

    ref_loss, ref_grads = _reference(mesh, ids, targets)

    all_layers = build_pipeline_layers(CONFIG, tp_group=mesh.tp_group)
    my_indices = range(len(all_layers))
    mine = partition_layers(all_layers, world, mesh.pp_rank)
    offset = next(
        i for i in my_indices if all_layers[i] is mine[0]
    )

    stage = PipelineStage(
        nn.Sequential(*mine),
        mesh,
        loss_fn=_loss_fn(mesh.tp_group) if mesh.is_last_stage else None,
    )
    schedule = build_schedule(schedule_name, stage, mesh, num_microbatches)
    loss = schedule.step(inputs=ids, targets=targets)

    if mesh.is_last_stage:
        assert_close(loss, ref_loss, f"[rank {rank}] {schedule_name} loss", tol=1e-5)

    for local_index, layer in enumerate(mine):
        expected = ref_grads[offset + local_index]
        for p_index, (param, want) in enumerate(zip(layer.parameters(), expected, strict=True)):
            assert param.grad is not None, (
                f"[rank {rank}] layer {offset + local_index} param {p_index} got no gradient"
            )
            assert_close(
                param.grad,
                want,
                f"[rank {rank}] {schedule_name} grad layer {offset + local_index} "
                f"param {p_index}",
                tol=2e-5,
            )


def _gpipe_body(rank, world):
    _pipeline_body(rank, world, "gpipe", 4)


def _one_f_one_b_body(rank, world):
    _pipeline_body(rank, world, "1f1b", 4)


def _single_microbatch_body(rank, world):
    """M = 1 degenerates to plain sequential execution — a useful edge case."""
    _pipeline_body(rank, world, "1f1b", 1)


def _schedules_agree_body(rank, world):
    """GPipe and 1F1B are reorderings of the same work: same loss, same grads."""
    mesh = ParallelMesh(tp_size=1, pp_size=world)
    ids, targets = _batch()

    results = {}
    for name in ("gpipe", "1f1b"):
        layers = partition_layers(
            build_pipeline_layers(CONFIG, tp_group=mesh.tp_group), world, mesh.pp_rank
        )
        stage = PipelineStage(
            nn.Sequential(*layers),
            mesh,
            loss_fn=_loss_fn(mesh.tp_group) if mesh.is_last_stage else None,
        )
        schedule = build_schedule(name, stage, mesh, 4)
        loss = schedule.step(inputs=ids, targets=targets)
        results[name] = (
            loss.clone() if loss is not None else None,
            [p.grad.clone() for p in stage.parameters()],
            schedule.max_activations_in_flight,
        )

    gpipe, onefoneb = results["gpipe"], results["1f1b"]
    if mesh.is_last_stage:
        assert_close(onefoneb[0], gpipe[0], f"[rank {rank}] 1f1b vs gpipe loss", tol=1e-6)
    for i, (a, b) in enumerate(zip(gpipe[1], onefoneb[1], strict=True)):
        assert_close(b, a, f"[rank {rank}] 1f1b vs gpipe grad {i}", tol=1e-6)

    # The whole point of 1F1B: fewer live activations for identical results.
    assert onefoneb[2] <= gpipe[2], (
        f"[rank {rank}] 1f1b held {onefoneb[2]} activations, gpipe held {gpipe[2]}"
    )


def _bubble_body(rank, world):
    """More microbatches must shrink the measured bubble, tracking the model."""
    mesh = ParallelMesh(tp_size=1, pp_size=world)
    ids, targets = _batch()

    measured = {}
    for num_microbatches in (1, 4):
        layers = partition_layers(
            build_pipeline_layers(CONFIG, tp_group=mesh.tp_group), world, mesh.pp_rank
        )
        stage = PipelineStage(
            nn.Sequential(*layers),
            mesh,
            loss_fn=_loss_fn(mesh.tp_group) if mesh.is_last_stage else None,
        )
        schedule = build_schedule("1f1b", stage, mesh, num_microbatches)
        schedule.step(inputs=ids, targets=targets)
        measured[num_microbatches] = schedule.report()

    for num_microbatches, report in measured.items():
        expected = theoretical_bubble_fraction(world, num_microbatches)
        assert report["theoretical_bubble"] == pytest.approx(expected)
        assert 0.0 <= report["measured_bubble"] <= 1.0
        assert report["busy_s"] > 0.0

    if not mesh.is_first_stage:
        # Non-first stages start idle waiting on their predecessor, so they
        # always show a real bubble.
        assert measured[1]["measured_bubble"] > 0.0, (
            f"[rank {rank}] expected a measurable bubble at M=1"
        )


def _partition_body(rank, world):
    layers = [nn.Linear(2, 2) for _ in range(7)]
    covered = []
    for stage in range(world):
        covered.extend(partition_layers(layers, world, stage))
    assert covered == layers, "partitioning must cover every layer exactly once"


@pytest.mark.parametrize("world", [2, 4])
def test_gpipe_matches_single_process(world):
    run_distributed(_gpipe_body, world)


@pytest.mark.parametrize("world", [2, 4])
def test_1f1b_matches_single_process(world):
    run_distributed(_one_f_one_b_body, world)


def test_single_microbatch():
    run_distributed(_single_microbatch_body, 2)


@pytest.mark.parametrize("world", [2, 4])
def test_schedules_agree(world):
    run_distributed(_schedules_agree_body, world)


@pytest.mark.parametrize("world", [2, 4])
def test_bubble_measurement(world):
    run_distributed(_bubble_body, world)


def test_layer_partitioning_is_a_cover():
    run_distributed(_partition_body, 3)
