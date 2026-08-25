"""The strategies have to compose, not just work in isolation.

Each of these runs a 2x2 mesh on four ranks and compares against the *same
model with one axis removed*.  Chaining those statements with the single-axis
tests gives the full claim: a 3D-parallel model computes what a single-device
model computes.

Using a reduced-axis reference rather than a single-device one keeps the
comparison exact — tensor-parallel shards line up element for element, so no
test has to re-derive which axis each layer is sharded along.
"""

from __future__ import annotations

import torch
import torch.distributed as dist
from torch import nn

from minidist.ddp import DistributedDataParallel
from minidist.fsdp import FullyShardedDataParallel
from minidist.mesh import ParallelMesh
from minidist.models import GPT, Block, GPTConfig, build_pipeline_layers
from minidist.pipeline import PipelineStage, build_schedule, partition_layers
from minidist.tensor_parallel import vocab_parallel_cross_entropy

from .common import assert_close, run_distributed

CONFIG = GPTConfig(vocab_size=64, block_size=16, n_layer=4, n_head=4, n_embd=32)
BATCH = 8


def _batch():
    g = torch.Generator().manual_seed(7)
    ids = torch.randint(0, CONFIG.vocab_size, (BATCH, CONFIG.block_size), generator=g)
    targets = torch.randint(0, CONFIG.vocab_size, (BATCH, CONFIG.block_size), generator=g)
    return ids, targets


def _loss_fn(mesh):
    def fn(logits, targets):
        return vocab_parallel_cross_entropy(
            logits, targets, group=mesh.tp_group,
            vocab_start=mesh.tp_rank * (CONFIG.vocab_size // mesh.tp_size),
        )
    return fn


def _tensor_parallel_reference(mesh, ids, targets):
    """The same tensor-parallel model, run end to end without pipelining."""
    layers = build_pipeline_layers(CONFIG, tp_group=mesh.tp_group)
    x = ids
    for layer in layers:
        x = layer(x)
    loss = _loss_fn(mesh)(x, targets)
    loss.backward()
    grads = [
        [p.grad.clone() for p in layer.parameters()] for layer in layers
    ]
    return loss.detach(), grads


def _tp_and_pp_body(rank, world):
    mesh = ParallelMesh(tp_size=2, pp_size=2)
    ids, targets = _batch()
    ref_loss, ref_grads = _tensor_parallel_reference(mesh, ids, targets)

    all_layers = build_pipeline_layers(CONFIG, tp_group=mesh.tp_group)
    mine = partition_layers(all_layers, mesh.pp_size, mesh.pp_rank)
    offset = next(i for i, layer in enumerate(all_layers) if layer is mine[0])

    stage = PipelineStage(
        nn.Sequential(*mine),
        mesh,
        loss_fn=_loss_fn(mesh) if mesh.is_last_stage else None,
    )
    loss = build_schedule("1f1b", stage, mesh, 4).step(inputs=ids, targets=targets)

    if mesh.is_last_stage:
        assert_close(loss, ref_loss, f"[rank {rank}] tp2xpp2 loss", tol=1e-5)

    for local_index, layer in enumerate(mine):
        expected = ref_grads[offset + local_index]
        for i, (param, want) in enumerate(zip(layer.parameters(), expected, strict=True)):
            assert_close(
                param.grad, want,
                f"[rank {rank}] tp2xpp2 grad layer {offset + local_index} param {i}",
                tol=2e-5,
            )


def _tp_and_dp_body(rank, world):
    """Data parallel across TP replicas: each DP replica sees a different shard."""
    mesh = ParallelMesh(tp_size=2, pp_size=1)
    assert mesh.dp_size == 2
    ids, targets = _batch()

    reference = GPT(CONFIG, tp_group=mesh.tp_group)
    _, ref_loss = reference(ids, targets)
    ref_loss.backward()

    model = DistributedDataParallel(
        GPT(CONFIG, tp_group=mesh.tp_group), process_group=mesh.dp_group
    )
    # Tensor-parallel peers must see the *same* data; data-parallel peers must
    # see different data. That is exactly `dp_rank`, and nothing else.
    my_ids = ids.chunk(mesh.dp_size, dim=0)[mesh.dp_rank]
    my_targets = targets.chunk(mesh.dp_size, dim=0)[mesh.dp_rank]
    _, loss = model(my_ids, my_targets)
    loss.backward()

    for (name, param), (_, expected) in zip(
        model.module.named_parameters(), reference.named_parameters(), strict=True
    ):
        assert_close(param.grad, expected.grad, f"[rank {rank}] tp2xdp2 grad {name}", tol=2e-5)


def _tp_and_fsdp_body(rank, world):
    """FSDP shards along the DP axis while TP shards within each replica."""
    mesh = ParallelMesh(tp_size=2, pp_size=1)
    ids, targets = _batch()
    my_ids = ids.chunk(mesh.dp_size, dim=0)[mesh.dp_rank]
    my_targets = targets.chunk(mesh.dp_size, dim=0)[mesh.dp_rank]

    torch.manual_seed(0)
    ddp = DistributedDataParallel(
        GPT(CONFIG, tp_group=mesh.tp_group), process_group=mesh.dp_group
    )
    _, ddp_loss = ddp(my_ids, my_targets)
    ddp_loss.backward()

    torch.manual_seed(0)
    fsdp = FullyShardedDataParallel(
        GPT(CONFIG, tp_group=mesh.tp_group),
        unit_types=(Block,),
        process_group=mesh.dp_group,
    )
    _, fsdp_loss = fsdp(my_ids, my_targets)
    fsdp_loss.backward()

    assert_close(fsdp_loss, ddp_loss, f"[rank {rank}] tp2 x fsdp2 loss", tol=1e-6)

    # Reassemble the sharded gradients and compare against DDP's.
    id_to_name = {id(p): n for n, p in fsdp.module.named_parameters()}
    ddp_grads = {n: p.grad for n, p in ddp.module.named_parameters()}
    for unit in fsdp._units:
        shard = unit.flat_shard.grad
        parts = [torch.empty_like(shard) for _ in range(mesh.dp_size)]
        dist.all_gather(parts, shard.contiguous(), group=mesh.dp_group)
        flat = torch.cat(parts)[: unit.total_numel]
        offset = 0
        for p, numel, shape in zip(unit.params, unit.numels, unit.shapes, strict=True):
            name = id_to_name[id(p)]
            assert_close(
                flat[offset : offset + numel].view(shape), ddp_grads[name],
                f"[rank {rank}] tp2 x fsdp2 grad {name}", tol=1e-6,
            )
            offset += numel


def test_tensor_and_pipeline_parallel():
    run_distributed(_tp_and_pp_body, 4)


def test_tensor_and_data_parallel():
    run_distributed(_tp_and_dp_body, 4)


def test_tensor_parallel_with_fsdp():
    run_distributed(_tp_and_fsdp_body, 4)
