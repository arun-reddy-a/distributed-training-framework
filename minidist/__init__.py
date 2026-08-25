"""minidist — a lightweight distributed training framework.

Implements the four parallelism strategies used to train large models, from
``torch.distributed`` collectives rather than by wrapping PyTorch's own
``DistributedDataParallel`` / ``FSDP``:

* :mod:`minidist.ddp`              — data parallelism, bucketed and overlapped
* :mod:`minidist.fsdp`             — ZeRO-3 parameter/gradient/optimizer sharding
* :mod:`minidist.tensor_parallel`  — Megatron-style intra-layer sharding
* :mod:`minidist.pipeline`         — GPipe and 1F1B inter-layer schedules
* :mod:`minidist.amp`              — mixed precision, incl. a collective grad scaler
* :mod:`minidist.profiling`        — collective breakdown and overlap analysis

The strategies compose along a 3D :class:`~minidist.mesh.ParallelMesh`.
"""

from .amp import DistributedGradScaler, PrecisionConfig, resolve_precision
from .comm import (
    barrier,
    comm_stats,
    get_rank,
    get_world_size,
    init_distributed,
    is_primary,
    print_rank0,
    record_comm,
    shutdown_distributed,
)
from .ddp import DDP, DistributedDataParallel
from .fsdp import FSDP, FullyShardedDataParallel, MixedPrecisionPolicy
from .mesh import ParallelMesh
from .pipeline import (
    PipelineStage,
    build_schedule,
    partition_layers,
    theoretical_bubble_fraction,
)
from .profiling import (
    analyze_trace,
    format_collective_table,
    profile_steps,
    summarize_collectives,
)
from .tensor_parallel import (
    ColumnParallelLinear,
    RowParallelLinear,
    VocabParallelEmbedding,
    vocab_parallel_cross_entropy,
)

__version__ = "0.1.0"

__all__ = [
    "DDP",
    "DistributedDataParallel",
    "FSDP",
    "FullyShardedDataParallel",
    "MixedPrecisionPolicy",
    "ParallelMesh",
    "PipelineStage",
    "ColumnParallelLinear",
    "RowParallelLinear",
    "VocabParallelEmbedding",
    "DistributedGradScaler",
    "PrecisionConfig",
    "vocab_parallel_cross_entropy",
    "build_schedule",
    "partition_layers",
    "theoretical_bubble_fraction",
    "resolve_precision",
    "init_distributed",
    "shutdown_distributed",
    "barrier",
    "get_rank",
    "get_world_size",
    "is_primary",
    "print_rank0",
    "comm_stats",
    "record_comm",
    "profile_steps",
    "summarize_collectives",
    "format_collective_table",
    "analyze_trace",
    "__version__",
]
