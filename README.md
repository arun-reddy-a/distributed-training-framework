# minidist — a lightweight distributed training framework

[![tests](https://github.com/arun-reddy-a/distributed-training-framework/actions/workflows/ci.yml/badge.svg)](https://github.com/arun-reddy-a/distributed-training-framework/actions/workflows/ci.yml)

Data, tensor, pipeline and fully-sharded parallelism for transformer training,
implemented directly on `torch.distributed` collectives — not by wrapping
PyTorch's own `DistributedDataParallel` or `FSDP` — plus a profiling layer that
measures where the communication time actually goes.

```
                      ┌─────────────────────────────────────────┐
   ParallelMesh       │  pipeline  ×  data  ×  tensor           │
                      │     (pp)      (dp)      (tp)            │
                      └─────────────────────────────────────────┘
                            │          │           │
              send/recv ────┘          │           └──── all-reduce (per layer)
              activations              │                 on the critical path
                                       │
                          all-reduce (DDP) ── once per step, overlapped
                       or all-gather + reduce-scatter (FSDP)
```

Everything composes: an 8-rank job can run as `tp=2 × pp=2 × dp=2` from one
command line, and the correctness tests check exactly that.

## Why build this

The four strategies are usually presented as a menu. They are better understood
as answers to four different resource limits, and the interesting content is in
*what each one costs* and *why the layers are cut where they are*:

| Strategy | Solves | Pays | Per step, per rank |
|---|---|---|---|
| **DDP** | throughput | one all-reduce over all parameters | `2·(W−1)/W · P` bytes |
| **FSDP** (ZeRO-3) | model state won't fit | 2 all-gathers + 1 reduce-scatter | `3·(W−1)/W · P` bytes (**1.5× DDP**) |
| **Tensor** | one *layer* won't fit | 4 all-reduces **per block**, unoverlappable | `4·b·s·h` per layer |
| **Pipeline** | depth won't fit; slow links | one activation send per stage boundary | `2·b·s·h` per stage |

The ordering of that table is the whole argument for the mesh layout: tensor
parallelism pays *per layer* and sits on the critical path between two matmuls,
so it must land on the fastest link; data parallelism pays once per step and
overlaps with backward, so it tolerates the slowest.

## Install and run

```bash
git clone https://github.com/arun-reddy-a/distributed-training-framework
cd distributed-training-framework
pip install -e ".[dev]"
```

```bash
# 8 GPUs, plain data parallel
torchrun --nproc_per_node=8 examples/train_gpt.py --strategy ddp

# 8 GPUs, ZeRO-3 with bf16 parameter all-gathers
torchrun --nproc_per_node=8 examples/train_gpt.py --strategy fsdp --fsdp-param-dtype bf16

# 8 GPUs as 2-way tensor × 2-way pipeline × 2-way data parallel
torchrun --nproc_per_node=8 examples/train_gpt.py \
    --tp 2 --pp 2 --strategy ddp --schedule 1f1b --microbatches 8

# no GPU required — Gloo on CPU runs every code path
torchrun --nproc_per_node=4 examples/train_gpt.py --tp 2 --pp 2 --backend gloo
```

```bash
make test          # 66 correctness tests, CPU only
make bench         # full benchmark suite -> results/<timestamp>/
```

## Design decisions

### The mesh: rank ordering is a bandwidth decision

A rank's coordinate is derived as

```
rank = pp_rank · (dp_size · tp_size) + dp_rank · tp_size + tp_rank
       \______ slowest varying ______/              \_ fastest varying _/
```

Tensor parallelism varies fastest, so a TP group is always a block of
*consecutive* ranks — under the usual launcher, the ranks sharing a node and
therefore NVLink. Pipeline parallelism varies slowest, so PP groups span nodes,
where their small point-to-point transfers are affordable. This follows
directly from the cost table above and is asserted in
[`tests/test_mesh.py`](tests/test_mesh.py).

One consequence worth stating because it is easy to get backwards: the data
seed must depend on `dp_rank` and **nothing else**. TP and PP ranks are
different pieces of *one* replica and must see identical batches; DP replicas
must see different ones.

### DDP: bucketing, overlap, and deterministic collective order

Two problems with the naive "backward, then all-reduce everything":

* **Latency.** Hundreds of small tensors (biases, LayerNorm weights) each
  become a collective whose cost is pure launch overhead. Fixed by packing
  gradients into ~25 MB flat buckets.
* **Serialisation.** The network idles through backward and the GPU idles
  through the all-reduce. Fixed by launching each bucket's all-reduce
  asynchronously the moment its last gradient lands, so it rides alongside the
  remaining backward compute.

The part that is genuinely subtle: **NCCL matches collectives by call order,
not by name.** If rank 0 issues all-reduce(bucket 2) while rank 1 issues
all-reduce(bucket 5), the two get paired — you silently reduce mismatched
buffers, or hang. Gradient-ready order is not guaranteed identical across
ranks.

[`minidist/ddp.py`](minidist/ddp.py) removes the hazard structurally rather than
hoping: bucket `k` is launched only once buckets `0..k-1` have launched.
Readiness still *triggers* the launch, so overlap is preserved, but the
observable sequence is 0, 1, 2, … on every rank, always. Buckets are built in
**reverse** parameter order — approximately the order backward produces
gradients — which is what keeps that in-order constraint from costing anything.

`tests/test_ddp.py` runs this with a bucket cap of 1 byte (one collective per
parameter, the configuration most likely to expose an ordering bug) and with a
single giant bucket, and requires both to reproduce the full-batch gradient.

### FSDP: freeing parameters that autograd still needs

Each unit's lifecycle is all-gather → compute → free → re-gather →
compute grads → reduce-scatter → free. Step three is the hard one: forward has
already saved those parameters for backward, so they cannot simply be dropped.

The mechanism is to free the **storage** while keeping the tensors:

```python
full.untyped_storage().resize_(0)        # free; views survive, data does not
full.untyped_storage().resize_(nbytes)   # realloc, then all-gather back into it
```

Each parameter's `.data` is a view into `full`, assigned once at construction
and never reassigned. A view resolves its address as
`storage.data_ptr() + offset` at *access* time, so re-allocating the storage
silently re-points every view. Autograd is undisturbed because it saved the
`Parameter` objects, whose version counters are independent of the buffer the
all-gather writes into. This is the same trick `torch.distributed.fsdp` uses,
and it is why FSDP parameters must not be touched outside forward/backward —
[`summon_full_params()`](minidist/fsdp.py) is the supported way in.

Two decisions worth calling out:

* **Wrap per block, not per model.** Peak memory is `state/W` plus the largest
  thing ever gathered at once, so unit granularity sets the peak.
* **Always reshard at the end of backward**, even for units configured not to
  reshard after forward. `reshard_after_forward` governs only whether
  parameters survive *between* forward and backward; holding them past backward
  would leave the gathered copy stale the instant the optimizer updates the
  shard, and the next forward would silently train on pre-update weights. This
  was a real bug caught by comparing weights (not just losses) after several
  steps.

### Tensor parallelism: why the MLP costs one all-reduce, not two

For `y = W₂ · GELU(W₁ · x)`, shard `W₁` by **columns**:

```
W₁ = [W₁₀ | W₁₁]   ⟹   GELU(x·W₁) = [GELU(x·W₁₀) | GELU(x·W₁₁)]
```

GELU is elementwise, so it commutes with the column split — no communication.
Now shard `W₂` by **rows**, matching the layout its input already arrives in:

```
W₂ = [W₂₀]         ⟹   y = (h₀·W₂₀) + (h₁·W₂₁)
     [W₂₁]                 \_ local _/  \_ all-reduce _/
```

One all-reduce for the whole block. Sharding `W₁` by rows instead would need a
collective *between* the two matmuls and another after — twice the traffic, all
of it on the critical path.

Attention shards the same way with heads as the unit, which keeps softmax
entirely local. One detail that is easy to get silently wrong: the QKV weight
is laid out as `[n_head, 3, head_dim]` — q, k, v **interleaved per head** —
rather than the familiar `[q | k | v]`. A column-parallel split takes a
contiguous slice, and under `[q | k | v]` at `tp=2` rank 0 would get all of q
plus half of k while rank 1 got half of k plus all of v.

**Replicated parameters need no extra all-reduce.** LayerNorm weights and
row-parallel biases are replicated across the TP group and their gradients must
agree on every rank. They do, for free: any gradient reaching them has already
passed backward through an `f` operator, which all-reduces. This is a property
of *where the collectives sit*, easy to break by moving one, so
`tests/test_tensor_parallel.py` asserts it directly.

**Vocab-parallel cross entropy.** The obvious implementation all-gathers the
logits; for a 128k vocabulary and 16k tokens that is an 8 GB fp32 tensor on
every rank, routinely the peak-memory event of the step. Instead, log-sum-exp
decomposes over a partition of the vocabulary into three scalar-per-token
reductions (`max`, `sum(exp)`, and the target's own logit). Traffic drops from
`O(b·s·V)` to `O(b·s)` and the full logit tensor never exists.

### Pipeline parallelism: 1F1B does not reduce the bubble

A widespread misreading. Both GPipe and 1F1B execute `M` forwards and `M`
backwards per stage with the same `P−1` steps of fill and drain, so **both have
bubble fraction `(P−1)/(M+P−1)`**. What 1F1B changes is *peak activation
memory*: GPipe runs all `M` forwards before the first backward and so holds `M`
microbatches of activations at stage 0, while 1F1B starts backpropagating as
early as possible and holds at most `P`.

Since `M` must be **large** to shrink the bubble and `P` is fixed and small,
that is the difference between "raise M until memory explodes" and "raise M
freely". 1F1B is what makes a low bubble *affordable*, not what makes it low.
The measured sweep below shows exactly this: identical bubble trends, and live
activations that grow 1→16 under GPipe but cap at 4 under 1F1B.

The point-to-point layer had one bug worth recording, because it only appears
with three or more stages: a middle stage both receives activations from `p−1`
and sends to `p+1`, and a single shape cache shared between the two directions
let the incoming header satisfy the outgoing channel's "already described"
check — so `p+1` waited for a header that was never sent and then mis-parsed the
payload as one. Send and receive caches are separate for this reason.

### Mixed precision: the loss scaler must be collective

bf16 keeps fp32's 8-bit exponent and spends the savings out of the mantissa, so
gradients that would flush to zero in fp16 stay representable and **no loss
scaling is needed at all**. fp16 has the mantissa but not the range, so it needs
adaptive scaling.

The distributed part has no single-GPU analogue. `inf`/`nan` detection is
inherently local — each rank inspects the gradients it holds — but **under FSDP
each rank holds a different shard, and under tensor parallelism each rank holds
a different slice of the weight matrix.** An overflow in one shard is invisible
to every other rank. If rank 2 skips its optimizer step while ranks 0, 1, 3
apply theirs, the replicas diverge silently, the loss still goes down, and
nothing looks wrong until an evaluation much later disagrees.

The fix is one all-reduce of the `found_inf` flag with `MAX` across every rank
holding a piece of the gradient — and it is not optional.
`tests/test_amp.py` injects an `inf` on exactly one rank and requires every rank
to skip and to end up with the same scale factor.

## Correctness

66 tests, all multi-process, all on the Gloo/CPU backend so they run in CI on
machines with no GPU. The pattern throughout: run the distributed
implementation across *N* spawned processes and compare against a reference
computed in one.

| Claim | Test |
|---|---|
| DDP reproduces the full-batch gradient, for any bucket size and with overlap on or off | `test_ddp.py` |
| FSDP's reassembled gradient equals DDP's (to 1e-6), and weights still agree after 4 steps | `test_fsdp.py` |
| A `tp=4` model is bit-comparable to a `tp=1` model built from the same seeds | `test_tensor_parallel.py` |
| Vocab-parallel cross entropy matches `F.cross_entropy` in value *and* gradient | `test_tensor_parallel.py` |
| Replicated (LayerNorm/bias) gradients are identical on every TP rank | `test_tensor_parallel.py` |
| GPipe and 1F1B both reproduce the non-pipelined loss and gradients | `test_pipeline.py` |
| 1F1B holds no more live activations than GPipe, for identical results | `test_pipeline.py` |
| `tp=2 × pp=2` and `tp=2 × fsdp=2` compose correctly | `test_composition.py` |
| One rank's overflow makes every rank skip, with scales staying in sync | `test_amp.py` |
| The overlap interval arithmetic does not double-count concurrent kernels | `test_profiling.py` |

A note on how the FSDP end-to-end test is written, because it is a real trap:
FSDP reduce-scatters where DDP all-reduces, so the two sum the same values in a
different order and their gradients differ by **~5e-8 relative** — pure fp32
rounding. Adam's update is `m/(√v + ε)`, which is scale-invariant and therefore
*amplifies* that relative noise rather than shrinking it; after four steps it
reaches 1e-3 on parameters with small gradients, which says nothing about
correctness. The multi-step comparison therefore uses SGD, and gradient
equivalence is checked directly at a 1e-6 tolerance.

## Benchmarks

Three harnesses, all launched through `scripts/run_benchmarks.sh`:

* **`bench_collectives.py`** — all-reduce / all-gather / reduce-scatter /
  broadcast across message sizes, reporting both algorithm bandwidth and **bus
  bandwidth** (the volume actually crossing the interconnect, comparable
  against the link's peak and against `nccl-tests`).
* **`bench_parallelism.py`** — the same model under each strategy: step time,
  measured communication volume *against the analytical prediction*, state
  memory, and exposed communication from the profiler trace. Includes
  `ddp-no-overlap` as a control, so overlap's value is measured rather than
  assumed.
* **`bench_pipeline.py`** — bubble sweep over microbatch count, measured against
  `(P−1)/(M+P−1)`, with per-stage busy times and an ASCII schedule timeline.

### Communication accounting

Every collective is instrumented ([`minidist/comm.py`](minidist/comm.py)), so
measured byte volume can be checked against theory rather than trusted:

```
  strategy           ms/step       tok/s   comm MB  predicted  state MB
  ----------------------------------------------------------------------
  none                184.32       22223       0.0        0.0     104.8
  ddp                 225.53       18162      39.3       39.3     104.8
  ddp-no-overlap      220.64       18564      39.3       39.3     104.8
  fsdp                283.73       14436      52.9       59.0      26.2
  fsdp-bf16          6076.50         674      36.3       39.3      26.2

  per-collective bus traffic (MB/step/rank):
    ddp             all_reduce=39.3
    fsdp            all_gather=33.2  reduce_scatter=19.7
    fsdp-bf16       all_gather=16.6  reduce_scatter=19.7
```
<sub>4 ranks, Gloo/CPU, 6.9M-parameter model. Timings are illustrative of the
accounting method rather than absolute performance; the byte counts and
memory ratios are the numbers worth trusting here.</sub>

DDP's measured traffic matches the prediction exactly. FSDP's measured 52.9 MB
comes in below the naive 59.0 MB prediction (1.34× DDP rather than 1.5×)
because the root unit — embeddings, final norm, LM head — is not resharded
after forward and so skips one of its two all-gathers. Sharded state is 0.25×
DDP's on 4 ranks, i.e. exactly `1/W`, which is the entire point of FSDP.

### Pipeline bubble

```
  schedule     M   ms/step   measured   theory     gap  live acts   per-stage busy ms
  ------------------------------------------------------------------------------------
  gpipe        1    422.89      78.7%    75.0%   +3.7%          1   98.8  132.4  96.0  91.3
  gpipe        2    295.47      66.9%    60.0%   +6.9%          2   102.9 145.1 107.3  98.6
  gpipe        4    247.13      50.2%    42.9%   +7.3%          4   120.5 160.5 126.7 129.4
  gpipe        8    225.81      38.4%    27.3%  +11.2%          8   135.0 172.7 136.1 140.1
  gpipe       16    247.99      34.6%    15.8%  +18.8%         16   171.0 221.5 164.4 171.9

  1f1b         1    463.53      78.2%    75.0%   +3.2%          1   97.0  131.1  97.0  93.5
  1f1b         2    280.47      63.5%    60.0%   +3.5%          2   103.2 143.2 108.9 102.6
  1f1b         4    233.25      50.2%    42.9%   +7.3%          4   117.8 158.4 129.9 132.1
  1f1b         8    221.33      37.4%    27.3%  +10.2%          4   142.5 184.3 138.9 139.0
  1f1b        16    243.66      30.2%    15.8%  +14.4%          4   167.4 206.2 167.1 176.6
```

```
  1f1b timeline, M=16   (F=forward  B=backward  <=recv  .=idle)
    stage 0  FFFFF<<<<<BFFFBBFFBBFF<BBFBBFFBBFF<BBFFBBFFBBFFBBFFBBFFBBFFBBB<BB<BBBBB.
    stage 1  <<FFFFFF<BBFFBBBFBBFFFBBFFBBFFBBFFBBFFBBBFFBBFFBBFBBBFBBBFBBBFBBB<BBBBB.
    stage 2  <<<<<<FFF<BFFBBFBBBFFBBBFBBBFBBFFBBFFBBBFBBBBFBBFFBBFFBBFBBBFBBBFBBBBBB.
    stage 3  <<<<<<<<<FBBFBBFBBFFBB<FBBBFBBBFFBBBFBBFFBBBFBBBFBBBFFBBFBBBFBBFFBBFFBB.

  gpipe timeline, M=16
    stage 0  FFFFFFFFFFFFFFFFFFFF<<<<<<<<<<<<<<BBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB.
    stage 1  <<FFFFFFFFFFFFFFFFFFFFFFFFFFF<<<<BBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB.
    stage 2  <<<<<<FFFFFFFFFFFFFFFFFFFFFFFFFFF<BBBBBBBBBBBBBBBBBBBBBBBBBBBBBB........
    stage 3  <<<<<<<<<<<<<<<<<FFFFFFFFFFFFFFFFFFFFFFFFFBBBBBBBBBBBBBBBBBBBBBBBBBBBBB.
```
<sub>4 stages, Gloo/CPU, 12-layer model, batch 16.</sub>

Three things this makes visible:

1. **The `live acts` column is the whole GPipe/1F1B story.** Identical bubble
   trends; GPipe's live activations grow 1 → 16 with `M` while 1F1B's cap at 4
   (= `P`). The timelines show why — GPipe is all-F-then-all-B, 1F1B alternates.
2. **Measured always exceeds theory, and the gap widens with `M`.** The formula
   counts only structural fill and drain. Real pipelines also pay for transfer
   time and stage imbalance — visible in the per-stage busy column, where stage
   1 is consistently ~1.3× stage 0 because uniform layer partitioning ignores
   that the embedding and LM head are not free.
3. **Step time bottoms out at `M=8` and rises at `M=16`.** Past some point the
   microbatches get too small to use the hardware efficiently, and that loss
   outweighs the shrinking bubble. The bubble is not the only cost.

### Comm/compute overlap

[`minidist/profiling.py`](minidist/profiling.py) parses the exported Chrome
trace and does interval arithmetic per stream:

```
   compute   ####----########--####
   comm      --######----######----
               ^^^^      ^^^^          overlapped
             ^^    ^^        ^^        exposed comm  ← the number that matters

   exposed_comm = |comm| − |comm ∩ compute|
```

Summing durations would double-count, because NCCL kernels and compute kernels
live on different streams and genuinely run at the same time. Merging
overlapping intervals per category first, then intersecting, gives numbers that
add up to the actual step. A step can spend 40% of its wall time inside NCCL and
still be compute-bound; what hurts is the *exposed* part.

```bash
torchrun --nproc_per_node=8 examples/train_gpt.py --strategy ddp --profile trace.json
python benchmarks/analyze_trace.py trace.json
```

## Repository layout

```
minidist/
  comm.py              process-group bootstrap, instrumented collectives, byte counters
  mesh.py              3D pipeline × data × tensor mesh and subgroup construction
  ddp.py               bucketed all-reduce overlapped with backward
  fsdp.py              ZeRO-3 sharding via storage resize + all-gather/reduce-scatter
  tensor_parallel.py   column/row parallel layers, f/g operators, vocab-parallel CE
  pipeline.py          GPipe and 1F1B schedules, P2P transport, timeline instrumentation
  amp.py               precision policy and the collective grad scaler
  profiling.py         PyTorch Profiler wrapper, collective breakdown, overlap analysis
  models/gpt.py        a GPT that is both TP-shardable and pipeline-splittable
examples/train_gpt.py  one entry point for every combination of strategies
benchmarks/            collectives, strategies, pipeline bubble, trace analysis
tests/                 66 multi-process correctness tests (Gloo/CPU)
scripts/               test and benchmark drivers
```

## Scope and limitations

Deliberately out of scope, and why:

* **Interleaved (virtual-stage) 1F1B**, which is what actually drives the bubble
  below `(P−1)/(M+P−1)`. The timeline instrumentation here is what you would use
  to evaluate it.
* **Distributed checkpointing.** `summon_full_params()` is provided for
  inspection; sharded checkpoint save/load is a substantial subsystem of its own.
* **Activation checkpointing**, ZeRO-1/2 as separate modes, and CPU offload.
* **Dropout inside a tensor-parallel region**, which needs rank-divergent RNG
  inside the parallel region and rank-identical RNG outside. Rather than ship a
  half-solution whose bugs are invisible in the loss curve, dropout defaults to
  0 and the correctness tests stay meaningful.
* **Unused-parameter handling in DDP.** The implementation assumes a static
  graph where every parameter receives a gradient — true for the models here,
  and the assumption is documented at the point where it matters.

## References

- Li et al., [PyTorch Distributed: Experiences on Accelerating Data Parallel Training](https://arxiv.org/abs/2006.15704) (2020) — DDP bucketing and overlap
- Rajbhandari et al., [ZeRO: Memory Optimizations Toward Training Trillion Parameter Models](https://arxiv.org/abs/1910.02054) (2019)
- Zhao et al., [PyTorch FSDP: Experiences on Scaling Fully Sharded Data Parallel](https://arxiv.org/abs/2304.11277) (2023)
- Shoeybi et al., [Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism](https://arxiv.org/abs/1909.08053) (2019)
- Huang et al., [GPipe: Efficient Training of Giant Neural Networks using Pipeline Parallelism](https://arxiv.org/abs/1811.06965) (2018)
- Narayanan et al., [Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM](https://arxiv.org/abs/2104.04473) (2021) — 1F1B and 3D parallelism
- Micikevicius et al., [Mixed Precision Training](https://arxiv.org/abs/1710.03740) (2017)

## License

[MIT](LICENSE)
