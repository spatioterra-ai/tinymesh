# Mac framework benchmark

This benchmark asks how equal graph operations behave in Tinymesh and the
PyTorch Geometric stack on one Apple M4 Mac.

## Decision

At Tinymesh revision
[`e644cf8`](https://github.com/spatioterra-ai/tinymesh/tree/e644cf820880bbe62f1d131f5db0b19a742317ae),
the three configured cross-framework paths agree within float32 precision.
JIT Tinymesh records lower best and median Metal/MPS wall time in all three
cases; PyG records lower CPU medians. The distributions are noisy enough that
this is implementation evidence, not a general framework speed claim.

Repeated Tinymesh execution should use `TinyJit`. Eager schedule construction
dominates its model timings. No runtime API or dependency changes follow from
this result.

## Equal work

The benchmark uses `N = 4,096`, directed degree eight, `E = 32,768`, and
feature and hidden widths of 32. TGCN adds one explicit self-loop per node, for
36,864 recurrent edges.

Aggregation sums source fields at each target. The two layers build different
models over that primitive.

### GraphSAGE

```text
neighbor fields X_j -- CSR mean -- Linear(F, H) --+
                                                   +--> X'_i
own field X_i -------------------- Linear(F, H) --+
```

This is PyG `SAGEConv` with its default mean aggregation, root transform, and
no projection or output normalization:

```text
X'_i = W_neighbor mean({X_j : j -> i}) + b + W_root X_i
```

Both implementations make one neighborhood aggregation and have `2FH + H`
parameters: 2,080 at `F = H = 32`. Tinymesh owns one homogeneous graph as CSR;
the measured PyG path receives COO `edge_index` and uses message passing. PyG's
additional bipartite, aggregation, projection, and normalization modes are
outside this comparison.

### TGCN

```text
X_t -- normalized graph sum -- Linear(F, 3H) -- split G_z, G_r, G_h
                                                        |
H_t-1 ---------------------------- three local gates ---+
                                                        |
                                                        v
                                                      H_t
```

The pinned PyG Temporal cell instead expresses the three graph projections as
three `GCNConv(F, H)` calls. Shared topology and linearity permit one exact
factorization:

```text
S = D^-1/2 A D^-1/2 X

PyG Temporal    [S W_z, S W_r, S W_h]    3 graph propagations
Tinymesh         S [W_z | W_r | W_h]      1 graph propagation
```

Both then apply the same update, reset, candidate, and hidden-state equations.
The parity mapping disables PyG Temporal's three graph-convolution biases and
copies every graph and gate weight. Those biases are redundant before gate
linears that already own a bias. Tinymesh therefore has
`3FH + 6H^2 + 3H` parameters, or 9,312 here; PyG Temporal registers another
`3H`, for 9,408. TGCN is an algebraically aligned cell comparison, not
identical kernel work.

### Model boundary

| | `SAGEConv` | `TGCN` |
| --- | --- | --- |
| Question | what do my neighbors say now? | what do my neighbors say now, and what should I remember? |
| Input | `X` | `X_t`, `H_t-1` |
| Graph operator | incoming mean | symmetric normalized sum with explicit self-loops |
| Own-node path | separate root linear | self-loop inside the graph sum |
| Sparse work | one propagation per layer | one propagation per time step |
| State | none | node-aligned hidden state |
| Parameters | `2FH + H` | `3FH + 6H^2 + 3H` |
| Reference | direct PyG default equation | fused PyG Temporal equation |

The timing rows are therefore not a race between GraphSAGE and TGCN. Each
Tinymesh component is compared with its own reference implementation; TGCN
performs additional recurrent work and returns persistent state.

| Device | Aggregation error | GraphSAGE error | TGCN error |
| --- | ---: | ---: | ---: |
| CPU | `0` | `7.45e-9` | `5.96e-8` |
| Metal/MPS | `4.77e-7` | `1.12e-8` | `5.96e-8` |

## Protocol

The Apple M4 ran macOS 26.5.2 and Python 3.12.13. Tinymesh used tinygrad
`33755a34`; the comparison used PyTorch 2.8.0, PyG 2.8.0, and PyG Temporal
source `fe555bc3`.

Each path receives ten warmups followed by 50 synchronized wall-time samples.
Topology construction, data transfer, and first compilation are excluded.
Tinymesh reports eager and `TinyJit` paths; the table below compares `TinyJit`
with eager PyG. PyTorch reported four CPU threads; both stacks otherwise used
their defaults. PyTorch compilation was not measured.

Values are minimum / median / p90 milliseconds:

| Device | Component | Tinymesh JIT | PyG / PyG Temporal |
| --- | --- | ---: | ---: |
| Metal/MPS | aggregation | `2.65 / 5.45 / 33.26` | `4.18 / 8.83 / 42.30` |
| Metal/MPS | GraphSAGE | `3.77 / 11.27 / 120.54` | `6.77 / 17.72 / 216.77` |
| Metal/MPS | TGCN | `4.66 / 9.05 / 81.47` | `13.10 / 82.77 / 375.99` |
| CPU | aggregation | `1.32 / 10.04 / 94.60` | `1.35 / 2.67 / 4.66` |
| CPU | GraphSAGE | `6.57 / 52.98 / 184.03` | `2.54 / 20.40 / 136.70` |
| CPU | TGCN | `25.76 / 322.22 / 495.17` | `16.20 / 156.11 / 394.59` |

Tinymesh eager medians were 23.16, 208.08, and 451.11 ms on Metal; and
65.24, 223.09, and 526.85 ms on CPU. They describe Python scheduling cost, not
the sparse kernels alone.

## Interpretation

The parity checks are the durable result: all three Tinymesh components have a
concrete mapping to the established PyG stack. Metal execution is promising,
especially where one shared normalized graph sum replaces three propagations.
CPU execution is not competitive in this run; even aggregation is only tied in
the best sample and loses clearly at the median.

The broad minimum-to-p90 ranges show that this workstation was not an isolated
benchmark host. Ratios, especially the TGCN Metal median, must not be promoted
as headline speedups. The synchronized minima show an achievable lower bound;
the medians show the experienced latency during this run.

## Limits

This record covers forward float32 execution over one balanced fixed graph. It
does not measure backward, optimizer steps, peak memory, compilation, topology
construction, changing topology, edge features, graph skew, multiple shapes,
batch throughput, or `torch.compile`.

The published PyG Temporal 0.56.2 package resolves `torch-sparse` 0.6.18 from
source on Python 3.12/macOS. The benchmark therefore loads only the exact TGCN
file from the pinned submodule. That is enough for this cell comparison, but it
is not a benchmark of the full installed PyG Temporal package.

## Reproduce

```console
git submodule update --init submodules/pytorch-geometric-temporal
uv run --locked --with torch==2.8.0 --with torch-geometric==2.8.0 python -m experiments.run framework_benchmark DEV=CPU DEGREE=8 HIDDEN=32 NODES=4096 SAMPLES=50 WARMUPS=10 WIDTH=32
uv run --locked --with torch==2.8.0 --with torch-geometric==2.8.0 python -m experiments.run framework_benchmark DEV=METAL DEGREE=8 HIDDEN=32 NODES=4096 SAMPLES=50 WARMUPS=10 WIDTH=32
```

The runner records the Tinymesh revision, every reference gitlink, explicit
settings, elapsed time, and the complete JSON observation under the ignored
`experiments/runs/` directory.

This benchmark is manual evidence, not routine CI. The tracked runner preserves
the protocol; this record preserves the representative result. The global
rerun and retention contract lives in [Experiments](../experiments.md#benchmark-retention).
