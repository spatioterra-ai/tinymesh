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

```text
aggregation   sum source fields at each target

GraphSAGE     Linear(mean_neighbors(X)) + Linear_root(X)

TGCN          graph-project X into update / reset / candidate inputs
              then apply the same three node-local recurrent gates
```

Aggregation and GraphSAGE are direct equation matches. Both GraphSAGE layers
have 2,080 parameters. For TGCN, the pinned PyG Temporal source calls three
`GCNConv` layers. Tinymesh concatenates their output weights into one graph
projection:

```text
PyG Temporal    GCN_z(X) + GCN_r(X) + GCN_h(X)    3 propagations
Tinymesh        GCN_[z,r,h](X)                     1 propagation
```

The mapping sets the three PyG convolution biases to zero, then copies the
same graph and gate weights. Tinymesh has 9,312 parameters; PyG Temporal retains
9,408 because its three zeroed convolution biases remain registered. This is
an algebraic component comparison, not identical kernel work.

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
especially where one wider sparse propagation replaces three smaller calls.
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
