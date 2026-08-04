# Mac framework benchmark

This benchmark asks how equal graph operations behave in Tinymesh and the
PyTorch Geometric stack on one Apple M4 Mac.

## Decision

At Tinymesh revision
[`c44270c`](https://github.com/spatioterra-ai/tinymesh/tree/c44270c90d715938bb7bb8867936dca752911dbf),
all eager, compiled, and algebraically factored TGCN paths agree within
`6.71e-8`.

Tinymesh JIT records 5.90 and 6.06 ms Metal medians in two repetitions, versus
14.44 and 14.33 ms for eager PyG Temporal. That is a real advantage over the
published eager cell, but not a framework victory: full-graph compiled PyG
Temporal records 3.28 and 3.91 ms, and the compiled one-propagation PyTorch
control records 4.74 and 2.81 ms. Both controls beat Tinymesh in both runs.

On CPU, Tinymesh JIT and PyG Temporal are tied at 20.47 and 20.81 ms; the
one-propagation compiled PyTorch control records 8.04 ms. The evidence supports
Tinymesh's TGCN factorization, CSR execution, and tinygrad-native composition.
It rejects the claim that Tinymesh generally outperforms PyG Temporal.

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
| CPU | `0` | `7.45e-9` | `<= 5.96e-8` |
| Metal/MPS | `4.77e-7` | `1.12e-8` | `<= 6.71e-8` |

## Why TGCN looked faster

The first benchmark compared Tinymesh JIT with eager PyG Temporal. Its 9.05
versus 82.77 ms Metal medians were unusually noisy and mixed three effects:

```text
PyG Temporal
X -- Linear(F,H) -- gather [E,H] -- edge weight -- scatter-add --+ x 3
                                                                  +--> gates

Tinymesh
X -- node scale -- CSR pull [N,F] -- node scale -- Linear(F,3H) -----> gates
```

1. **Algebra:** PyG Temporal calls three `GCNConv` layers. Tinymesh factors
   their shared normalization and input into one propagation.
2. **Sparse layout:** the measured PyG COO path gathers and weights an
   edge-sized message tensor, then uses `scatter_add_`. Tinymesh's CSR pull
   gives each output one owner, materializes no edge message tensor, and needs
   no atomics.
3. **Execution:** `TinyJit` captures and replays the complete Tinymesh call;
   the original PyG Temporal control used eager dispatch.

The factored PyTorch control keeps PyG's COO gather/scatter backend but uses the
same one-propagation equation and 9,312 parameters as Tinymesh. It separates
the first effect from the other two.

### Tinygrad alignment

The Tinymesh layer follows tinygrad's own shape: a plain callable class owns
tensors, hidden state is an explicit input and output, and JIT remains at the
outer execution boundary. Tinygrad's
[`LSTMCell`](https://github.com/tinygrad/tinygrad/blob/33755a34657d25920914badbe32a9d70489669c7/tinygrad/nn/__init__.py#L398-L424)
similarly groups independent gate projections into wide linear maps.

[`TinyJit`](https://github.com/tinygrad/tinygrad/blob/33755a34657d25920914badbe32a9d70489669c7/tinygrad/engine/jit.py#L225-L310)
captures the scheduled calls, compiles them, plans their memory, and replays
them. On Metal,
[`MetalGraph`](https://github.com/tinygrad/tinygrad/blob/33755a34657d25920914badbe32a9d70489669c7/tinygrad/runtime/graph/metal.py#L10-L90)
places the captured programs in one indirect command buffer. That is one
submission, not one fused kernel; the programs still execute behind barriers.

## Protocol

The Apple M4 ran macOS 26.5.2 and Python 3.12.13. Tinymesh used tinygrad
`33755a34`; the comparison used PyTorch 2.8.0, PyG 2.8.0, and PyG Temporal
source `fe555bc3`.

Each path receives ten warmups followed by 50 synchronized wall-time samples.
Topology construction, data transfer, and compilation are excluded before
timing. TGCN includes eager and `torch.compile(fullgraph=True)` PyTorch paths;
Tinymesh reports eager and `TinyJit`. PyTorch reported four CPU threads; both
stacks otherwise used their defaults.

Values are minimum / median / p90 milliseconds:

| Device | Component | Tinymesh JIT | PyG eager |
| --- | --- | ---: | ---: |
| Metal/MPS | aggregation | `2.40 / 3.34 / 6.14` | `5.70 / 8.66 / 13.29` |
| Metal/MPS | GraphSAGE | `2.46 / 3.32 / 4.70` | `4.84 / 7.16 / 12.29` |
| CPU | aggregation | `0.92 / 1.62 / 2.43` | `1.00 / 1.44 / 2.30` |
| CPU | GraphSAGE | `3.23 / 5.15 / 7.25` | `1.53 / 2.69 / 5.02` |

The direct-component Metal rows use the second repetition. TGCN retains every
causal control:

| Device | TGCN implementation | Source graph calls | Minimum / median / p90 ms |
| --- | --- | ---: | ---: |
| Metal/MPS | Tinymesh JIT | 1 CSR pull | `4.60 / 6.06 / 8.32` |
| Metal/MPS | PyG Temporal eager | 3 COO scatters | `10.38 / 14.33 / 20.63` |
| Metal/MPS | PyG Temporal compiled | 3 COO scatters | `1.74 / 3.91 / 9.08` |
| Metal/MPS | PyTorch factored eager | 1 COO scatter | `5.69 / 8.42 / 11.85` |
| Metal/MPS | PyTorch factored compiled | 1 COO scatter | `1.20 / 2.81 / 5.55` |
| CPU | Tinymesh JIT | 1 CSR pull | `12.62 / 20.47 / 27.82` |
| CPU | PyG Temporal eager | 3 COO scatters | `11.82 / 20.81 / 32.14` |
| CPU | PyG Temporal compiled | 3 COO scatters | `10.52 / 21.33 / 37.91` |
| CPU | PyTorch factored eager | 1 COO scatter | `4.24 / 8.55 / 12.61` |
| CPU | PyTorch factored compiled | 1 COO scatter | `4.80 / 8.04 / 10.92` |

The first Metal repetition recorded medians of 5.90, 14.44, 3.28, 10.46, and
4.74 ms in the same row order. The repeated direction is stable even though
the tails remain workstation-sensitive.

## Interpretation

The parity checks are the durable result: all three Tinymesh components have a
concrete mapping to the established PyG stack. Factoring three propagations
into one lowers eager PyTorch Metal median by 28–41% across repetitions and CPU
median by 59%. The algebra is useful independently of Tinymesh.

Tinymesh's CSR pull plus TinyJit beats eager PyG Temporal on Metal, but compiled
PyTorch controls are faster. On CPU, the factored PyTorch control is far faster
than either framework's published cell. The parameter difference of 96 zeroed
graph biases is negligible; propagation count, sparse representation, and
execution mode explain the result.

The broad minimum-to-p90 ranges show that this workstation was not an isolated
benchmark host. No ratio here supports a framework-level speed claim. The
balanced degree-nine recurrent graph also favors CSR pull; a high-degree hub
serializes one long row and may reverse that tradeoff.

## Limits

This record covers forward float32 execution over one balanced fixed graph. It
does not measure backward, optimizer steps, peak memory, compilation latency,
topology construction, changing topology, edge features, graph skew, multiple
shapes, batch throughput, or compiled GraphSAGE.

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
