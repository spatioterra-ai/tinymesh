# Sparse aggregation feasibility

## Decision

At Tinygrad revision
[`b1a7229`](https://github.com/tinygrad/tinygrad/tree/b1a72299ab3f2c94eceb644f1ed43117149679d1),
a Tinymesh-owned destination-CSR kernel passes the identity-message correctness,
gradient, and sparse-structure gates on CPU and Metal. Forward computes `A @ X`;
its custom gradient runs the same kernel over transpose CSR to compute
`A^T @ dY`. Neither direction uses atomics or constructs node-edge state.

This resolves the implementation boundary without adding a Tinygrad primitive,
but not the public API. The candidate remains in `experiments/`: it relies on
the alpha `Tensor.custom_kernel` surface and a newly landed data-dependent
`Ops.LOOP`; Tinygrad's default kernel option search still raises
`KeyError: dtypes.weakint`, so the kernel pins `opts_to_apply=()`. It covers
fixed-topology first-order sum aggregation only and serializes each CSR row
within one feature lane. Admit it to the package only after those contracts
prove stable enough for a real model caller.

## CSR result

One immutable topology stores both destination CSR and its transpose:

```text
forward:  destination row -> source columns -> sum source features
backward: source row      -> target columns -> sum target gradients
```

The transpose makes backward the same operation as forward. Preprocessing sorts
each row by column, so a fixed node labeling is deterministic regardless of COO
input order. Duplicate edges retain multiplicity; isolated rows return zero.
Hand-computed tests cover both directions, repeated sources and destinations,
duplicate edges, isolated nodes, empty topology, COO-order invariance, and
vertex-permutation equivariance on exact-valued fixtures on CPU and Metal. As
with any sequential floating-point reduction, relabeling can change summation
order and therefore the final rounding bits.

For balanced degree-eight graphs with feature width `H = 32`, the stored
topology has exactly `2E + 2(N + 1)` integer elements. Each direction launches
`N * H` lanes whose row loops make exactly `E * H` active edge visits:

| N | E | Topology elements | Forward lane work | Backward lane work |
|---:|---:|---:|---:|---:|
| 4,096 | 32,768 | 73,730 | 1,179,648 | 1,179,648 |
| 8,192 | 65,536 | 147,458 | 2,359,296 | 2,359,296 |
| 16,384 | 131,072 | 294,914 | 4,718,592 | 4,718,592 |

Doubling `N` and `E` therefore doubles useful lane work exactly and grows
topology storage linearly. This follows from the CSR representation and the
single dynamic row loop; it is the complexity proof. Tinygrad's counters cannot
observe data-dependent loop trips, so wall time is supporting evidence, not the
proof.

On a 32 GB Apple M4 MacBook Air running macOS 26.5.2, after five warmups and
across 20 samples, the largest cases produced these representative medians:

| Device | Topology | Forward (ms) | Backward (ms) |
|---|---|---:|---:|
| CPU | balanced | 3.81 | 4.03 |
| CPU | one destination hub | 6.86 | 4.67 |
| CPU | one source hub | 3.59 | 7.89 |
| Metal | balanced | 3.71 | 3.15 |
| Metal | one destination hub | 17.55 | 1.30 |
| Metal | one source hub | 2.41 | 17.38 |

The hub cases expose the present tradeoff: one high-degree row serializes its
edges across only `H` feature lanes. Preprocessing and compilation are excluded
from these kernel medians. Tinygrad's kernel option search is disabled as noted
above, and repeated balanced timings varied enough that no performance claim
should rest on the table alone.

The result does not yet cover weighted or edge-dependent messages, batching,
changing topology, higher-order gradients, or topology construction cost.
Empty graphs use Tinygrad's ordinary `values * 0` path because a custom kernel
cannot receive zero-length buffers. A one-node graph similarly reduces to
`values * E`, avoiding unnecessary traversal and a degenerate Metal loop scope.

## Rejected public composition

At the earlier Tinygrad revision
[`980748c`](https://github.com/tinygrad/tinygrad/tree/980748ccfc4e3900ac652d8451e2ead9bfb4d09a),
the smallest native candidate was:

```text
node state [N, H]
  -> gather source rows         [E, H]
  -> scatter_reduce(sum) by target [N, H]
```

It passed hand-computed forward values, isolated-node behavior,
finite-difference gradients, and vertex-permutation equivariance. It failed the
defining sparse invariant: work must scale with edges rather than node-edge
pairs.

### Evidence

For `E = 2N` and `H = 4`, doubling both `N` and `E` should approximately double
the work of an edge-linear implementation. The native candidate approaches a
fourfold increase instead:

| Device | N=16 | N=32 | N=64 | N=128 |
|---|---:|---:|---:|---:|
| CPU operations | 7,376 | 29,156 | 115,656 | 460,680 |
| Metal operations | 91,232 | 325,824 | 1,225,088 | 4,743,936 |

A fresh pinned-revision Metal rerun retained 24 `N * E * H` carriers at every
measured size. Tinygrad may fuse these rather than allocate the full tensors,
but fusion does not remove the quadratic work. A synthetic `N=30,000`,
`E=60,000`, `H=32` case therefore implies 57.6 billion logical
node-edge-feature lanes before useful message computation.

This matches the pinned implementation: `gather` uses a one-hot mask and
`scatter_reduce` expands source values and a destination mask before reduction
([source](https://github.com/tinygrad/tinygrad/blob/980748ccfc4e3900ac652d8451e2ead9bfb4d09a/tinygrad/mixin/op.py#L1011-L1119)).
The current upstream scatter-reduce comparisons remain forward-only
([tests](https://github.com/tinygrad/tinygrad/blob/980748ccfc4e3900ac652d8451e2ead9bfb4d09a/test/backend/test_ops.py#L3130-L3141)),
although this experiment observes correct sum gradients on its small cases.

Tinygrad's atomic embedding backward was not a complete tested alternative. It
is a custom UOp kernel limited to CPU and AMD, while `Tensor.custom_kernel` is
marked alpha; its forward lookup still uses one-hot reduction
([implementation](https://github.com/tinygrad/tinygrad/blob/980748ccfc4e3900ac652d8451e2ead9bfb4d09a/tinygrad/nn/__init__.py#L307-L396)).
This leaves custom UOps as a research candidate rather than an established
library path; it does not assign the solution to either repository.

## Precursor custom-kernel probe

An isolated 2026-07-21 probe at Tinygrad
[`f64f96ec`](https://github.com/tinygrad/tinygrad/tree/f64f96ec596082c5230cda9471b54af7b88b58cd)
made the unresolved boundary narrower. Current source still uses one-hot
expansion for public `gather` and `scatter_reduce`, while
`Tensor.custom_kernel` remains alpha
([indexed operations](https://github.com/tinygrad/tinygrad/blob/f64f96ec596082c5230cda9471b54af7b88b58cd/tinygrad/mixin/op.py#L1008-L1117),
[custom kernels](https://github.com/tinygrad/tinygrad/blob/f64f96ec596082c5230cda9471b54af7b88b58cd/tinygrad/tensor.py#L166-L172)):

| Path | Observed evidence | Remaining gap |
|---|---|---|
| Public `gather + scatter_reduce(sum)` | For `N=16,32,64,128`, `E=2N`, and `H=4`, CPU forward work grew `7,376 -> 29,156 -> 115,656 -> 460,680`; squared-loss gradient-evaluation work, including that forward computation, grew `13,616 -> 54,052 -> 214,600 -> 855,176` | Still node-edge scaling; backward-only work was not isolated |
| Direct UOp gather | Existing `INDEX`, `LOAD`, and `STORE` operations produced the expected small result on CPU and Metal | Alpha custom-kernel boundary; no integrated gradient or scaling result |
| Destination-CSR sum | One global worker per destination looped over its stored row and produced `[3, 6, 11, 0]` from values `[3, 1, 5, 4, 7]` and row pointers `[0, 1, 3, 5, 5]` on CPU and Metal | Backward, degree-skew performance, and default-optimized and instrumented execution were unproven |

The direct probes used existing UOps through `Tensor.custom_kernel`; they did
not require a new low-level operation. The CSR probe avoided atomics by giving
each destination row one writer. The checked-in experiment above completes its
first-order gradient with transpose CSR.

At that revision, default kernel optimization failed with
`KeyError: dtypes.weakint` unless options were fixed explicitly, and normal
statistics collection raised
`TypeError: _f() missing 1 required positional argument: 'core_id'` for the
data-dependent row loop. Tinygrad revision `b1a7229` supplies the loop form used
by the checked-in experiment, but default kernel optimization still fails with
the same `KeyError`, so the candidate pins `opts_to_apply=()`. The alpha API,
disabled option search, and unmeasured compilation and topology-preprocessing
costs remain limitations.

## How the PyTorch stack expresses it

PyTorch Geometric 2.8.0 uses the same semantic lowering Tinymesh needs, but its
backend primitives are genuinely indexed:

```text
node state [N, H]
  -> torch.index_select(source)       [E, H]
  -> message                          [E, H]
  -> zeros[N, H].scatter_add(target)  [N, H]
```

`MessagePassing` obtains source or target rows with `Tensor.index_select`
([PyG source](https://github.com/pyg-team/pytorch_geometric/blob/2.8.0/torch_geometric/nn/conv/message_passing.py#L263-L328)).
Its sum and mean aggregations use PyTorch's in-place `scatter_add_`
([PyG source](https://github.com/pyg-team/pytorch_geometric/blob/2.8.0/torch_geometric/utils/_scatter.py#L67-L80)).
The expanded target index has shape `[E, H]`; it never introduces an `N * E`
axis. For compatible layers and sparse adjacency inputs, PyG can instead fuse
message and aggregation into sparse matrix multiplication
([PyG source](https://github.com/pyg-team/pytorch_geometric/blob/2.8.0/torch_geometric/utils/_spmm.py#L12-L131)).

PyTorch owns the corresponding backward operations. The source gradient of
`scatter_add` is a gather
([PyTorch source](https://github.com/pytorch/pytorch/blob/v2.12.0/tools/autograd/derivatives.yaml#L1518-L1522));
the input gradient of `index_select` is a zero tensor followed by `index_add_`
([PyTorch source](https://github.com/pytorch/pytorch/blob/v2.12.0/aten/src/ATen/native/TensorAdvancedIndexing.cpp#L1878-L1890)).
Both directions therefore touch node or edge feature lanes, not every
node-edge pair. PyG owns graph semantics and orchestration; PyTorch owns indexed
loads, accumulation, autograd, and device kernels.

PyTorch Geometric Temporal adds no universal temporal graph kernel. Its static
signal container returns one ordinary PyG `Data` snapshot at a time while
reusing one `edge_index`
([source](https://github.com/benedekrozemberczki/pytorch_geometric_temporal/blob/fe555bc30ee197755c4b58a89407033a5f383415/torch_geometric_temporal/signal/static_graph_temporal_signal.py#L14-L134)).
The recurrent model then composes PyG operations across time:

- `GConvGRU` performs six `ChebConv` calls per time step—two for each GRU
  gate—and each Chebyshev order adds another sparse propagation
  ([cell](https://github.com/benedekrozemberczki/pytorch_geometric_temporal/blob/fe555bc30ee197755c4b58a89407033a5f383415/torch_geometric_temporal/nn/recurrent/gconv_gru.py#L55-L170),
  [convolution](https://github.com/pyg-team/pytorch_geometric/blob/2.8.0/torch_geometric/nn/conv/cheb_conv.py#L142-L182)).
  It remains edge-linear, with work roughly proportional to `T * K * E` times
  feature widths and gate constants.
- The ordinary `DCRNN` implementation is not sparse end to end. Its `DConv`
  constructs an `[N, N]` adjacency with `to_dense_adj` for degree calculation
  and reverse-edge construction before calling sparse `propagate`
  ([source](https://github.com/benedekrozemberczki/pytorch_geometric_temporal/blob/fe555bc30ee197755c4b58a89407033a5f383415/torch_geometric_temporal/nn/recurrent/dcrnn.py#L42-L111)).
  `DCRNN` invokes that layer for all three recurrent gates. The separate
  `BatchedDConv` path instead computes degrees with `scatter_add_` and reverses
  the edge list directly
  ([source](https://github.com/benedekrozemberczki/pytorch_geometric_temporal/blob/fe555bc30ee197755c4b58a89407033a5f383415/torch_geometric_temporal/nn/recurrent/dcrnn.py#L258-L325)).

The lesson is narrower than “PyG Temporal is sparse”: PyG's primitive path is
edge-linear, but every temporal architecture must still be audited for dense
normalization, adjacency conversion, batching, and repeated topology work.

## Reproduce

```console
uv run python -m unittest tests.test_sparse_aggregation
DEV=CPU uv run python -m unittest tests.test_csr_aggregation
DEV=METAL uv run python -m unittest tests.test_csr_aggregation
uv run python experiments/sparse_aggregation.py
DEV=CPU uv run python experiments/csr_aggregation.py
DEV=METAL uv run python experiments/csr_aggregation.py
```

The CSR experiment satisfies the structural gate for fixed-topology,
identity-message, first-order sum aggregation in both directions. More complex
messages add edge-local costs and require their own proof. Public package
admission is a separate gate: a real model caller must justify depending on the
alpha custom-kernel and dynamic-loop contracts, or a simpler stable Tinygrad
surface must replace them. The rejected public gather-and-scatter composition
must not be called sparse graph support.
