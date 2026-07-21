# Sparse aggregation feasibility

## Decision

Sparse message passing remains blocked at Tinygrad revision
[`980748c`](https://github.com/tinygrad/tinygrad/tree/980748ccfc4e3900ac652d8451e2ead9bfb4d09a).
No graph primitive enters the Tinymesh package in this stage.

The smallest native candidate is:

```text
node state [N, H]
  -> gather source rows         [E, H]
  -> scatter_reduce(sum) by target [N, H]
```

It passes hand-computed forward values, isolated-node behavior, finite-difference
gradients, and vertex-permutation equivariance. It fails the defining sparse
invariant: work must scale with edges rather than node-edge pairs.

## Evidence

For `E = 2N` and `H = 4`, doubling both `N` and `E` should approximately double
the work of an edge-linear implementation. The native candidate approaches a
fourfold increase instead:

| Device | N=16 | N=32 | N=64 | N=128 |
|---|---:|---:|---:|---:|
| CPU operations | 7,376 | 29,156 | 115,656 | 460,680 |
| Metal operations | 91,232 | 325,824 | 1,225,088 | 4,743,936 |

The lazy graph also contains `N * E * H` carriers. Tinygrad may fuse these rather
than allocate the full tensors, but fusion does not remove the quadratic work.
A synthetic `N=30,000`, `E=60,000`, `H=32` case therefore implies 57.6 billion
logical node-edge-feature lanes before useful message computation.

This matches the pinned implementation: `gather` uses a one-hot mask and
`scatter_reduce` expands source values and a destination mask before reduction
([source](https://github.com/tinygrad/tinygrad/blob/980748ccfc4e3900ac652d8451e2ead9bfb4d09a/tinygrad/mixin/op.py#L1011-L1119)).
The current upstream scatter-reduce comparisons remain forward-only
([tests](https://github.com/tinygrad/tinygrad/blob/980748ccfc4e3900ac652d8451e2ead9bfb4d09a/test/backend/test_ops.py#L3130-L3141)),
although this experiment observes correct sum gradients on its small cases.

Tinygrad's atomic embedding backward is not a general escape hatch. It is a
custom UOp kernel limited to CPU and AMD, while `Tensor.custom_kernel` is marked
alpha; its forward lookup still uses one-hot reduction
([implementation](https://github.com/tinygrad/tinygrad/blob/980748ccfc4e3900ac652d8451e2ead9bfb4d09a/tinygrad/nn/__init__.py#L307-L396)).
Depending on that path would create the private, backend-specific compiler layer
this stage explicitly excludes.

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
uv run python experiments/sparse_aggregation.py
DEV=CPU uv run python experiments/sparse_aggregation.py
```

The gate can reopen when Tinygrad exposes backend-neutral indexed gather and
segment-sum semantics whose forward and backward work are proportional to
`(N + E) * H`. Until then, Tinymesh can study tensor, temporal, and composition
contracts, but it must not call dense emulation sparse graph support.
