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
