# Mean GraphSAGE experiment

This record binds one learning result to an exact tinygrad revision. Read
[Message passing](../concepts/message-passing.md) first for the stable theory.

## Decision

At tinygrad revision
[`0bb36c9`](https://github.com/tinygrad/tinygrad/tree/0bb36c99899ba4742dbe1d5784397373998d81c3),
an experimental mean-GraphSAGE layer composes tinygrad linear maps with
tinymesh's CSR aggregation and trains on CPU and Metal.

The result proves composition and first-order parameter learning, not useful
model quality. At that revision the caller remained under `experiments/`.
After independent GCN, GAT, temporal, gradient, shape, and sparse-work
evidence established one reusable owner, the equation became
`tinymesh.nn.SAGEConv`; this experiment retains the learning witness.

## Learning witness

The experiment has two source nodes with features `1` and `-1`, two target nodes
with identical feature `0`, and edges:

```text
0 -> 2
1 -> 3
```

The targets are `1` and `-1`. A root-only function cannot distinguish nodes `2`
and `3`; their own features are identical. The neighbor parameter can:

```text
prediction at node 2 = W_neighbor * 1
prediction at node 3 = W_neighbor * -1
```

Starting both weights at zero gives loss `1` and neighbor gradient `-2`. One SGD
step at learning rate `0.5` sets the neighbor weight to `1` and the loss to `0`;
the root weight remains `0`.

At the recorded revision, the linear map ran before the mean. The output
gradient began at nodes `2` and `3`; transpose CSR returned it to nodes `0` and
`1`, then tinygrad differentiated those messages into `W_neighbor`.

The public class now computes the mean before the linear map so an optional bias
is applied once after aggregation. The witness disables bias, so its values and
parameter gradient are unchanged. On the current path the parameter gradient
consumes the sparse aggregate directly; dedicated `Graph.sum` tests retain the
transpose-CSR gradient evidence.

Fixed topology reuses realized CSR buffers per device. Integer degree remains a
lazy difference of cached row pointers. The layer derives inverse degree with
ordinary tinygrad operations, so topology owns only topology facts and
normalization does not enter model parameters.

## Limits

The witness says nothing about optimization difficulty, generalization, depth,
temporal learning, or a real dataset. The experiment deliberately omits bias,
activation, normalization, sampling, batching, and alternative aggregators.
PyG's pinned
[`SAGEConv`](https://github.com/pyg-team/pytorch_geometric/blob/2.8.0/torch_geometric/nn/conv/sage_conv.py)
implements that broader production surface.

The GraphSAGE paper is available as
[PDF](https://arxiv.org/pdf/1706.02216) and
[TeX source](https://arxiv.org/src/1706.02216).

## Reproduce

```console
DEV=CPU uv run --locked python -m unittest tests.test_mean_sage
DEV=METAL uv run --locked python -m unittest tests.test_mean_sage
uv run --locked python -m experiments.run mean_sage DEV=CPU
uv run --locked python -m experiments.run mean_sage DEV=METAL
```
