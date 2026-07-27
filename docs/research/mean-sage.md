# Mean GraphSAGE experiment

This record binds one learning result to an exact tinygrad revision. Read
[Message passing](../concepts/message-passing.md) first for the stable theory.

## Decision

At tinygrad revision
[`bdbb1d7`](https://github.com/tinygrad/tinygrad/tree/bdbb1d702f91c68ccfb0b93d93180b6f0947c7c1),
an experimental mean-GraphSAGE layer composes tinygrad linear maps with
tinymesh's CSR aggregation and trains on CPU and Metal.

The result proves composition and first-order parameter learning, not a stable
tinymesh API or useful model quality. The caller remains under `experiments/`
because the alpha custom-kernel boundary and disabled default kernel
optimization still make package admission premature.

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

The output gradient begins at nodes `2` and `3`. Transpose CSR returns it to
nodes `0` and `1`, then tinygrad differentiates their linear messages into
`W_neighbor`.

Fixed topology reuses realized CSR buffers per device and its inverse-degree
vector per device and dtype. The topology owns those derived caches, so they do
not enter model parameters.

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
DEV=CPU uv run python -m unittest tests.test_mean_sage
DEV=METAL uv run python -m unittest tests.test_mean_sage
DEV=CPU uv run python -m experiments.mean_sage
DEV=METAL uv run python -m experiments.mean_sage
```
