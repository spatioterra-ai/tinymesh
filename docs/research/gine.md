# GINE experiment

This stage asks whether vector-valued edge state needs another sparse backend
or can complete the existing message-passing algebra.

## Composition

```text
x_u [N, F] -- source gather --+
                                +-- ReLU -- target sum --+-- MLP --> y_v
e_uv [E, D] -- linear to F ----+                        |
                                                        +-- (1 + eps) x_v
```

`Graph.edge_values` already gathers node state into original COO edge order.
`Graph.sum_edges` exposes the CSR segment sum already used privately by target
softmax. The layer adds only ordinary tinygrad linear maps and ReLU:

```text
m(u -> v) = ReLU(x_u + W_edge e_uv)
y_v       = MLP((1 + epsilon) x_v + sum(m(u -> v)))
```

This follows GINE from [Hu et al.](https://arxiv.org/abs/1905.12265v3). The
[pinned PyG implementation](https://github.com/pyg-team/pytorch_geometric/blob/726310a486eae37a89cd6359072b82bbbbb71579/torch_geometric/nn/conv/gin_conv.py#L104-L210)
exposes the same edge projection, ReLU message, sum, self term, and
caller-supplied update network. Tinymesh owns a fixed two-linear update instead
of a module protocol. [Graph-JEPA's official implementation](https://github.com/geriskenderi/graph-jepa/blob/72df1b7704921001ea012a21f840300fbc792cdd/core/model.py)
uses GINE as its patch encoder, making MUTAG bond state the first live caller.

## Decision

Pending clean revision-bound CPU and Metal runs. Correct edge reduction and
GINE enter the public API independently of downstream model quality; patch
extraction, self-supervision, and representation evaluation remain experiments.

## Limits

The public layer accepts one homogeneous `[N, F]` node tensor and one aligned
`[E, D]` edge tensor. Its epsilon is fixed, its edge projection is always
learned, and its update is a two-linear ReLU MLP. Heterogeneous graphs,
trainable epsilon, leading model axes, edge-conditioned attention, and model
quality are not claimed.

## Reproduce

```console
DEV=CPU uv run --locked python -m unittest tests.test_edge_sum tests.test_gine
DEV=METAL uv run --locked python -m unittest tests.test_edge_sum tests.test_gine
uv run --locked python -m experiments.run gine DEV=CPU
uv run --locked python -m experiments.run gine DEV=METAL
```
