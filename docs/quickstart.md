# Quick start

This guide starts with one directed graph, runs sparse aggregation, follows its
gradient, and reaches the temporal model experiments. It assumes basic Python
and tensor knowledge.

`Graph` is an experimental 0.x API, not a stability promise. Run this guide
from a repository checkout with [uv](https://docs.astral.sh/uv/) and Python
3.11 or newer:

```console
git clone https://github.com/spatioterra-ai/tinymesh.git
cd tinymesh
uv sync --locked
```

## Aggregate a graph

Start with four nodes and three directed edges:

```text
0 ---> 2 <--- 1
              |
              v
              3
```

Each node has one scalar feature:

```python
from tinygrad import Device, Tensor
from tinymesh import Graph

graph = Graph(4, source=[1, 0, 1], target=[3, 2, 2])
state = Tensor([[2.0], [4.0], [8.0], [16.0]], device=Device.DEFAULT).realize()

output = graph.sum(state)
print(output.tolist())
# [[0.0], [0.0], [6.0], [4.0]]
```

The operation is:

```text
output[v] = sum(state[u] for every edge u -> v)
```

Node `2` receives `2 + 4`. Node `3` receives `4`. The operation aggregates
incoming neighbors; it does not add a node's own value unless the graph
contains a self-edge.

## See the sparse representation

`Graph` keeps ordered COO edge identity visible:

```python
print(graph.source, graph.target, graph.edges)
# ((1, 0, 1), (3, 2, 2), 3)

print(graph.in_degree(device=state.device).tolist())
# [0, 0, 2, 1]
```

Execution privately lowers those edges to compressed sparse row (CSR) so each
destination visits only its incoming neighbors:

```text
destination  incoming sources
0            []
1            []
2            [0, 1]
3            [1]
```

The integer degree tensor is derived lazily from cached CSR row pointers. No
`4 x 4` adjacency matrix is constructed. Read
[Sparse graph topology](concepts/topology.md) for COO, CSR, transpose, and why
the current kernel pulls rather than pushes messages.

## Differentiate it

Every output contributes a gradient of one:

```python
output = graph.sum(state)
gradient = output.sum().gradient(state)[0]
print(gradient.tolist())
# [[1.0], [2.0], [0.0], [0.0]]
```

Node `0` contributes to one output, so its gradient is `1`. Node `1`
contributes to two outputs, so its gradient is `2`.

Forward uses destination CSR to compute `A @ X`. Backward uses transpose CSR to
compute `A.T @ dY`. The same sparse sum implements both directions.

## Weight individual edges

Scalar edge values follow the `source` and `target` arrays passed to
`Graph`:

```python
edge_weight = Tensor([-1.0, 0.5, 2.0], device=state.device).realize()

weighted = graph.sum(state, edge_weight=edge_weight)
print(weighted.tolist())
# [[0.0], [0.0], [9.0], [-4.0]]
```

The first weight belongs to `1 -> 3`, the second to `0 -> 2`, and the third to
`1 -> 2`. Private maps select those weights after connectivity is grouped into
CSR; the public tensor stays in COO order.

Both inputs differentiate:

```python
node_gradient, edge_weight_gradient = weighted.sum().gradient(
    state,
    edge_weight,
)
print(node_gradient.tolist())
# [[0.5], [1.0], [0.0], [0.0]]
print(edge_weight_gradient.tolist())
# [4.0, 2.0, 4.0]
```

The edge-weight gradient follows the same COO order as `edge_weight`. Read
[Weighted aggregation experiment](research/weighted-aggregation.md) for the
formula, duplicate-edge evidence, and sparse-structure checks.

## Normalize attention over incoming edges

Node values can be projected to either endpoint in original COO order, then
normalized among edges with the same target:

```python
edge_score = graph.edge_values(state, endpoint="source").reshape(-1)
attention = graph.softmax(edge_score)
print(attention.tolist())
# [1.0, 0.11920292, 0.88079703]

attended = graph.sum(state, edge_weight=attention)
```

The first edge is the only edge ending at node `3`, so its attention is `1`.
The other two end at node `2` and normalize together. Both tensors retain the
original edge order: `1 -> 3`, `0 -> 2`, `1 -> 2`.

## Compose mean GraphSAGE

`SAGEConv` is a direct tinygrad-style object:

```text
neighbor = W_neighbor mean(x_u for every edge u -> v) + b_neighbor
output   = W_root x_v + neighbor
```

```python
from tinymesh.nn import SAGEConv

layer = SAGEConv(in_features=1, out_features=2)
output = layer(state, graph)
print(output.shape)
# (4, 2)
```

Run its one-step learning witness:

```console
uv run --locked python -m experiments.run mean_sage DEV=CPU
```

It starts with loss `1`, uses the sparse mean to distinguish the two target
nodes, updates only the neighbor weight, and reaches loss `0`:

```json
{
  "device": "CPU",
  "initial_loss": 1.0,
  "neighbor_gradient": -2.0,
  "final_loss": 0.0,
  "root_weight": 0.0,
  "neighbor_weight": 1.0
}
```

This proves that a tinygrad parameter can learn from the current sparse
aggregate. Dedicated `Graph.sum` gradient tests prove the transpose-CSR path.
It does not prove model quality, generalization, or temporal learning.
`tinymesh.nn` is still an experimental 0.x API. Read
[Message passing](concepts/message-passing.md) for the layer decomposition and
[Mean GraphSAGE experiment](research/mean-sage.md) for the exact witness.

## Reuse the sum in GCN

The second caller adds symmetric degree normalization around the same sparse
sum:

```text
output = (D^-1/2 A D^-1/2 X)W + b
```

Run its one-step witness:

```console
uv run --locked python -m experiments.run gcn DEV=CPU
```

For unit edges, the source and destination factors are node-wise tensor
operations. The existing CSR operation still owns only the sum; GCN owns its
normalization and linear map. Read [GCN experiment](research/gcn.md) for the
dense-reference, permutation, and learning evidence.

## Train graph attention

The third caller computes scalar source and target coefficients at nodes,
projects them to edges, normalizes by target, and reuses weighted sum:

```console
uv run --locked python -m experiments.run gat DEV=CPU
```

One SGD step lowers loss from `0.214323` to `0.126819` and updates the shared
attention parameter. Read [Sparse attention experiment](research/attention.md)
for the exact composition, gradient evidence, and limits.

A multi-head layer repeats normalization and aggregation independently, then
concatenates the node outputs:

```console
uv run --locked python -m experiments.run multi_head_gat DEV=CPU
```

The checked fixture gives two heads opposite initial attention parameters. One
step sends equal and opposite gradients to them and lowers loss from `0.214323`
to `0.168497`. No new `Graph` operation is involved.

## Carry state through time

Load the public PyTorch Geometric Temporal chickenpox signal without adding
PyTorch or NumPy to the runtime:

```python
from tinymesh.datasets import chickenpox

signal = chickenpox(lags=4, device=Device.DEFAULT)
train, test = signal.split(0.8)

print(len(train), len(test))
# 413 104
x, y = train[0]
print(x.shape, y.shape)
# (20, 4) (20, 1)
```

One `Graph` is shared by all `517` snapshots. Each input row holds four
previous weekly values for one county; each target row holds its following
value. The loader preserves all `102` source edges, including `20` self-loops.
Read the [Chickenpox data record](research/chickenpox-data.md) for the source,
lowering, parity, and window contract.

Recurrent models can instead expose history as a separate axis:

```python
sequence = chickenpox(lags=1, device=Device.DEFAULT)
values, target = next(sequence.batches(batch_size=32, history=8))

print(values.shape, target.shape)
# (32, 8, 20, 1) (32, 20, 1)
```

Every leading lane shares the same graph. `Graph.sum` folds those lanes into
feature width, runs one CSR operation, and restores `[B, ..., N, H]`.

Temporal recurrence reuses one graph while node fields and hidden state change:

```python
from tinymesh.nn import TGCN

temporal_graph = Graph(2, source=[0, 1, 0], target=[0, 1, 1])
snapshots = (
    Tensor([[1.0], [0.0]], device=Device.DEFAULT),
    Tensor([[0.0], [0.0]], device=Device.DEFAULT),
)
cell = TGCN(in_features=1, hidden_features=1)

hidden = cell(snapshots[0], temporal_graph)
for snapshot in snapshots[1:]:
    hidden = cell(snapshot, temporal_graph, hidden)
```

The topology and its CSR buffers stay fixed. One sparse reduction at input
width `F`, followed by one linear map to `3H` channels, computes the update,
reset, and candidate graph projections together. Run the checked two-snapshot
learning witness:

```console
uv run --locked python -m experiments.run tgcn DEV=CPU
```

The first snapshot crosses edge `0 -> 1`; the second contains no signal, so the
final prediction also requires temporal memory. One SGD step lowers loss from
`0.718740` to `0.686385`. Read [Time](concepts/time.md) for the data model and
[T-GCN experiment](research/tgcn.md) for the exact evidence.

## Move hidden state over the graph

T-GCN graph-mixes current input but keeps its recurrent gates node-local.
GConvGRU applies Chebyshev graph filters to both current input and hidden state:

```text
[update, reset] = Cheb_K([input, hidden])
candidate       = Cheb_K([input, reset * hidden])
```

Run the controlled comparison:

```console
uv run --locked python -m experiments.run gconv_gru DEV=CPU
```

Both cells see the same symmetric graph, two snapshots, hidden width, target,
and one full-model SGD step. With `K = 2`, T-GCN has `12` parameters and one
sparse call per step; GConvGRU has `15` parameters and two sparse calls. They
start at the same loss. Both learn, but their final toy losses are not a quality
ranking because their costs and optimization surfaces differ.

Read [GConvGRU experiment](research/gconv-gru.md) for the Chebyshev recurrence,
fusion, exact results, and limits.

## Train a real forecast

Run the controlled Chickenpox experiment:

```console
uv run --locked python -m experiments.run chickenpox_forecast DEV=CPU
```

It splits time before making eight-week windows, trains a node-local LSTM,
T-GCN, and GConvGRU with similar parameter counts, and reports MSE plus MAE
against zero and last-value baselines. Across seeds `0`, `1`, and `2`, LSTM and
GConvGRU are effectively tied; T-GCN trails both.

That result proves real batched training through the sparse boundary. It does
not yet show that the graph improves prediction. The exact configuration,
three-seed table, loop conventions, and data limitations live in the
[Chickenpox forecast](research/chickenpox-forecast.md).

## Read the source

The implementation is intentionally small:

- [`src/tinymesh/graph.py`](https://github.com/spatioterra-ai/tinymesh/blob/main/src/tinymesh/graph.py)
  owns public graph identity, validation, and methods;
- [`src/tinymesh/_csr.py`](https://github.com/spatioterra-ai/tinymesh/blob/main/src/tinymesh/_csr.py)
  owns private lowering, device caches, sparse forward, and sparse backward;
- [`src/tinymesh/nn/__init__.py`](https://github.com/spatioterra-ai/tinymesh/blob/main/src/tinymesh/nn/__init__.py)
  owns the direct spatial, temporal, and diffusion components;
- [`src/tinymesh/temporal.py`](https://github.com/spatioterra-ai/tinymesh/blob/main/src/tinymesh/temporal.py)
  owns aligned fixed-graph temporal signals;
- [`src/tinymesh/datasets.py`](https://github.com/spatioterra-ai/tinymesh/blob/main/src/tinymesh/datasets.py)
  owns pinned source lowering;
- [`experiments/__init__.py`](https://github.com/spatioterra-ai/tinymesh/blob/main/experiments/__init__.py)
  owns the evidence catalog; experiments retain training and observations;
- [`tests/`](https://github.com/spatioterra-ai/tinymesh/tree/main/tests)
  states the current contracts.

The public surface and alpha tinygrad execution boundary remain experimental.
