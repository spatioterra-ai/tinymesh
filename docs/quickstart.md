# Quick start

This guide starts with one directed graph, runs sparse aggregation, follows its
gradient, and trains two graph layers. It assumes basic Python and tensor
knowledge.

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
from tinygrad import Tensor
from tinymesh import Graph

graph = Graph(4, source=[1, 0, 1], target=[3, 2, 2])
state = Tensor([[2.0], [4.0], [8.0], [16.0]], device="CPU").realize()

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

The integer degree tensor is a cached topology fact. No `4 x 4` adjacency
matrix is constructed. Read
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
edge_weight = Tensor([-1.0, 0.5, 2.0], device="CPU").realize()

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

## Train mean GraphSAGE

The first model caller is mean GraphSAGE:

```text
neighbor = mean(W_neighbor x_u for every edge u -> v)
output   = W_root x_v + neighbor
```

Run its one-step learning witness:

```console
DEV=CPU uv run python -m experiments.mean_sage
```

It starts with loss `1`, sends the gradient through transpose CSR, updates only
the neighbor weight, and reaches loss `0`:

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

This proves that a tinygrad parameter can learn through the current sparse
boundary. It does not prove model quality, generalization, temporal learning,
or a stable model API. Read [Message passing](concepts/message-passing.md) for
the layer decomposition and [Mean GraphSAGE experiment](research/mean-sage.md)
for the exact witness.

## Reuse the sum in GCN

The second caller adds symmetric degree normalization around the same sparse
sum:

```text
output = D^-1/2 A D^-1/2 XW
```

Run its one-step witness:

```console
DEV=CPU uv run python -m experiments.gcn
```

For unit edges, the source and destination factors are node-wise tensor
operations. The existing CSR operation still owns only the sum; GCN owns its
normalization and linear map. Read [GCN experiment](research/gcn.md) for the
dense-reference, permutation, and learning evidence.

## Read the source

The implementation is intentionally small:

- [`src/tinymesh/graph.py`](https://github.com/spatioterra-ai/tinymesh/blob/main/src/tinymesh/graph.py)
  owns public graph identity, validation, and methods;
- [`src/tinymesh/_csr.py`](https://github.com/spatioterra-ai/tinymesh/blob/main/src/tinymesh/_csr.py)
  owns private lowering, device caches, sparse forward, and sparse backward;
- [`experiments/csr_aggregation.py`](https://github.com/spatioterra-ai/tinymesh/blob/main/experiments/csr_aggregation.py)
  retains the revision-bound CSR benchmark;
- [`experiments/mean_sage.py`](https://github.com/spatioterra-ai/tinymesh/blob/main/experiments/mean_sage.py)
  composes the first trainable model;
- [`experiments/gcn.py`](https://github.com/spatioterra-ai/tinymesh/blob/main/experiments/gcn.py)
  composes the normalized second caller;
- [`experiments/weighted_aggregation.py`](https://github.com/spatioterra-ai/tinymesh/blob/main/experiments/weighted_aggregation.py)
  records weighted forward and both gradients;
- [`tests/test_graph.py`](https://github.com/spatioterra-ai/tinymesh/blob/main/tests/test_graph.py)
  with [`tests/test_mean_sage.py`](https://github.com/spatioterra-ai/tinymesh/blob/main/tests/test_mean_sage.py)
  [`tests/test_gcn.py`](https://github.com/spatioterra-ai/tinymesh/blob/main/tests/test_gcn.py),
  and [`tests/test_weighted_aggregation.py`](https://github.com/spatioterra-ai/tinymesh/blob/main/tests/test_weighted_aggregation.py)
  state the current contracts.

`Graph` is public but experimental. The two model callers remain experiments,
and the alpha tinygrad execution boundary is not a stable contract.
