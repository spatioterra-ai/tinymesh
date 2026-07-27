# Quick start

This guide starts with one directed graph, runs sparse aggregation, follows its
gradient, and trains two graph layers. It assumes basic Python and tensor
knowledge.

tinymesh has no stable package API yet. Run this guide from a repository
checkout with [uv](https://docs.astral.sh/uv/) and Python 3.11 or newer:

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

from experiments.csr_aggregation import CSRTopology, csr_edge_sum

topology = CSRTopology(4, source=[0, 1, 1], target=[2, 2, 3])
state = Tensor([[2.0], [4.0], [8.0], [16.0]], device="CPU").realize()

output = csr_edge_sum(state, topology)
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

The edge list is convenient input. Execution uses compressed sparse row (CSR)
so each destination can visit only its incoming neighbors:

```python
print(topology.row_ptr)
# (0, 0, 0, 2, 3)

print(topology.column)
# (0, 1, 1)
```

The sources for destination `v` occupy:

```python
topology.column[topology.row_ptr[v]:topology.row_ptr[v + 1]]
```

No `4 x 4` adjacency matrix is constructed. Read
[Sparse graph topology](concepts/topology.md) for COO, CSR, transpose, and why
the current kernel pulls rather than pushes messages.

## Differentiate it

Every output contributes a gradient of one:

```python
output = csr_edge_sum(state, topology)
gradient = output.sum().gradient(state)[0]
print(gradient.tolist())
# [[1.0], [2.0], [0.0], [0.0]]
```

Node `0` contributes to one output, so its gradient is `1`. Node `1`
contributes to two outputs, so its gradient is `2`.

Forward uses destination CSR to compute `A @ X`. Backward uses transpose CSR to
compute `A.T @ dY`. The same sparse sum implements both directions.

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
or a stable API. Read [Message passing](concepts/message-passing.md) for the
layer decomposition and [Mean GraphSAGE experiment](research/mean-sage.md) for
the exact witness.

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

- [`experiments/csr_aggregation.py`](https://github.com/spatioterra-ai/tinymesh/blob/main/experiments/csr_aggregation.py)
  owns topology lowering, sparse forward, and sparse backward;
- [`experiments/mean_sage.py`](https://github.com/spatioterra-ai/tinymesh/blob/main/experiments/mean_sage.py)
  composes the first trainable model;
- [`experiments/gcn.py`](https://github.com/spatioterra-ai/tinymesh/blob/main/experiments/gcn.py)
  composes the normalized second caller;
- [`tests/test_csr_aggregation.py`](https://github.com/spatioterra-ai/tinymesh/blob/main/tests/test_csr_aggregation.py)
  with [`tests/test_mean_sage.py`](https://github.com/spatioterra-ai/tinymesh/blob/main/tests/test_mean_sage.py)
  and [`tests/test_gcn.py`](https://github.com/spatioterra-ai/tinymesh/blob/main/tests/test_gcn.py)
  state the current contracts.

These paths remain experimental. The two callers share topology and CSR sum,
but the alpha tinygrad execution boundary is not a stable public contract.
