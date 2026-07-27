# tinymesh

**Sparse structure through space and time, in tinygrad.**

tinymesh is an experimental library for learning over graphs and meshes with
[tinygrad](https://github.com/tinygrad/tinygrad). A graph is the smallest mesh:
sparse topology connects tensor fields. Geometry and time can extend that core
without changing it. tinymesh is tinygrad-native, not a compatibility layer
over another machine-learning framework.

[Documentation](https://spatioterra-ai.github.io/tinymesh/) |
[Quick start](https://spatioterra-ai.github.io/tinymesh/quickstart/) |
[Contributing](CONTRIBUTING.md)

## What works

The repository currently proves one narrow sparse core:

```text
directed edge list
      |
      v
CSR(A) + CSR(A.T)
      |
      v
sparse forward + sparse backward
      |                 |
      v                 v
mean GraphSAGE     unweighted GCN
```

- An unordered directed edge list lowers deterministically into destination CSR
  for forward propagation and transpose CSR for backward propagation.
- Sparse sum stores `O(N + E)` topology and each direction performs
  `O((N + E)H)` work for `N` nodes, `E` edges, and feature width `H`.
- One tinygrad custom kernel implements both `A @ X` and `A.T @ dY`; neither path
  constructs node-pair or node-edge state.
- A mean-GraphSAGE experiment sends gradients through the sparse boundary into
  a neighbor parameter on CPU and Metal.
- An unweighted GCN experiment composes source and destination degree scaling
  around the same sparse sum.
- Fixed topology owns and reuses its realized connectivity and degree buffers.

This is research code, not a stable API. The implementation remains under
`experiments/` because `Tensor.custom_kernel` is alpha and tinygrad's default
kernel optimization does not yet accept the data-dependent CSR loop.

## Run the proof

Install the exact locked tinygrad revision with
[uv](https://docs.astral.sh/uv/):

```console
uv sync --locked
```

Then run one sparse aggregation from the repository checkout:

```python
from tinygrad import Tensor

from experiments.csr_aggregation import CSRTopology, csr_edge_sum

topology = CSRTopology(4, source=[0, 1, 1], target=[2, 2, 3])
state = Tensor([[2.0], [4.0], [8.0], [16.0]], device="CPU").realize()

print(csr_edge_sum(state, topology).tolist())
# [[0.0], [0.0], [6.0], [4.0]]
```

Both trainable witnesses start with loss `1`, take one SGD step, and reach loss
`0` through graph propagation:

```console
DEV=CPU uv run python -m experiments.mean_sage
DEV=METAL uv run python -m experiments.mean_sage
DEV=CPU uv run python -m experiments.gcn
DEV=METAL uv run python -m experiments.gcn
```

## Learn

Start with the [documentation](https://spatioterra-ai.github.io/tinymesh/) or
run the [quick start](docs/quickstart.md):

- [Sparse graph topology](docs/concepts/topology.md) explains COO, CSR,
  transpose, lowering, and the push-pull tradeoff.
- [Message passing](docs/concepts/message-passing.md) explains
  message -> aggregate -> update and the gradient path.
- [Sparse aggregation feasibility](docs/research/sparse-aggregation.md) retains
  the revision-bound scaling and kernel evidence.
- [Mean GraphSAGE experiment](docs/research/mean-sage.md) retains the exact
  learning witness and its limits.
- [GCN experiment](docs/research/gcn.md) tests the same sparse sum under
  symmetric degree normalization.

## Direction

Mean GraphSAGE and GCN now share deterministic topology plus destination-CSR
sum. Topology caches only topology facts; each layer derives its own
normalization with ordinary tinygrad operations. The remaining package-admission
blocker is the alpha kernel and optimizer boundary. Weighted or edge-dependent
messages, batching, changing topology, and temporal recurrence remain
unimplemented.

Coordinates, coordinate-reference metadata, higher-dimensional cells, and
time-varying fields remain the wider mesh direction. They enter only when the
sparse graph core extends naturally; tinymesh is not a GIS, trainer framework,
application, or model zoo.

## Development

```console
uv sync --locked
uv run python -m unittest discover -s tests -p 'test_*.py'
uv build
```

Build or preview the documentation with the locked docs environment:

```console
uv run --locked --only-group docs zensical build --clean --strict
uv run --locked --only-group docs zensical serve
```

The pinned submodules are optional, reference-only source for studying tinygrad,
PyTorch Geometric, and PyTorch Geometric Temporal:

```console
git submodule update --init
```

See [CONTRIBUTING.md](CONTRIBUTING.md) before changing code.
