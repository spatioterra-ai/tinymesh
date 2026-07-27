# tinymesh

**Sparse structure through space and time, in tinygrad.**

tinymesh is an experimental library for learning over graphs and meshes with
[tinygrad](https://github.com/tinygrad/tinygrad). A graph is the smallest mesh:
sparse topology connects tensor fields. Geometry and time can extend that core
without changing it. tinymesh is tinygrad-native, not a compatibility layer
over another machine-learning framework.

## What works

The repository currently proves one narrow path:

```text
directed edge list
      |
      v
CSR(A) + CSR(A.T)
      |
      v
sparse forward + sparse backward
      |
      v
trainable mean GraphSAGE
```

- An unordered directed edge list lowers deterministically into destination CSR
  for forward propagation and transpose CSR for backward propagation.
- Sparse sum stores `O(N + E)` topology and each direction performs
  `O((N + E)H)` work for `N` nodes, `E` edges, and feature width `H`.
- One tinygrad custom kernel implements both `A @ X` and `A.T @ dY`; neither path
  constructs node-pair or node-edge state.
- A mean-GraphSAGE experiment sends gradients through the sparse boundary into
  a neighbor parameter on CPU and Metal.
- Fixed topology owns and reuses its realized device buffers.

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

The trainable witness starts with loss `1`, takes one SGD step, and reaches loss
`0` only through neighbor information:

```console
DEV=CPU uv run python -m experiments.mean_sage
DEV=METAL uv run python -m experiments.mean_sage
```

## Learn

Start with the [documentation map](docs/index.md):

- [Sparse graph topology](docs/concepts/topology.md) explains COO, CSR,
  transpose, lowering, and the push-pull tradeoff.
- [Message passing](docs/concepts/message-passing.md) explains
  message -> aggregate -> update and the gradient path.
- [Sparse aggregation feasibility](docs/research/sparse-aggregation.md) retains
  the revision-bound scaling and kernel evidence.
- [Mean GraphSAGE experiment](docs/research/mean-sage.md) retains the exact
  learning witness and its limits.

## Direction

The next decision is the smallest public topology and aggregation contract that
survives a second, genuinely different model caller. Weighted or edge-dependent
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

The pinned submodules are optional, reference-only source for studying tinygrad,
PyTorch Geometric, and PyTorch Geometric Temporal:

```console
git submodule update --init
```

See [CONTRIBUTING.md](CONTRIBUTING.md) before changing code.
