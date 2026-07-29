# Chickenpox temporal data

The first external temporal caller is the Hungary chickenpox signal published
with PyTorch Geometric Temporal. It is small, public, fixed-topology, and used
by that project's T-GCN and GConvGRU examples.

## Source contract

Tinymesh downloads the dataset at PyTorch Geometric Temporal revision
[`fe555bc`](https://github.com/benedekrozemberczki/pytorch_geometric_temporal/blob/fe555bc30ee197755c4b58a89407033a5f383415/dataset/chickenpox.json)
and verifies SHA-256
`724b48cfb274b2ecbb855bdb99b970b5ef9dd3671694fa477435dc1e08293735`.
The source contains:

```text
521 ordered weekly rows x 20 counties
102 directed edges, including 20 self-loops
one name -> row mapping for stable node identity
```

With four lags, lowering is:

```text
source FX [521, 20]
          |
          +--> x[t, node] = FX[t:t+4, node]  -> [517, 20, 4]
          |
          +--> y[t, node] = FX[t+4, node]    -> [517, 20, 1]

source edges + node rows
          |
          +--> one Graph + unit edge weights
```

The loader preserves source edge order and self-loops. A model may request a
different loop convention, but that is an explicit graph transform rather than
a hidden data mutation.

## The data boundary

`StaticGraphTemporalSignal` owns one `Graph`, stable node IDs, stacked `x` and
`y` tensors, and optional COO-aligned scalar edge weights. Integer indexing
returns `(x_t, y_t)`; contiguous slicing and `split()` reuse the same graph and
edge weights.

```python
from tinymesh.datasets import chickenpox

signal = chickenpox(lags=4, device="CPU")
train, test = signal.split(0.8)

x, y = train[0]
print(x.shape, y.shape)
# (20, 4) (20, 1)
```

The tensor axes are fixed:

```text
x  [time, node, feature]
y  [time, node, target]
```

This container now pays rent: it rejects time, node, edge, dtype, and device
misalignment that a tuple of tensors could not name. It does not invent dates,
masks, or irregular-time semantics absent from this source.

## Reference parity

Against `torch-geometric-temporal==0.56.2`, all `517` feature and target
snapshots match under NumPy's default `assert_allclose`; edge indices and unit
weights match in source order. The full pinned-source witness reports the same
contract on CPU and Metal:

```console
uv run --locked python -m experiments.run chickenpox_data DEV=CPU
uv run --locked python -m experiments.run chickenpox_data DEV=METAL
```

```json
{
  "device": "CPU",
  "nodes": 20,
  "edges": 102,
  "self_loops": 20,
  "snapshots": 517,
  "train_snapshots": 413,
  "test_snapshots": 104,
  "x_shape": [20, 4],
  "y_shape": [20, 1]
}
```

## Causal window batches

For recurrent models, load one feature lag and make temporal history explicit:

```python
signal = chickenpox(lags=1, device="CPU")
values, target = next(signal.batches(batch_size=32, history=8))

print(values.shape, target.shape)
# (32, 8, 20, 1) (32, 20, 1)
```

Each sample contains eight consecutive node fields and predicts the week after
the last field. The final short batch is retained. Topology is not repeated:
every batch reuses `signal.graph`.

PyG's `TemporalDataLoader` batches continuous edge events, not ordered fields
over one fixed graph. At the pinned revision, PyG Temporal's `IndexDataset`
uses a PyTorch `DataLoader` to gather an input sequence and an equally long
future target sequence. Tinymesh instead exposes sequence-to-one windows
directly from the signal because that is the first current model caller:

```text
PyG Temporal IndexDataset   input [B, L, N, F] -> target [B, L, N, F]
Tinymesh batches            input [B, L, N, F] -> target [B, N, Y]
```

This is not a generic worker, shuffle, prefetch, or multiple-graph loader. It
is the smallest deterministic window contract over resident tinygrad tensors.
The [Chickenpox forecast](chickenpox-forecast.md) records the first end-to-end
training result and its limits.
