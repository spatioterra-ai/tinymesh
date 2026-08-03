# MUTAG molecular graphs

MUTAG is Tinymesh's first collection of variable-size graphs. This stage only
establishes a pinned, sparse data boundary for graph-level experiments; it
makes no representation or classification claim.

## Pinned source

The loader fetches the canonical [TU Dortmund
archive](https://www.chrsmrrs.com/graphkerneldatasets/MUTAG.zip):

```text
MUTAG.zip
24,550 bytes
SHA-256 c419bdc853c367d2d83da4973c45100954ae15e10f5ae2cddde6ca431f8207f6
```

The default path uses a ten-second request, reads at most 32 KiB, and verifies
the exact checksum and byte count. An explicit ZIP path supports offline use
and fixtures; it remains archive- and member-size bounded and is validated
structurally. No dataset payload is committed or redistributed. The archive
README does not state a license, so Tinymesh makes no redistribution claim.

## Public boundary

```python
from tinymesh.datasets import mutag

data = mutag(device="CPU")
graph, atom, bond, label = data[0]
node_features = atom.one_hot(len(data.node_types)).float()

print(len(data), graph.nodes, graph.edges)
# 188 17 38
print(node_features.shape, bond.shape, label)
# (17, 7) (38,) 1
```

`MUTAG` owns four aligned facts per molecule:

```text
graph        sparse directed storage of an undirected molecular graph
node_labels  [N] categorical atom type in node_types order
edge_labels  [E] categorical bond type in bond_types order
labels       graph class remapped from source {-1, 1} to {0, 1}
```

Each chemical bond is represented by two matching directed edges because
`Graph` has one directed primitive. The loader preserves source edge order,
keeps edge labels aligned with COO identity, and rejects cross-graph edges,
self-loops, duplicates, missing reverse edges, invalid label domains, malformed
member rows, and oversized input.

The pinned PyG
[`TUDataset`](https://github.com/pyg-team/pytorch_geometric/blob/726310a486eae37a89cd6359072b82bbbbb71579/torch_geometric/datasets/tu_dataset.py)
reports the same 188 graphs, seven one-hot node features, four one-hot edge
features, and two classes. Tinymesh retains integer source labels as the single
fact; ordinary tinygrad `one_hot` derives model inputs.

## Full-source witness

Pending a clean revision-bound CPU and Metal run.

```console
uv run --locked python -m experiments.run mutag_data DEV=CPU
uv run --locked python -m experiments.run mutag_data DEV=METAL
```

The next stage will define train-only self-supervision, frozen probes, splits,
and controls. Those policies do not belong in the loader.
