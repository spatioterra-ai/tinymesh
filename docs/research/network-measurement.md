# Network measurement

Tinymesh's dataset cards described node and edge identity but did not answer
whether their graphs were connected, mutually reachable, or locally shallow.
Those are data assumptions, not properties implied by sparse message passing.

## Measurement contract

The experiment measures each public `Graph` exactly:

```text
ordered COO topology
        |
        +--> unique non-self adjacency O(N + E)
                    |
                    +--> weak + strong components
                    +--> in/out neighbor distributions
                    +--> exact directed BFS from every node
                                  |
                                  +--> reachability + hop-distance distribution
```

Self-loops and duplicate COO rows remain separately visible. Components,
degrees, and distances use unique non-self neighbors: a node with only a
self-loop is structurally isolated. Directed distance excludes self-pairs and
unreachable pairs. `p90` is the nearest-rank 90th percentile.

The implementation stores no dense adjacency or distance matrix. Its storage
is `O(N + E)`, while exact all-pairs traversal costs `O(N(N + E))`. That bound
is suitable for the current graphs, whose largest node count is 675, but it is
not a network-scale training primitive.

## Reference boundary

The pinned libraries separate these concerns differently:

- PyG [`Data.connected_components`](https://github.com/pyg-team/pytorch_geometric/blob/5c6461b2305ad068a6d61165b3c55852a11aaa41/torch_geometric/data/data.py#L908)
  returns component subgraphs through union-find, while
  [`LargestConnectedComponents`](https://github.com/pyg-team/pytorch_geometric/blob/5c6461b2305ad068a6d61165b3c55852a11aaa41/torch_geometric/transforms/largest_connected_components.py)
  delegates explicit weak or strong connectivity to SciPy. Its
  [`k_hop_subgraph`](https://github.com/pyg-team/pytorch_geometric/blob/5c6461b2305ad068a6d61165b3c55852a11aaa41/torch_geometric/utils/_subgraph.py#L249)
  serves sampled computation, not dataset measurement.
- PyG Temporal's
  [`StaticGraphTemporalSignal`](https://github.com/benedekrozemberczki/pytorch_geometric_temporal/blob/fe555bc30ee197755c4b58a89407033a5f383415/torch_geometric_temporal/signal/static_graph_temporal_signal.py)
  carries one edge index through time but does not characterize its topology.
- TSL's
  [`Dataset.get_connectivity`](https://github.com/TorchSpatiotemporal/tsl/blob/aa5f313e000d192bdec270748b8d01df5912e58e/tsl/datasets/prototypes/dataset.py#L380)
  derives thresholded, k-nearest, symmetric, normalized, or dense connectivity
  from similarities. Tinymesh keeps each source-specific graph recipe in its
  dataset adapter instead of importing that policy surface.

Tinymesh therefore borrows the distinction between weak and strong
connectivity, not the generic containers, transforms, dense conversions, or
dependency stack.

## Dataset observations

Revision
[`586f5ca`](https://github.com/spatioterra-ai/tinymesh/tree/586f5ca0f843f7d1f44e97e0a5528246e1fad6ed)
produced this CPU witness:

```console
uv run --locked python -m experiments.run network_measurement
```

| Dataset | Nodes / edges | Weak components | Strong components (largest) | Reachable ordered pairs | Directed distance mean / p90 / max |
| --- | ---: | ---: | ---: | ---: | ---: |
| Chickenpox | 20 / 102 | 1 | 1 (20) | 380 / 380 (100%) | 2.51 / 4 / 6 |
| Montevideo | 675 / 690 | 1 | 675 (1) | 77,427 / 454,950 (17.0%) | 40.47 / 71 / 114 |
| METR-LA | 207 / 1,722 | 2 | 13 (195) | 40,008 / 42,642 (93.8%) | 5.85 / 10 / 17 |

MUTAG's 188 molecular graphs are all weakly connected and contain no isolated
nodes. Because every bond is stored in both directions, weak and directed
distance agree; per-graph directed diameter has mean 8.22, p90 11, and maximum
15.

The degree evidence sharpens the interpretation:

| Dataset | Non-self in-degree mean / p90 / max | Non-self out-degree mean / p90 / max | Sources / sinks / isolates |
| --- | ---: | ---: | ---: |
| Chickenpox | 4.10 / 6 / 7 | 4.10 / 6 / 7 | 0 / 0 / 0 |
| Montevideo | 1.02 / 1 / 3 | 1.02 / 1 / 3 | 9 / 7 / 0 |
| METR-LA | 7.32 / 13 / 17 | 7.32 / 13 / 18 | 1 / 4 / 1 |

Montevideo is one weak component but a directed acyclic relation: it has no
reciprocal edge and every node is its own strong component. A single forward
message step is local, while repeated forward propagation cannot connect most
ordered pairs. Reverse diffusion is therefore a deliberate second relation,
not a property already present in the source graph.

METR-LA's 207 self-loops do not make every sensor structurally connected. One
sensor has no non-self neighbor, the weak graph has a 206-node component plus
that isolate, and the directed graph has one 195-node strong core plus 12
smaller components. Model evaluation should retain every sensor but must not
describe the affinity as fully connected.

## Decision

Add the exact measurement as a research-only evidence owner. Do not add
components, shortest paths, k-hop sampling, similarity graphs, or a general
analytics namespace to `tinymesh.Graph`:

```text
src/tinymesh.Graph              sparse differentiable message algebra
tinymesh.datasets               source-specific topology truth
experiments.network_measurement exact bounded topology evidence
```

The public core is not missing a primitive required by its current learning
callers. A runtime subgraph primitive becomes justified when a second
non-research caller needs aligned node and edge selection. Approximate or
sampled distance becomes justified only when a larger dataset question cannot
afford this exact witness.
