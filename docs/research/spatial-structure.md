# Spatial structure

A graph is already spatial in the topological sense: its edges say which nodes
can interact. Metric space adds positions, displacement, and distance without
changing that connectivity.

## The contract

Keep the four facts separate:

```text
topology             G = (source, target)
coordinate frame     CRS, origin, axes, units
position             P[N, D]
time-varying field   X[T, N, F]
```

For each edge `e = u -> v`, derive geometry in the graph's original COO order:

```text
source position      P_u
target position      P_v
displacement         delta_e = P_v - P_u       [E, D]
distance             r_e = ||delta_e||          [E]
```

The current public API can already express a radial spatial message:

```python
from tinygrad import Tensor
from tinymesh import Graph

graph = Graph(3, source=[0, 1], target=[1, 2])
position = Tensor([[0.0, 0.0], [3.0, 4.0], [3.0, 8.0]])
state = Tensor([[1.0], [2.0], [3.0]])

source = graph.edge_values(position, endpoint="source")
target = graph.edge_values(position, endpoint="target")
delta = target - source
distance = (delta * delta).sum(axis=-1).sqrt()
weight = 1 / (1 + distance)
output = graph.sum(state, edge_weight=weight)
```

`Graph` owns sparse edge identity and aggregation. Position is an ordinary
tinygrad node field. The coordinate frame belongs to the data boundary because
a tensor cannot say whether `[3, 4]` means degrees, metres, or a local
simulation frame.

For a utility network, physical pipe connectivity should remain authoritative.
Two pipes crossing on a map are not connected unless the source network says
they share a junction. Radius or nearest-neighbor topology is useful when
connectivity is absent, not as a silent replacement for domain topology.

## Space and time are orthogonal

The first useful spatiotemporal form is:

```text
fixed topology G
static position P[N, D]
node history X[T, N, F]
           |
           v
edge geometry -> spatial message -> temporal cell
```

A moving mesh changes position to `P[T, N, D]`. A changing network changes
topology to `G_t`. These are different contracts:

```text
static geometry      G,   P[N, D],    X[T, N, F]
moving geometry      G,   P[T, N, D], X[T, N, F]
changing topology    G_t, P_t,         X_t
```

The current fixed-graph signal covers only the first topology case. A caller
can slice moving positions one snapshot at a time, but Tinymesh does not yet
own their temporal alignment.

## Reference implementations

The pinned sources answer three different questions.

### PyTorch Geometric: graph geometry

At revision
[`726310a`](https://github.com/pyg-team/pytorch_geometric/tree/726310a486eae37a89cd6359072b82bbbbb71579),
PyG keeps position in `data.pos`:

- [`KNNGraph`](https://github.com/pyg-team/pytorch_geometric/blob/726310a486eae37a89cd6359072b82bbbbb71579/torch_geometric/transforms/knn_graph.py)
  and
  [`RadiusGraph`](https://github.com/pyg-team/pytorch_geometric/blob/726310a486eae37a89cd6359072b82bbbbb71579/torch_geometric/transforms/radius_graph.py)
  derive connectivity from position.
- [`Distance`](https://github.com/pyg-team/pytorch_geometric/blob/726310a486eae37a89cd6359072b82bbbbb71579/torch_geometric/transforms/distance.py)
  and
  [`Cartesian`](https://github.com/pyg-team/pytorch_geometric/blob/726310a486eae37a89cd6359072b82bbbbb71579/torch_geometric/transforms/cartesian.py)
  derive scalar or vector edge attributes from linked positions.

That decomposition matches Tinymesh: topology, node position, and derived edge
geometry remain distinct. It does not require a PyG-compatible data container.

### TorchGeo: geospatial alignment

At revision
[`468c670`](https://github.com/torchgeo/torchgeo/tree/468c670bc94c961eb80e6c0ad32ed147852c367b),
TorchGeo's
[`GeoDataset`](https://github.com/torchgeo/torchgeo/blob/468c670bc94c961eb80e6c0ad32ed147852c367b/torchgeo/datasets/geo.py)
owns coordinate reference system, resolution, bounds, and spatiotemporal
queries. Its
[`GeoSampler`](https://github.com/torchgeo/torchgeo/blob/468c670bc94c961eb80e6c0ad32ed147852c367b/torchgeo/samplers/base.py)
selects regions in that coordinate space.

This is a data-alignment reference, not a graph-compute reference. Tinymesh
should accept already aligned numeric positions; a geospatial adapter should
project longitude and latitude into an appropriate metric frame before
Euclidean distance enters a model. The
[TorchGeo paper](https://arxiv.org/abs/2111.08872) describes the dataset,
sampler, and multispectral-data boundary.

### TerraTorch: model composition

At revision
[`375356c`](https://github.com/torchgeo/terratorch/tree/375356c9ba1d919c39816abaf6b499afc303497f),
TerraTorch's
[`EncoderDecoderFactory`](https://github.com/torchgeo/terratorch/blob/375356c9ba1d919c39816abaf6b499afc303497f/terratorch/models/encoder_decoder_factory.py)
composes a backbone, optional necks, decoder, and task head.

That separation is useful when Tinymesh has several proven interchangeable
model parts. It does not justify a registry or factory before those callers
exist, and it does not define spatial graph semantics. The
[TerraTorch paper](https://arxiv.org/abs/2503.20563) describes the wider
fine-tuning and benchmarking toolkit.

## Research lineage

- [DCRNN](https://arxiv.org/abs/1707.01926) treats directed network diffusion
  as the spatial operator and recurrence as the temporal operator.
- [E(n)-equivariant GNNs](https://arxiv.org/abs/2102.09844) show how coordinate
  differences and radial distances can preserve translation, rotation, and
  reflection symmetries.
- [MeshGraphNets](https://arxiv.org/abs/2010.03409) uses message passing over a
  simulation mesh and predicts physical dynamics.
- [MultiScale MeshGraphNets](https://arxiv.org/abs/2210.00612) adds coarse
  connectivity when spatially close points remain far apart in fine-mesh graph
  distance.

These papers describe progressively stronger needs. Scalar distance weighting
does not provide directional equivariance, mesh cells, adaptive remeshing, or
multiscale propagation.

## Decision

Add no public spatial type yet:

```text
Graph                         owns topology
Tensor[N, D]                  owns numeric position
data adapter                  owns CRS, projection, and units
edge_values + tensor math     derive COO edge geometry
Graph.sum                     aggregates scalar-weighted node messages
```

The [spatial geometry experiment](spatial-geometry.md) now proves that direct
composition on one fixed graph preserves COO identity, sparse intermediate
shapes, first-order gradients, vertex relabeling, translation, and rotation
contracts on CPU and Metal. A vector edge-message aggregation primitive earns
its place only when directional displacement is a real caller; a `Mesh` type
earns its place only when faces, cells, or hierarchy need an owner.
