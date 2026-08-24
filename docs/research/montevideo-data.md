# Montevideo spatial-temporal data

The Montevideo bus dataset is the first real source with topology, metric
position, scalar edge facts, and an ordered node signal. Tinymesh validates the
host values first, then lowers their shared node and edge identity into
tinygrad tensors.

## Pinned source

Tinymesh reads
[PyG Temporal `fe555bc`](https://github.com/benedekrozemberczki/pytorch_geometric_temporal/blob/fe555bc30ee197755c4b58a89407033a5f383415/dataset/montevideo_bus.json)
with SHA-256
`37d9c6286d474077b5c05173c1570c4da42c387013116daa8862c7a6cab86a75`.
The JSON payload is `3,131,884` bytes. The reader accepts at most 4 MiB and
checks that limit before JSON parsing.

An explicit path supports offline runs and fixtures. The default path uses a
10-second standard-library request, reads at most 4 MiB, and verifies the exact
checksum above. Python's standard library owns acquisition, JSON parsing, and
source validation. No geo or dataframe package enters the runtime.

## Source contract

```text
ordered nodes
  bus_stop            stable source identity
  lon, lat            two finite numeric position values
  X.y                 ordered feature observations
  y                   ordered target observations

ordered links
  source, target      directed bus_stop identities
  weight              positive finite road distance
```

The parser resolves link identities to node rows while preserving both source
orders. It rejects duplicate node identities, duplicate directed links, missing
endpoints, non-finite geometry, non-positive distance, unequal time axes, and
oversized input. It returns plain immutable Python values; it creates no tensor,
normalization statistic, or dense topology.

The source calls its projected position fields `lon` and `lat`. Tinymesh
preserves the numeric values without interpreting or transforming them at the
source boundary.

## Aligned tensors

```python
from tinymesh.datasets import montevideo_bus

data = montevideo_bus(lags=4, device="CPU")
signal = data.signal

print(signal.x.shape, signal.y.shape)
# (740, 675, 4) (740, 675, 1)
print(data.position.shape, data.road_distance.shape)
# (675, 2) (690,)
```

`MontevideoBus` composes the existing `StaticGraphTemporalSignal` with one
node-aligned position tensor and one COO-edge-aligned road-distance tensor. The
record rejects shape, dtype, or device misalignment. It is dataset-specific;
it does not establish a generic spatial container.

```text
Graph                       675 nodes, 690 directed edges
signal.x                    [740, 675, 4] raw lagged inflow
signal.y                    [740, 675, 1] raw next-step inflow
position                    [675, 2] node order
road_distance               [690] original COO link order
coordinate frame            EPSG:32721
length unit                 m
```

The [Uruguay open-data catalog](https://catalogodatos.gub.uy/dataset/transporte-colectivo-paradas-puntos-de-control-y-recorridos-de-omnibus/resource/f30c15b5-2638-4315-b6a1-4868f9e6e02d)
identifies the stop positions as WGS 84 / UTM zone 21S. The
[Uruguay spatial-data recommendation](https://montevideo.gub.uy/sites/default/files/biblioteca/sistemareferenciaproyeccionesrecomendacionesideuy.pdf)
maps that frame to EPSG:32721, whose position unit is metres. PyG Temporal calls
the link weight road distance but does not label its unit. Its magnitude and
agreement with UTM edge distance support metres; Tinymesh records `m` as this
dataset interpretation and performs no projection.

Road distance remains data, not an aggregation coefficient:
`signal.edge_weight` is `None`. The loader also retains raw passenger counts.
Train-only normalization belongs to forecasting, after a forward split.

## Full-source witness

```console
uv run --locked python -m experiments.run montevideo_source
```

At the pinned revision:

```json
{
  "nodes": 675,
  "edges": 690,
  "steps": 744,
  "duplicate_edges": 0,
  "self_loops": 0,
  "position_dimensions": 2,
  "minimum_road_distance": 23.8,
  "maximum_road_distance": 1991.3
}
```

This establishes source structure, not model quality or framework parity.

## Tensor witness

```console
uv run --locked python -m experiments.run montevideo_data DEV=CPU
uv run --locked python -m experiments.run montevideo_data DEV=METAL
```

Both devices report the aligned shapes above. For the first source edge, the
straight-line coordinate distance is `171.59254 m` and the observed road
distance is `172.2 m`. Keeping these as separate tensors makes the later
forecast comparison explicit rather than silently treating either one as the
graph weight.

This stage proves deterministic lowering and real sparse composition. It makes
no forecasting or performance claim.

The cross-dataset [network measurement](network-measurement.md) adds the
missing structural qualification. All 675 nodes form one weak component, but
the source has no reciprocal edges, 675 singleton strong components, and only
17.0% directed pair reachability. Its mean reachable distance is 40.47 hops and
its directed diameter is 114; the graph is a sparse directed route relation,
not a mutually reachable proximity network.
