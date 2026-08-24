# METR-LA sensor data

METR-LA is Tinymesh's first real fixed sensor network with explicit timestamps,
missing observations, and asymmetric road-network affinity. This stage only
establishes the data boundary; it makes no forecast or topology-value claim.

## Pinned sources

Traffic readings come from the CC BY 4.0
[METR-LA CSV on Zenodo](https://zenodo.org/records/5724362):

```text
METR-LA.csv
72,467,662 bytes
SHA-256 8d67a35472db1719d7d4be851f2bf64cb21d9c52577c8a6b4b873d43205af381
```

Sensor order and road distances come from DCRNN revision
[`602afd9d`](https://github.com/liyaguang/DCRNN/tree/602afd9d767d3aa1c9b3eac51710d6aeee12c227):

```text
graph_sensor_ids.txt
1,448 bytes
SHA-256 3ba026caa2e6263ab0ea54b0fa1b125dbfa7216544cd05313b555e826292b990

distances_la_2012.csv
6,393,348 bytes
SHA-256 a576a2a3e28dbb959be6da22688e24dd1b246b81264595e129147c256cd53de5
```

The default loader uses tinygrad's cache and verifies every remote checksum and
exact byte count. An explicit directory supports offline use and fixtures; it
must contain those three canonical names, remains size-bounded, and is
validated structurally rather than forced to match the remote checksums.

The CSV identifies values as traffic speed but does not label a unit, so
Tinymesh does not invent one. Its timestamps have no zone offset and remain
naive source datetimes.

## Source identity and missingness

The traffic header matches all 207 DCRNN sensor IDs in exact order. The payload
contains 34,272 uninterrupted five-minute rows:

```text
2012-03-01 00:00:00
        ...
2012-06-27 23:55:00
```

The [DCRNN paper](https://arxiv.org/abs/1707.01926) describes a broader March 1
through June 30 collection period, while this published CSV ends on June 27.
Tinymesh preserves the payload rather than filling the difference.

DCRNN's pinned
[training code](https://github.com/liyaguang/DCRNN/blob/602afd9d767d3aa1c9b3eac51710d6aeee12c227/model/dcrnn_supervisor.py#L78-L80)
uses zero as the missing-value sentinel. The CSV has:

```text
7,094,304 total values
6,519,002 observed values
  575,302 missing sentinels
```

The nonzero count exactly matches the paper's reported number of observed
METR-LA points. `METRLA.observed` therefore derives `speed != 0` on demand;
the loader neither imputes zero nor stores a second source of truth.

## Directed affinity

Tinymesh reproduces DCRNN's pinned
[graph recipe](https://github.com/liyaguang/DCRNN/blob/602afd9d767d3aa1c9b3eac51710d6aeee12c227/scripts/gen_adj_mx.py)
directly from selected road-distance rows:

```text
sigma = population standard deviation of selected distances
w(i,j) = exp(-(distance(i,j) / sigma)^2)
keep w(i,j) >= 0.1
```

Direction is retained. The result is 1,722 COO-ordered edges, including 207
self-edges and 1,111 edges whose reverse is absent. This is a sparse,
distance-derived sensor affinity—not a street graph, learned correlation
graph, or claim that each edge is a physical road segment.

The loader never creates an `[N, N]` tensor. It filters selected distance facts,
sorts the surviving source-target pairs, and emits one `Graph` plus its aligned
`affinity[E]`.

An independent audit against the pinned
[`adj_mx.pkl`](https://github.com/liyaguang/DCRNN/blob/602afd9d767d3aa1c9b3eac51710d6aeee12c227/data/sensor_graph/adj_mx.pkl)
found identical sensor order and sparse support. The largest weight difference
was one float32 rounding step, `1.1920928955078125e-07`.

## Public boundary

```python
from tinymesh.datasets import metr_la

data = metr_la(device="CPU")

print(data.speed.shape, data.observed.shape)
# (34272, 207) (34272, 207)
print(data.graph.nodes, data.graph.edges, data.affinity.shape)
# 207 1722 (1722,)
```

`METRLA` owns only facts this source can support:

```text
graph          sparse directed sensor connectivity
sensor_ids     [N] stable source identity
timestamps     [T] ordered naive datetimes
speed          [T, N] raw float32 readings
observed       [T, N] derived zero-sentinel mask
affinity       [E] positive COO-aligned graph weight
sample_minutes 5
```

It rejects mismatched columns, irregular timestamps, duplicate identities or
selected edges, missing zero-distance self-edges, non-finite or negative
values, malformed schemas, oversized local files, and tensor-axis
misalignment.

No generic temporal container changes in this stage. Forecast windows,
time-of-day fields, imputation, normalization, split policy, targets, losses,
and model unrolling need a task contract and remain outside the loader.

## Full-source witness

Revision `33e81efd26ecf370c21b5ca95880491105a1c994` produced matching CPU and
Metal observations:

```console
uv run --locked python -m experiments.run metr_la_data DEV=CPU
uv run --locked python -m experiments.run metr_la_data DEV=METAL
```

```json
{
  "nodes": 207,
  "edges": 1722,
  "steps": 34272,
  "first_timestamp": "2012-03-01 00:00:00",
  "last_timestamp": "2012-06-27 23:55:00",
  "sample_minutes": 5,
  "values": 7094304,
  "observed_values": 6519002,
  "missing_values": 575302,
  "self_loops": 207,
  "asymmetric_edges": 1111,
  "minimum_affinity": 0.10008395463228226,
  "maximum_affinity": 1.0,
  "affinity_sha256": "d4db2dce0cfd83ec40c115372881cd409059e3ddcb2c4adf8f34502b7a7a00e5"
}
```

The matching affinity digest proves device-independent lowering at this
revision. It does not show that the affinity improves a forecast. The
[METR-LA forecast](metr-la-forecast.md) now owns the forward split, train-only
preprocessing, missing-value-aware loss, temporal controls, and false
topologies; its learned graph comparison remains pending.

The cross-dataset [network measurement](network-measurement.md) finds a
195-node strong core, 13 strong components, and two weak components. One sensor
has only its retained self-loop and is therefore isolated from non-self
messages. Directed reachability is 93.8%, with mean reachable distance 5.85
hops and diameter 17. The affinity is broad but not fully connected.
