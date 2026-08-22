# MBTA topology signal

Status: Stage 4 retained; learned test opened once after validation freeze.

## Decision

Retain a bounded topology-specific result. A 353-parameter residual model using
true upstream operational neighbors reaches 149.398 seconds mean test MAE across
three seeds. It beats both Stage 3 baselines and the station-local, reversed,
and degree-preserving permuted controls in every seed, as well as on route-macro
MAE and p90 error.

```text
Stage 3 exact event target at cutoff
                  |
       +----------+-----------+
       |                      |
       v                      v
route-fitted plan/time   latest exact headway
anchor                   on sparse neighbor(s)
       |                      |
       +----------+-----------+
                  |
          353-parameter MLP
                  |
       +----------+----------+----------+
       |          |          |          |
     self        true      reverse   permuted
    150.272    145.793     150.682    149.896 s
                  |
           validation frozen
                  |
          single test passed
```

The claim is deliberately narrow: within this frozen retrospective MBTA task,
the train-observed direction of service carries predictive next-headway
information that matched false topologies do not.

## Frozen information boundary

Stage 4 inherits all 940,551 physical targets, dates, cutoffs, masks, and metrics
from the checksum-bound [Stage 3 task](mbta-headway-task.md). It changes none of
them. Every dynamic feature is known strictly before the target cutoff.

The non-graph anchor fits one public-plan weight per route from train targets
only, over the fixed grid `{0.00, 0.05, ..., 1.00}`:

| Route | Plan weight | Temporal weight |
| --- | ---: | ---: |
| Blue | 0.10 | 0.90 |
| Green-B | 0.00 | 1.00 |
| Green-C | 0.00 | 1.00 |
| Green-D | 0.05 | 0.95 |
| Green-E | 0.00 | 1.00 |
| Mattapan | 0.65 | 0.35 |
| Orange | 0.55 | 0.45 |
| Red | 0.55 | 0.45 |

This anchor reaches 152.176 seconds validation MAE at full coverage, improving
slightly on the 152.648-second public-plan and 157.152-second temporal Stage 3
controls without reading validation labels during fitting.

## Sparse topology and messages

Nodes are the 259 observed `(parent_station, trunk_route_id, direction_id)`
lanes. True directed edges are all 831 distinct station transitions in exact,
unambiguous run relations from train dates only. Edge affinity is train support;
no validation/test movement chooses or weights an edge.

For each target and each incoming edge, the builder performs an as-of lookup of
the latest exact completed source-node headway in the same service date. It then
computes the affinity-weighted headway and age across available neighbors. Work
scales with target-edge incidences; no node-pair matrix, global clock, or
target-by-node snapshot is constructed.

```text
source history --ASOF(< cutoff)--+
source history --ASOF(< cutoff)--+-- support-weighted message --> target node
source history --ASOF(< cutoff)--+
```

The four arms differ only in that edge relation:

- `self`: 259 self edges; the station-local learned control;
- `true`: train-observed direction of service;
- `reverse`: every true edge reversed with its affinity unchanged;
- `permuted`: one frozen vertex relabeling within each trunk-direction
  component, preserving edge count, affinity multiset, and degree sequence.

Missing plan, persistence, and message values impute to their declared causal
control and retain explicit masks. All learned arms therefore predict every
target. Feature means/scales and target transformation fit train only.

## Matched model

Every arm and seed uses the same zero-initialized residual model:

```text
20 standardized inputs -> tanh(16) -> scalar log-headway residual
```

Inputs are logarithmic temporal, plan, persistence, topology-message, and
message-age values; their presence masks; weekday/hour sine and cosine; and the
eight route indicators. The target and anchor use a train-standardized
`log1p(seconds)` transform, so restored predictions are positive without label
clipping. Metrics remain unbounded seconds.

Each seed trains exactly 500 batches of 4,096 examples with Adam at `0.003` and
a Huber delta of `0.25`. Checkpoints occur every 100 steps and select validation
micro MAE. Each arm processes 2,048,000 sampled train examples per seed. Seeds
`(0, 1, 2)`, data order, topology permutation, model initialization, and batch
sampling are fixed.

## Validation evidence

| Arm | Mean MAE | Mean route-macro MAE | Mean p90 AE |
| --- | ---: | ---: | ---: |
| self | 150.272 s | 170.705 s | 321.030 s |
| **true** | **145.793 s** | **166.533 s** | **309.366 s** |
| reverse | 150.682 s | 171.165 s | 321.669 s |
| permuted | 149.896 s | 170.613 s | 319.320 s |

The true arm's seed MAEs are 145.742, 146.026, and 145.610 seconds. It beats
the best false-topology arm by 4.103 seconds on mean micro MAE and preserves the
direction in every seed. The station-local model also clears its gate at
150.272 seconds, demonstrating useful learned residual signal before topology.

The validation gate requires true topology to beat both Stage 3 baselines,
every topology control in every seed, route-macro MAE, and p90 absolute error.
All clauses pass. RMSE remains diagnostic rather than the selection metric.

## Single test evidence

The learned test split opened once after commit `4078a5117c`. No model was
retrained: the evaluator rebuilt the causal features, matched the protocol,
loaded the exact validation-selected states, and evaluated all 12 arms together.

| Arm | Mean MAE | Mean route-macro MAE | Mean p90 AE |
| --- | ---: | ---: | ---: |
| self | 155.042 s | 152.978 s | 344.283 s |
| **true** | **149.398 s** | **148.167 s** | **330.573 s** |
| reverse | 155.241 s | 153.405 s | 344.604 s |
| permuted | 154.771 s | 152.894 s | 343.124 s |

True-topology seed MAEs are 149.049, 149.824, and 149.322 seconds. All beat
their paired false-topology controls. The true arm also improves on the
159.788-second temporal baseline, 166.062-second public plan, and
156.384-second route-fitted anchor. It wins seven of eight route MAEs;
Mattapan is the exception, so the result is not a universal per-route gain.

The Schedule-resolved/unresolved mean MAEs are 140.845/208.423 seconds for true
topology versus 146.261/215.637 for self-only. True topology therefore improves
both provenance slices rather than winning by filtering unresolved observations.
All arms cover all 176,568 test targets and produce zero nonpositive values.

## Determinism and reproduction

An initial characterization found run-to-run drift despite fixed RNG seeds. The
cause was an unordered SQL `row_number()`: identical random batch indices named
different physical targets after reconstruction. Target identity now owns row
order, and one-thread feature aggregation fixes floating reduction order. Two
fresh full builds produce byte-identical artifacts:

| Artifact | SHA-256 |
| --- | --- |
| protocol | `e3c44f17757820eea46c7c6e066fe914991c920388e1a56a4e17f7c013303db8` |
| validation | `9e3fb262e8be83314f0606b2553011d6a0a0a6d5ec96b24366334719048c3069` |
| test | `d2cf0d52bbb0aabba48fc25b561a49bfc7102feb2865764f469e94d0133c9e9a` |

```console
uv run --locked --with duckdb==1.4.1 --with numpy==2.3.2 -m experiments.tools.mbta_topology --source-dir /tmp/mbta-population-source --population-audit experiments/fixtures/mbta_population/audit.json --task-protocol experiments/fixtures/mbta_headway_task/protocol.json --output-dir /tmp/mbta-topology
uv run --locked python -m experiments.run mbta_topology
uv run --locked --with duckdb==1.4.1 --with numpy==2.3.2 -m experiments.tools.mbta_topology --source-dir /tmp/mbta-population-source --population-audit experiments/fixtures/mbta_population/audit.json --task-protocol experiments/fixtures/mbta_headway_task/protocol.json --output-dir /tmp/mbta-topology --test
uv run --locked python -m experiments.run mbta_topology TEST=1
```

DuckDB and NumPy are one-off evidence-builder dependencies, not TinyMesh runtime
dependencies. The executable record pins tinygrad plus the Stage 3 TSL,
LibCity, and PyG Temporal study references.

## Limits

The frozen claim gate required true topology to beat both Stage 3 test
baselines, all controls in every seed, route-macro MAE, and p90 error. Every
clause passed. This establishes incremental directional signal, not that this
353-parameter model is an optimal forecast or that every route benefits.

The topology comes from observed train run relations, not a universal transit
ontology. Rare operational transitions remain with low affinity rather than an
arbitrary support threshold. The experiment covers one MBTA summer interval,
uses retrospective event time, and says nothing about online inference,
service quality, other systems, causal intervention, or richer context. Stage 5
remains closed until error analysis names a residual and a causally available
context source.
