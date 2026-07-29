# Montevideo delayed edges

This record asks whether the real directed stop graph carries causal signal
beyond the selected seasonal floor.

## Decision

At tinygrad revision
[`6ea7d366`](https://github.com/tinygrad/tinygrad/tree/6ea7d366fa92842c0bc8b7b080e26e83a7406252),
delayed residuals on real edges fail the frozen promotion gate.

The real graph selects a one-hour lag. It improves validation RMSE by only
`0.000062` while worsening MAE, and both structural controls achieve lower
validation RMSE. On test, the real graph worsens both MAE and RMSE relative to
the seasonal floor.

Tinymesh retains the experiment and negative result. It adds no edge-field
primitive, recurrent architecture, or public API.

## Question

The selected baseline predicts one value for each node and ordered
hour-of-week phase. Subtracting it leaves a node residual field:

```text
target[v,t] - seasonal[v,t] = residual[v,t]

residual[u,t-lag] -- real edges u -> v -- incoming mean --> signal[v,t,lag]

prediction[v,t] = seasonal[v,t] + alpha * signal[v,t,lag]
```

For every structure and lag, one global scalar minimizes training squared
residual error:

```text
alpha = sum(signal * residual) / sum(signal**2)
```

A zero denominator gives `alpha = 0`. There is no node, edge, or phase-specific
coefficient.

## Controls

The lag set is fixed at `{1, 2, 3, 6, 12, 24}` hours. The same fit and
validation selection run on:

```text
real       original directed edges, original node fields
reverse    every directed edge reversed, original node fields
permuted   original edges, node fields shifted left by one node index
```

The reverse control tests edge direction. The cyclic field control preserves
the graph, tensor shapes, value distribution, and temporal order while
breaking stop identity.

Each structure selects minimum validation RMSE, then MAE, then the smaller lag.
All three choices are frozen before test metrics are computed.

## Candidate evidence

CPU candidate values are below. Metal differs only in the last digits of some
coefficients and block reductions; rounded metrics, selections, and the gate
are unchanged.

| Structure | Lag | Alpha | Validation MAE | Validation RMSE |
| --- | ---: | ---: | ---: | ---: |
| Real | **1** | 0.024279 | **0.398313** | **1.133975** |
| Real | 2 | 0.034118 | 0.399917 | 1.134102 |
| Real | 3 | 0.026545 | 0.399133 | 1.134529 |
| Real | 6 | 0.009056 | 0.396654 | 1.134074 |
| Real | 12 | 0.002601 | 0.395550 | 1.134053 |
| Real | 24 | -0.008707 | 0.396369 | 1.134066 |
| Reverse | 1 | 0.040038 | 0.400361 | 1.134856 |
| Reverse | 2 | 0.021596 | 0.397982 | 1.134101 |
| Reverse | **3** | 0.026643 | **0.398905** | **1.133462** |
| Reverse | 6 | 0.016014 | 0.397964 | 1.133871 |
| Reverse | 12 | -0.000856 | 0.395080 | 1.134028 |
| Reverse | 24 | -0.007997 | 0.396163 | 1.134108 |
| Permuted | 1 | 0.063858 | 0.400670 | 1.136070 |
| Permuted | 2 | 0.063903 | 0.401378 | 1.134140 |
| Permuted | **3** | 0.057031 | **0.401152** | **1.133179** |
| Permuted | 6 | 0.041106 | 0.401414 | 1.134874 |
| Permuted | 12 | 0.002067 | 0.395315 | 1.134103 |
| Permuted | 24 | 0.007347 | 0.395594 | 1.133965 |

Bold rows are selected by RMSE, not by MAE.

## Selected test

| Structure | Covered nodes | Validation MAE | Validation RMSE | Test MAE | Test RMSE |
| --- | ---: | ---: | ---: | ---: | ---: |
| Seasonal floor | 675 / 675 | 0.394855 | 1.134036 | 0.453352 | 1.224738 |
| Real, lag 1 | 666 / 675 | 0.398313 | 1.133975 | 0.457050 | 1.224942 |
| Reverse, lag 3 | 668 / 675 | 0.398905 | 1.133462 | 0.457736 | 1.224921 |
| Permuted, lag 3 | 666 / 675 | 0.401152 | 1.133179 | 0.459708 | 1.223832 |

Isolated nodes receive a zero spatial residual, so their prediction remains
the seasonal floor. Metrics retain all 675 nodes.

## Gate

Each split is divided into three contiguous blocks. Real edges must improve
both overall metrics and each metric in at least two blocks against every
comparator.

| Split | Against | Overall MAE better | Overall RMSE better | MAE blocks | RMSE blocks | Pass |
| --- | --- | --- | --- | ---: | ---: | --- |
| Validation | Floor | No | Yes | 0 / 3 | 2 / 3 | No |
| Validation | Reverse | Yes | No | 3 / 3 | 0 / 3 | No |
| Validation | Permuted | Yes | No | 3 / 3 | 1 / 3 | No |
| Test | Floor | No | No | 0 / 3 | 1 / 3 | No |
| Test | Reverse | Yes | No | 3 / 3 | 2 / 3 | No |
| Test | Permuted | Yes | No | 3 / 3 | 1 / 3 | No |

The tiny validation RMSE movement is neither metric-consistent nor
structure-specific. It reverses on test.

## Sparse path

`Graph.sum` folds the 743 time rows into feature width and reduces the complete
field with one `csr_sum` call per structure:

```text
value buffer          [675, 743]
forward row pointer   [676]
forward column        [690]
transpose row pointer [676]
transpose column      [690]
```

Each graph sum owns `2 * (676 + 690) = 2,732` device `int32` topology values.
UOp inspection finds one sparse call and no `[675,675]` topology carrier. The
experiment uses node fields of shape `[743,675,1]`, not dense node-pair state.

An independent host implementation matches incoming means, the closed-form
coefficient, predictions, and metrics on a small directed periodic fixture.
Boundary tests prove lagged predictions read only earlier rows; perturbing test
rows leaves every fit, validation metric, and selected lag unchanged.

## Reference boundary

| Reference | Revision | Role |
| --- | --- | --- |
| [tinygrad](https://github.com/tinygrad/tinygrad/tree/6ea7d366fa92842c0bc8b7b080e26e83a7406252) | `6ea7d366` | Tensor execution and UOp evidence |
| [PyG](https://github.com/pyg-team/pytorch_geometric/tree/726310a486eae37a89cd6359072b82bbbbb71579) | `726310a` | Source-to-target message and aggregation reference |
| [PyG Temporal](https://github.com/benedekrozemberczki/pytorch_geometric_temporal/tree/fe555bc30ee197755c4b58a89407033a5f383415) | `fe555bc` | DCRNN forward/reverse diffusion reference |
| [TorchGeo](https://github.com/microsoft/torchgeo/tree/468c670bc94c961eb80e6c0ad32ed147852c367b) | `468c670` | Geospatial dataset-boundary reference |
| [TerraTorch](https://github.com/IBM/terratorch/tree/375356c9ba1d919c39816abaf6b499afc303497f) | `375356c` | Modular data/model composition reference |

PyG's `MessagePassing` surface separates messages from target aggregation.
This experiment needs only the existing Tinymesh composition
`Graph.sum(field) / in_degree`; it does not justify another abstraction. PyG
Temporal's DCRNN motivates directional diffusion; this stage tests its simplest
identifiable component before adding recurrent complexity.

TorchGeo and TerraTorch remain reference-only because no raster adapter,
geospatial sampler, backbone, neck, or head participates in this test. No
PyTorch, NumPy, or geo runtime is added.

## Meaning

This result rejects one narrow claim: an unweighted incoming mean of delayed
stop residuals is not a useful next-hour correction on this month of data.

It does not prove that graphs or spatiotemporal graph neural networks are
irrelevant. Route identity, within-hour travel time, bus trajectories, service
frequency, traffic, weather, mobile-agent attachment, learned edge state, and
more observations are absent. Those facts could define a different graph
signal. They do not justify adding speculative Tinymesh machinery before a
caller demonstrates it.

## Reproduce

```console
DEV=CPU uv run --locked python -m unittest tests.test_montevideo_delayed_edges
DEV=METAL uv run --locked python -m unittest tests.test_montevideo_delayed_edges
uv run --locked python -m experiments.run montevideo_delayed_edges DEV=CPU
uv run --locked python -m experiments.run montevideo_delayed_edges DEV=METAL
```
