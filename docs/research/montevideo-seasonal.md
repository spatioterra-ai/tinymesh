# Montevideo seasonal floor

This record selects the temporal control that a later graph experiment must
beat.

## Decision

At tinygrad revision
[`6ea7d366`](https://github.com/tinygrad/tinygrad/tree/6ea7d366fa92842c0bc8b7b080e26e83a7406252),
a train-only node mean for each ordered hour-of-week phase is the strongest
validation baseline. It reduces validation MAE by 19.0% and RMSE by 27.1%
relative to persistence.

Only that selected baseline is evaluated on test. It reduces test MAE by 22.4%
and RMSE by 32.8% relative to the persistence result recorded by the preceding
forecast study.

This is evidence for a strong periodic node field, not for graph structure.
The source has ordered hourly rows but no timestamp field. `t mod 168` is a
periodic hypothesis; it is not a claim about a civil weekday or local time.

## Protocol

The one-lag signal pairs the observation at raw hour `r` with the target at raw
hour `r + 1`:

```text
signal row r       x[r] ----------------------> y[r]
raw hour             r                           r + 1

target rows        0:520       520:594       594:743
                   train       validation     test
```

Five controls are fixed before validation:

```text
zero            0
persistence     x[r]
daily           x[r + 1 - 24]
weekly          x[r + 1 - 168]
hour_of_week    mean(y[train rows with (r + 1) mod 168 = phase], axis=time)
```

The phase mean falls back to the node's overall training mean when training has
not observed that phase. All 168 phases occur in the pinned training rows, so
the real run uses no fallback.

Validation RMSE selects one control. MAE and then the displayed baseline order
break exact ties. Test is read only for the selected control.

## Result

CPU and Metal return the same values. Every validation candidate covers all 74
rows, 49,950 node targets, and a zero fraction of `0.824324`.

| Validation baseline | MAE | RMSE |
| --- | ---: | ---: |
| Zero | 0.617618 | 2.923249 |
| Persistence | 0.487187 | 1.555442 |
| Daily | 0.559640 | 1.875419 |
| Weekly | 0.443103 | 1.383088 |
| Hour of week | **0.394855** | **1.134036** |

The selected hour-of-week control covers all 149 test rows and 100,575 node
targets:

| Test baseline | Zero fraction | MAE | RMSE |
| --- | ---: | ---: | ---: |
| Hour of week | 0.795029 | 0.453352 | 1.224738 |

The weekly naive control already beats persistence on validation. Averaging
each node's repeated training phases removes additional noise without an
optimizer.

## Leakage boundary

Daily and weekly lookups use only raw hours before their target. Validation and
test may read preceding observations across a split boundary; no later target
enters a training statistic.

The phase table reads only `y[0:520]`. A perturbation test changes every test
row while leaving the fitted phase table, all validation metrics, and baseline
selection unchanged. A separate periodic fixture matches an independent
Python calculation.

## Reference boundary

The comparison is revision-bound:

| Reference | Revision | Role |
| --- | --- | --- |
| [tinygrad](https://github.com/tinygrad/tinygrad/tree/6ea7d366fa92842c0bc8b7b080e26e83a7406252) | `6ea7d366` | Tensor execution on CPU and Metal |
| [PyG](https://github.com/pyg-team/pytorch_geometric/tree/726310a486eae37a89cd6359072b82bbbbb71579) | `726310a` | Sparse graph data reference |
| [PyG Temporal](https://github.com/benedekrozemberczki/pytorch_geometric_temporal/tree/fe555bc30ee197755c4b58a89407033a5f383415) | `fe555bc` | Source dataset and fixed-graph signal reference |
| [TorchGeo](https://github.com/microsoft/torchgeo/tree/468c670bc94c961eb80e6c0ad32ed147852c367b) | `468c670` | Geospatial dataset-boundary reference |
| [TerraTorch](https://github.com/IBM/terratorch/tree/375356c9ba1d919c39816abaf6b499afc303497f) | `375356c` | Modular data/model composition reference |

PyG Temporal defaults to four lagged features and standardizes the complete
series in its loader. Tinymesh retains raw one-lag values so split ownership,
train-only fitting, and passenger-count metrics remain explicit. This
topology-free control needs no PyTorch, NumPy, geo adapter, or new Tinymesh
primitive.

## Consequence

The previous recurrent models were compared against a floor that was too weak.
The follow-up [delayed-edge study](montevideo-delayed-edges.md) subtracts this
prediction and tests real directed edges against reverse-edge and
permuted-field controls. It fails the promotion gate, so Tinymesh adds no
edge-field API or model.

## Limits

This result covers one pinned 31-day ordered series, one-step targets, one
168-hour hypothesis, and per-node phase means. It does not establish calendar
semantics, seasonality outside this interval, moving buses, route travel time,
topology value, geometry value, or production accuracy.

## Reproduce

```console
DEV=CPU uv run --locked python -m experiments.montevideo_seasonal
DEV=METAL uv run --locked python -m experiments.montevideo_seasonal
DEV=CPU uv run --locked python -m unittest tests.test_montevideo_seasonal
DEV=METAL uv run --locked python -m unittest tests.test_montevideo_seasonal
```
