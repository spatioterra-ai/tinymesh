# Chickenpox forecast

This experiment asks whether the fixed-graph data path can train recurrent
models on real temporal windows, and whether moving hidden state across county
edges helps.

## Decision

At tinygrad revision
[`7d48926`](https://github.com/tinygrad/tinygrad/tree/7d48926293484b78ea2f5a3c73108c3f52a36274),
Tinymesh trains node-local LSTM, T-GCN, and Chebyshev GConvGRU forecasts through
the same causal window and sparse-autograd path on CPU.

GConvGRU and the matched LSTM are tied across three seeds. GConvGRU's mean test
MSE is `0.4%` lower, while its mean test MAE is `0.3%` higher, and the winner
changes with the seed. T-GCN is consistently worse. This proves end-to-end
execution and exposes a model comparison; it does not yet show that graph
recurrence improves this forecast.

## Data flow

The pinned source already contains standardized weekly values. Tinymesh loads
one lag so each snapshot pairs the current week with the next week, splits
along time, then forms eight-week sequence-to-one windows:

```text
source FX [521, 20]
        |
        v
signal x [520, 20, 1] ---- time split ----> train 364
       y [520, 20, 1]                     validation 52
                                             test 104
                                                  |
                                                  v
                                   causal windows [B, 8, 20, 1]
                                           target [B, 20, 1]
```

Splitting before windowing prevents a window from crossing a split boundary.
It also discards the first seven possible targets inside validation and test,
leaving `357`, `45`, and `97` windows.

The source graph has `102` edges, including one self-loop per county. Operator
conventions are explicit:

```text
source graph, 102 edges -----------> T-GCN GCN
          |
          +-- remove 20 self-loops -> GConvGRU Chebyshev graph, 82 edges

node rows only --------------------> LSTM
```

Tinymesh GCN expects its caller to supply the renormalization self-loops.
Chebyshev convolution removes self-loops before constructing its Laplacian.
Forcing both cells through one loop convention would change one of the
operators, not make the comparison fair.

## Execution

`Graph.sum` treats every leading axis as an independent lane around one shared
node axis:

```text
[B, L, N, H]
      |
      | move N first, fold the other lanes
      v
[N, B * L * H] -- one CSR call --> [N, B * L * H]
      |                                  |
      +------------ restore -------------+
                         |
                         v
                  [B, L, N, H]
```

Static scalar edge weights are shared by all lanes. Their backward gradient
sums contributions from every lane. The experiment unrolls each recurrent cell
over `L`; it does not send the time axis through the graph at once.

`StaticGraphTemporalSignal.batches()` constructs a batch from `L` contiguous
time slices rather than one tensor expression per window. That keeps window
expression size proportional to history length instead of batch count; the
materialized tensor still contains `B * L * N * F` values.

## Controlled run

All models use eight input weeks, hidden width four, one scalar readout, MSE
loss, Adam at `0.01`, 50 full-batch epochs, and seeds `0`, `1`, and `2`.
Parameter counts are close:

```text
node-local LSTM     117
T-GCN               125
GConvGRU K=2        137
```

The source values are standardized, so these errors are dimensionless and
cannot be read as chickenpox case counts. Values are mean plus sample standard
deviation across the three seeds:

| Model | Validation MSE | Validation MAE | Test MSE | Test MAE |
| --- | ---: | ---: | ---: | ---: |
| LSTM | 0.433390 ± 0.026565 | 0.459462 ± 0.009720 | 0.739218 ± 0.040890 | 0.547016 ± 0.012959 |
| T-GCN | 0.547173 ± 0.002051 | 0.512327 ± 0.003576 | 0.931795 ± 0.002444 | 0.622671 ± 0.001488 |
| GConvGRU K=2 | 0.431246 ± 0.018080 | 0.459169 ± 0.004954 | 0.735942 ± 0.037898 | 0.548614 ± 0.007640 |

Two parameter-free test baselines are identical across seeds:

| Baseline | Test MSE | Test MAE |
| --- | ---: | ---: |
| zero climatology | 0.993437 | 0.628739 |
| last-value persistence | 2.856533 | 1.081488 |

Both LSTM and GConvGRU learn useful temporal signal. Persistence is weak because
these standardized county values change sharply week to week. Beating it alone
would not establish model quality; the zero and node-local baselines are the
important controls.

## Interpretation

GConvGRU wins on both validation metrics for seeds `0` and `1`; LSTM wins for
seed `2`. Their mean errors are effectively equal. The current evidence
supports neither “the graph helps” nor “the graph is useless.”

T-GCN graph-mixes the current input before node-local recurrence. GConvGRU's
Chebyshev basis includes a local `T_0` path and graph-mixes remembered state.
That richer cell can fall back toward a node-local function, while T-GCN cannot
undo all smoothing of a volatile node signal. This is a plausible explanation
for the gap, not a causal result.

## Framework comparison

The data tensors match PyG Temporal, but this training result is not a framework
benchmark. The pinned PyG Temporal tree does not record expected T-GCN or
GConvGRU outputs. Its
[Chickenpox DCRNN example](https://github.com/benedekrozemberczki/pytorch_geometric_temporal/blob/fe555bc30ee197755c4b58a89407033a5f383415/docs/source/notes/introduction.rst#L236-L304)
reports test MSE `0.7418`, but uses four lag features, hidden width `32`, a
20/80 split, and 200 epochs. Its model and evaluation protocol therefore differ
from this run.

A valid framework comparison must hold the windows, chronological splits,
model dimensions, seeds, optimizer, epochs, checkpoint rule, and metrics fixed.
The numerical proximity between `0.7418` and this GConvGRU's `0.735942` is not
evidence of parity or improvement.

## Limits

- The pinned `FX` values are already standardized across the source series;
  the JSON has no inverse transform or causal preprocessing record.
- The source has ordered weeks but no dates or missingness mask.
- This is one small graph, one target, one horizon, one split, and three seeds.
- Configuration was developed while test output was visible. This is an
  engineering witness, not an untouched benchmark result.
- The run reports the final epoch; it does not select or restore a validation
  checkpoint.
- Full-batch training makes the small witness deterministic and cheap after JIT
  capture. Smaller batches work but change optimization and compile more
  shapes.
- No matched PyG Temporal training result exists yet.

## Reproduce

```console
uv run --locked python -m experiments.run chickenpox_forecast DEV=CPU SEED=0 EPOCHS=50 HISTORY=8 BS=357 HIDDEN=4 LR=0.01
uv run --locked python -m experiments.run chickenpox_forecast DEV=CPU SEED=1 EPOCHS=50 HISTORY=8 BS=357 HIDDEN=4 LR=0.01
uv run --locked python -m experiments.run chickenpox_forecast DEV=CPU SEED=2 EPOCHS=50 HISTORY=8 BS=357 HIDDEN=4 LR=0.01
```

Forecast unrolling, readouts, and training remain under `experiments/`.
`TGCN` and `GConvGRU` now live in `tinymesh.nn`. The other public result is the
batched fixed-graph execution and temporal-window contract.
