# Montevideo forecast

This record fixes the causal evaluation boundary and compares one node-local
forecast with three matched directed-diffusion forecasts.

The later [seasonal-floor study](montevideo-seasonal.md) finds a stronger
train-only temporal control. The learned comparison below predates that floor.

## Decision

At tinygrad revision
[`dd16d5a`](https://github.com/tinygrad/tinygrad/tree/dd16d5aead62e0207c0c3c50c19bc8b67e176c55),
all four learned models lose to persistence on three CPU seeds. Coordinate and
road distance do not improve unit diffusion under the frozen promotion rule.
Tinymesh retains the loader, sparse operator, recurrent witness, protocol, and
negative result; it promotes no geometry API.

The comparison loads one raw lag, assigns examples by target hour, derives
normalization from training rows only, and evaluates predictions in raw
passenger-count units.

```text
raw hourly signal[743,675,1]
          |
          +--> train targets       0:520
          +--> validation targets 520:594
          +--> test targets       594:743
          |
          v
24-hour causal windows
```

Validation and test windows may read observations before their target
boundary. They never send a later target into training, normalization,
checkpoint selection, or optimizer state.

## Why one loader lag

The source feature and target are the same hourly passenger series. With
`lags=1`, signal row `t` pairs observation `t` with target `t+1`. A 24-step
recurrent window then has one owner for each hour:

```text
X[t:t+24] -> y[t+24]
```

Loading four lags and then unrolling a recurrent history would send overlapping
lag vectors through time and obscure that contract.

## Split

The pinned signal has 743 next-hour examples after one loader lag:

| Split | Target rows | Windows | Historical context |
| --- | ---: | ---: | ---: |
| Train | 520 | 497 | none before row 0 |
| Validation | 74 | 74 | preceding 23 train hours |
| Test | 149 | 149 | preceding 23 development hours |

Splitting the target axis first preserves all validation and test targets.
Only inputs cross a boundary, and every crossed input predates its target.

## Normalization

Input and target mean and population standard deviation are fitted separately
for every node from training rows:

```text
x_norm = (x - mean_x_train) / scale_x_train
y_norm = (y - mean_y_train) / scale_y_train
```

A constant training node receives scale one, preserving its observed mean
without division by zero. Perturbation tests change every validation and test
value while leaving both training standardizers unchanged.

Learned models optimize normalized targets. Metrics restore the target
standardizer first; MAE and RMSE therefore use passenger-count units.

## Baselines

Three raw-unit controls are fixed before model training:

```text
zero          predict 0
persistence   predict the latest observed hour
train mean    predict each node's mean training target
```

The zero baseline is necessary because about 80% of target cells are zero. A
small aggregate error without this control would be easy to misread.

At tinygrad revision `dd16d5a`, CPU returns:

| Split | Targets | Zero fraction | Baseline | MAE | RMSE |
| --- | ---: | ---: | --- | ---: | ---: |
| Train | 335,475 | 0.804856 | Zero | 0.744622 | 3.388523 |
| Train | 335,475 | 0.804856 | Persistence | 0.554301 | 1.744267 |
| Train | 335,475 | 0.804856 | Train mean | 0.731815 | 2.312357 |
| Validation | 49,950 | 0.824324 | Zero | 0.617618 | 2.923249 |
| Validation | 49,950 | 0.824324 | Persistence | 0.487187 | 1.555442 |
| Validation | 49,950 | 0.824324 | Train mean | 0.661758 | 2.049604 |
| Test | 100,575 | 0.795029 | Zero | 0.795058 | 3.524940 |
| Test | 100,575 | 0.795029 | Persistence | 0.584131 | 1.822791 |
| Test | 100,575 | 0.795029 | Train mean | 0.746667 | 2.367824 |

Three nodes are constant in the training input and target rows. Persistence is
the strongest baseline on both validation and test. A learned model must beat
it before any graph comparison is useful.

## Directed recurrence

The three geometry comparisons use the same recurrent class:

```text
S = [X, H]
B(S) = [S, forward(S), reverse(S)]
Z, R = sigmoid(Linear(B(S)))
C = [X, R * H]
H_next = Z * H + (1 - Z) * tanh(Linear(B(C)))
```

One gate projection and one candidate projection require four sparse sums per
step. A hand-written two-step edge loop matches the cell, gradients reach its
candidate parameters through time, and UOp inspection finds no `[N,N]` or
`[N,E]` carrier. The node-local LSTM is contextual only because its cell and
parameter count differ:

| Model | Parameters | Sparse calls per step |
| --- | ---: | ---: |
| Node-local LSTM | 117 | 0 |
| Unit diffusion | 197 | 4 |
| Coordinate diffusion | 197 | 4 |
| Road diffusion | 197 | 4 |

Resetting the tinygrad seed before constructing each diffusion model gives the
three variants identical learned tensors. They differ only in fixed affinity:

```text
unit        1
coordinate  1 / Euclidean edge distance
road        1 / observed road distance
```

Each affinity is source-normalized independently in the forward and reverse
graphs. The graph, reverse graph, and normalized edge fields are realized once
per variant and reused across every recurrent step.

## Identifiability

Most nodes have at most one relevant edge. Source normalization cancels any
positive scalar affinity on those rows, so geometry can change only branching
rows:

| Affinity | Forward weights changed from unit | Reverse weights changed from unit |
| --- | ---: | ---: |
| Unit | 0 / 690 | 0 / 690 |
| Coordinate | 42 / 690 | 46 / 690 |
| Road | 42 / 690 | 46 / 690 |

Seven nodes have no outgoing edge and nine have no incoming edge. This narrow
identifiability bound makes the later near-equality expected rather than
surprising.

## Controlled run

Every run uses a 24-hour history, hidden width `4`, batch size `32`, normalized
MSE, Adam at `0.01` with `fused=False`, and at most ten epochs. Checkpoints are
evaluated at epochs `0`, `5`, and `10`; the earliest minimum validation RMSE is
restored before one test evaluation. All runs selected epoch `10`.

Values below are raw-unit `MAE / RMSE`. Fit time includes initial validation,
TinyJit compilation, training, and checkpoint validation; it excludes data
loading and final test evaluation.

| Model | Seed | Epoch 0 validation | Epoch 5 validation | Epoch 10 validation | Test | Fit seconds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| LSTM | 0 | 0.773565 / 2.130590 | 0.594309 / 1.970278 | 0.587521 / 1.963438 | 0.700037 / 2.375411 | 338.5 |
| LSTM | 1 | 0.943795 / 2.560979 | 0.597095 / 1.983778 | 0.590856 / 1.972007 | 0.702229 / 2.379200 | 254.0 |
| LSTM | 2 | 0.629104 / 2.070458 | 0.604032 / 2.009833 | 0.595015 / 1.997978 | 0.708976 / 2.412658 | 252.7 |
| Unit | 0 | 1.005479 / 2.618773 | 0.614382 / 2.035433 | 0.604846 / 2.013533 | 0.708483 / 2.385347 | 122.5 |
| Unit | 1 | 0.691149 / 2.126590 | 0.601928 / 2.006748 | 0.592666 / 2.005164 | 0.706813 / 2.418844 | 97.0 |
| Unit | 2 | 0.838236 / 2.326164 | 0.605914 / 2.020039 | 0.600949 / 2.009671 | 0.705073 / 2.390109 | 104.5 |
| Coordinate | 0 | 1.005492 / 2.618968 | 0.614508 / 2.036291 | 0.605023 / 2.014548 | 0.708628 / 2.385691 | 137.9 |
| Coordinate | 1 | 0.691129 / 2.126598 | 0.601965 / 2.006673 | 0.592752 / 2.005073 | 0.706838 / 2.418273 | 139.6 |
| Coordinate | 2 | 0.838212 / 2.326117 | 0.606007 / 2.020301 | 0.601066 / 2.009920 | 0.705156 / 2.390023 | 132.1 |
| Road | 0 | 1.005478 / 2.618756 | 0.614580 / 2.036709 | 0.605080 / 2.014892 | 0.708605 / 2.385422 | 150.7 |
| Road | 1 | 0.691108 / 2.126406 | 0.601978 / 2.006829 | 0.592820 / 2.005310 | 0.706816 / 2.417806 | 105.8 |
| Road | 2 | 0.838167 / 2.325736 | 0.606054 / 2.020547 | 0.601117 / 2.010133 | 0.705174 / 2.389914 | 183.1 |

The twelve fits took `33.6` minutes in aggregate on an Apple M4 MacBook Air.
Mean plus sample standard deviation across seeds is:

| Model | Validation MAE | Validation RMSE | Test MAE | Test RMSE |
| --- | ---: | ---: | ---: | ---: |
| Persistence | 0.487187 | 1.555442 | 0.584131 | 1.822791 |
| LSTM | 0.591131 ± 0.003754 | 1.977808 ± 0.017986 | 0.703747 ± 0.004659 | 2.389090 ± 0.020498 |
| Unit | 0.599487 ± 0.006220 | 2.009456 ± 0.004189 | 0.706790 ± 0.001705 | 2.398100 ± 0.018122 |
| Coordinate | 0.599614 ± 0.006263 | 2.009847 ± 0.004738 | 0.706874 ± 0.001736 | 2.397995 ± 0.017694 |
| Road | 0.599672 ± 0.006256 | 2.010112 ± 0.004791 | 0.706865 ± 0.001716 | 2.397714 ± 0.017544 |

## Promotion decision

A geometry variant must beat unit diffusion on mean validation MAE and RMSE,
with the paired direction agreeing in at least two seeds. Negative deltas are
better:

| Variant minus unit | Validation MAE delta | Paired wins | Validation RMSE delta | Paired wins | Qualifies |
| --- | ---: | ---: | ---: | ---: | --- |
| Coordinate | +0.000127 | 0 / 3 | +0.000391 | 1 / 3 | No |
| Road | +0.000185 | 0 / 3 | +0.000656 | 0 / 3 | No |

Neither variant reaches the test-confirmation gate. For completeness, their
test MAE is also worse in every seed; small mean RMSE changes
(`-0.000105` coordinate and `-0.000386` road) improve in two seeds, while MAE
improves in zero. The rule requires both metrics.

The learned models also remain well behind persistence. This experiment proves
that the complete sparse causal path trains and compares fixed geometry
without leakage. It does not show that topology or geometry improves this
forecast.

## Limits

This result covers one-hour-ahead prediction, one fixed graph, one 24-hour
history, three seeds, regular hourly rows, and per-node standardization.
Roughly 80% of targets are zero, and scalar geometry can distinguish only a
small fraction of edges after normalization. There is no calendar, weather,
missingness mask, multi-horizon decoder, architecture search, learned affinity,
generic trainer, public model factory, or production claim. The model and
operator remain experiments.

## Reproduce

```console
DEV=CPU uv run python -m experiments.montevideo_forecast
DEV=CPU MODEL=all SEED=-1 EPOCHS=10 HISTORY=24 BS=32 HIDDEN=4 LR=0.01 CHECKPOINT_EVERY=5 uv run python -m experiments.montevideo_forecast
DEV=METAL uv run python -m unittest tests.test_montevideo_forecast tests.test_directed_gru
```
