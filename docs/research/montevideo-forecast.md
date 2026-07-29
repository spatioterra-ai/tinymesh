# Montevideo forecast

This record fixes the causal evaluation boundary before comparing learned
directed models.

## Decision

Tinymesh loads one raw lag, assigns examples by target hour, derives
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

## Limits

This protocol covers one-hour-ahead prediction, one fixed graph, a 24-hour
history, regular hourly rows, and per-node standardization. It has no calendar,
weather, missingness, mask, multi-horizon decoder, randomized split, generic
trainer, or model-quality claim.

## Reproduce

```console
DEV=CPU uv run python -m experiments.montevideo_forecast
DEV=METAL uv run python -m unittest tests.test_montevideo_forecast
```
