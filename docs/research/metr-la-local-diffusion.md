# METR-LA local diffusion

True directed transport beats both matched controls on every declared metric
in seeds 0 and 1. The earlier self-only `DiffusionGRU` still has lower MAE and
RMSE than this factorized model. The result isolates a repeatable topology
signal; it does not improve the best forecaster or add a public API.

## Factorization

At each five-minute step:

```text
                         +--> local GRU --> local residual --------+
x[t] -------------------+                                         |
                         |                                         v
                         +--> directed diffusion               forecast
                                  |                                ^
                         [forward - x, reverse - x]                 |
                                  |                                |
                             spatial GRU --> spatial residual -- tanh(gate[H])
                                                                   |
persistence -------------------------------------------------------+
```

The forecast is:

```text
persistence + local residual + tanh(gate[h]) * spatial residual
```

The horizon gate starts at exactly zero. The residual head also starts the
local readout at zero, so the complete model initially equals causal
persistence. The local path receives no graph input.

The transported field keeps direction explicit:

```text
[forward(x) - x, reverse(x) - x]
```

Self-only diffusion makes this field exactly zero. True and permuted affinity
therefore differ only through transported information, while every model keeps
the same node-local path and parameter shapes.

## Sparse work

Each period performs one forward and one reverse sparse sum. Twelve history
steps therefore require 24 sparse calls, half the calls of the entangled
`DiffusionGRU`. Topology storage remains `O(N + E)` and no node-by-node,
node-edge, or product space-time adjacency is materialized.

## Controls

```text
true       directed METR-LA affinity
permuted   one fixed isomorphic node relabeling with aligned affinity
self       self edges with unit affinity
```

Features, target windows, initialization, optimizer, loss, checkpoint policy,
and budget remain matched. Model test stays closed.

The frozen validation protocol uses:

```text
head=residual   loss=mae        hidden=32
epochs=3        batch=512       learning_rate=0.001
checkpoint=each epoch
```

True topology advances only if it beats both controls on overall RMSE and
30/60-minute RMSE without exceeding self-only overall MAE in at least two of
three matched seeds. Stop once that gate is satisfied or impossible.

## Validation result

Values are mean ± sample standard deviation across seeds 0 and 1:

| Topology | Validation MAE | Validation RMSE |
| --- | ---: | ---: |
| true transport | **3.5159 ± 0.0185** | **7.1642 ± 0.0085** |
| permuted transport | 3.5317 ± 0.0084 | 7.2086 ± 0.0001 |
| self-only transport | 3.5456 ± 0.0076 | 7.2256 ± 0.0042 |

Across two seeds, two metrics, and two controls, true transport wins all eight
paired overall comparisons. Its mean MAE is 0.45% lower than permuted and 0.84%
lower than self-only; its mean RMSE is 0.62% and 0.85% lower.

Mean validation RMSE shows the topology signal at every reported horizon:

| Topology | 15 minutes | 30 minutes | 60 minutes |
| --- | ---: | ---: | ---: |
| true transport | **5.6689** | **7.0878** | **8.9371** |
| permuted transport | 5.7105 | 7.1389 | 8.9789 |
| self-only transport | 5.7180 | 7.1561 | 8.9984 |

True transport beats both controls at 30 and 60 minutes in each seed, satisfies
the declared gate twice, and stops the protocol before seed 2.

## Incumbent comparison

Passing a topology-identification gate is not the same as improving the model:

| Architecture | Validation MAE | Validation RMSE |
| --- | ---: | ---: |
| earlier self-only `DiffusionGRU` | **3.4560 ± 0.0085** | 7.1330 ± 0.0277 |
| earlier true `DiffusionGRU` | 3.5264 ± 0.0212 | **7.0685 ± 0.0071** |
| factorized true transport | 3.5159 ± 0.0185 | 7.1642 ± 0.0085 |

The earlier self-only model dominates the factorized model: its MAE is 1.73%
lower and its RMSE is 0.44% lower. Factorization improves MAE by 0.30% relative
to the earlier true graph model but worsens RMSE by 1.35%.

Every factorized curve is still improving at epoch 3. The frozen comparison
proves the topology signal at that budget; it does not establish converged or
equal-time model quality.

## Decision

```text
topology identification   pass
predictive improvement    fail against incumbent
public API promotion      no
model test                closed
```

The next experiment should first compare models at a bounded plateau or equal
execution budget. If the local floor still degrades, train and freeze the
incumbent local forecast before fitting a zero-gated transport residual to its
errors. Do not add dynamic edges, node identity, or learned adjacency yet.

## Ownership

`DirectedDiffusion` remains the public spatial primitive. Root subtraction and
direction concatenation stay direct tensor composition in the experiment. The
local cell, readouts, and horizon gate also remain experiment-owned because the
model does not improve the incumbent. This separation follows the motivation of
[D2STGNN](https://arxiv.org/abs/2206.09112), not an implementation claim of
paper parity.

## Reproduce

The matched Metal comparison used:

```console
uv run --locked python -m experiments.run metr_la_local_diffusion DEV=METAL EPOCHS=3 MODEL=true HEAD=residual LOSS=mae SEED=0 BS=512 HIDDEN=32 LR=0.001 CHECKPOINT_EVERY=1
uv run --locked python -m experiments.run metr_la_local_diffusion DEV=METAL EPOCHS=3 MODEL=permuted HEAD=residual LOSS=mae SEED=0 BS=512 HIDDEN=32 LR=0.001 CHECKPOINT_EVERY=1
uv run --locked python -m experiments.run metr_la_local_diffusion DEV=METAL EPOCHS=3 MODEL=self HEAD=residual LOSS=mae SEED=0 BS=512 HIDDEN=32 LR=0.001 CHECKPOINT_EVERY=1
uv run --locked python -m experiments.run metr_la_local_diffusion DEV=METAL EPOCHS=3 MODEL=true HEAD=residual LOSS=mae SEED=1 BS=512 HIDDEN=32 LR=0.001 CHECKPOINT_EVERY=1
uv run --locked python -m experiments.run metr_la_local_diffusion DEV=METAL EPOCHS=3 MODEL=permuted HEAD=residual LOSS=mae SEED=1 BS=512 HIDDEN=32 LR=0.001 CHECKPOINT_EVERY=1
uv run --locked python -m experiments.run metr_la_local_diffusion DEV=METAL EPOCHS=3 MODEL=self HEAD=residual LOSS=mae SEED=1 BS=512 HIDDEN=32 LR=0.001 CHECKPOINT_EVERY=1
```

All runs use revision
[`b62ca7c6`](https://github.com/spatioterra-ai/tinymesh/commit/b62ca7c68d4fafc4875b2da9fc11fd5ff11d3777),
record `evaluate_test=false` and `test=null`, and finish inside the fixed
600-second bound.
