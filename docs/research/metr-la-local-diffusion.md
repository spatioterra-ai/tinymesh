# METR-LA local diffusion

This experiment asks whether directed transport can improve METR-LA's
long-horizon RMSE without giving up the lower MAE of node-local recurrence.
It composes existing sparse diffusion inside the experiment; it adds no public
API.

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

## Ownership

`DirectedDiffusion` remains the public spatial primitive. The local cell,
readouts, and horizon gate remain experiment-owned until repeated evidence
reveals a smaller stable API. This separation follows the motivation of
[D2STGNN](https://arxiv.org/abs/2206.09112), not an implementation claim of
paper parity.

## Reproduce

Inspect the protocol or run one bounded optimizer step:

```console
uv run --locked python -m experiments.run metr_la_local_diffusion DEV=CPU
uv run --locked python -m experiments.run metr_la_local_diffusion DEV=METAL STEPS=1 SEED=0 BS=512 HIDDEN=32 HEAD=residual LOSS=mae
```

Matched validation begins only after the implementation revision is merged.
