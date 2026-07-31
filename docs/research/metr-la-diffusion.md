# METR-LA directed diffusion

Directed affinity lowers validation RMSE against both permuted and self-only
controls in two matched seeds, with the advantage growing at 30 and 60
minutes. Self-only recurrence still has lower MAE. This supports a
topology-specific long-horizon error signal, not a model that dominates its
node-local control.

## Factorized mesh

METR-LA supplies a fixed directed spatial domain and an ordered five-minute
time domain:

```text
space G                 time T
sensor u -> sensor v    ... -> t-1 -> t
             \                 /
              joint state (v,t)
```

The conceptual domain is `G x T`, following the
[time-vertex framework](https://arxiv.org/abs/1705.02307). Execution stays
factorized: one lowered `Graph` is reused while `DiffusionGRU` advances hidden
state through 12 snapshots. Tinymesh never constructs product adjacency or a
dense node-pair tensor.

This is a time-varying traffic field on a directed sensor mesh. The DCRNN
affinity is not a complete street graph, and the source does not declare the
speed or distance units needed to invent physical travel times.

## Inputs

Every input fact is available at its snapshot:

```text
[B,12,207,6]
       |
       +-- normalized speed; missing input becomes 0 after normalization
       +-- observed mask
       +-- sin/cos daily phase
       +-- sin/cos weekly phase
```

Daily and weekly phase are continuous across midnight. Speed normalization,
sensor fallback means, and all baseline statistics use observed training rows
only. Targets, masks, target-time splits, raw-speed metrics, and persistence
remain identical to the existing [METR-LA forecast](metr-la-forecast.md).

## Model

For each period, public Tinymesh components construct three aligned channels.
This is the smallest [DCRNN-style](https://arxiv.org/abs/1707.01926) recurrent
test that preserves the source affinity:

```text
local values --------------------------+
forward affinity diffusion ------------+--> DiffusionGRU --> H[t]
reverse affinity diffusion ------------+
H[t-1] --------------------------------+

H[11] --> ReLU --> Linear(32,12) --> persistence + residual
```

`DirectedDiffusion` source-normalizes the original affinity in both graph
directions. `DiffusionGRU` keeps the local channel beside both transported
channels inside its update, reset, and candidate paths. Sequential unrolling
makes period `t` depend on earlier hidden state; unlike A3T-GCN, periods are not
encoded independently from the same zero state.

At hidden width 32 the model has 11,436 parameters. Twelve recurrent steps
perform 48 sparse sums after the fixed diffusion weights are prepared once.
Topology storage remains `O(N + E)` and no `[N,N]` or `[N,E]` carrier enters
the model.

## Controls

The same initialization, parameters, features, split, batches, loss, and
checkpoint policy receive:

```text
true       original directed support + affinity
permuted   one fixed node relabeling + aligned affinity
self       self edges with unit affinity
```

The declared graph gate required true affinity to beat both controls on MAE
and RMSE in at least two of three matched seeds. Model test stayed closed.

## Model selection

```text
head=residual   loss=mae        hidden=32
epochs=3        batch=512       learning_rate=0.001
seeds=0,1       checkpoint=each epoch
```

At seed 0, one epoch made self-only look best on both metrics. Three epochs
changed the structural conclusion: every topology continued improving, while
true affinity became best on RMSE. This is direct evidence that the one-epoch
probe undertrained the graph model.

Hidden width 16 reached `3.8025` MAE and `7.4732` RMSE after one epoch, versus
`3.7765` and `7.3958` at width 32. It removed 8,016 parameters but only 13.92
seconds end to end in those ordered runs, so width 32 earned the frozen
quality budget. That timing is model-selection context, not a performance
benchmark.

## Validation result

Values are mean ± sample standard deviation across seeds 0 and 1:

| Topology | Validation MAE | Validation RMSE |
| --- | ---: | ---: |
| persistence | 3.8547 | 7.6462 |
| true affinity | 3.5264 ± 0.0212 | **7.0685 ± 0.0071** |
| permuted affinity | 3.5389 ± 0.0023 | 7.1375 ± 0.0211 |
| self-only | **3.4560 ± 0.0085** | 7.1330 ± 0.0277 |

True affinity lowers mean RMSE by 0.97% relative to permuted affinity and
0.90% relative to self-only. It wins all four paired overall RMSE
comparisons. It does not win MAE: self-only wins both paired comparisons, and
permuted affinity narrowly wins seed 1.

The RMSE separation grows with forecast distance:

| Topology | 15 minutes | 30 minutes | 60 minutes |
| --- | ---: | ---: | ---: |
| true affinity | 5.6797 | **7.0243** | **8.7231** |
| permuted affinity | 5.6832 | 7.0708 | 8.8692 |
| self-only | **5.6664** | 7.0701 | 8.8654 |

At 30 and 60 minutes, true affinity beats both controls in each individual
seed. At 15 minutes self-only is slightly better on mean RMSE. Huber loss
repeats the tradeoff at seed 0: true affinity reaches `3.6931` MAE and
`6.8997` RMSE, while self-only reaches `3.5669` and `6.9805`. Changing the
loss does not recover both objectives.

## Decision

The declared all-metric graph gate fails. After true affinity lost self-only
MAE in seeds 0 and 1, seed 2 could not produce the required two passing seeds
and was not run. Model test remains unopened.

The narrower result is useful:

```text
local recurrence     lower typical absolute error
directed diffusion   lower large and long-horizon error
                                  |
                                  v
next: explicit local path + gated spatial residual
```

The current GRU concatenates local and transported fields inside every gate.
The next architecture should keep the inherent/local forecast intact and add
a zero-initialized spatial correction, then repeat true, permuted, and
self-only controls. This follows the separation motivation in
[D2STGNN](https://arxiv.org/abs/2206.09112) without yet adding dynamic edges,
node identity, or a learned graph.

Typed relations, hierarchy, and physical integration from
[MeshGraphNets](https://huggingface.co/papers/2010.03409),
[RIGNO](https://huggingface.co/papers/2501.19205), and
[PhyMPGN](https://huggingface.co/papers/2410.01337) remain broader Tinymesh
research, not claims supported by this result.

No public component changes. `DirectedDiffusion` and `DiffusionGRU` remain
correct sparse compositions; the result constrains how a forecast should
combine them.

## Reproduce

The Metal comparison used these commands with `MODEL=true`, `permuted`, and
`self`, once for each seed:

```console
uv run --locked python -m experiments.run metr_la_diffusion DEV=METAL EPOCHS=3 MODEL=true HEAD=residual LOSS=mae SEED=0 BS=512 HIDDEN=32 LR=0.001 CHECKPOINT_EVERY=1
uv run --locked python -m experiments.run metr_la_diffusion DEV=METAL EPOCHS=3 MODEL=permuted HEAD=residual LOSS=mae SEED=0 BS=512 HIDDEN=32 LR=0.001 CHECKPOINT_EVERY=1
uv run --locked python -m experiments.run metr_la_diffusion DEV=METAL EPOCHS=3 MODEL=self HEAD=residual LOSS=mae SEED=0 BS=512 HIDDEN=32 LR=0.001 CHECKPOINT_EVERY=1
uv run --locked python -m experiments.run metr_la_diffusion DEV=METAL EPOCHS=3 MODEL=true HEAD=residual LOSS=mae SEED=1 BS=512 HIDDEN=32 LR=0.001 CHECKPOINT_EVERY=1
uv run --locked python -m experiments.run metr_la_diffusion DEV=METAL EPOCHS=3 MODEL=permuted HEAD=residual LOSS=mae SEED=1 BS=512 HIDDEN=32 LR=0.001 CHECKPOINT_EVERY=1
uv run --locked python -m experiments.run metr_la_diffusion DEV=METAL EPOCHS=3 MODEL=self HEAD=residual LOSS=mae SEED=1 BS=512 HIDDEN=32 LR=0.001 CHECKPOINT_EVERY=1
```

True and self-only ran at revision
[`1d81c7be`](https://github.com/spatioterra-ai/tinymesh/commit/1d81c7be7da49a6f6b687d1cd845f7a8fb7ab68d).
Permuted controls ran at
[`87dddb14`](https://github.com/spatioterra-ai/tinymesh/commit/87dddb14114020685fac23bc46ddd77d9445568b),
whose only execution change moves the fixed timeout into the catalog. Model,
data, features, splits, optimization, references, and metrics are identical.
Every model result records `evaluate_test=false` and `test=null`.

The original implementation smoke at
[`90f53dca`](https://github.com/spatioterra-ai/tinymesh/commit/90f53dca8ce19833771cf0b35a92672f081139c9)
processed 512 full-source windows with 11,436 parameters and 48 sparse calls.
