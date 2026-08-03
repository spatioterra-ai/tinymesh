# MUTAG masked JEPA

This stage asks whether the working JEPA mechanism changes useful graph-level
representations on real variable-size graphs. It is an intentionally smaller
question than Graph-JEPA parity.

## Protocol

```text
train graphs only
      |
      +-- mask every third atom -- online 2xSAGE -- predictor --+
      |                                                       MSE
      +-- complete atom view --- EMA 2xSAGE -- stop gradient --+

complete graph -- frozen target encoder -- linear probe -- class
```

Each fold builds one sparse disjoint union of its training molecules. A second
sparse graph maps nodes to graph rows for mean pooling; neither path creates a
dense node-pair tensor. The model sees seven one-hot atom types plus one mask
bit and the source topology. It does not see bond labels.

Five stratified folds are rebuilt for each seed. Self-supervision and the
linear probe use only training graphs and training labels respectively. The
frozen target encoder embeds held-out graphs after pretraining. All probes use
the same fixed optimizer budget and train-only standardization.

## Controls and decision

```text
majority       label-frequency floor
summary        atom proportions + bond proportions + node count
random encoder same frozen SAGE before JEPA training
JEPA encoder   same frozen SAGE after JEPA training
```

Pretraining earns continuation only if the JEPA probe improves over the random
encoder without driving target variation toward zero. The summary is a strong
non-neural reference, not a required promotion threshold. Train accuracy is
recorded so an underfit probe cannot masquerade as a representation failure.

This protocol uses one fixed masked whole-graph target. It has no graph
partition, target position, random-walk structural encoding, attention,
edge-aware GNN, or hyperbolic objective. A negative result therefore rejects
this simple task, not JEPA or Graph-JEPA.

## Sources

[I-JEPA v3](https://arxiv.org/abs/2301.08243v3) supplies the asymmetric encoder,
stop-gradient EMA target, smaller predictor, and latent L2 mechanism.
[Graph-JEPA v3](https://arxiv.org/abs/2309.16014v3) supplies the train-only
pretraining and frozen linear-probe evaluation boundary. Its graph-specific
partition, position, and objective choices are deferred deliberately.

## Decision

At revision
[`bc44ff8`](https://github.com/spatioterra-ai/tinymesh/tree/bc44ff86596562ab0f2b3caeb7e351930939bf1d),
three Metal runs produced 15 held-out folds. Values below are mean and
population standard deviation across the three seed-level fold means.

| Measurement | Result |
| --- | ---: |
| Initial latent MSE | 0.07993 ± 0.00302 |
| Final latent MSE | 0.000099 ± 0.000011 |
| Initial target variation | 0.00971 ± 0.00022 |
| Final target variation | 0.00882 ± 0.00124 |
| Majority accuracy | 66.50% ± 0.00% |
| Summary accuracy | 83.48% ± 0.44% |
| Random encoder accuracy | 76.23% ± 0.55% |
| JEPA encoder accuracy | 76.24% ± 1.54% |
| JEPA minus random | +0.00 ± 1.13 points |

The seed-level JEPA deltas were `-0.51`, `+1.58`, and `-1.05` percentage
points. It won four folds, tied five, and lost six. Latent MSE fell by `99.88%`
and every target gradient was zero, but target variation contracted by `9.2%`
and the frozen representation did not improve consistently over its identical
random initialization.

The simple masked whole-graph objective therefore stops here. Low pretraining
loss is not evidence of useful transfer. The next experiment must add the
graph-specific information this task omitted—multiple patches and target
position—before considering a reusable JEPA API. Every component remains
research-only; `Graph`, `MUTAG`, and `SAGEConv` already own the reusable
contracts.

## Reproduce

```console
uv run --locked python -m experiments.run mutag_jepa DEV=METAL EMA=0.99 FOLDS=5 HIDDEN=16 LR=0.01 MASK_EVERY=3 PROBE_LR=0.05 PROBE_STEPS=150 SEED=0 STEPS=100
uv run --locked python -m experiments.run mutag_jepa DEV=METAL EMA=0.99 FOLDS=5 HIDDEN=16 LR=0.01 MASK_EVERY=3 PROBE_LR=0.05 PROBE_STEPS=150 SEED=1 STEPS=100
uv run --locked python -m experiments.run mutag_jepa DEV=METAL EMA=0.99 FOLDS=5 HIDDEN=16 LR=0.01 MASK_EVERY=3 PROBE_LR=0.05 PROBE_STEPS=150 SEED=2 STEPS=100
```
