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

Pending clean revision-bound runs for seeds 0, 1, and 2. Every component in
this file remains research-only; `Graph`, `MUTAG`, and `SAGEConv` already own
the reusable contracts.

## Reproduce

```console
uv run --locked python -m experiments.run mutag_jepa DEV=METAL EMA=0.99 FOLDS=5 HIDDEN=16 LR=0.01 MASK_EVERY=3 PROBE_LR=0.05 PROBE_STEPS=150 SEED=0 STEPS=100
uv run --locked python -m experiments.run mutag_jepa DEV=METAL EMA=0.99 FOLDS=5 HIDDEN=16 LR=0.01 MASK_EVERY=3 PROBE_LR=0.05 PROBE_STEPS=150 SEED=1 STEPS=100
uv run --locked python -m experiments.run mutag_jepa DEV=METAL EMA=0.99 FOLDS=5 HIDDEN=16 LR=0.01 MASK_EVERY=3 PROBE_LR=0.05 PROBE_STEPS=150 SEED=2 STEPS=100
```
