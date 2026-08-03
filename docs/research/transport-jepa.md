# Controlled node-time JEPA

This stage asks whether latent future prediction improves a reusable
representation of a known spatiotemporal process—and whether the correct joint
mesh matters. It is a controlled mechanism test, not a traffic benchmark.

## Data and causal boundary

The existing controlled transport generator evolves independent zero-mean
fields over one directed 24-node graph:

```text
x[t+1] = 0.25 x[t] + 0.55 forward(x[t]) + 0.20 reverse(x[t])

context                   hidden target
x[0] x[1] x[2] x[3]  ->  x[4] x[5] x[6] x[7]
```

Training, validation, and test contain 128, 32, and 32 independent trajectories
from the already pinned data seeds. Self-supervision sees training trajectories
only. The downstream probe sees training targets only; validation decides and
test confirms.

## Product mesh

Each four-step block lowers through `time.cartesian(space)`:

```text
node token       (t, v) -> t * 24 + v

relation 0       (t-1, v) -> (t, v)             temporal
relation 1       (t, u)   -> (t, v)             forward affinity
relation 2       (t, v)   -> (t, u)             reverse affinity
```

Forward and reverse edge values use the same source- and destination-normalized
coefficients as the stated transport law. They multiply source messages; they
are never added as standalone features. A zero field therefore produces zero
state, preventing static topology from becoming the prediction shortcut.

| Arm | Time edges | Space edges | Product edges |
| --- | ---: | ---: | ---: |
| `true` | 72 | 288 true | 360 |
| `permuted` | 72 | 288 degree-matched false | 360 |
| `temporal` | 72 | 0 | 72 |
| `spatial` | 0 | 288 true | 288 |

`temporal` tests node-local history. `spatial` gives the frozen probe every
time token but forbids messages between them. The matched four arms distinguish
correct space, false space, no space, and no learned time edge.

## Objective and frozen evaluation

```text
context product -> online 3xMeshConv -> LayerNorm -> predictor -- L1 --+
                                                                    |
future product  -> EMA target 3xMeshConv -> LayerNorm -> stopgrad ---+
```

`MeshConv` is a research-only relation-weighted sum composed from
`Graph.edge_values` and `Graph.sum_edges`. Three layers span the four-step path.
The final token-wise LayerNorm removes scale as an objective shortcut. The
online encoder and predictor receive gradients; the full target encoder starts
identically and follows the online encoder with EMA `0.998`.

All four context token rows remain in the frozen node representation. One
shared linear head sees `[node, 4 * hidden]` and predicts the four future field
values. Its inputs use training-only mean and standard deviation. Every arm is
compared with its identical target encoder before pretraining and uses the same
probe initialization and budget. Persistence and a standardized raw-history
probe are non-neural references.

V-JEPA v1 grounds latent L1 prediction, an EMA target, masked spatiotemporal
tokens, and frozen evaluation. This experiment deliberately uses its causal
mask ablation rather than the paper's stronger non-causal multi-block mask: the
whole future block is unavailable to the context encoder. It replaces video
attention with sparse relation messages and is not a V-JEPA reproduction.

## Registered decision

Seed `17` was used before registration to verify execution, eliminate a final
ReLU collapse, reject additive topology-only messages, retain all token rows,
and select V-JEPA's slower target regime. It is excluded from evidence below.
Seeds `0`, `1`, and `2` are untouched.

The representation earns continuation only if `true` improves validation RMSE
over its own random encoder on mean and in at least two of three seeds, with
zero target gradient and final embedding variation at least half its initial
value. The explicit joint mesh is supported only if trained `true` also beats
trained `permuted`, `temporal`, and `spatial` in at least two of three paired
validation runs. Test repeats the validation decision; it does not select one.
No seed, arm, budget, feature, or threshold changes after clean formal results
are visible.

## Decision

Pending clean revision-bound runs for seeds 0, 1, and 2. `Graph.cartesian`,
batched endpoint gathering, and batched edge sums already own the reusable
contracts. `Mesh`, `MeshConv`, encoders, predictor, loss, data policy, probes,
and controls remain research-only.

## Reproduce

```console
uv run --locked python -m experiments.run transport_jepa DEV=METAL EMA=0.998 HIDDEN=8 HISTORY=4 HORIZON=4 LR=0.001 PROBE_LR=0.05 PROBE_STEPS=150 SEED=0 STEPS=100
uv run --locked python -m experiments.run transport_jepa DEV=METAL EMA=0.998 HIDDEN=8 HISTORY=4 HORIZON=4 LR=0.001 PROBE_LR=0.05 PROBE_STEPS=150 SEED=1 STEPS=100
uv run --locked python -m experiments.run transport_jepa DEV=METAL EMA=0.998 HIDDEN=8 HISTORY=4 HORIZON=4 LR=0.001 PROBE_LR=0.05 PROBE_STEPS=150 SEED=2 STEPS=100
```
