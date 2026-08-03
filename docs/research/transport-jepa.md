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

At revision
[`d7dadea`](https://github.com/spatioterra-ai/tinymesh/tree/d7dadea2d6bd2aee7cc22d4d156b083c1fcd4003),
three Metal runs produced the registered seed-level results below. RMSE values
are mean and population standard deviation across seeds. Gain is each trained
target encoder's random-initialization RMSE minus its trained RMSE, so positive
is better.

| Arm | Random validation | Trained validation | Gain | Seed wins | Trained test | Test gain |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `true` | 0.047428 ± 0.003955 | 0.046152 ± 0.004810 | **+0.001276 ± 0.001287** | **3/3** | 0.047617 ± 0.005229 | **+0.001617 ± 0.001282** |
| `permuted` | 0.052101 ± 0.005740 | 0.053413 ± 0.006710 | -0.001312 ± 0.001507 | 1/3 | 0.053861 ± 0.005884 | -0.000783 ± 0.000806 |
| `temporal` | 0.079379 ± 0.002475 | 0.078719 ± 0.003006 | +0.000661 ± 0.000939 | 2/3 | 0.080902 ± 0.003426 | +0.000530 ± 0.000928 |
| `spatial` | 0.044436 ± 0.004942 | **0.043257 ± 0.002213** | +0.001179 ± 0.003353 | 1/3 | **0.043544 ± 0.001709** | +0.000794 ± 0.002884 |

Persistence validation RMSE was `0.097644`; the standardized raw-history probe
reached `0.055889 ± 0.000010`. The `true` representation beat both in every
seed. Its validation gains were `+0.000166`, `+0.003081`, and `+0.000580`, and
all three test gains remained positive. Mean latent loss fell from
`0.894219` to `0.656218`. Target gradients were exactly zero, the EMA target
moved by `2.840953 ± 0.230082`, and embedding variation changed from
`0.961556 ± 0.003974` to `0.962472 ± 0.004822` rather than collapsing.

The representation gate passes. Causal latent prediction consistently improves
the true product encoder over its identical random initialization, and test
repeats the validation direction.

The joint-mesh gate fails. Trained `true` beat `permuted` and `temporal` in all
three validation seeds by mean RMSE margins of `0.007260` and `0.032566`, but
beat `spatial` in only one. `spatial` was better on mean validation RMSE by
`0.002895` and on mean test RMSE by `0.004073`, again winning two seeds. The
probe already sees all four spatially encoded history rows, so these results do
not justify learned temporal edges on this task.

Continue causal JEPA as a research mechanism, with `spatial` as the architecture
incumbent and the full product as a control. Do not promote `Mesh`, `MeshConv`,
encoders, predictor, objective, data policy, probes, or controls. `Graph.cartesian`,
batched endpoint gathering, and batched edge sums already own the reusable
contracts.

## Reproduce

```console
uv run --locked python -m experiments.run transport_jepa DEV=METAL EMA=0.998 HIDDEN=8 HISTORY=4 HORIZON=4 LR=0.001 PROBE_LR=0.05 PROBE_STEPS=150 SEED=0 STEPS=100
uv run --locked python -m experiments.run transport_jepa DEV=METAL EMA=0.998 HIDDEN=8 HISTORY=4 HORIZON=4 LR=0.001 PROBE_LR=0.05 PROBE_STEPS=150 SEED=1 STEPS=100
uv run --locked python -m experiments.run transport_jepa DEV=METAL EMA=0.998 HIDDEN=8 HISTORY=4 HORIZON=4 LR=0.001 PROBE_LR=0.05 PROBE_STEPS=150 SEED=2 STEPS=100
```
