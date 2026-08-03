# MUTAG Graph-JEPA ablation

The masked whole-graph task fit its latent target but tied the same random
encoder. This stage isolates the graph-specific information that task omitted:
local patches, target position, and prediction geometry.

## Protocol

```text
MUTAG graph
   |
   +-- balanced random partition -- one-hop expansion --+
   |                                                     |
   +-- exact node RWSE -------------------------- max per patch
                                                         |
atoms + bonds -- 2xGINE -- mean patch pool --------------+
                         |
       one context ------+-- online MLP -----------------+-- predictor --+
       three targets ----+-- stopgrad EMA target MLP --------------------+-- loss
       target RWSE ------- position projection -----------+
```

Eight non-empty base partitions are balanced by a seeded node permutation and
fixed for one fold. Each gains its undirected one-hop neighborhood; induced
edges preserve source COO order and aligned bond labels. A two-layer GINE patch
encoder consumes one-hot atoms and bonds, then sparse mean pooling produces
one vector per patch.

RWSE records each node's probability of returning to itself after steps one
through eight of the outgoing random walk. The implementation advances one
sparse probability map per origin and stores only `[N, K]` output—not a dense
`[N, N]` adjacency or carrier. Its worst-case preprocessing cost is
`O(KN(N + E))` time and `O(NK + N + E)` memory, so it remains research-only.
Elementwise maximum gives each expanded patch a structural position.

Each optimizer step chooses one context and three distinct targets per graph.
The context MLP and smaller predictor receive gradients. The target MLP starts
as an exact copy, receives no gradient, and follows the context MLP by EMA. The
shared patch encoder learns only through selected context patches. There is no
test-graph self-supervision.

## Fixed arms

| Arm | Target position | Target code | Loss |
| --- | --- | --- | --- |
| `euclidean` | RWSE | full latent vector | MSE |
| `hyperbola` | RWSE | `(cosh(mean(z)), sinh(mean(z)))` | smooth L1 |
| `hyperbola_mse` | RWSE | same 2D hyperbola | MSE |
| `positionless` | zero | same 2D hyperbola | smooth L1 |

All arms share partitions, context-target choices, encoder initialization,
optimizer budget, folds, and probe initialization where shapes permit. The
hyperbola changes target dimensionality, so its comparison with Euclidean is
directional rather than a parameter-matched geometry claim. `hyperbola` versus
`hyperbola_mse` isolates loss; `hyperbola` versus `positionless` isolates
structural conditioning.

## Evaluation and decision

Five stratified folds are rebuilt for seeds 0, 1, and 2. Self-supervision sees
only training graphs; frozen probes see only training labels and train-only
standardization. Every arm is compared with its own identical frozen encoder
before pretraining. Majority and atom/bond/count summary controls retain the
same fixed probe protocol as the preceding stage.

An objective earns continuation only if its trained encoder beats its own
random initialization on mean held-out accuracy and does so in at least two of
three seed-level fold means, without collapsing graph-level target variation.
The summary remains a reference, not a promotion threshold. Position or loss
matters only through the matched arm comparisons above. No arm, seed, budget,
or threshold changes after clean results are visible.

## Deliberate departures from paper parity

This is a bounded architectural ablation, not a reproduction of the reported
MUTAG score. It uses random rather than METIS partitions, MLP rather than
attention encoders, hidden width 16 rather than 512, eight rather than 32
patches, fixed rather than online partitions, five folds over three seeds, and
a fixed tinygrad probe rather than scikit-learn logistic regression.

These choices are evidence-backed reductions. Graph-JEPA reports a competitive
random-partition ablation and a close MLP ablation. MUTAG graphs have only 10 to
28 nodes, so 32 base patches would necessarily be empty or singleton-centered.
The smaller protocol isolates the ideas without adding METIS, PyTorch,
scikit-learn, NumPy, or any other runtime dependency.

## Sources

[Graph-JEPA v3](https://arxiv.org/abs/2309.16014v3) supplies one-hop expanded
patches, node-level RWSE pooled by maximum, one context with multiple targets,
the EMA target, MLP ablation, unit-hyperbola target, and smooth-L1 objective.
[GINE v3](https://arxiv.org/abs/1905.12265v3) grounds the edge-aware patch
encoder. Exact PDF and TeX revisions are pinned by the paper registry.

## Decision

Pending clean revision-bound runs for seeds 0, 1, and 2. Patch construction,
RWSE, encoder orchestration, objectives, training, and probes remain
research-only. `Graph`, `Graph.sum_edges`, `GINEConv`, `MUTAG`, and the shared
MUTAG evaluation protocol already own the reusable contracts.

## Reproduce

```console
uv run --locked python -m experiments.run mutag_graph_jepa DEV=METAL EMA=0.99 FOLDS=5 HIDDEN=16 LR=0.005 PATCHES=8 PROBE_LR=0.05 PROBE_STEPS=150 RW=8 SEED=0 STEPS=80 TARGETS=3
uv run --locked python -m experiments.run mutag_graph_jepa DEV=METAL EMA=0.99 FOLDS=5 HIDDEN=16 LR=0.005 PATCHES=8 PROBE_LR=0.05 PROBE_STEPS=150 RW=8 SEED=1 STEPS=80 TARGETS=3
uv run --locked python -m experiments.run mutag_graph_jepa DEV=METAL EMA=0.99 FOLDS=5 HIDDEN=16 LR=0.005 PATCHES=8 PROBE_LR=0.05 PROBE_STEPS=150 RW=8 SEED=2 STEPS=80 TARGETS=3
```
