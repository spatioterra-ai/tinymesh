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

Each frozen representation also serves as an exact in-memory search index.
Train-only standardization and unit normalization define cosine similarity; each
held-out graph takes the label of its nearest training graph. Test graphs are
queries, never indexed candidates. This measures whether the latent geometry is
searchable without adding an approximate index or storage dependency.

An objective earns continuation only if its trained encoder beats its own
random initialization on mean held-out accuracy and does so in at least two of
three seed-level fold means, without collapsing graph-level target variation.
The summary remains a reference, not a promotion threshold. Position or loss
matters only through the matched arm comparisons above. No arm, seed, budget,
or threshold changes after clean results are visible.

## Deliberate departures from paper parity

The experiment catalog marks this as an `ablation`, not a `reproduction` of the
reported MUTAG score. It uses the same canonical MUTAG dataset, but random
rather than METIS partitions, MLP rather than
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

At revision
[`ddea5f2`](https://github.com/spatioterra-ai/tinymesh/tree/ddea5f26cd9299f2e57eebcc309071c862fe23b7), three
Metal runs produced 15 held-out folds. Values below are mean and population
standard deviation across the three seed-level fold means.

| Arm | Initial -> final loss | Initial -> final variation | Random | Trained | Delta | Seed wins | Fold W/T/L |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `euclidean` | 0.08650 -> 0.000274 | 0.00498 -> 0.00579 | 84.23% | 82.25% | -1.98 ± 3.11 pt | 1/3 | 3/2/10 |
| `hyperbola` | 0.29347 -> 0.01290 | 0.00498 -> 0.01053 | 84.23% | 82.61% | -1.62 ± 2.61 pt | 1/3 | 4/4/7 |
| `hyperbola_mse` | 0.59835 -> 0.02689 | 0.00498 -> 0.01046 | 84.23% | 82.80% | -1.44 ± 3.11 pt | 2/3 | 6/1/8 |
| `positionless` | 0.29542 -> 0.01836 | 0.00295 -> 0.00902 | 78.90% | 81.38% | **+2.48 ± 1.23 pt** | **3/3** | **8/3/4** |

Majority accuracy was `66.50%`; the atom/bond/count summary reached
`83.48% ± 0.44%`. Every target gradient was zero and no arm collapsed its
graph-level variation.

Only `positionless` passes the registered continuation gate. Its seed deltas
were `+4.22`, `+1.61`, and `+1.61` points. The same hyperbola objective with
RWSE started `5.33` points higher but pretraining erased most of that advantage:
its paired delta was `4.10` points worse. MSE and smooth L1 both fail on mean
paired transfer, so loss choice does not explain the result. Low objective loss
again does not predict a useful representation.

Continue patch prediction without structural position as the minimal incumbent;
retain RWSE only as a negative control. This is evidence for the learning
mechanism, not a competitive MUTAG model: every trained arm remains below the
fixed summary. Patch construction, RWSE, encoder orchestration, objectives,
training, and probes remain research-only. `Graph`, `Graph.sum_edges`,
`GINEConv`, `MUTAG`, and the shared MUTAG evaluation protocol already own the
reusable contracts.

### Search extension

At revision
[`f66ab38`](https://github.com/spatioterra-ai/tinymesh/tree/f66ab383c4c6c88d74b183d06ba35564270d8937),
the same three registered runs also measured exact standardized-cosine 1-nearest
neighbor accuracy. Values remain the mean and population standard deviation
across the three seed-level fold means.

| Representation | Random search | Trained search | Delta |
| --- | ---: | ---: | ---: |
| Fixed summary | — | **83.67% ± 1.03%** | — |
| `euclidean` | 79.44% ± 1.11% | 76.96% ± 2.93% | -2.48 ± 3.88 pt |
| `hyperbola` | 79.44% ± 1.11% | 77.47% ± 1.93% | -1.97 ± 1.61 pt |
| `hyperbola_mse` | 79.44% ± 1.11% | 77.65% ± 1.97% | -1.79 ± 1.36 pt |
| `positionless` | 76.21% ± 0.24% | 71.61% ± 3.70% | -4.60 ± 3.52 pt |

Every trained arm regresses its matched random search on mean accuracy, and the
fixed summary beats every learned representation. The positive positionless
linear-probe result therefore does not imply useful nearest-neighbor geometry:
a trained readout can recover label information that cosine neighborhoods do
not expose.

Keep exact search as an evaluation control. Promote no embedding, retrieval, or
index API, and add no vector database. A later retrieval stage needs an explicit
similarity objective or independent structural relevance labels before it can
claim semantically searchable graphs.

## Reproduce

```console
uv run --locked python -m experiments.run mutag_graph_jepa DEV=METAL EMA=0.99 FOLDS=5 HIDDEN=16 LR=0.005 PATCHES=8 PROBE_LR=0.05 PROBE_STEPS=150 RW=8 SEED=0 STEPS=80 TARGETS=3
uv run --locked python -m experiments.run mutag_graph_jepa DEV=METAL EMA=0.99 FOLDS=5 HIDDEN=16 LR=0.005 PATCHES=8 PROBE_LR=0.05 PROBE_STEPS=150 RW=8 SEED=1 STEPS=80 TARGETS=3
uv run --locked python -m experiments.run mutag_graph_jepa DEV=METAL EMA=0.99 FOLDS=5 HIDDEN=16 LR=0.005 PATCHES=8 PROBE_LR=0.05 PROBE_STEPS=150 RW=8 SEED=2 STEPS=80 TARGETS=3
```

## Paper reproduction

`mutag_graph_jepa_reproduction` is a separate, frozen port of the official
MUTAG configuration at revision `72df1b7704921001ea012a21f840300fbc792cdd`:
32 patch slots, 15-step RWSE, two GINE layers, width 512, four attention layers,
one context, three targets, smooth L1, 50 epochs, ten stratified folds, five
published seeds, and scikit-learn logistic regression.

The executable reference schedules learning rate from held-out test loss despite
the paper's claim that test data is unseen during pretraining. This port preserves
that behavior for code parity; its score must not be read as a clean blind test.

Every MUTAG graph has fewer than 32 nodes. The official `metis_subgraph`
therefore never invokes METIS for this dataset: it permutes 32 slots, assigns
one node to each occupied slot, shifts the highest occupied slot to 31, expands
each occupied patch by one hop, and masks the empty slots. The tinygrad port
matches that behavior directly without adding METIS, PyTorch, or PyG.

This command is intentionally expensive and isolated from routine verification:

```console
uv run --locked --with scikit-learn==1.7.2 python -m experiments.run mutag_graph_jepa_reproduction DEV=METAL
```

The reproduction has not produced a retained score yet. Until the complete
command finishes on a clean revision, it establishes executable protocol parity,
not numerical reproduction of the paper's reported `91.25 ± 2.10` accuracy.
The reference's final printer instead reports the mean of the five within-run
ten-fold standard deviations; the reproduction output preserves that metric so
the discrepancy stays visible.
