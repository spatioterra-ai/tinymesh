# Research

Research records bind a claim to an exact revision, protocol, and measurement.
They are evidence, not API promises. This page is the current decision ledger;
[`experiments.CATALOG`](../experiments.md) owns the runnable inventory.

## Current decisions

| Question | Current evidence | Decision |
| --- | --- | --- |
| Can sparse graph learning stay native to tinygrad? | Unit, scalar-weighted, and edge-vector aggregation, endpoint projection, target softmax, and first-order gradients remain sparse on CPU and Metal. | Yes, for fixed topology. Keep the CSR backend private while `Tensor.custom_kernel` remains alpha. |
| How does the current runtime compare with PyG on this Mac? | Exact mappings agree within `4.8e-7`. Tinymesh TGCN JIT is about 2.4x faster than eager PyG Temporal on Metal, but compiled PyG Temporal and a compiled one-propagation PyTorch control both beat it in two repetitions; the factored control also wins clearly on CPU. | Require JIT for repeated Tinymesh execution. Keep the one-propagation algebra; reject a framework speed claim and expand shapes, skew, backward, and memory before optimizing the private CSR backend. |
| Does metric geometry require a geo dependency? | Position tensors compose into displacement, distance, radial weights, and sparse messages. | No. Data adapters own coordinate frames and units; add a type only when it must own a new invariant. |
| Can the models use spatial structure? | True-topology diffusion wins controlled transport and transfers to unseen graph sizes. | Yes under an identifiable local law. Preserve false-topology and node-local controls. |
| Does graph structure improve the real forecasts tested so far? | Chickenpox is tied with a node-local model and Montevideo geometry loses to persistence. On METR-LA, factorized true transport beats both topology controls on MAE and RMSE in two seeds, but the earlier self-only model still dominates it. | Graph structure measurably affects METR-LA forecasts; it has not improved the best model. Compare bounded convergence before changing architecture or API. |
| Can tinygrad express JEPA learning over graph patches? | Aligned latent loss falls 90.9%; reversed and positionless controls lose on CPU and Metal while the EMA target receives zero gradient. | Yes. Keep the mechanism research-only until a real representation task justifies a reusable owner. |
| Can variable-size graph collections stay dependency-free and sparse? | MUTAG lowers 188 graphs and 7,442 labeled directed edges identically on CPU and Metal from one bounded, pinned archive. | Yes. Keep source labels canonical and derive model features with tinygrad operations. |
| Does Graph-JEPA improve MUTAG representations? | Positionless patch prediction gains 2.48 probe points over its paired random encoder in all three seeds, but exact cosine search regresses every trained arm and every learned arm trails the fixed graph summary. | Continue the minimal positionless mechanism only as representation research. Claim no searchable geometry; keep JEPA, retrieval, and indexing research-only. |
| Can one sparse graph represent a bounded node-time mesh? | Cartesian products, batched endpoint gathers, and batched GINE pass 30 focused tests on CPU and Metal without dense adjacency. | Yes when joint message passing needs explicit node-time vertices. Keep long fixed-topology sequences factorized. |
| Does causal JEPA learn a useful node-time representation? | The true-mesh encoder improves over its paired random initialization in all three seeds and test agrees, but spatial-only features beat the joint product in two of three. | Continue the objective as research; reject explicit temporal edges as the incumbent on this task. Promote no JEPA API. |
| Does real traffic support factorized causal JEPA? | Latent loss falls 75.6%, but the trained factorized encoder regresses its paired random encoder in all three seeds and loses every simpler control on mean validation RMSE. | Reject this encoder, mask, objective, and budget. Keep test closed and promote no JEPA API. |
| Which LAMP field can support an operational forecast? | Arrival, travel time, and dwell inherit mixed stop provenance. Movement-derived trunk headway reproduces exactly for all 4,274 full-day Blue Line labels and 26,558 physical departures across a seven-day audit. | Retain MBTA and extend the headway replay before specifying a task. Add no public adapter or model yet. |
| Can a regular clock provide a matched carrier for frozen MBTA headway targets? | Across 947,489 departures and 940,551 targets, 30/60/300-second clocks merge 1,459/3,823/120,752 predecessor-target pairs. Even 30 seconds is 93.4% empty within active lane-day spans. | No. Close the snapshot branch and retain reversible departure events as source truth. |
| Does the public LAMP export support a forecast-sufficient population? | A capped 28-day acquisition lowers 947,489 physical departures and reproduces 940,776 movement-headway labels. It measures 201 mismatches, 1,007 boundary labels, 384 ambiguous-order rows, and 78.2% exact Schedule identity. | Advance to a retrospective event-time task. Keep the full source population, make carrier and Schedule masks explicit, and make no online claim. |
| What is the retrospective MBTA next-headway floor? | The frozen task retains 940,551 physical targets. Public plan leads validation at 152.648 s MAE, but the train-only temporal median leads the single test open at 159.788 s versus plan at 166.062 s and persistence at 229.802 s. | Carry both strong controls into Stage 4 and require improvement over each. Keep the claim retrospective. |
| Does directed MBTA topology add next-headway signal? | On the single test opening, the true upstream arm reaches 149.398 s mean MAE versus 155.042 self-only, 155.241 reversed, and 154.771 degree-preserving permuted controls. It wins every seed, route-macro MAE, p90 error, and both Schedule-provenance slices. | Yes, within the frozen retrospective MBTA task. Retain the bounded directional signal; make no universal, online, or every-route claim. |
| Does event-native memory improve the MBTA topology model? | True-topology memory beats every matched memory control in every validation seed, but its 147.726 s mean MAE loses to the frozen 145.793 s topology MLP in every seed. | No for this bounded design and budget. Stop at validation, keep test closed, and add no runtime memory abstraction. |

## Sparse core

- [Mac framework benchmark](framework-benchmark.md) — exact component mappings
  against PyG and pinned PyG Temporal source, with synchronized CPU and Metal
  wall time.
- [Sparse aggregation](sparse-aggregation.md) — destination CSR, transpose
  backward, scaling evidence, and the alpha-kernel boundary.
- [Mean GraphSAGE](mean-sage.md) — the first trainable caller of sparse mean.
- [GCN](gcn.md) — degree normalization composed around the same sum.
- [GINE](gine.md) — learned edge-vector messages over sparse COO-to-CSR reduction.
- [Weighted aggregation](weighted-aggregation.md) — COO edge identity through
  forward, node gradients, and scalar edge gradients.
- [Sparse attention](attention.md) — endpoint projection, target softmax, and
  independently trainable heads.

## Space and direction

- [Spatial structure](spatial-structure.md) — topology, geometry, coordinate
  frames, and ownership.
- [Spatial geometry](spatial-geometry.md) — differentiable metric composition
  with no geo runtime.
- [Directed diffusion](directed-diffusion.md) — sparse source-normalized
  propagation in both graph directions.

## Time and recurrence

- [Sparse Cartesian products](cartesian-product.md) — explicit product lowering,
  edge order, batched edge messages, and the factorized alternative.
- [T-GCN](tgcn.md) — fixed-graph recurrence through one spatial and temporal
  transition.
- [GConvGRU](gconv-gru.md) — Chebyshev graph mixing inside recurrent gates.

## Representation

- [JEPA mechanics](jepa-mechanics.md) — asymmetric latent prediction over graph
  patches with an EMA target and shortcut controls.
- [MUTAG masked JEPA](mutag-jepa.md) — frozen graph representations against
  random-encoder and non-neural controls.
- [MUTAG Graph-JEPA ablation](mutag-graph-jepa.md) — graph patches, structural
  position, and latent objective under one frozen probe protocol.
- [Controlled node-time JEPA](transport-jepa.md) — causal latent prediction over
  sparse product meshes with matched space and time controls.
- [METR-LA factorized JEPA](metr-la-jepa.md) — real causal traffic blocks with
  sparse space, bounded time, and paired representation controls.

## Data boundaries

- [Chickenpox](chickenpox-data.md) — weekly node fields over one pinned graph.
- [Montevideo bus](montevideo-data.md) — hourly fields, projected positions,
  and road distance.
- [METR-LA](metr-la-data.md) — five-minute speed, timestamps, missingness, and
  directed affinity.
- [MUTAG](mutag-data.md) — variable-size molecular graphs with aligned atom,
  bond, and graph labels.
- [GTFS semantic boundary](gtfs-boundary.md) — Schedule and Realtime facts
  lowered into deterministic sparse views without a public transport API.
- [MBTA Blue Line replay](mbta-replay-data.md) — version-aligned operations,
  rejected mixed-stop targets, and validated movement-derived headway.
- [MBTA departure-event mesh](mbta-event-mesh.md) — reversible physical
  departures, typed causal relations, strict prefixes, and measured clock
  alternatives.
- [MBTA event population](mbta-population.md) — bounded 28-day acquisition,
  event lowering, and explicit Schedule-identity limits.
- [MBTA clock boundary](mbta-clock.md) — full-population identity, causal
  collision, and empty-work audit for 30/60/300-second projections.

## Forecast evidence

- [MBTA next-headway task](mbta-headway-task.md) — frozen event-time target,
  temporal split, leakage-safe baselines, validation evidence, and test gate.
- [MBTA topology signal](mbta-topology.md) — sparse as-of neighbor messages,
  matched false-topology controls, and frozen learned validation evidence.
- [MBTA event memory](mbta-event-memory.md) — causal event folds, bounded lane
  state, matched topology controls, and a validation-only stop decision.

- [Chickenpox forecast](chickenpox-forecast.md) — matched node-local and graph
  recurrence without a stable graph advantage.
- [Montevideo forecast](montevideo-forecast.md) — causal evaluation and a
  negative geometry comparison.
- [Montevideo seasonal floor](montevideo-seasonal.md) — the stronger temporal
  control later graph experiments must beat.
- [Montevideo delayed edges](montevideo-delayed-edges.md) — causal graph
  residuals against reversed and permuted controls.
- [METR-LA forecast](metr-la-forecast.md) — A3T-GCN task parity, trustworthy
  evaluation, and a temporal rather than spatial gain.
- [METR-LA directed diffusion](metr-la-diffusion.md) — topology-specific
  long-horizon RMSE gains with a remaining local-MAE tradeoff.
- [METR-LA local diffusion](metr-la-local-diffusion.md) — a zero-gated
  transported residual that identifies topology but loses to the incumbent.
- [Controlled transport](transport-forecast.md) — a positive identifiable
  topology witness.
- [Controlled transfer](transport-transfer.md) — frozen local parameters on
  unseen graph sizes and longer rollouts.
