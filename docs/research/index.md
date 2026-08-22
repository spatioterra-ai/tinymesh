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
| Should MBTA headway use events or a fixed clock as source truth? | A 597-event, 1,109-relation mesh reproduces all 575 derivable labels. 30/60-second clocks preserve identity but are 89.1%/78.2% empty; five-minute bins merge 90 identities. | Retain reversible departure events. Derive clocks only as matched controls after the task fixes their aggregation and missingness policy. |
| Does the public LAMP export support a forecast-sufficient population? | A capped 28-day acquisition retains 1,050,259 rows, but 228,746 lack reproducible active-Schedule identity; 221,220 are added-trip rows, and missing identity removes whole route-days. | Stop before task design. Do not filter disrupted service or treat unnamed scheduled values as reversible Schedule truth. |

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
- [MBTA Blue Line replay](mbta-replay-data.md) — version-aligned operations,
  rejected mixed-stop targets, and validated movement-derived headway.
- [MBTA departure-event mesh](mbta-event-mesh.md) — reversible physical
  departures, typed causal relations, strict prefixes, and measured clock
  alternatives.
- [MBTA event population](mbta-population.md) — bounded 28-day acquisition and
  a negative Schedule-identity decision that blocks task design.

## Forecast evidence

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
