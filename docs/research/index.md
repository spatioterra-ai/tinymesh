# Research

Research records bind a claim to an exact revision, protocol, and measurement.
They are evidence, not API promises. This page is the current decision ledger;
[`experiments.CATALOG`](../experiments.md) owns the runnable inventory.

## Current decisions

| Question | Current evidence | Decision |
| --- | --- | --- |
| Can sparse graph learning stay native to tinygrad? | Unit and scalar-weighted aggregation, endpoint projection, target softmax, and first-order gradients remain sparse on CPU and Metal. | Yes, for fixed topology. Keep the CSR backend private while `Tensor.custom_kernel` remains alpha. |
| Does metric geometry require a geo dependency? | Position tensors compose into displacement, distance, radial weights, and sparse messages. | No. Data adapters own coordinate frames and units; add a type only when it must own a new invariant. |
| Can the models use spatial structure? | True-topology diffusion wins controlled transport and transfers to unseen graph sizes. | Yes under an identifiable local law. Preserve false-topology and node-local controls. |
| Does graph structure improve the real forecasts tested so far? | Chickenpox is tied with a node-local model and Montevideo geometry loses to persistence. On METR-LA, factorized true transport beats both topology controls on MAE and RMSE in two seeds, but the earlier self-only model still dominates it. | Graph structure measurably affects METR-LA forecasts; it has not improved the best model. Compare bounded convergence before changing architecture or API. |
| Can tinygrad express JEPA learning over graph patches? | Aligned latent loss falls 90.9%; reversed and positionless controls lose on CPU and Metal while the EMA target receives zero gradient. | Yes. Keep the mechanism research-only until a real representation task justifies a reusable owner. |
| Can variable-size graph collections stay dependency-free and sparse? | MUTAG lowers 188 graphs and 7,442 labeled directed edges identically on CPU and Metal from one bounded, pinned archive. | Yes. Keep source labels canonical and derive model features with tinygrad operations. |

## Sparse core

- [Sparse aggregation](sparse-aggregation.md) — destination CSR, transpose
  backward, scaling evidence, and the alpha-kernel boundary.
- [Mean GraphSAGE](mean-sage.md) — the first trainable caller of sparse mean.
- [GCN](gcn.md) — degree normalization composed around the same sum.
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

- [T-GCN](tgcn.md) — fixed-graph recurrence through one spatial and temporal
  transition.
- [GConvGRU](gconv-gru.md) — Chebyshev graph mixing inside recurrent gates.

## Representation

- [JEPA mechanics](jepa-mechanics.md) — asymmetric latent prediction over graph
  patches with an EMA target and shortcut controls.
- [MUTAG masked JEPA](mutag-jepa.md) — frozen graph representations against
  random-encoder and non-neural controls.

## Data boundaries

- [Chickenpox](chickenpox-data.md) — weekly node fields over one pinned graph.
- [Montevideo bus](montevideo-data.md) — hourly fields, projected positions,
  and road distance.
- [METR-LA](metr-la-data.md) — five-minute speed, timestamps, missingness, and
  directed affinity.
- [MUTAG](mutag-data.md) — variable-size molecular graphs with aligned atom,
  bond, and graph labels.

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
