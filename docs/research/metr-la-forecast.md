# METR-LA forecast

Tinymesh reproduces the PyG Temporal A3T-GCN task shape without inheriting its
evaluation policy. A persistence-anchored residual head now improves the
temporal floor slightly and consistently, but self-only topology beats the
real sensor graph in every matched seed. The current result supports temporal
residual learning, not spatial message passing.

## Architecture

The pinned PyG Temporal
[`A3TGCN2`](https://github.com/benedekrozemberczki/pytorch_geometric_temporal/blob/fe555bc30ee197755c4b58a89407033a5f383415/torch_geometric_temporal/nn/recurrent/attentiontemporalgcn.py#L83-L164)
applies one shared T-GCN independently to each input period, then learns one
softmax weight per period. It is temporal attention over fixed-state T-GCN
encodings, not recurrent unrolling from one period into the next:

```text
x[t-11] -- TGCN(H0) -- a[0] --+
x[t-10] -- TGCN(H0) -- a[1] --+
              ...              +--> sum --> ReLU --> Linear --> y[t+1:t+12]
x[t]    -- TGCN(H0) -- a[11] --+
```

`tinymesh.nn.PeriodAttention` owns the learned convex mixture over `P`
same-shaped states. `A3TGCN` composes it with the existing `TGCN`. One class
handles both `[P,N,F]` and `[B,P,N,F]` because Tinymesh graph operations already
preserve arbitrary leading axes; no batch-size constructor argument or
`A3TGCN2` alias is needed.

The 2→32 encoder plus 32→12 task head has 6,840 parameters and 12 sparse calls
per forward pass. PyG's equivalent has 6,936 parameters. Tinymesh omits three
redundant graph-convolution biases: each would feed immediately into a biased
gate projection and can be folded into that gate bias without changing the
represented functions.

The direct head preserves PyG parity. The experiment-only residual head asks
the network to predict a correction to the causal persistence forecast:

```text
latest observed speed ----------------------+
                                            |
x[t-11:t] --> A3TGCN --> ReLU --> Linear --> + --> y[t+1:t+12]
```

Its linear layer starts at zero, so epoch zero is exactly persistence.
Checkpoint selection can therefore keep the known floor when learned
corrections are harmful. The anchor, head, objective, and selection policy
remain experiment concerns; none expands the public `A3TGCN` contract.

## Parity and policy

Both paths use 207 sensors, the 1,722-edge thresholded sensor graph, two input
features, 12 five-minute input rows, and 12 future speed targets. The full
source produces the same 34,249 sliding windows and the PyG example's 27,399
80% training-window count.

The tutorial passes only `edge_index`, so both parity paths treat the selected
edges as unit-weight. The loader's distance affinity remains aligned and
available, but it is not evidence for this architecture; a weighted comparison
requires its own matched control. The
[directed-diffusion experiment](metr-la-diffusion.md) owns that next question.

The trustworthy experiment deliberately differs after that structural parity:

| Boundary | PyG example | Tinymesh evidence |
| --- | --- | --- |
| split | 80/20 after overlapping windows | 70/10/20 by target time before windows |
| speed fit | all rows, including future and zero sentinels | observed training rows only |
| missing input | normalized zero sentinel | training-mean value in normalized space |
| loss | unmasked standardized MSE | observed-target masked MSE, MAE, or Huber |
| metrics | standardized MSE | raw-speed MAE and RMSE |
| controls | none | train mean, persistence, daily mean, permuted graph, self graph |

```text
raw speed + timestamps + observed mask
                  |
          target-time 70/10/20
                  |
       train-only speed/time fit
                  |
       gather starts x fixed offsets
                  |
      [B,12,207,2] + persistence anchor
                  |
               A3TGCN
                  |
         masked loss / raw metrics
```

The split retains already-observed history across a boundary but drops the 11
windows whose future targets would cross each boundary. That yields 23,967
training, 3,416 validation, and 6,844 test windows.

## Temporal floor

The controls use only training observations. Persistence carries the latest
observed value in the 12-row history and falls back to the per-sensor training
mean. Daily mean is fitted per sensor and five-minute slot.

| Control | Validation MAE | Validation RMSE | Test MAE | Test RMSE |
| --- | ---: | ---: | ---: | ---: |
| train mean | 7.516 | 11.640 | 7.520 | 11.940 |
| persistence | **3.855** | **7.646** | **4.232** | **8.145** |
| daily mean | 5.823 | 9.952 | 5.155 | 9.004 |

Persistence is therefore the model gate. At 15, 30, and 60 minutes its test
MAE is 3.494, 4.196, and 5.363; test RMSE is 6.390, 8.008, and 10.280.

## Execution boundary

Normalized features, targets, and masks transfer to the execution device once.
Each batch is one tensor gather from `starts × offsets`; partial batches are
padded to the compiled shape with a false target mask. Topology remains sparse
through all 12 A3T-GCN calls. The residual anchor is derived causally from the
same input history and falls back to the per-sensor training mean.

The first bounded smoke at revision
[`827b5f2`](https://github.com/spatioterra-ai/tinymesh/commit/827b5f2074af7bebd06932b722f26a3e05f75f9b)
proved full-size sparse forward, backward, and optimizer execution. Batch 512
then made complete local training practical. No Modal or paid compute was used.

Prediction JITs are scoped to one evaluation pass. TinyJit captures parameter
buffers, so a predictor must not survive an optimizer update. A regression
guards this boundary after an early full-epoch run exposed stale checkpoint
evaluation.

## Model selection

The PyG-parity direct head does not clear persistence. At seed 0, 12 epochs of
masked MSE reach validation MAE 6.554 and RMSE 9.962. A residual head is the
smallest useful correction: the known causal forecast flows directly to the
output while A3T-GCN learns only a delta.

Masked MSE improves RMSE but worsens MAE. Huber narrows that conflict. Masked
MAE with the exact persistence anchor is the first validation-selected
configuration to improve both metrics, so the matched comparison freezes:

```text
head=residual   loss=mae        hidden=32
epochs=3        batch=512       learning_rate=0.001
seeds=0,1,2     checkpoint=each epoch
```

## Matched topology result

Revision
[`508805e`](https://github.com/spatioterra-ai/tinymesh/commit/508805ea920df477be051c251f48773431b1275d)
ran true, isomorphically permuted, and self-only topology under the frozen
budget. The three seed envelopes completed on Metal in 356.42, 341.67, and
323.60 seconds.

```console
uv run --locked python -m experiments.run metr_la_forecast DEV=METAL EPOCHS=3 MODEL=all HEAD=residual LOSS=mae SEED=0 BS=512 HIDDEN=32 LR=0.001 CHECKPOINT_EVERY=1
uv run --locked python -m experiments.run metr_la_forecast DEV=METAL EPOCHS=3 MODEL=all HEAD=residual LOSS=mae SEED=1 BS=512 HIDDEN=32 LR=0.001 CHECKPOINT_EVERY=1
uv run --locked python -m experiments.run metr_la_forecast DEV=METAL EPOCHS=3 MODEL=all HEAD=residual LOSS=mae SEED=2 BS=512 HIDDEN=32 LR=0.001 CHECKPOINT_EVERY=1
```

Values are mean ± sample standard deviation across seeds:

| Topology | Validation MAE | Validation RMSE | Test MAE | Test RMSE |
| --- | ---: | ---: | ---: | ---: |
| persistence | 3.8547 | 7.6462 | 4.2323 | 8.1450 |
| true | 3.8342 ± 0.0001 | 7.5651 ± 0.0027 | 4.2090 ± 0.0001 | 8.0664 ± 0.0025 |
| permuted | 3.8461 ± 0.0013 | 7.6106 ± 0.0020 | 4.2229 ± 0.0012 | 8.1105 ± 0.0018 |
| self-only | **3.8230 ± 0.0009** | **7.4836 ± 0.0068** | **4.1931 ± 0.0008** | **7.9745 ± 0.0070** |

True topology improves validation MAE by 0.53% and RMSE by 1.06% over
persistence; its test improvements are 0.55% and 0.96%. It beats the permuted
graph on MAE and RMSE in every seed on both splits. That is not a graph win:
self-only topology beats true topology on the same 12 paired comparisons and
improves test MAE by 0.93% and RMSE by 2.09% over persistence.

```text
persistence < permuted < true < self-only
              learned temporal correction ----^
              useful neighbor mixing: no
```

The result says the residual temporal head learns a small repeatable
correction. Unit-weight GCN neighbor mixing removes useful node-local
information in this architecture. Distance affinity, directed diffusion, and
other graph operators remain separate hypotheses.

These test numbers are development evidence, not an untouched benchmark
claim. The original runner evaluated test for each model-selection probe even
though decisions used validation. Current training defaults to
validation-only; `TEST=1` is now an explicit final-evaluation boundary.

## Promotion and limits

`PeriodAttention` is public because it owns one parameterized equation and one
shape invariant independent of T-GCN. `A3TGCN` is public because it is the
small standard `TGCN` plus `PeriodAttention` composition with batched and
unbatched shape, gradient, sparse-call, and live METR-LA evidence. The task
head, persistence anchor, objectives, target-time split, normalization, mask
policy, baselines, checkpointing, false graphs, and run modes remain under
`experiments/`.

The evidence proves framework-independent task construction, full-size sparse
training, and a small node-local temporal improvement. It rejects a spatial
advantage from this unit-weight A3T-GCN configuration. It does not prove
PyTorch numeric parity, benchmark rank, affinity value, production accuracy,
or a general absence of graph value.
