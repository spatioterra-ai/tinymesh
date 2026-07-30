# METR-LA forecast

Tinymesh now reproduces the PyG Temporal A3T-GCN task shape without inheriting
its evaluation policy. The protocol establishes the temporal floor and a
runnable sparse model path; it does not yet establish that the sensor graph
improves traffic forecasting.

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

`tinymesh.nn.A3TGCN` is the same composition over the existing `TGCN`. One
class handles both `[P,N,F]` and `[B,P,N,F]` because Tinymesh graph operations
already preserve arbitrary leading axes; no batch-size constructor argument or
`A3TGCN2` alias is needed.

The 2→32 encoder plus 32→12 task head has 6,840 parameters and 12 sparse calls
per forward pass. PyG's equivalent has 6,936 parameters. Tinymesh omits three
redundant graph-convolution biases: each would feed immediately into a biased
gate projection and can be folded into that gate bias without changing the
represented functions.

## Parity and policy

Both paths use 207 sensors, the 1,722-edge thresholded sensor graph, two input
features, 12 five-minute input rows, and 12 future speed targets. The full
source produces the same 34,249 sliding windows and the PyG example's 27,399
80% training-window count.

The tutorial passes only `edge_index`, so both parity paths treat the selected
edges as unit-weight. The loader's distance affinity remains aligned and
available, but it is not evidence for this architecture; a weighted comparison
requires its own matched control.

The trustworthy experiment deliberately differs after that structural parity:

| Boundary | PyG example | Tinymesh evidence |
| --- | --- | --- |
| split | 80/20 after overlapping windows | 70/10/20 by target time before windows |
| speed fit | all rows, including future and zero sentinels | observed training rows only |
| missing input | normalized zero sentinel | training-mean value in normalized space |
| loss | unmasked standardized MSE | observed-target masked standardized MSE |
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
      [B,12,207,2] -> [B,207,12]
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
through all 12 A3T-GCN calls.

Protocol and bounded optimizer-step evidence are separate:

```console
uv run --locked python -m experiments.run metr_la_forecast DEV=CPU
uv run --locked python -m experiments.run metr_la_forecast DEV=METAL STEPS=3 SEED=0 BS=32 HIDDEN=32 LR=0.001
```

Revision
[`827b5f2`](https://github.com/spatioterra-ai/tinymesh/commit/827b5f2074af7bebd06932b722f26a3e05f75f9b)
recorded the complete protocol in 21.96 seconds on CPU. Its Metal smoke
completed in 135.11 seconds overall, with 65.47 seconds inside three optimizer
steps over 96 windows:

```json
{
  "parameters": 6840,
  "sparse_calls": 12,
  "first_loss": 1.123005986213684,
  "last_loss": 0.8795691728591919
}
```

The loss movement proves the masked objective reaches A3T-GCN parameters
through the full 207-node sparse graph. Three shuffled batches are not a model
comparison or convergence result.

On the current local Metal device, one full width-32 epoch did not complete
inside the runner's 600-second bound at batch 32 or 128. Those stopped
diagnostics produce no model score. A matched three-seed comparison of true,
isomorphically relabeled, and self-only topology therefore remains pending
explicit bounded GPU execution.

## Promotion and limits

`A3TGCN` is public because it is a small standard composition with batched and
unbatched shape, gradient, sparse-call, and live METR-LA evidence. The task
head, target-time split, normalization, mask policy, baselines, checkpointing,
false graphs, and run modes remain under `experiments/`.

The current evidence proves framework-independent task construction and
full-size sparse forward/backward execution. It does not prove PyTorch numeric
parity, benchmark rank, graph value, affinity value, production accuracy, or
completed training.
