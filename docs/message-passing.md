# Mean message passing

## Decision

At Tinygrad revision
[`0f98212`](https://github.com/tinygrad/tinygrad/tree/0f98212e80f18b0722d04811798bfefd4bf6d93d),
an experimental mean-GraphSAGE layer composes Tinygrad linear maps with
Tinymesh's CSR aggregation and trains on CPU and Metal. This is the first real
model caller for the sparse primitive. It remains in `experiments/`: the result
proves composition and first-order parameter learning, not a stable Tinymesh API
or useful model quality.

No code enters `src/tinymesh` in this stage. The caller supplies the missing
model evidence, but the alpha custom-kernel boundary and disabled default kernel
optimization still make package admission premature.

## One message-passing layer

For node state `x`, incoming neighborhood `N(v)`, and indegree `d_v`, the layer
computes:

```text
message:    m(u -> v) = W_neighbor x_u
aggregate:  a_v       = sum(m(u -> v)) / max(1, d_v)
update:     y_v       = W_root x_v + a_v
```

The neighbor transform has no bias, so transforming before the mean is
equivalent to transforming the mean. Placing it before aggregation makes its
parameter gradient cross the transpose-CSR backward and therefore tests the
actual sparse composition. An isolated node has `a_v = 0` and retains only its
root path.

Because this experiment uses fixed topology, its realized CSR buffers are
reused per device and its inverse-degree vector is reused per device and dtype.
The topology owns both derived caches; they never enter model parameters.

This is the mean special case of
[GraphSAGE](https://arxiv.org/abs/1706.02216), which learns node representations
by aggregating local neighborhood features rather than assigning every node an
independent embedding. The paper is available as
[PDF](https://arxiv.org/pdf/1706.02216) and
[TeX source](https://arxiv.org/src/1706.02216). PyG's pinned
[`SAGEConv`](https://github.com/pyg-team/pytorch_geometric/blob/2.8.0/torch_geometric/nn/conv/sage_conv.py)
implements the broader production surface; this experiment deliberately omits
bias, activation, normalization, sampling, batching, and alternative
aggregators.

## Learning witness

The checked-in witness has two source nodes with features `1` and `-1`, two
target nodes with identical feature `0`, and edges:

```text
0 -> 2
1 -> 3
```

The targets are `1` and `-1`. A root-only function cannot distinguish nodes 2
and 3 because their own features are identical. The neighbor parameter can:

```text
prediction at node 2 = W_neighbor * 1
prediction at node 3 = W_neighbor * -1
```

Starting both weights at zero gives loss `1` and neighbor gradient `-2`.
One SGD step at learning rate `0.5` sets the neighbor weight to `1` and the loss
to `0`; the root weight remains `0`. The output gradient begins at nodes 2 and
3, transpose CSR returns it to nodes 0 and 1, and Tinygrad then differentiates
their linear messages into `W_neighbor`.

This constructed case establishes that topology supplies indispensable signal
and that gradients reach a parameter on the source side of the sparse
operation. It says nothing about optimization difficulty, generalization,
depth, temporal learning, or a real dataset.

## Reproduce

```console
DEV=CPU uv run python -m unittest tests.test_mean_sage
DEV=METAL uv run python -m unittest tests.test_mean_sage
DEV=CPU uv run python -m experiments.mean_sage
DEV=METAL uv run python -m experiments.mean_sage
```
