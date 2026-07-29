# Spatial geometry experiment

This record asks whether metric edge geometry needs a Tinymesh abstraction or
another runtime dependency.

## Decision

At tinygrad revision
[`291ee435b`](https://github.com/tinygrad/tinygrad/tree/291ee435bb5c545428dc9ef86ecc91406b2c1022),
the existing public operations express a differentiable radial spatial
message:

```text
delta_e = P_target[e] - P_source[e]
r_e     = ||delta_e||
w_e     = exp(-alpha r_e)
Y_v     = sum(w_e X_u)                 for every edge e: u -> v
```

No new public type or runtime dependency is needed. `Graph` owns topology and
COO edge identity. Ordinary tinygrad tensors own position, node values, and the
radial parameter. A data adapter still owns coordinate frame, projection, and
units.

This result keeps the public surface unchanged. It does not add
`SpatialGraph`, a geometry transform, or a coordinate container.

## Symmetry contract

Displacement points along the declared directed edge from source to target.
The focused tests establish:

```text
translation P' = P + t       delta, r, w, and Y are unchanged
rotation    P' = R P         delta' = R delta; r, w, and Y are unchanged
relabeling  i' = pi(i)       node outputs and gradients follow pi
edge order  e' = sigma(e)    edge fields follow sigma; node results do not
```

Duplicate edges keep separate COO positions and contribute separately. Empty
topology produces empty edge fields, zero node output, and zero gradients.

Distance-only weighting is invariant to reflection as well as rotation. That
is useful for an isotropic interaction, but it intentionally discards
direction.

## Exact witness

The runnable fixture uses four nodes in a two-dimensional metric frame:

```text
edge                 0 -> 2    1 -> 2    0 -> 1    2 -> 3
distance                   4          5          3          3
alpha                   0.25       0.25       0.25       0.25
weight              0.367879   0.286505   0.472367   0.472367
```

CPU and Metal return, within `1e-5`:

```text
Y = [[ 0.000000,  0.000000],
     [ 0.944733, -0.472367],
     [ 1.022264,  0.491635],
     [-0.944733,  1.889466]]

dL/dw     = [5, 6, 4, 10]
dL/dalpha = -35.792130

dL/dP = [[ 0.459849,  0.472367],
         [ 0.343806, -0.730221],
         [-0.803655,  1.438771],
         [ 0.000000, -1.180916]]

dL/dX = [[ 1.576005, -0.576854],
         [ 0.859514,  0.286505],
         [-0.472367,  0.944733],
         [ 0.000000,  0.000000]]
```

An independent Python edge loop computes displacement, distance, output, and
all four gradients. Tinygrad matches that host reference; neither PyTorch nor
NumPy is used as an oracle.

## Sparse work

For the structural fixture `N=5`, `E=7`, coordinate width `D=3`, and feature
width `H=2`, the scheduled UOps are:

```text
source projection     E * D = 21 independent owners
target projection     E * D = 21 independent owners
radial weight         E = 7 owners, each reducing D = 3
weighted sum          N * H = 10 owners, each traversing one CSR row
```

The declared intermediates have shapes `[E,D]`, `[E]`, `[E]`, and `[N,H]`.
UOp inspection rejects `[N,N]` and `[N,E]` carriers. Topology storage remains
`O(N + E)` and geometry storage is `O(ED)`. This is a complexity result, not a
speed claim.

The existing endpoint-projection and weighted-aggregation tests separately
inspect sparse first-order backward kernels. The combined spatial test adds an
exact host comparison for gradients through both boundaries.

## Dependency boundary

Installed Tinymesh metadata names only `tinygrad` as a runtime requirement.
PyTorch Geometric and TorchGeo remain pinned, reference-only submodules.

At PyG revision
[`726310a`](https://github.com/pyg-team/pytorch_geometric/tree/726310a486eae37a89cd6359072b82bbbbb71579),
`Distance` derives linked-node distance from `data.pos`. `Cartesian` stores the
opposite displacement sign, source minus target; Tinymesh deliberately chooses
target minus source so the vector points along `u -> v`. The scalar distance is
the same.

At TorchGeo revision
[`468c670`](https://github.com/torchgeo/torchgeo/tree/468c670bc94c961eb80e6c0ad32ed147852c367b),
`GeoDataset` owns CRS, resolution, bounds, and reprojection. That remains the
right boundary: longitude and latitude must be projected into a suitable
metric frame before Euclidean distance enters this computation. Zero geo
dependencies do not make unprojected angular coordinates metric.

## Limits

The experiment proves static floating-point positions, Euclidean distance, one
isotropic exponential weight, fixed directed topology, one device, and
first-order gradients. It does not prove geodesy, learned vector messages,
radius or nearest-neighbor graph construction, moving coordinates, faces,
cells, hierarchy, model quality, or performance.

The position-gradient witness uses strictly positive edge distances. The
Euclidean norm is not differentiable at zero, so coincident endpoints remain
outside this gradient contract unless a caller chooses an explicit
regularization or subgradient convention.

A vector reducer earns a public operation only when directional edge features
have a caller. A `Mesh` earns a type only when faces, cells, or hierarchy need
an owner.

## Reproduce

```console
DEV=CPU uv run python -m unittest tests.test_spatial_geometry
DEV=METAL uv run python -m unittest tests.test_spatial_geometry
DEV=CPU uv run python -m experiments.spatial_geometry
DEV=METAL uv run python -m experiments.spatial_geometry
```
