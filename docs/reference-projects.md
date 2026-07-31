# Reference projects

tinymesh has one runtime dependency: tinygrad. Five pinned, read-only
submodules make design comparisons reproducible without becoming runtime
imports.

| Project | Revision | What tinymesh studies | What tinymesh does not copy |
| --- | --- | --- | --- |
| tinygrad | [`1095bbe4`](https://github.com/tinygrad/tinygrad/tree/1095bbe409f5ed3cbeca74aa3c2ca09bef634309) | Tensor, autograd, compilation, devices, direct module shape, minimal code | mesh semantics or a compatibility wrapper |
| PyTorch Geometric | [`726310a4`](https://github.com/pyg-team/pytorch_geometric/tree/726310a486eae37a89cd6359072b82bbbbb71579) | graph-layer equations, conventional names, host and dense comparisons | PyTorch, `MessagePassing`, registries, aggregation plug-ins |
| PyTorch Geometric Temporal | [`fe555bc3`](https://github.com/benedekrozemberczki/pytorch_geometric_temporal/tree/fe555bc30ee197755c4b58a89407033a5f383415) | recurrent and period-attention equations, pinned temporal datasets | PyTorch, batch-specific aliases, trainer surfaces, dense adjacency construction |
| TorchGeo | [`468c670b`](https://github.com/torchgeo/torchgeo/tree/468c670bc94c961eb80e6c0ad32ed147852c367b) | explicit geospatial query, dataset, and sampling boundaries | GIS dependencies or general coordinate types without a caller |
| TerraTorch | [`375356c9`](https://github.com/torchgeo/terratorch/tree/375356c9ba1d919c39816abaf6b499afc303497f) | backbone, neck, head, and task composition seams | factories or registries before configuration-driven composition exists |

The priority is deliberate:

1. tinygrad governs API shape and implementation style;
2. tinymesh owns sparse mesh semantics and evidence;
3. PyG and PyG Temporal constrain mathematical parity;
4. TorchGeo and TerraTorch inform future boundaries.

This produces direct classes with ordinary Tensor attributes and `__call__`,
not a framework inside a framework. `tinymesh.nn` contains only components
already used by current experiments. Sequence unrolling, task heads, trainers,
datasets, and evaluation policy stay outside those classes.

At the pinned tinygrad revision, `Linear` and `LSTMCell` are ordinary classes
with Tensor state and `__call__`; there is no module base class or `forward`
protocol. Tinymesh keeps that shape. Stateful graph compositions are direct
classes, while stateless math remains Tensor and `Graph` operations.

Gitlinks move only for an intentional study. A pin update records its upstream
delta and compatibility evidence; it never silently changes runtime behavior.
