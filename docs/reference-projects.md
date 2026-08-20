# Reference projects

tinymesh has one runtime dependency: tinygrad. Two pinned, read-only submodules
are consumed by executable evidence without becoming runtime imports.

| Project | Revision | What tinymesh studies | What tinymesh does not copy |
| --- | --- | --- | --- |
| tinygrad | [`33755a34`](https://github.com/tinygrad/tinygrad/tree/33755a34657d25920914badbe32a9d70489669c7) | Tensor, autograd, compilation, devices, direct module shape, minimal code | mesh semantics or a compatibility wrapper |
| PyTorch Geometric Temporal | [`fe555bc3`](https://github.com/benedekrozemberczki/pytorch_geometric_temporal/tree/fe555bc30ee197755c4b58a89407033a5f383415) | recurrent and period-attention equations, pinned temporal datasets | PyTorch, batch-specific aliases, trainer surfaces, dense adjacency construction |

The priority is deliberate:

1. tinygrad governs API shape and implementation style;
2. tinymesh owns sparse mesh semantics and evidence;
3. PyG Temporal constrains the executable framework comparison.

PyTorch Geometric, TorchGeo, and TerraTorch informed earlier equation and
boundary decisions. Their research records retain exact revision links. They
remain historical sources rather than repository state because no executable
evidence reads their checkouts.

This produces direct classes with ordinary Tensor attributes and `__call__`,
not a framework inside a framework. `tinymesh.nn` contains only components
already used by current experiments. Sequence unrolling, task heads, trainers,
datasets, and evaluation policy stay outside those classes.

At the pinned tinygrad revision, `Linear` and `LSTMCell` are ordinary classes
with Tensor state and `__call__`; there is no module base class or `forward`
protocol. Tinymesh keeps that shape. Stateful graph compositions are direct
classes, while stateless math remains Tensor and `Graph` operations.

Gitlinks move only for an intentional executable study. A pin update records its
upstream delta and compatibility evidence; it never silently changes runtime
behavior. An experiment envelope records only the gitlinks declared by its
catalog entry.
