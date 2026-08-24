# Reference projects

tinymesh has one runtime dependency: tinygrad. Pinned, read-only submodules
constrain executable evidence and active staged research without becoming
runtime imports.

| Project | Revision | What tinymesh studies | What tinymesh does not copy |
| --- | --- | --- | --- |
| tinygrad | [`33755a34`](https://github.com/tinygrad/tinygrad/tree/33755a34657d25920914badbe32a9d70489669c7) | Tensor, autograd, compilation, devices, direct module shape, minimal code | mesh semantics or a compatibility wrapper |
| PyTorch Geometric Temporal | [`fe555bc3`](https://github.com/benedekrozemberczki/pytorch_geometric_temporal/tree/fe555bc30ee197755c4b58a89407033a5f383415) | recurrent and period-attention equations, pinned temporal datasets, fixed-topology carriage | PyTorch, batch-specific aliases, trainer surfaces, dense adjacency construction |
| PyTorch Geometric | [`5c6461b2`](https://github.com/pyg-team/pytorch_geometric/tree/5c6461b2305ad068a6d61165b3c55852a11aaa41) | continuous-time event identity, batching, and topology diagnostics | PyTorch, generic storage machinery, graph transforms, or an event-container API |
| Torch Spatiotemporal | [`aa5f313e`](https://github.com/TorchSpatiotemporal/tsl/tree/aa5f313e000d192bdec270748b8d01df5912e58e) | masks, covariates, windows, horizons, scalers, and connectivity derivation | its dataset, trainer, configuration, dense similarity, or conversion framework |
| TorchGeo | [`a9822d4b`](https://github.com/torchgeo/torchgeo/tree/a9822d4b76feb2cf824cd5bb062712d76d6187a4) | coordinate-aware dataset composition and raster sampling boundaries | imagery machinery for transit events or a geospatial base class |
| TerraTorch | [`703f002b`](https://github.com/torchgeo/terratorch/tree/703f002b7102bcf0eaf6f67f6d788fbb81a73838) | the boundary between scalar operations and Earth-observation foundation models | model registries, trainers, or foundation-model dependencies |
| LibCity | [`5a6391d4`](https://github.com/LibCity/Bigscity-LibCity/tree/5a6391d41944e937f2c15e9be85ab7f40ac8b23e) | urban-forecast task taxonomy, controls, and model coverage | its unified executor, configuration, dataset format, or model warehouse |

The priority is deliberate:

1. tinygrad governs API shape and implementation style;
2. tinymesh owns sparse mesh semantics and evidence;
3. PyG, PyG Temporal, TSL, and LibCity constrain event and forecast comparisons;
4. TorchGeo and TerraTorch bound optional geographic context and remain unused
   unless error evidence gives that context a causal role.

The additional study checkouts are shallow, durable, revision-bound references.
A stage that names one must cite the exact files that can change its design.
Retention does not authorize runtime imports or require routine pin updates;
gitlinks move only when an intentional study needs a different revision.

This produces direct classes with ordinary Tensor attributes and `__call__`,
not a framework inside a framework. `tinymesh.nn` contains only components
already used by current experiments. Sequence unrolling, task heads, trainers,
datasets, and evaluation policy stay outside those classes.

At the pinned tinygrad revision, `Linear` and `LSTMCell` are ordinary classes
with Tensor state and `__call__`; there is no module base class or `forward`
protocol. Tinymesh keeps that shape. Stateful graph compositions are direct
classes, while stateless math remains Tensor and `Graph` operations.

Gitlinks move only for an intentional executable study or when re-grounding an
active staged specification that names their source. A pin update records its
upstream delta and compatibility evidence; it never silently changes runtime
behavior. An experiment envelope records only the gitlinks declared by its
catalog entry.
