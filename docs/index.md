# tinymesh documentation

tinymesh documentation separates concepts from evidence:

- **Concepts** explain graph and learning semantics that should survive an
  implementation change.
- **Research records** bind decisions and measurements to exact code and
  dependency revisions.
- **Source and tests** own current behavior.

The package does not expose a stable graph API yet.

## Start here

1. [Sparse graph topology](concepts/topology.md) explains how a directed edge
   list becomes the CSR representation used by the current kernel.
2. [Message passing](concepts/message-passing.md) explains how models compose
   messages, aggregation, and node updates over that topology.
3. [Sparse aggregation feasibility](research/sparse-aggregation.md) records why
   the native gather-and-scatter candidate was rejected and what the custom CSR
   path proves.
4. [Mean GraphSAGE experiment](research/mean-sage.md) records the first
   trainable caller and what its exact witness does not prove.

```text
theory                    revision-bound proof

topology ----------------> sparse aggregation
   |                             |
   v                             v
message passing ---------> mean GraphSAGE
```

## Current boundary

Implemented:

- fixed directed topology;
- deterministic COO-to-CSR lowering;
- destination-CSR sum and transpose-CSR first-order backward;
- fixed-topology device-buffer reuse;
- one trainable mean-GraphSAGE composition;
- CPU and Metal verification.

Not implemented:

- a public `tinymesh` graph API;
- weighted or edge-dependent messages;
- segment softmax, batching, sampling, or changing topology;
- higher-order gradients;
- coordinates, higher-dimensional cells, temporal fields, or recurrence.

This boundary is intentional. A second model caller must demonstrate the public
contract before experimental code moves into `src/tinymesh`.
