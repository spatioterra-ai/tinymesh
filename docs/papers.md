# Papers

[`papers/registry.toml`](https://github.com/spatioterra-ai/tinymesh/blob/main/papers/registry.toml)
is the source of truth for papers that directly guide current research or
implementation. Each entry owns one publication citation, one exact arXiv
revision, and the source license shown by arXiv.

```text
tracked registry
      |
      +--> publication citation
      +--> arXiv /abs/<exact revision>
      +--> arXiv /pdf/<exact revision>
      +--> arXiv /src/<exact revision>
                         |
                         v
              papers/_cache/  (ignored)
```

List the registry, print BibTeX, or fetch the rendered paper and TeX archive:

```console
uv run --locked python -m papers.run --list
uv run --locked python -m papers.run --cite i-jepa graph-jepa
uv run --locked python -m papers.run i-jepa graph-jepa
```

With no paper names, each command selects the whole registry. Fetches are
bounded, atomic, and stored under the paper key and exact arXiv revision. The
command prints a SHA-256 digest for every local file.

The current sources are
[I-JEPA v3](https://arxiv.org/abs/2301.08243v3), published at
[CVPR 2023](https://openaccess.thecvf.com/content/CVPR2023/html/Assran_Self-Supervised_Learning_From_Images_With_a_Joint-Embedding_Predictive_Architecture_CVPR_2023_paper.html),
and [Graph-JEPA v3](https://arxiv.org/abs/2309.16014v3), published in
[TMLR](https://openreview.net/forum?id=v47f4DwYZb), plus
[GINE v3](https://arxiv.org/abs/1905.12265v3), published at
[ICLR 2020](https://openreview.net/forum?id=HJlWWJSFDH), and
[V-JEPA v1](https://arxiv.org/abs/2404.08471v1), and
[TS-JEPA v1](https://arxiv.org/abs/2509.25449v1), presented at the
[NeurIPS 2024 time-series workshop](https://openreview.net/forum?id=FIdbozebmy).

## Source rule

Add a paper only when a live research question or implementation uses it. Pin
an explicit `vN`, copy citation metadata from the publication venue, and record
the license linked by that arXiv revision. Cite the publication in prose and
name the exact arXiv revision when an equation, algorithm, loss, or evaluation
protocol affects the work.

Read the PDF or TeX before making an implementation claim. Do not commit paper
copies or extracted source; the cache is reproducible and replaceable. A source
license may permit less than redistribution, which is another reason the
tracked boundary contains metadata rather than paper bodies.
