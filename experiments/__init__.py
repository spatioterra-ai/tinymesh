"""Revision-bound evidence for tinymesh contracts and research decisions."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Experiment:
    group: str
    owner: str
    settings: tuple[str, ...] = ("DEV",)


CATALOG = {
    "sparse_aggregation": Experiment("kernel", "tinymesh.Graph"),
    "csr_aggregation": Experiment(
        "kernel",
        "tinymesh.Graph",
        ("DEV", "DEGREE", "SAMPLES", "SIZES", "WARMUPS", "WIDTH"),
    ),
    "weighted_aggregation": Experiment("primitive", "tinymesh.Graph.sum"),
    "spatial_geometry": Experiment("primitive", "research-only"),
    "mean_sage": Experiment("layer", "tinymesh.nn.SAGEConv"),
    "gcn": Experiment("layer", "tinymesh.nn.GCNConv"),
    "gat": Experiment("layer", "tinymesh.nn.GATConv"),
    "multi_head_gat": Experiment("layer", "tinymesh.nn.GATConv"),
    "tgcn": Experiment("layer", "tinymesh.nn.TGCN"),
    "gconv_gru": Experiment("layer", "tinymesh.nn.GConvGRU"),
    "directed_diffusion": Experiment("layer", "tinymesh.nn.DirectedDiffusion"),
    "chickenpox_data": Experiment("data", "tinymesh.datasets"),
    "montevideo_source": Experiment("data", "tinymesh.datasets", ()),
    "montevideo_data": Experiment("data", "tinymesh.datasets"),
    "chickenpox_forecast": Experiment(
        "forecast",
        "research-only",
        ("DEV", "BS", "EPOCHS", "HIDDEN", "HISTORY", "LR", "SEED"),
    ),
    "montevideo_forecast": Experiment(
        "forecast",
        "research-only",
        (
            "DEV",
            "BS",
            "CHECKPOINT_EVERY",
            "EPOCHS",
            "HIDDEN",
            "HISTORY",
            "LR",
            "MODEL",
            "NODES",
            "SEED",
        ),
    ),
    "montevideo_seasonal": Experiment("forecast", "research-only"),
    "montevideo_delayed_edges": Experiment("forecast", "research-only"),
    "transport_forecast": Experiment(
        "forecast",
        "research-only",
        (
            "DEV",
            "BS",
            "EPOCHS",
            "HIDDEN",
            "HISTORY",
            "HORIZON",
            "LR",
            "MODEL",
            "SEED",
            "TOPOLOGY",
        ),
    ),
    "transport_transfer": Experiment(
        "forecast",
        "research-only",
        (
            "DEV",
            "BS",
            "EPOCHS",
            "HIDDEN",
            "HISTORY",
            "HORIZON",
            "INITIAL",
            "LR",
            "MODEL",
            "NODES",
            "SEED",
        ),
    ),
}
