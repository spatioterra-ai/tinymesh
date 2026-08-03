"""Revision-bound evidence for tinymesh contracts and research decisions."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Experiment:
    group: str
    owner: str
    settings: tuple[str, ...] = ("DEV",)
    timeout_seconds: int = 600


METR_LA_SETTINGS = (
    "DEV",
    "EPOCHS",
    "STEPS",
    "MODEL",
    "HEAD",
    "LOSS",
    "TEST",
    "SEED",
    "HISTORY",
    "HORIZON",
    "BS",
    "HIDDEN",
    "LR",
    "CHECKPOINT_EVERY",
)


CATALOG = {
    "sparse_aggregation": Experiment("kernel", "tinymesh.Graph"),
    "csr_aggregation": Experiment(
        "kernel",
        "tinymesh.Graph",
        ("DEV", "DEGREE", "SAMPLES", "SIZES", "WARMUPS", "WIDTH"),
    ),
    "weighted_aggregation": Experiment("primitive", "tinymesh.Graph.sum"),
    "spatial_geometry": Experiment("primitive", "research-only"),
    "jepa_mechanics": Experiment(
        "representation",
        "research-only",
        ("DEV", "EMA", "HIDDEN", "LR", "SAMPLES", "SEED", "STEPS"),
    ),
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
    "metr_la_data": Experiment("data", "tinymesh.datasets"),
    "metr_la_forecast": Experiment(
        "forecast",
        "tinymesh.nn.A3TGCN",
        METR_LA_SETTINGS,
    ),
    "metr_la_diffusion": Experiment(
        "forecast",
        "tinymesh.nn.DiffusionGRU",
        METR_LA_SETTINGS,
        timeout_seconds=900,
    ),
    "metr_la_local_diffusion": Experiment(
        "forecast",
        "research-only",
        METR_LA_SETTINGS,
    ),
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
