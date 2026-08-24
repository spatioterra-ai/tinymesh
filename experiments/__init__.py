"""Revision-bound evidence for tinymesh contracts and research decisions."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Experiment:
    group: str
    owner: str
    settings: tuple[str, ...] = ("DEV",)
    timeout_seconds: int = 600
    papers: tuple[str, ...] = ()
    fidelity: str = "original"
    references: tuple[str, ...] = ("submodules/tinygrad",)


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
    "framework_benchmark": Experiment(
        "benchmark",
        "research-only",
        ("DEV", "DEGREE", "HIDDEN", "NODES", "SAMPLES", "WARMUPS", "WIDTH"),
        references=("submodules/tinygrad", "submodules/pytorch-geometric-temporal"),
    ),
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
        papers=("i-jepa", "graph-jepa"),
        fidelity="mechanism",
    ),
    "mean_sage": Experiment("layer", "tinymesh.nn.SAGEConv"),
    "gcn": Experiment("layer", "tinymesh.nn.GCNConv"),
    "gine": Experiment("layer", "tinymesh.nn.GINEConv", papers=("gine",), fidelity="mechanism"),
    "gat": Experiment("layer", "tinymesh.nn.GATConv"),
    "multi_head_gat": Experiment("layer", "tinymesh.nn.GATConv"),
    "tgcn": Experiment("layer", "tinymesh.nn.TGCN"),
    "gconv_gru": Experiment("layer", "tinymesh.nn.GConvGRU"),
    "directed_diffusion": Experiment("layer", "tinymesh.nn.DirectedDiffusion"),
    "chickenpox_data": Experiment("data", "tinymesh.datasets"),
    "montevideo_source": Experiment("data", "tinymesh.datasets", ()),
    "montevideo_data": Experiment("data", "tinymesh.datasets"),
    "metr_la_data": Experiment("data", "tinymesh.datasets"),
    "mutag_data": Experiment("data", "tinymesh.datasets"),
    "network_measurement": Experiment(
        "data",
        "research-only",
        (),
        references=(
            "submodules/pytorch-geometric",
            "submodules/pytorch-geometric-temporal",
            "submodules/torch-spatiotemporal",
        ),
    ),
    "gtfs_schedule": Experiment("data", "research-only", ()),
    "gtfs_realtime": Experiment("data", "research-only", ()),
    "gtfs_transition": Experiment("data", "research-only", ()),
    "gtfs_snapshot": Experiment("data", "research-only", ("DEV",)),
    "gtfs_replay": Experiment("data", "research-only", ()),
    "gtfs_event_mesh": Experiment(
        "data",
        "research-only",
        (),
        references=(
            "submodules/pytorch-geometric",
            "submodules/pytorch-geometric-temporal",
            "submodules/torch-spatiotemporal",
        ),
    ),
    "mbta_population": Experiment("data", "research-only", (), references=()),
    "mbta_clock": Experiment(
        "data",
        "research-only",
        (),
        references=("submodules/pytorch-geometric-temporal",),
    ),
    "mbta_headway_task": Experiment(
        "forecast",
        "research-only",
        ("TEST",),
        references=(
            "submodules/libcity",
            "submodules/pytorch-geometric-temporal",
            "submodules/torch-spatiotemporal",
        ),
    ),
    "mbta_topology": Experiment(
        "forecast",
        "research-only",
        ("TEST",),
        references=(
            "submodules/libcity",
            "submodules/pytorch-geometric-temporal",
            "submodules/tinygrad",
            "submodules/torch-spatiotemporal",
        ),
    ),
    "mbta_event_memory": Experiment(
        "forecast",
        "research-only",
        (),
        references=(
            "submodules/pytorch-geometric",
            "submodules/pytorch-geometric-temporal",
            "submodules/tinygrad",
        ),
    ),
    "mutag_jepa": Experiment(
        "representation",
        "research-only",
        ("DEV", "EMA", "FOLDS", "HIDDEN", "LR", "MASK_EVERY", "PROBE_LR", "PROBE_STEPS", "SEED", "STEPS"),
        papers=("graph-jepa",),
        fidelity="ablation",
    ),
    "mutag_graph_jepa": Experiment(
        "representation",
        "research-only",
        ("DEV", "EMA", "FOLDS", "HIDDEN", "LR", "PATCHES", "PROBE_LR", "PROBE_STEPS", "RW", "SEED", "STEPS", "TARGETS"),
        timeout_seconds=1800,
        papers=("graph-jepa", "gine"),
        fidelity="ablation",
    ),
    "mutag_graph_jepa_reproduction": Experiment(
        "representation",
        "research-only",
        timeout_seconds=86_400,
        papers=("graph-jepa",),
        fidelity="reproduction",
    ),
    "transport_jepa": Experiment(
        "representation",
        "research-only",
        ("DEV", "EMA", "HIDDEN", "HISTORY", "HORIZON", "LR", "PROBE_LR", "PROBE_STEPS", "SEED", "STEPS"),
        timeout_seconds=1800,
        papers=("v-jepa",),
        fidelity="ablation",
    ),
    "metr_la_jepa": Experiment(
        "representation",
        "research-only",
        ("DEV", "BS", "EMA", "EVAL_SAMPLES", "HIDDEN", "HISTORY", "HORIZON", "LR", "PROBE_LR", "PROBE_STEPS", "SAMPLES", "SEED", "STEPS", "TEST"),
        timeout_seconds=1800,
        papers=("ts-jepa",),
        fidelity="ablation",
    ),
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
