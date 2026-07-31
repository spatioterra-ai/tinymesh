"""Run sequential directed diffusion on METR-LA."""

import json
from dataclasses import asdict

from experiments.metr_la_forecast import run


if __name__ == "__main__":
    print(json.dumps(asdict(run("diffusion_gru")), indent=2))
