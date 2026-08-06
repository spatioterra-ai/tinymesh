"""Run one cataloged experiment and write its revision-bound observation."""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

from experiments import CATALOG, Experiment


ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "experiments" / "runs"


def main() -> None:
    if len(sys.argv) == 2 and sys.argv[1] == "--list":
        for name, experiment in CATALOG.items():
            papers = ",".join(experiment.papers) or "-"
            print(f"{name:28} {experiment.group:14} {experiment.fidelity:9} {papers:20} {experiment.owner}")
        return
    if len(sys.argv) < 2:
        raise SystemExit("usage: python -m experiments.run EXPERIMENT [KEY=VALUE ...]")

    name, *raw_settings = sys.argv[1:]
    if name not in CATALOG:
        raise SystemExit(f"unknown experiment {name!r}; use --list")
    revision, references = _revision()
    settings = _settings(CATALOG[name], raw_settings)
    started_at = datetime.now(timezone.utc)
    start = perf_counter()
    timeout = CATALOG[name].timeout_seconds
    result = _run(name, settings, timeout)
    elapsed = perf_counter() - start
    envelope = {
        "schema": 2,
        "experiment": name,
        "group": CATALOG[name].group,
        "owner": CATALOG[name].owner,
        "papers": CATALOG[name].papers,
        "fidelity": CATALOG[name].fidelity,
        "started_at": started_at.isoformat(),
        "elapsed_seconds": round(elapsed, 6),
        "timeout_seconds": timeout,
        "python": platform.python_version(),
        "revision": revision,
        "references": references,
        "settings": settings,
        "result": result,
    }
    print(_write(envelope, started_at, name, revision).relative_to(ROOT))


def _revision() -> tuple[str, dict[str, str]]:
    dirty = _git("status", "--porcelain", "--untracked-files=no", "--ignore-submodules=none")
    if dirty:
        raise SystemExit("refusing to run against a dirty tracked worktree")
    revision = _git("rev-parse", "HEAD")
    references = {}
    for line in _git("ls-tree", "-r", "HEAD").splitlines():
        metadata, path = line.split("\t", 1)
        mode, kind, commit = metadata.split()
        if (mode, kind) == ("160000", "commit"):
            references[path] = commit
    return revision, references


def _settings(experiment: Experiment, raw: list[str]) -> dict[str, str]:
    settings = {}
    for item in raw:
        key, separator, value = item.partition("=")
        if not separator or not key or not value:
            raise SystemExit(f"setting must be KEY=VALUE, got {item!r}")
        if key not in experiment.settings:
            allowed = ", ".join(experiment.settings) or "none"
            raise SystemExit(f"{key} is not a setting for this experiment; allowed: {allowed}")
        if key in settings:
            raise SystemExit(f"duplicate setting {key}")
        settings[key] = value
    return dict(sorted(settings.items()))


def _run(name: str, settings: dict[str, str], timeout: int) -> dict:
    env = os.environ.copy()
    for key in {key for experiment in CATALOG.values() for key in experiment.settings}:
        env.pop(key, None)
    env.update(settings)
    process = subprocess.run(
        [sys.executable, "-m", f"experiments.{name}"],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        timeout=timeout,
        check=True,
    )
    result = json.loads(process.stdout)
    if not isinstance(result, dict):
        raise RuntimeError(f"{name} must print one JSON object")
    return result


def _write(envelope: dict, started_at: datetime, name: str, revision: str) -> Path:
    RUNS.mkdir(parents=True, exist_ok=True)
    stamp = started_at.strftime("%Y%m%dT%H%M%S%fZ")
    path = RUNS / f"{stamp}-{name}-{revision[:10]}.json"
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(envelope, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)
    return path


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    ).stdout.strip()


if __name__ == "__main__":
    main()
