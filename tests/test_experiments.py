import json
import shlex
import subprocess
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

from experiments import CATALOG
from experiments.run import ROOT, _revision, _run, _settings, _write


class CatalogTest(unittest.TestCase):
    def test_catalog_owns_every_runnable_experiment(self) -> None:
        directory = ROOT / "experiments"
        runnable = {
            path.stem
            for path in directory.glob("*.py")
            if path.stem != "run" and 'if __name__ == "__main__"' in path.read_text()
        }

        self.assertEqual(set(CATALOG), runnable)
        self.assertTrue(all(
            experiment.owner == "research-only" or experiment.owner.startswith("tinymesh.")
            for experiment in CATALOG.values()
        ))

    def test_settings_are_explicit_and_unique(self) -> None:
        experiment = CATALOG["chickenpox_forecast"]

        self.assertEqual(
            _settings(experiment, ["SEED=0", "DEV=CPU"]),
            {"DEV": "CPU", "SEED": "0"},
        )
        with self.assertRaisesRegex(SystemExit, "duplicate"):
            _settings(experiment, ["SEED=0", "SEED=1"])
        with self.assertRaisesRegex(SystemExit, "not a setting"):
            _settings(experiment, ["TOKEN=secret"])

    def test_experiment_timeouts_are_positive_and_revisioned(self) -> None:
        self.assertTrue(all(experiment.timeout_seconds > 0 for experiment in CATALOG.values()))
        self.assertEqual(CATALOG["mean_sage"].timeout_seconds, 600)
        self.assertEqual(CATALOG["metr_la_diffusion"].timeout_seconds, 900)
        self.assertEqual(CATALOG["metr_la_local_diffusion"].timeout_seconds, 600)

    def test_documented_runs_use_the_locked_runner(self) -> None:
        paths = [
            ROOT / "README.md",
            ROOT / "CONTRIBUTING.md",
            ROOT / "experiments" / "README.md",
            *(ROOT / "docs").rglob("*.md"),
        ]
        documented = set()
        for path in paths:
            for number, line in enumerate(path.read_text().splitlines(), start=1):
                with self.subTest(path=path.relative_to(ROOT), line=number):
                    if "uv run" in line:
                        self.assertIn("uv run --locked", line)
                    if "python -m experiments." in line or "python experiments/" in line:
                        self.assertIn("python -m experiments.run", line)
                    if "python -m experiments.run" not in line:
                        continue
                    command = shlex.split(line)
                    runner = ["python", "-m", "experiments.run"]
                    offset = next(
                        (index for index in range(len(command) - 2) if command[index:index + 3] == runner),
                        None,
                    )
                    if offset is not None and "--list" not in line and "<experiment>" not in line:
                        name, *settings = command[offset + 3:]
                        self.assertIn(name, CATALOG)
                        _settings(CATALOG[name], settings)
                        documented.add(name)
        self.assertEqual(documented, set(CATALOG))


class RunnerTest(unittest.TestCase):
    def test_revision_covers_every_reference_gitlink(self) -> None:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=True,
        ).stdout.strip()
        tree = subprocess.run(
            ["git", "ls-tree", "-r", "HEAD"],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=True,
        ).stdout.strip()
        with patch("experiments.run._git", side_effect=["", revision, tree]):
            actual_revision, references = _revision()

        self.assertEqual(actual_revision, revision)
        self.assertEqual(
            set(references),
            {
                "submodules/pytorch-geometric",
                "submodules/pytorch-geometric-temporal",
                "submodules/terratorch",
                "submodules/tinygrad",
                "submodules/torchgeo",
            },
        )
        self.assertTrue(all(len(commit) == 40 for commit in references.values()))
        reference_doc = (ROOT / "docs" / "reference-projects.md").read_text()
        self.assertTrue(all(commit in reference_doc for commit in references.values()))
        tinygrad = references["submodules/tinygrad"]
        self.assertIn(f"tinygrad.git@{tinygrad}", (ROOT / "pyproject.toml").read_text())
        self.assertIn(tinygrad, (ROOT / "uv.lock").read_text())

    def test_dirty_revision_is_rejected(self) -> None:
        with patch("experiments.run._git", return_value=" M src/tinymesh/graph.py"):
            with self.assertRaisesRegex(SystemExit, "dirty"):
                _revision()

    def test_run_clears_inherited_settings_and_parses_one_object(self) -> None:
        completed = subprocess.CompletedProcess([], 0, stdout='{"device": "CPU"}\n', stderr="")
        with patch.dict("os.environ", {"DEV": "METAL", "EPOCHS": "99"}):
            with patch("experiments.run.subprocess.run", return_value=completed) as run:
                result = _run("mean_sage", {"DEV": "CPU"}, timeout=10)

        self.assertEqual(result, {"device": "CPU"})
        environment = run.call_args.kwargs["env"]
        self.assertEqual(environment["DEV"], "CPU")
        self.assertNotIn("EPOCHS", environment)
        self.assertEqual(run.call_args.kwargs["timeout"], 10)

    def test_write_is_atomic_json(self) -> None:
        started = datetime(2026, 7, 29, tzinfo=timezone.utc)
        envelope = {"schema": 1, "result": {"value": 2}}
        with tempfile.TemporaryDirectory() as directory:
            with patch("experiments.run.RUNS", Path(directory)):
                path = _write(envelope, started, "witness", "a" * 40)

            self.assertEqual(json.loads(path.read_text()), envelope)
            self.assertEqual(list(Path(directory).glob("*.tmp")), [])


if __name__ == "__main__":
    unittest.main()
