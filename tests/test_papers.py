import re
import subprocess
import sys
import tempfile
import tomllib
import unittest
from pathlib import Path

from papers.run import _validate


ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "papers" / "registry.toml"


class PapersTest(unittest.TestCase):
    def test_registry_pins_sources_and_citations(self) -> None:
        papers = tomllib.loads(REGISTRY.read_text())["paper"]

        self.assertEqual(set(papers), {"i-jepa", "graph-jepa", "gine", "v-jepa"})
        for name, paper in papers.items():
            with self.subTest(paper=name):
                self.assertRegex(paper["arxiv"], r"^\d{4}\.\d{5}v\d+$")
                self.assertTrue(paper["license"].startswith("https://"))
                self.assertIn(paper["citation"]["type"], {"article", "inproceedings"})
                self.assertTrue(paper["citation"]["key"])
                self.assertTrue(paper["citation"]["title"])
                self.assertTrue(paper["citation"]["author"])
                self.assertTrue(paper["citation"]["url"].startswith("https://"))

    def test_cli_lists_and_cites_every_paper(self) -> None:
        listing = self._run("--list")
        citations = self._run("--cite")

        self.assertIn("i-jepa       2301.08243v3", listing)
        self.assertIn("graph-jepa   2309.16014v3", listing)
        self.assertIn("gine         1905.12265v3", listing)
        self.assertIn("v-jepa       2404.08471v1", listing)
        self.assertEqual(len(re.findall(r"^@", citations, re.MULTILINE)), 4)
        self.assertIn("@inproceedings{Assran_2023_CVPR,", citations)
        self.assertIn("@article{skenderi2025graph,", citations)
        self.assertIn("@inproceedings{Hu_2020_ICLR,", citations)
        self.assertIn("@article{Bardes_2024_VJEPA,", citations)

    def test_cli_rejects_unknown_paper(self) -> None:
        process = subprocess.run(
            [sys.executable, "-m", "papers.run", "missing"],
            cwd=ROOT,
            text=True,
            capture_output=True,
        )

        self.assertNotEqual(process.returncode, 0)
        self.assertIn("unknown paper 'missing'; use --list", process.stderr)

    def test_download_validation_rejects_error_pages(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "paper.pdf"
            path.write_text("<html>try again</html>")

            with self.assertRaisesRegex(RuntimeError, "is not a PDF"):
                _validate(path, "paper.pdf")

    @staticmethod
    def _run(*args: str) -> str:
        return subprocess.run(
            [sys.executable, "-m", "papers.run", *args],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=True,
        ).stdout


if __name__ == "__main__":
    unittest.main()
