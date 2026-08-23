import re
import tomllib
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESEARCH = ROOT / "docs" / "research"


class DocsTest(unittest.TestCase):
  def test_every_research_record_is_discoverable(self) -> None:
    pages = {
      f"research/{path.name}"
      for path in RESEARCH.glob("*.md")
      if path.name != "index.md"
    }

    nav = tomllib.loads((ROOT / "zensical.toml").read_text())["project"]["nav"]
    research_nav = next(item["Research"] for item in nav if isinstance(item, dict) and "Research" in item)
    self.assertEqual(pages, set(research_nav) - {"research/index.md"})

    ledger = (RESEARCH / "index.md").read_text()
    links = {
      f"research/{target}"
      for target in re.findall(r"\]\(([^/)]+\.md)(?:#[^)]+)?\)", ledger)
    }
    self.assertEqual(pages, links)


if __name__ == "__main__":
  unittest.main()
