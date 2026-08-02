"""List, cite, or fetch exact paper revisions."""

from __future__ import annotations

import argparse
import hashlib
import tarfile
import tomllib
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "papers" / "registry.toml"
CACHE = ROOT / "papers" / "_cache"
MAX_BYTES = 64 << 20


def main() -> None:
  parser = argparse.ArgumentParser()
  action = parser.add_mutually_exclusive_group()
  action.add_argument("--list", action="store_true")
  action.add_argument("--cite", action="store_true")
  parser.add_argument("paper", nargs="*")
  args = parser.parse_args()

  catalog = _catalog()
  selected = _select(catalog, args.paper)
  if args.list:
    for name in selected:
      print(f"{name:12} {catalog[name]['arxiv']:16} {catalog[name]['citation']['title']}")
  elif args.cite:
    print("\n\n".join(_bibtex(catalog[name]) for name in selected))
  else:
    for name in selected:
      _fetch(name, catalog[name])


def _catalog() -> dict[str, dict]:
  return tomllib.loads(REGISTRY.read_text())["paper"]


def _select(catalog: dict[str, dict], names: list[str]) -> list[str]:
  selected = names or sorted(catalog)
  unknown = [name for name in selected if name not in catalog]
  if unknown:
    raise SystemExit(f"unknown paper {unknown[0]!r}; use --list")
  return selected


def _bibtex(paper: dict) -> str:
  citation = paper["citation"]
  fields = [(name, value) for name, value in citation.items() if name not in {"type", "key"}]
  width = max(len(name) for name, _ in fields)
  body = "\n".join(f"  {name:<{width}} = {{{value}}}," for name, value in fields)
  return f"@{citation['type']}{{{citation['key']},\n{body}\n}}"


def _fetch(name: str, paper: dict) -> None:
  arxiv = paper["arxiv"]
  directory = CACHE / name / arxiv
  for filename, url in {
    "paper.pdf": f"https://arxiv.org/pdf/{arxiv}",
    "source.tar": f"https://arxiv.org/src/{arxiv}",
  }.items():
    path = directory / filename
    digest = _download(url, path)
    print(f"{path.relative_to(ROOT)}  sha256:{digest}")


def _download(url: str, path: Path) -> str:
  if path.is_file():
    _validate(path, path.name)
    return _sha256(path)

  path.parent.mkdir(parents=True, exist_ok=True)
  temporary = path.with_suffix(path.suffix + ".tmp")
  request = urllib.request.Request(url, headers={"User-Agent": "tinymesh papers"})
  try:
    with urllib.request.urlopen(request, timeout=30) as response, temporary.open("wb") as output:
      length = int(response.headers.get("content-length", 0))
      if length > MAX_BYTES:
        raise RuntimeError(f"refusing {length}-byte download; limit is {MAX_BYTES}")
      size = 0
      while chunk := response.read(1 << 20):
        size += len(chunk)
        if size > MAX_BYTES:
          raise RuntimeError(f"download exceeded {MAX_BYTES}-byte limit")
        output.write(chunk)
    _validate(temporary, path.name)
    temporary.replace(path)
  finally:
    temporary.unlink(missing_ok=True)
  return _sha256(path)


def _validate(path: Path, name: str) -> None:
  with path.open("rb") as source:
    prefix = source.read(5)
  if name == "paper.pdf" and prefix != b"%PDF-":
    raise RuntimeError(f"{path} is not a PDF")
  if name == "source.tar" and not tarfile.is_tarfile(path):
    raise RuntimeError(f"{path} is not a source archive")


def _sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as source:
    while chunk := source.read(1 << 20):
      digest.update(chunk)
  return digest.hexdigest()


if __name__ == "__main__":
  main()
