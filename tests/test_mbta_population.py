import hashlib
import json
import shutil
import tempfile
import unittest
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from io import BytesIO
from pathlib import Path
from unittest.mock import patch

from experiments.mbta_population import FIXTURE, PopulationError as AuditError, observe
from experiments.tools.mbta_population import (
  CAP_BYTES,
  END_DATE,
  INDEX_FIELDS,
  INDEX_URL,
  PERFORMANCE_BASE,
  START_DATE,
  Plan,
  PopulationError,
  Source,
  acquire,
  plan,
  read,
  seal,
  _write,
)


OBSERVED_AT = datetime(2026, 8, 22, 20, tzinfo=timezone.utc)


class MbtaPopulationTest(unittest.TestCase):
  def test_exact_plan(self) -> None:
    index = self._index()
    result = plan(index, OBSERVED_AT)

    self.assertEqual((result.start_date, result.end_date), ("2026-07-24", "2026-08-20"))
    self.assertEqual(len(result.sources), 34)
    self.assertEqual(sum(source.name == "performance" for source in result.sources), 28)
    self.assertEqual(result.source_bytes, 76_434_973 + 28)
    self.assertEqual(result.index_bytes, len(index))
    self.assertEqual(result.index_sha256, hashlib.sha256(index).hexdigest())
    self.assertLess(result.source_bytes, result.cap_bytes)
    self.assertEqual(result.availability, "retrospective_event_time_only:no_generation_or_ingestion_clock")

  def test_plan_is_deterministic_and_round_trips(self) -> None:
    value = plan(self._index(), OBSERVED_AT)
    with tempfile.TemporaryDirectory() as directory:
      path = Path(directory) / "plan.json"
      _write(path, value)

      self.assertEqual(read(path), value)
      self.assertEqual(path.read_text(), path.read_text())

  def test_missing_duplicate_future_and_cap_fail_before_acquisition(self) -> None:
    rows = self._rows()
    with self.assertRaisesRegex(PopulationError, "missing service dates"):
      plan(self._index(rows[:-1]), OBSERVED_AT)
    with self.assertRaisesRegex(PopulationError, "duplicate service_date"):
      plan(self._index(rows + [rows[0]]), OBSERVED_AT)

    future_observation = datetime(2026, 8, 20, 12, tzinfo=timezone.utc)
    published_rows = [row.replace("2026-08-21 00:00:00+00:00", "2026-08-19 00:00:00+00:00") for row in rows]
    with self.assertRaisesRegex(PopulationError, "incomplete service date"):
      plan(self._index(published_rows), future_observation)

    oversized = rows.copy()
    oversized[0] = oversized[0].replace("1,", f"{CAP_BYTES},", 1)
    with self.assertRaisesRegex(PopulationError, "exceeds cap"):
      plan(self._index(oversized), OBSERVED_AT)

  def test_wrong_url_and_publication_time_fail_with_source_date(self) -> None:
    rows = self._rows()
    rows[0] = rows[0].replace(PERFORMANCE_BASE, "https://example.invalid")
    with self.assertRaisesRegex(PopulationError, "invalid source identity 2026-07-24"):
      plan(self._index(rows), OBSERVED_AT)

    rows = self._rows()
    rows[0] = rows[0].replace("2026-08-21 00:00:00+00:00", "2026-08-23 00:00:00+00:00")
    with self.assertRaisesRegex(PopulationError, "impossible publication time 2026-07-24"):
      plan(self._index(rows), OBSERVED_AT)

  def test_acquisition_publishes_only_after_every_exact_source(self) -> None:
    value = self._small_plan()
    with tempfile.TemporaryDirectory() as directory:
      target = Path(directory) / "source"
      with patch(
        "experiments.tools.mbta_population.urlopen",
        side_effect=[BytesIO(bytes([index])) for index in range(28)],
      ):
        sealed = acquire(value, target)

      self.assertTrue((target / "manifest.json").is_file())
      self.assertEqual(read(target / "manifest.json"), sealed)
      self.assertTrue(all(source.sha256 is not None for source in sealed.sources))
      self.assertEqual(len(tuple(target.glob("*.parquet"))), 28)

  def test_acquisition_failure_leaves_no_target_or_staging_directory(self) -> None:
    value = self._small_plan()
    with tempfile.TemporaryDirectory() as directory:
      target = Path(directory) / "source"
      responses = [BytesIO(b"x"), BytesIO(b"")]
      with patch("experiments.tools.mbta_population.urlopen", side_effect=responses):
        with self.assertRaisesRegex(PopulationError, "expected 1 bytes, got 0"):
          acquire(value, target)

      self.assertFalse(target.exists())
      self.assertEqual(tuple(Path(directory).iterdir()), ())

  def test_checksum_drift_leaves_no_acquisition(self) -> None:
    value = self._small_plan()
    first = replace(value.sources[0], sha256="0" * 64)
    value = replace(value, sources=(first, *value.sources[1:]))
    with tempfile.TemporaryDirectory() as directory:
      target = Path(directory) / "source"
      with patch("experiments.tools.mbta_population.urlopen", return_value=BytesIO(b"x")):
        with self.assertRaisesRegex(PopulationError, "checksum drift"):
          acquire(value, target)

      self.assertFalse(target.exists())

  def test_seal_reuses_only_exact_planned_bytes(self) -> None:
    value = self._small_plan()
    with tempfile.TemporaryDirectory() as directory:
      source = Path(directory)
      for item in value.sources:
        (source / item.filename).write_bytes(b"x")

      sealed = seal(value, source)

      self.assertEqual(read(source / "manifest.json"), sealed)
      self.assertTrue(all(item.sha256 == hashlib.sha256(b"x").hexdigest() for item in sealed.sources))

  def test_exact_retained_population_decision(self) -> None:
    result = observe()

    self.assertEqual((result.source_files, result.source_bytes, result.cap_bytes), (34, 110_610_188, CAP_BYTES))
    self.assertEqual((result.dates, result.routes, result.source_rows, result.trip_instances), (28, 8, 1_050_259, 98_767))
    self.assertEqual((result.missing_movement, result.source_headways), (55_164, 941_984))
    self.assertEqual(
      (
        result.schedule_resolved,
        result.schedule_unresolved,
        result.unresolved_added,
        result.unresolved_nonrevenue,
        result.unresolved_other,
      ),
      (821_513, 228_746, 221_220, 4_363, 3_163),
    )
    self.assertEqual(result.schedule_versions, 12)
    self.assertEqual(result.decision, "stop:insufficient_schedule_identity")
    self.assertEqual(result, observe())

  def test_retained_manifest_and_audit_drift_fail_closed(self) -> None:
    with tempfile.TemporaryDirectory() as directory:
      fixture = Path(directory) / "fixture"
      shutil.copytree(FIXTURE, fixture)
      manifest_path = fixture / "manifest.json"
      manifest = json.loads(manifest_path.read_text())
      manifest["sources"][0]["bytes"] += 1
      manifest_path.write_text(json.dumps(manifest))
      with self.assertRaisesRegex(AuditError, "source byte drift"):
        observe(fixture)

    with tempfile.TemporaryDirectory() as directory:
      fixture = Path(directory) / "fixture"
      shutil.copytree(FIXTURE, fixture)
      audit_path = fixture / "audit.json"
      audit = json.loads(audit_path.read_text())
      audit["decision"] = "advance:stage_3"
      audit_path.write_text(json.dumps(audit))
      with self.assertRaisesRegex(AuditError, "decision does not follow evidence"):
        observe(fixture)

  @staticmethod
  def _rows() -> list[str]:
    rows = []
    day = START_DATE
    while day <= END_DATE:
      filename = f"{day}-subway-on-time-performance-v1.parquet"
      rows.append(f"1,2026-08-21 00:00:00+00:00,{day},{PERFORMANCE_BASE}/{filename}")
      day += timedelta(days=1)
    return rows

  @classmethod
  def _index(cls, rows: list[str] | None = None) -> bytes:
    lines = [",".join(INDEX_FIELDS), *(rows if rows is not None else cls._rows())]
    return ("\n".join(lines) + "\n").encode()

  @staticmethod
  def _small_plan() -> Plan:
    sources = tuple(
      Source(
        "performance",
        f"{START_DATE + timedelta(days=index)}.parquet",
        f"https://example.test/{index}",
        1,
        OBSERVED_AT.isoformat(),
        (START_DATE + timedelta(days=index)).isoformat(),
        None,
      )
      for index in range(28)
    )
    return Plan(
      1,
      OBSERVED_AT.isoformat(),
      START_DATE.isoformat(),
      END_DATE.isoformat(),
      CAP_BYTES,
      INDEX_URL,
      1,
      "0" * 64,
      "retrospective_event_time_only:no_generation_or_ingestion_clock",
      sources,
    )


if __name__ == "__main__":
  unittest.main()
