import io
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
from zipfile import ZIP_DEFLATED, ZipFile

from experiments.gtfs_schedule import SOURCE_MAX_BYTES, ScheduleError, load, parse


FIXTURE = Path(__file__).parent / "fixtures" / "gtfs_sample"


def archive(*, replace: tuple[str, str, str] | None = None, omit: str | None = None) -> bytes:
  output = io.BytesIO()
  with ZipFile(output, "w", ZIP_DEFLATED) as target:
    for path in sorted(FIXTURE.iterdir()):
      if path.name == omit:
        continue
      text = path.read_text()
      if replace is not None and path.name == replace[0]:
        if replace[1] not in text:
          raise AssertionError(f"missing fixture text {replace[1]!r}")
        text = text.replace(replace[1], replace[2], 1)
      target.writestr(path.name, text)
  return output.getvalue()


class GtfsScheduleTest(unittest.TestCase):
  def test_parses_canonical_source_facts_and_relations(self) -> None:
    schedule = parse(archive())

    self.assertEqual(
      (
        len(schedule.agencies),
        len(schedule.services),
        len(schedule.exceptions),
        len(schedule.stops),
        len(schedule.routes),
        len(schedule.trips),
        len(schedule.stop_times),
        len(schedule.frequencies),
        len(schedule.segments),
      ),
      (1, 2, 1, 9, 5, 11, 28, 11, 15),
    )
    self.assertEqual(schedule.stops[0].stop_id, "FUR_CREEK_RES")
    self.assertEqual(schedule.stop_times[0].source.line, 2)
    segment = next(item for item in schedule.segments if item.route_id == "CITY" and item.from_stop_id == "NANAA")
    self.assertEqual(segment.to_stop_id, "NADAV")
    self.assertEqual({item.trip_id for item in segment.occurrences}, {"CITY1"})
    routes = {trip.trip_id: trip.route_id for trip in schedule.trips}
    calls = dict(schedule.ordered_calls)
    expected = {
      (routes[trip_id], left.stop_id, right.stop_id, trip_id, left.stop_sequence, right.stop_sequence)
      for trip_id, trip_calls in calls.items()
      for left, right in zip(trip_calls, trip_calls[1:])
    }
    actual = {
      (segment.route_id, segment.from_stop_id, segment.to_stop_id, occurrence.trip_id, occurrence.from_sequence, occurrence.to_sequence)
      for segment in schedule.segments
      for occurrence in segment.occurrences
    }
    self.assertEqual(actual, expected)

  def test_derives_call_order_independently_of_source_rows(self) -> None:
    stop_times = (FIXTURE / "stop_times.txt").read_text().splitlines()
    shuffled_calls = "\n".join((stop_times[0], *reversed(stop_times[1:]))) + "\n"
    schedule = parse(archive(replace=("stop_times.txt", (FIXTURE / "stop_times.txt").read_text(), shuffled_calls)))

    self.assertEqual(schedule.stop_times[0].trip_id, "AAMV4")
    city = dict(schedule.ordered_calls)["CITY1"]
    self.assertEqual([call.stop_sequence for call in city], [1, 2, 3, 4, 5])

    trips = (FIXTURE / "trips.txt").read_text().splitlines()
    shuffled_trips = "\n".join((trips[0], *reversed(trips[1:]))) + "\n"
    schedule = parse(archive(replace=("trips.txt", (FIXTURE / "trips.txt").read_text(), shuffled_trips)))

    self.assertEqual(schedule.trips[0].trip_id, "AAMV4")
    self.assertEqual([trip_id for trip_id, _ in schedule.ordered_calls], sorted(trip.trip_id for trip in schedule.trips))

  def test_accepts_post_midnight_time_and_sequence_gap(self) -> None:
    data = archive(replace=("stop_times.txt", "STBA,6:20:00,6:20:00,BEATTY_AIRPORT,2", "STBA,25:20:00,25:20:00,BEATTY_AIRPORT,23"))
    schedule = parse(data)
    stop_time = dict(schedule.ordered_calls)["STBA"][-1]

    self.assertEqual(stop_time.arrival, 25 * 3600 + 20 * 60)
    self.assertEqual(stop_time.stop_sequence, 23)

  def test_rejects_identity_and_reference_failures(self) -> None:
    cases = (
      ("duplicate", "stop_times.txt", "STBA,6:20:00,6:20:00,BEATTY_AIRPORT,2", "STBA,6:20:00,6:20:00,BEATTY_AIRPORT,1"),
      ("duplicate", "stops.txt", "BEATTY_AIRPORT,", "FUR_CREEK_RES,"),
      ("unknown reference", "trips.txt", "AB,FULLW,AB1", "UNKNOWN,FULLW,AB1"),
      ("unknown reference", "stop_times.txt", "STBA,6:20:00", "UNKNOWN,6:20:00"),
      ("unknown reference", "stop_times.txt", "BEATTY_AIRPORT,2", "UNKNOWN,2"),
      ("unknown reference", "calendar_dates.txt", "FULLW,20070604", "UNKNOWN,20070604"),
    )
    for message, name, old, new in cases:
      with self.subTest(message=message, name=name), self.assertRaisesRegex(ScheduleError, message):
        parse(archive(replace=(name, old, new)))

  def test_rejects_invalid_values(self) -> None:
    cases = (
      ("service-day time", "stop_times.txt", "6:20:00", "-1:20:00"),
      ("service-day time", "stop_times.txt", "6:20:00", "6:60:00"),
      ("precedes arrival", "stop_times.txt", "6:05:00,6:07:00", "6:05:00,6:04:00"),
      (r"expected \[-90", "stops.txt", "36.425288", "91.0"),
      ("must follow", "frequencies.txt", "6:00:00,22:00:00", "6:00:00,6:00:00"),
      ("at least 1", "frequencies.txt", "1800", "0"),
      ("overlap", "frequencies.txt", "CITY1,8:00:00", "CITY1,7:00:00"),
    )
    for message, name, old, new in cases:
      with self.subTest(message=message), self.assertRaisesRegex(ScheduleError, message):
        parse(archive(replace=(name, old, new)))

  def test_rejects_non_platform_and_flexible_calls(self) -> None:
    stops = (FIXTURE / "stops.txt").read_text()
    non_platform = stops.replace("stop_url\n", "stop_url,location_type\n").replace(
      "-117.133162,,\n", "-117.133162,,,1\n", 1
    )
    with self.assertRaisesRegex(ScheduleError, "stops.txt:2:location_type"):
      parse(archive(replace=("stops.txt", stops, non_platform)))

    stop_times = (FIXTURE / "stop_times.txt").read_text()
    flexible = stop_times.replace("shape_dist_traveled\n", "shape_dist_traveled,location_group_id\n").replace(
      "STBA,6:00:00,6:00:00,STAGECOACH,1,,,,\n",
      "STBA,6:00:00,6:00:00,,1,,,,,ZONE\n",
    )
    with self.assertRaisesRegex(ScheduleError, "flexible locations"):
      parse(archive(replace=("stop_times.txt", stop_times, flexible)))

  def test_rejects_missing_terminal_time(self) -> None:
    with self.assertRaisesRegex(ScheduleError, "first and last"):
      parse(archive(replace=("stop_times.txt", "STBA,6:00:00,6:00:00", "STBA,,")))

  def test_rejects_archive_and_acquisition_failures(self) -> None:
    with self.assertRaisesRegex(ScheduleError, "ZIP"):
      parse(b"not a zip")
    with self.assertRaisesRegex(ScheduleError, "missing agency.txt"):
      parse(archive(omit="agency.txt"))
    with self.assertRaisesRegex(ScheduleError, "exceeds"):
      parse(b"x" * (SOURCE_MAX_BYTES + 1))

    malformed = archive(replace=("agency.txt", "America/Los_Angeles", "America/Los_Angeles,extra"))
    with self.assertRaisesRegex(ScheduleError, "malformed CSV"):
      parse(malformed)

  def test_local_override_requires_pinned_checksum(self) -> None:
    with TemporaryDirectory() as temporary:
      path = Path(temporary) / "fixture.zip"
      path.write_bytes(archive())
      with self.assertRaisesRegex(ScheduleError, "sha256 expected"):
        load(path)

  def test_remote_failure_retains_context(self) -> None:
    with patch("experiments.gtfs_schedule.urlopen", side_effect=TimeoutError("timed out")):
      with self.assertRaisesRegex(ScheduleError, "acquisition failed.*timed out"):
        load()


if __name__ == "__main__":
  unittest.main()
