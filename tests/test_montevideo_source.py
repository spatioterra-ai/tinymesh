import copy
import hashlib
import json
import unittest
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from tinymesh.datasets import (
    _MONTEVIDEO_MAX_BYTES,
    _MONTEVIDEO_TIMEOUT,
    _MONTEVIDEO_URL,
    _parse_montevideo,
    _read_montevideo,
)

SOURCE = {
    "directed": True,
    "multigraph": False,
    "nodes": [
        {
            "bus_stop": 20,
            "lon": 1,
            "lat": 2,
            "X": {"y": [1, 2, 3]},
            "y": [2, 3, 4],
        },
        {
            "bus_stop": 10,
            "lon": 3.5,
            "lat": 4.5,
            "X": {"y": [10, 20, 30]},
            "y": [20, 30, 40],
        },
    ],
    "links": [
        {"source": 10, "target": 20, "weight": 7.5},
        {"source": 20, "target": 10, "weight": 8},
    ],
}


class MontevideoSourceTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = TemporaryDirectory()
        self.path = Path(self.temporary.name) / "montevideo.json"
        self.path.write_text(json.dumps(SOURCE))

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_preserves_source_order_and_values(self) -> None:
        source = _read_montevideo(self.path)

        self.assertEqual(source.node_ids, (20, 10))
        self.assertEqual(source.source, (1, 0))
        self.assertEqual(source.target, (0, 1))
        self.assertEqual(source.position, ((1.0, 2.0), (3.5, 4.5)))
        self.assertEqual(source.road_distance, (7.5, 8.0))
        self.assertEqual(source.features, ((1.0, 2.0, 3.0), (10.0, 20.0, 30.0)))
        self.assertEqual(source.targets, ((2.0, 3.0, 4.0), (20.0, 30.0, 40.0)))

    def test_default_source_is_revision_bound(self) -> None:
        payload = self.path.read_bytes()
        checksum = hashlib.sha256(payload).hexdigest()
        with (
            patch("tinymesh.datasets._MONTEVIDEO_SHA256", checksum),
            patch("tinymesh.datasets.urlopen", return_value=BytesIO(payload)) as urlopen,
        ):
            _read_montevideo()

        request = urlopen.call_args.args[0]
        self.assertEqual(request.full_url, _MONTEVIDEO_URL)
        self.assertEqual(urlopen.call_args.kwargs, {"timeout": _MONTEVIDEO_TIMEOUT})

    def test_rejects_default_source_with_wrong_checksum(self) -> None:
        with (
            patch("tinymesh.datasets._MONTEVIDEO_SHA256", "0" * 64),
            patch("tinymesh.datasets.urlopen", return_value=BytesIO(self.path.read_bytes())),
            self.assertRaisesRegex(RuntimeError, "checksum"),
        ):
            _read_montevideo()

    def test_rejects_oversized_source_before_json_parsing(self) -> None:
        self.path.write_bytes(b"{" + b" " * _MONTEVIDEO_MAX_BYTES)

        with (
            patch("tinymesh.datasets.json.loads") as loads,
            self.assertRaisesRegex(ValueError, "exceeds"),
        ):
            _read_montevideo(self.path)
        loads.assert_not_called()

    def test_validates_graph_envelope(self) -> None:
        for value, message in (
            ([], "JSON object"),
            (dict(SOURCE, directed=False), "directed"),
            (dict(SOURCE, multigraph=True), "multigraph"),
            (dict(SOURCE, nodes=[]), "nodes"),
            (dict(SOURCE, links=[]), "links"),
        ):
            with self.subTest(message=message), self.assertRaisesRegex((TypeError, ValueError), message):
                _parse_montevideo(value)

    def test_validates_node_identity_geometry_and_observations(self) -> None:
        malformed: list[tuple[dict[str, object], str]] = []

        duplicate = copy.deepcopy(SOURCE)
        duplicate["nodes"][1]["bus_stop"] = 20
        malformed.append((duplicate, "unique"))

        invalid_position = copy.deepcopy(SOURCE)
        invalid_position["nodes"][0]["lon"] = float("inf")
        malformed.append((invalid_position, "finite"))

        invalid_feature = copy.deepcopy(SOURCE)
        invalid_feature["nodes"][0]["X"]["y"][0] = True
        malformed.append((invalid_feature, "numeric"))

        unequal_feature_target = copy.deepcopy(SOURCE)
        unequal_feature_target["nodes"][0]["y"].pop()
        malformed.append((unequal_feature_target, "lengths differ"))

        unequal_nodes = copy.deepcopy(SOURCE)
        unequal_nodes["nodes"][1]["y"].pop()
        unequal_nodes["nodes"][1]["X"]["y"].pop()
        malformed.append((unequal_nodes, "observations"))

        for value, message in malformed:
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                _parse_montevideo(value)

    def test_validates_link_identity_and_distance(self) -> None:
        malformed: list[tuple[dict[str, object], str]] = []

        missing_endpoint = copy.deepcopy(SOURCE)
        missing_endpoint["links"][0]["target"] = 99
        malformed.append((missing_endpoint, "resolve"))

        invalid_endpoint = copy.deepcopy(SOURCE)
        invalid_endpoint["links"][0]["source"] = True
        malformed.append((invalid_endpoint, "integers"))

        duplicate = copy.deepcopy(SOURCE)
        duplicate["links"].append(copy.deepcopy(duplicate["links"][0]))
        malformed.append((duplicate, "duplicates"))

        invalid_distance = copy.deepcopy(SOURCE)
        invalid_distance["links"][0]["weight"] = 0
        malformed.append((invalid_distance, "positive"))

        for value, message in malformed:
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                _parse_montevideo(value)


if __name__ == "__main__":
    unittest.main()
