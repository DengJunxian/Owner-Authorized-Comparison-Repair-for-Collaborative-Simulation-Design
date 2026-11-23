"""Recovery accepts serialization-equivalent arrays, never changed values."""

from policytwin.provenance import canonical_json_bytes
from scripts.cscwd2027_review2_canonical_recovery import serialized_equal


def test_canonical_comparison_is_exact(tmp_path):
    path = tmp_path / "record.json"
    path.write_bytes(canonical_json_bytes({"contract": [("c", 1)], "v": 4}))
    assert serialized_equal(path, {"contract": [["c", 1]], "v": 4})
    assert serialized_equal(path, {"contract": [("c", 1)], "v": 4})
    assert not serialized_equal(path, {"contract": [["c", 1]], "v": 5})
    assert not serialized_equal(path, {"contract": [["c", 2]], "v": 4})
