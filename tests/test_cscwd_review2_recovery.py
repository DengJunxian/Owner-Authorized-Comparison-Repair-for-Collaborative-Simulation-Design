"""Serialization repair must not alter scientific endpoints or observations."""

from scripts.cscwd2027_review2_recovery import normalize_summary


def test_only_histogram_key_type_is_repaired():
    value = {
        "operation_count_distribution": {0: 5, 1: 7},
        "regret": [1, 4],
        "records": [{"flipped": False}],
    }
    repaired = normalize_summary(value)
    assert repaired["operation_count_distribution"] == {"0": 5, "1": 7}
    assert repaired["regret"] == value["regret"]
    assert repaired["records"] == value["records"]
    assert list(value["operation_count_distribution"]) == [0, 1]
