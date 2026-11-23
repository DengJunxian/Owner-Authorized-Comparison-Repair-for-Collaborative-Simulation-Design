"""The legacy-schema view must not alter numerical observations."""

import copy

import pytest

from scripts.cscwd2027_mechanism_analysis_amendment import linked_view


def test_join_adds_only_session_and_preserves_original_object():
    old = {
        "artifact_analysis": {
            "original_rows": [{"case_id": "a", "value": 3}],
            "mixed_rows": [{"value": 4}],
        },
        "workflow": {"value": 5},
    }
    before = copy.deepcopy(old)
    new = linked_view(old, [{"id": "a", "session": "s"}])
    assert old == before
    assert new["artifact_analysis"]["original_rows"] == [
        {"case_id": "a", "value": 3, "session": "s"}
    ]
    assert (
        new["artifact_analysis"]["mixed_rows"] == old["artifact_analysis"]["mixed_rows"]
    )
    assert new["workflow"] == old["workflow"]


def test_unknown_or_conflicting_case_fails_closed():
    with pytest.raises(KeyError):
        linked_view(
            {"artifact_analysis": {"original_rows": [{"case_id": "unknown"}]}}, []
        )
    with pytest.raises(RuntimeError):
        linked_view(
            {
                "artifact_analysis": {
                    "original_rows": [{"case_id": "a", "session": "wrong"}]
                }
            },
            [{"id": "a", "session": "s"}],
        )
