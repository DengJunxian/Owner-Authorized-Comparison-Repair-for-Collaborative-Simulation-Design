from pathlib import Path

from scripts.verify_original_evidence import verify

ROOT = Path(__file__).resolve().parents[1]


def test_registered_evidence_is_immutable_and_recomputable() -> None:
    result = verify(
        ROOT / "results" / "frozen" / "aaai27-paired-execution-v2",
        ROOT / "results" / "registered_evidence_lock.json",
    )
    assert result["status"] == "VERIFIED"
    assert result["market_row_count"] == 3072
