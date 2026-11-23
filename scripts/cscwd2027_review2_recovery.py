"""Transparent, serialization-only recovery; every original output stays intact."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from unittest.mock import patch

from scripts import cscwd2027_review2 as parent
from scripts.cscwd2027_design_contract.store import save

ROOT = parent.ROOT
SEAL = ROOT / "configs/cscwd2027/review_consequences_v2_recovery.json"
RECOVERED = parent.OUT / "artifacts/recovered_summary.json"


def normalize_summary(value: dict[str, Any]) -> dict[str, Any]:
    """Only stringify the operation-count histogram keys; preserve all values."""
    return {
        **value,
        "operation_count_distribution": {
            str(key): count
            for key, count in value["operation_count_distribution"].items()
        },
    }


def freeze() -> None:
    """Record the failed file and recovery code before rebuilding the summary."""
    parent.verify()
    files = [
        Path(__file__).resolve(),
        ROOT / "tests/test_cscwd_review2_recovery.py",
        parent.SEAL,
    ]
    files += sorted(p for p in (parent.OUT / "artifacts").rglob("*") if p.is_file())
    failed = parent.OUT / "artifacts/summary.json"
    assert failed.is_file() and failed.stat().st_size == 0
    save(
        SEAL,
        {
            "created_at_utc": datetime.now(UTC).isoformat(),
            "reason": "MetadataValidationError: non-string object key at $.operation_count_distribution",
            "scope": "stringify histogram keys only; verify all saved generated panels and decisions by deterministic reconstruction; no new simulation or endpoint",
            "failed_original_file": str(failed.relative_to(ROOT)),
            "members": {str(p.relative_to(ROOT)): parent.sha(p) for p in files},
        },
    )
    print("RECOVERY_SEALED", parent.sha(SEAL))


def check() -> dict[str, Any]:
    """Verify the recovery amendment and every retained original artifact."""
    parent.verify()
    value = parent.read(SEAL)
    for path, expected in value["members"].items():
        assert parent.sha(ROOT / path) == expected, path
    return value


def run() -> None:
    """Reconstruct all outputs, checking rather than overwriting existing files."""
    check()
    compared = []

    def checked_save(path: Path, value: object) -> None:
        if path == parent.OUT / "artifacts/summary.json":
            assert isinstance(value, dict)
            save(RECOVERED, normalize_summary(value))
        else:
            assert parent.read(path) == value, str(path)
            compared.append(str(path.relative_to(ROOT)))

    with patch.object(parent, "save", checked_save):
        parent.artifact_analysis()
    check()
    save(
        parent.OUT / "artifacts/recovery_receipt.json",
        {
            "amendment_sha256": parent.sha(SEAL),
            "verified_original_outputs": compared,
            "recovered_summary_sha256": parent.sha(RECOVERED),
            "original_failed_file_retained": True,
            "new_simulation_pairs": 0,
        },
    )
    print("RECOVERED_WITH_RAW_OUTPUT_MATCHES", len(compared))


def consolidate() -> None:
    """Bind workflow results and the separately recovered artifact summary."""
    check()
    summary = parent.read(parent.OUT / "summary.json")
    save(
        parent.OUT / "publication_summary.json",
        {
            "workflow": summary,
            "artifact_analysis": parent.read(RECOVERED),
            "original_analysis_serialization_failure": parent.read(SEAL)["reason"],
            "recovery_receipt": parent.read(
                parent.OUT / "artifacts/recovery_receipt.json"
            ),
        },
    )
    save(
        parent.OUT / "publication_manifest.json",
        {
            "parent_freeze_sha256": parent.sha(parent.SEAL),
            "recovery_amendment_sha256": parent.sha(SEAL),
            "publication_summary_sha256": parent.sha(
                parent.OUT / "publication_summary.json"
            ),
        },
    )
    print("PUBLICATION_SUMMARY_BOUND")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("freeze", "run", "check", "consolidate"))
    action = parser.parse_args().action
    if action == "freeze":
        freeze()
    elif action == "run":
        run()
    elif action == "consolidate":
        consolidate()
    else:
        check()
        print("RECOVERY_CHECK_OK")
