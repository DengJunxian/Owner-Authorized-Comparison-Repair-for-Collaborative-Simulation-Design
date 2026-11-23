"""Reconstruct serialized observations byte-for-byte without altering raw files."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from unittest.mock import patch

from policytwin.provenance import canonical_json_bytes
from scripts import cscwd2027_review2_recovery as recovery
from scripts import cscwd2027_review2_recovery_runner as directory_adapter
from scripts.cscwd2027_design_contract.store import save

PARENT = recovery.parent
SEAL = PARENT.ROOT / "configs/cscwd2027/review2_canonical_recovery.json"


def serialized_equal(path: Path, value: object) -> bool:
    """Compare the actual encoding, including normalization of tuple arrays."""
    return path.read_bytes() == canonical_json_bytes(value)


def check() -> None:
    """Bind the new adapter and all previous failed attempts."""
    recovery.check()
    directory_seal = PARENT.read(directory_adapter.SEAL)
    for path, expected in directory_seal["members"].items():
        assert PARENT.sha(PARENT.ROOT / path) == expected, path
    for path, expected in PARENT.read(SEAL)["members"].items():
        assert PARENT.sha(PARENT.ROOT / path) == expected, path


def freeze() -> None:
    """Seal the encoding-only correction before reconstructing any summary."""
    recovery.check()
    paths = [
        Path(__file__).resolve(),
        PARENT.ROOT / "tests/test_cscwd_canonical_recovery.py",
        directory_adapter.SEAL,
        recovery.SEAL,
    ]
    save(
        SEAL,
        {
            "created_at_utc": datetime.now(UTC).isoformat(),
            "reason": "JSON arrays decode to lists while contract fields are tuples",
            "scope": "compare canonical serialized bytes; only stringify summary "
            "histogram keys; no new simulation or endpoint change",
            "members": {
                str(p.relative_to(PARENT.ROOT)): PARENT.sha(p) for p in paths
            },
        },
    )
    print("CANONICAL_RECOVERY_SEALED", PARENT.sha(SEAL))


def run() -> None:
    """Verify every retained panel/report before writing a new summary path."""
    check()
    compared: list[str] = []
    original_mkdir = Path.mkdir

    def mkdir(path: Path, *args: Any, **kwargs: Any) -> Any:
        return directory_adapter.allow_existing_root(
            original_mkdir, path, *args, **kwargs
        )

    def checked_save(path: Path, value: object) -> None:
        if path == PARENT.OUT / "artifacts/summary.json":
            assert isinstance(value, dict)
            save(recovery.RECOVERED, recovery.normalize_summary(value))
        else:
            assert serialized_equal(path, value), str(path)
            compared.append(str(path.relative_to(PARENT.ROOT)))

    with patch.object(Path, "mkdir", mkdir), patch.object(
        PARENT, "save", checked_save
    ):
        PARENT.artifact_analysis()
    check()
    save(
        PARENT.OUT / "artifacts/recovery_receipt.json",
        {
            "amendment_sha256": PARENT.sha(SEAL),
            "verified_original_outputs": compared,
            "comparison": "exact canonical serialized bytes",
            "recovered_summary_sha256": PARENT.sha(recovery.RECOVERED),
            "original_failed_file_retained": True,
            "new_simulation_pairs": 0,
            "retained_failures": [
                "integer-key summary serialization",
                "existing reconstruction directory",
                "decoded list versus in-memory tuple equality",
            ],
        },
    )
    print("CANONICAL_RAW_OUTPUT_MATCHES", len(compared))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("freeze", "run", "check"))
    action = parser.parse_args().action
    if action == "freeze":
        freeze()
    elif action == "run":
        run()
    else:
        check()
        print("CANONICAL_RECOVERY_CHECK_OK")
