"""Separate pre-execution seal; checks original v1 without replacing its manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import zipfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from policytwin.provenance import canonical_json_bytes
from scripts.cscwd2027.freeze_study import check as original_check
from scripts.cscwd2027_revision2.baseline import verify
from scripts.cscwd2027_revision2.design import CONFIG, schedule

ROOT = Path(__file__).resolve().parents[2]
DEST = ROOT / "configs/cscwd2027/revision2"


def digest(path: Path) -> str:
    """Hash a file in bounded memory."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def check() -> dict[str, Any]:
    """Fail on a changed scientific input, implementation or analysis."""
    original_check()
    path = DEST / "freeze.json"
    if digest(path) != (DEST / "freeze.sha256").read_text().strip():
        raise RuntimeError("revision2 freeze manifest changed")
    value: dict[str, Any] = json.loads(path.read_text())
    for name, sha in value["members"].items():
        if not (ROOT / name).is_file() or digest(ROOT / name) != sha:
            raise RuntimeError(f"revision2 frozen member changed: {name}")
    if digest(ROOT / value["source_archive"]) != value["source_archive_sha256"]:
        raise RuntimeError("revision2 source archive changed")
    return value


def freeze() -> None:
    """Save all primary choices and create a new version once."""
    verify()
    original = original_check()
    DEST.mkdir(parents=True, exist_ok=False)
    (DEST / "study.yaml").write_bytes(canonical_json_bytes(CONFIG))
    rows = schedule()
    (DEST / "schedule.json").write_bytes(canonical_json_bytes(rows))
    files = {ROOT / name for name in original["members"]}
    for directory in (
        "src/policytwin/coordination_validation",
        "scripts/cscwd2027_revision2",
        "tests/cscwd2027_revision2",
    ):
        files.update((ROOT / directory).glob("*.py"))
    files.update((ROOT / "docs/cscwd2027/revision2").glob("*.md"))
    files.update({DEST / "study.yaml", DEST / "schedule.json"})
    archive = DEST / "source.zip"
    with zipfile.ZipFile(archive, "x", zipfile.ZIP_DEFLATED) as out:
        for path in sorted(files):
            out.write(path, str(path.relative_to(ROOT)))
    value = {
        "schema": "cscwd-post-registration-revision2-v1",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "timestamp_authority": "local clock, not third-party attestation",
        "python": sys.version,
        "platform": platform.platform(),
        "original_v1_freeze_sha256": digest(
            ROOT / "configs/cscwd2027/freeze_manifest_v1.json"
        ),
        "protected_baseline_sha256": digest(
            ROOT / "results/cscwd2027/revision2/baseline/protected.json"
        ),
        "total_attempts": len(rows),
        "members": {str(p.relative_to(ROOT)): digest(p) for p in sorted(files)},
        "source_archive": str(archive.relative_to(ROOT)),
        "source_archive_sha256": digest(archive),
    }
    path = DEST / "freeze.json"
    path.write_bytes(canonical_json_bytes(value))
    (DEST / "freeze.sha256").write_text(digest(path) + "\n")
    print(
        json.dumps(
            {
                "status": "FROZEN",
                "attempts": len(rows),
                "sha256": digest(path),
                "created_at_utc": value["created_at_utc"],
            }
        )
    )


def main() -> None:
    """Freeze a fresh study or check its unchanged members."""
    parser = argparse.ArgumentParser()
    parser.add_argument("operation", choices=("freeze", "check"))
    args = parser.parse_args()
    if args.operation == "freeze":
        freeze()
    else:
        value = check()
        print(
            json.dumps(
                {"status": "FREEZE_OK", "created_at_utc": value["created_at_utc"]}
            )
        )


if __name__ == "__main__":
    main()
