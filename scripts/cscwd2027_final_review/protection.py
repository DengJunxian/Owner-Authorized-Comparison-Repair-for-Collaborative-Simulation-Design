"""Protect revision2 evidence and delivery bytes before final-review changes."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
DEST = ROOT / "results/cscwd2027/final_review/protection"
TREES = (
    "configs/cscwd2027/revision2",
    "results/cscwd2027/revision2",
    "reports/cscwd2027/revision2",
    "scripts/cscwd2027_revision2",
    "src/policytwin/coordination_validation",
    "tests/cscwd2027_revision2",
    "docs/cscwd2027/revision2",
    "paper/cscwd2027_revision2",
    "release/cscwd2027/review_20261025_revision2_v1",
)


def digest(path: Path) -> str:
    """Hash an existing file without changing it."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def capture() -> dict[str, Any]:
    """Seal existing members once, before any manuscript revision."""
    DEST.mkdir(parents=True, exist_ok=False)
    paths = sorted(
        p
        for tree in TREES
        for p in (ROOT / tree).rglob("*")
        if p.is_file()
        and not p.is_symlink()
        and "__pycache__" not in p.parts
        and p.suffix not in {".pyc", ".pyo"}
        and p.name != ".DS_Store"
    )
    record = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "scope": "revision2 preservation, not a new experiment registration",
        "members": {str(p.relative_to(ROOT)): digest(p) for p in paths},
    }
    path = DEST / "manifest.json"
    path.write_text(json.dumps(record, indent=2) + "\n")
    path.with_suffix(".sha256").write_text(digest(path) + "\n")
    return {"files": len(paths), "sha256": digest(path)}


def verify() -> dict[str, Any]:
    """Require all protected files to retain their exact captured bytes."""
    path = DEST / "manifest.json"
    if digest(path) != path.with_suffix(".sha256").read_text().strip():
        raise RuntimeError("final-review preservation manifest changed")
    record = json.loads(path.read_text())
    changed = [
        name
        for name, sha in record["members"].items()
        if not (ROOT / name).is_file() or digest(ROOT / name) != sha
    ]
    if changed:
        raise RuntimeError(f"protected revision2 files changed: {changed}")
    return {"unchanged": True, "files": len(record["members"])}


def main() -> None:
    """Capture or verify the separate final-review protection manifest."""
    parser = argparse.ArgumentParser()
    parser.add_argument("operation", choices=("capture", "verify"))
    args = parser.parse_args()
    print(json.dumps(capture() if args.operation == "capture" else verify()))


if __name__ == "__main__":
    main()
