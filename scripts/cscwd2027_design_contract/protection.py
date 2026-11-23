"""Protect the previous final-review deliverable and verify all older baselines."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from typing import Any

from scripts.cscwd2027_final_review.protection import ROOT as ROOT
from scripts.cscwd2027_final_review.protection import digest as digest
from scripts.cscwd2027_final_review.protection import verify as verify_revision2
from scripts.cscwd2027_revision2.baseline import verify as verify_original

DEST = ROOT / "results/cscwd2027/design_contract_v1/protection"
TREES = (
    "scripts/cscwd2027_final_review",
    "tests/cscwd2027_final_review",
    "results/cscwd2027/final_review",
    "reports/cscwd2027/final_review",
    "paper/cscwd2027_final_review",
    "release/cscwd2027/review_20261025_final_review_v1",
)


def verify() -> dict[str, Any]:
    """Check every prior protected byte without modifying any old member."""
    older = {"original": verify_original(), "revision2": verify_revision2()}
    path = DEST / "manifest.json"
    if digest(path) != path.with_suffix(".sha256").read_text().strip():
        raise RuntimeError("previous-delivery protection changed")
    members = json.loads(path.read_text())["members"]
    changed = [
        name
        for name, sha in members.items()
        if not (ROOT / name).is_file() or digest(ROOT / name) != sha
    ]
    if changed:
        raise RuntimeError(f"previous-delivery bytes changed: {changed}")
    return {**older, "final_review": {"unchanged": True, "files": len(members)}}


def main() -> None:
    """Capture once or verify the complete prior-delivery chain."""
    parser = argparse.ArgumentParser()
    parser.add_argument("operation", choices=("capture", "verify"))
    args = parser.parse_args()
    if args.operation == "verify":
        print(json.dumps(verify()))
        return
    verify_original()
    verify_revision2()
    DEST.mkdir(parents=True, exist_ok=False)
    files = [
        p
        for prefix in TREES
        for p in (ROOT / prefix).rglob("*")
        if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc"
    ]
    files.append(ROOT / "docs/cscwd2027/final_review.md")
    value = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "purpose": "prior final-review bytes, not experiment preregistration",
        "members": {str(p.relative_to(ROOT)): digest(p) for p in sorted(files)},
    }
    path = DEST / "manifest.json"
    path.write_text(json.dumps(value, indent=2) + "\n")
    path.with_suffix(".sha256").write_text(digest(path) + "\n")
    print(json.dumps({"files": len(files), "sha256": digest(path)}))


if __name__ == "__main__":
    main()
