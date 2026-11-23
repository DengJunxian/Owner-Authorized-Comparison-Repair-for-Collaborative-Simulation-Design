"""Protect every existing CSCWD study/source/delivery byte before revision two."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
DEST = ROOT / "results/cscwd2027/revision2/baseline"
TREES = (
    "configs/cscwd2027",
    "results/cscwd2027",
    "scripts/cscwd2027",
    "src/policytwin/coordination",
    "src/policytwin/market",
    "src/policytwin/counterfactual",
    "tests/cscwd2027",
    "docs/cscwd2027",
    "paper/cscwd2027",
    "release/cscwd2027",
)


def digest(path: Path) -> str:
    """Stream an exact byte digest."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verify() -> dict[str, Any]:
    """Verify only the captured old members, without changing them."""
    path = DEST / "protected.json"
    if digest(path) != (DEST / "protected.sha256").read_text().strip():
        raise RuntimeError("revision baseline manifest changed")
    record: dict[str, Any] = json.loads(path.read_text())
    changed = [
        name
        for name, value in record["members"].items()
        if not (ROOT / name).is_file() or digest(ROOT / name) != value["sha256"]
    ]
    if changed:
        raise RuntimeError(f"protected original CSCWD bytes changed: {changed}")
    return {"unchanged": True, "files": len(record["members"])}


def main() -> None:
    """Create a new baseline once or verify the old one."""
    parser = argparse.ArgumentParser()
    parser.add_argument("operation", choices=("capture", "verify"))
    args = parser.parse_args()
    if args.operation == "verify":
        print(json.dumps(verify()))
        return
    DEST.mkdir(parents=True, exist_ok=False)
    files = set()
    for prefix in TREES:
        for path in (ROOT / prefix).rglob("*"):
            if not path.is_file() or path.is_symlink():
                continue
            relative = path.relative_to(ROOT)
            if "revision2" in relative.parts or "__pycache__" in relative.parts:
                continue
            if path.suffix in {".pyc", ".pyo"} or path.name == ".DS_Store":
                continue
            files.add(path)
    record = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "purpose": "pre-revision2 protection; does not replace the v1 freeze",
        "members": {
            str(path.relative_to(ROOT)): {
                "sha256": digest(path),
                "bytes": path.stat().st_size,
            }
            for path in sorted(files)
        },
    }
    path = DEST / "protected.json"
    path.write_text(json.dumps(record, indent=2) + "\n")
    (DEST / "protected.sha256").write_text(digest(path) + "\n")
    print(json.dumps({"files": len(files), "sha256": digest(path)}))


if __name__ == "__main__":
    main()
