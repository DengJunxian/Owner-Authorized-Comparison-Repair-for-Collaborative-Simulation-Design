"""Create/check an immutable local pre-run source/configuration manifest."""

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
from scripts.cscwd2027.study_design import CONFIG, schedule

ROOT = Path(__file__).resolve().parents[2]
DEST = ROOT / "configs/cscwd2027"


def digest(path: Path) -> str:
    """Read exact source bytes, including all local uncommitted modifications."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def members() -> list[Path]:
    """Include reused runtime and all study/analysis code before execution."""
    paths: list[Path] = []
    for prefix in (
        "src/policytwin/coordination",
        "src/policytwin/market",
        "src/policytwin/counterfactual",
        "scripts/cscwd2027",
        "tests/cscwd2027",
    ):
        paths.extend((ROOT / prefix).glob("*.py"))
    paths += [
        ROOT / "src/policytwin/provenance.py",
        ROOT / "pyproject.toml",
        ROOT / "uv.lock",
    ]
    paths += list((ROOT / "docs/cscwd2027").glob("*.md"))
    paths += [DEST / "study_v1.yaml", DEST / "schedule_v1.json"]
    return sorted(set(paths))


def check() -> dict[str, Any]:
    """Fail on a changed member, manifest or source snapshot."""
    manifest_path = DEST / "freeze_manifest_v1.json"
    expected = (DEST / "freeze_manifest_v1.sha256").read_text().strip()
    if digest(manifest_path) != expected:
        raise RuntimeError("freeze manifest changed")
    manifest: dict[str, Any] = json.loads(manifest_path.read_text())
    if not isinstance(manifest, dict):
        raise ValueError("freeze manifest must be an object")
    changed = [
        name
        for name, sha in manifest["members"].items()
        if not (ROOT / name).is_file() or digest(ROOT / name) != sha
    ]
    if changed:
        raise RuntimeError(f"frozen source/configuration changed: {changed}")
    archive = ROOT / manifest["source_archive"]
    if digest(archive) != manifest["source_archive_sha256"]:
        raise RuntimeError("frozen source archive changed")
    return manifest


def freeze() -> None:
    """Write the complete design first, then seal it with source and UTC receipt."""
    DEST.mkdir(parents=True, exist_ok=False)
    grid = schedule()
    (DEST / "study_v1.yaml").write_bytes(canonical_json_bytes(CONFIG))
    (DEST / "schedule_v1.json").write_bytes(canonical_json_bytes(grid))
    archive = DEST / "source_v1.zip"
    with zipfile.ZipFile(archive, "x", zipfile.ZIP_DEFLATED) as out:
        for path in members():
            out.write(path, str(path.relative_to(ROOT)))
    counts = {
        group: sum(r["group"] == group for r in grid)
        for group in sorted({r["group"] for r in grid})
    }
    manifest = {
        "schema": "cscwd-local-freeze-v1",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "timestamp_authority": "local clock; not external attestation",
        "python": sys.version,
        "platform": platform.platform(),
        "counts": counts,
        "total_paired_attempts": len(grid),
        "members": {str(p.relative_to(ROOT)): digest(p) for p in members()},
        "source_archive": str(archive.relative_to(ROOT)),
        "source_archive_sha256": digest(archive),
    }
    path = DEST / "freeze_manifest_v1.json"
    path.write_bytes(canonical_json_bytes(manifest))
    (DEST / "freeze_manifest_v1.sha256").write_text(digest(path) + "\n")
    print(
        json.dumps(
            {
                "status": "FROZEN",
                "sha256": digest(path),
                "counts": counts,
                "total_paired_attempts": len(grid),
            }
        ),
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("operation", choices=("freeze", "check"))
    args = parser.parse_args()
    if args.operation == "freeze":
        freeze()
    else:
        result = check()
        print(
            json.dumps(
                {"status": "FREEZE_OK", "created_at_utc": result["created_at_utc"]}
            )
        )


if __name__ == "__main__":
    main()
