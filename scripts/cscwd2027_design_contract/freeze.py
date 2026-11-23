"""Pre-execution seal for the separate comparison-contract study."""

from __future__ import annotations

import argparse
import json
import platform
import sys
import zipfile
from dataclasses import asdict
from datetime import UTC, datetime
from typing import Any

from scripts.cscwd2027_design_contract.design import (
    CONFIG,
    attempts,
    cases,
    contract,
    sessions,
)
from scripts.cscwd2027_design_contract.protection import ROOT, digest, verify
from scripts.cscwd2027_design_contract.store import save
from scripts.cscwd2027_revision2.freeze import check as previous_check

DEST = ROOT / "configs/cscwd2027/design_contract_v1"


def check() -> dict[str, Any]:
    """Reject altered source/configuration/analysis; never rewrite a prior seal."""
    previous_check()
    path = DEST / "freeze.json"
    if digest(path) != path.with_suffix(".sha256").read_text().strip():
        raise RuntimeError("comparison-contract freeze changed")
    value: dict[str, Any] = json.loads(path.read_text())
    for name, sha in value["members"].items():
        if not (ROOT / name).is_file() or digest(ROOT / name) != sha:
            raise RuntimeError(f"frozen member changed: {name}")
    if digest(DEST / "source.zip") != value["source_archive_sha256"]:
        raise RuntimeError("source snapshot changed")
    return value


def freeze() -> None:
    """Seal every endpoint, recipe, case, sample and source before formal runs."""
    verify()
    previous = previous_check()
    DEST.mkdir(parents=True, exist_ok=False)
    blocks = sessions()
    save(DEST / "study.json", CONFIG)
    save(DEST / "sessions.json", blocks)
    save(DEST / "attempts.json", [row for s in blocks for row in attempts(s)])
    save(DEST / "handoffs.json", [row for s in blocks for row in cases(s)])
    save(
        DEST / "contracts.json",
        {s["id"]: {str(v): asdict(contract(s, v)) for v in (1, 2)} for s in blocks},
    )
    files = {ROOT / name for name in previous["members"]}
    for prefix in (
        "src/policytwin/design_contract",
        "scripts/cscwd2027_design_contract",
        "tests/cscwd2027_design_contract",
    ):
        files.update((ROOT / prefix).glob("*.py"))
    files.update(
        {
            ROOT / "docs/cscwd2027/design_contract_v1_protocol.md",
            ROOT / "AGENTS.md",
            ROOT / "pyproject.toml",
            ROOT / "uv.lock",
        }
    )
    files.update(DEST.glob("*.json"))
    with zipfile.ZipFile(DEST / "source.zip", "x", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(files):
            archive.write(path, str(path.relative_to(ROOT)))
    value = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "timestamp_authority": "local pre-execution clock, not external attestation",
        "label": CONFIG["label"],
        "python": sys.version,
        "platform": platform.platform(),
        "planned_sessions": len(blocks),
        "planned_attempts": sum(len(attempts(s)) for s in blocks),
        "planned_handoffs": sum(len(cases(s)) for s in blocks),
        "previous_freeze_sha256": digest(
            ROOT / "configs/cscwd2027/revision2/freeze.json"
        ),
        "members": {str(p.relative_to(ROOT)): digest(p) for p in sorted(files)},
        "source_archive_sha256": digest(DEST / "source.zip"),
    }
    save(DEST / "freeze.json", value)
    (DEST / "freeze.sha256").write_text(digest(DEST / "freeze.json") + "\n")
    print(
        json.dumps(
            {
                "status": "FROZEN",
                "sha256": digest(DEST / "freeze.json"),
                "created_at_utc": value["created_at_utc"],
                "attempts": value["planned_attempts"],
            }
        )
    )


def main() -> None:
    """Create this version once or check its exact members."""
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
