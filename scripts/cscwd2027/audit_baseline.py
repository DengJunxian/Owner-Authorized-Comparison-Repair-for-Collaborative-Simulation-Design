"""Capture and verify the pre-CSCWD file set without modifying historical files."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import zipfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "results/cscwd2027/baseline"
TREES = ("src", "configs", "scripts", "tests", "docs", "paper", "results", "dai2026")
SMALL_TREES = {"src", "configs", "scripts", "tests", "docs", "paper"}


def digest(path: Path) -> str:
    """Hash a file in bounded memory."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def protected_files() -> list[Path]:
    """Enumerate the actual pre-existing files, including untracked evidence."""
    paths: set[Path] = set()
    for tree in TREES:
        for path in (ROOT / tree).rglob("*"):
            relative = path.relative_to(ROOT)
            if not path.is_file() or path.is_symlink():
                continue
            if set(relative.parts) & {"cscwd2027", "coordination", "__pycache__"}:
                continue
            if path.suffix in {".pyc", ".pyo"} or path.name == ".DS_Store":
                continue
            paths.add(path)
    paths.update(
        p for p in (ROOT / "release/aaai27_final_submission").iterdir() if p.is_file()
    )
    for name in ("AGENTS.md", "pyproject.toml", "uv.lock", ".python-version"):
        if (ROOT / name).is_file():
            paths.add(ROOT / name)
    return sorted(paths)


def run_check(command: list[str], cwd: Path = ROOT) -> dict[str, Any]:
    """Retain the exact command, exit status and output of a diagnostic."""
    result = subprocess.run(command, cwd=cwd, capture_output=True, text=True)
    return {
        "command": command,
        "cwd_relative": str(cwd.relative_to(ROOT)),
        "exit_code": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }


def historical_search() -> dict[str, Any]:
    """Search scoped local source copies and AAAI archives for the old bytes."""
    freeze = json.loads((ROOT / "configs/revision_v2_freeze_manifest.json").read_text())
    target = next(
        m["sha256"]
        for m in freeze["members"]
        if m["path"].endswith("coupling_contract.py")
    )
    suffix = "src/policytwin/counterfactual/coupling_contract.py"
    found: list[dict[str, Any]] = []
    candidates = [ROOT / suffix]
    for directory in (
        ROOT / "release",
        ROOT / "tmp",
        ROOT.parent / "aaai27_rebuttal_20261024/sources",
    ):
        candidates.extend(directory.rglob("coupling_contract.py"))
    for path in sorted(set(candidates)):
        if path.is_file():
            observed = digest(path)
            found.append(
                {
                    "path": str(path.relative_to(ROOT.parent)),
                    "sha256": observed,
                    "match": observed == target,
                }
            )
    for path in sorted((ROOT / "release").rglob("*.zip")):
        if "dai2026" in path.parts:
            continue
        with zipfile.ZipFile(path) as archive:
            for name in archive.namelist():
                if name.endswith(suffix):
                    observed = hashlib.sha256(archive.read(name)).hexdigest()
                    found.append(
                        {
                            "path": str(path.relative_to(ROOT.parent)),
                            "member": name,
                            "sha256": observed,
                            "match": observed == target,
                        }
                    )
    members = [
        {
            **m,
            "observed_sha256": digest(ROOT / m["path"]),
            "match": digest(ROOT / m["path"]) == m["sha256"],
        }
        for m in freeze["members"]
    ]
    return {
        "expected_sha256": target,
        "current_members": members,
        "candidates": found,
        "recovered": any(m["match"] for m in found),
        "scope": "Current tree, local AAAI release/tmp and rebuttal source copies.",
    }


def capture() -> None:
    """Write a new baseline; never replace an earlier capture."""
    BASE.mkdir(parents=True, exist_ok=False)
    files = protected_files()
    members = [
        {
            "path": str(p.relative_to(ROOT)),
            "bytes": p.stat().st_size,
            "sha256": digest(p),
        }
        for p in files
    ]
    archive_path = BASE / "source_and_authorities.zip"
    with zipfile.ZipFile(archive_path, "x", zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            relative = path.relative_to(ROOT)
            if (
                relative.parts[0] in SMALL_TREES
                or str(relative).startswith(
                    (
                        "release/aaai27_final_submission/",
                        "results/frozen/",
                        "results/registered_evidence_lock.json",
                    )
                )
                or len(relative.parts) == 1
            ):
                archive.write(path, str(relative))
    checks = [
        run_check(
            [
                sys.executable,
                "scripts/check_registered_metadata.py",
                "paper/main_revised.tex",
            ]
        ),
        run_check([sys.executable, "scripts/verify_original_evidence.py"]),
        run_check(
            ["shasum", "-a", "256", "-c", "SHA256SUMS.txt"],
            ROOT / "release/aaai27_final_submission",
        ),
    ]
    payload = {
        "schema_version": "policytwin.cscwd-baseline.v1",
        "captured_at_utc": datetime.now(UTC).isoformat(),
        "git_head": run_check(["git", "rev-parse", "HEAD"])["stdout"].strip(),
        "git_status": run_check(["git", "status", "--short"])["stdout"],
        "python": platform.python_version(),
        "members": members,
        "source_archive_sha256": digest(archive_path),
        "checks": checks,
        "historical_v2_freeze": historical_search(),
    }
    path = BASE / "protected_files.json"
    with path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
    with (BASE / "protected_files.sha256").open("x") as stream:
        stream.write(digest(path) + "  protected_files.json\n")
    print(
        json.dumps(
            {
                "protected_files": len(members),
                "protected_bytes": sum(m["bytes"] for m in members),
                "checks_pass": all(c["exit_code"] == 0 for c in checks),
                "historical_v2_recovered": payload["historical_v2_freeze"]["recovered"],
                "manifest_sha256": digest(path),
            },
            indent=2,
        )
    )
    if not all(c["exit_code"] == 0 for c in checks):
        raise RuntimeError("original evidence diagnostics failed; inspect baseline")


def verify() -> None:
    """Fail on deletion or byte change of any protected member."""
    path = BASE / "protected_files.json"
    expected = (BASE / "protected_files.sha256").read_text().split()[0]
    if digest(path) != expected:
        raise RuntimeError("baseline manifest changed")
    payload = json.loads(path.read_text())
    failures = [
        m["path"]
        for m in payload["members"]
        if not (ROOT / m["path"]).is_file() or digest(ROOT / m["path"]) != m["sha256"]
    ]
    if digest(BASE / "source_and_authorities.zip") != payload["source_archive_sha256"]:
        failures.append("source_and_authorities.zip")
    if failures:
        raise RuntimeError(f"protected files changed: {failures}")
    print(f"BASELINE_UNCHANGED: {len(payload['members'])} files")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("operation", choices=("capture", "verify"))
    args = parser.parse_args()
    capture() if args.operation == "capture" else verify()
