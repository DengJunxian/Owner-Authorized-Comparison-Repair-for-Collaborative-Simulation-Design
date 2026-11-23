"""Write-once pre-outcome freeze, execution and complete retained-record analysis."""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing
import platform
import subprocess
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

from policytwin.consent_projection.analysis import summarize
from policytwin.consent_projection.runtime import execute_session
from policytwin.consent_projection.study import STUDY, schedule
from policytwin.typed_revision.runtime import read
from scripts.cscwd2027_design_contract.store import save
from scripts.cscwd2027_owner_policy import verify as previous_verify

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/cscwd2027/consent_projection_v1"
OUT = ROOT / "results/cscwd2027/consent_projection_v1"
REPORTS = ROOT / "reports/cscwd2027/consent_projection_v1"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify() -> dict[str, Any]:
    previous_verify()
    frozen: dict[str, Any] = read(CONFIG / "freeze.json")
    for name, digest in frozen["members"].items():
        if not (ROOT / name).is_file() or sha(ROOT / name) != digest:
            raise RuntimeError("sealed projection source differs: " + name)
    for name, digest in read(CONFIG / "protected_inputs.json")["members"].items():
        if not (ROOT / name).is_file() or sha(ROOT / name) != digest:
            raise RuntimeError("protected prior input differs: " + name)
    if (OUT / "manifest.json").exists():
        for name, digest in read(OUT / "manifest.json")["members"].items():
            if not (OUT / name).is_file() or sha(OUT / name) != digest:
                raise RuntimeError("raw projection artifact differs: " + name)
    return frozen


def freeze() -> None:
    previous_verify()
    if CONFIG.exists() or OUT.exists():
        raise FileExistsError("projection namespace already exists")
    tests_path = REPORTS / "preseal_tests.xml"
    tests = ET.parse(tests_path).getroot()
    count = sum(int(s.get("tests", 0)) for s in tests.iter("testsuite"))
    if count == 0 or any(
        int(s.get(k, 0))
        for s in tests.iter("testsuite")
        for k in ("failures", "errors", "skipped")
    ):
        raise RuntimeError("preseal tests absent, failed or skipped")
    CONFIG.mkdir(parents=True, exist_ok=False)
    save(CONFIG / "study.json", STUDY)
    save(CONFIG / "schedule.json", schedule())
    save(
        CONFIG / "preseal_tests.json",
        {
            "tests": count,
            "junit_sha256": sha(tests_path),
            "failures": 0,
            "errors": 0,
            "skipped": 0,
        },
    )
    protected = {}
    for tree in ("configs", "results"):
        for path in sorted((ROOT / tree).rglob("*")):
            if (
                path.is_file()
                and CONFIG not in path.parents
                and OUT not in path.parents
            ):
                protected[str(path.relative_to(ROOT))] = sha(path)
    save(CONFIG / "protected_inputs.json", {"members": protected})
    paths = list((ROOT / "src/policytwin/consent_projection").glob("*.py"))
    paths += [
        Path(__file__),
        ROOT / "tests/test_cscwd_consent_projection.py",
        ROOT / "docs/cscwd2027/consent_projection.md",
        ROOT / "uv.lock",
        ROOT / "pyproject.toml",
        ROOT / "AGENTS.md",
    ]
    paths += [p for p in CONFIG.rglob("*") if p.is_file()]
    with zipfile.ZipFile(
        CONFIG / "consent_projection_source.zip", "x", zipfile.ZIP_DEFLATED
    ) as archive:
        for path in sorted(set(paths)):
            archive.write(path, str(path.relative_to(ROOT)))
    paths.append(CONFIG / "consent_projection_source.zip")
    save(
        CONFIG / "freeze.json",
        {
            "created_at_utc": datetime.now(UTC).isoformat(),
            "time_authority": "local pre-run ordering; no external attestation",
            "git_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip(),
            "working_tree": "dirty; exact member hashes authoritative",
            "python": platform.python_version(),
            "parent_freeze_sha256": sha(
                ROOT / "configs/cscwd2027/owner_policy_v1/freeze.json"
            ),
            "study": STUDY,
            "scientific_blocks": 4,
            "planning_requests": 384,
            "physical_requests": 12,
            "physical_method_panels": 48,
            "parent_parallelism": 2,
            "members": {str(p.relative_to(ROOT)): sha(p) for p in sorted(set(paths))},
        },
    )
    print(
        "CONSENT_PROJECTION_FROZEN",
        sha(CONFIG / "freeze.json"),
        len(protected),
        flush=True,
    )


def run() -> None:
    verify()
    OUT.mkdir(parents=True, exist_ok=False)
    seal = sha(CONFIG / "freeze.json")
    save(
        OUT / "run_started.json",
        {"started_at_utc": datetime.now(UTC).isoformat(), "freeze_sha256": seal},
    )
    successes, failures = [], []
    with ProcessPoolExecutor(
        max_workers=2, mp_context=multiprocessing.get_context("spawn")
    ) as pool:
        futures = {
            pool.submit(execute_session, block, OUT, seal): block["session"]["id"]
            for block in read(CONFIG / "schedule.json")
        }
        for future in as_completed(futures):
            try:
                successes.append(future.result())
            except Exception as error:
                failure = {
                    "session": futures[future],
                    "error": f"{type(error).__name__}: {error}",
                }
                failures.append(failure)
                save(OUT / "session_failures" / (futures[future] + ".json"), failure)
            print("PROJECTION_SESSION", len(successes), len(failures), flush=True)
    save(
        OUT / "run_completed.json",
        {
            "completed_at_utc": datetime.now(UTC).isoformat(),
            "sessions": successes,
            "failures": failures,
        },
    )
    verify()
    if failures:
        raise RuntimeError("session failures retained; no retries")


def analyze(destination: Path | None = None) -> None:
    frozen = verify()
    completed = read(OUT / "run_completed.json")
    if completed["failures"] or len(completed["sessions"]) != 4:
        raise RuntimeError("incomplete schedule; failures retained")
    started = read(OUT / "run_started.json")
    if not (
        datetime.fromisoformat(frozen["created_at_utc"])
        < datetime.fromisoformat(started["started_at_utc"])
        < datetime.fromisoformat(completed["completed_at_utc"])
    ) or started["freeze_sha256"] != sha(CONFIG / "freeze.json"):
        raise RuntimeError("pre-run ordering or run seal differs")
    target = OUT if destination is None else destination
    if (target / "summary.json").exists():
        raise FileExistsError("analysis already exists")
    summary = summarize(OUT, target)
    if destination is None:
        save(
            OUT / "manifest.json",
            {
                "members": {
                    str(p.relative_to(OUT)): sha(p)
                    for p in sorted(OUT.rglob("*"))
                    if p.is_file() and p.name != "manifest.json"
                }
            },
        )
    print(json.dumps(summary), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("operation", choices=("freeze", "check", "run", "analyze"))
    parser.add_argument("--destination", type=Path)
    args = parser.parse_args()
    if args.operation == "freeze":
        freeze()
    elif args.operation == "run":
        run()
    elif args.operation == "analyze":
        analyze(args.destination)
    else:
        verify()
        print("CONSENT_PROJECTION_UNCHANGED", sha(CONFIG / "freeze.json"))


if __name__ == "__main__":
    main()
