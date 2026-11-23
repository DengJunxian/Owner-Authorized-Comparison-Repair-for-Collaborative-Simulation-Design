"""Write-once pre-run seal and execution of negotiated comparison repair."""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing
import platform
import shutil
import subprocess
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

from policytwin.negotiated_repair.analysis import summarize
from policytwin.negotiated_repair.design import STUDY, schedule
from policytwin.negotiated_repair.runtime import execute_session
from policytwin.typed_revision.runtime import read
from scripts.cscwd2027_design_contract.store import save
from scripts.cscwd2027_typed_revision import verify as previous_verify

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/cscwd2027/negotiated_repair_v1"
OUT = ROOT / "results/cscwd2027/negotiated_repair_v1"
WORK = ROOT / "ccfa-workfiles/writing/paired-comparison"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify() -> dict[str, Any]:
    previous_verify()
    frozen: dict[str, Any] = read(CONFIG / "freeze.json")
    for name, digest in frozen["members"].items():
        if not (ROOT / name).is_file() or sha(ROOT / name) != digest:
            raise RuntimeError("negotiated repair source differs: " + name)
    for name, digest in read(CONFIG / "protected_inputs.json")["members"].items():
        if not (ROOT / name).is_file() or sha(ROOT / name) != digest:
            raise RuntimeError("prior scientific input differs: " + name)
    return frozen


def freeze() -> None:
    previous_verify()
    if CONFIG.exists() or OUT.exists():
        raise FileExistsError("negotiated repair namespace already exists")
    tests = ET.parse(WORK / "build/negotiation_preseal.xml").getroot()
    if any(
        int(s.get("failures", 0)) or int(s.get("errors", 0))
        for s in tests.iter("testsuite")
    ):
        raise RuntimeError("pre-seal tests failed")
    CONFIG.mkdir(parents=True, exist_ok=False)
    save(CONFIG / "study.json", STUDY)
    save(CONFIG / "schedule.json", schedule())
    save(
        CONFIG / "preseal_tests.json",
        {
            "tests": sum(int(s.get("tests", 0)) for s in tests.iter("testsuite")),
            "failures": 0,
            "errors": 0,
            "junit_sha256": sha(WORK / "build/negotiation_preseal.xml"),
        },
    )
    protected = {}
    for tree in ("configs/cscwd2027", "results/cscwd2027"):
        for path in sorted((ROOT / tree).rglob("*")):
            if (
                path.is_file()
                and CONFIG not in path.parents
                and OUT not in path.parents
            ):
                protected[str(path.relative_to(ROOT))] = sha(path)
    save(CONFIG / "protected_inputs.json", {"members": protected})
    preserved = CONFIG / "previous_author_review"
    preserved.mkdir()
    for name in (
        "main.pdf",
        "technical_supplement.pdf",
        "source_for_author_review.zip",
    ):
        shutil.copyfile(
            ROOT / "paper/cscwd2027_design_contract" / name, preserved / name
        )
    paths = list((ROOT / "src/policytwin/negotiated_repair").glob("*.py"))
    paths += [
        Path(__file__),
        ROOT / "tests/test_cscwd_negotiated_repair.py",
        ROOT / "docs/cscwd2027/owner_authorized_repair.md",
        WORK / "negotiated_repair_plan.md",
        ROOT / "uv.lock",
        ROOT / "pyproject.toml",
        ROOT / "AGENTS.md",
    ]
    paths += [p for p in CONFIG.rglob("*") if p.is_file()]
    with zipfile.ZipFile(
        CONFIG / "negotiated_repair_source.zip", "x", zipfile.ZIP_DEFLATED
    ) as archive:
        for path in sorted(set(paths)):
            archive.write(path, str(path.relative_to(ROOT)))
    paths.append(CONFIG / "negotiated_repair_source.zip")
    save(
        CONFIG / "freeze.json",
        {
            "created_at_utc": datetime.now(UTC).isoformat(),
            "time_authority": "local pre-run ordering; not external attestation",
            "git_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip(),
            "working_tree": "dirty; exact source members take precedence",
            "python": platform.python_version(),
            "study": STUDY,
            "sessions": 4,
            "tickets": 24,
            "method_panels": 96,
            "parent_parallelism": 2,
            "members": {str(p.relative_to(ROOT)): sha(p) for p in sorted(set(paths))},
        },
    )
    print(
        "NEGOTIATED_REPAIR_FROZEN",
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
            pool.submit(execute_session, b, OUT, seal): b["session"]["id"]
            for b in read(CONFIG / "schedule.json")
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
            print("NEGOTIATED_SESSION", len(successes), len(failures), flush=True)
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
        raise RuntimeError("session failures retained; no retry is authorized")


def analyze() -> None:
    verify()
    completed = read(OUT / "run_completed.json")
    if completed["failures"] or len(completed["sessions"]) != 4:
        raise RuntimeError("cannot claim complete analysis of an incomplete schedule")
    if (OUT / "summary.json").exists():
        raise FileExistsError(
            "analysis already materialized; recompute in a fresh directory"
        )
    summary = summarize(OUT)
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
    operation = parser.parse_args().operation
    if operation == "freeze":
        freeze()
    elif operation == "run":
        run()
    elif operation == "analyze":
        analyze()
    else:
        verify()
        print(
            "NEGOTIATED_REPAIR_INPUTS_UNCHANGED",
            sha(CONFIG / "freeze.json"),
            flush=True,
        )


if __name__ == "__main__":
    main()
