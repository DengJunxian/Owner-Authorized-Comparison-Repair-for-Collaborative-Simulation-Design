"""Write-once seal, actual role execution and complete typed-revision analysis."""

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

from policytwin.typed_revision.analysis import summarize
from policytwin.typed_revision.design import STUDY, schedule
from policytwin.typed_revision.runtime import execute_session, read
from scripts.cscwd2027_design_contract.store import save
from scripts.cscwd2027_mechanism_review import protect_check as prior_protect
from scripts.cscwd2027_mechanism_review import verify as prior_verify

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/cscwd2027/typed_revision_v1"
OUT = ROOT / "results/cscwd2027/typed_revision_v1"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify() -> dict:
    prior_verify()
    prior_protect()
    seal = read(CONFIG / "freeze.json")
    for name, expected in seal["members"].items():
        if not (ROOT / name).is_file() or sha(ROOT / name) != expected:
            raise RuntimeError("sealed typed-revision input differs: " + name)
    for name, expected in read(CONFIG / "protected_inputs.json")["members"].items():
        if not (ROOT / name).is_file() or sha(ROOT / name) != expected:
            raise RuntimeError("prior scientific file differs: " + name)
    return seal


def freeze() -> None:
    prior_verify()
    prior_protect()
    if (CONFIG / "freeze.json").exists() or OUT.exists():
        raise FileExistsError("typed revision namespace already used")
    CONFIG.mkdir(parents=True, exist_ok=True)
    save(CONFIG / "study.json", STUDY)
    blocks = schedule()
    save(CONFIG / "schedule.json", blocks)
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
    previous = CONFIG / "previous_author_review"
    previous.mkdir(exist_ok=False)
    for name in (
        "main.pdf",
        "technical_supplement.pdf",
        "source_for_author_review.zip",
    ):
        shutil.copyfile(
            ROOT / "paper/cscwd2027_design_contract" / name, previous / name
        )
    paths = list((ROOT / "src/policytwin/typed_revision").glob("*.py"))
    paths += [
        Path(__file__),
        ROOT / "tests/test_cscwd_typed_revision.py",
        ROOT / "uv.lock",
        ROOT / "pyproject.toml",
        ROOT / "AGENTS.md",
    ]
    paths += [p for p in CONFIG.rglob("*") if p.is_file()]
    with zipfile.ZipFile(
        CONFIG / "typed_revision_source.zip", "x", zipfile.ZIP_DEFLATED
    ) as archive:
        for path in sorted(set(paths)):
            archive.write(path, str(path.relative_to(ROOT)))
    paths.append(CONFIG / "typed_revision_source.zip")
    save(
        CONFIG / "freeze.json",
        {
            "created_at_utc": datetime.now(UTC).isoformat(),
            "time_authority": "local pre-run ordering, not external attestation",
            "git_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip(),
            "working_tree": "dirty; exact source member hashes take precedence",
            "python": platform.python_version(),
            "study": STUDY,
            "sessions": len(blocks),
            "requested_panels_per_method": 72,
            "fresh_reference_panels": 24,
            "parent_parallelism": 2,
            "members": {str(p.relative_to(ROOT)): sha(p) for p in sorted(set(paths))},
        },
    )
    print(
        "TYPED_REVISION_FROZEN", sha(CONFIG / "freeze.json"), len(protected), flush=True
    )


def run() -> None:
    verify()
    OUT.mkdir(parents=True, exist_ok=False)
    seal_hash = sha(CONFIG / "freeze.json")
    save(
        OUT / "run_started.json",
        {"created_at_utc": datetime.now(UTC).isoformat(), "freeze_sha256": seal_hash},
    )
    rows, failures = [], []
    with ProcessPoolExecutor(
        max_workers=2, mp_context=multiprocessing.get_context("spawn")
    ) as pool:
        futures = {
            pool.submit(execute_session, b, OUT, seal_hash): b["session"]["id"]
            for b in read(CONFIG / "schedule.json")
        }
        for future in as_completed(futures):
            try:
                rows.append(future.result())
            except Exception as error:
                failure = {
                    "session": futures[future],
                    "error": f"{type(error).__name__}: {error}",
                }
                failures.append(failure)
                save(OUT / "session_failures" / (futures[future] + ".json"), failure)
            print("TYPED_SESSION", len(rows), len(failures), flush=True)
    save(
        OUT / "run_completed.json",
        {
            "completed_at_utc": datetime.now(UTC).isoformat(),
            "sessions": rows,
            "failures": failures,
        },
    )
    verify()
    if failures:
        raise RuntimeError("session failures retained; no retry is authorized")


def analyze() -> None:
    verify()
    completed = read(OUT / "run_completed.json")
    if completed["failures"] or len(completed["sessions"]) != 8:
        raise RuntimeError("cannot analyze an incomplete session schedule as complete")
    if (OUT / "summary.json").exists():
        raise FileExistsError("analysis already materialized; recompute elsewhere")
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
    print(
        json.dumps(
            {k: summary[k] for k in ("actual_paired_attempts", "typed", "opaque")}
        ),
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("operation", choices=("freeze", "check", "run", "analyze"))
    command = parser.parse_args().operation
    if command == "freeze":
        freeze()
    elif command == "run":
        run()
    elif command == "analyze":
        analyze()
    else:
        seal = verify()
        print(
            "TYPED_REVISION_INPUTS_UNCHANGED",
            sha(CONFIG / "freeze.json"),
            seal["sessions"],
            flush=True,
        )


if __name__ == "__main__":
    main()
