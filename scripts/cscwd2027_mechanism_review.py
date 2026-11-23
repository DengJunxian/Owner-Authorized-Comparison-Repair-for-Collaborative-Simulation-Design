"""Seal, execute and analyze a separate post-review CSCWD mechanism study."""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing
import platform
import shutil
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path

from policytwin.cscwd_mechanism.analysis import (
    auxiliary_analysis,
    coupling_analysis,
    read,
    repair_analysis,
)
from policytwin.cscwd_mechanism.coupling import execute
from policytwin.cscwd_mechanism.design import design
from scripts.cscwd2027_design_contract.freeze import check as original_check
from scripts.cscwd2027_design_contract.store import save

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/cscwd2027/mechanism_review_v1"
OUT = ROOT / "results/cscwd2027/mechanism_review_v1"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify():
    original_check()
    seal = read(CONFIG / "freeze.json")
    for name, expected in seal["members"].items():
        if sha(ROOT / name) != expected:
            raise RuntimeError(f"sealed mechanism input differs: {name}")
    return seal


def protect_check():
    values = read(CONFIG / "protected_inputs.json")["members"]
    changed = [
        p for p, h in values.items() if not (ROOT / p).is_file() or sha(ROOT / p) != h
    ]
    if changed:
        raise RuntimeError(f"prior scientific inputs changed: {changed[:8]}")
    return {"unchanged": True, "files": len(values)}


def freeze():
    original_check()
    if (CONFIG / "freeze.json").exists():
        raise FileExistsError("mechanism study is already sealed")
    contexts, attempts = design(
        read(ROOT / "configs/cscwd2027/design_contract_v1/sessions.json")
    )
    if len(contexts) != 52 or len(attempts) != 1632:
        raise RuntimeError("protocol allocation differs from the predeclared design")
    save(CONFIG / "contexts.json", contexts)
    save(CONFIG / "attempts.json", attempts)
    preserved = []
    for tree in (
        "configs/cscwd2027/design_contract_v1",
        "results/cscwd2027/design_contract_v1",
        "results/cscwd2027/review_response_v1",
        "results/cscwd2027/review_consequences_v2",
        "release/cscwd2027/review_20261025_design_contract_v1",
    ):
        preserved.extend(
            p
            for p in (ROOT / tree).rglob("*")
            if p.is_file() and "__pycache__" not in p.parts
        )
    # Preserve previous protocols/programs, including failed recovery sources.
    preserved.extend(
        p for p in (ROOT / "configs/cscwd2027").glob("review*.json") if p.is_file()
    )
    preserved.extend(
        p for p in (ROOT / "scripts").glob("cscwd2027_review*.py") if p.is_file()
    )
    preserved.extend(
        p for p in (ROOT / "tests").glob("test_cscwd_review*.py") if p.is_file()
    )
    save(
        CONFIG / "protected_inputs.json",
        {"members": {str(p.relative_to(ROOT)): sha(p) for p in sorted(set(preserved))}},
    )
    archive = CONFIG / "previous_author_review"
    archive.mkdir(exist_ok=False)
    for name in (
        "source_for_author_review.zip",
        "main.pdf",
        "technical_supplement.pdf",
    ):
        shutil.copyfile(ROOT / "paper/cscwd2027_design_contract" / name, archive / name)
    paths = list((ROOT / "src/policytwin/cscwd_mechanism").glob("*.py"))
    paths += [
        Path(__file__),
        ROOT / "tests/test_cscwd_mechanism_review.py",
        ROOT / "uv.lock",
        ROOT / "pyproject.toml",
        ROOT / "AGENTS.md",
    ]
    paths += [p for p in CONFIG.rglob("*") if p.is_file()]
    with zipfile.ZipFile(
        CONFIG / "mechanism_source.zip", "x", zipfile.ZIP_DEFLATED
    ) as z:
        for p in sorted(set(paths)):
            z.write(p, str(p.relative_to(ROOT)))
    paths.append(CONFIG / "mechanism_source.zip")
    save(
        CONFIG / "freeze.json",
        {
            "created_at_utc": datetime.now(UTC).isoformat(),
            "time_authority": "local pre-run ordering, not external attestation",
            "python": platform.python_version(),
            "parent_parallelism": 2,
            "study": read(CONFIG / "study.json"),
            "contexts": len(contexts),
            "pairs": len(attempts),
            "members": {str(p.relative_to(ROOT)): sha(p) for p in sorted(set(paths))},
        },
    )
    print("MECHANISM_FROZEN", sha(CONFIG / "freeze.json"), len(attempts), flush=True)


def run():
    verify()
    protect_check()
    OUT.mkdir(parents=True, exist_ok=False)
    save(
        OUT / "run_started.json",
        {
            "created_at_utc": datetime.now(UTC).isoformat(),
            "freeze_sha256": sha(CONFIG / "freeze.json"),
            "parent_parallelism": 2,
        },
    )
    attempts = read(CONFIG / "attempts.json")
    results = []
    with ProcessPoolExecutor(
        max_workers=2, mp_context=multiprocessing.get_context("spawn")
    ) as pool:
        tasks = {
            pool.submit(execute, row, OUT, sha(CONFIG / "freeze.json")): row["id"]
            for row in attempts
        }
        for future in as_completed(tasks):
            result = future.result()
            results.append(
                {
                    k: result[k]
                    for k in ("id", "status", "local_pass", "elapsed_seconds")
                }
            )
            if len(results) % 24 == 0 or not result["local_pass"]:
                print(
                    "MECHANISM_PAIR",
                    len(results),
                    len(attempts),
                    result["id"],
                    result["status"],
                    result["local_pass"],
                    flush=True,
                )
    save(
        OUT / "run_completed.json",
        {"finished_at_utc": datetime.now(UTC).isoformat(), "rows": results},
    )
    verify()
    print("MECHANISM_RUN_COMPLETE", len(results), flush=True)


def analyze():
    verify()
    if not (OUT / "run_completed.json").exists():
        raise RuntimeError(
            "all scheduled attempts must finish before mechanism analysis"
        )
    config = read(CONFIG / "study.json")
    coupling = coupling_analysis(
        OUT, read(CONFIG / "contexts.json"), read(CONFIG / "attempts.json"), config
    )
    repair = repair_analysis(ROOT, OUT)
    auxiliary = auxiliary_analysis(ROOT, OUT)
    save(
        OUT / "summary.json",
        {
            "freeze_sha256": sha(CONFIG / "freeze.json"),
            "coupling": coupling,
            "repair": repair,
            "auxiliary": auxiliary,
            "protection": protect_check(),
        },
    )
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
    verify()
    print(
        "MECHANISM_ANALYSIS_COMPLETE", coupling["pairs"], repair["panels"], flush=True
    )


def main():
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
        print(
            json.dumps(
                {
                    "freeze_sha256": sha(CONFIG / "freeze.json"),
                    "seal": verify()["pairs"],
                    "protection": protect_check(),
                }
            )
        )


if __name__ == "__main__":
    main()
