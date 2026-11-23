"""Write-once freeze and separately bounded preparation, timing and execution."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import multiprocessing
import platform
import subprocess
import sys
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

from policytwin.revision_workload.analysis import summarize
from policytwin.revision_workload.design import STUDY, schedule
from policytwin.revision_workload.runtime import benchmark, physical_session, prepare
from policytwin.typed_revision.runtime import read
from scripts.cscwd2027_consent_projection import verify as previous_verify
from scripts.cscwd2027_design_contract.store import save

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/cscwd2027/revision_workload_v1"
OUT = ROOT / "results/cscwd2027/revision_workload_v1"
REPORTS = ROOT / "reports/cscwd2027/revision_workload_v1"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def local_module(name: str) -> Path | None:
    for parent in (ROOT / "src", ROOT):
        candidate = parent / name.replace(".", "/")
        for path in (candidate.with_suffix(".py"), candidate / "__init__.py"):
            if path.is_file():
                return path
    return None


def source_closure() -> list[Path]:
    """Conservative recursive local imports, including imports inside functions."""
    pending = [Path(__file__)] + list(
        (ROOT / "src/policytwin/revision_workload").glob("*.py")
    )
    pending += [ROOT / "tests/test_cscwd_revision_workload.py"]
    seen: set[Path] = set()
    while pending:
        path = pending.pop().resolve()
        if path in seen:
            continue
        seen.add(path)
        relative = path.relative_to(
            ROOT / "src" if path.is_relative_to(ROOT / "src") else ROOT
        )
        module = ".".join(relative.with_suffix("").parts)
        package = (
            module.removesuffix(".__init__")
            if path.name == "__init__.py"
            else module.rpartition(".")[0]
        )
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            names = []
            if isinstance(node, ast.Import):
                names = [n.name for n in node.names]
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    base = package.split(".")[
                        : len(package.split(".")) - node.level + 1
                    ]
                    full = ".".join([*base, *([node.module] if node.module else [])])
                else:
                    full = node.module or ""
                names = [full] + [
                    full + "." + n.name for n in node.names if n.name != "*"
                ]
            for name in names:
                for size in range(1, len(name.split(".")) + 1):
                    found = local_module(".".join(name.split(".")[:size]))
                    if found is not None and found not in seen:
                        pending.append(found)
    # Imported modules complement static resolution of dynamic package loading.
    for module_object in list(sys.modules.values()):
        filename = getattr(module_object, "__file__", None)
        if filename:
            path = Path(filename).resolve()
            if (
                path.suffix == ".py"
                and path.is_relative_to(ROOT)
                and ".venv" not in path.parts
            ):
                seen.add(path)
    return sorted(seen)


def verify() -> dict[str, Any]:
    previous_verify()
    frozen: dict[str, Any] = read(CONFIG / "freeze.json")
    for name, digest in frozen["members"].items():
        if not (ROOT / name).is_file() or sha(ROOT / name) != digest:
            raise RuntimeError("revision workload sealed source differs: " + name)
    for name, digest in read(CONFIG / "protected_inputs.json")["members"].items():
        if not (ROOT / name).is_file() or sha(ROOT / name) != digest:
            raise RuntimeError("prior protected artifact differs: " + name)
    if (OUT / "manifest.json").exists():
        for name, digest in read(OUT / "manifest.json")["members"].items():
            if not (OUT / name).is_file() or sha(OUT / name) != digest:
                raise RuntimeError("raw revision workload differs: " + name)
    return frozen


def freeze() -> None:
    previous_verify()
    if CONFIG.exists() or OUT.exists():
        raise FileExistsError("revision workload namespace already exists")
    tests_path = REPORTS / "preseal_tests.xml"
    tests = ET.parse(tests_path).getroot()
    count = sum(int(s.get("tests", 0)) for s in tests.iter("testsuite"))
    if not count or any(
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
    for tree in ("configs", "results", "release"):
        for path in sorted((ROOT / tree).rglob("*")):
            if (
                path.is_file()
                and not path.is_relative_to(CONFIG)
                and not path.is_relative_to(OUT)
            ):
                protected[str(path.relative_to(ROOT))] = sha(path)
    save(CONFIG / "protected_inputs.json", {"members": protected})
    paths = source_closure()
    save(
        CONFIG / "source_dependency_closure.json",
        {
            "members": {str(p.relative_to(ROOT)): sha(p) for p in paths},
            "third_party_environment": "uv.lock and current Python version",
        },
    )
    paths += [
        ROOT / "docs/cscwd2027/revision_workload.md",
        ROOT / "uv.lock",
        ROOT / "pyproject.toml",
        ROOT / "AGENTS.md",
    ]
    paths += [p for p in CONFIG.iterdir() if p.is_file()]
    with zipfile.ZipFile(
        CONFIG / "revision_workload_source.zip", "x", zipfile.ZIP_DEFLATED
    ) as archive:
        for path in sorted(set(paths)):
            archive.write(path, str(path.relative_to(ROOT)))
    paths.append(CONFIG / "revision_workload_source.zip")
    save(
        CONFIG / "freeze.json",
        {
            "created_at_utc": datetime.now(UTC).isoformat(),
            "time_authority": "local chronological record; not external attestation",
            "git_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip(),
            "working_tree": "dirty; exact member hashes authoritative",
            "python": platform.python_version(),
            "parent_freeze_sha256": sha(
                ROOT / "configs/cscwd2027/consent_projection_v1/freeze.json"
            ),
            "study": STUDY,
            "scientific_blocks": 4,
            "planning_contexts": 156,
            "route_measurements": 3900,
            "physical_requests": 16,
            "physical_method_panels": 48,
            "parent_parallelism": 2,
            "timing_parallelism": 1,
            "members": {str(p.relative_to(ROOT)): sha(p) for p in sorted(set(paths))},
        },
    )
    print(
        "REVISION_WORKLOAD_FROZEN",
        sha(CONFIG / "freeze.json"),
        len(protected),
        flush=True,
    )


def parallel_phase(name: str, function: Any) -> None:
    verify()
    phase_path = OUT / f"{name}_started.json"
    if phase_path.exists():
        raise FileExistsError("phase already attempted; no retries")
    save(phase_path, {"started_at_utc": datetime.now(UTC).isoformat()})
    successes, failures = [], []
    with ProcessPoolExecutor(
        max_workers=2, mp_context=multiprocessing.get_context("spawn")
    ) as pool:
        futures = {
            pool.submit(function, block, OUT, sha(CONFIG / "freeze.json")): block[
                "session"
            ]["id"]
            for block in read(CONFIG / "schedule.json")
        }
        for future in as_completed(futures):
            try:
                successes.append(future.result())
            except Exception as error:
                failure = {
                    "session": futures[future],
                    "phase": name,
                    "error": f"{type(error).__name__}: {error}",
                }
                failures.append(failure)
                save(OUT / "failures" / f"{name}-{futures[future]}.json", failure)
            print(
                "REVISION_PHASE_PROGRESS",
                name,
                len(successes),
                len(failures),
                flush=True,
            )
    save(
        OUT / f"{name}_completed.json",
        {
            "completed_at_utc": datetime.now(UTC).isoformat(),
            "sessions": successes,
            "failures": failures,
        },
    )
    verify()
    if failures:
        raise RuntimeError("phase failures retained; no retry or replacement")


def prepare_run() -> None:
    verify()
    OUT.mkdir(parents=True, exist_ok=False)
    save(
        OUT / "run_started.json",
        {
            "started_at_utc": datetime.now(UTC).isoformat(),
            "freeze_sha256": sha(CONFIG / "freeze.json"),
        },
    )
    parallel_phase("prepare", prepare)


def benchmark_run() -> None:
    verify()
    if read(OUT / "prepare_completed.json")["failures"]:
        raise RuntimeError("preparation incomplete")
    if (OUT / "benchmark_started.json").exists():
        raise FileExistsError("timing already attempted")
    save(
        OUT / "benchmark_started.json",
        {
            "started_at_utc": datetime.now(UTC).isoformat(),
            "system": platform.system(),
            "architecture": platform.machine(),
            "python": platform.python_version(),
            "warmup": "none",
            "parallelism": 1,
        },
    )
    sessions, failures = [], []
    for block in read(CONFIG / "schedule.json"):
        try:
            sessions.append(benchmark(block, OUT))
        except Exception as error:
            failure = {
                "session": block["session"]["id"],
                "error": f"{type(error).__name__}: {error}",
            }
            failures.append(failure)
            save(OUT / "failures" / f"benchmark-{block['session']['id']}.json", failure)
    save(
        OUT / "benchmark_completed.json",
        {
            "completed_at_utc": datetime.now(UTC).isoformat(),
            "sessions": sessions,
            "failures": failures,
        },
    )
    verify()
    if failures:
        raise RuntimeError("timing failures retained")


def execute_run() -> None:
    if read(OUT / "benchmark_completed.json")["failures"]:
        raise RuntimeError("planning incomplete")
    parallel_phase("execute", physical_session)
    save(
        OUT / "run_completed.json", {"completed_at_utc": datetime.now(UTC).isoformat()}
    )


def analyze(destination: Path | None = None) -> None:
    frozen = verify()
    chronology = [
        frozen["created_at_utc"],
        read(OUT / "run_started.json")["started_at_utc"],
    ]
    for phase in ("prepare", "benchmark", "execute"):
        chronology += [
            read(OUT / f"{phase}_started.json")["started_at_utc"],
            read(OUT / f"{phase}_completed.json")["completed_at_utc"],
        ]
    chronology += [read(OUT / "run_completed.json")["completed_at_utc"]]
    parsed = [datetime.fromisoformat(t) for t in chronology]
    if any(a >= b for a, b in zip(parsed, parsed[1:], strict=False)) or read(
        OUT / "run_started.json"
    )["freeze_sha256"] != sha(CONFIG / "freeze.json"):
        raise RuntimeError("pre-outcome chronology or freeze binding differs")
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
    print(
        json.dumps(
            {
                k: summary[k]
                for k in (
                    "planning_contexts",
                    "route_measurements",
                    "physical_panels",
                    "methods",
                    "physical_methods",
                    "actual_pairs",
                    "failed_attempts",
                    "stale_approvals_admitted",
                )
            }
        ),
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "operation",
        choices=("freeze", "check", "prepare", "benchmark", "execute", "analyze"),
    )
    parser.add_argument("--destination", type=Path)
    args = parser.parse_args()
    operations = {
        "freeze": freeze,
        "prepare": prepare_run,
        "benchmark": benchmark_run,
        "execute": execute_run,
    }
    if args.operation in operations:
        operations[args.operation]()
    elif args.operation == "analyze":
        analyze(args.destination)
    else:
        frozen = verify()
        print(
            "REVISION_WORKLOAD_UNCHANGED",
            sha(CONFIG / "freeze.json"),
            len(frozen["members"]),
        )


if __name__ == "__main__":
    main()
