"""All-record descriptive analysis; timing repetitions are technical repeats."""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from statistics import median, quantiles
from typing import Any

from policytwin.coordination.schema import commitment
from policytwin.revision_workload.design import METHODS, PHYSICAL_METHODS
from policytwin.typed_revision.runtime import read
from scripts.cscwd2027_design_contract.store import save


def describe(numbers: list[int | float]) -> dict[str, Any]:
    if not numbers:
        return {
            "n": 0,
            "median": None,
            "min": None,
            "max": None,
            "q1": None,
            "q3": None,
        }
    q = (
        quantiles(numbers, n=4, method="inclusive")
        if len(numbers) > 1
        else [numbers[0]] * 3
    )
    return {
        "n": len(numbers),
        "median": median(numbers),
        "min": min(numbers),
        "max": max(numbers),
        "q1": q[0],
        "q3": q[2],
    }


def summarize(root: Path, destination: Path) -> dict[str, Any]:
    planning, timing, physical, replays = [], [], [], []
    selection_by_context: dict[tuple[Any, ...], str] = {}
    repeat_disagreements = []
    for path in sorted((root / "planning").glob("*.json")):
        record = read(path)
        reference = record["methods"]["exhaustive"]["selected"]
        for method, result in record["methods"].items():
            selected = result["selected"]
            row = {
                k: record[k]
                for k in (
                    "session",
                    "agents",
                    "regime",
                    "domain_size",
                    "episode",
                    "stage",
                )
            }
            row.update(
                method=method,
                agreement=selected is not None,
                target_id=selected["target_id"] if selected else None,
                target=selected["target"] if selected else None,
                new_pairs=selected["new_pairs"] if selected else None,
                reevaluations=selected["reevaluations"] if selected else None,
                reused_evaluations=selected["reused_evaluations"] if selected else None,
                exact_selection_and_work=selected == reference,
                false_abstention=selected is None and reference is not None,
                false_agreement=selected is not None and reference is None,
                request_sha256=record["request"]["request_sha256"],
                catalog_sha256=record["catalog_sha256"],
                **result["counts"],
            )
            planning.append(row)
            key = (
                record["session"],
                record["domain_size"],
                record["episode"],
                record["stage"],
                method,
            )
            selection_by_context[key] = commitment(selected)
    for path in sorted((root / "timing").glob("*.json")):
        record = read(path)
        for method, result in record["methods"].items():
            row = {
                k: record[k]
                for k in (
                    "session",
                    "agents",
                    "regime",
                    "domain_size",
                    "episode",
                    "stage",
                    "repeat",
                )
            }
            row.update(
                method=method,
                position=record["method_order"].index(method),
                agreement=result["selected"] is not None,
                cold_trajectory_compilations=result["counts"][
                    "cold_trajectory_compilations"
                ],
                cold_evaluation_compilations=result["counts"][
                    "cold_evaluation_compilations"
                ],
                **result["timing"],
            )
            row["counts"] = result["counts"]
            row["timing"] = result["timing"]
            row["cold_dependency_route"] = bool(
                row["cold_trajectory_compilations"]
                or row["cold_evaluation_compilations"]
            )
            timing.append(row)
            key = (
                record["session"],
                record["domain_size"],
                record["episode"],
                record["stage"],
                method,
            )
            if selection_by_context.get(key) != commitment(result["selected"]):
                repeat_disagreements.append(
                    {**row, "source": str(path.relative_to(root))}
                )
    for path in sorted((root / "handoffs").glob("*/result.json")):
        physical.append(read(path))
    for path in sorted((root / "approval_replays").glob("*.json")):
        replays.extend(read(path))
    methods = {}
    for method in METHODS:
        rows = [r for r in planning if r["method"] == method]
        calls = [
            r["search_predicate_calls"]
            for r in rows
            if r["search_predicate_calls"] is not None
        ]
        methods[method] = {
            "contexts": len(rows),
            "agreements": sum(r["agreement"] for r in rows),
            "no_agreement": sum(not r["agreement"] for r in rows),
            "exact_selection_and_work": sum(
                r["exact_selection_and_work"] for r in rows
            ),
            "false_abstention": sum(r["false_abstention"] for r in rows),
            "false_agreement": sum(r["false_agreement"] for r in rows),
            "search_predicate_calls": sum(calls) if calls else None,
            "projection_calls": sum(r["projection_calls"] for r in rows),
            "selection_predicate_calls": sum(
                r["selection_predicate_calls"] for r in rows
            ),
            "negative_hits": sum(r["negative_hits"] for r in rows),
            "positive_hits": sum(r["positive_hits"] for r in rows),
        }
    timing_groups = []
    balanced_timings = []
    for size in (32, 256, 1024):
        for method in METHODS:
            rows = [
                r for r in timing if r["domain_size"] == size and r["method"] == method
            ]
            warm = [r for r in rows if not r["cold_dependency_route"]]
            timing_groups.append(
                {
                    "domain_size": size,
                    "method": method,
                    "route_wall_ns": describe([r["wall_ns"] for r in rows]),
                    "route_cpu_ns": describe([r["cpu_ns"] for r in rows]),
                    "warm_route_wall_ns": describe([r["wall_ns"] for r in warm]),
                    "warm_route_cpu_ns": describe([r["cpu_ns"] for r in warm]),
                    "cold_routes": sum(r["cold_dependency_route"] for r in rows),
                    "cold_trajectory_compilations": sum(
                        r["cold_trajectory_compilations"] for r in rows
                    ),
                    "cold_evaluation_compilations": sum(
                        r["cold_evaluation_compilations"] for r in rows
                    ),
                    "phases_wall_ns": {
                        name: describe([r["phases"][name]["wall_ns"] for r in rows])
                        for name in rows[0]["phases"]
                    }
                    if rows
                    else {},
                }
            )
            repeat_totals = []
            for repeat in range(5):
                measured = [r for r in rows if r["repeat"] == repeat]
                if not measured:
                    continue
                repeat_totals.append(
                    {
                        "repeat": repeat,
                        "contexts": len(measured),
                        "wall_ns_per_context": sum(r["wall_ns"] for r in measured)
                        / len(measured),
                        "cpu_ns_per_context": sum(r["cpu_ns"] for r in measured)
                        / len(measured),
                        "search_predicate_calls": None
                        if method == "complete_sql"
                        else sum(
                            r["counts"]["search_predicate_calls"] for r in measured
                        ),
                        "phases_wall_ns_per_context": {
                            name: sum(r["phases"][name]["wall_ns"] for r in measured)
                            / len(measured)
                            for name in measured[0]["phases"]
                        },
                    }
                )
            balanced_timings.append(
                {
                    "domain_size": size,
                    "method": method,
                    "repeat_totals": repeat_totals,
                    "wall_ns_per_context": describe(
                        [r["wall_ns_per_context"] for r in repeat_totals]
                    ),
                    "cpu_ns_per_context": describe(
                        [r["cpu_ns_per_context"] for r in repeat_totals]
                    ),
                }
            )
    physical_methods = {}
    for method in PHYSICAL_METHODS:
        rows = [r for r in physical if r["method"] == method]
        physical_methods[method] = {
            "panels": len(rows),
            "released": sum(r["authorized"] for r in rows),
            "no_agreement": sum(r["status"] == "no_agreement" for r in rows),
            "exact_panels": sum(r["exact"] is True for r in rows),
            **{
                key: sum(r[key] for r in rows)
                for key in (
                    "new_pairs",
                    "reevaluations",
                    "reused_evaluations",
                    "compared_slots",
                    "matching_slots",
                    "authorization_predicate_calls",
                )
            },
        }
    attempts = []
    for path in sorted((root / "objects").glob("*/result.json")):
        raw = read(path)
        attempts.append(
            {
                "id": raw["id"],
                "status": raw["status"],
                "local_pass": bool(raw.get("local_pass", False)),
                "elapsed_seconds": raw.get("elapsed_seconds"),
                "timing_seconds": raw.get("timing_seconds"),
                "error": raw.get("error"),
            }
        )
    episode_groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in planning:
        episode_groups[
            row["domain_size"], row["episode"], row["stage"], row["method"]
        ].append(row)
    episodes = [
        {
            "domain_size": key[0],
            "episode": key[1],
            "stage": key[2],
            "method": key[3],
            "contexts": len(rows),
            "agreements": sum(r["agreement"] for r in rows),
            "search_predicate_calls": None
            if key[3] == "complete_sql"
            else sum(r["search_predicate_calls"] for r in rows),
            "negative_hits": sum(r["negative_hits"] for r in rows),
            "positive_hits": sum(r["positive_hits"] for r in rows),
            "work_if_selected": [
                {
                    k: row[k]
                    for k in (
                        "session",
                        "new_pairs",
                        "reevaluations",
                        "reused_evaluations",
                        "target",
                    )
                }
                for row in rows
            ],
        }
        for key, rows in sorted(episode_groups.items())
    ]
    summary = {
        "schema": "revision-workload-summary.v1",
        "expected_planning_contexts": 156,
        "planning_contexts": len(planning) // len(METHODS),
        "expected_route_measurements": 3900,
        "route_measurements": len(timing),
        "expected_physical_panels": 48,
        "physical_panels": len(physical),
        "methods": methods,
        "physical_methods": physical_methods,
        "timing_by_domain_method": timing_groups,
        "planning_by_episode_stage": episodes,
        "balanced_timing_by_domain_method": balanced_timings,
        "repeat_selection_disagreements": repeat_disagreements,
        "approval_replay_probes": len(replays),
        "stale_approvals_admitted": sum(r["stale_records_authorized"] for r in replays),
        "actual_pairs": len(attempts),
        "local_passes": sum(r["local_pass"] for r in attempts),
        "failed_attempts": sum(
            r["status"] != "completed" or not r["local_pass"] for r in attempts
        ),
        "physical_statuses": dict(Counter(r["status"] for r in physical)),
        "interpretation": {
            "timing_repetitions": "five descriptive technical repeats on one host",
            "domain_order": (
                "32 then256 then1024; immutable dependency cache retained, "
                "not independent cold scaling"
            ),
            "generic_memo": (
                "complete dependency annotations and both Boolean values, "
                "exact approval always renewed"
            ),
            "scope": (
                "software role protocol over the same synthetic CLOB; "
                "no human coordination measurement"
            ),
        },
    }
    for name, rows in (
        ("planning_rows", planning),
        ("timing_rows", timing),
        ("physical_rows", physical),
        ("approval_replay_rows", replays),
        ("attempt_rows", attempts),
    ):
        save(destination / (name + ".json"), rows)
    save(destination / "summary.json", summary)
    return summary
