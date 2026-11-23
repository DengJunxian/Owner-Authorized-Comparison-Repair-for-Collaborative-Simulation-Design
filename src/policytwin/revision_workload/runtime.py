"""Actual evidence arrivals, counterbalanced planners, and paired acquisitions."""

from __future__ import annotations

from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter_ns
from typing import Any

from policytwin.coordination.schema import commitment
from policytwin.negotiated_repair.runtime import new_pair
from policytwin.revision_workload.design import (
    INITIAL,
    METHODS,
    PHYSICAL_METHODS,
    STUDY,
    Target,
    recipes,
    slots,
)
from policytwin.revision_workload.planner import (
    DependencyCache,
    SemanticMemo,
    build_work,
    catalog_identity,
    declared_contract,
    plan,
)
from policytwin.revision_workload.policy import authorized, compile_policies
from policytwin.typed_revision.contract import (
    EvaluationToken,
    TraceToken,
    decide,
    trajectory_key,
)
from policytwin.typed_revision.metrics import MetricSpec, evaluation_key
from policytwin.typed_revision.runtime import (
    acquired_trace,
    checked_evaluation,
    expectations,
    read,
    trace_matches,
)
from scripts.cscwd2027_design_contract.store import save


def measured(
    trace: TraceToken, capsule: dict[str, Any], metric: MetricSpec, path: Path
) -> EvaluationToken:
    if trace.local_pass:
        return checked_evaluation(trace, capsule, metric, path)
    return EvaluationToken(
        "rejected-" + trace.object_id,
        *trace.slot,
        trace.trajectory_key,
        evaluation_key(trace.trajectory_key, metric),
        trace.capsule_sha256,
        trace.plan_sha256,
        "",
        False,
        None,
    )


def prepare(block: dict[str, Any], root: Path, seal: str) -> dict[str, Any]:
    session = block["session"]
    initial, revised = {}, {}
    for slot, recipe in recipes(session, INITIAL).items():
        identifier = f"{session['id']}-initial-{slot[0]}-{slot[1]}"
        trace, capsule = new_pair(identifier, recipe, 2, root, seal)
        evaluation = measured(
            trace,
            capsule,
            INITIAL.metric,
            root / "derivations" / f"{identifier}-notional0.json",
        )
        v0 = measured(
            trace,
            capsule,
            MetricSpec("volume", 0),
            root / "derivations" / f"{identifier}-volume0.json",
        )
        v4 = measured(
            trace,
            capsule,
            MetricSpec("volume", 4),
            root / "derivations" / f"{identifier}-volume4.json",
        )
        initial[slot] = trace, evaluation, v0, v4
    changed = Target(2, "shared", "volume", 4, 2, 12)
    changed_recipes = recipes(session, changed)
    for slot in slots(session, 2):
        if slot[0] != session["target"]:
            continue
        identifier = f"{session['id']}-revised-{slot[0]}-{slot[1]}"
        trace, capsule = new_pair(identifier, changed_recipes[slot], 2, root, seal)
        evaluation = measured(
            trace,
            capsule,
            changed.metric,
            root / "derivations" / f"{identifier}-volume4.json",
        )
        revised[slot] = trace, evaluation
    catalogs = []
    for episode in block["episodes"]:
        for stage in episode["stages"]:
            spec = stage["catalog"]
            traces, evaluations = [], []
            for slot in slots(session, spec["base_per_branch"]):
                trace, base, v0, v4 = initial[slot]
                traces.append(trace)
                evaluations.append(base)
                if spec["unrelated_volume0"]:
                    evaluations.append(v0)
                if spec["base_volume4"]:
                    evaluations.append(v4)
            if spec["revised_epoch2"]:
                for trace, evaluation in revised.values():
                    traces.append(trace)
                    if spec["revised_volume4"]:
                        evaluations.append(evaluation)
            row = {
                "session": session,
                "episode": episode["episode"],
                "stage": stage["stage"],
                "traces": [asdict(t) for t in traces],
                "evaluations": [asdict(e) for e in evaluations],
                "catalog_sha256": catalog_identity(traces, evaluations),
                "recorded_at_utc": datetime.now(UTC).isoformat(),
            }
            name = f"{session['id']}-{episode['episode']}-{stage['stage']}"
            save(root / "catalogs" / (name + ".json"), row)
            catalogs.append(name)
    return {
        "session": session["id"],
        "initial_pairs": len(initial),
        "revised_pairs": len(revised),
        "catalogs": catalogs,
    }


def catalog(
    root: Path, session_id: str, episode: str, stage: int
) -> tuple[list[TraceToken], list[EvaluationToken]]:
    row = read(root / "catalogs" / f"{session_id}-{episode}-{stage}.json")
    traces = [TraceToken(**t) for t in row["traces"]]
    evaluations = [EvaluationToken(**e) for e in row["evaluations"]]
    if catalog_identity(traces, evaluations) != row["catalog_sha256"]:
        raise RuntimeError("catalog commitment differs")
    return traces, evaluations


def benchmark(block: dict[str, Any], root: Path) -> dict[str, Any]:
    """Serial measured routes: the executor and independent audits are outside."""
    session = block["session"]
    count = 0
    for repeat in range(STUDY["timing_repeats"]):
        caches = {m: DependencyCache() for m in METHODS}
        for size in (32, 256, 1024):
            for ei, episode in enumerate(block["episodes"]):
                memos = {m: SemanticMemo() for m in METHODS}
                for stage in episode["stages"]:
                    traces, evaluations = catalog(
                        root, session["id"], episode["episode"], stage["stage"]
                    )
                    shift = (repeat + stage["stage"] + ei) % len(METHODS)
                    order = METHODS[shift:] + METHODS[:shift]
                    results = {}
                    for method in order:
                        results[method] = plan(
                            session,
                            size,
                            stage["request"],
                            stage["policies"],
                            traces,
                            evaluations,
                            method,
                            caches[method],
                            memos[method],
                        )
                    selection_hashes = {
                        m: commitment(r["selected"]) for m, r in results.items()
                    }
                    row = {
                        "session": session["id"],
                        "agents": session["agents"],
                        "regime": session["regime"],
                        "domain_size": size,
                        "episode": episode["episode"],
                        "stage": stage["stage"],
                        "repeat": repeat,
                        "method_order": order,
                        "request": stage["request"],
                        "catalog_sha256": catalog_identity(traces, evaluations),
                        "methods": results,
                        "all_selection_and_work_equal": len(
                            set(selection_hashes.values())
                        )
                        == 1,
                        "recorded_at_utc": datetime.now(UTC).isoformat(),
                    }
                    identifier = (
                        f"{session['id']}-d{size}-{episode['episode']}"
                        f"-s{stage['stage']}-r{repeat}"
                    )
                    save(root / "timing" / (identifier + ".json"), row)
                    if repeat == 0:
                        save(
                            root
                            / "planning"
                            / (identifier.removesuffix("-r0") + ".json"),
                            row,
                        )
                    count += len(METHODS)
            print("REVISION_TIMING_DOMAIN", session["id"], repeat, size, flush=True)
    return {"session": session["id"], "route_measurements": count}


def replay_diagnostics(block: dict[str, Any], root: Path) -> list[dict[str, Any]]:
    """Reapply actual prior approval records to current work and current policies."""
    result = []
    session = block["session"]
    for size in (32, 256, 1024):
        cache = DependencyCache()
        for episode in block["episodes"]:
            previous: dict[str, Any] | None = None
            for stage in episode["stages"]:
                name = f"{session['id']}-d{size}-{episode['episode']}-s{stage['stage']}"
                record = read(root / "planning" / (name + ".json"))["methods"][
                    "persistent"
                ]
                traces, evaluations = catalog(
                    root, session["id"], episode["episode"], stage["stage"]
                )
                if previous is not None:
                    layouts = cache.build(session, size)
                    layout = next(
                        o
                        for o in layouts
                        if o["target_id"] == previous["selected"]["target_id"]
                    )
                    current = build_work([layout], traces, evaluations, False)[0]
                    current["contract"] = declared_contract(session, current)
                    current["waived"] = sorted(
                        f
                        for f, v in stage["request"]["proposed"].items()
                        if current["target"][f] != v
                    )
                    current_policies = compile_policies(stage["policies"])
                    admitted = authorized(
                        stage["request"],
                        record["catalog_sha256"],
                        current,
                        current_policies,
                        previous["approvals"],
                    )
                    result.append(
                        {
                            "session": session["id"],
                            "domain_size": size,
                            "episode": episode["episode"],
                            "stage": stage["stage"],
                            "old_approvals": previous["approvals"],
                            "old_target": previous["selected"]["target"],
                            "current_request": stage["request"]["request_sha256"],
                            "current_catalog": record["catalog_sha256"],
                            "current_policies": stage["policies"],
                            "current_work": current["work_order"],
                            "stale_records_authorized": admitted,
                            "current_target_selected": record["selected"] is not None,
                        }
                    )
                if record["selected"] is not None:
                    previous = record
    save(root / "approval_replays" / (session["id"] + ".json"), result)
    return result


def fresh_references(
    block: dict[str, Any], physical: list[dict[str, Any]], root: Path, seal: str
) -> dict[str, dict[str, Any]]:
    session = block["session"]
    grouped: dict[tuple[int, str, int], list[Target]] = {}
    for item in physical:
        for method in PHYSICAL_METHODS:
            selected = item["record"]["methods"][method]["selected"]
            if selected is not None:
                target = Target.from_mapping(selected["target"])
                group = grouped.setdefault(
                    (target.epoch, target.coupling, target.workers), []
                )
                if target not in group:
                    group.append(target)
    result = {}
    for index, (_, desired) in enumerate(sorted(grouped.items())):
        target = desired[0]
        traces = []
        evaluations: dict[str, list[dict[str, Any]]] = {t.identity: [] for t in desired}
        for slot, recipe in recipes(session, target).items():
            identifier = f"{session['id']}-reference{index}-{slot[0]}-{slot[1]}"
            trace, capsule = new_pair(identifier, recipe, target.workers, root, seal)
            traces.append(asdict(trace))
            for measured_target in desired:
                token = measured(
                    trace,
                    capsule,
                    measured_target.metric,
                    root
                    / "reference_derivations"
                    / f"{identifier}-{measured_target.identity}.json",
                )
                evaluations[measured_target.identity].append(asdict(token))
        for measured_target in desired:
            result[measured_target.identity] = {
                "traces": traces,
                "evaluations": evaluations[measured_target.identity],
            }
    save(root / "references" / (session["id"] + ".json"), result)
    return result


def acquire(
    session: dict[str, Any],
    item: dict[str, Any],
    method: str,
    reference: dict[str, Any] | None,
    root: Path,
    seal: str,
) -> dict[str, Any]:
    started = perf_counter_ns()
    stage, record = item["stage"], item["record"]
    request = stage["request"]
    episode = item["episode"]
    decision = record["methods"][method]
    selected = decision["selected"]
    identifier = f"{session['id']}-{episode}-s{stage['stage']}-{method}"
    folder = root / "handoffs" / identifier
    row: dict[str, Any] = {
        "id": identifier,
        "session": session["id"],
        "agents": session["agents"],
        "regime": session["regime"],
        "domain_size": 256,
        "episode": episode,
        "stage": stage["stage"],
        "method": method,
        "status": "no_agreement",
        "authorized": False,
        "exact": None,
        "new_pairs": 0,
        "reevaluations": 0,
        "reused_evaluations": 0,
        "compared_slots": 0,
        "matching_slots": 0,
        "authorization_predicate_calls": 0,
    }
    save(folder / "selection.json", decision)
    if selected is None:
        save(folder / "result.json", row)
        return row
    save(
        folder / "acquisition_started.json",
        {
            "started_at_utc": datetime.now(UTC).isoformat(),
            "approvals": decision["approvals"],
            "actions": selected["work_order"],
        },
    )
    compiled = compile_policies(stage["policies"])
    if not authorized(
        request, decision["catalog_sha256"], selected, compiled, decision["approvals"]
    ):
        raise RuntimeError("no exact current approval before acquisition")
    target = Target.from_mapping(selected["target"])
    declared = selected["contract"]
    expected = expectations(declared)
    desired = recipes(session, target)
    old_traces, old_evaluations = catalog(root, session["id"], episode, stage["stage"])
    current_expected = {
        slot: (
            trajectory_key(recipe),
            evaluation_key(trajectory_key(recipe), target.metric),
        )
        for slot, recipe in desired.items()
    }
    layout = {
        "target": target.to_mapping(),
        "target_id": target.identity,
        "expected": current_expected,
        "plan": declared["allowed_plans"][0],
    }
    current = build_work([layout], old_traces, old_evaluations, False)[0]
    if (
        declared_contract(session, layout) != declared
        or current["work_order"] != selected["work_order"]
    ):
        raise RuntimeError("current target, contract or work differs before execution")
    acquisition_start = perf_counter_ns()
    members, collected = [], []
    for action in selected["work_order"]:
        slot = action["candidate"], action["replicate"]
        if action["action"] == "execute":
            trace, capsule = new_pair(
                f"{identifier}-{slot[0]}-{slot[1]}",
                desired[slot],
                target.workers,
                root,
                seal,
            )
        else:
            if action["action"] == "reuse":
                stored = next(
                    e for e in old_evaluations if e.object_id == action["object_id"]
                )
                old = next(
                    t
                    for t in old_traces
                    if t.slot == stored.slot
                    and t.capsule_sha256 == stored.capsule_sha256
                )
            else:
                old = next(t for t in old_traces if t.object_id == action["object_id"])
            trace, capsule = acquired_trace(
                old.object_id, root, folder / "replay" / f"{slot[0]}-{slot[1]}"
            )
        token = measured(
            trace,
            capsule,
            target.metric,
            folder / "evaluations" / f"{slot[0]}-{slot[1]}.json",
        )
        if action["action"] == "reuse" and token != next(
            e for e in old_evaluations if e.object_id == action["object_id"]
        ):
            raise RuntimeError("stored evaluation changed on exact recomputation")
        members.append(token)
        collected.append(trace)
    numerical = decide(
        declared["contract_sha256"], expected, tuple(declared["allowed_plans"]), members
    )
    owner_ok = authorized(
        request,
        decision["catalog_sha256"],
        selected,
        compiled,
        read(folder / "selection.json")["approvals"],
    )
    released = bool(numerical["authorized"] and owner_ok)
    acquisition_end = perf_counter_ns()
    exact, matching, compared = None, 0, 0
    if reference is not None and released:
        ref_traces = {
            (t["candidate"], t["replicate"]): TraceToken(**t)
            for t in reference["traces"]
        }
        ref_members = [EvaluationToken(**e) for e in reference["evaluations"]]
        ref_decision = decide(
            declared["contract_sha256"],
            expected,
            tuple(declared["allowed_plans"]),
            ref_members,
        )
        ref_values = {e.slot: e.value for e in ref_members}
        if ref_decision["authorized"]:
            for trace, token in zip(collected, members, strict=True):
                _, observed = acquired_trace(trace.object_id, root)
                _, intended = acquired_trace(ref_traces[trace.slot].object_id, root)
                compared += 1
                matching += int(
                    trace_matches(observed, intended)
                    and token.value == ref_values[trace.slot]
                )
            exact = matching == compared and all(
                numerical["report"][k] == ref_decision["report"][k]
                for k in ("means", "top1", "top2")
            )
    row.update(
        status="released" if released else "execution_abstention",
        authorized=released,
        exact=exact,
        new_pairs=selected["new_pairs"],
        reevaluations=selected["reevaluations"],
        reused_evaluations=selected["reused_evaluations"],
        compared_slots=compared,
        matching_slots=matching,
        authorization_predicate_calls=6,
    )
    row["physical_timing"] = {
        "wall_ns": perf_counter_ns() - started,
        "preacquisition_validation_ns": acquisition_start - started,
        "acquisition_replay_derivation_gate_ns": acquisition_end - acquisition_start,
        "reference_comparison_ns": perf_counter_ns() - acquisition_end,
    }
    save(
        folder / "response.json",
        {
            "traces": [asdict(t) for t in collected],
            "evaluations": [asdict(e) for e in members],
        },
    )
    save(folder / "numerical_decision.json", numerical)
    save(folder / "result.json", row)
    print("REVISION_HANDOFF", identifier, row["status"], exact, flush=True)
    return row


def physical_session(block: dict[str, Any], root: Path, seal: str) -> dict[str, Any]:
    session = block["session"]
    physical = []
    for episode in block["episodes"]:
        for stage in episode["stages"]:
            if stage["physical"]:
                name = f"{session['id']}-d256-{episode['episode']}-s{stage['stage']}"
                physical.append(
                    {
                        "episode": episode["episode"],
                        "stage": stage,
                        "record": read(root / "planning" / (name + ".json")),
                    }
                )
    references = fresh_references(block, physical, root, seal)
    rows = []
    for item in physical:
        for method in PHYSICAL_METHODS:
            selected = item["record"]["methods"][method]["selected"]
            reference = (
                references.get(selected["target_id"]) if selected is not None else None
            )
            rows.append(acquire(session, item, method, reference, root, seal))
    replays = replay_diagnostics(block, root)
    return {
        "session": session["id"],
        "physical_panels": len(rows),
        "approval_replay_probes": len(replays),
    }
