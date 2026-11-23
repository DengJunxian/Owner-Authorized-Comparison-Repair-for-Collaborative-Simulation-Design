"""Verify immutable registered evidence and recompute the registered claims."""

from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.derive_paired_execution_paper_artifacts import (  # noqa: E402
    _aggregate_modes,
    _fault_rows,
    _read_json,
    _scenario_rows,
    _validate_grid,
    _verify_artifact,
)


def _as_float(value: object) -> float:
    if not isinstance(value, str | bytes | bytearray | int | float):
        raise TypeError(f"expected a numeric JSON scalar, got {type(value).__name__}")
    return float(value)


def _as_int(value: object) -> int:
    if not isinstance(value, str | bytes | bytearray | int | float):
        raise TypeError(f"expected an integer JSON scalar, got {type(value).__name__}")
    return int(value)


def verify(root: Path, lock_path: Path) -> dict[str, object]:
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if lock.get("schema_version") != "policytwin.registered-evidence-lock.v1":
        raise RuntimeError("unsupported registered evidence lock schema")
    for relative, expected in lock["files"].items():
        payload = (root / relative).read_bytes()
        if len(payload) != expected["size_bytes"]:
            raise RuntimeError(f"registered evidence size differs: {relative}")
        if hashlib.sha256(payload).hexdigest() != expected["sha256"]:
            raise RuntimeError(f"registered evidence SHA-256 differs: {relative}")

    artifact, provenance = _verify_artifact(root)
    plan = _read_json(root / "plan.json")
    market_rows = _validate_grid(artifact, plan)
    coupling = _aggregate_modes(_scenario_rows(artifact, plan), artifact)
    faults = _fault_rows(artifact)
    observed_mismatch = {
        str(row["coupling_mode"]): _as_float(row["mean_shock_mismatch_rate"])
        for row in coupling
    }
    observed_ratios = {
        str(row["coupling_mode"]): _as_float(row["variance_sum_ratio_vs_event_keyed"])
        for row in coupling
    }
    claims = lock["claims"]
    if len(market_rows) != claims["market_row_count"]:
        raise RuntimeError("registered market row count differs")
    for field, observed in (
        ("coupling_mismatch_rates", observed_mismatch),
        ("variance_sum_ratios_vs_event_keyed", observed_ratios),
    ):
        expected = claims[field]
        if set(observed) != set(expected) or any(
            not math.isclose(observed[key], expected[key], rel_tol=0.0, abs_tol=1e-15)
            for key in observed
        ):
            raise RuntimeError(f"registered claim differs: {field}")
    detections = [_as_int(row["fully_detected_fault_classes"]) for row in faults]
    if detections != claims["complete_fault_detection_by_layer"]:
        raise RuntimeError("registered fault detection progression differs")
    if any(
        _as_int(row["clean_repetition_count"]) != claims["clean_controls_per_layer"]
        or _as_int(row["clean_false_positive_count"]) != 0
        for row in faults
    ):
        raise RuntimeError("registered clean-control claim differs")
    return {
        "coupling_mismatch_rates": observed_mismatch,
        "market_row_count": len(market_rows),
        "provenance": provenance,
        "registered_fault_detection": detections,
        "status": "VERIFIED",
        "variance_sum_ratios_vs_event_keyed": observed_ratios,
    }


def main() -> int:
    try:
        result = verify(
            Path("results/frozen/aaai27-paired-execution-v2"),
            Path("results/registered_evidence_lock.json"),
        )
    except (OSError, KeyError, TypeError, ValueError, RuntimeError) as error:
        print(f"REGISTERED_EVIDENCE_FAILED: {error}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
