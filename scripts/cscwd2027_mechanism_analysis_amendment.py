"""Pre-analysis schema amendment: derive absent old session fields by case ID.

No original source/result or mechanism-study seal is edited. The originally
sealed analysis is reused with an in-memory legacy-schema view; numerical
observations, metrics, references, endpoints and simulation schedule are fixed.
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch

from policytwin.cscwd_mechanism import analysis
from scripts.cscwd2027_design_contract.store import save
from scripts.cscwd2027_mechanism_review import (
    CONFIG,
    OUT,
    ROOT,
    protect_check,
    sha,
    verify,
)

SEAL = CONFIG / "analysis_schema_amendment.json"
PUBLICATION = ROOT / "results/cscwd2027/review_consequences_v2/publication_summary.json"
HANDOFFS = ROOT / "configs/cscwd2027/design_contract_v1/handoffs.json"


def linked_view(data, handoffs):
    """Add only the independently declared session label, never a numeric field."""
    mapping = {row["id"]: row["session"] for row in handoffs}
    old = data["artifact_analysis"]["original_rows"]
    linked = []
    for row in old:
        session = mapping[row["case_id"]]
        if "session" in row and row["session"] != session:
            raise RuntimeError("old session label conflicts with original schedule")
        linked.append({**row, "session": session})
    return {
        **data,
        "artifact_analysis": {**data["artifact_analysis"], "original_rows": linked},
    }


def check():
    verify()
    seal = analysis.read(SEAL)
    for path, digest in seal["members"].items():
        if sha(ROOT / path) != digest:
            raise RuntimeError("pre-analysis schema amendment differs")
    return seal


def freeze():
    verify()
    if (OUT / "analysis").exists():
        raise RuntimeError("schema amendment must precede scientific analysis outputs")
    before = analysis.read(PUBLICATION)
    after = linked_view(before, analysis.read(HANDOFFS))
    assert all(
        {k: v for k, v in b.items() if k != "session"} == a
        for a, b in zip(
            before["artifact_analysis"]["original_rows"],
            after["artifact_analysis"]["original_rows"],
            strict=True,
        )
    )
    paths = [
        Path(__file__),
        ROOT / "tests/test_cscwd_mechanism_schema_amendment.py",
        PUBLICATION,
        HANDOFFS,
        CONFIG / "freeze.json",
    ]
    save(
        SEAL,
        {
            "created_at_utc": datetime.now(UTC).isoformat(),
            "reason": (
                "legacy original_rows omit session; "
                "join case_id to original sealed handoff schedule"
            ),
            "timing": (
                "before mechanism analysis; simulation protocol and source unchanged"
            ),
            "changes_numeric_fields": False,
            "added_simulation_pairs": 0,
            "changed_endpoints": False,
            "members": {str(p.relative_to(ROOT)): sha(p) for p in paths},
        },
    )
    print("MECHANISM_SCHEMA_AMENDMENT_FROZEN", sha(SEAL), flush=True)


def run():
    check()
    if not (OUT / "run_completed.json").exists():
        raise RuntimeError("scheduled paired execution is incomplete")
    raw_read = analysis.read
    handoffs = raw_read(HANDOFFS)

    def compatible_read(path):
        value = raw_read(path)
        return linked_view(value, handoffs) if path == PUBLICATION else value

    coupling = analysis.coupling_analysis(
        OUT,
        raw_read(CONFIG / "contexts.json"),
        raw_read(CONFIG / "attempts.json"),
        raw_read(CONFIG / "study.json"),
    )
    repair = analysis.repair_analysis(ROOT, OUT)
    with patch.object(analysis, "read", compatible_read):
        auxiliary = analysis.auxiliary_analysis(ROOT, OUT)
    save(
        OUT / "summary.json",
        {
            "freeze_sha256": sha(CONFIG / "freeze.json"),
            "analysis_schema_amendment_sha256": sha(SEAL),
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
    check()
    print(
        "MECHANISM_ANALYSIS_COMPLETE", coupling["pairs"], repair["panels"], flush=True
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("operation", choices=("freeze", "check", "run"))
    operation = parser.parse_args().operation
    if operation == "freeze":
        freeze()
    elif operation == "run":
        run()
    else:
        print(json.dumps(check()))
