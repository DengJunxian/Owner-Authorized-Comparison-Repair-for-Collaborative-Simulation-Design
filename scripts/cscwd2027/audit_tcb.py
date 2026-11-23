"""Post-freeze engineering audit; does not alter experimental analysis or data."""

from __future__ import annotations

import ast
import importlib.util
import json
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    """Report actual source sizes, JUnit counts, and unavailable test instruments."""
    components = {
        "C/P compiler": ["coordination/plan.py", "counterfactual/coupling_contract.py"],
        "Canonical serialization": ["provenance.py", "coordination/schema.py"],
        "Event-key mapping": ["counterfactual/coupling_contract.py"],
        "Worker and coordinator": [
            "coordination/worker_runtime.py",
            "coordination/coordinator.py",
        ],
        "Receipt auditor": ["coordination/auditor.py"],
        "Actual replay": ["coordination/replay.py", "coordination/workload.py"],
        "Evidence gate/reporter": ["coordination/release_gate.py"],
    }
    rows = []
    for name, members in components.items():
        loc = statements = 0
        for member in members:
            text = (ROOT / "src/policytwin" / member).read_text()
            loc += sum(
                bool(line.strip()) and not line.lstrip().startswith("#")
                for line in text.splitlines()
            )
            statements += sum(
                isinstance(node, ast.stmt) for node in ast.walk(ast.parse(text))
            )
        rows.append(
            {
                "component": name,
                "members": members,
                "nonblank_noncomment_lines": loc,
                "ast_statements": statements,
            }
        )
    suites = ET.parse(ROOT / "reports/cscwd2027/pre_freeze_tests.xml").getroot()
    counts = {
        key: sum(int(s.get(key, "0")) for s in suites.iter("testsuite"))
        for key in ("tests", "failures", "errors", "skipped")
    }
    own = list((ROOT / "tests/cscwd2027").glob("test_*.py"))
    properties = sum("@given(" in p.read_text() for p in own)
    report = {
        "components": rows,
        "junit": counts,
        "source_loc_definition": (
            "nonblank physical lines excluding full-line comments; "
            "overlaps not additive"
        ),
        "property_test_files": properties,
        "coverage_tool_present": importlib.util.find_spec("coverage") is not None,
        "statement_coverage": "unavailable",
        "branch_coverage": "unavailable",
        "software_mutation_score": (
            "unavailable; runtime fault detection is a separate measure"
        ),
    }
    dest = ROOT / "reports/cscwd2027"
    (dest / "tcb_audit.json").write_text(json.dumps(report, indent=2) + "\n")
    lines = [
        "# Trusted computing base audit",
        "",
        f"JUnit result: {counts}.",
        "",
        "Coverage and software mutation score: unavailable; not inferred from faults.",
        "LOC excludes blank lines and full-line comments. Component sizes overlap.",
        "",
        "| Component | LOC | AST statements |",
        "|---|---:|---:|",
    ]
    lines += [
        f"| {r['component']} | {r['nonblank_noncomment_lines']} | "
        f"{r['ast_statements']} |"
        for r in rows
    ]
    lines += [
        "",
        "Tests cover exact schemas, real processes/IPC, independent task sets, views,",
        "event scopes, faults, actual replay, reporting files, worker crash cleanup,",
        "selection-bias counterexamples, frozen grids and manual statistics fixtures.",
        "Hypothesis checks the new view-policy adapter against the existing rule.",
        "This is same-repository validation, not independent implementation evidence.",
    ]
    (dest / "tcb_audit.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"junit": counts, "components": len(rows)}))


if __name__ == "__main__":
    main()
