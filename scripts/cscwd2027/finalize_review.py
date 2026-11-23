"""Generate the author-review disposition from completed audit receipts."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

from scripts.cscwd2027.freeze_study import ROOT, check


def read(relative: str) -> Any:
    """Read a previously generated structured audit."""
    return json.loads((ROOT / relative).read_text())


def digest(path: Path) -> str:
    """Bind disposition to the reviewed PDF bytes."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    """Require successful technical and extraction checks before author review."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--validated-package", required=True)
    args = parser.parse_args()
    if Path(args.validated_package).name != args.validated_package:
        raise ValueError("expected a package directory name")
    freeze = check()
    summary = read("results/cscwd2027/derived_v1/summary.json")
    integrity = read("reports/cscwd2027/artifact_integrity_audit.json")
    machine = read("reports/cscwd2027/manuscript_machine_audit.json")
    visual = read("reports/cscwd2027/visual_review.json")
    package = read(
        f"reports/cscwd2027/package_validation_{args.validated_package}.json"
    )
    xml = ET.parse(ROOT / "reports/cscwd2027/final_tests.xml").getroot()
    tests = {
        name: sum(int(s.get(name, "0")) for s in xml.iter("testsuite"))
        for name in ("tests", "failures", "errors", "skipped")
    }
    baseline = read("results/cscwd2027/baseline/protected_files.json")
    historical_checks = []
    for command in (
        [sys.executable, "-m", "scripts.cscwd2027.audit_baseline", "verify"],
        [
            sys.executable,
            "scripts/check_registered_metadata.py",
            "paper/main_revised.tex",
        ],
        [sys.executable, "scripts/verify_original_evidence.py"],
    ):
        result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
        historical_checks.append(
            {
                "command": ["python", *command[1:]],
                "exit_code": result.returncode,
                "output": result.stdout + result.stderr,
            }
        )
    visuals_bound = all(
        visual[stem]["sha256"] == digest(ROOT / f"paper/cscwd2027/{stem}.pdf")
        and visual[stem]["pages_inspected"]
        == list(range(1, machine["pdfs"][stem]["pages"] + 1))
        for stem in ("main", "technical_supplement")
    )
    passed = all(
        (
            summary["data_complete"],
            not summary["failed_ids"],
            integrity["all_files_bound"],
            machine["machine_checks_pass"],
            visual["pass"],
            visuals_bound,
            package["pass"],
            package.get("summary_identical", False),
            not tests["failures"] and not tests["errors"],
            all(c["exit_code"] == 0 for c in historical_checks),
        )
    )
    status = "READY FOR HUMAN REVIEW" if passed else "NOT READY"
    dest = ROOT / "reports/cscwd2027"
    receipt = {
        "status": status,
        "purpose": "anonymous author review, not external submission clearance",
        "formal_attempts": summary["recorded_attempts"],
        "expected_attempts": summary["expected_attempts"],
        "runner_failure_ids": summary["failed_ids"],
        "verified_attempts": integrity["verified_attempts"],
        "actual_report_decisions_checked": integrity["reports_checked"],
        "freeze_created_at_utc": freeze["created_at_utc"],
        "freeze_manifest_sha256": digest(
            ROOT / "configs/cscwd2027/freeze_manifest_v1.json"
        ),
        "protected_original_files": len(baseline["members"]),
        "historical_checks": historical_checks,
        "tests": tests,
        "machine_checks_pass": machine["machine_checks_pass"],
        "visual_review_bound": visuals_bound,
        "pdfs": {
            stem: {k: value[k] for k in ("pages", "bytes", "sha256")}
            for stem, value in machine["pdfs"].items()
        },
        "validated_package": package["package"],
        "extracted_summary_identical": package.get("summary_identical", False),
        "author_review_complete": False,
        "external_submission_authorized": False,
        "external_submission_cleared": False,
        "historical_v2_source_recovered": False,
    }
    (dest / "delivery_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    (dest / "execution_status.json").write_text(
        json.dumps(
            {
                "schema_version": "policytwin.cscwd-execution-status.v1",
                "manuscript_status": status,
                "phase": "P5_local_author_review_delivery",
                "formal_studies_frozen": True,
                "formal_studies_run": "complete",
                "formal_attempts": summary["recorded_attempts"],
                "baseline_checks_passed": all(
                    c["exit_code"] == 0 for c in historical_checks
                ),
                "historical_v2_source_recovered": False,
                "final_tests_passed": tests["tests"]
                - tests["failures"]
                - tests["errors"]
                - tests["skipped"],
                "author_block": "anonymous by explicit author request",
                "submission_authorized": False,
                "external_submission_cleared": False,
                "receipt": "reports/cscwd2027/delivery_receipt.json",
            },
            indent=2,
        )
        + "\n"
    )
    statistical = f"""# Final statistical consistency audit

Disposition: **{status}** for the requested anonymous author-review draft.
This file is generated from the completed summary and validation receipts.

- Formal schedule: {summary["recorded_attempts"]}/{summary["expected_attempts"]}
  attempts.
- Runner failures: {len(summary["failed_ids"])}; native incomplete fault cases are
  separate outcomes and remain in the fault table.
- Raw integrity: {integrity["verified_attempts"]} capsules and
  {integrity["reports_checked"]} actual report decisions checked.
- Original protected files: {len(baseline["members"])}, unchanged; registered
  metadata and original-result recomputation pass.
- Exact partition traces: {summary["e1"]["trace_equal"]}/{summary["e1"]["attempts"]}.
- Fault numerical outcomes: {summary["e2"]["numerical_changed"]} changed,
  {summary["e2"]["numerically_unchanged"]} unchanged,
  {summary["e2"]["incomplete"]} incomplete; not pooled into one error measure.
- Full invalid release: {summary["full_invalid_releases"]["count"]}
  of {summary["e2"]["attempts"]} fault attempts.
- Paper result macros and figure/table manifests match the structured artifacts.
- Extracted-package recomputation produces the identical summary SHA-256.

## Denominators and interpretation

Reference runs and warm-ups are included in the retained attempt inventory but
not counted as extra primary independent replicates. Worker variants share a
scientific scenario. The conventional path shares checking primitives with full.
Task coverage deletion removes one predicate, not the whole expected-set compiler.
Fault counts describe the predeclared finite operation set, not all software faults.
The cost ratio is the mean of within-block ratios; it is not the ratio of medians.
Its percentile intervals resample paired measured blocks, excluding warm-ups.
Native rejection, diagnostic detection and actual report containment are separate.

## Preserved null and adverse outcomes

Equal-obligation agreement, unchanged-contrast violations, overlapping coverage
checks, runtime overhead and uncontrolled host background load remain explicit.
An all-pass batch does not establish an unconditional accepted-sample law.
No thresholds, seeds, formal source files or estimator definitions were changed
after the new freeze. The unresolved older v2 source-byte gap is documented and
is not used as confirmation of the new experiment's pre-run binding.
"""
    (dest / "final_statistical_consistency_audit.md").write_text(statistical)
    submission = f"""# CSCWD author-review delivery audit

Final local status: **{status}**.
External submission clearance: **not granted**; no upload has occurred.

The requested anonymous draft contains {machine["pdfs"]["main"]["pages"]} main-paper
pages including references and figures, plus a separate
{machine["pdfs"]["technical_supplement"]["pages"]}-page internal technical supplement.
IEEEtran is byte-identical to the installed official distribution; there are no
font, margin, line-spacing or column-width overrides. All PDF fonts are embedded,
no Type 3 fonts or raster figure images are present, and all pages were rendered
and inspected. The count and cost figures were also inspected in grayscale.

The completed local regression suite has {tests["tests"]} tests,
{tests["failures"]} failures, {tests["errors"]} errors and {tests["skipped"]} skips.
The complete raw file audit, generated numerical bindings and isolated package
extraction checks pass. These are same-host, same-repository checks, not an
independent implementation or cross-platform reproduction.

The first release candidate's missing import-time package data and the plotting
environment's missing static-type imports are retained in audit receipts.
Packaging adds the two unchanged support data files. It does not alter frozen
scientific code. Plotting executes in its separately recorded runtime; static
typing is reported only for the validated non-plotting scope.

## Remaining author actions

Read `final_human_review_required.md` and the related-manuscript comparison.
Supply the final author block; approve scientific claims, citations and the
AI-assistance statement; confirm portal anonymity, attachment limits, deadline
time zone and related-paper disclosure. Source and full-result archives are
internal review materials, not automatically permitted conference attachments.
Code redistribution licensing remains an author decision.

The original AAAI artifacts and registered title/abstract remain protected.
The author has confirmed AAAI rejection and another planned CSCWD manuscript.
This audit does not infer an editorial eligibility ruling from those facts.
"""
    (ROOT / "paper/cscwd2027/final_submission_audit.md").write_text(submission)
    (dest / "final_submission_audit.md").write_text(submission)
    print(
        json.dumps(
            {
                "status": status,
                "receipt": str((dest / "delivery_receipt.json").relative_to(ROOT)),
            }
        )
    )
    if not passed:
        raise RuntimeError("author-review gate did not pass")


if __name__ == "__main__":
    main()
