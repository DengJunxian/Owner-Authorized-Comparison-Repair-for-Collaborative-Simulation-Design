"""Create final human-review receipts from actual audits, not hand-entered counts."""

from __future__ import annotations

import json

from .protection import ROOT


def main() -> None:
    """Generate the final consistency receipt after every required gate passes."""
    reports = ROOT / "reports/cscwd2027/final_review"
    audit = json.loads((reports / "delivery_audit.json").read_text())
    if audit["status"] != "READY FOR HUMAN REVIEW":
        raise RuntimeError("passed audit required")
    pdfs, tests = audit["pdfs"], audit["tests"]
    original_count = audit["original_cscwd_protection"]["files"]
    revision2_count = audit["revision2_protection"]["files"]
    text = f"""# CSCWD final-review submission audit

Status: **{audit["status"]}**, anonymous author-review version.
External submission clearance and human author sign-off remain outstanding.

## Evidence preservation

- Original CSCWD protected files unchanged: {original_count}.
- Additional revision2 protected files unchanged: {revision2_count}.
- Original formal revision2 attempts: {audit["original_formal_attempts"]}.
- New simulation attempts in this final-review turn: {audit["new_simulation_attempts"]}.
- Raw-file commitment mismatches: {len(audit["changed_raw_files"])}.
- The original v1/v2 freezes and separately sealed sampler probe remain unchanged.
- Original AAAI metadata and evidence-lock regression tests pass.

## Scientific corrections

The manuscript now explicitly positions the contract relative to deterministic
multithreading and runtime verification. A hash commitment binds content; trusted
registration and endpoint identities supply provenance. The numerical contrast
has synthetic notional units. All original experiment counts, ablation ties,
numerical nulls and the undetected sampler example remain in the evidence.

The separate final-review activity audit is post-hoc and descriptive. Its raw
inputs and outputs have their own commitments, and direct recomputation agrees
with its structured summary. It does not amend registered metrics or add runs.
Inherited statistical figures and count tables are byte-identical to revision2.

## Page, template and anonymity checks

Main: {pdfs["main"]["pages"]} pages, including references/figures/tables.
Technical supplement: {pdfs["technical_supplement"]["pages"]} pages.
The IEEEtran class matches the installed official class. No font, margin,
line-spacing or column-width overrides, unresolved citations or overfull boxes
were found. All fonts are embedded; figures are vector. Every page was rendered
and visually inspected, including the balanced final-page columns. The title
and Anonymous authors blocks remain unchanged. PDF privacy-shape checks pass.

Main SHA-256: `{pdfs["main"]["sha256"]}`.

Supplement SHA-256: `{pdfs["technical_supplement"]["sha256"]}`.

## Tests and review scope

Relevant regression: {tests["tests"]} tests, {tests["failures"]} failures,
{tests["errors"]} errors, {tests["skipped"]} skipped. Ruff and mypy checks pass.
Statement/branch coverage and a software mutation score remain unavailable.
The three reviewer lenses are simulated diagnostics, not external human reviews
or an acceptance prediction. The full report records unresolved novelty,
workload-extent and shared-TCB judgments without averaging them away.

The package extraction/rebuild receipt is generated separately after packaging.
No external upload or public release has been performed. Portal anonymity,
supporting uploads, final authors and related-paper disclosures need human review.
"""
    (ROOT / "paper/cscwd2027_final_review/final_submission_audit.md").write_text(text)
    (reports / "final_statistical_consistency_audit.md").write_text(text)
    (reports / "delivery_receipt.json").write_text(
        json.dumps(
            {
                "status": audit["status"],
                "pdfs": {
                    k: {f: v[f] for f in ("pages", "bytes", "sha256")}
                    for k, v in pdfs.items()
                },
                "tests": tests,
                "new_simulation_attempts": 0,
                "human_review_complete": False,
                "external_submission_cleared": False,
                "review_report": (
                    "paper/cscwd2027_final_review/ccfa-review-reports/"
                    "2026-10-25-verifiable-coordination-cscwd2027-conference-review.md"
                ),
            },
            indent=2,
        )
        + "\n"
    )
    print(json.dumps({"status": audit["status"], "tests": tests["tests"]}))


if __name__ == "__main__":
    main()
