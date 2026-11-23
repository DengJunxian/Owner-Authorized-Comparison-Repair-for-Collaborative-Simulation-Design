"""Post-freeze author-review reports generated from the completed audits."""

from __future__ import annotations

import json

from scripts.cscwd2027_revision2.freeze import ROOT, check, digest


def main() -> None:
    """Bind concise disposition/claim reports to actual artifacts and PDFs."""
    freeze = check()
    reports = ROOT / "reports/cscwd2027/revision2"
    audit = json.loads((reports / "delivery_audit.json").read_text())
    summary = json.loads(
        (ROOT / "results/cscwd2027/revision2/derived/summary.json").read_text()
    )
    visual = json.loads((reports / "visual_review.json").read_text())
    paper = ROOT / "paper/cscwd2027_revision2"
    if (
        not audit["pass"]
        or not visual["pass"]
        or any(
            visual[s]["sha256"] != digest(paper / f"{s}.pdf")
            for s in ("main", "technical_supplement")
        )
    ):
        raise RuntimeError("complete, current machine and visual audit required")
    extension = summary["extension"]
    mutants = [v for k, v in summary["common_mode"].items() if k != "clean"]
    clean = summary["common_mode"]["clean"]
    report = f"""# Final statistical and evidence consistency audit

Status: **READY FOR HUMAN REVIEW**, for an anonymous author-review draft.
This is not submission clearance or human sign-off.

## Separate evidence layers

- Original CSCWD v1 source/configuration seal and results remain unchanged.
- Pre-revision protection rechecked {audit["protected_original"]["files"]} old
  source/result/delivery files without a changed byte.
- New main grid: {summary["attempts"]}/{summary["expected_attempts"]} attempted
  pairs, {len(summary["failed"])} runner failures,
  {len(summary["integrity_findings"])} raw-file/report binding findings.
- Extension: {sum(r["trace_equal"] for r in extension)} exact trace matches in
  {sum(r["attempts"] for r in extension)} actual four-process pairs; the matched
  in-process references are a separate denominator.
- Shared-code diagnostic: {sum(r["attempts"] for r in mutants)} altered pairs,
  all retained; full releases {sum(r["full_release"] for r in mutants)} and the
  witnessed route releases {sum(r["witness_release"] for r in mutants)}.
- Witnessed clean acceptance: {clean["witness_release"]}/{clean["attempts"]}.
- Altered numerical contrasts: {sum(r["contrast_changed"] for r in mutants)}
  changed, {sum(r["contrast_unchanged"] for r in mutants)} unchanged.
- The separate exploratory residual probe is not pooled into these counts.
  Its changed sampler is undetected by both routes, and the numerical files
  remain in the raw evidence. It is not a false-negative-rate estimate.

## Number and format binding

All new paper count macros/tables come from the frozen analysis summary. Old
count macros, original tables and figure bytes remain equal to v1. Main and
supplement use the same numerical macros. The original registered AAAI metadata
and 3,072-row evidence checks also passed without editing those files.
The new CSCWD title is unchanged; its abstract is revised for the new evidence.

The main PDF has {audit["pdfs"]["main"]["pages"]} pages including references and
figures; technical support has {audit["pdfs"]["technical_supplement"]["pages"]}.
The unmodified IEEEtran conference class is retained. There are no font,
margin, line-spacing or column-width overrides. Standard column breaks balance
the final pages. All pages are rendered and inspected, all fonts are embedded,
there are no unresolved citations or overfull boxes, and the figures remain
vector with redundant grayscale cues.

## Test and interpretation boundaries

The current relevant regression suite contains {audit["tests"]["tests"]} passing
cases, zero failures/errors/skips. The witness module has
{audit["witness_module"]["nonblank_noncomment_loc"]} nonblank/noncomment physical
lines and imports only future annotations and typing. Statement/branch coverage
and a software mutation score are unavailable. The declared diagnostic operators
are not substituted for a general software-mutation score.

Worker variants share scientific inputs; they are not extra independent seeds.
Exact agreement is a finite test result, not a fault-population confidence bound.
Extension dimensions are separate cells, not a maximum-dimension factorial.
Old cost intervals describe the original full route and measured host, not the
new witness route's incremental performance. Acceptance remains conditional on
the complete acceptance event; no all-pass batch is used to claim unbiasedness.

## Freeze identity

The new main study was sealed at {freeze["created_at_utc"]} with SHA-256
`{digest(ROOT / "configs/cscwd2027/revision2/freeze.json")}`.
The residual probe has its own source/configuration seal and is explicitly
exploratory, not a blind held-out campaign. No old/new frozen member, formal
threshold, sample list, seed or analysis definition was changed after sealing.
"""
    (reports / "final_statistical_consistency_audit.md").write_text(report)
    claim_rows = [
        (
            "Coordination and replay enforce non-equivalent obligations",
            "Observation 1; original E2 findings; obligation_separation.md",
            "Constructive witnesses, not universal minimality",
        ),
        (
            "Tested invariance extends across map, horizon and population",
            "Original E1 plus separately sealed extension_results.md",
            "Fixed schedule; same CLOB; separate dimension cells",
        ),
        (
            "Action/ledger identities complement shared replay",
            "common_mode_results.md; actual report files",
            "Four designed shared defects; partial specification",
        ),
        (
            "Partial witnesses leave a trusted sampler boundary",
            "residual_probe/summary.json; preserved altered reports",
            "Exploratory counterexample, not a detection-rate estimate",
        ),
        (
            "Release differs from numerical change and diagnostic alarm",
            "Original ablations and both new report routes",
            "Native incompletion and numerical nulls are separate",
        ),
        (
            "Original evidence is preserved",
            "v1 seal; protected baseline; registered checks",
            "Earlier source-byte gap retained, not retroactively repaired",
        ),
        (
            "Anonymous PDFs meet requested local format constraints",
            "delivery_audit.json; visual_review.json",
            "Six-page main; no upload or author sign-off",
        ),
    ]
    claims = "\n".join(
        [
            "# Revised claim–evidence map",
            "",
            "| Claim | Evidence | Boundary |",
            "|---|---|---|",
        ]
        + ["| " + " | ".join(row) + " |" for row in claim_rows]
        + [
            "",
            "Not claimed: independent/cross-platform reproduction, arbitrary TCB",
            "correctness, malicious-component security, new RNG primitives, universal",
            "speedup, real-market validity, or exclusive capability over an",
            "equal-obligation conventional system.",
            "",
        ]
    )
    (reports / "claim_evidence_matrix.md").write_text(claims)
    conflicts = """# Evidence-scope and conflict audit

The original full route's 0/256 invalid releases refers to its declared eight
runtime-fault operations. The new four-operator diagnostic deliberately violates
the shared-code TCB premise and records 96/96 releases under that unchanged
route. These are different operation universes and assumptions. Both results
are retained; neither is substituted for the other or pooled into a new class
count. The witnessed route's 0/96 applies only to those four diagnostic operators.

The separately sealed all-zero-sampler probe is accepted by both routes and
changes the numerical contrast. It prevents extending the 0/96 result to all
common-mode defects. The attempted and released artifacts remain available.
No original registered abstract number or metadata was edited to reconcile it.
The revised CSCWD abstract and main text state the scoped results and retain
the sampler counterexample. No numerical contradiction is hidden by changing
the original definitions, seeds, result files or seals.
"""
    (reports / "evidence_scope_conflict_audit.md").write_text(conflicts)
    receipt = {
        "status": "READY FOR HUMAN REVIEW",
        "author_block": "anonymous, explicitly requested",
        "main_pages": audit["pdfs"]["main"]["pages"],
        "supplement_pages": audit["pdfs"]["technical_supplement"]["pages"],
        "pdfs": {
            k: {field: value[field] for field in ("sha256", "bytes", "pages")}
            for k, value in audit["pdfs"].items()
        },
        "attempts_in_sealed_main_grid": summary["attempts"],
        "additional_separately_sealed_exploratory_attempts": 2,
        "protected_old_files_unchanged": audit["protected_original"]["files"],
        "tests": audit["tests"],
        "human_review_complete": False,
        "external_submission_cleared": False,
        "public_release_authorized": False,
        "machine_audit": "reports/cscwd2027/revision2/delivery_audit.json",
        "visual_review": "reports/cscwd2027/revision2/visual_review.json",
        "post_build_extraction_validation": "separate receipt after archive creation",
    }
    (reports / "delivery_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    (paper / "final_submission_audit.md").write_text(report)
    print(
        json.dumps(
            {
                "status": receipt["status"],
                "pages": [receipt["main_pages"], receipt["supplement_pages"]],
            }
        )
    )


if __name__ == "__main__":
    main()
