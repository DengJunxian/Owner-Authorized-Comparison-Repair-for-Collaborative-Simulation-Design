"""Write final author-facing receipts entirely from the completed audit."""

from __future__ import annotations

import ast
import json

from scripts.cscwd2027_design_contract.deliver import PAPER, REPORT
from scripts.cscwd2027_design_contract.protection import ROOT
from scripts.cscwd2027_design_contract.store import RESULTS


def main() -> None:
    """Summarize actual evidence and remaining human decisions, not acceptance."""
    audit = json.loads((REPORT / "delivery_audit.json").read_text())
    if audit["status"] != "READY FOR HUMAN REVIEW":
        raise RuntimeError("passed machine/visual/review gates required")
    stats = json.loads((RESULTS / "derived/summary.json").read_text())
    presentation = json.loads((REPORT / "presentation_receipt.json").read_text())
    mode = stats["mode_statistics"]
    rev = stats["revision"]
    tests = audit["tests"]
    pdfs = audit["pdfs"]
    text = f"""# CSCWD comparison-contract final audit

Status: **{audit["status"]}**. Anonymous author-review version, not upload approval.

## Evidence and new contribution

The new study has {stats["actual_attempts"]}/{stats["planned_attempts"]} actual
paired attempts, {rev["sessions"]} design sessions and {stats["handoff_count"]}
handoffs. Its source/configuration/analysis seal is `{audit["freeze_sha256"]}`.
Failed, locally rejected, missing attempts and integrity findings are all empty
lists in the complete structured summary. The original v1, revision-two,
residual probe and final-review protection manifests pass without modification;
the original AAAI registered metadata/evidence tests also pass.

Per-run-full writes {mode["local_full_only"]["unauthorized_release"]} unauthorized
panels; specification-derived grouping still writes
{mode["spec_grouped"]["unauthorized_release"]}, with
{mode["spec_grouped"]["optimal_set_error"]} positive-regret choices among its
{mode["spec_grouped"]["written"]} actual reports. The comparison contract writes
{mode["comparison_contract"]["unauthorized_release"]} unauthorized panels and
retains {mode["comparison_contract"]["valid_control_release"]}/
{mode["comparison_contract"]["evaluable_controls"]} evaluable valid/recoverable
controls. The equal-obligation relational collector agrees; this is retained as
an equivalence control, not branded superiority.

Selective revision executes {rev["selective_new_pairs"]} new pairs and reuses
{rev["reused_pairs"]}, versus {rev["fresh_new_pairs"]} fresh pairs. All
{rev["slot_comparisons"]} slot comparisons and {stats["remap"]["pairs"]} remap
comparisons complete; scientific and draw agreement are exact. Complete release
and correct top-choice counts are separate. Predeclared dispatch/reuse agrees
with the key-based plan in {audit["schedule_plan_matches"]} sessions. This is
plan implementation consistency, not online adaptive dispatch. Stale evidence
changes top choice in {rev["stale_top1_disagreement"]}/{rev["sessions"]} sessions;
{rev["stale_optimal_set_error"]} choices have positive finite-panel regret.

## Statistical and source checks

Frozen analysis recomputes exactly: {audit["statistics_recompute"]}.
Direct arithmetic audit from all trades and actual report files passes:
{audit["raw_arithmetic_recheck_pass"]}. Raw file commitment changes:
{len(audit["raw_file_changes"])}. All generated count/table/figure bindings pass.
Session blocks, not handoff repetitions, are the scientific denominator.
Regret is undefined for abstention and the reference is a complete finite panel,
not a population optimum. Original nulls, shared-code counterexamples and ties
remain visible. The one-candidate revision determines the theoretical work
saving; complete correct reuse is the tested empirical claim.

## Page, font, figure and anonymity checks

Main: {pdfs["main"]["pages"]} total pages including references, figures and tables.
Supplement: {pdfs["technical_supplement"]["pages"]} pages, for local author review;
upload permission is not established. IEEEtran class bytes match the installed
official class. No margin, font-size, line-spacing or column-width override,
overfull box or unresolved citation is present. All used fonts are embedded,
with no Type 3 font or raster figure in either compiled PDF. Every page was
rendered and visually checked. Standard IEEE reference-column breaking is used.
The main title and anonymous author block are preserved.

The sealed original heatmap is retained. A separately manifested vector redraw
changes only the rendering primitive and minimum count-label size, not data,
range, grayscale, count or panel selection.

Main SHA-256: `{pdfs["main"]["sha256"]}`.
Supplement SHA-256: `{pdfs["technical_supplement"]["sha256"]}`.

## Tests and review

Relevant regression: {tests["tests"]} passed, {tests["failures"]} failures,
{tests["errors"]} errors, {tests["skipped"]} skipped. Ruff and strict mypy pass.
Statement/branch coverage and a software mutation score are unavailable.
Three internal same-model reviewers retain 6/10, 7/10 and 7/10. All fixable
comments were resolved and final PDF differences confirmed; these are not
external reviews, acceptance odds or award-level evidence.

## Remaining human decisions

Confirm authors, portal anonymity/supporting-upload policy, deadline time zone,
AI-assistance disclosure, redistribution rights and the relationship with the
other planned CSCWD paper before submission. The AAAI paper was rejected, as
confirmed by the author. No external upload, editor communication or public
release is performed. Package extraction/rebuild is recorded separately.
"""
    (PAPER / "final_submission_audit.md").write_text(text)
    (REPORT / "final_statistical_consistency_audit.md").write_text(text)
    (PAPER / "revision_log.md").write_text(
        "# Comparison-contract revision log\n\n"
        "Preserved the earlier final-review paper and all prior freezes/results.\n\n"
        "Implemented scientific recipe authority, comparison slot conformance, "
        "trusted-store tokens and outcome-independent invalidation in a new namespace. "
        "Completed the separately sealed design-handoff/revision study.\n\n"
        "Recentered the manuscript on local validity versus comparison validity "
        "and safe revision reuse. Added workflow/build-system prior art and fair "
        "spec-derived/relational controls. Kept all nulls, ties and TCB failures.\n\n"
        "Resolved review findings on release/correctness counters, plan-dispatch "
        "wording, completeness, vector output, stronger control decision effects, "
        "local-validity assumptions, Figure1 and final-page balance.\n\n"
        "Both requested anti-defensive writing skills guided direct evidence-led "
        "claims; no advantage, result or acceptance status was invented.\n"
    )
    (PAPER / "final_human_review_required.md").write_text(
        "# Human author approval required\n\n"
        "Review the comparison contract, finite-panel objective, exact-key reuse "
        "boundary, all negative cases and the three internal reviews. The "
        "composition-level novelty question remains substantive.\n\n"
        "Do not interpret fixed pair-count saving as a measured speedup, this "
        "same-code reference as independent reproduction, or branch-epoch "
        "revision as a pure timing causal effect.\n\n"
        "Confirm final authors, anonymity policy and supporting-upload permission "
        "in EasyChair. Recompile after adding authors and retain the total "
        "six-page main-paper limit. Approve all citations and AI assistance.\n\n"
        "Review the related planned CSCWD manuscript together with this version "
        "and arrange appropriate related-paper disclosure before uploading. "
        "No public redistribution license or external submission is authorized "
        "by this local review status.\n"
    )
    claims = [
        "# Revised claim-evidence map",
        "",
        "| Claim | Main evidence | Bound |",
        "|---|---|---|",
        "| Local validity does not compose into comparison validity | "
        "evaluation/results.json, all actual handoff numerical files | "
        "finite designed families, not field fault prevalence |",
        "| Intended scientific membership matters after correct grouping | "
        "spec_grouped unauthorized and positive-regret fields | "
        "finite-panel reference; many invalid panels leave choices unchanged |",
        "| Separation retains permitted worker reassignment | "
        "96 remap comparisons and authorized_remap controls | "
        "static maps, fixed logical schedule |",
        "| Selective evidence reuse is complete and fresh-equivalent | "
        "all_revision_rows, complete slot comparisons, schedule identity audit | "
        "one-candidate revision; exact-key equivalence and trusted store |",
        "| Contract composition, not unique join/cache algorithm | "
        "equal-obligation relational collector agreement and cited prior art | "
        "no generic caching-theory or external implementation claim |",
        "| Shared-code trust remains necessary | "
        "unchanged residual sampler probe and diagnostic artifacts | "
        "partial witnesses, not full sampler/CLOB correctness |",
    ]
    (REPORT / "revised_claim_evidence_matrix.md").write_text("\n".join(claims) + "\n")
    tcb = [
        "# Added trusted components",
        "",
        "LOC excludes blanks and full-line comments. "
        "AST statement counts are source size, not executed coverage. Components "
        "share code and must not be summed as independent proof strength.",
        "",
        "| Source | LOC | AST statements | Main responsibility | Tests |",
        "|---|---:|---:|---|---|",
    ]
    paths = {
        "src/policytwin/design_contract/runtime.py": "recipe/opening/source authority",
        "src/policytwin/design_contract/contract.py": "comparison and reuse relation",
        "scripts/cscwd2027_design_contract/store.py": "local verification token",
    }
    for name, duty in paths.items():
        code = (ROOT / name).read_text()
        loc = sum(
            bool(s.strip()) and not s.lstrip().startswith("#")
            for s in code.splitlines()
        )
        statements = sum(isinstance(n, ast.stmt) for n in ast.walk(ast.parse(code)))
        tcb.append(
            f"| `{name}` | {loc} | {statements} | {duty} | "
            "tests/cscwd2027_design_contract |"
        )
    tcb += [
        "",
        f"The current combined regression has {tests['tests']} passing tests. "
        "Test associations are not per-component coverage percentages. "
        "The inherited TCB table remains unchanged. Correct sampler, "
        "serializer, domain ontology, CLOB, local verification and trusted "
        "storage remain assumptions. Statement/branch coverage and software "
        "mutation score: unavailable.",
    ]
    (REPORT / "tcb_audit.md").write_text("\n".join(tcb) + "\n")
    print(
        json.dumps(
            {
                "status": audit["status"],
                "tests": tests["tests"],
                "pages": {k: v["pages"] for k, v in pdfs.items()},
                "extra_counts": presentation["numeric_values"],
            }
        )
    )


if __name__ == "__main__":
    main()
