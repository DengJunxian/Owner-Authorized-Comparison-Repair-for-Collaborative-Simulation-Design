"""Final machine gate for the separate design-contract author-review delivery."""

from __future__ import annotations

import ast
import json
import re
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

from scripts.check_registered_metadata import check_registered_metadata
from scripts.cscwd2027 import audit_baseline
from scripts.cscwd2027_design_contract.analyze import analyze
from scripts.cscwd2027_design_contract.deliver import PAPER, REPORT
from scripts.cscwd2027_design_contract.freeze import DEST, check
from scripts.cscwd2027_design_contract.protection import ROOT, digest, verify
from scripts.cscwd2027_design_contract.store import RESULTS
from scripts.cscwd2027_final_review import audit as previous_audit


def bound_manifest(path: Path) -> bool:
    """Verify declared generated artifacts against actual bytes."""
    value = json.loads(path.read_text())
    passed = all(digest(ROOT / p) == sha for p, sha in value["outputs"].items())
    if "input" in value:
        passed &= digest(ROOT / value["input"]) == value["input_sha256"]
    if isinstance(value.get("source"), dict):
        passed &= all(digest(ROOT / p) == sha for p, sha in value["source"].items())
    return passed


def main() -> None:
    """Recompute all endpoints, preserve evidence, and check actual compiled PDFs."""
    seal = check()
    protection = verify()
    audit_baseline.verify()
    check_registered_metadata(
        ROOT / "paper/main_revised.tex", ROOT / "paper/registered_metadata_lock.json"
    )
    summary = json.loads((RESULTS / "derived/summary.json").read_text())
    recomputes = summary == analyze()
    raw_changes = [
        name
        for name, sha in summary["raw_file_manifest"].items()
        if digest(RESULTS / name) != sha
    ]
    receipt = json.loads((REPORT / "presentation_receipt.json").read_text())
    raw = json.loads((REPORT / "raw_arithmetic_recheck.json").read_text())
    raw_ok = (
        raw["pass"] and raw["raw_capsules_recomputed"] == summary["actual_attempts"]
    )
    raw_ok &= raw["source_sha256"] == digest(Path(__file__).with_name("raw_recheck.py"))
    raw_ok &= raw["evaluation_sha256"] == digest(RESULTS / "evaluation/results.json")
    previous_audit.PAPER = PAPER
    pdfs = {
        name: previous_audit.pdf_audit(name)
        for name in ("main", "technical_supplement")
    }
    for name, item in pdfs.items():
        item["pass"] = (
            not (
                item["unembedded_fonts"]
                or item["type3_fonts"]
                or item["raster_images"]
                or item["errors"]
                or any(item["privacy"].values())
            )
            and item["pages"] > 0
            and (name != "main" or item["pages"] <= 6)
        )
    official = Path(
        subprocess.check_output(["kpsewhich", "IEEEtran.cls"], text=True).strip()
    )
    template = digest(official) == digest(PAPER / "IEEEtran.cls")
    sources = [
        PAPER / "main.tex",
        PAPER / "technical_supplement.tex",
        *sorted((PAPER / "sections").glob("*.tex")),
    ]
    override = re.compile(
        r"\\(?:geometry|fontsize|baselinestretch|linespread|vspace|enlargethispage)\b"
        r"|\\setlength\s*\{\\(?:textwidth|textheight|columnsep|topmargin|oddsidemargin)"
    )
    overrides = [
        str(p.relative_to(PAPER)) for p in sources if override.search(p.read_text())
    ]
    injection = re.compile(
        r"ignore (?:all |the )?(?:previous|prior) instructions"
        r"|(?:reviewer|assistant).{0,25}(?:assign|give).{0,15}(?:score|accept)",
        re.I,
    )
    hidden = [
        str(p.relative_to(PAPER)) for p in sources if injection.search(p.read_text())
    ]
    bib = set(re.findall(r"@\w+\{([^,]+),", (PAPER / "references.bib").read_text()))
    cited = {
        key.strip()
        for p in sources
        for group in re.findall(r"\\cite\{([^}]+)\}", p.read_text())
        for key in group.split(",")
    }
    inherited = all(
        digest(p) == digest(PAPER / "generated" / p.name)
        for p in (ROOT / "paper/cscwd2027_final_review/generated").iterdir()
        if p.is_file()
    )
    title_pattern = r"\\title\{([\s\S]*?)\}"
    title = re.search(title_pattern, (PAPER / "main.tex").read_text())
    prior = re.search(
        title_pattern, (ROOT / "paper/cscwd2027_final_review/main.tex").read_text()
    )
    title_equal = bool(title and prior and title[1] == prior[1])
    generated = all(
        bound_manifest(p)
        for p in (
            PAPER / "generated/design_manifest.json",
            PAPER / "generated/figure_manifest_design.json",
            PAPER / "generated/figure_manifest_vector.json",
            REPORT / "presentation_receipt.json",
        )
    )
    suites = ET.parse(REPORT / "final_tests.xml").getroot()
    tests = {
        key: sum(int(s.get(key, "0")) for s in suites.iter("testsuite"))
        for key in ("tests", "failures", "errors", "skipped")
    }
    visual_path = REPORT / "visual_review.json"
    visual = json.loads(visual_path.read_text()) if visual_path.exists() else {}
    visual_ok = visual.get("pass", False) and all(
        visual.get("pdfs", {}).get(k, {}).get("sha256") == v["sha256"]
        and visual.get("pdfs", {}).get(k, {}).get("pages_inspected")
        == list(range(1, v["pages"] + 1))
        for k, v in pdfs.items()
    )
    reviews = REPORT / "three_reviewer_results.json"
    review = json.loads(reviews.read_text()) if reviews.exists() else {}
    review_ok = len(review.get("reviewers", [])) == 3 and review.get(
        "revisions_resolved", False
    )
    review_ok &= review.get("main_sha256") == pdfs["main"]["sha256"]
    review_ok &= (
        review.get("supplement_sha256") == pdfs["technical_supplement"]["sha256"]
    )
    footprint = {}
    for p in (ROOT / "src/policytwin/design_contract").glob("*.py"):
        code = p.read_text()
        footprint[p.name] = {
            "nonblank_noncomment_loc": sum(
                bool(s.strip()) and not s.lstrip().startswith("#")
                for s in code.splitlines()
            ),
            "ast_statements": sum(
                isinstance(n, ast.stmt) for n in ast.walk(ast.parse(code))
            ),
        }
    passed = all(
        (
            recomputes,
            raw_ok,
            not raw_changes,
            not summary["missing_attempts"],
            not summary["integrity_findings"],
            template,
            not overrides,
            not hidden,
            not cited - bib,
            inherited,
            title_equal,
            generated,
            tests["tests"] >= 129,
            not tests["failures"],
            not tests["errors"],
            not tests["skipped"],
            all(r["pass"] for r in pdfs.values()),
            visual_ok,
            review_ok,
            receipt["all_slot_comparisons_complete"],
            receipt["all_remap_comparisons_complete"],
            receipt["numeric_values"]["DesignPlanMatches"] == seal["planned_sessions"],
        )
    )
    value: dict[str, Any] = {
        "status": "READY FOR HUMAN REVIEW" if passed else "NOT READY",
        "freeze_sha256": digest(DEST / "freeze.json"),
        "old_protected_evidence": protection,
        "original_aaai_metadata_lock": "PASS",
        "statistics_recompute": recomputes,
        "raw_arithmetic_recheck_pass": raw_ok,
        "raw_file_changes": raw_changes,
        "actual_attempts": summary["actual_attempts"],
        "pdfs": pdfs,
        "tests": tests,
        "template_equal": template,
        "geometry_overrides": overrides,
        "hidden_instructions": hidden,
        "missing_citations": sorted(cited - bib),
        "inherited_generated_unchanged": inherited,
        "title_unchanged": title_equal,
        "new_generated_bindings": generated,
        "visual_review_bound": visual_ok,
        "internal_reviews_complete": review_ok,
        "new_tcb_source_footprint": footprint,
        "coverage": "unavailable",
        "software_mutation_score": "unavailable",
        "all_comparisons_complete": receipt["all_slot_comparisons_complete"]
        and receipt["all_remap_comparisons_complete"],
        "schedule_plan_matches": receipt["numeric_values"]["DesignPlanMatches"],
        "external_submission_cleared": False,
        "portal_anonymity_and_supplement_permission": "human confirmation required",
    }
    (REPORT / "delivery_audit.json").write_text(json.dumps(value, indent=2) + "\n")
    print(
        json.dumps(
            {
                "status": value["status"],
                "pdfs": {
                    k: {"pages": r["pages"], "pass": r["pass"], "errors": r["errors"]}
                    for k, r in pdfs.items()
                },
                "tests": tests,
            }
        )
    )


if __name__ == "__main__":
    main()
