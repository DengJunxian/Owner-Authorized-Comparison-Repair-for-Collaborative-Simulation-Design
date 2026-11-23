"""Post-freeze manuscript/build and evidence audits; no new scientific estimator."""

from __future__ import annotations

import ast
import json
import re
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

from scripts.cscwd2027_revision2.baseline import verify
from scripts.cscwd2027_revision2.freeze import ROOT, check, digest

PAPER = ROOT / "paper/cscwd2027_revision2"
REPORT = ROOT / "reports/cscwd2027/revision2"


def output(command: list[str]) -> str:
    """Read actual local diagnostic output."""
    return subprocess.check_output(command, cwd=ROOT, text=True)


def pdf_audit(stem: str) -> dict[str, Any]:
    """Inspect compiled pages, font embedding and errors, not only source."""
    path = PAPER / f"{stem}.pdf"
    info = output(["pdfinfo", str(path)])
    match = re.search(r"^Pages:\s+(\d+)$", info, re.M)
    if match is None:
        raise RuntimeError("PDF page count unavailable")
    pages = int(match.group(1))
    fonts = output(["pdffonts", str(path)])
    unembedded = [
        r.split()[0] for r in fonts.splitlines()[2:] if r.split()[-5] != "yes"
    ]
    type3 = [r for r in fonts.splitlines()[2:] if "Type 3" in r]
    text = output(["pdftotext", "-layout", str(path), "-"])
    log = (PAPER / f"{stem}.log").read_text(errors="replace")
    warnings = [
        r
        for r in log.splitlines()
        if r.startswith("!")
        or "Overfull" in r
        or "undefined" in r.lower()
        or "Rerun to get" in r
    ]
    privacy = {
        "personal_path": bool(re.search(r"/Users/[A-Za-z0-9_.-]+/", info + text)),
        "email": bool(re.search(r"\b[\w.%+-]+@[\w.-]+\.[A-Za-z]{2,}\b", info + text)),
        "credential_shape": bool(re.search(r"\bsk-[A-Za-z0-9_-]{24,}\b", info + text)),
        "anonymous_block_missing": "Anonymous authors" not in text,
    }
    raster_rows = output(["pdfimages", "-list", str(path)]).splitlines()[2:]
    return {
        "sha256": digest(path),
        "bytes": path.stat().st_size,
        "pages": pages,
        "fonts": fonts,
        "unembedded": unembedded,
        "type3": type3,
        "warnings": warnings,
        "privacy": privacy,
        "raster_rows": raster_rows,
        "pass": not (
            unembedded or type3 or warnings or any(privacy.values()) or raster_rows
        )
        and (0 < pages <= 6 if stem == "main" else pages > 0),
    }


def main() -> None:
    """Check all completed evidence and manuscript constraints before review."""
    freeze = check()
    protected = verify()
    summary = json.loads(
        (ROOT / "results/cscwd2027/revision2/derived/summary.json").read_text()
    )
    complete = (
        summary["data_complete"]
        and not summary["failed"]
        and not summary["integrity_findings"]
    )
    if not complete:
        raise RuntimeError("complete verified study required")
    changed_raw = [
        name
        for name, sha in summary["raw_file_manifest"].items()
        if digest(ROOT / name) != sha
    ]
    presentation = json.loads((PAPER / "generated/revision_manifest.json").read_text())
    changed_presentation = [
        name
        for name, sha in presentation["outputs"].items()
        if digest(ROOT / name) != sha
    ]
    input_matches = digest(ROOT / presentation["input"]) == presentation["input_sha256"]
    official = Path(output(["kpsewhich", "IEEEtran.cls"]).strip())
    template_matches = digest(official) == digest(PAPER / "IEEEtran.cls")
    sources = [
        PAPER / "main.tex",
        PAPER / "technical_supplement.tex",
        *list((PAPER / "sections").glob("*.tex")),
    ]
    forbidden = re.compile(
        r"\\(?:geometry|fontsize|baselinestretch|linespread|vspace|enlargethispage)\b|\\setlength\s*\{\\(?:textwidth|textheight|columnsep|topmargin|oddsidemargin)"
    )
    overrides = [
        str(p.relative_to(ROOT)) for p in sources if forbidden.search(p.read_text())
    ]
    bib = set(re.findall(r"@\w+\{([^,]+),", (PAPER / "references.bib").read_text()))
    cited = {
        key
        for p in sources
        for group in re.findall(r"\\cite\{([^}]+)\}", p.read_text())
        for key in group.split(",")
    }
    unknown = sorted(cited - bib)
    figures = {}
    for name in ("invalid_release_publication.pdf", "cost.pdf"):
        current = PAPER / "generated" / name
        original = ROOT / "paper/cscwd2027/generated" / name
        figures[name] = {
            "sha256": digest(current),
            "unchanged_from_v1": digest(current) == digest(original),
        }
    inherited_values_unchanged = all(
        digest(PAPER / "generated" / name)
        == digest(ROOT / "paper/cscwd2027/generated" / name)
        for name in (
            "numbers.tex",
            "presentation_numbers.tex",
            "partition_table.tex",
            "fault_outcomes_compact.tex",
            "cost_a6.tex",
            "cost_a24.tex",
        )
    )
    title_pattern = r"\\title\{([\s\S]*?)\}"
    current_title = re.search(title_pattern, (PAPER / "main.tex").read_text())
    old_title = re.search(
        title_pattern, (ROOT / "paper/cscwd2027/main.tex").read_text()
    )
    title_matches = (
        current_title is not None
        and old_title is not None
        and current_title.group(1) == old_title.group(1)
    )
    pdfs = {stem: pdf_audit(stem) for stem in ("main", "technical_supplement")}
    suite = ET.parse(REPORT / "final_tests.xml").getroot()
    test_counts = {
        key: sum(int(s.get(key, "0")) for s in suite.iter("testsuite"))
        for key in ("tests", "failures", "errors", "skipped")
    }
    witness_source = ROOT / "src/policytwin/coordination_validation/witnesses.py"
    tree = ast.parse(witness_source.read_text())
    imports = [
        node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
    ]
    witness_loc = sum(
        bool(line.strip()) and not line.lstrip().startswith("#")
        for line in witness_source.read_text().splitlines()
    )
    probe_path = ROOT / "configs/cscwd2027/revision2/residual_probe_freeze.json"
    probe_seal = json.loads(probe_path.read_text())
    probe_unchanged = digest(probe_path) == probe_path.with_suffix(
        ".sha256"
    ).read_text().strip() and all(
        digest(ROOT / name) == sha for name, sha in probe_seal["members"].items()
    )
    probe = json.loads(
        (ROOT / "results/cscwd2027/revision2/residual_probe/summary.json").read_text()
    )
    passed = all(
        (
            protected["unchanged"],
            complete,
            not changed_raw,
            not changed_presentation,
            input_matches,
            template_matches,
            not overrides,
            not unknown,
            all(p["pass"] for p in pdfs.values()),
            not test_counts["failures"],
            not test_counts["errors"],
            imports == ["__future__", "typing"],
            probe_unchanged,
            inherited_values_unchanged,
            title_matches,
        )
    )
    report = {
        "status": "READY FOR HUMAN REVIEW" if passed else "NOT READY",
        "deliverable": "anonymous author-review draft",
        "external_submission_cleared": False,
        "new_freeze": {
            "sha256": digest(ROOT / "configs/cscwd2027/revision2/freeze.json"),
            "created_at_utc": freeze["created_at_utc"],
        },
        "protected_original": protected,
        "formal_attempts": summary["attempts"],
        "runner_failures": summary["failed"],
        "changed_raw_files": changed_raw,
        "changed_generated_files": changed_presentation,
        "input_matches": input_matches,
        "official_template_unchanged": template_matches,
        "original_generated_values_unchanged": inherited_values_unchanged,
        "cscwd_title_unchanged": title_matches,
        "layout_overrides": overrides,
        "unknown_citations": unknown,
        "tests": test_counts,
        "pdfs": pdfs,
        "figures": figures,
        "residual_probe": {
            "source_seal_unchanged": probe_unchanged,
            "actual_summary": probe,
        },
        "witness_module": {
            "nonblank_noncomment_loc": witness_loc,
            "imports": imports,
            "statement_coverage": "unavailable",
            "branch_coverage": "unavailable",
            "software_mutation_score": (
                "unavailable; four common-mode hooks are a separate diagnostic"
            ),
        },
        "visual_review": "A separate hash-bound page-by-page record is required",
        "pass": passed,
    }
    REPORT.mkdir(parents=True, exist_ok=True)
    (REPORT / "delivery_audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                "pass": passed,
                "pages": {k: v["pages"] for k, v in pdfs.items()},
                "protected_files": protected["files"],
                "tests": test_counts,
            }
        )
    )
    if not passed:
        raise RuntimeError("delivery audit has findings")


if __name__ == "__main__":
    main()
