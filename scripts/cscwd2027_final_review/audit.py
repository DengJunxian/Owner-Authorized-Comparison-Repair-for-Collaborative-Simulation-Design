"""Final-review evidence, manuscript, anonymity and build gate; no simulation."""

from __future__ import annotations

import json
import re
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

from scripts.cscwd2027_revision2.baseline import verify as verify_v1
from scripts.cscwd2027_revision2.freeze import check

from .activity import analyze
from .protection import ROOT, digest, verify

PAPER = ROOT / "paper/cscwd2027_final_review"
REPORT = ROOT / "reports/cscwd2027/final_review"


def output(command: list[str]) -> str:
    """Collect actual diagnostic output without editing its input."""
    return subprocess.check_output(command, text=True, cwd=ROOT)


def pdf_audit(stem: str) -> dict[str, Any]:
    """Inspect actual pages, embedded fonts, metadata and final build log."""
    path = PAPER / f"{stem}.pdf"
    info = output(["pdfinfo", str(path)])
    found = re.search(r"^Pages:\s+(\d+)$", info, re.M)
    if found is None:
        raise RuntimeError("missing PDF page count")
    pages = int(found.group(1))
    text = output(["pdftotext", "-layout", str(path), "-"])
    fonts = output(["pdffonts", str(path)])
    unembedded = [r for r in fonts.splitlines()[2:] if r.split()[-5] != "yes"]
    type3 = [r for r in fonts.splitlines()[2:] if "Type 3" in r]
    rasters = output(["pdfimages", "-list", str(path)]).splitlines()[2:]
    logs = (PAPER / f"{stem}.log").read_text(errors="replace")
    errors = [
        line
        for line in logs.splitlines()
        if line.startswith("!")
        or "Overfull" in line
        or "undefined" in line.lower()
        or "Rerun to get" in line
    ]
    privacy = {
        "personal_path": bool(re.search(r"/Users/[A-Za-z0-9_.-]+/", text + info)),
        "email": bool(re.search(r"\b[\w.%+-]+@[\w.-]+\.[A-Za-z]{2,}\b", text)),
        "credential": bool(re.search(r"\bsk-[A-Za-z0-9_-]{24,}\b", text)),
        "anonymous_block_missing": "Anonymous authors" not in text,
    }
    return {
        "pages": pages,
        "sha256": digest(path),
        "bytes": path.stat().st_size,
        "font_report": fonts,
        "unembedded_fonts": unembedded,
        "type3_fonts": type3,
        "raster_images": rasters,
        "errors": errors,
        "privacy": privacy,
        "pass": not (unembedded or type3 or rasters or errors or any(privacy.values()))
        and 0 < pages <= 6,
    }


def main() -> None:
    """Require byte preservation, derived-number binding and anonymous six-page PDFs."""
    check()
    old_protection, revision2_protection = verify_v1(), verify()
    summary_path = ROOT / "results/cscwd2027/revision2/derived/summary.json"
    summary = json.loads(summary_path.read_text())
    changed_raw = [
        name
        for name, sha in summary["raw_file_manifest"].items()
        if not (ROOT / name).is_file() or digest(ROOT / name) != sha
    ]
    activity_path = (
        ROOT / "results/cscwd2027/final_review/derived/activity_summary.json"
    )
    activity = json.loads(activity_path.read_text())
    activity_matches = activity == analyze()
    manifest = json.loads((PAPER / "generated/final_review_manifest.json").read_text())
    generated_bound = digest(ROOT / manifest["input"]) == manifest["input_sha256"]
    generated_bound &= all(
        digest(ROOT / name) == sha for name, sha in manifest["outputs"].items()
    )
    inherited = {
        p.name: digest(p) == digest(PAPER / "generated" / p.name)
        for p in (ROOT / "paper/cscwd2027_revision2/generated").iterdir()
        if p.is_file()
    }
    sources = [
        PAPER / "main.tex",
        PAPER / "technical_supplement.tex",
        *sorted((PAPER / "sections").glob("*.tex")),
    ]
    forbidden = re.compile(
        r"\\(?:geometry|fontsize|baselinestretch|linespread|vspace|enlargethispage)\b"
        r"|\\setlength\s*\{\\(?:textwidth|textheight|columnsep|topmargin|oddsidemargin)"
    )
    overrides = [p.name for p in sources if forbidden.search(p.read_text())]
    injection = re.compile(
        r"ignore (?:all |the )?(?:previous|prior) instructions"
        r"|(?:reviewer|assistant).{0,25}(?:assign|give).{0,15}(?:score|accept)",
        re.I,
    )
    hidden = [p.name for p in sources if injection.search(p.read_text())]
    bib = set(re.findall(r"@\w+\{([^,]+),", (PAPER / "references.bib").read_text()))
    cited = {
        key.strip()
        for p in sources
        for group in re.findall(r"\\cite\{([^}]+)\}", p.read_text())
        for key in group.split(",")
    }
    official = Path(output(["kpsewhich", "IEEEtran.cls"]).strip())
    template_equal = digest(official) == digest(PAPER / "IEEEtran.cls")
    title = re.search(r"\\title\{([\s\S]*?)\}", (PAPER / "main.tex").read_text())
    old_title = re.search(
        r"\\title\{([\s\S]*?)\}",
        (ROOT / "paper/cscwd2027_revision2/main.tex").read_text(),
    )
    title_equal = bool(title and old_title and title.group(1) == old_title.group(1))
    suites = ET.parse(REPORT / "final_tests.xml").getroot()
    tests = {
        key: sum(int(s.get(key, "0")) for s in suites.iter("testsuite"))
        for key in ("tests", "failures", "errors", "skipped")
    }
    pdfs = {stem: pdf_audit(stem) for stem in ("main", "technical_supplement")}
    visual_path = REPORT / "visual_review.json"
    visual = json.loads(visual_path.read_text()) if visual_path.exists() else {}
    visual_bound = all(
        visual.get("pdfs", {}).get(stem, {}).get("sha256") == item["sha256"]
        and visual.get("pdfs", {}).get(stem, {}).get("pages_inspected")
        == list(range(1, item["pages"] + 1))
        for stem, item in pdfs.items()
    ) and visual.get("pass", False)
    residual_seal = ROOT / "configs/cscwd2027/revision2/residual_probe_freeze.json"
    residual = json.loads(residual_seal.read_text())
    residual_unchanged = digest(residual_seal) == residual_seal.with_suffix(
        ".sha256"
    ).read_text().strip() and all(
        digest(ROOT / name) == sha for name, sha in residual["members"].items()
    )
    passed = all(
        (
            summary["data_complete"],
            not summary["failed"],
            not summary["integrity_findings"],
            not changed_raw,
            activity_matches,
            generated_bound,
            all(inherited.values()),
            not overrides,
            not hidden,
            not cited - bib,
            template_equal,
            title_equal,
            not tests["failures"],
            not tests["errors"],
            not tests["skipped"],
            tests["tests"] >= 106,
            all(d["pass"] for d in pdfs.values()),
            visual_bound,
            residual_unchanged,
        )
    )
    value = {
        "status": "READY FOR HUMAN REVIEW" if passed else "NOT READY",
        "review_kind": "three simulated lenses, not external peer review",
        "external_submission_cleared": False,
        "original_cscwd_protection": old_protection,
        "revision2_protection": revision2_protection,
        "original_formal_attempts": summary["attempts"],
        "new_simulation_attempts": 0,
        "changed_raw_files": changed_raw,
        "posthoc_activity_recomputes": activity_matches,
        "activity_input_sha256": digest(activity_path),
        "generated_numbers_bound": generated_bound,
        "inherited_presentation_unchanged": inherited,
        "official_ieeetran_unchanged": template_equal,
        "title_unchanged": title_equal,
        "layout_overrides": overrides,
        "hidden_instruction_findings": hidden,
        "unknown_citations": sorted(cited - bib),
        "citation_count": len(cited),
        "tests": tests,
        "pdfs": pdfs,
        "visual_review_bound": visual_bound,
        "residual_probe_seal_unchanged": residual_unchanged,
    }
    (REPORT / "delivery_audit.json").write_text(json.dumps(value, indent=2) + "\n")
    print(
        json.dumps(
            {k: value[k] for k in ("status", "tests", "new_simulation_attempts")}
        )
    )
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
