"""Machine checks for the IEEE manuscript; visual and author review stay explicit."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
PAPER = ROOT / "paper/cscwd2027"


def command(arguments: list[str]) -> str:
    """Capture an exact local diagnostic without editing the inspected files."""
    completed = subprocess.run(
        arguments, cwd=ROOT, capture_output=True, text=True, check=True
    )
    return completed.stdout


def digest(path: Path) -> str:
    """Content identity used by generated presentation manifests."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pdf_checks(stem: str) -> dict[str, Any]:
    """Inspect pages, embedded fonts, raster images, text and compilation warnings."""
    path = PAPER / f"{stem}.pdf"
    info = command(["pdfinfo", str(path)])
    page_match = re.search(r"^Pages:\s+(\d+)$", info, re.M)
    if page_match is None:
        raise RuntimeError("pdfinfo did not return a page count")
    pages = int(page_match.group(1))
    fonts = command(["pdffonts", str(path)])
    rows = fonts.splitlines()[2:]
    unembedded = [row.split()[0] for row in rows if row.split()[-5] != "yes"]
    type3 = [row.split()[0] for row in rows if "Type 3" in row]
    text = command(["pdftotext", "-layout", str(path), "-"])
    log = (PAPER / f"{stem}.log").read_text(errors="replace")
    errors = [line for line in log.splitlines() if line.startswith("!")]
    warnings = [
        line
        for line in log.splitlines()
        if "Overfull" in line or "undefined" in line.lower() or "Rerun to get" in line
    ]
    privacy = {
        "personal_path": bool(re.search(r"/Users/[A-Za-z0-9_.-]+/", text + info)),
        "anonymous_author_block_missing": "Anonymous authors" not in text,
        "email_address": bool(
            re.search(
                r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b", text + info
            )
        ),
        "credential_shape": bool(re.search(r"\bsk-[A-Za-z0-9_-]{24,}\b", text + info)),
    }
    image_list = command(["pdfimages", "-list", str(path)])
    images = image_list.splitlines()[2:]
    return {
        "sha256": digest(path),
        "bytes": path.stat().st_size,
        "pages": pages,
        "pdfinfo": info,
        "pdffonts": fonts,
        "raster_image_rows": images,
        "unembedded_fonts": unembedded,
        "type3_fonts": type3,
        "compile_errors": errors,
        "unresolved_or_overfull": warnings,
        "privacy_findings": privacy,
        "checks_pass": not (
            unembedded or type3 or errors or warnings or any(privacy.values())
        )
        and (pages <= 6 if stem == "main" else pages > 0),
    }


def main() -> None:
    """Check actual outputs, never infer successful compilation from source alone."""
    summary = json.loads(
        (ROOT / "results/cscwd2027/derived_v1/summary.json").read_text()
    )
    if not summary["data_complete"]:
        raise RuntimeError("paper audit requires complete formal data")
    original_class = Path(command(["kpsewhich", "IEEEtran.cls"]).strip())
    class_equal = digest(PAPER / "IEEEtran.cls") == digest(original_class)
    sources = [PAPER / "main.tex", PAPER / "technical_supplement.tex"]
    sources += list((PAPER / "sections").glob("*.tex"))
    forbidden = re.compile(
        r"\\(?:geometry|fontsize|baselinestretch|linespread|vspace|enlargethispage)\b"
        r"|\\setlength\s*\{\\(?:textwidth|textheight|columnsep|topmargin|oddsidemargin)"
    )
    overrides = [
        str(p.relative_to(ROOT)) for p in sources if forbidden.search(p.read_text())
    ]
    manifest_findings = []
    for name in (
        "figure_manifest.json",
        "presentation_manifest.json",
        "publication_figure_manifest.json",
    ):
        manifest = json.loads((PAPER / "generated" / name).read_text())
        if (
            "input" in manifest
            and digest(ROOT / manifest["input"]) != manifest["input_sha256"]
        ):
            manifest_findings.append(f"{name}: input changed")
        for path, sha in manifest.get("inputs", {}).items():
            if digest(ROOT / path) != sha:
                manifest_findings.append(f"{name}: input {path}")
        for path, sha in manifest["outputs"].items():
            if digest(ROOT / path) != sha:
                manifest_findings.append(f"{name}: output {path}")
        if (
            "protocol_diagram_source" in manifest
            and digest(ROOT / manifest["protocol_diagram_source"])
            != manifest["protocol_diagram_source_sha256"]
        ):
            manifest_findings.append(f"{name}: protocol source changed")
    macros = dict(
        re.findall(
            r"\\newcommand\{\\([A-Za-z]+)\}\{([^{}]+)\}",
            (PAPER / "generated/numbers.tex").read_text(),
        )
    )
    expected = {
        "StudyAttempts": summary["recorded_attempts"],
        "PartitionPairs": summary["e1"]["attempts"],
        "PartitionEqual": summary["e1"]["trace_equal"],
        "FaultAttempts": summary["e2"]["attempts"],
        "FaultTriggered": summary["e2"]["triggered"],
        "FullInvalidReleases": summary["full_invalid_releases"]["count"],
        "NumericalChanged": summary["e2"]["numerical_changed"],
        "NumericalUnchanged": summary["e2"]["numerically_unchanged"],
        "IncompleteFaults": summary["e2"]["incomplete"],
        "ControlPairs": sum(c["attempts"] for c in summary["controls"].values()),
        "ControlAccepted": sum(
            c["modes"]["full"]["count"] for c in summary["controls"].values()
        ),
    }
    number_match = macros == {k: str(v) for k, v in expected.items()}
    bibliography = set(
        re.findall(r"@\w+\{([^,]+),", (PAPER / "references.bib").read_text())
    )
    citations = set()
    for path in sources:
        for group in re.findall(r"\\cite\{([^}]+)\}", path.read_text()):
            citations.update(group.split(","))
    unknown_citations = sorted(citations - bibliography)
    pdfs = {stem: pdf_checks(stem) for stem in ("main", "technical_supplement")}
    machine_pass = (
        class_equal
        and not overrides
        and not manifest_findings
        and number_match
        and not unknown_citations
        and all(p["checks_pass"] for p in pdfs.values())
    )
    report = {
        "mode": "venue-format, package-check and numerical binding",
        "official_main_page_limit": 6,
        "template_unchanged": class_equal,
        "template_sha256": digest(PAPER / "IEEEtran.cls"),
        "layout_override_files": overrides,
        "manifests_findings": manifest_findings,
        "result_macros_match": number_match,
        "unknown_citations": unknown_citations,
        "pdfs": pdfs,
        "machine_checks_pass": machine_pass,
        "visual_review": "Separate page-by-page inspection record required",
        "deliverable": "author-requested anonymous review draft",
        "external_submission_cleared": False,
        "official_rules": ["https://cscwd2027.dailyeliteevents.com.au/submission.html"],
    }
    (ROOT / "reports/cscwd2027/manuscript_machine_audit.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(
        json.dumps(
            {
                "machine_checks_pass": machine_pass,
                "pages": {k: v["pages"] for k, v in pdfs.items()},
                "template_unchanged": class_equal,
                "result_macros_match": number_match,
                "warnings": {k: v["unresolved_or_overfull"] for k, v in pdfs.items()},
            }
        )
    )
    if not machine_pass:
        raise RuntimeError("manuscript checks failed; inspect retained audit")


if __name__ == "__main__":
    main()
