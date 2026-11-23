"""Create a separate closed author-review package; never overwrite old releases."""

from __future__ import annotations

import json
import shutil
import xml.etree.ElementTree as ET
import zipfile

from scripts.cscwd2027_design_contract.deliver import PAPER, REPORT
from scripts.cscwd2027_design_contract.protection import ROOT, digest, verify
from scripts.cscwd2027_design_contract.store import RESULTS
from scripts.cscwd2027_final_review.package import append_files, ordinary

DEST = ROOT / "release/cscwd2027/review_20261025_design_contract_v2"
PREVIOUS = ROOT / "release/cscwd2027/review_20261025_final_review_v1"


def checksums() -> None:
    """Bind every top-level payload except the checksum file itself."""
    (DEST / "SHA256SUMS.txt").write_text(
        "".join(
            f"{digest(p)}  {p.name}\n"
            for p in sorted(DEST.iterdir())
            if p.is_file() and p.name != "SHA256SUMS.txt"
        )
    )


def main() -> None:
    """Retain all raw attempts, actual outputs, source and anonymous PDFs."""
    verify()
    audit = json.loads((REPORT / "delivery_audit.json").read_text())
    if audit["status"] != "READY FOR HUMAN REVIEW":
        raise RuntimeError("passed delivery audit required")
    DEST.mkdir(parents=True, exist_ok=False)
    for name in ("main", "technical_supplement"):
        shutil.copyfile(PAPER / f"{name}.pdf", DEST / f"{name}.pdf")
    files = [
        p
        for p in PAPER.rglob("*")
        if p.is_file()
        and p.suffix in {".tex", ".bib", ".cls", ".pdf", ".svg", ".json"}
        and p not in (PAPER / "main.pdf", PAPER / "technical_supplement.pdf")
    ]
    with zipfile.ZipFile(
        DEST / "manuscript_source.zip", "x", zipfile.ZIP_DEFLATED
    ) as z:
        append_files(z, files)
    additions = [
        p
        for prefix in (
            "src/policytwin/design_contract",
            "scripts/cscwd2027_design_contract",
            "tests/cscwd2027_design_contract",
        )
        for p in (ROOT / prefix).glob("*.py")
    ]
    additions += [
        p
        for p in (ROOT / "configs/cscwd2027/design_contract_v1").iterdir()
        if p.is_file()
    ]
    additions += [
        ROOT / "AGENTS.md",
        ROOT / "docs/cscwd2027/design_contract_v1_protocol.md",
        ROOT / "docs/cscwd2027/design_contract_review.md",
    ]
    base = PREVIOUS / "source_and_analysis.zip"
    with (
        zipfile.ZipFile(base) as old,
        zipfile.ZipFile(
            DEST / "source_and_analysis.zip", "x", zipfile.ZIP_DEFLATED
        ) as z,
    ):
        for name in old.namelist():
            ordinary(name)
            z.writestr(name, old.read(name))
        append_files(z, additions)
    with zipfile.ZipFile(
        DEST / "design_contract_complete_results.zip", "x", zipfile.ZIP_DEFLATED
    ) as z:
        append_files(z, [p for p in RESULTS.rglob("*") if p.is_file()])
    redactions = {}
    with zipfile.ZipFile(DEST / "review_and_audit.zip", "x", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(REPORT.iterdir()):
            if p.suffix not in {".json", ".md", ".xml"}:
                continue
            name = str(p.relative_to(ROOT))
            if p.suffix == ".xml":
                xml = ET.fromstring(p.read_bytes())
                for suite in xml.iter("testsuite"):
                    suite.attrib.pop("hostname", None)
                z.writestr(name, ET.tostring(xml, encoding="utf-8"))
                redactions[name] = {"original_sha256": digest(p), "removed": "hostname"}
            else:
                z.write(p, name)
        review = (
            PAPER
            / "ccfa-review-reports"
            / "2026-10-25-comparison-contract-cscwd2027-conference-review.md"
        )
        z.write(review, str(review.relative_to(ROOT)))
    shutil.copyfile(
        ROOT / "docs/cscwd2027/design_contract_review.md", DEST / "REVIEW_GUIDE.md"
    )
    prior = (
        ROOT
        / "release/cscwd2027/review_20261025_revision2_v1"
        / "revision2_complete_results.zip"
    )
    value = {
        "status": audit["status"],
        "author_mode": "anonymous author review, no upload permission asserted",
        "new_attempts": audit["actual_attempts"],
        "payloads": {p.name: digest(p) for p in sorted(DEST.iterdir()) if p.is_file()},
        "inherited_source_archive_sha256": digest(base),
        "required_prior_results": {
            "relative_path": str(prior.relative_to(ROOT)),
            "sha256": digest(prior),
        },
        "junit_archive_redactions": redactions,
        "original_artifacts_overwritten": False,
        "new_raw_results_complete": True,
        "prior_failed_package_retained": "review_20261025_design_contract_v1",
        "external_submission_cleared": False,
        "supporting_upload_permission": "not established",
    }
    (DEST / "MANIFEST.json").write_text(json.dumps(value, indent=2) + "\n")
    checksums()
    print(
        json.dumps(
            {
                "package": str(DEST.relative_to(ROOT)),
                "new_attempts": value["new_attempts"],
            }
        )
    )


if __name__ == "__main__":
    main()
