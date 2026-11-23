"""Assemble a new closed review package, retaining all old packages unchanged."""

from __future__ import annotations

import json
import shutil
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any

from .protection import ROOT, digest, verify

DEST = ROOT / "release/cscwd2027/review_20261025_final_review_v1"
PREVIOUS = ROOT / "release/cscwd2027/review_20261025_revision2_v1"
PAPER = ROOT / "paper/cscwd2027_final_review"


def ordinary(name: str) -> None:
    """Reject non-relative or traversing archive members."""
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or "\\" in name:
        raise ValueError("unsafe archive member")


def append_files(archive: zipfile.ZipFile, files: list[Path]) -> None:
    """Add local ordinary files without duplicates or generated Python caches."""
    names = set(archive.namelist())
    for path in sorted(set(files)):
        name = str(path.relative_to(ROOT))
        ordinary(name)
        if name in names:
            raise ValueError(f"duplicate archive member: {name}")
        names.add(name)
        archive.write(path, name)


def main() -> None:
    """Package only after audits pass, never overwriting another release."""
    verify()
    audit = json.loads(
        (ROOT / "reports/cscwd2027/final_review/delivery_audit.json").read_text()
    )
    if audit["status"] != "READY FOR HUMAN REVIEW":
        raise RuntimeError("passed final review audit required")
    DEST.mkdir(parents=True, exist_ok=False)
    for name in ("main", "technical_supplement"):
        shutil.copyfile(PAPER / f"{name}.pdf", DEST / f"{name}.pdf")
    manuscript = [
        p
        for p in PAPER.rglob("*")
        if p.is_file()
        and p.suffix in {".tex", ".bib", ".cls", ".pdf", ".svg", ".json", ".md"}
        and p not in (PAPER / "main.pdf", PAPER / "technical_supplement.pdf")
    ]
    with zipfile.ZipFile(
        DEST / "manuscript_source.zip", "w", zipfile.ZIP_DEFLATED
    ) as archive:
        append_files(archive, manuscript)
    base = PREVIOUS / "source_and_protocol.zip"
    additions = [
        p
        for prefix in ("scripts/cscwd2027_final_review", "tests/cscwd2027_final_review")
        for p in (ROOT / prefix).rglob("*.py")
    ]
    additions += [
        ROOT / "docs/cscwd2027/final_review.md",
        ROOT / "results/cscwd2027/final_review/derived/activity_summary.json",
    ]
    with (
        zipfile.ZipFile(base) as original,
        zipfile.ZipFile(
            DEST / "source_and_analysis.zip", "w", zipfile.ZIP_DEFLATED
        ) as archive,
    ):
        for name in original.namelist():
            ordinary(name)
            archive.writestr(name, original.read(name))
        append_files(archive, additions)
    redactions: dict[str, Any] = {}
    with zipfile.ZipFile(
        DEST / "review_and_audit.zip", "w", zipfile.ZIP_DEFLATED
    ) as archive:
        for path in sorted((ROOT / "reports/cscwd2027/final_review").iterdir()):
            if path.suffix not in {".json", ".md", ".xml"}:
                continue
            name = str(path.relative_to(ROOT))
            if path.suffix == ".xml":
                element = ET.fromstring(path.read_bytes())
                for suite in element.iter("testsuite"):
                    suite.attrib.pop("hostname", None)
                archive.writestr(name, ET.tostring(element, encoding="utf-8"))
                redactions[name] = {
                    "original_sha256": digest(path),
                    "removed": "hostname",
                }
            else:
                archive.write(path, name)
        review = next((PAPER / "ccfa-review-reports").glob("*-conference-review.md"))
        archive.write(review, str(review.relative_to(ROOT)))
    guide = (ROOT / "docs/cscwd2027/final_review.md").read_text()
    (DEST / "REVIEW_GUIDE.md").write_text(guide)
    payloads = {p.name: digest(p) for p in sorted(DEST.iterdir()) if p.is_file()}
    manifest = {
        "status": audit["status"],
        "author_mode": "anonymous author review, not a portal-policy assertion",
        "payloads": payloads,
        "inherited_source_archive_sha256": digest(base),
        "required_unchanged_results_archive": {
            "relative_path": (
                "../review_20261025_revision2_v1/revision2_complete_results.zip"
            ),
            "sha256": digest(PREVIOUS / "revision2_complete_results.zip"),
        },
        "old_packages_overwritten": False,
        "new_simulation_attempts": 0,
        "posthoc_analysis": "all existing extension references, no new endpoints",
        "junit_metadata_redactions": redactions,
        "external_submission_cleared": False,
        "supporting_upload_permission": "not confirmed",
    }
    (DEST / "MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (DEST / "SHA256SUMS.txt").write_text(
        "".join(
            f"{digest(p)}  {p.name}\n"
            for p in sorted(DEST.iterdir())
            if p.is_file() and p.name != "SHA256SUMS.txt"
        )
    )
    print(
        json.dumps({"package": str(DEST.relative_to(ROOT)), "payloads": len(payloads)})
    )


if __name__ == "__main__":
    main()
