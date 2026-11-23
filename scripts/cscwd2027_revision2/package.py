"""Export a new anonymous internal review package, without touching v1 releases."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import xml.etree.ElementTree as ET
import zipfile
from typing import Any

from scripts.cscwd2027_revision2.freeze import ROOT, check, digest


def add(archive: zipfile.ZipFile, name: str, data: bytes) -> None:
    """Add one relative ordinary file with a deterministic archive timestamp."""
    item = zipfile.ZipInfo(name, (2026, 10, 25, 0, 0, 0))
    item.compress_type = zipfile.ZIP_DEFLATED
    item.external_attr = 0o100644 << 16
    archive.writestr(item, data)


def main() -> None:
    """Archive scoped source, all new attempts, and the rebuilt manuscript."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="review_20261025_revision2_v1")
    args = parser.parse_args()
    if not re.fullmatch(r"review_[A-Za-z0-9_-]+", args.name):
        raise ValueError("expected a single review_* directory name")
    freeze = check()
    report = ROOT / "reports/cscwd2027/revision2"
    audit = json.loads((report / "delivery_audit.json").read_text())
    visual = json.loads((report / "visual_review.json").read_text())
    if not audit["pass"] or not visual["pass"]:
        raise RuntimeError("complete audit and visual review required")
    paper = ROOT / "paper/cscwd2027_revision2"
    if any(
        visual[stem]["sha256"] != digest(paper / f"{stem}.pdf")
        for stem in ("main", "technical_supplement")
    ):
        raise RuntimeError("visual review is stale")
    destination = ROOT / "release/cscwd2027" / args.name
    destination.mkdir(parents=True, exist_ok=False)
    old = ROOT / "release/cscwd2027/review_20261025_v1"
    # Reuse the exact old complete data archive rather than rewriting its members.
    shutil.copy2(
        old / "cscwd_complete_results.zip",
        destination / "original_v1_complete_results.zip",
    )
    inherited = old / "cscwd_source_and_protocol.zip"
    extras = {ROOT / name for name in freeze["members"]}
    for prefix in (
        "scripts/cscwd2027_revision2",
        "src/policytwin/coordination_validation",
        "tests/cscwd2027_revision2",
    ):
        extras.update((ROOT / prefix).glob("*.py"))
    extras.update((ROOT / "configs/cscwd2027/revision2").glob("*"))
    extras.update((ROOT / "docs/cscwd2027/revision2").glob("*.md"))
    extras.update(report.glob("*.xml"))
    data_manifest = {}
    redactions = {}
    inherited_names = set()
    with zipfile.ZipFile(
        destination / "source_and_protocol.zip", "x", zipfile.ZIP_DEFLATED
    ) as archive:
        with zipfile.ZipFile(inherited) as original:
            for member in original.infolist():
                inherited_names.add(member.filename)
                add(archive, member.filename, original.read(member))
        for path in sorted(extras):
            if not path.is_file():
                continue
            name = str(path.relative_to(ROOT))
            if name in inherited_names:
                with zipfile.ZipFile(inherited) as original:
                    if original.read(name) != path.read_bytes():
                        raise RuntimeError("inherited frozen member differs")
                continue
            data = path.read_bytes()
            if path.suffix == ".xml":
                tree = ET.fromstring(data)
                for suite in tree.iter("testsuite"):
                    suite.attrib.pop("hostname", None)
                cleaned = ET.tostring(tree, encoding="utf-8", xml_declaration=True)
                redactions[name] = {
                    "original_sha256": digest(path),
                    "review_copy_sha256": hashlib.sha256(cleaned).hexdigest(),
                    "operation": "remove JUnit hostname only",
                }
                data = cleaned
            if re.search(rb"/Users/[A-Za-z0-9_.-]+/|\bsk-[A-Za-z0-9_-]{24,}\b", data):
                raise RuntimeError(f"source privacy pattern: {name}")
            add(archive, name, data)
            data_manifest[name] = hashlib.sha256(data).hexdigest()
        add(
            archive,
            "REVISION2_README.md",
            (ROOT / "docs/cscwd2027/revision2/reproduction.md").read_bytes(),
        )
    with zipfile.ZipFile(
        destination / "revision2_complete_results.zip", "x", zipfile.ZIP_DEFLATED
    ) as archive:
        for path in sorted((ROOT / "results/cscwd2027/revision2").rglob("*")):
            if (
                path.is_file()
                and "baseline"
                not in path.relative_to(ROOT / "results/cscwd2027/revision2").parts
            ):
                add(archive, str(path.relative_to(ROOT)), path.read_bytes())
    with zipfile.ZipFile(
        destination / "manuscript_source.zip", "x", zipfile.ZIP_DEFLATED
    ) as archive:
        for path in sorted(paper.rglob("*")):
            if path.is_file() and path.suffix in {
                ".tex",
                ".bib",
                ".cls",
                ".pdf",
                ".svg",
                ".json",
                ".md",
            }:
                add(archive, str(path.relative_to(ROOT)), path.read_bytes())
    with zipfile.ZipFile(
        destination / "internal_audit_reports.zip", "x", zipfile.ZIP_DEFLATED
    ) as archive:
        for path in sorted(report.rglob("*")):
            if path.is_file() and path.suffix in {".md", ".json"}:
                add(archive, str(path.relative_to(ROOT)), path.read_bytes())
    for name in ("main.pdf", "technical_supplement.pdf"):
        shutil.copy2(paper / name, destination / name)
    guide = """# Anonymous CSCWD revision2 author-review package

The main paper is at most six pages, including references and figures.
The technical supplement, source archives and complete result archives are
internal review materials. Portal permission and size limits are unconfirmed.
No external upload, public code release or author sign-off has been performed.

- main.pdf and technical_supplement.pdf: current anonymous review documents.
- manuscript_source.zip: editable LaTeX, figures and generated tables.
- source_and_protocol.zip: unchanged inherited v1 source plus separate r2 code.
- original_v1_complete_results.zip: byte-identical original result archive.
- revision2_complete_results.zip: every new formal attempt and engineering pilot.
- internal_audit_reports.zip: evidence, negative cases and remaining author checks.

Read REVISION2_README.md in the source archive for exact validation commands.
Do not infer that this collection is a set of permitted conference attachments.
"""
    (destination / "UPLOAD_GUIDE.md").write_text(guide)
    manifest: dict[str, Any] = {
        "purpose": "anonymous author review; no external submission clearance",
        "new_freeze_sha256": digest(ROOT / "configs/cscwd2027/revision2/freeze.json"),
        "inherited_source_archive_sha256": digest(inherited),
        "old_results_byte_identical": digest(
            destination / "original_v1_complete_results.zip"
        )
        == digest(old / "cscwd_complete_results.zip"),
        "added_source_members": data_manifest,
        "review_copy_redactions": redactions,
        "files": {
            p.name: {"bytes": p.stat().st_size, "sha256": digest(p)}
            for p in sorted(destination.iterdir())
            if p.is_file()
        },
    }
    (destination / "MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (destination / "SHA256SUMS.txt").write_text(
        "\n".join(
            f"{digest(p)}  {p.name}"
            for p in sorted(destination.iterdir())
            if p.is_file()
        )
        + "\n"
    )
    print(
        json.dumps(
            {
                "destination": str(destination.relative_to(ROOT)),
                "files": len(manifest["files"]),
            }
        )
    )


if __name__ == "__main__":
    main()
