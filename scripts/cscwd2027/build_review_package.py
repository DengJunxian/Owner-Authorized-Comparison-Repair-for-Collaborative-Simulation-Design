"""Build scoped internal review archives, preserving original experimental bytes."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import shutil
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path
from typing import Any

from scripts.cscwd2027.freeze_study import ROOT, check


def digest(path: Path) -> str:
    """Hash a file without retaining its entire contents in memory."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def local_module(name: str) -> Path | None:
    """Resolve only known project Python modules, without importing them."""
    if not name.startswith("policytwin"):
        return None
    prefix = ROOT / "src" / name.replace(".", "/")
    for path in (prefix.with_suffix(".py"), prefix / "__init__.py"):
        if path.is_file():
            return path
    return None


def source_closure(seeds: set[Path]) -> set[Path]:
    """Conservatively include local import dependencies and package initializers."""
    found = set(seeds)
    pending = list(seeds)
    while pending:
        path = pending.pop()
        if path.suffix != ".py":
            continue
        candidates = set()
        for parent in path.parents:
            if parent == ROOT:
                break
            initializer = parent / "__init__.py"
            if initializer.is_file():
                candidates.add(initializer)
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
                names = [node.module] + [
                    f"{node.module}.{alias.name}" for alias in node.names
                ]
            for name in names:
                module = local_module(name)
                if module:
                    candidates.add(module)
        new = candidates - found
        found.update(new)
        pending.extend(new)
    return found


def inventory() -> tuple[set[Path], dict[str, Any]]:
    """Identify source/support bytes and their pre-run or post-run authority."""
    freeze = check()
    frozen = {ROOT / name for name in freeze["members"]}
    seeds = frozen | set((ROOT / "scripts/cscwd2027").glob("*.py"))
    seeds |= set((ROOT / "tests/cscwd2027").glob("*.py"))
    seeds.add(ROOT / "tests/conftest.py")
    # The original policy fixture imports a package initializer that reads these
    # existing states. Include their protected bytes, without using them as new
    # CSCWD observations or modifying the frozen test's import path.
    package_data = {
        ROOT / "src/policytwin" / name
        for name in (
            "exact_finite_scenario_selection.json",
            "exact_finite_scenario_selection_pending.json",
        )
    }
    seeds |= package_data
    closure = source_closure(seeds)
    baseline = json.loads(
        (ROOT / "results/cscwd2027/baseline/protected_files.json").read_text()
    )
    earlier = {m["path"]: m["sha256"] for m in baseline["members"]}
    support = {}
    post_freeze = {}
    for path in sorted(closure - frozen):
        name = str(path.relative_to(ROOT))
        sha = digest(path)
        if name in earlier:
            if sha != earlier[name]:
                raise RuntimeError(f"protected support source changed: {name}")
            support[name] = sha
        elif name.startswith(("scripts/cscwd2027/", "tests/cscwd2027/")):
            post_freeze[name] = sha
        else:
            raise RuntimeError(f"support source lacks a recorded authority: {name}")
    record = {
        "frozen_members": len(frozen),
        "source_and_dependency_members": len(closure),
        "source_bytes": sum(p.stat().st_size for p in closure),
        "earlier_baseline_captured_at_utc": baseline["captured_at_utc"],
        "earlier_baseline_manifest_sha256": digest(
            ROOT / "results/cscwd2027/baseline/protected_files.json"
        ),
        "support_bound_by_earlier_baseline": support,
        "post_freeze_audit_presentation_packaging": post_freeze,
        "transitive_package_initialization_data": [
            str(p.relative_to(ROOT)) for p in sorted(package_data)
        ],
        "scope": (
            "Frozen scientific source and conservative import closure; "
            "no prior AAAI/DAI result trees, private git history or environment files"
        ),
    }
    return closure, record


def archive(
    path: Path, members: set[Path], extras: dict[str, bytes] | None = None
) -> None:
    """Write a new relative-path archive with reproducible member timestamps."""
    with zipfile.ZipFile(path, "x", zipfile.ZIP_DEFLATED) as out:
        for member in sorted(members):
            if member.is_symlink() or not member.is_file():
                raise RuntimeError("package member must be an ordinary file")
            name = str(member.relative_to(ROOT))
            if extras and name in extras:
                continue
            info = zipfile.ZipInfo(name, (2026, 10, 25, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            out.writestr(info, member.read_bytes())
        for name, content in sorted((extras or {}).items()):
            info = zipfile.ZipInfo(name, (2026, 10, 25, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            out.writestr(info, content)


def scan(members: set[Path]) -> list[dict[str, str]]:
    """Find personal paths and credential-shaped strings without printing values."""
    patterns = {
        "personal_path": re.compile(rb"/Users/[A-Za-z0-9_.-]+/"),
        "private_key": re.compile(
            rb"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"
        ),
        "credential_shape": re.compile(
            rb"\b(?:sk-[A-Za-z0-9_-]{24,}|ghp_[A-Za-z0-9]{24,}|AKIA[0-9A-Z]{16})\b"
        ),
    }
    findings = []
    for path in sorted(members):
        if path.suffix in {".zip", ".gz", ".pdf", ".png"}:
            continue
        content = path.read_bytes()
        for name, pattern in patterns.items():
            if pattern.search(content):
                findings.append({"path": str(path.relative_to(ROOT)), "kind": name})
    return findings


def main() -> None:
    """Prepare files for author review; never upload, publish or withdraw anything."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--inventory-only", action="store_true")
    parser.add_argument("--directory", default="review_20261025_v1")
    args = parser.parse_args()
    sources, source_record = inventory()
    source_scan = scan(sources)
    source_record["sensitive_pattern_findings"] = source_scan
    report_root = ROOT / "reports/cscwd2027"
    (report_root / "package_source_inventory.json").write_text(
        json.dumps(source_record, indent=2) + "\n"
    )
    if args.inventory_only:
        print(
            json.dumps(
                {
                    "members": len(sources),
                    "source_bytes": source_record["source_bytes"],
                    "pattern_findings": source_scan,
                }
            )
        )
        return
    if source_scan:
        raise RuntimeError("source privacy scan has findings; preserve and inspect")
    integrity = json.loads((report_root / "artifact_integrity_audit.json").read_text())
    if not integrity["all_files_bound"]:
        raise RuntimeError("complete artifact integrity audit required")
    paper_root = ROOT / "paper/cscwd2027"
    for name in ("main.pdf", "technical_supplement.pdf"):
        if not (paper_root / name).is_file():
            raise RuntimeError(f"missing compiled document: {name}")
    if not re.fullmatch(r"review_[A-Za-z0-9_-]+", args.directory):
        raise ValueError("review directory must be one plain review_* name")
    destination = ROOT / "release/cscwd2027" / args.directory
    destination.mkdir(parents=True, exist_ok=False)
    sources |= set((ROOT / "configs/cscwd2027").glob("*"))
    sources |= set((ROOT / "docs/cscwd2027").glob("*.md"))
    sources |= {
        report_root / name
        for name in (
            "pre_freeze_tests.xml",
            "post_freeze_binding_tests.xml",
            "final_tests.xml",
            "tcb_audit.json",
            "hardware_addendum.json",
            "package_source_inventory.json",
        )
    }
    readme = (ROOT / "docs/cscwd2027/reproduction.md").read_bytes()
    extras = {"README.md": readme}
    redactions = {}
    for name in (
        "pre_freeze_tests.xml",
        "post_freeze_binding_tests.xml",
        "final_tests.xml",
    ):
        path = report_root / name
        tree = ET.fromstring(path.read_bytes())
        for suite in tree.iter("testsuite"):
            suite.attrib.pop("hostname", None)
        cleaned = ET.tostring(tree, encoding="utf-8", xml_declaration=True)
        relative = str(path.relative_to(ROOT))
        extras[relative] = cleaned
        redactions[relative] = {
            "original_sha256": digest(path),
            "review_copy_sha256": hashlib.sha256(cleaned).hexdigest(),
            "operation": "remove JUnit hostname only; all counts and cases retained",
        }
    archive(destination / "cscwd_source_and_protocol.zip", sources, extras)
    raw = {p for p in (ROOT / "results/cscwd2027/study_v1").rglob("*") if p.is_file()}
    raw |= {p for p in (ROOT / "results/cscwd2027/derived_v1").glob("*") if p.is_file()}
    archive(destination / "cscwd_complete_results.zip", raw)
    manuscript = {
        p
        for p in paper_root.rglob("*")
        if p.is_file()
        and p.suffix in {".tex", ".bib", ".cls", ".pdf", ".svg", ".json", ".md"}
    }
    archive(destination / "cscwd_manuscript_source.zip", manuscript)
    for name in ("main.pdf", "technical_supplement.pdf"):
        shutil.copy2(paper_root / name, destination / name)
    reports = {
        p for p in report_root.glob("*") if p.is_file() and p.suffix in {".md", ".json"}
    }
    archive(destination / "internal_audit_reports.zip", reports)
    guide = (
        "# CSCWD 2027 internal review package\n\n"
        "Anonymous author-review package, requested by the author. "
        "It is not a submitted or submission-approved packet. "
        "Author details, related-paper disclosures and portal rules remain "
        "pre-upload checks.\n\n"
        "- main.pdf: six-page IEEE-format candidate main paper.\n"
        "- technical_supplement.pdf: internal technical support; "
        "upload permission unconfirmed.\n"
        "- cscwd_manuscript_source.zip: editable LaTeX, references, "
        "vector figures and manifests.\n"
        "- cscwd_source_and_protocol.zip: implementation, frozen design, "
        "dependencies and audit scripts.\n"
        "- cscwd_complete_results.zip: every formal attempt, "
        "including negative outcomes and warm-ups.\n"
        "- internal_audit_reports.zip: internal review evidence, "
        "not a conference attachment requirement.\n\n"
        "Only the main PDF upload is established by the public conference page. "
        "Do not import AAAI page, size or attachment rules into CSCWD. "
        "No file has been submitted by this task. Read README.md in the source archive "
        "for reproduction commands and licensing/author-review boundaries.\n"
    )
    (destination / "UPLOAD_GUIDE.md").write_text(guide)
    records = {
        path.name: {"bytes": path.stat().st_size, "sha256": digest(path)}
        for path in sorted(destination.iterdir())
        if path.is_file()
    }
    manifest = {
        "purpose": "internal author review; no external submission",
        "files": records,
        "source_inventory": source_record,
        "review_copy_redactions": redactions,
        "raw_attempt_count": integrity["verified_attempts"],
        "prior_submission_documents_or_result_trees_included": False,
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
            {"destination": str(destination.relative_to(ROOT)), "files": len(records)}
        )
    )


if __name__ == "__main__":
    main()
