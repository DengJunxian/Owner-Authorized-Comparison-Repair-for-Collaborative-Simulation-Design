"""Validate the closed final-review package in an isolated same-host extraction."""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Any

from .package import DEST, PREVIOUS, ordinary
from .protection import ROOT, digest


def canonical_hash(path: Path) -> str:
    """Compare parsed JSON independently of whitespace and object-key order."""
    return hashlib.sha256(
        json.dumps(json.loads(path.read_text()), sort_keys=True).encode()
    ).hexdigest()


def unpack(path: Path, root: Path) -> None:
    """Extract ordinary files once, refusing collisions and path traversal."""
    with zipfile.ZipFile(path) as archive:
        for member in archive.infolist():
            ordinary(member.filename)
            target = root / member.filename
            if member.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            if (member.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError("symbolic-link archive member")
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("xb") as stream:
                stream.write(archive.read(member))


def main() -> None:
    """Check hashes, scoped tests, descriptive recomputation and both PDF builds."""
    checksums = {}
    for line in (DEST / "SHA256SUMS.txt").read_text().splitlines():
        expected, name = line.split("  ", 1)
        if Path(name).name != name or digest(DEST / name) != expected:
            raise RuntimeError(f"package hash mismatch: {name}")
        checksums[name] = expected
    manifest = json.loads((DEST / "MANIFEST.json").read_text())
    raw_archive = PREVIOUS / "revision2_complete_results.zip"
    if digest(raw_archive) != manifest["required_unchanged_results_archive"]["sha256"]:
        raise RuntimeError("prior raw result archive changed")
    findings = []
    sensitive = re.compile(
        rb"/Users/[A-Za-z0-9_.-]+/|\bsk-[A-Za-z0-9_-]{24,}"
        rb"|-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"
    )
    for archive_path in DEST.glob("*.zip"):
        with zipfile.ZipFile(archive_path) as archive:
            for name in archive.namelist():
                ordinary(name)
                if Path(name).suffix in {
                    ".py",
                    ".json",
                    ".md",
                    ".tex",
                    ".bib",
                    ".xml",
                } and sensitive.search(archive.read(name)):
                    findings.append(f"{archive_path.name}:{name}")
    if findings:
        raise RuntimeError(f"privacy-pattern findings: {findings}")
    root = Path(tempfile.mkdtemp(prefix="cscwd-final-verify-", dir=ROOT / "tmp"))
    for path in (
        DEST / "source_and_analysis.zip",
        DEST / "manuscript_source.zip",
        raw_archive,
    ):
        unpack(path, root)
    environment = {
        **os.environ,
        "PYTHONPATH": str(root / "src") + os.pathsep + str(root),
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    commands: list[dict[str, Any]] = []

    def run(command: list[str]) -> str:
        value = subprocess.run(
            command,
            cwd=root,
            env=environment,
            capture_output=True,
            text=True,
            timeout=240,
            check=False,
        )
        result = value.stdout + value.stderr
        commands.append(
            {"command": command, "exit_code": value.returncode, "output": result}
        )
        return result

    run(
        [
            sys.executable,
            "-c",
            "import pathlib,policytwin.coordination_validation as m; "
            "p=pathlib.Path(m.__file__).resolve(); "
            "assert p.is_relative_to(pathlib.Path.cwd()); "
            "print('EXTRACTED_IMPORT_OK')",
        ]
    )
    run([sys.executable, "-m", "scripts.cscwd2027.freeze_study", "check"])
    run([sys.executable, "-m", "scripts.cscwd2027_revision2.freeze", "check"])
    run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "-p",
            "no:cacheprovider",
            "tests/cscwd2027",
            "tests/cscwd2027_revision2",
            "tests/cscwd2027_final_review",
        ]
    )
    stats = root / "results/cscwd2027/final_review/derived/activity_summary.json"
    before = canonical_hash(stats)
    run([sys.executable, "-m", "scripts.cscwd2027_final_review.activity"])
    identical_stats = before == canonical_hash(stats)
    identical_text = {}
    pages = {}
    for stem in ("main", "technical_supplement"):
        run(
            [
                "/Library/TeX/texbin/latexmk",
                "-cd",
                "-pdf",
                "-interaction=nonstopmode",
                "-halt-on-error",
                f"paper/cscwd2027_final_review/{stem}.tex",
            ]
        )
        rebuilt = root / "paper/cscwd2027_final_review" / f"{stem}.pdf"
        text = subprocess.check_output(
            ["pdftotext", "-layout", str(rebuilt), "-"], text=True
        )
        delivered_text = subprocess.check_output(
            ["pdftotext", "-layout", str(DEST / f"{stem}.pdf"), "-"], text=True
        )
        identical_text[stem] = text == delivered_text
        info = subprocess.check_output(["pdfinfo", str(rebuilt)], text=True)
        match = re.search(r"^Pages:\s+(\d+)$", info, re.M)
        pages[stem] = int(match.group(1)) if match else None
    passed = (
        all(c["exit_code"] == 0 for c in commands)
        and identical_stats
        and all(identical_text.values())
        and all(n == 6 for n in pages.values())
    )
    result = {
        "pass": passed,
        "package": str(DEST.relative_to(ROOT)),
        "checked_sha256": checksums,
        "privacy_pattern_findings": findings,
        "privacy_scope": (
            "top-level text members; PDF checks separate; prior archive unchanged"
        ),
        "extraction": str(root.relative_to(ROOT)),
        "commands": commands,
        "descriptive_statistics_identical": identical_stats,
        "rebuilt_pdf_text_identical": identical_text,
        "rebuilt_pages": pages,
        "new_simulation_grid_run": False,
        "independent_implementation": False,
        "cross_platform": False,
    }
    destination = ROOT / "reports/cscwd2027/final_review/package_validation.json"
    destination.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                k: result[k]
                for k in ("pass", "rebuilt_pages", "descriptive_statistics_identical")
            }
        )
    )
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
