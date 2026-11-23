"""Same-host clean-extraction verification, not independent reproduction."""

from __future__ import annotations

import argparse
import gzip
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any

from policytwin.coordination.schema import commitment
from scripts.cscwd2027_revision2.freeze import ROOT, digest


def scan(name: str, data: bytes) -> list[dict[str, str]]:
    """Recursively inspect text receipts without disclosing any matched value."""
    if name.endswith(".zip"):
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            return [
                finding
                for p in archive.infolist()
                for finding in scan(name + "!" + p.filename, archive.read(p))
            ]
    if name.endswith(".gz"):
        data = gzip.decompress(data)
    if name.endswith(".pdf"):
        return []
    patterns = {
        "personal_path": rb"/Users/[A-Za-z0-9_.-]+/",
        "private_key": rb"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----",
        "credential_shape": (
            rb"\b(?:sk-[A-Za-z0-9_-]{24,}|ghp_[A-Za-z0-9]{24,}|AKIA[0-9A-Z]{16})\b"
        ),
    }
    return [
        {"member": name, "kind": k}
        for k, pattern in patterns.items()
        if re.search(pattern, data)
    ]


def unpack(path: Path, target: Path) -> None:
    """Reject absolute, escaping, duplicate, symlink and overwriting members."""
    with zipfile.ZipFile(path) as archive:
        seen = set()
        for member in archive.infolist():
            pure = PurePosixPath(member.filename)
            if (
                pure.is_absolute()
                or ".." in pure.parts
                or member.filename in seen
                or (target / member.filename).exists()
            ):
                raise ValueError("unsafe or conflicting archive member")
            if member.external_attr >> 16 & 0o170000 == 0o120000:
                raise ValueError("archive symlink")
            seen.add(member.filename)
        archive.extractall(target)


def main() -> None:
    """Recompute the frozen counts and compile the exact exported sources."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="review_20261025_revision2_v1")
    args = parser.parse_args()
    if not re.fullmatch(r"review_[A-Za-z0-9_-]+", args.name):
        raise ValueError("invalid package directory")
    package = ROOT / "release/cscwd2027" / args.name
    checked = {}
    for line in (package / "SHA256SUMS.txt").read_text().splitlines():
        sha, name = line.split("  ", 1)
        if PurePosixPath(name).name != name or digest(package / name) != sha:
            raise ValueError("package checksum mismatch")
        checked[name] = sha
    findings = [
        finding
        for path in package.iterdir()
        if path.is_file()
        for finding in scan(path.name, path.read_bytes())
    ]
    if findings:
        raise RuntimeError(json.dumps({"privacy_pattern_findings": findings}))
    target = Path(tempfile.mkdtemp(prefix="cscwd-r2-verify-", dir=ROOT / "tmp"))
    for name in (
        "source_and_protocol.zip",
        "original_v1_complete_results.zip",
        "revision2_complete_results.zip",
        "manuscript_source.zip",
    ):
        unpack(package / name, target)
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join((str(target / "src"), str(target)))
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    summary_path = target / "results/cscwd2027/revision2/derived/summary.json"
    before = commitment(json.loads(summary_path.read_text()))
    text_before = {
        stem: subprocess.check_output(
            [
                "pdftotext",
                "-layout",
                str(target / f"paper/cscwd2027_revision2/{stem}.pdf"),
                "-",
            ]
        )
        for stem in ("main", "technical_supplement")
    }
    latexmk = shutil.which("latexmk")
    if not latexmk:
        raise RuntimeError("latexmk unavailable")
    commands = [
        [
            sys.executable,
            "-c",
            "import pathlib,policytwin.coordination_validation as m; "
            "assert pathlib.Path(m.__file__).resolve().is_relative_to("
            "pathlib.Path.cwd()); print('EXTRACTED_IMPORT_OK')",
        ],
        [sys.executable, "-m", "scripts.cscwd2027.freeze_study", "check"],
        [sys.executable, "-m", "scripts.cscwd2027_revision2.freeze", "check"],
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "-p",
            "no:cacheprovider",
            "tests/cscwd2027",
            "tests/cscwd2027_revision2",
        ],
        [sys.executable, "-m", "scripts.cscwd2027_revision2.analyze"],
        [sys.executable, "-m", "scripts.cscwd2027_revision2.present"],
        *[
            [
                latexmk,
                "-cd",
                "-pdf",
                "-interaction=nonstopmode",
                "-halt-on-error",
                f"paper/cscwd2027_revision2/{stem}.tex",
            ]
            for stem in ("main", "technical_supplement")
        ],
    ]
    receipts = []
    for command in commands:
        result = subprocess.run(
            command,
            cwd=target,
            env=environment,
            capture_output=True,
            text=True,
            timeout=1800,
        )
        label = "latexmk" if command[0] == latexmk else "python"
        receipts.append(
            {
                "command": [label, *command[1:]],
                "exit_code": result.returncode,
                "output": (result.stdout + result.stderr).replace(
                    str(target), "<extraction>"
                ),
            }
        )
        print(
            json.dumps(
                {"command": receipts[-1]["command"], "exit_code": result.returncode}
            ),
            flush=True,
        )
        if result.returncode:
            break
    after = commitment(json.loads(summary_path.read_text()))
    same_text = {
        stem: value
        == subprocess.check_output(
            [
                "pdftotext",
                "-layout",
                str(target / f"paper/cscwd2027_revision2/{stem}.pdf"),
                "-",
            ]
        )
        for stem, value in text_before.items()
    }
    passed = (
        all(r["exit_code"] == 0 for r in receipts)
        and len(receipts) == len(commands)
        and before == after
        and all(same_text.values())
    )
    report: dict[str, Any] = {
        "package": str(package.relative_to(ROOT)),
        "checked_sha256": checked,
        "privacy_pattern_findings": findings,
        "privacy_scope": (
            "recursive path/key/credential scan; PDFs have separate checks; "
            "not a proof against reidentification"
        ),
        "extraction_relative": str(target.relative_to(ROOT)),
        "commands": receipts,
        "summary_canonical_sha256_before": before,
        "summary_canonical_sha256_after": after,
        "summary_semantically_identical": before == after,
        "rebuilt_pdf_text_identical": same_text,
        "pass": passed,
        "independent_implementation": False,
        "cross_platform": False,
        "new_formal_grid_run": False,
    }
    destination = ROOT / "reports/cscwd2027/revision2/package_validation.json"
    destination.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"pass": passed, "report": str(destination.relative_to(ROOT))}))
    if not passed:
        raise RuntimeError("package validation failed; preserve receipt")


if __name__ == "__main__":
    main()
