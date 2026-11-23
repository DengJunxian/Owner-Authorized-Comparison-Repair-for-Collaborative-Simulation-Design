"""Check exported bytes and rerun local validation from a clean extraction."""

from __future__ import annotations

import argparse
import gzip
import hashlib
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

ROOT = Path(__file__).resolve().parents[2]
PATTERNS = {
    "personal_path": re.compile(rb"/Users/[A-Za-z0-9_.-]+/"),
    "private_key": re.compile(rb"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    "credential_shape": re.compile(
        rb"\b(?:sk-[A-Za-z0-9_-]{24,}|ghp_[A-Za-z0-9]{24,}|AKIA[0-9A-Z]{16})\b"
    ),
}


def digest(path: Path) -> str:
    """Hash exact package bytes."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def inspect_bytes(name: str, content: bytes) -> list[dict[str, str]]:
    """Inspect nested archives and compressed receipts without printing secrets."""
    if name.endswith(".zip"):
        with zipfile.ZipFile(io.BytesIO(content)) as archive:
            return [
                finding
                for member in archive.infolist()
                for finding in inspect_bytes(
                    f"{name}!{member.filename}", archive.read(member)
                )
            ]
    if name.endswith(".gz"):
        content = gzip.decompress(content)
    if name.endswith(".pdf"):
        # PDF content and metadata have a separate rendered/text/font audit.
        return []
    return [
        {"member": name, "kind": kind}
        for kind, pattern in PATTERNS.items()
        if pattern.search(content)
    ]


def extract(archive_path: Path, target: Path) -> None:
    """Extract only safe, ordinary, relative members into a new working directory."""
    with zipfile.ZipFile(archive_path) as archive:
        names = set()
        for member in archive.infolist():
            path = PurePosixPath(member.filename)
            if path.is_absolute() or ".." in path.parts or member.filename in names:
                raise ValueError("unsafe or duplicate archive path")
            if member.external_attr >> 16 & 0o170000 == 0o120000:
                raise ValueError("symlink in archive")
            names.add(member.filename)
            if (target / member.filename).exists():
                raise ValueError("archives would overwrite an extracted file")
        archive.extractall(target)


def main() -> None:
    """Validate a local review snapshot; do not launch a new scientific grid."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", required=True)
    parser.add_argument("--hashes-only", action="store_true")
    parser.add_argument("--plot-python")
    args = parser.parse_args()
    if not re.fullmatch(r"review_[A-Za-z0-9_-]+", args.directory):
        raise ValueError("expected a plain review_* directory name")
    package = ROOT / "release/cscwd2027" / args.directory
    manifest = json.loads((package / "MANIFEST.json").read_text())
    checked = {}
    for line in (package / "SHA256SUMS.txt").read_text().splitlines():
        sha, name = line.split("  ", 1)
        if PurePosixPath(name).name != name or digest(package / name) != sha:
            raise ValueError("package SHA-256 mismatch")
        checked[name] = sha
    for name, record in manifest["files"].items():
        if checked[name] != record["sha256"]:
            raise ValueError("manifest and checksum list disagree")
        if (package / name).stat().st_size != record["bytes"]:
            raise ValueError("package byte count mismatch")
    findings = []
    for path in sorted(package.iterdir()):
        if path.is_file():
            findings.extend(inspect_bytes(path.name, path.read_bytes()))
    if findings:
        raise ValueError(json.dumps({"sensitive_pattern_findings": findings}))
    report: dict[str, Any] = {
        "package": str(package.relative_to(ROOT)),
        "checked_sha256": checked,
        "sensitive_pattern_findings": findings,
        "privacy_scope": (
            "Personal-path/private-key/credential patterns, recursively including "
            "zip and gzip; PDFs use separate page/text/metadata audit. "
            "Not a proof against every form of reidentification."
        ),
        "cross_platform_test": False,
        "independent_implementation": False,
        "new_formal_study_run": False,
        "commands": [],
    }
    if not args.hashes_only:
        target = Path(tempfile.mkdtemp(prefix="cscwd-review-", dir=ROOT / "tmp"))
        for name in (
            "cscwd_source_and_protocol.zip",
            "cscwd_complete_results.zip",
            "cscwd_manuscript_source.zip",
        ):
            extract(package / name, target)
        report["extraction_relative"] = str(target.relative_to(ROOT))
        environment = dict(os.environ)
        environment["PYTHONPATH"] = os.pathsep.join((str(target / "src"), str(target)))
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        environment["MPLCONFIGDIR"] = str(target / "tmp/matplotlib")
        report["summary_sha256_before"] = digest(
            target / "results/cscwd2027/derived_v1/summary.json"
        )
        commands = [
            [
                sys.executable,
                "-c",
                "import pathlib,policytwin.coordination as m; "
                "assert pathlib.Path(m.__file__).resolve().is_relative_to("
                "pathlib.Path.cwd()); "
                "print('EXTRACTED_IMPORT_OK')",
            ],
            [sys.executable, "-m", "scripts.cscwd2027.freeze_study", "check"],
            [
                sys.executable,
                "-m",
                "pytest",
                "-q",
                "-p",
                "no:cacheprovider",
                "tests/cscwd2027",
                "--junitxml=reports/cscwd2027/extracted_tests.xml",
            ],
            [sys.executable, "-m", "scripts.cscwd2027.verify_artifacts"],
            [sys.executable, "-m", "scripts.cscwd2027.derive_results"],
            [sys.executable, "-m", "scripts.cscwd2027.audit_tcb"],
            [sys.executable, "-m", "scripts.cscwd2027.prepare_manuscript"],
        ]
        latexmk = shutil.which("latexmk")
        before_text = {}
        if args.plot_python:
            if not Path(args.plot_python).is_file() or not latexmk:
                raise ValueError("render validation needs plotting Python and latexmk")
            for stem in ("main", "technical_supplement"):
                before_text[stem] = subprocess.check_output(
                    [
                        "pdftotext",
                        "-layout",
                        str(target / f"paper/cscwd2027/{stem}.pdf"),
                        "-",
                    ]
                )
            commands.extend(
                [args.plot_python, "-m", f"scripts.cscwd2027.{module}"]
                for module in ("build_figures", "refine_figure_style")
            )
            commands.extend(
                [
                    latexmk,
                    "-cd",
                    "-pdf",
                    "-interaction=nonstopmode",
                    "-halt-on-error",
                    f"paper/cscwd2027/{stem}.tex",
                ]
                for stem in ("main", "technical_supplement")
            )
            commands.append(
                [sys.executable, "-m", "scripts.cscwd2027.audit_manuscript"]
            )
        for command in commands:
            completed = subprocess.run(
                command,
                cwd=target,
                env=environment,
                capture_output=True,
                text=True,
                timeout=1800,
            )
            output = (completed.stdout + completed.stderr).replace(
                str(target), "<extraction>"
            )
            executable_label = (
                "latexmk"
                if command[0] == latexmk
                else "plot-python"
                if command[0] == args.plot_python
                else "python"
            )
            record = {
                "command": [executable_label, *command[1:]],
                "exit_code": completed.returncode,
                "output": output,
            }
            report["commands"].append(record)
            print(
                json.dumps(
                    {"command": record["command"], "exit_code": completed.returncode}
                ),
                flush=True,
            )
            if completed.returncode:
                report["pass"] = False
                break
        report["summary_sha256_after"] = digest(
            target / "results/cscwd2027/derived_v1/summary.json"
        )
        report["summary_identical"] = (
            report["summary_sha256_before"] == report["summary_sha256_after"]
        )
        report["pass"] = (
            all(r["exit_code"] == 0 for r in report["commands"])
            and report["summary_identical"]
        )
        if before_text:
            report["rebuilt_pdf_text_identical"] = {
                stem: value
                == subprocess.check_output(
                    [
                        "pdftotext",
                        "-layout",
                        str(target / f"paper/cscwd2027/{stem}.pdf"),
                        "-",
                    ]
                )
                for stem, value in before_text.items()
            }
            report["pass"] = report["pass"] and all(
                report["rebuilt_pdf_text_identical"].values()
            )
    else:
        report["pass"] = True
    destination = (
        ROOT / "reports/cscwd2027" / f"package_validation_{args.directory}.json"
    )
    destination.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {"pass": report["pass"], "report": str(destination.relative_to(ROOT))}
        )
    )
    if not report["pass"]:
        raise RuntimeError("package validation failed; inspect retained receipt")


if __name__ == "__main__":
    main()
