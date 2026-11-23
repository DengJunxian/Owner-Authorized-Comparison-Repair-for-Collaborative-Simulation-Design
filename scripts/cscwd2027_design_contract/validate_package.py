"""Recompute statistics/tests/PDF text in a fresh same-host scoped extraction."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Any

from scripts.cscwd2027_design_contract.package import DEST, checksums
from scripts.cscwd2027_design_contract.protection import ROOT, digest
from scripts.cscwd2027_final_review.package import ordinary
from scripts.cscwd2027_final_review.validate_package import canonical_hash, unpack


def main() -> None:
    """Check closed new-study inputs without rerunning the formal simulation grid."""
    for line in (DEST / "SHA256SUMS.txt").read_text().splitlines():
        sha, name = line.split("  ", 1)
        if Path(name).name != name or digest(DEST / name) != sha:
            raise RuntimeError("payload checksum changed")
    sensitive = re.compile(
        rb"/Users/[A-Za-z0-9_.-]+/|\bsk-[A-Za-z0-9_-]{24,}"
        rb"|-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"
    )
    privacy = []
    for path in DEST.glob("*.zip"):
        with zipfile.ZipFile(path) as z:
            for name in z.namelist():
                ordinary(name)
                if Path(name).suffix in {
                    ".py",
                    ".json",
                    ".tex",
                    ".bib",
                    ".md",
                    ".xml",
                } and sensitive.search(z.read(name)):
                    privacy.append(f"{path.name}:{name}")
    if privacy:
        raise RuntimeError(f"privacy-pattern findings: {privacy}")
    root = Path(tempfile.mkdtemp(prefix="cscwd-design-verify-", dir=ROOT / "tmp"))
    for name in (
        "source_and_analysis.zip",
        "manuscript_source.zip",
        "design_contract_complete_results.zip",
    ):
        unpack(DEST / name, root)
    env = {
        **os.environ,
        "PYTHONPATH": str(root / "src") + os.pathsep + str(root),
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    commands: list[dict[str, Any]] = []

    def run(command: list[str]) -> None:
        result = subprocess.run(
            command,
            cwd=root,
            env=env,
            capture_output=True,
            text=True,
            timeout=240,
            check=False,
        )
        safe_command = [c.replace(str(ROOT), "<local-repository>") for c in command]
        output = (result.stdout + result.stderr).replace(
            str(ROOT), "<local-repository>"
        )
        commands.append(
            {"command": safe_command, "exit_code": result.returncode, "output": output}
        )
        if result.returncode:
            raise RuntimeError(output)

    run(
        [
            sys.executable,
            "-c",
            "import pathlib,policytwin.design_contract as m; "
            "p=pathlib.Path(m.__file__).resolve(); "
            "assert p.is_relative_to(pathlib.Path.cwd()); "
            "print('EXTRACTED_IMPORT_OK')",
        ]
    )
    run([sys.executable, "-m", "scripts.cscwd2027_design_contract.freeze", "check"])
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
            "tests/cscwd2027_design_contract",
        ]
    )
    summary = root / "results/cscwd2027/design_contract_v1/derived/summary.json"
    before = canonical_hash(summary)
    generated = root / "paper/cscwd2027_design_contract/generated"
    tex_before = {p.name: digest(p) for p in generated.glob("*.tex")}
    run([sys.executable, "-m", "scripts.cscwd2027_design_contract.analyze"])
    stats_equal = before == canonical_hash(summary)
    run([sys.executable, "-m", "scripts.cscwd2027_design_contract.present"])
    run(
        [
            sys.executable,
            "-m",
            "scripts.cscwd2027_design_contract.deliver",
            "--scoped-extraction",
        ]
    )
    run([sys.executable, "-m", "scripts.cscwd2027_design_contract.raw_recheck"])
    tex_equal = all(digest(generated / name) == sha for name, sha in tex_before.items())
    pdf_text_equal = {}
    pages = {}
    for stem in ("main", "technical_supplement"):
        run(
            [
                "/Library/TeX/texbin/latexmk",
                "-silent",
                "-cd",
                "-pdf",
                "-interaction=nonstopmode",
                "-halt-on-error",
                f"paper/cscwd2027_design_contract/{stem}.tex",
            ]
        )
        pdf = root / "paper/cscwd2027_design_contract" / f"{stem}.pdf"
        actual = subprocess.check_output(
            ["pdftotext", "-layout", str(pdf), "-"], text=True
        )
        reference = subprocess.check_output(
            ["pdftotext", "-layout", str(DEST / f"{stem}.pdf"), "-"], text=True
        )
        pdf_text_equal[stem] = actual == reference
        info = subprocess.check_output(["pdfinfo", str(pdf)], text=True)
        pages[stem] = int(re.search(r"Pages:\s+(\d+)", info).group(1))  # type: ignore[union-attr]
    passed = (
        stats_equal
        and tex_equal
        and all(pdf_text_equal.values())
        and pages["main"] <= 6
    )
    value = {
        "pass": passed,
        "privacy_pattern_findings": privacy,
        "scope": "new study closed extraction on the same host, "
        "not independent or cross-platform reproduction",
        "extraction": str(root.relative_to(ROOT)),
        "statistics_identical": stats_equal,
        "generated_tex_identical": tex_equal,
        "rebuilt_pdf_text_identical": pdf_text_equal,
        "pages": pages,
        "commands": commands,
        "formal_grid_rerun": False,
        "local_original_preservation_manifests_checked_in_working_tree": True,
    }
    (DEST / "PACKAGE_VALIDATION.json").write_text(json.dumps(value, indent=2) + "\n")
    manifest = json.loads((DEST / "MANIFEST.json").read_text())
    manifest["package_validation_pass"] = passed
    manifest["payloads"]["PACKAGE_VALIDATION.json"] = digest(
        DEST / "PACKAGE_VALIDATION.json"
    )
    (DEST / "MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
    checksums()
    print(
        json.dumps(
            {
                k: value[k]
                for k in (
                    "pass",
                    "statistics_identical",
                    "generated_tex_identical",
                    "rebuilt_pdf_text_identical",
                    "pages",
                )
            }
        )
    )
    if not passed:
        raise RuntimeError("package rebuild mismatch")


if __name__ == "__main__":
    main()
