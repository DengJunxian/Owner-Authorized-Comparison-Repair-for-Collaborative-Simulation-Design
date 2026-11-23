"""Fail closed when registered paper metadata differs from its frozen lock."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import unicodedata
from pathlib import Path
from typing import NoReturn


class RegisteredMetadataError(RuntimeError):
    """Raised when frozen submission metadata or manuscript layout differs."""


def _fail(message: str) -> NoReturn:
    raise RegisteredMetadataError(message)


def _strip_comments(source: str) -> str:
    lines: list[str] = []
    for line in source.splitlines(keepends=True):
        for index, character in enumerate(line):
            if character != "%":
                continue
            backslashes = 0
            cursor = index - 1
            while cursor >= 0 and line[cursor] == "\\":
                backslashes += 1
                cursor -= 1
            if backslashes % 2 == 0:
                line = line[:index] + ("\n" if line.endswith("\n") else "")
                break
        lines.append(line)
    return "".join(lines)


def normalize_text(value: str) -> str:
    """Return the versioned semantic normalization used by the lock."""
    value = value.replace(r"\%", "%").replace(r"\&", "&")
    value = value.replace("’", "'").replace("‘", "'")
    value = unicodedata.normalize("NFKC", value)
    return " ".join(value.split())


def _extract(source: str, pattern: str, field: str) -> str:
    match = re.search(
        pattern,
        _strip_comments(source),
        flags=re.DOTALL | re.MULTILINE,
    )
    if match is None:
        _fail(f"could not extract {field}")
    return normalize_text(match.group("value"))


def extract_registered_metadata(path: Path) -> dict[str, str]:
    source = path.read_text(encoding="utf-8")
    uncommented = _strip_comments(source)
    if re.search(r"(?:Primary|Secondary) Content Area", uncommented):
        _fail(
            "content areas must remain in submission metadata, not rendered paper text"
        )
    return {
        "title": _extract(source, r"\\title\s*\{(?P<value>[^{}]*)\}", "title"),
        "abstract": _extract(
            source,
            r"\\begin\{abstract\}(?P<value>.*?)\\end\{abstract\}",
            "abstract",
        ),
        "paper_id": _extract(
            source,
            r"\\(?:new|provide)command\s*\{\\aaaiPaperID\}"
            r"\s*\{(?P<value>[^{}]*)\}",
            "paper ID",
        ),
    }


def extract_submission_content_areas(path: Path) -> dict[str, str]:
    """Read the OpenReview-only areas without placing them in rendered LaTeX."""
    source = path.read_text(encoding="utf-8")
    primary = _extract(
        source,
        r"^- Primary:\s*`(?P<value>[^`\n]+)`\s*$",
        "primary content area",
    )
    secondary_matches = re.findall(
        r"^- Secondary:\s*`(?P<value>[^`\n]+)`\s*$",
        source,
        flags=re.MULTILINE,
    )
    if not secondary_matches:
        _fail("could not extract secondary content areas")
    return {
        "primary": primary,
        "secondary": normalize_text("; ".join(secondary_matches)),
    }


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def check_registered_metadata(
    paper_path: Path,
    lock_path: Path,
    submission_fields_path: Path | None = None,
) -> None:
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if lock.get("schema_version") != "policytwin.registered-metadata-lock.v2":
        _fail("unsupported registered metadata lock schema")
    observed = extract_registered_metadata(paper_path)
    for field in ("title", "abstract"):
        expected = normalize_text(lock[field]["text"])
        expected_hash = lock[field]["normalized_sha256"]
        if _sha256(expected) != expected_hash:
            _fail(f"lock file {field} text does not match its own SHA-256")
        if observed[field] != expected or _sha256(observed[field]) != expected_hash:
            _fail(
                f"registered {field} differs: expected SHA-256 {expected_hash}, "
                f"observed {_sha256(observed[field])}"
            )
    expected_paper_id = normalize_text(lock["paper_id"])
    if observed["paper_id"] != expected_paper_id:
        _fail(
            f"paper ID differs: expected {expected_paper_id}, "
            f"observed {observed['paper_id']}"
        )
    fields_path = (
        submission_fields_path or lock_path.parent / "openreview_submission_fields.md"
    )
    observed_areas = extract_submission_content_areas(fields_path)
    for field in ("primary", "secondary"):
        expected = normalize_text(lock["content_areas"][field])
        if observed_areas[field] != expected:
            _fail(f"registered {field} content area differs")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("paper", type=Path)
    parser.add_argument(
        "--lock",
        type=Path,
        default=Path("paper/registered_metadata_lock.json"),
    )
    parser.add_argument(
        "--submission-fields",
        type=Path,
        default=Path("paper/openreview_submission_fields.md"),
    )
    arguments = parser.parse_args(argv)
    try:
        check_registered_metadata(
            arguments.paper,
            arguments.lock,
            arguments.submission_fields,
        )
    except (
        OSError,
        KeyError,
        TypeError,
        json.JSONDecodeError,
        RegisteredMetadataError,
    ) as error:
        print(f"REGISTERED_METADATA_LOCK_FAILED: {error}", file=sys.stderr)
        return 1
    print(f"REGISTERED_METADATA_LOCK_OK: {arguments.paper}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
