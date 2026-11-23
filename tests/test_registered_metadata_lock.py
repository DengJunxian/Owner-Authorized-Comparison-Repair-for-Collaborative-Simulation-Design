from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.check_registered_metadata import (
    RegisteredMetadataError,
    check_registered_metadata,
)

ROOT = Path(__file__).resolve().parents[1]
LOCK = ROOT / "paper" / "registered_metadata_lock.json"
SUBMISSION_FIELDS = ROOT / "paper" / "openreview_submission_fields.md"


def test_revised_paper_matches_registered_metadata() -> None:
    check_registered_metadata(ROOT / "paper" / "main_revised.tex", LOCK)


def test_title_change_fails_closed(tmp_path: Path) -> None:
    source = (ROOT / "paper" / "main_revised.tex").read_text(encoding="utf-8")
    changed = source.replace("From Event Keys", "From Random Seeds", 1)
    paper = tmp_path / "changed.tex"
    paper.write_text(changed, encoding="utf-8")
    with pytest.raises(RegisteredMetadataError, match="registered title differs"):
        check_registered_metadata(paper, LOCK)


def test_paper_id_change_fails_closed(tmp_path: Path) -> None:
    source = (ROOT / "paper" / "main_revised.tex").read_text(encoding="utf-8")
    changed = source.replace(
        r"\providecommand{\aaaiPaperID}{29236}",
        r"\providecommand{\aaaiPaperID}{99999}",
        1,
    )
    paper = tmp_path / "changed-id.tex"
    paper.write_text(changed, encoding="utf-8")
    with pytest.raises(RegisteredMetadataError, match="paper ID differs"):
        check_registered_metadata(paper, LOCK)


def test_content_area_stays_in_submission_metadata_only(tmp_path: Path) -> None:
    fields = SUBMISSION_FIELDS.read_text(encoding="utf-8")
    changed = fields.replace("Agent-Based Simulation", "Agent Learning", 1)
    changed_fields = tmp_path / "changed-fields.md"
    changed_fields.write_text(changed, encoding="utf-8")
    with pytest.raises(RegisteredMetadataError, match="primary content area differs"):
        check_registered_metadata(
            ROOT / "paper" / "main_revised.tex",
            LOCK,
            changed_fields,
        )


def test_rendered_content_area_fails_closed(tmp_path: Path) -> None:
    source = (ROOT / "paper" / "main_revised.tex").read_text(encoding="utf-8")
    changed = source.replace(
        r"\maketitle",
        "\\maketitle\nPrimary Content Area: MAS: Agent-Based Simulation",
        1,
    )
    assert changed != source
    paper = tmp_path / "rendered-area.tex"
    paper.write_text(changed, encoding="utf-8")
    with pytest.raises(RegisteredMetadataError, match="submission metadata"):
        check_registered_metadata(paper, LOCK)


def test_lock_hash_is_self_authenticating(tmp_path: Path) -> None:
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    lock["abstract"]["text"] += " Changed."
    changed_lock = tmp_path / "changed-lock.json"
    changed_lock.write_text(json.dumps(lock), encoding="utf-8")
    with pytest.raises(RegisteredMetadataError, match="does not match its own SHA-256"):
        check_registered_metadata(ROOT / "paper" / "main_revised.tex", changed_lock)
