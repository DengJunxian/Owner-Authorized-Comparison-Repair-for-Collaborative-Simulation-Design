"""Fail-closed Git source-snapshot and release preflight.

The scientific artifacts in this project name a Git commit as part of their
identity.  This module verifies that such an identity actually exists and
builds a content manifest from committed Git blobs, never from mutable working
tree files.  It is a reproducibility and packaging control, not a license,
privacy, or scientific-validity authorization.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import shutil
import stat
import sys
import tarfile
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path, PurePosixPath
from typing import BinaryIO, NoReturn, Self, cast

from policytwin.git_boundary import (
    GitBoundaryError,
    GitCliIdentity,
    assert_trusted_repository_layout,
    discover_repository_root,
    require_same_git_cli_identity,
    run_git_command,
    verify_git_cli_identity,
)
from policytwin.provenance import (
    MetadataValidationError,
    canonical_json_bytes,
    read_json_bytes_object,
    sha256_bytes,
    sha256_file,
)

SOURCE_MANIFEST_SCHEMA_VERSION = "policytwin.source-snapshot.v1"
SOURCE_ARCHIVE_SCHEMA_VERSION = "policytwin.source-archive.v2"
_ARCHIVE_PREFIX = "policytwin-source"

_OBJECT_ID = re.compile(r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SAFE_MODES = frozenset({"100644", "100755"})
_SAFE_ROW_ID = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
_GENERATED_PAPER_SUFFIXES = frozenset(
    {".aux", ".bbl", ".blg", ".fdb_latexmk", ".fls", ".log", ".out", ".pdf"}
)
_PRIVATE_FILE_SUFFIXES = frozenset({".key", ".p12", ".pem", ".pfx"})
_CACHE_PARTS = frozenset(
    {
        ".hypothesis",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".venv",
        "__pycache__",
    }
)
_CREDENTIAL_PATTERNS: tuple[tuple[str, re.Pattern[bytes]], ...] = (
    (
        "private_key",
        re.compile(rb"-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----"),
    ),
    ("aws_access_key", re.compile(rb"AKIA[0-9A-Z]{16}")),
    ("github_token", re.compile(rb"gh[pousr]_[A-Za-z0-9_]{20,}")),
    ("openai_key", re.compile(rb"sk-[A-Za-z0-9_-]{20,}")),
)

# This is a semantic sentinel for the two outcome-summary exposures already
# disclosed to the design-influencing team.  Updating or appending the ledger
# must be an explicit reviewed source change; re-sealing a rewritten history is
# not sufficient to pass release preflight.
_SEC_OUTCOME_KNOWLEDGE_SENTINEL_SEMANTIC_SHA256 = (
    "f32d3add9099b79f337299859d41137f7d7c461a0a4e009e2613aac3e6b2e114"
)


class PreflightRuleCode(StrEnum):
    """Stable machine-readable source-preflight outcomes."""

    NOT_GIT_REPOSITORY = "NOT_GIT_REPOSITORY"
    REPOSITORY_ROOT_MISMATCH = "REPOSITORY_ROOT_MISMATCH"
    GIT_TRUST_BOUNDARY_FAILED = "GIT_TRUST_BOUNDARY_FAILED"
    GIT_COMMAND_FAILED = "GIT_COMMAND_FAILED"
    MISSING_HEAD = "MISSING_HEAD"
    TRACKED_CHANGES = "TRACKED_CHANGES"
    UNTRACKED_FILES = "UNTRACKED_FILES"
    REQUIRED_FILE_MISSING = "REQUIRED_FILE_MISSING"
    UNSAFE_TREE_ENTRY = "UNSAFE_TREE_ENTRY"
    FORBIDDEN_PATH = "FORBIDDEN_PATH"
    FILE_TOO_LARGE = "FILE_TOO_LARGE"
    CREDENTIAL_PATTERN = "CREDENTIAL_PATTERN"
    OUTPUT_PATH_INVALID = "OUTPUT_PATH_INVALID"
    OUTPUT_EXISTS = "OUTPUT_EXISTS"
    PARTIAL_OUTPUT_CLEANUP_FAILED = "PARTIAL_OUTPUT_CLEANUP_FAILED"
    MANIFEST_INVALID = "MANIFEST_INVALID"
    MANIFEST_REFERENCE_MISSING = "MANIFEST_REFERENCE_MISSING"
    MANIFEST_REFERENCE_MISMATCH = "MANIFEST_REFERENCE_MISMATCH"
    MANIFEST_UNDECLARED_ARTIFACT = "MANIFEST_UNDECLARED_ARTIFACT"


class SourcePreflightError(ValueError):
    """Raised when an export request cannot safely be completed."""

    def __init__(self, message: str, rule_code: PreflightRuleCode) -> None:
        super().__init__(message)
        self.rule_code = rule_code


def _raise(rule_code: PreflightRuleCode, message: str) -> NoReturn:
    raise SourcePreflightError(message, rule_code)


def _safe_relative_path(value: str) -> bool:
    if not value or "\\" in value or any(ord(character) < 32 for character in value):
        return False
    path = PurePosixPath(value)
    return (
        not path.is_absolute()
        and path.as_posix() == value
        and all(part not in {"", ".", "..", ".git"} for part in path.parts)
    )


def _strict_fields(
    value: Mapping[str, object], expected: frozenset[str], context: str
) -> None:
    actual = frozenset(value)
    if actual != expected:
        _raise(
            PreflightRuleCode.MANIFEST_INVALID,
            f"{context} fields differ from the closed schema",
        )


def _mapping(value: object, context: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        _raise(PreflightRuleCode.MANIFEST_INVALID, f"{context} must be an object")
    return cast(Mapping[str, object], value)


def _integer(value: object, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        _raise(PreflightRuleCode.MANIFEST_INVALID, f"{context} must be an integer")
    return value


def _text(value: object, context: str) -> str:
    if not isinstance(value, str) or not value:
        _raise(
            PreflightRuleCode.MANIFEST_INVALID,
            f"{context} must be a non-empty string",
        )
    return value


@dataclass(frozen=True, slots=True)
class SourcePreflightConfig:
    """Immutable local policy for one committed source-tree review."""

    max_file_size_bytes: int = 5 * 1024 * 1024
    required_paths: tuple[str, ...] = (
        ".github/workflows/clean-reproduction.yml",
        ".github/workflows/ci.yml",
        ".gitignore",
        ".python-version",
        "AGENTS.md",
        "README.md",
        "docs/literature_search_manifest.json",
        "docs/literature_search_manifest.sha256",
        # These two files preserve the self-reported hindsight ledger in every
        # source snapshot.  The companion checksum is not an external authority;
        # real-study release still requires the protocol's independent anchor.
        "configs/policies/sec_tick_size_pilot_outcome_knowledge.json",
        "configs/policies/sec_tick_size_pilot_outcome_knowledge.sha256",
        "paper/artifact_manifest.json",
        "paper/artifact_manifest.sha256",
        "pyproject.toml",
        "uv.lock",
    )
    closed_artifact_manifests: tuple[str, ...] = (
        "experiments/smoke_matrix/run-manifest.json",
    )

    def __post_init__(self) -> None:
        if (
            isinstance(self.max_file_size_bytes, bool)
            or not isinstance(self.max_file_size_bytes, int)
            or self.max_file_size_bytes <= 0
        ):
            _raise(
                PreflightRuleCode.FILE_TOO_LARGE,
                "max_file_size_bytes must be a positive integer",
            )
        if (
            not isinstance(self.required_paths, tuple)
            or not self.required_paths
            or any(not _safe_relative_path(path) for path in self.required_paths)
            or len(set(self.required_paths)) != len(self.required_paths)
        ):
            _raise(
                PreflightRuleCode.REQUIRED_FILE_MISSING,
                "required_paths must be a unique tuple of safe relative paths",
            )
        if (
            not isinstance(self.closed_artifact_manifests, tuple)
            or any(
                not _safe_relative_path(path) for path in self.closed_artifact_manifests
            )
            or len(set(self.closed_artifact_manifests))
            != len(self.closed_artifact_manifests)
        ):
            _raise(
                PreflightRuleCode.MANIFEST_INVALID,
                "closed_artifact_manifests must contain unique safe paths",
            )


@dataclass(frozen=True, slots=True)
class PreflightBlocker:
    """One exact reason that a source snapshot cannot be released."""

    rule_code: PreflightRuleCode
    message: str
    path: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.rule_code, PreflightRuleCode):
            _raise(PreflightRuleCode.GIT_COMMAND_FAILED, "invalid preflight rule code")
        if not isinstance(self.message, str) or not self.message:
            _raise(PreflightRuleCode.GIT_COMMAND_FAILED, "empty preflight message")
        if self.path is not None and not _safe_relative_path(self.path):
            _raise(PreflightRuleCode.UNSAFE_TREE_ENTRY, "unsafe blocker path")

    def to_mapping(self) -> dict[str, object]:
        return {
            "message": self.message,
            "path": self.path,
            "rule_code": self.rule_code.value,
        }


@dataclass(frozen=True, slots=True)
class SourceFileDigest:
    """Portable identity of one regular file stored in the commit."""

    path: str
    mode: str
    size_bytes: int
    sha256: str

    def __post_init__(self) -> None:
        if not _safe_relative_path(self.path):
            _raise(PreflightRuleCode.UNSAFE_TREE_ENTRY, "unsafe source file path")
        if self.mode not in _SAFE_MODES:
            _raise(PreflightRuleCode.UNSAFE_TREE_ENTRY, "unsafe source file mode")
        if (
            isinstance(self.size_bytes, bool)
            or not isinstance(self.size_bytes, int)
            or self.size_bytes < 0
        ):
            _raise(PreflightRuleCode.GIT_COMMAND_FAILED, "invalid source file size")
        if not isinstance(self.sha256, str) or _SHA256.fullmatch(self.sha256) is None:
            _raise(PreflightRuleCode.GIT_COMMAND_FAILED, "invalid source file digest")

    def to_mapping(self) -> dict[str, object]:
        return {
            "mode": self.mode,
            "path": self.path,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> Self:
        """Strictly parse one committed regular-file identity."""
        _strict_fields(
            value,
            frozenset({"mode", "path", "sha256", "size_bytes"}),
            "source file digest",
        )
        return cls(
            path=_text(value["path"], "source file path"),
            mode=_text(value["mode"], "source file mode"),
            size_bytes=_integer(value["size_bytes"], "source file size"),
            sha256=_text(value["sha256"], "source file SHA-256"),
        )


@dataclass(frozen=True, slots=True)
class SourceSnapshotManifest:
    """Deterministic SHA-256 inventory derived entirely from committed blobs."""

    schema_version: str
    commit: str
    git_object_format: str
    files: tuple[SourceFileDigest, ...]
    file_count: int
    total_size_bytes: int
    tree_sha256: str

    def __post_init__(self) -> None:
        if self.schema_version != SOURCE_MANIFEST_SCHEMA_VERSION:
            _raise(PreflightRuleCode.GIT_COMMAND_FAILED, "invalid manifest schema")
        if (
            not isinstance(self.commit, str)
            or _OBJECT_ID.fullmatch(self.commit) is None
        ):
            _raise(PreflightRuleCode.GIT_COMMAND_FAILED, "invalid commit object ID")
        if self.git_object_format not in {"sha1", "sha256"}:
            _raise(PreflightRuleCode.GIT_COMMAND_FAILED, "unknown Git object format")
        if tuple(sorted(self.files, key=lambda item: item.path)) != self.files:
            _raise(
                PreflightRuleCode.GIT_COMMAND_FAILED, "manifest files are not sorted"
            )
        if len({entry.path for entry in self.files}) != len(self.files):
            _raise(PreflightRuleCode.GIT_COMMAND_FAILED, "duplicate manifest path")
        if self.file_count != len(self.files):
            _raise(PreflightRuleCode.GIT_COMMAND_FAILED, "manifest file count mismatch")
        if self.total_size_bytes != sum(entry.size_bytes for entry in self.files):
            _raise(PreflightRuleCode.GIT_COMMAND_FAILED, "manifest byte count mismatch")
        expected = sha256_bytes(
            canonical_json_bytes([entry.to_mapping() for entry in self.files])
        )
        if self.tree_sha256 != expected:
            _raise(
                PreflightRuleCode.GIT_COMMAND_FAILED, "manifest tree digest mismatch"
            )

    def to_mapping(self) -> dict[str, object]:
        return {
            "commit": self.commit,
            "file_count": self.file_count,
            "files": [entry.to_mapping() for entry in self.files],
            "git_object_format": self.git_object_format,
            "schema_version": self.schema_version,
            "total_size_bytes": self.total_size_bytes,
            "tree_sha256": self.tree_sha256,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> Self:
        """Strictly parse and self-validate a committed source manifest."""
        _strict_fields(
            value,
            frozenset(
                {
                    "commit",
                    "file_count",
                    "files",
                    "git_object_format",
                    "schema_version",
                    "total_size_bytes",
                    "tree_sha256",
                }
            ),
            "source snapshot manifest",
        )
        raw_files = value["files"]
        if not isinstance(raw_files, list):
            _raise(
                PreflightRuleCode.MANIFEST_INVALID,
                "source snapshot files must be an array",
            )
        return cls(
            schema_version=_text(value["schema_version"], "manifest schema"),
            commit=_text(value["commit"], "manifest commit"),
            git_object_format=_text(
                value["git_object_format"], "manifest object format"
            ),
            files=tuple(
                SourceFileDigest.from_mapping(_mapping(item, "source file digest"))
                for item in raw_files
            ),
            file_count=_integer(value["file_count"], "manifest file count"),
            total_size_bytes=_integer(value["total_size_bytes"], "manifest total size"),
            tree_sha256=_text(value["tree_sha256"], "manifest tree SHA-256"),
        )


@dataclass(frozen=True, slots=True)
class SourcePreflightReport:
    """Machine-readable decision separating snapshot readiness from licensing."""

    repository_root: str | None
    head_commit: str | None
    clean_worktree: bool
    source_snapshot_ready: bool
    public_release_authorized: bool
    git_cli_identity: GitCliIdentity | None
    manifest: SourceSnapshotManifest | None
    blockers: tuple[PreflightBlocker, ...]

    def __post_init__(self) -> None:
        if self.public_release_authorized:
            _raise(
                PreflightRuleCode.GIT_COMMAND_FAILED,
                "source preflight cannot authorize legal or public release",
            )
        expected_ready = (
            self.repository_root is not None
            and self.head_commit is not None
            and self.clean_worktree
            and self.git_cli_identity is not None
            and self.manifest is not None
            and not self.blockers
        )
        if self.source_snapshot_ready != expected_ready:
            _raise(
                PreflightRuleCode.GIT_COMMAND_FAILED, "inconsistent preflight result"
            )

    def to_mapping(self) -> dict[str, object]:
        return {
            "blockers": [blocker.to_mapping() for blocker in self.blockers],
            "clean_worktree": self.clean_worktree,
            "head_commit": self.head_commit,
            "git_cli_identity": (
                None
                if self.git_cli_identity is None
                else self.git_cli_identity.to_mapping()
            ),
            "manifest": None if self.manifest is None else self.manifest.to_mapping(),
            "public_release_authorized": self.public_release_authorized,
            "repository_root": self.repository_root,
            "schema_version": "policytwin.source-preflight.v2",
            "source_snapshot_ready": self.source_snapshot_ready,
        }


@dataclass(frozen=True, slots=True)
class SourceArchiveReceipt:
    """Digest receipt for an exported deterministic Git archive and manifest."""

    archive_sha256: str
    archive_size_bytes: int
    manifest_path: str
    git_cli_identity: GitCliIdentity
    source_manifest: SourceSnapshotManifest

    def __post_init__(self) -> None:
        if _SHA256.fullmatch(self.archive_sha256) is None:
            _raise(PreflightRuleCode.GIT_COMMAND_FAILED, "invalid archive digest")
        if self.archive_size_bytes <= 0:
            _raise(PreflightRuleCode.GIT_COMMAND_FAILED, "archive must not be empty")
        if not self.manifest_path:
            _raise(PreflightRuleCode.OUTPUT_PATH_INVALID, "empty manifest path")
        if not isinstance(self.git_cli_identity, GitCliIdentity):
            _raise(
                PreflightRuleCode.GIT_TRUST_BOUNDARY_FAILED,
                "archive receipt lacks a valid Git CLI identity",
            )

    def to_mapping(self) -> dict[str, object]:
        return {
            "archive_sha256": self.archive_sha256,
            "archive_size_bytes": self.archive_size_bytes,
            "git_cli_identity": self.git_cli_identity.to_mapping(),
            "manifest_path": self.manifest_path,
            "schema_version": SOURCE_ARCHIVE_SCHEMA_VERSION,
            "source_manifest": self.source_manifest.to_mapping(),
        }


@dataclass(frozen=True, slots=True)
class _GitResult:
    returncode: int
    stdout: bytes


@dataclass(frozen=True, slots=True)
class _StableExportFileCapture:
    """Identity and contents read from one continuously held archive fd."""

    device: int
    inode: int
    mode: int
    link_count: int
    size_bytes: int
    mtime_ns: int
    ctime_ns: int
    sha256: str


def _capture_stable_export_file(
    descriptor: int,
    path: Path,
    *,
    expected_identity: tuple[int, int],
    expected_link_count: int,
    context: str,
) -> _StableExportFileCapture:
    """Hash one held regular file and prove its pathname stayed on that inode."""
    try:
        before = os.fstat(descriptor)
        linked_before = os.lstat(path)
    except OSError as error:
        raise SourcePreflightError(
            f"cannot inspect {context}",
            PreflightRuleCode.MANIFEST_REFERENCE_MISMATCH,
        ) from error
    stable_fields = (
        "st_dev",
        "st_ino",
        "st_mode",
        "st_nlink",
        "st_size",
        "st_mtime_ns",
        "st_ctime_ns",
    )
    if (
        not stat.S_ISREG(before.st_mode)
        or stat.S_ISLNK(linked_before.st_mode)
        or not stat.S_ISREG(linked_before.st_mode)
        or (before.st_dev, before.st_ino) != expected_identity
        or (linked_before.st_dev, linked_before.st_ino) != expected_identity
        or before.st_nlink != expected_link_count
        or linked_before.st_nlink != expected_link_count
        or any(
            getattr(before, field) != getattr(linked_before, field)
            for field in stable_fields
        )
    ):
        _raise(
            PreflightRuleCode.MANIFEST_REFERENCE_MISMATCH,
            f"{context} identity, type, or link count changed",
        )
    digest = hashlib.sha256()
    total = 0
    try:
        os.lseek(descriptor, 0, os.SEEK_SET)
        while total < before.st_size:
            chunk = os.read(descriptor, min(1024 * 1024, before.st_size - total))
            if not chunk:
                raise OSError("short read while hashing archive")
            total += len(chunk)
            digest.update(chunk)
        if os.read(descriptor, 1):
            raise OSError("archive grew while it was hashed")
        after = os.fstat(descriptor)
        linked_after = os.lstat(path)
    except OSError as error:
        raise SourcePreflightError(
            f"cannot stably hash {context}",
            PreflightRuleCode.MANIFEST_REFERENCE_MISMATCH,
        ) from error
    if (
        total != before.st_size
        or any(
            getattr(before, field) != getattr(after, field) for field in stable_fields
        )
        or any(
            getattr(before, field) != getattr(linked_after, field)
            for field in stable_fields
        )
    ):
        _raise(
            PreflightRuleCode.MANIFEST_REFERENCE_MISMATCH,
            f"{context} changed while it was hashed",
        )
    return _StableExportFileCapture(
        device=before.st_dev,
        inode=before.st_ino,
        mode=before.st_mode,
        link_count=before.st_nlink,
        size_bytes=before.st_size,
        mtime_ns=before.st_mtime_ns,
        ctime_ns=before.st_ctime_ns,
        sha256=digest.hexdigest(),
    )


def _git(root: Path, identity: GitCliIdentity, *arguments: str) -> _GitResult:
    try:
        result = run_git_command(
            root,
            arguments,
            expected_cli_identity=identity,
        )
    except GitBoundaryError as error:
        raise SourcePreflightError(
            f"Git trust boundary failed: {error}",
            PreflightRuleCode.GIT_TRUST_BOUNDARY_FAILED,
        ) from error
    return _GitResult(result.returncode, result.stdout)


def _decode_line(result: _GitResult) -> str | None:
    if result.returncode != 0:
        return None
    try:
        return result.stdout.decode("utf-8", errors="strict").strip()
    except UnicodeError:
        return None


def _verify_git_object_database(
    root: Path, identity: GitCliIdentity, commit: str
) -> None:
    result = _git(
        root,
        identity,
        "fsck",
        "--full",
        "--strict",
        "--no-reflogs",
        "--no-dangling",
        commit,
    )
    if result.returncode != 0:
        _raise(
            PreflightRuleCode.GIT_TRUST_BOUNDARY_FAILED,
            "Git object database failed strict content verification",
        )


def _forbidden_path_reason(path_text: str) -> str | None:
    path = PurePosixPath(path_text)
    if any(part in _CACHE_PARTS for part in path.parts):
        return "cache or local environment path"
    if path.name == ".env" or path.name.startswith(".env."):
        return "environment-secret file"
    if path.suffix.lower() in _PRIVATE_FILE_SUFFIXES:
        return "private credential file suffix"
    if path.parts and path.parts[0] == "runs":
        return "generated run bundle"
    if path_text.startswith("experiments/smoke_matrix/runs/") and not (
        len(path.parts) == 5
        and path.parts[:3] == ("experiments", "smoke_matrix", "runs")
        and _SAFE_ROW_ID.fullmatch(path.parts[3]) is not None
        and path.parts[4] in {"result.json", "result.sha256"}
    ):
        return "undeclared or noncanonical experiment-run path"
    if (
        len(path.parts) >= 3
        and path.parts[:2]
        in {
            ("data", "raw"),
            ("data", "snapshots"),
            ("data", "processed"),
        }
        and path.name != ".gitkeep"
    ):
        return "local research data"
    if (
        path.parts
        and path.parts[0] == "paper"
        and path.suffix in _GENERATED_PAPER_SUFFIXES
    ):
        return "generated manuscript output"
    return None


def _closed_manifest_blockers(
    manifest_path: str,
    contents: Mapping[str, bytes],
    entries: Mapping[str, SourceFileDigest],
) -> list[PreflightBlocker]:
    """Require a committed experiment run manifest to be self-contained."""
    content = contents.get(manifest_path)
    if content is None:
        return []
    try:
        manifest = read_json_bytes_object(content, source=manifest_path)
    except MetadataValidationError:
        return [
            PreflightBlocker(
                PreflightRuleCode.MANIFEST_INVALID,
                "closed artifact manifest is not strict JSON",
                manifest_path,
            )
        ]
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        return [
            PreflightBlocker(
                PreflightRuleCode.MANIFEST_INVALID,
                "closed artifact manifest must contain a non-empty artifacts array",
                manifest_path,
            )
        ]

    blockers: list[PreflightBlocker] = []
    manifest_parent = PurePosixPath(manifest_path).parent
    referenced: set[str] = set()
    for raw_artifact in artifacts:
        if not isinstance(raw_artifact, Mapping):
            blockers.append(
                PreflightBlocker(
                    PreflightRuleCode.MANIFEST_INVALID,
                    "closed artifact manifest contains a non-object entry",
                    manifest_path,
                )
            )
            continue
        raw_path = raw_artifact.get("path")
        expected_sha256 = raw_artifact.get("sha256")
        expected_size = raw_artifact.get("size_bytes")
        if (
            not isinstance(raw_path, str)
            or not _safe_relative_path(raw_path)
            or not isinstance(expected_sha256, str)
            or _SHA256.fullmatch(expected_sha256) is None
            or isinstance(expected_size, bool)
            or not isinstance(expected_size, int)
            or expected_size < 0
        ):
            blockers.append(
                PreflightBlocker(
                    PreflightRuleCode.MANIFEST_INVALID,
                    "closed artifact manifest entry has invalid path, hash, or size",
                    manifest_path,
                )
            )
            continue
        committed_path = (manifest_parent / PurePosixPath(raw_path)).as_posix()
        if not _safe_relative_path(committed_path) or committed_path in referenced:
            blockers.append(
                PreflightBlocker(
                    PreflightRuleCode.MANIFEST_INVALID,
                    "closed artifact manifest has an unsafe or duplicate reference",
                    manifest_path,
                )
            )
            continue
        referenced.add(committed_path)
        entry = entries.get(committed_path)
        if entry is None:
            blockers.append(
                PreflightBlocker(
                    PreflightRuleCode.MANIFEST_REFERENCE_MISSING,
                    "closed artifact manifest references bytes absent from HEAD",
                    committed_path,
                )
            )
        elif entry.sha256 != expected_sha256 or entry.size_bytes != expected_size:
            blockers.append(
                PreflightBlocker(
                    PreflightRuleCode.MANIFEST_REFERENCE_MISMATCH,
                    "closed artifact manifest hash or size differs from HEAD",
                    committed_path,
                )
            )

    committed_run_paths = {
        path for path in entries if path.startswith("experiments/smoke_matrix/runs/")
    }
    for undeclared_path in sorted(committed_run_paths - referenced):
        blockers.append(
            PreflightBlocker(
                PreflightRuleCode.MANIFEST_UNDECLARED_ARTIFACT,
                "versioned synthetic run artifact is absent from its closed manifest",
                undeclared_path,
            )
        )

    seal_path = f"{manifest_path.removesuffix('.json')}.sha256"
    seal = contents.get(seal_path)
    expected_seal = f"{sha256_bytes(canonical_json_bytes(manifest))}\n".encode()
    if seal is None:
        blockers.append(
            PreflightBlocker(
                PreflightRuleCode.MANIFEST_REFERENCE_MISSING,
                "closed artifact manifest seal is absent from HEAD",
                seal_path,
            )
        )
    elif seal != expected_seal:
        blockers.append(
            PreflightBlocker(
                PreflightRuleCode.MANIFEST_REFERENCE_MISMATCH,
                "closed artifact manifest seal differs from committed bytes",
                seal_path,
            )
        )
    return blockers


_BLOCKED_PAPER_MANIFEST_FIELDS = frozenset(
    {
        "config_sha256",
        "figure_spec_sha256",
        "outputs",
        "reason",
        "registry_sha256",
        "result_sha256",
        "schema_version",
        "status",
        "synthetic_only",
    }
)


def _blocked_paper_manifest_blockers(
    contents: Mapping[str, bytes],
    entries: Mapping[str, SourceFileDigest],
) -> list[PreflightBlocker]:
    """Keep the checked-in source tree free of unregistered paper results."""

    manifest_path = "paper/artifact_manifest.json"
    seal_path = "paper/artifact_manifest.sha256"
    content = contents.get(manifest_path)
    if content is None:
        return []
    blockers: list[PreflightBlocker] = []
    try:
        manifest = read_json_bytes_object(content, source=manifest_path)
    except MetadataValidationError:
        blockers.append(
            PreflightBlocker(
                PreflightRuleCode.MANIFEST_INVALID,
                "paper artifact sentinel is not strict JSON",
                manifest_path,
            )
        )
        manifest = None

    if manifest is not None:
        if frozenset(manifest) != _BLOCKED_PAPER_MANIFEST_FIELDS:
            blockers.append(
                PreflightBlocker(
                    PreflightRuleCode.MANIFEST_INVALID,
                    "paper artifact sentinel fields differ from the closed schema",
                    manifest_path,
                )
            )
        elif (
            manifest["schema_version"] != "policytwin.paper-artifact-manifest.v1"
            or manifest["status"] != "blocked"
            or manifest["synthetic_only"] is not True
            or not isinstance(manifest["reason"], str)
            or not manifest["reason"].strip()
            or any(
                manifest[field] is not None
                for field in (
                    "config_sha256",
                    "result_sha256",
                    "registry_sha256",
                    "figure_spec_sha256",
                )
            )
            or manifest["outputs"] != []
        ):
            blockers.append(
                PreflightBlocker(
                    PreflightRuleCode.MANIFEST_INVALID,
                    (
                        "paper artifact sentinel must remain blocked with no claims "
                        "or outputs"
                    ),
                    manifest_path,
                )
            )

    expected_seal = f"{sha256_bytes(content)}\n".encode("ascii")
    seal = contents.get(seal_path)
    if seal is None:
        blockers.append(
            PreflightBlocker(
                PreflightRuleCode.MANIFEST_REFERENCE_MISSING,
                "paper artifact sentinel seal is absent from HEAD",
                seal_path,
            )
        )
    elif seal != expected_seal:
        blockers.append(
            PreflightBlocker(
                PreflightRuleCode.MANIFEST_REFERENCE_MISMATCH,
                "paper artifact sentinel seal differs from committed bytes",
                seal_path,
            )
        )

    generated_prefixes = ("paper/figures/", "paper/tables/")
    for path in sorted(entries):
        if path.endswith("/.gitkeep"):
            continue
        if path.startswith(generated_prefixes):
            blockers.append(
                PreflightBlocker(
                    PreflightRuleCode.MANIFEST_UNDECLARED_ARTIFACT,
                    (
                        "paper figure or table is not allowed in the blocked source "
                        "snapshot"
                    ),
                    path,
                )
            )
    return blockers


def _literature_scoping_manifest_blockers(
    contents: Mapping[str, bytes],
) -> list[PreflightBlocker]:
    """Require the checked-in literature map to remain sealed and non-authorizing."""

    manifest_path = "docs/literature_search_manifest.json"
    seal_path = "docs/literature_search_manifest.sha256"
    content = contents.get(manifest_path)
    if content is None:
        return []
    blockers: list[PreflightBlocker] = []
    try:
        manifest = read_json_bytes_object(content, source=manifest_path)
    except MetadataValidationError:
        blockers.append(
            PreflightBlocker(
                PreflightRuleCode.MANIFEST_INVALID,
                "literature search sentinel is not strict JSON",
                manifest_path,
            )
        )
        manifest = None
    if manifest is not None:
        expected_fields = frozenset(
            {
                "limitations",
                "schema_version",
                "screening",
                "search_runs",
                "searched_at",
                "sources",
                "status",
            }
        )
        screening = manifest.get("screening")
        if frozenset(manifest) != expected_fields or not isinstance(screening, Mapping):
            blockers.append(
                PreflightBlocker(
                    PreflightRuleCode.MANIFEST_INVALID,
                    "literature search sentinel fields differ from the closed schema",
                    manifest_path,
                )
            )
        else:
            expected_screening_fields = frozenset(
                {
                    "dual_reviewer_complete",
                    "novelty_claim_authorized",
                    "reviewer_count",
                    "systematic_review_complete",
                }
            )
            if (
                frozenset(screening) != expected_screening_fields
                or manifest["schema_version"] != "policytwin.literature-search.v1"
                or manifest["status"] != "scoping_search_incomplete"
                or screening["dual_reviewer_complete"] is not False
                or screening["novelty_claim_authorized"] is not False
                or screening["systematic_review_complete"] is not False
                or screening["reviewer_count"] != 1
                or not isinstance(manifest["search_runs"], list)
                or not manifest["search_runs"]
                or not isinstance(manifest["sources"], list)
                or not manifest["sources"]
                or not isinstance(manifest["limitations"], list)
                or not manifest["limitations"]
            ):
                blockers.append(
                    PreflightBlocker(
                        PreflightRuleCode.MANIFEST_INVALID,
                        (
                            "literature search sentinel must remain an incomplete, "
                            "single-reviewer, non-authorizing scoping record"
                        ),
                        manifest_path,
                    )
                )
    expected_seal = f"{sha256_bytes(content)}\n".encode("ascii")
    seal = contents.get(seal_path)
    if seal is None:
        blockers.append(
            PreflightBlocker(
                PreflightRuleCode.MANIFEST_REFERENCE_MISSING,
                "literature search sentinel seal is absent from HEAD",
                seal_path,
            )
        )
    elif seal != expected_seal:
        blockers.append(
            PreflightBlocker(
                PreflightRuleCode.MANIFEST_REFERENCE_MISMATCH,
                "literature search sentinel seal differs from committed bytes",
                seal_path,
            )
        )
    return blockers


def _outcome_knowledge_manifest_blockers(
    contents: Mapping[str, bytes],
) -> list[PreflightBlocker]:
    """Keep the checked-in hindsight ledger present and non-authorizing.

    The adjacent digest detects accidental byte drift inside one committed
    snapshot.  It is deliberately not treated as an external registration or
    verification receipt; the default source release accepts only the current
    self-reported, locator-only status.
    """

    manifest_path = "configs/policies/sec_tick_size_pilot_outcome_knowledge.json"
    seal_path = "configs/policies/sec_tick_size_pilot_outcome_knowledge.sha256"
    content = contents.get(manifest_path)
    if content is None:
        return []
    blockers: list[PreflightBlocker] = []
    try:
        manifest = read_json_bytes_object(content, source=manifest_path)
    except MetadataValidationError:
        blockers.append(
            PreflightBlocker(
                PreflightRuleCode.MANIFEST_INVALID,
                "outcome-knowledge ledger is not strict JSON",
                manifest_path,
            )
        )
        manifest = None
    if manifest is not None:
        from policytwin.outcome_knowledge import (
            OutcomeKnowledgeError,
            OutcomeKnowledgeManifest,
        )

        try:
            typed_manifest = OutcomeKnowledgeManifest.from_mapping(manifest)
        except OutcomeKnowledgeError:
            typed_manifest = None
        if (
            typed_manifest is None
            or typed_manifest.sha256 != _SEC_OUTCOME_KNOWLEDGE_SENTINEL_SEMANTIC_SHA256
        ):
            blockers.append(
                PreflightBlocker(
                    PreflightRuleCode.MANIFEST_INVALID,
                    (
                        "checked-in outcome-knowledge ledger must preserve the "
                        "reviewed hindsight-exposure sentinel exactly; updating "
                        "the companion checksum alone is not authority"
                    ),
                    manifest_path,
                )
            )
    expected_seal = f"{sha256_bytes(content)}\n".encode("ascii")
    seal = contents.get(seal_path)
    if seal is None:
        blockers.append(
            PreflightBlocker(
                PreflightRuleCode.MANIFEST_REFERENCE_MISSING,
                "outcome-knowledge companion checksum is absent from HEAD",
                seal_path,
            )
        )
    elif seal != expected_seal:
        blockers.append(
            PreflightBlocker(
                PreflightRuleCode.MANIFEST_REFERENCE_MISMATCH,
                "outcome-knowledge companion checksum differs from committed bytes",
                seal_path,
            )
        )
    return blockers


def _tree_manifest(
    root: Path,
    commit: str,
    object_format: str,
    git_cli_identity: GitCliIdentity,
    config: SourcePreflightConfig,
) -> tuple[SourceSnapshotManifest | None, list[PreflightBlocker]]:
    blockers: list[PreflightBlocker] = []
    listing = _git(root, git_cli_identity, "ls-tree", "-r", "-z", "--full-tree", commit)
    if listing.returncode != 0:
        return None, [
            PreflightBlocker(
                PreflightRuleCode.GIT_COMMAND_FAILED,
                "cannot enumerate the committed source tree",
            )
        ]

    entries: list[SourceFileDigest] = []
    contents: dict[str, bytes] = {}
    seen: set[str] = set()
    for record in listing.stdout.split(b"\0"):
        if not record:
            continue
        try:
            metadata, raw_path = record.split(b"\t", maxsplit=1)
            mode, object_type, object_id = metadata.decode("ascii").split(" ")
            path_text = raw_path.decode("utf-8", errors="strict")
        except (UnicodeError, ValueError):
            blockers.append(
                PreflightBlocker(
                    PreflightRuleCode.UNSAFE_TREE_ENTRY,
                    "Git tree entry is not a canonical UTF-8 regular-file record",
                )
            )
            continue
        if not _safe_relative_path(path_text) or path_text in seen:
            blockers.append(
                PreflightBlocker(
                    PreflightRuleCode.UNSAFE_TREE_ENTRY,
                    "Git tree contains an unsafe or duplicate path",
                )
            )
            continue
        seen.add(path_text)
        if object_type != "blob" or mode not in _SAFE_MODES:
            blockers.append(
                PreflightBlocker(
                    PreflightRuleCode.UNSAFE_TREE_ENTRY,
                    "only regular non-symlink files may enter a source snapshot",
                    path_text,
                )
            )
            continue
        expected_object_id_length = 40 if object_format == "sha1" else 64
        if re.fullmatch(rf"[0-9a-f]{{{expected_object_id_length}}}", object_id) is None:
            _raise(
                PreflightRuleCode.GIT_TRUST_BOUNDARY_FAILED,
                "Git tree contains an invalid blob object ID",
            )
        reason = _forbidden_path_reason(path_text)
        if reason is not None:
            blockers.append(
                PreflightBlocker(
                    PreflightRuleCode.FORBIDDEN_PATH,
                    f"committed path is forbidden: {reason}",
                    path_text,
                )
            )
            continue
        size_result = _git(root, git_cli_identity, "cat-file", "-s", object_id)
        size_text = _decode_line(size_result)
        try:
            size_bytes = int(size_text) if size_text is not None else -1
        except ValueError:
            size_bytes = -1
        if size_bytes < 0:
            blockers.append(
                PreflightBlocker(
                    PreflightRuleCode.GIT_COMMAND_FAILED,
                    "cannot determine committed blob size",
                    path_text,
                )
            )
            continue
        if size_bytes > config.max_file_size_bytes:
            blockers.append(
                PreflightBlocker(
                    PreflightRuleCode.FILE_TOO_LARGE,
                    f"committed blob exceeds {config.max_file_size_bytes} bytes",
                    path_text,
                )
            )
            continue
        content_result = _git(root, git_cli_identity, "cat-file", "blob", object_id)
        if content_result.returncode != 0 or len(content_result.stdout) != size_bytes:
            blockers.append(
                PreflightBlocker(
                    PreflightRuleCode.GIT_COMMAND_FAILED,
                    "cannot read the exact committed blob",
                    path_text,
                )
            )
            continue
        object_hasher = hashlib.new(object_format)
        object_hasher.update(f"blob {size_bytes}\0".encode("ascii"))
        object_hasher.update(content_result.stdout)
        if object_hasher.hexdigest() != object_id:
            _raise(
                PreflightRuleCode.GIT_TRUST_BOUNDARY_FAILED,
                "committed blob bytes do not reproduce their Git object ID",
            )
        credential_kind = next(
            (
                name
                for name, pattern in _CREDENTIAL_PATTERNS
                if pattern.search(content_result.stdout) is not None
            ),
            None,
        )
        if credential_kind is not None:
            blockers.append(
                PreflightBlocker(
                    PreflightRuleCode.CREDENTIAL_PATTERN,
                    f"committed blob matches credential pattern: {credential_kind}",
                    path_text,
                )
            )
            continue
        entries.append(
            SourceFileDigest(
                path=path_text,
                mode=mode,
                size_bytes=size_bytes,
                sha256=sha256_bytes(content_result.stdout),
            )
        )
        contents[path_text] = content_result.stdout

    for required_path in config.required_paths:
        if required_path not in seen:
            blockers.append(
                PreflightBlocker(
                    PreflightRuleCode.REQUIRED_FILE_MISSING,
                    "required reproducibility file is absent from HEAD",
                    required_path,
                )
            )
    entries_by_path = {entry.path: entry for entry in entries}
    for manifest_path in config.closed_artifact_manifests:
        blockers.extend(
            _closed_manifest_blockers(manifest_path, contents, entries_by_path)
        )
    blockers.extend(_blocked_paper_manifest_blockers(contents, entries_by_path))
    blockers.extend(_literature_scoping_manifest_blockers(contents))
    blockers.extend(_outcome_knowledge_manifest_blockers(contents))
    if blockers:
        return None, blockers
    files = tuple(sorted(entries, key=lambda entry: entry.path))
    return (
        SourceSnapshotManifest(
            schema_version=SOURCE_MANIFEST_SCHEMA_VERSION,
            commit=commit,
            git_object_format=object_format,
            files=files,
            file_count=len(files),
            total_size_bytes=sum(entry.size_bytes for entry in files),
            tree_sha256=sha256_bytes(
                canonical_json_bytes([entry.to_mapping() for entry in files])
            ),
        ),
        blockers,
    )


def run_source_preflight(
    project_root: Path,
    *,
    config: SourcePreflightConfig | None = None,
) -> SourcePreflightReport:
    """Inspect HEAD and the live checkout, returning a fail-closed decision."""
    policy = config if config is not None else SourcePreflightConfig()
    try:
        git_cli_identity = verify_git_cli_identity()
        requested_root = project_root.expanduser().resolve(strict=True)
        discovered_root = discover_repository_root(requested_root)
    except (OSError, GitBoundaryError) as error:
        raise SourcePreflightError(
            f"Git trust boundary failed: {error}",
            PreflightRuleCode.GIT_TRUST_BOUNDARY_FAILED,
        ) from error
    if discovered_root is None:
        blocker = PreflightBlocker(
            PreflightRuleCode.NOT_GIT_REPOSITORY,
            "project root is not inside a standalone Git repository",
        )
        return SourcePreflightReport(
            repository_root=None,
            head_commit=None,
            clean_worktree=False,
            source_snapshot_ready=False,
            public_release_authorized=False,
            git_cli_identity=git_cli_identity,
            manifest=None,
            blockers=(blocker,),
        )
    repository_root = discovered_root
    replace_result = _git(
        repository_root,
        git_cli_identity,
        "for-each-ref",
        "--format=%(refname)",
        "refs/replace/",
    )
    if replace_result.returncode != 0 or replace_result.stdout.strip():
        raise SourcePreflightError(
            "repository contains or cannot exclude commit replacement refs",
            PreflightRuleCode.GIT_TRUST_BOUNDARY_FAILED,
        )
    root_result = _git(
        repository_root, git_cli_identity, "rev-parse", "--show-toplevel"
    )
    root_text = _decode_line(root_result)
    if root_text is None:
        blocker = PreflightBlocker(
            PreflightRuleCode.NOT_GIT_REPOSITORY,
            "project root is not inside a readable Git repository",
        )
        return SourcePreflightReport(
            repository_root=None,
            head_commit=None,
            clean_worktree=False,
            source_snapshot_ready=False,
            public_release_authorized=False,
            git_cli_identity=git_cli_identity,
            manifest=None,
            blockers=(blocker,),
        )
    reported_root = Path(root_text).resolve(strict=True)
    blockers: list[PreflightBlocker] = []
    if reported_root != repository_root:
        blockers.append(
            PreflightBlocker(
                PreflightRuleCode.GIT_TRUST_BOUNDARY_FAILED,
                "Git-reported root differs from the filesystem trust root",
            )
        )
    if repository_root != requested_root:
        blockers.append(
            PreflightBlocker(
                PreflightRuleCode.REPOSITORY_ROOT_MISMATCH,
                "project_root must be the exact Git repository root",
            )
        )

    commit_result = _git(
        repository_root,
        git_cli_identity,
        "rev-parse",
        "--verify",
        "HEAD^{commit}",
    )
    commit = _decode_line(commit_result)
    if commit is None or _OBJECT_ID.fullmatch(commit) is None:
        blockers.append(
            PreflightBlocker(
                PreflightRuleCode.MISSING_HEAD,
                "repository has no valid committed HEAD",
            )
        )
        commit = None

    tracked_clean = commit is not None
    if commit is not None:
        tracked_state_failed = False
        for arguments in (
            ("diff", "--no-ext-diff", "--no-textconv", "--quiet", "HEAD", "--"),
            (
                "diff",
                "--cached",
                "--no-ext-diff",
                "--no-textconv",
                "--quiet",
                "HEAD",
                "--",
            ),
        ):
            result = _git(repository_root, git_cli_identity, *arguments)
            if result.returncode == 1:
                tracked_clean = False
            elif result.returncode != 0:
                tracked_clean = False
                tracked_state_failed = True
        if tracked_state_failed:
            blockers.append(
                PreflightBlocker(
                    PreflightRuleCode.GIT_COMMAND_FAILED,
                    "cannot determine tracked working-tree state",
                )
            )
        elif not tracked_clean:
            blockers.append(
                PreflightBlocker(
                    PreflightRuleCode.TRACKED_CHANGES,
                    "tracked or staged bytes differ from HEAD",
                )
            )

    untracked_result = _git(
        repository_root,
        git_cli_identity,
        "ls-files",
        "--others",
        "--exclude-standard",
        "-z",
    )
    if untracked_result.returncode != 0:
        blockers.append(
            PreflightBlocker(
                PreflightRuleCode.GIT_COMMAND_FAILED,
                "cannot enumerate untracked files",
            )
        )
        untracked = True
    else:
        untracked = bool(untracked_result.stdout)
    if untracked:
        blockers.append(
            PreflightBlocker(
                PreflightRuleCode.UNTRACKED_FILES,
                "non-ignored untracked files are present",
            )
        )

    manifest: SourceSnapshotManifest | None = None
    if commit is not None:
        _verify_git_object_database(repository_root, git_cli_identity, commit)
        format_result = _git(
            repository_root, git_cli_identity, "rev-parse", "--show-object-format"
        )
        object_format = _decode_line(format_result)
        if object_format not in {"sha1", "sha256"}:
            blockers.append(
                PreflightBlocker(
                    PreflightRuleCode.GIT_COMMAND_FAILED,
                    "cannot determine Git object format",
                )
            )
        else:
            manifest, tree_blockers = _tree_manifest(
                repository_root,
                commit,
                object_format,
                git_cli_identity,
                policy,
            )
            blockers.extend(tree_blockers)
            _verify_git_object_database(repository_root, git_cli_identity, commit)

    blockers = sorted(
        blockers,
        key=lambda blocker: (
            blocker.rule_code.value,
            blocker.path or "",
            blocker.message,
        ),
    )
    clean_worktree = tracked_clean and not untracked
    ready = (
        repository_root == requested_root
        and commit is not None
        and clean_worktree
        and reported_root == repository_root
        and manifest is not None
        and not blockers
    )
    try:
        assert_trusted_repository_layout(repository_root)
        require_same_git_cli_identity(
            git_cli_identity, context="after source preflight"
        )
    except GitBoundaryError as error:
        raise SourcePreflightError(
            f"Git trust boundary changed during preflight: {error}",
            PreflightRuleCode.GIT_TRUST_BOUNDARY_FAILED,
        ) from error
    return SourcePreflightReport(
        repository_root=str(repository_root),
        head_commit=commit,
        clean_worktree=clean_worktree,
        source_snapshot_ready=ready,
        public_release_authorized=False,
        git_cli_identity=git_cli_identity,
        manifest=manifest,
        blockers=tuple(blockers),
    )


def _verify_archive_against_manifest(
    archive_path: Path, manifest: SourceSnapshotManifest
) -> None:
    """Require every generated tar member to match the committed manifest."""
    expected = {entry.path: entry for entry in manifest.files}
    expected_directories = {""}
    for relative in expected:
        parent = PurePosixPath(relative).parent
        while parent.as_posix() != ".":
            expected_directories.add(parent.as_posix())
            parent = parent.parent
    seen_files: set[str] = set()
    seen_directories: set[str] = set()
    try:
        with tarfile.open(archive_path, mode="r:") as archive:
            for member in archive:
                archive_relative = _archive_relative(member.name)
                if archive_relative is None:
                    _raise(
                        PreflightRuleCode.UNSAFE_TREE_ENTRY,
                        "generated source archive contains an unsafe path",
                    )
                if member.isdir():
                    if (
                        archive_relative not in expected_directories
                        or archive_relative in seen_directories
                    ):
                        _raise(
                            PreflightRuleCode.MANIFEST_REFERENCE_MISMATCH,
                            "generated archive directory set differs from manifest",
                        )
                    seen_directories.add(archive_relative)
                    continue
                entry = expected.get(archive_relative)
                if (
                    entry is None
                    or archive_relative in seen_files
                    or not member.isfile()
                    or member.size != entry.size_bytes
                    or bool(member.mode & 0o111) != (entry.mode == "100755")
                    or bool(member.mode & 0o7000)
                ):
                    _raise(
                        PreflightRuleCode.MANIFEST_REFERENCE_MISMATCH,
                        "generated archive entry differs from committed manifest",
                    )
                stream = archive.extractfile(member)
                if stream is None:
                    _raise(
                        PreflightRuleCode.MANIFEST_REFERENCE_MISSING,
                        "generated archive file bytes are unavailable",
                    )
                digest = hashlib.sha256()
                total = 0
                while True:
                    chunk = stream.read(1024 * 1024)
                    if not chunk:
                        break
                    total += len(chunk)
                    if total > entry.size_bytes:
                        _raise(
                            PreflightRuleCode.MANIFEST_REFERENCE_MISMATCH,
                            "generated archive file exceeds its committed size",
                        )
                    digest.update(chunk)
                if total != entry.size_bytes or digest.hexdigest() != entry.sha256:
                    _raise(
                        PreflightRuleCode.MANIFEST_REFERENCE_MISMATCH,
                        "generated archive bytes differ from committed manifest",
                    )
                seen_files.add(archive_relative)
    except (tarfile.TarError, OSError) as error:
        raise SourcePreflightError(
            "generated source archive cannot be verified",
            PreflightRuleCode.MANIFEST_REFERENCE_MISMATCH,
        ) from error
    if seen_files != set(expected) or seen_directories != expected_directories:
        _raise(
            PreflightRuleCode.MANIFEST_REFERENCE_MISMATCH,
            "generated source archive path set differs from committed manifest",
        )


def _cleanup_partial_export_file(
    path: Path,
    *,
    expected_parent: Path,
    expected_name: str | None,
    expected_prefix: str | None,
    expected_identity: tuple[int, int] | None,
    context: str,
    primary_error: BaseException,
) -> None:
    """Remove one private export residual and prove its pathname is absent."""
    absolute = Path(os.path.abspath(path))
    parent = Path(os.path.abspath(expected_parent))
    safe_name = (
        expected_name is not None
        and expected_prefix is None
        and absolute.name == expected_name
    ) or (
        expected_name is None
        and expected_prefix is not None
        and absolute.name.startswith(expected_prefix)
    )
    if absolute.parent != parent or not safe_name:
        raise SourcePreflightError(
            f"refusing unsafe {context} cleanup path: {absolute}",
            PreflightRuleCode.PARTIAL_OUTPUT_CLEANUP_FAILED,
        ) from primary_error
    try:
        info = os.lstat(absolute)
    except FileNotFoundError:
        return
    except OSError as error:
        raise SourcePreflightError(
            f"cannot inspect {context} during cleanup: {error}",
            PreflightRuleCode.PARTIAL_OUTPUT_CLEANUP_FAILED,
        ) from primary_error
    if (
        stat.S_ISLNK(info.st_mode)
        or not stat.S_ISREG(info.st_mode)
        or (
            expected_identity is not None
            and (info.st_dev, info.st_ino) != expected_identity
        )
    ):
        raise SourcePreflightError(
            f"{context} changed identity or type before cleanup",
            PreflightRuleCode.PARTIAL_OUTPUT_CLEANUP_FAILED,
        ) from primary_error
    try:
        absolute.unlink()
        os.lstat(absolute)
    except FileNotFoundError:
        return
    except OSError as error:
        raise SourcePreflightError(
            f"{context} cleanup failed: {error}",
            PreflightRuleCode.PARTIAL_OUTPUT_CLEANUP_FAILED,
        ) from primary_error
    raise SourcePreflightError(
        f"{context} remains after cleanup returned",
        PreflightRuleCode.PARTIAL_OUTPUT_CLEANUP_FAILED,
    ) from primary_error


def _write_exclusive_json_file(
    path: Path,
    value: Mapping[str, object],
    identity_state: list[tuple[int, int]],
) -> None:
    """Create and fsync strict JSON without ever replacing an existing path."""
    content = canonical_json_bytes(value) + b"\n"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags, 0o600)
    try:
        before = os.fstat(descriptor)
        identity_state.append((before.st_dev, before.st_ino))
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise OSError("exclusive JSON destination is not a private regular file")
        written = 0
        while written < len(content):
            count = os.write(descriptor, content[written:])
            if count <= 0:
                raise OSError("short write while publishing source archive receipt")
            written += count
        os.fsync(descriptor)
        after = os.fstat(descriptor)
        linked = os.lstat(path)
        if (
            (after.st_dev, after.st_ino) != identity_state[0]
            or (linked.st_dev, linked.st_ino) != identity_state[0]
            or after.st_size != len(content)
        ):
            raise OSError("source archive receipt changed while it was published")
    finally:
        os.close(descriptor)


def export_source_snapshot(
    project_root: Path,
    destination: Path,
    *,
    config: SourcePreflightConfig | None = None,
) -> SourceArchiveReceipt:
    """Export a deterministic tar from verified HEAD and write its digest receipt."""
    report = run_source_preflight(project_root, config=config)
    if not report.source_snapshot_ready or report.manifest is None:
        rules = ",".join(blocker.rule_code.value for blocker in report.blockers)
        _raise(
            report.blockers[0].rule_code
            if report.blockers
            else PreflightRuleCode.GIT_COMMAND_FAILED,
            f"source snapshot preflight failed: {rules}",
        )
    if (
        report.repository_root is None
        or report.head_commit is None
        or report.git_cli_identity is None
    ):
        _raise(PreflightRuleCode.GIT_COMMAND_FAILED, "preflight identity is incomplete")
    repository_root = Path(report.repository_root)
    target = destination.expanduser().absolute()
    if not target.parent.is_dir() or target.name in {"", ".", ".."}:
        _raise(
            PreflightRuleCode.OUTPUT_PATH_INVALID,
            "archive parent must be an existing directory",
        )
    resolved_parent = target.parent.resolve()
    target = resolved_parent / target.name
    manifest_path = target.with_name(f"{target.name}.manifest.json")
    if target == repository_root or repository_root in target.parents:
        _raise(
            PreflightRuleCode.OUTPUT_PATH_INVALID,
            "archive destination must be outside the Git repository",
        )
    if (
        target.exists()
        or target.is_symlink()
        or manifest_path.exists()
        or manifest_path.is_symlink()
    ):
        _raise(
            PreflightRuleCode.OUTPUT_EXISTS,
            "archive or receipt destination already exists",
        )

    temporary_path: Path | None = None
    temporary_identity: tuple[int, int] | None = None
    temporary_descriptor: int | None = None
    verified_capture: _StableExportFileCapture | None = None
    published_identity: tuple[int, int] | None = None
    receipt_identity_state: list[tuple[int, int]] = []
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=resolved_parent, prefix=f".{target.name}.", delete=False
        ) as stream:
            temporary_path = Path(stream.name)
            try:
                completed = run_git_command(
                    repository_root,
                    (
                        "archive",
                        "--format=tar",
                        "--prefix=policytwin-source/",
                        report.head_commit,
                    ),
                    expected_cli_identity=report.git_cli_identity,
                    stdout=cast(BinaryIO, stream),
                )
            except GitBoundaryError as error:
                raise SourcePreflightError(
                    f"Git archive trust boundary failed: {error}",
                    PreflightRuleCode.GIT_TRUST_BOUNDARY_FAILED,
                ) from error
            stream.flush()
            os.fsync(stream.fileno())
            temporary_info = os.fstat(stream.fileno())
            temporary_identity = (temporary_info.st_dev, temporary_info.st_ino)
        if completed.returncode != 0:
            _raise(PreflightRuleCode.GIT_COMMAND_FAILED, "Git archive failed")
        assert temporary_path is not None
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        temporary_descriptor = os.open(temporary_path, flags)
        assert temporary_identity is not None
        preverification_capture = _capture_stable_export_file(
            temporary_descriptor,
            temporary_path,
            expected_identity=temporary_identity,
            expected_link_count=1,
            context="temporary source archive before manifest verification",
        )
        _verify_archive_against_manifest(temporary_path, report.manifest)
        verified_capture = _capture_stable_export_file(
            temporary_descriptor,
            temporary_path,
            expected_identity=temporary_identity,
            expected_link_count=1,
            context="verified temporary source archive",
        )
        if verified_capture != preverification_capture:
            _raise(
                PreflightRuleCode.MANIFEST_REFERENCE_MISMATCH,
                "temporary source archive changed during manifest verification",
            )
        _verify_git_object_database(
            repository_root, report.git_cli_identity, report.head_commit
        )
        try:
            assert_trusted_repository_layout(repository_root)
            require_same_git_cli_identity(
                report.git_cli_identity, context="after archive verification"
            )
        except GitBoundaryError as error:
            raise SourcePreflightError(
                f"Git trust boundary changed during archive export: {error}",
                PreflightRuleCode.GIT_TRUST_BOUNDARY_FAILED,
            ) from error
        receipt = SourceArchiveReceipt(
            archive_sha256=verified_capture.sha256,
            archive_size_bytes=verified_capture.size_bytes,
            manifest_path=manifest_path.name,
            git_cli_identity=report.git_cli_identity,
            source_manifest=report.manifest,
        )
        try:
            os.link(temporary_path, target, follow_symlinks=False)
        except FileExistsError as error:
            raise SourcePreflightError(
                "archive destination appeared before exclusive publication",
                PreflightRuleCode.OUTPUT_EXISTS,
            ) from error
        published_identity = temporary_identity
        published_info = os.lstat(target)
        if (
            published_info.st_dev,
            published_info.st_ino,
        ) != published_identity or published_info.st_nlink != 2:
            raise OSError("published source archive identity changed after linking")
        temporary_path.unlink()
        try:
            os.lstat(temporary_path)
        except FileNotFoundError:
            temporary_path = None
        else:
            raise OSError("temporary source archive remains after publication")
        try:
            _write_exclusive_json_file(
                manifest_path, receipt.to_mapping(), receipt_identity_state
            )
        except FileExistsError as error:
            raise SourcePreflightError(
                "source archive receipt appeared before exclusive publication",
                PreflightRuleCode.OUTPUT_EXISTS,
            ) from error
        final_preverification_capture = _capture_stable_export_file(
            temporary_descriptor,
            target,
            expected_identity=published_identity,
            expected_link_count=1,
            context="published source archive before manifest verification",
        )
        if (
            final_preverification_capture.device != verified_capture.device
            or final_preverification_capture.inode != verified_capture.inode
            or final_preverification_capture.mode != verified_capture.mode
            or final_preverification_capture.size_bytes != verified_capture.size_bytes
            or final_preverification_capture.sha256 != verified_capture.sha256
        ):
            _raise(
                PreflightRuleCode.MANIFEST_REFERENCE_MISMATCH,
                "published source archive differs from the verified archive",
            )
        _verify_archive_against_manifest(target, report.manifest)
        final_capture = _capture_stable_export_file(
            temporary_descriptor,
            target,
            expected_identity=published_identity,
            expected_link_count=1,
            context="verified published source archive",
        )
        if final_capture != final_preverification_capture:
            _raise(
                PreflightRuleCode.MANIFEST_REFERENCE_MISMATCH,
                "published source archive changed during manifest verification",
            )
        return receipt
    except Exception as error:
        cleanup_failures: list[str] = []
        cleanup_requests: list[
            tuple[Path, str | None, str | None, tuple[int, int] | None, str]
        ] = []
        if published_identity is not None:
            cleanup_requests.append(
                (
                    target,
                    target.name,
                    None,
                    published_identity,
                    "published source archive",
                )
            )
        if receipt_identity_state:
            cleanup_requests.append(
                (
                    manifest_path,
                    manifest_path.name,
                    None,
                    receipt_identity_state[0],
                    "source archive receipt",
                )
            )
        if temporary_path is not None:
            cleanup_requests.append(
                (
                    temporary_path,
                    None,
                    f".{target.name}.",
                    temporary_identity,
                    "temporary source archive",
                )
            )
        for (
            cleanup_path,
            expected_name,
            expected_prefix,
            expected_identity,
            context,
        ) in cleanup_requests:
            try:
                _cleanup_partial_export_file(
                    cleanup_path,
                    expected_parent=resolved_parent,
                    expected_name=expected_name,
                    expected_prefix=expected_prefix,
                    expected_identity=expected_identity,
                    context=context,
                    primary_error=error,
                )
            except SourcePreflightError as cleanup_error:
                cleanup_failures.append(str(cleanup_error))
        if cleanup_failures:
            raise SourcePreflightError(
                "source archive export failed and cleanup was incomplete; "
                f"primary_error={error}; cleanup_errors="
                + " | ".join(cleanup_failures),
                PreflightRuleCode.PARTIAL_OUTPUT_CLEANUP_FAILED,
            ) from error
        raise
    finally:
        if temporary_descriptor is not None:
            os.close(temporary_descriptor)


def _archive_relative(value: str) -> str | None:
    if "\\" in value or any(ord(character) < 32 for character in value):
        return None
    path = PurePosixPath(value)
    if path.is_absolute() or not path.parts or path.parts[0] != _ARCHIVE_PREFIX:
        return None
    relative = PurePosixPath(*path.parts[1:])
    if any(part in {"", ".", ".."} for part in relative.parts):
        return None
    return relative.as_posix() if relative.parts else ""


def read_source_snapshot_manifest(path: Path) -> SourceSnapshotManifest:
    """Read one bounded canonical source manifest without following a symlink."""
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise SourcePreflightError(
            "cannot open source snapshot manifest",
            PreflightRuleCode.MANIFEST_INVALID,
        ) from error
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode) or info.st_size > 8 * 1024 * 1024:
            _raise(
                PreflightRuleCode.MANIFEST_INVALID,
                "source snapshot manifest must be a bounded regular file",
            )
        content = b""
        while len(content) <= info.st_size:
            chunk = os.read(descriptor, min(1024 * 1024, info.st_size + 1))
            if not chunk:
                break
            content += chunk
    finally:
        os.close(descriptor)
    try:
        mapping = read_json_bytes_object(content, source=str(path))
    except MetadataValidationError as error:
        raise SourcePreflightError(
            "source snapshot manifest is not strict JSON",
            PreflightRuleCode.MANIFEST_INVALID,
        ) from error
    if canonical_json_bytes(mapping) != content:
        _raise(
            PreflightRuleCode.MANIFEST_INVALID,
            "source snapshot manifest must use canonical JSON",
        )
    return SourceSnapshotManifest.from_mapping(mapping)


def verify_source_snapshot_directory(
    root: Path, manifest: SourceSnapshotManifest
) -> None:
    """Require an extracted tree to equal the committed regular-file manifest."""
    try:
        root_info = os.lstat(root)
    except OSError as error:
        raise SourcePreflightError(
            "source snapshot directory does not exist",
            PreflightRuleCode.MANIFEST_REFERENCE_MISSING,
        ) from error
    if not stat.S_ISDIR(root_info.st_mode) or stat.S_ISLNK(root_info.st_mode):
        _raise(
            PreflightRuleCode.UNSAFE_TREE_ENTRY,
            "source snapshot root must be a real directory",
        )
    expected = {entry.path: entry for entry in manifest.files}
    expected_directories: set[str] = set()
    for relative in expected:
        parent = PurePosixPath(relative).parent
        while parent.as_posix() != ".":
            expected_directories.add(parent.as_posix())
            parent = parent.parent
    actual_files: dict[str, Path] = {}
    actual_directories: set[str] = set()
    for current, directory_names, file_names in os.walk(root, followlinks=False):
        current_path = Path(current)
        for name in directory_names:
            candidate = current_path / name
            info = os.lstat(candidate)
            if not stat.S_ISDIR(info.st_mode) or stat.S_ISLNK(info.st_mode):
                _raise(
                    PreflightRuleCode.UNSAFE_TREE_ENTRY,
                    "source snapshot contains a non-directory tree component",
                )
            actual_directories.add(candidate.relative_to(root).as_posix())
        for name in file_names:
            candidate = current_path / name
            relative = candidate.relative_to(root).as_posix()
            info = os.lstat(candidate)
            if not stat.S_ISREG(info.st_mode) or stat.S_ISLNK(info.st_mode):
                _raise(
                    PreflightRuleCode.UNSAFE_TREE_ENTRY,
                    "source snapshot contains a special or symbolic entry",
                )
            if not _safe_relative_path(relative) or relative in actual_files:
                _raise(
                    PreflightRuleCode.UNSAFE_TREE_ENTRY,
                    "source snapshot contains an unsafe or duplicate path",
                )
            actual_files[relative] = candidate
    if set(actual_files) != set(expected) or actual_directories != expected_directories:
        _raise(
            PreflightRuleCode.MANIFEST_REFERENCE_MISMATCH,
            "source snapshot path set differs from the committed manifest",
        )
    for relative, entry in expected.items():
        path = actual_files[relative]
        info = os.lstat(path)
        executable = bool(info.st_mode & 0o111)
        if (
            info.st_size != entry.size_bytes
            or sha256_file(path) != entry.sha256
            or executable != (entry.mode == "100755")
            or bool(info.st_mode & 0o7000)
        ):
            _raise(
                PreflightRuleCode.MANIFEST_REFERENCE_MISMATCH,
                "source snapshot bytes or mode differ from the committed manifest",
            )


def _make_snapshot_directories_removable(target: Path) -> None:
    """Restore owner access without following links before checked removal."""

    def raise_walk_error(error: OSError) -> NoReturn:
        raise error

    for current, _, _ in os.walk(
        target,
        topdown=True,
        onerror=raise_walk_error,
        followlinks=False,
    ):
        current_path = Path(current)
        info = os.lstat(current_path)
        if not stat.S_ISDIR(info.st_mode) or stat.S_ISLNK(info.st_mode):
            raise OSError("partial snapshot contains a non-directory tree component")
        os.chmod(current_path, 0o700, follow_symlinks=False)


def _cleanup_partial_source_snapshot(
    target: Path,
    primary_error: SourcePreflightError,
) -> None:
    """Remove a failed extraction and prove that no destination remains."""
    cleanup_error: Exception | None = None
    try:
        _make_snapshot_directories_removable(target)
        shutil.rmtree(target)
    except Exception as error:
        cleanup_error = error

    try:
        os.lstat(target)
    except FileNotFoundError:
        return
    except OSError as error:
        cleanup_error = error
    else:
        if cleanup_error is None:
            cleanup_error = OSError(
                "partial snapshot destination still exists after cleanup returned"
            )

    assert cleanup_error is not None
    raise SourcePreflightError(
        "source snapshot extraction failed and partial-output cleanup failed; "
        f"destination={target}; "
        f"primary_rule={primary_error.rule_code.value}; "
        f"primary_error={primary_error}; cleanup_error={cleanup_error}",
        PreflightRuleCode.PARTIAL_OUTPUT_CLEANUP_FAILED,
    ) from primary_error


def extract_source_snapshot(
    archive_path: Path,
    destination: Path,
    receipt: SourceArchiveReceipt,
) -> Path:
    """Safely extract only committed regular files and reverify the full tree."""
    try:
        archive_info = os.lstat(archive_path)
    except OSError as error:
        raise SourcePreflightError(
            "source archive does not exist",
            PreflightRuleCode.MANIFEST_REFERENCE_MISSING,
        ) from error
    if (
        not stat.S_ISREG(archive_info.st_mode)
        or stat.S_ISLNK(archive_info.st_mode)
        or archive_info.st_size != receipt.archive_size_bytes
        or sha256_file(archive_path) != receipt.archive_sha256
    ):
        _raise(
            PreflightRuleCode.MANIFEST_REFERENCE_MISMATCH,
            "source archive differs from its export receipt",
        )
    target = destination.expanduser().absolute()
    if target.exists() or target.is_symlink() or not target.parent.is_dir():
        _raise(
            PreflightRuleCode.OUTPUT_PATH_INVALID,
            "snapshot destination must be absent with an existing parent",
        )
    target = target.parent.resolve() / target.name
    expected = {entry.path: entry for entry in receipt.source_manifest.files}
    seen_files: set[str] = set()
    seen_directories: set[str] = set()
    target.mkdir(mode=0o700)
    try:
        with tarfile.open(archive_path, mode="r:") as archive:
            for member in archive:
                relative = _archive_relative(member.name)
                if relative is None:
                    _raise(
                        PreflightRuleCode.UNSAFE_TREE_ENTRY,
                        "source archive contains an unsafe path",
                    )
                if member.isdir():
                    if relative:
                        if relative in seen_directories:
                            _raise(
                                PreflightRuleCode.UNSAFE_TREE_ENTRY,
                                "source archive contains a duplicate directory",
                            )
                        seen_directories.add(relative)
                        (target / relative).mkdir(parents=True, exist_ok=True)
                    continue
                if not member.isfile() or not relative or relative in seen_files:
                    _raise(
                        PreflightRuleCode.UNSAFE_TREE_ENTRY,
                        "source archive contains a special, linked, or duplicate entry",
                    )
                expected_entry = expected.get(relative)
                if expected_entry is None or member.size != expected_entry.size_bytes:
                    _raise(
                        PreflightRuleCode.MANIFEST_REFERENCE_MISMATCH,
                        "source archive entry differs from the committed manifest",
                    )
                stream = archive.extractfile(member)
                if stream is None:
                    _raise(
                        PreflightRuleCode.MANIFEST_REFERENCE_MISSING,
                        "cannot read source archive entry",
                    )
                content = stream.read(expected_entry.size_bytes + 1)
                if (
                    len(content) != expected_entry.size_bytes
                    or sha256_bytes(content) != expected_entry.sha256
                ):
                    _raise(
                        PreflightRuleCode.MANIFEST_REFERENCE_MISMATCH,
                        "source archive bytes differ from the committed manifest",
                    )
                output = target / relative
                output.parent.mkdir(parents=True, exist_ok=True)
                flags = (
                    os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
                )
                descriptor = os.open(output, flags, 0o700)
                try:
                    written = 0
                    while written < len(content):
                        count = os.write(descriptor, content[written:])
                        if count <= 0:
                            _raise(
                                PreflightRuleCode.GIT_COMMAND_FAILED,
                                "short write while extracting source snapshot",
                            )
                        written += count
                    os.fsync(descriptor)
                finally:
                    os.close(descriptor)
                output.chmod(0o755 if expected_entry.mode == "100755" else 0o644)
                seen_files.add(relative)
        if seen_files != set(expected):
            _raise(
                PreflightRuleCode.MANIFEST_REFERENCE_MISSING,
                "source archive is missing committed files",
            )
        verify_source_snapshot_directory(target, receipt.source_manifest)
        for path in sorted(
            (item for item in target.rglob("*") if item.is_file()),
            key=lambda item: len(item.parts),
            reverse=True,
        ):
            expected_entry = expected[path.relative_to(target).as_posix()]
            path.chmod(0o555 if expected_entry.mode == "100755" else 0o444)
        for path in sorted(
            (item for item in target.rglob("*") if item.is_dir()),
            key=lambda item: len(item.parts),
            reverse=True,
        ):
            path.chmod(0o555)
        target.chmod(0o555)
        verify_source_snapshot_directory(target, receipt.source_manifest)
        return target
    except SourcePreflightError as error:
        _cleanup_partial_source_snapshot(target, error)
        raise
    except (OSError, tarfile.TarError) as error:
        try:
            raise SourcePreflightError(
                f"cannot safely extract source snapshot: {error}",
                PreflightRuleCode.UNSAFE_TREE_ENTRY,
            ) from error
        except SourcePreflightError as primary_error:
            _cleanup_partial_source_snapshot(target, primary_error)
            raise


def _emit(value: Mapping[str, object], *, error: bool = False) -> None:
    stream = sys.stderr.buffer if error else sys.stdout.buffer
    stream.write(canonical_json_bytes(value) + b"\n")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m policytwin.release_preflight",
        description="Verify and export an exact committed PolicyTwin source snapshot.",
    )
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--archive", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the machine-readable source preflight or export command."""
    args = _parser().parse_args(argv)
    try:
        if args.archive is None:
            report = run_source_preflight(args.project_root)
            _emit(report.to_mapping(), error=not report.source_snapshot_ready)
            return 0 if report.source_snapshot_ready else 1
        receipt = export_source_snapshot(args.project_root, args.archive)
        _emit(receipt.to_mapping())
        return 0
    except SourcePreflightError as error:
        _emit(
            {
                "error": str(error),
                "rule_code": error.rule_code.value,
                "schema_version": "policytwin.source-preflight-error.v1",
                "status": "error",
            },
            error=True,
        )
        return 2


if __name__ == "__main__":  # pragma: no cover - exercised through the API.
    raise SystemExit(main())


__all__ = [
    "PreflightBlocker",
    "PreflightRuleCode",
    "SOURCE_ARCHIVE_SCHEMA_VERSION",
    "SOURCE_MANIFEST_SCHEMA_VERSION",
    "SourceArchiveReceipt",
    "SourceFileDigest",
    "SourcePreflightConfig",
    "SourcePreflightError",
    "SourcePreflightReport",
    "SourceSnapshotManifest",
    "extract_source_snapshot",
    "export_source_snapshot",
    "main",
    "read_source_snapshot_manifest",
    "run_source_preflight",
    "verify_source_snapshot_directory",
]
