"""Strict command-line boundary for an externally committed AAMAS study.

The CLI deliberately accepts plans, study commitments, and caller-supplied
raw digests as separate files.  It never derives an external commitment from a
bundle under audit, and it emits only bounded machine-readable status data.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import stat
import sys
import tempfile
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import NoReturn, cast

from policytwin.evaluation.os_isolation import (
    DEFAULT_RECEIPT_NAME,
    IsolationBackend,
    IsolationError,
    IsolationInputMode,
    IsolationReceipt,
    audit_isolation_receipt,
)
from policytwin.experiments.aamas_publication import (
    AAMASPublicationAuditReport,
    AAMASPublicationError,
    AAMASPublicationIntegrityError,
    AAMASPublicationPlan,
    AAMASPublicationStudyCommitment,
    audit_aamas_publication,
    run_aamas_publication,
    write_aamas_publication,
)
from policytwin.provenance import (
    MetadataValidationError,
    canonical_json_bytes,
    read_json_bytes_object,
    sha256_bytes,
)

_SHA256_LINE = re.compile(rb"[0-9a-f]{64}\n")
_RELEASE_SCHEMA_VERSION = "policytwin.aamas-synthetic-release.v2"
_RELEASE_MANIFEST_SCHEMA_VERSION = "policytwin.aamas-synthetic-release-manifest.v2"
_RELEASE_SEAL_SCHEMA_VERSION = "policytwin.aamas-synthetic-release-seal.v2"
_RELEASE_ROOT_FILES = frozenset(
    {
        "command-manifest.json",
        "command-manifest.sha256",
        "manifest.json",
        "release.json",
        "seal.json",
    }
)
_RELEASE_ROOT_DIRECTORIES = frozenset({"isolated-job", "source-snapshot"})
_ISOLATED_JOB_OUTPUT_MEMBER = "publication"
_POST_SCAN_RELEASE_FILES = frozenset({"manifest.json", "release.json", "seal.json"})
_SECRET_SCAN_SCHEMA_VERSION = "policytwin.high-confidence-secret-scan.v1"
_HIGH_CONFIDENCE_SECRET_PATTERNS = (
    re.compile(rb"(?<![A-Za-z0-9])sk-[A-Za-z0-9_-]{16,}"),
    re.compile(rb"(?<![A-Za-z0-9])AKIA[0-9A-Z]{16}(?![0-9A-Z])"),
    re.compile(rb"(?<![A-Za-z0-9])gh[pousr]_[A-Za-z0-9]{30,}"),
    re.compile(rb"(?<![A-Za-z0-9])xox[baprs]-[A-Za-z0-9-]{20,}"),
)
_SENSITIVE_ARG = re.compile(
    r"(?i)(?:sk-[a-z0-9_-]{12,}|api[_-]?key|access[_-]?token|"
    r"authorization|bearer|password|private[_-]?key)"
)
_EXPECTED_COMPLETE_OUTPUT_STATUSES = (
    ("core_scientific_result", "complete"),
    ("integrated_branch_artifact", "complete"),
    ("misspecification_negative_control", "complete_detected"),
    ("core_frozen_reporting", "complete"),
    ("integrated_branch_frozen_reporting", "complete"),
    ("root_validity_certificate", "abstain_missing_external_authority"),
)
_MAX_STABLE_FILE_BYTES = 512 * 1024 * 1024


class AAMASFormalCLIError(RuntimeError):
    """A bounded CLI-input failure whose message is safe to serialize."""

    def __init__(self, error_code: str) -> None:
        super().__init__(error_code)
        self.error_code = error_code


class _MachineArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> NoReturn:
        del message
        raise AAMASFormalCLIError("invalid_cli_arguments")


def _emit(value: Mapping[str, object], *, error: bool = False) -> None:
    payload = json.dumps(
        dict(value),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    stream = sys.stderr if error else sys.stdout
    stream.write(payload + "\n")


def _stat_signature(item: os.stat_result) -> tuple[int, ...]:
    return (
        item.st_dev,
        item.st_ino,
        item.st_mode,
        item.st_nlink,
        item.st_uid,
        item.st_gid,
        item.st_size,
        item.st_mtime_ns,
        item.st_ctime_ns,
    )


def _directory_stat_signature(item: os.stat_result) -> tuple[int, ...]:
    """Bind a directory inode without rejecting unrelated entry churn."""

    return (
        item.st_dev,
        item.st_ino,
        item.st_mode,
        item.st_uid,
        item.st_gid,
    )


def _open_rooted_directory_chain(
    directory: Path,
    *,
    error_code: str,
) -> tuple[list[int], tuple[tuple[int, ...], ...]]:
    """Open an absolute directory from its filesystem root without following links."""

    nofollow = getattr(os, "O_NOFOLLOW", None)
    if nofollow is None:
        raise AAMASFormalCLIError(error_code)
    absolute = Path(os.path.abspath(directory.expanduser()))
    if not absolute.is_absolute() or not absolute.anchor:
        raise AAMASFormalCLIError(error_code)
    flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_CLOEXEC", 0)
        | nofollow
    )
    descriptors: list[int] = []
    signatures: list[tuple[int, ...]] = []
    try:
        root_descriptor = os.open(absolute.anchor, flags)
        descriptors.append(root_descriptor)
        root_info = os.fstat(root_descriptor)
        if not stat.S_ISDIR(root_info.st_mode):
            raise AAMASFormalCLIError(error_code)
        signatures.append(_directory_stat_signature(root_info))
        current = root_descriptor
        for part in absolute.parts[1:]:
            before = os.stat(part, dir_fd=current, follow_symlinks=False)
            if stat.S_ISLNK(before.st_mode) or not stat.S_ISDIR(before.st_mode):
                raise AAMASFormalCLIError(error_code)
            child = os.open(part, flags, dir_fd=current)
            descriptors.append(child)
            opened = os.fstat(child)
            if _directory_stat_signature(opened) != _directory_stat_signature(before):
                raise AAMASFormalCLIError(error_code)
            signatures.append(_directory_stat_signature(opened))
            current = child
        return descriptors, tuple(signatures)
    except AAMASFormalCLIError:
        for descriptor in reversed(descriptors):
            os.close(descriptor)
        raise
    except OSError as error:
        for descriptor in reversed(descriptors):
            os.close(descriptor)
        raise AAMASFormalCLIError(error_code) from error


def _read_exact_descriptor(
    descriptor: int,
    *,
    expected_size: int,
    error_code: str,
) -> bytes:
    if not 0 <= expected_size <= _MAX_STABLE_FILE_BYTES:
        raise AAMASFormalCLIError(error_code)
    chunks: list[bytes] = []
    observed = 0
    try:
        while observed < expected_size:
            chunk = os.read(descriptor, min(1024 * 1024, expected_size - observed))
            if not chunk:
                break
            chunks.append(chunk)
            observed += len(chunk)
        if observed != expected_size or os.read(descriptor, 1):
            raise AAMASFormalCLIError(error_code)
    except AAMASFormalCLIError:
        raise
    except OSError as error:
        raise AAMASFormalCLIError(error_code) from error
    return b"".join(chunks)


def _stable_regular_bytes(path: Path, *, error_code: str) -> bytes:
    """Read through held root-anchored FDs and reject path or content drift."""

    absolute = Path(os.path.abspath(path.expanduser()))
    if not absolute.name:
        raise AAMASFormalCLIError(error_code)
    descriptors, directory_signatures = _open_rooted_directory_chain(
        absolute.parent,
        error_code=error_code,
    )
    file_descriptor: int | None = None
    try:
        parent_descriptor = descriptors[-1]
        before = os.stat(
            absolute.name,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
        if (
            stat.S_ISLNK(before.st_mode)
            or not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
        ):
            raise AAMASFormalCLIError(error_code)
        file_descriptor = os.open(
            absolute.name,
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NONBLOCK", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=parent_descriptor,
        )
        opened = os.fstat(file_descriptor)
        if _stat_signature(opened) != _stat_signature(before):
            raise AAMASFormalCLIError(error_code)
        payload = _read_exact_descriptor(
            file_descriptor,
            expected_size=before.st_size,
            error_code=error_code,
        )
        after_open = os.fstat(file_descriptor)
        after_path = os.stat(
            absolute.name,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
        expected = _stat_signature(before)
        if (
            _stat_signature(after_open) != expected
            or _stat_signature(after_path) != expected
            or tuple(_directory_stat_signature(os.fstat(item)) for item in descriptors)
            != directory_signatures
        ):
            raise AAMASFormalCLIError(error_code)
    except AAMASFormalCLIError:
        raise
    except OSError as error:
        raise AAMASFormalCLIError(error_code) from error
    finally:
        if file_descriptor is not None:
            os.close(file_descriptor)
        for descriptor in reversed(descriptors):
            os.close(descriptor)

    second_descriptors, second_signatures = _open_rooted_directory_chain(
        absolute.parent,
        error_code=error_code,
    )
    try:
        second_path = os.stat(
            absolute.name,
            dir_fd=second_descriptors[-1],
            follow_symlinks=False,
        )
        if second_signatures != directory_signatures or _stat_signature(
            second_path
        ) != _stat_signature(before):
            raise AAMASFormalCLIError(error_code)
    except AAMASFormalCLIError:
        raise
    except OSError as error:
        raise AAMASFormalCLIError(error_code) from error
    finally:
        for descriptor in reversed(second_descriptors):
            os.close(descriptor)
    return payload


def _read_sha256_file(path: Path, *, error_code: str) -> str:
    payload = _stable_regular_bytes(path, error_code=error_code)
    if _SHA256_LINE.fullmatch(payload) is None:
        raise AAMASFormalCLIError(error_code)
    return payload[:-1].decode("ascii")


def _read_plan(path: Path) -> tuple[AAMASPublicationPlan, bytes]:
    payload = _stable_regular_bytes(path, error_code="invalid_plan_file")
    try:
        mapping = read_json_bytes_object(payload, source="external publication plan")
        plan = AAMASPublicationPlan.from_mapping(mapping)
    except (AAMASPublicationError, MetadataValidationError) as error:
        raise AAMASFormalCLIError("invalid_plan_file") from error
    if payload != canonical_json_bytes(plan.to_mapping()):
        raise AAMASFormalCLIError("noncanonical_plan_file")
    return plan, payload


def _read_commitment(
    path: Path,
) -> tuple[AAMASPublicationStudyCommitment, bytes]:
    payload = _stable_regular_bytes(path, error_code="invalid_commitment_file")
    try:
        mapping = read_json_bytes_object(
            payload,
            source="external publication study commitment",
        )
        commitment = AAMASPublicationStudyCommitment.from_mapping(mapping)
    except (AAMASPublicationError, MetadataValidationError) as error:
        raise AAMASFormalCLIError("invalid_commitment_file") from error
    if payload != canonical_json_bytes(commitment.to_mapping()):
        raise AAMASFormalCLIError("noncanonical_commitment_file")
    return commitment, payload


def _load_external_inputs(
    *,
    plan_path: Path,
    plan_sha256_path: Path,
    commitment_path: Path,
    commitment_sha256_path: Path,
) -> tuple[AAMASPublicationPlan, str, AAMASPublicationStudyCommitment, str]:
    plan, plan_payload = _read_plan(plan_path)
    plan_sha256 = _read_sha256_file(
        plan_sha256_path,
        error_code="invalid_plan_sha256_file",
    )
    if plan_sha256 != plan.sha256 or plan_sha256 != sha256_bytes(plan_payload):
        raise AAMASFormalCLIError("plan_sha256_mismatch")

    commitment, commitment_payload = _read_commitment(commitment_path)
    commitment_sha256 = _read_sha256_file(
        commitment_sha256_path,
        error_code="invalid_commitment_sha256_file",
    )
    if commitment_sha256 != commitment.sha256 or commitment_sha256 != sha256_bytes(
        commitment_payload
    ):
        raise AAMASFormalCLIError("commitment_sha256_mismatch")
    if commitment.plan_sha256 != plan_sha256:
        raise AAMASFormalCLIError("commitment_plan_mismatch")
    return plan, plan_sha256, commitment, commitment_sha256


def _synthetic_output_gate(report: AAMASPublicationAuditReport) -> bool:
    """Accept complete inner outputs without claiming OS-isolated release status."""

    return (
        report.integrity_valid
        and report.submission_evidence_complete
        and report.output_statuses == _EXPECTED_COMPLETE_OUTPUT_STATUSES
        and not report.failed_outputs
        and not report.blocked_outputs
        and report.abstained_outputs == ("root_validity_certificate",)
    )


def _audit_payload(
    report: AAMASPublicationAuditReport,
    *,
    command: str,
) -> dict[str, object]:
    outputs_complete = _synthetic_output_gate(report)
    return {
        "abstained_outputs": list(report.abstained_outputs),
        "blocked_outputs": list(report.blocked_outputs),
        "command": command,
        "os_isolation_attested": False,
        "failed_outputs": list(report.failed_outputs),
        "incomplete_outputs": list(report.incomplete_outputs),
        "integrity_valid": report.integrity_valid,
        "output_statuses": [
            {"output_id": output_id, "status": status}
            for output_id, status in report.output_statuses
        ],
        "plan_sha256": report.plan_sha256,
        "publication_eligible": report.publication_eligible,
        "release_gate_passed": False,
        "root_seal_sha256": report.root_seal_sha256,
        "schema_version": 1,
        "scientific_result_sha256": report.scientific_result_sha256,
        "scientifically_complete": report.scientifically_complete,
        "status": (
            "synthetic_outputs_complete_unfinalized"
            if outputs_complete
            else "synthetic_outputs_incomplete"
        ),
        "study_commitment_sha256": report.study_commitment_sha256,
        "submission_evidence_complete": report.submission_evidence_complete,
        "synthetic_only": True,
        "synthetic_outputs_complete": outputs_complete,
        "validity_level": report.validity_level.value,
    }


def _write_new(path: Path, payload: bytes) -> None:
    descriptor: int | None = None
    try:
        descriptor = os.open(
            path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0),
            0o600,
        )
        remaining = memoryview(payload)
        while remaining:
            written = os.write(descriptor, remaining)
            if written <= 0:
                raise AAMASFormalCLIError("release_short_write")
            remaining = remaining[written:]
        os.fsync(descriptor)
    except AAMASFormalCLIError:
        raise
    except OSError as error:
        raise AAMASFormalCLIError("release_write_failed") from error
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _directory_identity(details: os.stat_result) -> tuple[int, int]:
    return details.st_dev, details.st_ino


def _quarantine_owned_directory(
    path: Path,
    *,
    expected_identity: tuple[int, int],
) -> bool:
    """Atomically detach an owned directory without deleting a replacement.

    The quarantined tree is deliberately retained.  Path-based recursive
    deletion after an identity check has an unavoidable check/use race: a
    different directory can be substituted before ``rmtree`` opens it.  A
    descriptor-relative rename followed by an identity check instead ensures
    that an unexpected replacement is preserved, even if it wins that race.
    """

    parent_descriptor: int | None = None
    try:
        parent_descriptor = os.open(
            path.parent,
            os.O_RDONLY
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
        )
        current = os.stat(
            path.name,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
        if (
            not stat.S_ISDIR(current.st_mode)
            or _directory_identity(current) != expected_identity
        ):
            return False

        for _ in range(8):
            quarantine_name = f".{path.name}.quarantine-{uuid.uuid4().hex}"
            try:
                # Reserving the destination prevents rename from overwriting
                # any pre-existing user entry with the same name.
                os.mkdir(quarantine_name, mode=0o700, dir_fd=parent_descriptor)
            except FileExistsError:
                continue
            try:
                os.rename(
                    path.name,
                    quarantine_name,
                    src_dir_fd=parent_descriptor,
                    dst_dir_fd=parent_descriptor,
                )
            except OSError:
                # Keep the reserved directory as a harmless tombstone.  It is
                # safer than another check-then-delete sequence during error
                # handling.
                return False
            moved = os.stat(
                quarantine_name,
                dir_fd=parent_descriptor,
                follow_symlinks=False,
            )
            return (
                stat.S_ISDIR(moved.st_mode)
                and _directory_identity(moved) == expected_identity
            )
    except OSError:
        return False
    finally:
        if parent_descriptor is not None:
            os.close(parent_descriptor)
    return False


def _strict_tree_rows(root: Path) -> tuple[dict[str, object], ...]:
    if not root.is_dir() or root.is_symlink():
        raise AAMASFormalCLIError("invalid_release_tree")
    rows: list[dict[str, object]] = []
    for path in sorted(root.rglob("*"), key=lambda item: item.as_posix()):
        try:
            details = path.lstat()
        except OSError as error:
            raise AAMASFormalCLIError("invalid_release_tree") from error
        if stat.S_ISLNK(details.st_mode):
            raise AAMASFormalCLIError("invalid_release_tree")
        if stat.S_ISDIR(details.st_mode):
            continue
        if not stat.S_ISREG(details.st_mode) or details.st_nlink != 1:
            raise AAMASFormalCLIError("invalid_release_tree")
        relative = path.relative_to(root).as_posix()
        payload = _stable_regular_bytes(path, error_code="invalid_release_tree")
        rows.append(
            {
                "path": relative,
                "sha256": sha256_bytes(payload),
                "size_bytes": len(payload),
            }
        )
    return tuple(rows)


def _strict_directories(root: Path) -> frozenset[str]:
    if not root.is_dir() or root.is_symlink():
        raise AAMASFormalCLIError("invalid_release_tree")
    directories: set[str] = set()
    for path in root.rglob("*"):
        try:
            details = path.lstat()
        except OSError as error:
            raise AAMASFormalCLIError("invalid_release_tree") from error
        if stat.S_ISLNK(details.st_mode):
            raise AAMASFormalCLIError("invalid_release_tree")
        if stat.S_ISDIR(details.st_mode):
            directories.add(path.relative_to(root).as_posix())
    return frozenset(directories)


def _scan_high_confidence_secrets(
    root: Path,
    *,
    identity_exclusions: frozenset[str] = frozenset(),
) -> dict[str, object]:
    """Fail closed on a small, versioned set of high-confidence token forms."""

    rows = _strict_tree_rows(root)
    for row in rows:
        relative = cast(str, row["path"])
        payload = _stable_regular_bytes(
            root / relative,
            error_code="secret_scan_input_changed",
        )
        if any(
            pattern.search(payload) is not None
            for pattern in _HIGH_CONFIDENCE_SECRET_PATTERNS
        ):
            raise AAMASFormalCLIError("high_confidence_secret_detected")
    if _strict_tree_rows(root) != rows:
        raise AAMASFormalCLIError("secret_scan_input_changed")
    identity_rows = tuple(row for row in rows if row["path"] not in identity_exclusions)
    return {
        "file_count": len(identity_rows),
        "passed": True,
        "schema_version": _SECRET_SCAN_SCHEMA_VERSION,
        "scope": "pre_seal_release_payload_tree",
        "tree_sha256": sha256_bytes(canonical_json_bytes(identity_rows)),
    }


def _copy_strict_tree(source: Path, destination: Path) -> None:
    source_rows = _strict_tree_rows(source)
    source_directories = _strict_directories(source)
    destination.mkdir(parents=False, exist_ok=False)
    for relative in sorted(
        source_directories, key=lambda item: (item.count("/"), item)
    ):
        (destination / relative).mkdir(exist_ok=False)
    for row in source_rows:
        relative = cast(str, row["path"])
        payload = _stable_regular_bytes(
            source / relative,
            error_code="source_changed_during_release_copy",
        )
        if len(payload) != row["size_bytes"] or sha256_bytes(payload) != row["sha256"]:
            raise AAMASFormalCLIError("source_changed_during_release_copy")
        _write_new(destination / relative, payload)
    if (
        _strict_tree_rows(source) != source_rows
        or _strict_directories(source) != source_directories
        or _strict_tree_rows(destination) != source_rows
        or _strict_directories(destination) != source_directories
    ):
        raise AAMASFormalCLIError("source_changed_during_release_copy")


def _source_manifest_rows(
    manifest: Mapping[str, object],
) -> tuple[dict[str, object], ...]:
    expected_fields = frozenset(
        {
            "authority",
            "complete_for_scope",
            "files",
            "schema_version",
            "scope",
            "source_count",
            "tree_sha256",
            "uv_lock_sha256",
        }
    )
    if (
        frozenset(manifest) != expected_fields
        or manifest.get("authority") != "local_worktree_bytes_not_external_attestation"
        or manifest.get("complete_for_scope") is not True
        or manifest.get("schema_version") != 1
        or manifest.get("scope")
        != "AGENTS_pyproject_uv_lock_and_all_regular_src_policytwin_files"
    ):
        raise AAMASFormalCLIError("invalid_publication_source_manifest")
    raw_rows = manifest.get("files")
    if not isinstance(raw_rows, list):
        raise AAMASFormalCLIError("invalid_publication_source_manifest")
    rows: list[dict[str, object]] = []
    previous: str | None = None
    for raw in raw_rows:
        if not isinstance(raw, Mapping) or frozenset(raw) != {
            "path",
            "sha256",
            "size_bytes",
        }:
            raise AAMASFormalCLIError("invalid_publication_source_manifest")
        relative = raw.get("path")
        digest = raw.get("sha256")
        size = raw.get("size_bytes")
        if (
            not isinstance(relative, str)
            or not relative
            or relative.startswith("/")
            or "\\" in relative
            or any(ord(character) < 32 for character in relative)
            or any(part in {"", ".", ".."} for part in relative.split("/"))
            or (
                relative not in {"AGENTS.md", "pyproject.toml", "uv.lock"}
                and not relative.startswith("src/policytwin/")
            )
            or not isinstance(digest, str)
            or re.fullmatch(r"[0-9a-f]{64}", digest) is None
            or isinstance(size, bool)
            or not isinstance(size, int)
            or size < 0
            or (previous is not None and relative <= previous)
        ):
            raise AAMASFormalCLIError("invalid_publication_source_manifest")
        previous = relative
        rows.append({"path": relative, "sha256": digest, "size_bytes": size})
    result = tuple(rows)
    if (
        manifest.get("source_count") != len(result)
        or manifest.get("tree_sha256") != sha256_bytes(canonical_json_bytes(result))
        or not isinstance(manifest.get("uv_lock_sha256"), str)
        or re.fullmatch(r"[0-9a-f]{64}", cast(str, manifest["uv_lock_sha256"])) is None
    ):
        raise AAMASFormalCLIError("invalid_publication_source_manifest")
    return result


def _expected_source_directories(
    rows: Sequence[Mapping[str, object]],
) -> frozenset[str]:
    directories: set[str] = set()
    for row in rows:
        parent = Path(cast(str, row["path"])).parent
        while parent != Path("."):
            directories.add(parent.as_posix())
            parent = parent.parent
    return frozenset(directories)


def _verify_source_snapshot(
    snapshot: Path,
    manifest: Mapping[str, object],
    commitment: AAMASPublicationStudyCommitment,
) -> tuple[tuple[dict[str, object], ...], str]:
    rows = _source_manifest_rows(manifest)
    tree_sha256 = cast(str, manifest["tree_sha256"])
    if (
        tree_sha256 != commitment.source_tree_sha256
        or manifest["uv_lock_sha256"] != commitment.uv_lock_sha256
        or _strict_tree_rows(snapshot) != rows
        or _strict_directories(snapshot) != _expected_source_directories(rows)
    ):
        raise AAMASFormalCLIError("embedded_source_snapshot_mismatch")
    return rows, tree_sha256


def _copy_source_snapshot(
    *,
    project_root: Path,
    publication: Path,
    destination: Path,
    commitment: AAMASPublicationStudyCommitment,
) -> tuple[int, str]:
    manifest_payload = _stable_regular_bytes(
        publication / "source-manifest.json",
        error_code="invalid_publication_source_manifest",
    )
    try:
        manifest = read_json_bytes_object(
            manifest_payload,
            source="publication source manifest",
        )
    except MetadataValidationError as error:
        raise AAMASFormalCLIError("invalid_publication_source_manifest") from error
    rows = _source_manifest_rows(manifest)
    required_directories = _expected_source_directories(rows)
    try:
        project_details = project_root.lstat()
        if not stat.S_ISDIR(project_details.st_mode) or stat.S_ISLNK(
            project_details.st_mode
        ):
            raise AAMASFormalCLIError("source_snapshot_directory_link")
        for relative in required_directories:
            details = (project_root / relative).lstat()
            if not stat.S_ISDIR(details.st_mode) or stat.S_ISLNK(details.st_mode):
                raise AAMASFormalCLIError("source_snapshot_directory_link")
    except AAMASFormalCLIError:
        raise
    except OSError as error:
        raise AAMASFormalCLIError("source_snapshot_directory_link") from error
    destination.mkdir(parents=False, exist_ok=False)
    for relative in sorted(
        required_directories,
        key=lambda item: (item.count("/"), item),
    ):
        (destination / relative).mkdir(exist_ok=False)
    for row in rows:
        relative = cast(str, row["path"])
        payload = _stable_regular_bytes(
            project_root / relative,
            error_code="source_snapshot_input_mismatch",
        )
        if len(payload) != row["size_bytes"] or sha256_bytes(payload) != row["sha256"]:
            raise AAMASFormalCLIError("source_snapshot_input_mismatch")
        _write_new(destination / relative, payload)
    for relative in required_directories:
        details = (project_root / relative).lstat()
        if not stat.S_ISDIR(details.st_mode) or stat.S_ISLNK(details.st_mode):
            raise AAMASFormalCLIError("source_snapshot_directory_link")
    verified, tree_sha256 = _verify_source_snapshot(
        destination,
        manifest,
        commitment,
    )
    return len(verified), tree_sha256


def _read_command_manifest(path: Path) -> tuple[tuple[str, ...], bytes, str]:
    payload = _stable_regular_bytes(path, error_code="invalid_command_manifest")
    try:
        raw = json.loads(payload.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError, ValueError) as error:
        raise AAMASFormalCLIError("invalid_command_manifest") from error
    if (
        not isinstance(raw, list)
        or not raw
        or not all(
            isinstance(item, str) and item and len(item) <= 4096 and "\x00" not in item
            for item in raw
        )
    ):
        raise AAMASFormalCLIError("invalid_command_manifest")
    argv = tuple(cast(list[str], raw))
    if payload != canonical_json_bytes(list(argv)):
        raise AAMASFormalCLIError("noncanonical_command_manifest")
    if any(_SENSITIVE_ARG.search(item) is not None for item in argv):
        raise AAMASFormalCLIError("sensitive_command_manifest")
    return argv, payload, sha256_bytes(payload)


def _command_paths(argv: Sequence[str]) -> dict[str, Path]:
    if (
        len(argv) != 17
        or argv[1:4] != ("-m", "policytwin.aamas_formal", "run")
        or argv[4] != "--plan"
        or argv[6] != "--plan-sha256-file"
        or argv[8] != "--study-commitment"
        or argv[10] != "--study-commitment-sha256-file"
        or argv[12] != "--project-root"
        or argv[14] != "--allow-formal-scale"
        or argv[15] != "--output"
    ):
        raise AAMASFormalCLIError("command_manifest_not_aamas_run")
    executable = Path(argv[0])
    paths = {
        "executable": executable,
        "plan": Path(argv[5]),
        "plan_sha256": Path(argv[7]),
        "study_commitment": Path(argv[9]),
        "study_commitment_sha256": Path(argv[11]),
        "project_root": Path(argv[13]),
        "output": Path(argv[16]),
    }
    if not all(path.is_absolute() for path in paths.values()):
        raise AAMASFormalCLIError("command_manifest_paths_not_absolute")
    return paths


def _validate_run_command(
    argv: Sequence[str],
    *,
    bind_current_executable: bool = False,
    plan_path: Path | None = None,
    plan_sha256_path: Path | None = None,
    commitment_path: Path | None = None,
    commitment_sha256_path: Path | None = None,
    project_root: Path | None = None,
    output: Path | None = None,
) -> dict[str, Path]:
    paths = _command_paths(argv)
    expected = {
        "executable": Path(sys.executable) if bind_current_executable else None,
        "plan": plan_path,
        "plan_sha256": plan_sha256_path,
        "study_commitment": commitment_path,
        "study_commitment_sha256": commitment_sha256_path,
        "project_root": project_root,
        "output": output,
    }
    for role, expected_path in expected.items():
        if (
            expected_path is not None
            and paths[role].resolve() != expected_path.resolve()
        ):
            raise AAMASFormalCLIError("command_manifest_path_mismatch")
    return paths


def _isolation_member_names(receipt: IsolationReceipt) -> frozenset[str]:
    base = DEFAULT_RECEIPT_NAME
    names = {base, f"{base}.sha256"}
    if receipt.backend is IsolationBackend.MACOS_SANDBOX:
        names.add(f"{base}.macos-runtime.json")
    else:
        names.update(
            {
                f"{base}.docker-runtime.json",
                f"{base}.source-manifest.json",
            }
        )
    return frozenset(names)


def _audit_isolated_job_membership(
    directory: Path,
    receipt: IsolationReceipt,
) -> None:
    if not directory.is_dir() or directory.is_symlink():
        raise AAMASFormalCLIError("invalid_isolated_job_membership")
    expected = _isolation_member_names(receipt) | {_ISOLATED_JOB_OUTPUT_MEMBER}
    entries = {item.name: item for item in directory.iterdir()}
    if frozenset(entries) != expected:
        raise AAMASFormalCLIError("invalid_isolated_job_membership")
    publication = entries[_ISOLATED_JOB_OUTPUT_MEMBER]
    details = publication.lstat()
    if not stat.S_ISDIR(details.st_mode) or stat.S_ISLNK(details.st_mode):
        raise AAMASFormalCLIError("invalid_isolated_job_membership")
    for name in _isolation_member_names(receipt):
        details = entries[name].lstat()
        if (
            not stat.S_ISREG(details.st_mode)
            or stat.S_ISLNK(details.st_mode)
            or details.st_nlink != 1
        ):
            raise AAMASFormalCLIError("invalid_isolated_job_membership")


def _require_committed_isolation_source(
    receipt: IsolationReceipt,
    commitment: AAMASPublicationStudyCommitment,
) -> None:
    """Require an execution-time source snapshot, not a post-run live checkout."""

    if (
        receipt.backend is not IsolationBackend.DOCKER_NONE
        or receipt.source_identity.mode
        is not IsolationInputMode.COMMITTED_SOURCE_SNAPSHOT
    ):
        raise AAMASFormalCLIError("isolation_source_not_committed_snapshot")
    if (
        commitment.git_commit is None
        or commitment.git_dirty_at_precommit
        or receipt.source_identity.commit != commitment.git_commit
    ):
        raise AAMASFormalCLIError("isolation_source_commitment_mismatch")


def _copy_isolation_evidence(
    receipt_path: Path,
    destination: Path,
) -> IsolationReceipt:
    if receipt_path.name != DEFAULT_RECEIPT_NAME:
        raise AAMASFormalCLIError("unexpected_isolation_receipt_name")
    try:
        receipt = audit_isolation_receipt(receipt_path, require_success=True)
    except IsolationError as error:
        raise AAMASFormalCLIError("invalid_isolation_receipt") from error
    names = _isolation_member_names(receipt)
    if not destination.is_dir() or destination.is_symlink():
        raise AAMASFormalCLIError("invalid_isolated_job_membership")
    if frozenset(item.name for item in destination.iterdir()) != {
        _ISOLATED_JOB_OUTPUT_MEMBER
    }:
        raise AAMASFormalCLIError("invalid_isolated_job_membership")
    for name in sorted(names):
        payload = _stable_regular_bytes(
            receipt_path.parent / name,
            error_code="invalid_isolation_receipt_members",
        )
        _write_new(destination / name, payload)
    _audit_isolated_job_membership(destination, receipt)
    try:
        copied = audit_isolation_receipt(
            destination / DEFAULT_RECEIPT_NAME,
            require_success=True,
        )
    except IsolationError as error:
        raise AAMASFormalCLIError("copied_isolation_receipt_invalid") from error
    if copied != receipt:
        raise AAMASFormalCLIError("copied_isolation_receipt_changed")
    _audit_isolated_job_membership(destination, copied)
    return copied


@dataclass(frozen=True, slots=True)
class AAMASSyntheticReleaseAuditReport:
    """Local semantic audit result for one source-recoverable synthetic release."""

    release_gate_passed: bool
    release_seal_sha256: str
    publication_root_seal_sha256: str
    plan_sha256: str
    study_commitment_sha256: str
    scientific_result_sha256: str
    isolation_receipt_sha256: str
    isolation_output_tree_sha256: str
    isolation_output_file_count: int
    isolation_output_directory_count: int
    command_sha256: str
    source_tree_sha256: str
    source_file_count: int

    def to_mapping(self, *, command: str) -> dict[str, object]:
        return {
            "clean_git_attested": False,
            "committed_source_receipt_binding_semantically_valid": True,
            "command": command,
            "command_sha256": self.command_sha256,
            "external_or_temporal_isolation_attested": False,
            "independent_custody": False,
            "isolation_receipt_sha256": self.isolation_receipt_sha256,
            "isolation_receipt_semantically_valid": True,
            "isolation_output_directory_count": (self.isolation_output_directory_count),
            "isolation_output_file_count": self.isolation_output_file_count,
            "isolation_output_tree_sha256": self.isolation_output_tree_sha256,
            "plan_sha256": self.plan_sha256,
            "preexisting_or_independent_retention_attested": False,
            "publication_eligible": False,
            "publication_root_seal_sha256": self.publication_root_seal_sha256,
            "release_gate_meaning": "synthetic_local_evidence_validation_only",
            "release_gate_passed": self.release_gate_passed,
            "release_seal_sha256": self.release_seal_sha256,
            "schema_version": 2,
            "scientific_result_sha256": self.scientific_result_sha256,
            "source_file_count": self.source_file_count,
            "source_recoverable_from_embedded_snapshot": True,
            "source_tree_sha256": self.source_tree_sha256,
            "status": "synthetic_release_locally_validated",
            "study_commitment_sha256": self.study_commitment_sha256,
            "synthetic_only": True,
            "validity_level": "abstain",
        }


def _read_canonical_object(path: Path, *, error_code: str) -> dict[str, object]:
    payload = _stable_regular_bytes(path, error_code=error_code)
    try:
        mapping = read_json_bytes_object(payload, source="sealed synthetic release")
    except MetadataValidationError as error:
        raise AAMASFormalCLIError(error_code) from error
    if payload != canonical_json_bytes(mapping):
        raise AAMASFormalCLIError(error_code)
    return mapping


def _release_mapping(
    *,
    publication: AAMASPublicationAuditReport,
    receipt: IsolationReceipt,
    command_sha256: str,
    command_paths: Mapping[str, Path],
    source_tree_sha256: str,
    source_file_count: int,
    secret_scan: Mapping[str, object],
) -> dict[str, object]:
    if not _synthetic_output_gate(publication):
        raise AAMASFormalCLIError("publication_outputs_incomplete")
    path_commitments = {
        role: sha256_bytes(canonical_json_bytes(str(path)))
        for role, path in sorted(command_paths.items())
    }
    return {
        "caller_supplied_isolation_receipt_raw_digest_verified": True,
        "claim_scope": "synthetic_local_evidence_validation_only",
        "clean_git_attested": False,
        "committed_source_receipt_binding_semantically_valid": True,
        "command_path_commitments": path_commitments,
        "command_sha256": command_sha256,
        "external_or_temporal_isolation_attested": False,
        "independent_custody": False,
        "isolation": {
            "backend": receipt.backend.value,
            "isolation_config_sha256": receipt.isolation_config_sha256,
            "output_root_member": _ISOLATED_JOB_OUTPUT_MEMBER,
            "output_tree": receipt.output_tree.to_mapping(),
            "receipt_sha256": receipt.sha256,
            "runtime_controls_sha256": receipt.runtime_controls_sha256,
            "runtime_identity": receipt.runtime_identity,
            "status": receipt.status.value,
        },
        "isolation_receipt_semantically_valid": True,
        "plan_sha256": publication.plan_sha256,
        "preexisting_or_independent_retention_attested": False,
        "publication_eligible": False,
        "publication_root_seal_sha256": publication.root_seal_sha256,
        "release_gate_meaning": "synthetic_local_evidence_validation_only",
        "release_gate_passed": True,
        "schema_version": _RELEASE_SCHEMA_VERSION,
        "scientific_result_sha256": publication.scientific_result_sha256,
        "scientifically_complete": False,
        "secret_scan": dict(secret_scan),
        "source_file_count": source_file_count,
        "source_recoverable_from_embedded_snapshot": True,
        "source_tree_sha256": source_tree_sha256,
        "study_commitment_sha256": publication.study_commitment_sha256,
        "submission_evidence_complete": True,
        "synthetic_only": True,
        "validity_level": "abstain",
    }


def _release_manifest_mapping(directory: Path) -> dict[str, object]:
    rows = tuple(
        row
        for row in _strict_tree_rows(directory)
        if row["path"] not in {"manifest.json", "seal.json"}
    )
    return {
        "file_count": len(rows),
        "files": list(rows),
        "fixed_root_directories": sorted(_RELEASE_ROOT_DIRECTORIES),
        "fixed_root_files": sorted(_RELEASE_ROOT_FILES),
        "schema_version": _RELEASE_MANIFEST_SCHEMA_VERSION,
        "tree_sha256": sha256_bytes(canonical_json_bytes(rows)),
    }


def _release_seal_mapping(
    *,
    release_payload: bytes,
    manifest_payload: bytes,
    release_mapping: Mapping[str, object],
) -> dict[str, object]:
    isolation = release_mapping.get("isolation")
    if not isinstance(isolation, Mapping):
        raise AAMASFormalCLIError("invalid_release_mapping")
    return {
        "caller_supplied_receipt_raw_digest_required_for_reaudit": True,
        "command_sha256": release_mapping["command_sha256"],
        "isolation_receipt_sha256": isolation["receipt_sha256"],
        "manifest_raw_sha256": sha256_bytes(manifest_payload),
        "plan_sha256": release_mapping["plan_sha256"],
        "publication_root_seal_sha256": release_mapping["publication_root_seal_sha256"],
        "release_raw_sha256": sha256_bytes(release_payload),
        "release_semantic_sha256": sha256_bytes(
            canonical_json_bytes(dict(release_mapping))
        ),
        "schema_version": _RELEASE_SEAL_SCHEMA_VERSION,
        "source_tree_sha256": release_mapping["source_tree_sha256"],
        "study_commitment_sha256": release_mapping["study_commitment_sha256"],
    }


def _verify_release_root(directory: Path) -> None:
    if not directory.is_dir() or directory.is_symlink():
        raise AAMASFormalCLIError("invalid_finalized_release")
    entries = {item.name: item for item in directory.iterdir()}
    if frozenset(entries) != _RELEASE_ROOT_FILES | _RELEASE_ROOT_DIRECTORIES:
        raise AAMASFormalCLIError("invalid_finalized_release_membership")
    for name in _RELEASE_ROOT_FILES:
        details = entries[name].lstat()
        if (
            not stat.S_ISREG(details.st_mode)
            or stat.S_ISLNK(details.st_mode)
            or details.st_nlink != 1
        ):
            raise AAMASFormalCLIError("invalid_finalized_release_membership")
    for name in _RELEASE_ROOT_DIRECTORIES:
        details = entries[name].lstat()
        if not stat.S_ISDIR(details.st_mode) or stat.S_ISLNK(details.st_mode):
            raise AAMASFormalCLIError("invalid_finalized_release_membership")
    _strict_tree_rows(directory)


def _read_publication_source_manifest(publication: Path) -> dict[str, object]:
    payload = _stable_regular_bytes(
        publication / "source-manifest.json",
        error_code="invalid_publication_source_manifest",
    )
    try:
        return read_json_bytes_object(
            payload,
            source="publication source manifest",
        )
    except MetadataValidationError as error:
        raise AAMASFormalCLIError("invalid_publication_source_manifest") from error


def _audit_finalized_release_snapshot(
    directory: Path,
    *,
    plan: AAMASPublicationPlan,
    plan_sha256: str,
    commitment: AAMASPublicationStudyCommitment,
    commitment_sha256: str,
    publication_root_seal_sha256: str,
    expected_release_seal_sha256: str,
    expected_isolation_receipt_sha256: str,
    allow_formal_scale: bool,
) -> AAMASSyntheticReleaseAuditReport:
    _verify_release_root(directory)
    seal_payload = _stable_regular_bytes(
        directory / "seal.json",
        error_code="invalid_release_seal",
    )
    if sha256_bytes(seal_payload) != expected_release_seal_sha256:
        raise AAMASFormalCLIError("release_seal_sha256_mismatch")
    seal = _read_canonical_object(
        directory / "seal.json",
        error_code="invalid_release_seal",
    )
    release = _read_canonical_object(
        directory / "release.json",
        error_code="invalid_release_mapping",
    )
    manifest = _read_canonical_object(
        directory / "manifest.json",
        error_code="invalid_release_manifest",
    )
    expected_manifest = _release_manifest_mapping(directory)
    if manifest != expected_manifest:
        raise AAMASFormalCLIError("release_manifest_mismatch")

    command_path = directory / "command-manifest.json"
    argv, command_payload, command_sha256 = _read_command_manifest(command_path)
    command_seal = _read_sha256_file(
        directory / "command-manifest.sha256",
        error_code="invalid_command_manifest_sha256_file",
    )
    if command_seal != command_sha256:
        raise AAMASFormalCLIError("command_manifest_sha256_mismatch")
    command_paths = _validate_run_command(argv)

    isolated_job = directory / "isolated-job"
    publication_directory = isolated_job / _ISOLATED_JOB_OUTPUT_MEMBER
    receipt_path = isolated_job / DEFAULT_RECEIPT_NAME
    receipt_payload = _stable_regular_bytes(
        receipt_path,
        error_code="invalid_isolation_receipt",
    )
    if sha256_bytes(receipt_payload) != expected_isolation_receipt_sha256:
        raise AAMASFormalCLIError("external_isolation_receipt_sha256_mismatch")
    try:
        receipt = audit_isolation_receipt(receipt_path, require_success=True)
    except IsolationError as error:
        raise AAMASFormalCLIError("invalid_isolation_receipt") from error
    if receipt.sha256 != expected_isolation_receipt_sha256:
        raise AAMASFormalCLIError("external_isolation_receipt_sha256_mismatch")
    _require_committed_isolation_source(receipt, commitment)
    if receipt.command_sha256 != command_sha256:
        raise AAMASFormalCLIError("isolation_command_mismatch")
    _audit_isolated_job_membership(isolated_job, receipt)

    source_manifest = _read_publication_source_manifest(publication_directory)
    source_rows, source_tree_sha256 = _verify_source_snapshot(
        directory / "source-snapshot",
        source_manifest,
        commitment,
    )
    publication = audit_aamas_publication(
        publication_directory,
        expected_plan=plan,
        expected_plan_sha256=plan_sha256,
        expected_study_commitment_sha256=commitment_sha256,
        expected_root_seal_sha256=publication_root_seal_sha256,
        project_root=directory / "source-snapshot",
        allow_formal_scale=allow_formal_scale,
    )
    if not _synthetic_output_gate(publication):
        raise AAMASFormalCLIError("publication_outputs_incomplete")
    secret_scan = _scan_high_confidence_secrets(
        directory,
        identity_exclusions=_POST_SCAN_RELEASE_FILES,
    )
    expected_release = _release_mapping(
        publication=publication,
        receipt=receipt,
        command_sha256=command_sha256,
        command_paths=command_paths,
        source_tree_sha256=source_tree_sha256,
        source_file_count=len(source_rows),
        secret_scan=secret_scan,
    )
    if release != expected_release:
        raise AAMASFormalCLIError("release_mapping_mismatch")
    expected_seal = _release_seal_mapping(
        release_payload=canonical_json_bytes(release),
        manifest_payload=canonical_json_bytes(manifest),
        release_mapping=release,
    )
    if seal != expected_seal:
        raise AAMASFormalCLIError("release_seal_mapping_mismatch")
    if (
        seal.get("schema_version") != _RELEASE_SEAL_SCHEMA_VERSION
        or seal.get("caller_supplied_receipt_raw_digest_required_for_reaudit")
        is not True
        or command_payload != canonical_json_bytes(list(argv))
    ):
        raise AAMASFormalCLIError("invalid_release_seal")
    return AAMASSyntheticReleaseAuditReport(
        release_gate_passed=True,
        release_seal_sha256=expected_release_seal_sha256,
        publication_root_seal_sha256=publication.root_seal_sha256,
        plan_sha256=publication.plan_sha256,
        study_commitment_sha256=publication.study_commitment_sha256,
        scientific_result_sha256=publication.scientific_result_sha256,
        isolation_receipt_sha256=receipt.sha256,
        isolation_output_tree_sha256=receipt.output_tree.tree_sha256,
        isolation_output_file_count=receipt.output_tree.file_count,
        isolation_output_directory_count=receipt.output_tree.directory_count,
        command_sha256=command_sha256,
        source_tree_sha256=source_tree_sha256,
        source_file_count=len(source_rows),
    )


def _audit_finalized_release(
    directory: Path,
    *,
    plan: AAMASPublicationPlan,
    plan_sha256: str,
    commitment: AAMASPublicationStudyCommitment,
    commitment_sha256: str,
    publication_root_seal_sha256: str,
    expected_release_seal_sha256: str,
    expected_isolation_receipt_sha256: str,
    allow_formal_scale: bool,
) -> AAMASSyntheticReleaseAuditReport:
    """Audit only a private snapshot and reject any live-tree drift."""

    _verify_release_root(directory)
    live_rows = _strict_tree_rows(directory)
    live_directories = _strict_directories(directory)
    with tempfile.TemporaryDirectory(
        prefix="policytwin-aamas-finalized-release-audit-"
    ) as raw:
        snapshot = Path(raw).resolve() / "release"
        try:
            shutil.copytree(directory, snapshot, symlinks=True)
        except OSError as error:
            raise AAMASFormalCLIError("cannot_snapshot_finalized_release") from error
        if (
            _strict_tree_rows(snapshot) != live_rows
            or _strict_directories(snapshot) != live_directories
        ):
            raise AAMASFormalCLIError("finalized_release_changed_during_snapshot")
        report = _audit_finalized_release_snapshot(
            snapshot,
            plan=plan,
            plan_sha256=plan_sha256,
            commitment=commitment,
            commitment_sha256=commitment_sha256,
            publication_root_seal_sha256=publication_root_seal_sha256,
            expected_release_seal_sha256=expected_release_seal_sha256,
            expected_isolation_receipt_sha256=(expected_isolation_receipt_sha256),
            allow_formal_scale=allow_formal_scale,
        )
        if (
            _strict_tree_rows(snapshot) != live_rows
            or _strict_directories(snapshot) != live_directories
        ):
            raise AAMASFormalCLIError("private_finalized_release_snapshot_changed")
    _verify_release_root(directory)
    if (
        _strict_tree_rows(directory) != live_rows
        or _strict_directories(directory) != live_directories
    ):
        raise AAMASFormalCLIError("finalized_release_changed_during_audit")
    return report


def _finalize_release(
    directory: Path,
    *,
    bundle: Path,
    receipt_path: Path,
    receipt_sha256_path: Path,
    command_manifest_path: Path,
    command_manifest_sha256_path: Path,
    plan_path: Path,
    plan_sha256_path: Path,
    commitment_path: Path,
    commitment_sha256_path: Path,
    plan: AAMASPublicationPlan,
    plan_sha256: str,
    commitment: AAMASPublicationStudyCommitment,
    commitment_sha256: str,
    publication_root_seal_sha256: str,
    project_root: Path,
    allow_formal_scale: bool,
) -> AAMASSyntheticReleaseAuditReport:
    if directory.exists() or directory.is_symlink():
        raise AAMASFormalCLIError("release_output_already_exists")
    if directory.resolve().is_relative_to(
        project_root.resolve() / "src" / "policytwin"
    ):
        raise AAMASFormalCLIError("release_output_inside_source_scope")
    if bundle.name != _ISOLATED_JOB_OUTPUT_MEMBER:
        raise AAMASFormalCLIError("unexpected_isolated_output_member")
    publication = audit_aamas_publication(
        bundle,
        expected_plan=plan,
        expected_plan_sha256=plan_sha256,
        expected_study_commitment_sha256=commitment_sha256,
        expected_root_seal_sha256=publication_root_seal_sha256,
        project_root=project_root,
        allow_formal_scale=allow_formal_scale,
    )
    if not _synthetic_output_gate(publication):
        raise AAMASFormalCLIError("publication_outputs_incomplete")
    if receipt_path.parent.resolve() != bundle.parent.resolve():
        raise AAMASFormalCLIError("isolation_receipt_bundle_job_mismatch")
    if receipt_sha256_path.resolve().is_relative_to(receipt_path.parent.resolve()):
        raise AAMASFormalCLIError(
            "isolation_receipt_sha256_not_external_to_isolated_job"
        )
    external_receipt_sha256 = _read_sha256_file(
        receipt_sha256_path,
        error_code="invalid_external_isolation_receipt_sha256_file",
    )
    receipt_payload = _stable_regular_bytes(
        receipt_path,
        error_code="invalid_isolation_receipt",
    )
    if sha256_bytes(receipt_payload) != external_receipt_sha256:
        raise AAMASFormalCLIError("external_isolation_receipt_sha256_mismatch")
    try:
        receipt = audit_isolation_receipt(receipt_path, require_success=True)
    except IsolationError as error:
        raise AAMASFormalCLIError("invalid_isolation_receipt") from error
    if receipt.sha256 != external_receipt_sha256:
        raise AAMASFormalCLIError("external_isolation_receipt_sha256_mismatch")
    _require_committed_isolation_source(receipt, commitment)
    _audit_isolated_job_membership(receipt_path.parent, receipt)
    argv, command_payload, command_sha256 = _read_command_manifest(
        command_manifest_path
    )
    external_command_sha256 = _read_sha256_file(
        command_manifest_sha256_path,
        error_code="invalid_command_manifest_sha256_file",
    )
    if (
        command_sha256 != external_command_sha256
        or command_sha256 != receipt.command_sha256
    ):
        raise AAMASFormalCLIError("isolation_command_mismatch")
    command_paths = _validate_run_command(
        argv,
        bind_current_executable=True,
        plan_path=plan_path,
        plan_sha256_path=plan_sha256_path,
        commitment_path=commitment_path,
        commitment_sha256_path=commitment_sha256_path,
        project_root=project_root,
        output=bundle,
    )

    directory.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{directory.name}.staging-", dir=directory.parent)
    )
    staging_identity = _directory_identity(staging.lstat())
    try:
        isolated_job = staging / "isolated-job"
        isolated_job.mkdir(exist_ok=False)
        copied_publication_directory = isolated_job / _ISOLATED_JOB_OUTPUT_MEMBER
        _copy_strict_tree(bundle, copied_publication_directory)
        source_file_count, source_tree_sha256 = _copy_source_snapshot(
            project_root=project_root,
            publication=copied_publication_directory,
            destination=staging / "source-snapshot",
            commitment=commitment,
        )
        copied_receipt = _copy_isolation_evidence(
            receipt_path,
            isolated_job,
        )
        if copied_receipt != receipt:
            raise AAMASFormalCLIError("copied_isolation_receipt_changed")
        _write_new(staging / "command-manifest.json", command_payload)
        _write_new(
            staging / "command-manifest.sha256",
            (command_sha256 + "\n").encode("ascii"),
        )
        copied_publication = audit_aamas_publication(
            copied_publication_directory,
            expected_plan=plan,
            expected_plan_sha256=plan_sha256,
            expected_study_commitment_sha256=commitment_sha256,
            expected_root_seal_sha256=publication_root_seal_sha256,
            project_root=staging / "source-snapshot",
            allow_formal_scale=allow_formal_scale,
        )
        if copied_publication != publication:
            raise AAMASFormalCLIError("copied_publication_changed")
        secret_scan = _scan_high_confidence_secrets(
            staging,
            identity_exclusions=_POST_SCAN_RELEASE_FILES,
        )
        release = _release_mapping(
            publication=copied_publication,
            receipt=copied_receipt,
            command_sha256=command_sha256,
            command_paths=command_paths,
            source_tree_sha256=source_tree_sha256,
            source_file_count=source_file_count,
            secret_scan=secret_scan,
        )
        release_payload = canonical_json_bytes(release)
        _write_new(staging / "release.json", release_payload)
        manifest = _release_manifest_mapping(staging)
        manifest_payload = canonical_json_bytes(manifest)
        _write_new(staging / "manifest.json", manifest_payload)
        seal = _release_seal_mapping(
            release_payload=release_payload,
            manifest_payload=manifest_payload,
            release_mapping=release,
        )
        seal_payload = canonical_json_bytes(seal)
        _write_new(staging / "seal.json", seal_payload)
        if (
            _scan_high_confidence_secrets(
                staging,
                identity_exclusions=_POST_SCAN_RELEASE_FILES,
            )
            != secret_scan
        ):
            raise AAMASFormalCLIError("secret_scan_input_changed")
        release_seal_sha256 = sha256_bytes(seal_payload)
        _audit_finalized_release(
            staging,
            plan=plan,
            plan_sha256=plan_sha256,
            commitment=commitment,
            commitment_sha256=commitment_sha256,
            publication_root_seal_sha256=publication_root_seal_sha256,
            expected_release_seal_sha256=release_seal_sha256,
            expected_isolation_receipt_sha256=external_receipt_sha256,
            allow_formal_scale=allow_formal_scale,
        )
        if directory.exists() or directory.is_symlink():
            raise AAMASFormalCLIError("release_output_appeared_before_publish")
        published_identity = staging_identity
        os.replace(staging, directory)
        try:
            return _audit_finalized_release(
                directory,
                plan=plan,
                plan_sha256=plan_sha256,
                commitment=commitment,
                commitment_sha256=commitment_sha256,
                publication_root_seal_sha256=publication_root_seal_sha256,
                expected_release_seal_sha256=release_seal_sha256,
                expected_isolation_receipt_sha256=external_receipt_sha256,
                allow_formal_scale=allow_formal_scale,
            )
        except BaseException:
            _quarantine_owned_directory(
                directory,
                expected_identity=published_identity,
            )
            raise
    except BaseException:
        _quarantine_owned_directory(
            staging,
            expected_identity=staging_identity,
        )
        raise


def _add_external_inputs(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256-file", type=Path, required=True)
    parser.add_argument("--study-commitment", type=Path, required=True)
    parser.add_argument("--study-commitment-sha256-file", type=Path, required=True)
    parser.add_argument("--project-root", type=Path)
    parser.add_argument(
        "--allow-formal-scale",
        action="store_true",
        required=True,
        help="explicitly authorize the committed plan's formal-scale guard",
    )


def _parser() -> argparse.ArgumentParser:
    parser = _MachineArgumentParser(
        prog="python -m policytwin.aamas_formal",
        description="Run or audit one externally committed synthetic AAMAS bundle.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    run_parser = commands.add_parser("run", help="execute and publish one bundle")
    _add_external_inputs(run_parser)
    run_parser.add_argument("--output", type=Path, required=True)

    audit_parser = commands.add_parser("audit", help="replay-audit one bundle")
    _add_external_inputs(audit_parser)
    audit_parser.add_argument("--bundle", type=Path, required=True)
    audit_parser.add_argument("--root-seal-sha256-file", type=Path, required=True)

    finalize_parser = commands.add_parser(
        "finalize",
        help="bind a successful OS-isolation receipt and embedded source snapshot",
    )
    _add_external_inputs(finalize_parser)
    finalize_parser.add_argument("--bundle", type=Path, required=True)
    finalize_parser.add_argument(
        "--root-seal-sha256-file",
        type=Path,
        required=True,
    )
    finalize_parser.add_argument("--isolation-receipt", type=Path, required=True)
    finalize_parser.add_argument(
        "--isolation-receipt-sha256-file",
        type=Path,
        required=True,
    )
    finalize_parser.add_argument("--command-manifest", type=Path, required=True)
    finalize_parser.add_argument(
        "--command-manifest-sha256-file",
        type=Path,
        required=True,
    )
    finalize_parser.add_argument("--release-output", type=Path, required=True)

    audit_finalize_parser = commands.add_parser(
        "audit-finalize",
        help="re-audit one locally validated source-recoverable synthetic release",
    )
    _add_external_inputs(audit_finalize_parser)
    audit_finalize_parser.add_argument("--release", type=Path, required=True)
    audit_finalize_parser.add_argument(
        "--publication-root-seal-sha256-file",
        type=Path,
        required=True,
    )
    audit_finalize_parser.add_argument(
        "--release-seal-sha256-file",
        type=Path,
        required=True,
    )
    audit_finalize_parser.add_argument(
        "--isolation-receipt-sha256-file",
        type=Path,
        required=True,
    )
    return parser


def _execute_run(arguments: argparse.Namespace) -> tuple[dict[str, object], int]:
    plan, plan_sha256, commitment, commitment_sha256 = _load_external_inputs(
        plan_path=cast(Path, arguments.plan),
        plan_sha256_path=cast(Path, arguments.plan_sha256_file),
        commitment_path=cast(Path, arguments.study_commitment),
        commitment_sha256_path=cast(Path, arguments.study_commitment_sha256_file),
    )
    output = cast(Path, arguments.output)
    if output.exists() or output.is_symlink():
        raise AAMASFormalCLIError("output_already_exists")
    project_root = cast(Path | None, arguments.project_root)
    allow_formal_scale = cast(bool, arguments.allow_formal_scale)
    run = run_aamas_publication(
        plan,
        expected_plan_sha256=plan_sha256,
        study_commitment=commitment,
        expected_study_commitment_sha256=commitment_sha256,
        project_root=project_root,
        allow_formal_scale=allow_formal_scale,
    )
    written = write_aamas_publication(
        output,
        run=run,
        expected_plan_sha256=plan_sha256,
        expected_study_commitment_sha256=commitment_sha256,
        project_root=project_root,
        allow_formal_scale=allow_formal_scale,
    )
    report = audit_aamas_publication(
        output,
        expected_plan=plan,
        expected_plan_sha256=plan_sha256,
        expected_study_commitment_sha256=commitment_sha256,
        expected_root_seal_sha256=written.root_seal_sha256,
        project_root=project_root,
        allow_formal_scale=allow_formal_scale,
    )
    if (
        written.plan_sha256 != report.plan_sha256
        or written.study_commitment_sha256 != report.study_commitment_sha256
        or written.root_seal_sha256 != report.root_seal_sha256
        or written.scientific_result_sha256 != report.scientific_result_sha256
        or written.submission_evidence_complete
        is not report.submission_evidence_complete
    ):
        raise AAMASFormalCLIError("write_audit_disagreement")
    payload = _audit_payload(report, command="run")
    return payload, 0 if cast(bool, payload["synthetic_outputs_complete"]) else 1


def _execute_audit(arguments: argparse.Namespace) -> tuple[dict[str, object], int]:
    plan, plan_sha256, commitment, commitment_sha256 = _load_external_inputs(
        plan_path=cast(Path, arguments.plan),
        plan_sha256_path=cast(Path, arguments.plan_sha256_file),
        commitment_path=cast(Path, arguments.study_commitment),
        commitment_sha256_path=cast(Path, arguments.study_commitment_sha256_file),
    )
    root_seal_sha256 = _read_sha256_file(
        cast(Path, arguments.root_seal_sha256_file),
        error_code="invalid_root_seal_sha256_file",
    )
    if commitment.plan_sha256 != plan_sha256:
        raise AAMASFormalCLIError("commitment_plan_mismatch")
    report = audit_aamas_publication(
        cast(Path, arguments.bundle),
        expected_plan=plan,
        expected_plan_sha256=plan_sha256,
        expected_study_commitment_sha256=commitment_sha256,
        expected_root_seal_sha256=root_seal_sha256,
        project_root=cast(Path | None, arguments.project_root),
        allow_formal_scale=cast(bool, arguments.allow_formal_scale),
    )
    payload = _audit_payload(report, command="audit")
    return payload, 0 if cast(bool, payload["synthetic_outputs_complete"]) else 1


def _execute_finalize(arguments: argparse.Namespace) -> tuple[dict[str, object], int]:
    plan, plan_sha256, commitment, commitment_sha256 = _load_external_inputs(
        plan_path=cast(Path, arguments.plan),
        plan_sha256_path=cast(Path, arguments.plan_sha256_file),
        commitment_path=cast(Path, arguments.study_commitment),
        commitment_sha256_path=cast(Path, arguments.study_commitment_sha256_file),
    )
    project_root = cast(Path | None, arguments.project_root)
    if project_root is None:
        raise AAMASFormalCLIError("finalize_requires_project_root")
    publication_root_seal_sha256 = _read_sha256_file(
        cast(Path, arguments.root_seal_sha256_file),
        error_code="invalid_root_seal_sha256_file",
    )
    report = _finalize_release(
        cast(Path, arguments.release_output),
        bundle=cast(Path, arguments.bundle),
        receipt_path=cast(Path, arguments.isolation_receipt),
        receipt_sha256_path=cast(
            Path,
            arguments.isolation_receipt_sha256_file,
        ),
        command_manifest_path=cast(Path, arguments.command_manifest),
        command_manifest_sha256_path=cast(
            Path,
            arguments.command_manifest_sha256_file,
        ),
        plan_path=cast(Path, arguments.plan),
        plan_sha256_path=cast(Path, arguments.plan_sha256_file),
        commitment_path=cast(Path, arguments.study_commitment),
        commitment_sha256_path=cast(Path, arguments.study_commitment_sha256_file),
        plan=plan,
        plan_sha256=plan_sha256,
        commitment=commitment,
        commitment_sha256=commitment_sha256,
        publication_root_seal_sha256=publication_root_seal_sha256,
        project_root=project_root,
        allow_formal_scale=cast(bool, arguments.allow_formal_scale),
    )
    return report.to_mapping(command="finalize"), 0


def _execute_audit_finalize(
    arguments: argparse.Namespace,
) -> tuple[dict[str, object], int]:
    plan, plan_sha256, commitment, commitment_sha256 = _load_external_inputs(
        plan_path=cast(Path, arguments.plan),
        plan_sha256_path=cast(Path, arguments.plan_sha256_file),
        commitment_path=cast(Path, arguments.study_commitment),
        commitment_sha256_path=cast(Path, arguments.study_commitment_sha256_file),
    )
    release = cast(Path, arguments.release)
    receipt_sha256_path = cast(Path, arguments.isolation_receipt_sha256_file)
    if receipt_sha256_path.resolve().is_relative_to(release.resolve()):
        raise AAMASFormalCLIError("isolation_receipt_sha256_not_external_to_release")
    expected_isolation_receipt_sha256 = _read_sha256_file(
        receipt_sha256_path,
        error_code="invalid_external_isolation_receipt_sha256_file",
    )
    publication_root_seal_sha256 = _read_sha256_file(
        cast(Path, arguments.publication_root_seal_sha256_file),
        error_code="invalid_root_seal_sha256_file",
    )
    release_seal_sha256 = _read_sha256_file(
        cast(Path, arguments.release_seal_sha256_file),
        error_code="invalid_release_seal_sha256_file",
    )
    report = _audit_finalized_release(
        release,
        plan=plan,
        plan_sha256=plan_sha256,
        commitment=commitment,
        commitment_sha256=commitment_sha256,
        publication_root_seal_sha256=publication_root_seal_sha256,
        expected_release_seal_sha256=release_seal_sha256,
        expected_isolation_receipt_sha256=expected_isolation_receipt_sha256,
        allow_formal_scale=cast(bool, arguments.allow_formal_scale),
    )
    return report.to_mapping(command="audit-finalize"), 0


def main(argv: Sequence[str] | None = None) -> int:
    """Execute the strict AAMAS publication boundary."""

    command: str | None = None
    try:
        arguments = _parser().parse_args(argv)
        command = cast(str, arguments.command)
        if command == "run":
            payload, exit_code = _execute_run(arguments)
        elif command == "audit":
            payload, exit_code = _execute_audit(arguments)
        elif command == "finalize":
            payload, exit_code = _execute_finalize(arguments)
        elif command == "audit-finalize":
            payload, exit_code = _execute_audit_finalize(arguments)
        else:  # pragma: no cover - argparse enforces the finite command set.
            raise AAMASFormalCLIError("invalid_cli_arguments")
        _emit(payload, error=exit_code != 0)
        return exit_code
    except AAMASFormalCLIError as error:
        _emit(
            {
                "command": command,
                "error_code": error.error_code,
                "schema_version": 1,
                "status": "error",
            },
            error=True,
        )
        return 2
    except AAMASPublicationIntegrityError:
        _emit(
            {
                "command": command,
                "error_code": "publication_integrity_error",
                "schema_version": 1,
                "status": "error",
            },
            error=True,
        )
        return 2
    except (
        AAMASPublicationError,
        IsolationError,
        MetadataValidationError,
        OSError,
        ValueError,
    ):
        _emit(
            {
                "command": command,
                "error_code": "publication_execution_error",
                "schema_version": 1,
                "status": "error",
            },
            error=True,
        )
        return 2


if __name__ == "__main__":  # pragma: no cover - exercised through ``main``.
    raise SystemExit(main())


__all__ = ["AAMASFormalCLIError", "AAMASSyntheticReleaseAuditReport", "main"]
