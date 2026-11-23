"""Integrity and provenance audit for PolicyTwin run bundles."""

from __future__ import annotations

import os
import stat
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Literal

from policytwin.config import DataManifest, SmokeExperimentConfig
from policytwin.counterfactual.randomness import EVENT_KEY_SCHEMA_VERSION
from policytwin.experiments.smoke import (
    CLAIM_SCOPE,
    EMPIRICAL_STATUS,
    run_smoke_experiment,
)
from policytwin.provenance import (
    MetadataValidationError,
    RunMetadata,
    build_event_key_metadata,
    canonical_json_bytes,
    collect_git_metadata,
    read_json_bytes_object,
    read_jsonl_bytes_objects,
    sha256_bytes,
    sha256_file,
)
from policytwin.run import (
    RUN_BUNDLE_FILES,
    RunCompletion,
    RunCreationError,
    derive_run_identity,
    find_project_root,
)

_MAX_BUNDLE_FILE_BYTES = 128 * 1024 * 1024
_StatSignature = tuple[int, int, int, int, int, int, int]


@dataclass(frozen=True, slots=True)
class _CapturedBundle:
    root_signature: _StatSignature
    entry_names: frozenset[str]
    files: Mapping[str, bytes]
    file_signatures: Mapping[str, _StatSignature]


def _stat_signature(info: os.stat_result) -> _StatSignature:
    return (
        info.st_dev,
        info.st_ino,
        info.st_mode,
        info.st_nlink,
        info.st_size,
        info.st_mtime_ns,
        info.st_ctime_ns,
    )


def _capture_run_bundle(
    root: Path, expected_files: frozenset[str]
) -> tuple[_CapturedBundle, list[str]]:
    """Read stable non-linked files through a no-follow directory descriptor."""
    errors: list[str] = []
    try:
        linked_root = os.lstat(root)
    except OSError as error:
        raise ValueError(f"run directory does not exist: {root}: {error}") from error
    if stat.S_ISLNK(linked_root.st_mode) or not stat.S_ISDIR(linked_root.st_mode):
        raise ValueError("run directory must be a real non-symlink directory")
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(root, flags)
    except OSError as error:
        raise ValueError(f"cannot safely open run directory: {error}") from error
    try:
        opened_root = os.fstat(descriptor)
        if _stat_signature(opened_root) != _stat_signature(linked_root):
            raise ValueError("run directory changed while it was opened")
        entry_names = frozenset(os.listdir(descriptor))
        files: dict[str, bytes] = {}
        signatures: dict[str, _StatSignature] = {}
        unsafe_entries: set[str] = set()
        for name in sorted(expected_files & entry_names):
            file_descriptor: int | None = None
            try:
                file_descriptor = os.open(
                    name,
                    os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
                    dir_fd=descriptor,
                )
                before = os.fstat(file_descriptor)
                if (
                    not stat.S_ISREG(before.st_mode)
                    or before.st_nlink != 1
                    or before.st_size > _MAX_BUNDLE_FILE_BYTES
                ):
                    unsafe_entries.add(name)
                    continue
                chunks: list[bytes] = []
                remaining = before.st_size
                while remaining:
                    chunk = os.read(file_descriptor, min(1024 * 1024, remaining))
                    if not chunk:
                        raise OSError("short read while capturing run file")
                    chunks.append(chunk)
                    remaining -= len(chunk)
                if os.read(file_descriptor, 1):
                    raise OSError("run file grew while it was captured")
                after = os.fstat(file_descriptor)
                linked_after = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
                signature = _stat_signature(before)
                if (
                    _stat_signature(after) != signature
                    or _stat_signature(linked_after) != signature
                ):
                    raise OSError("run file changed while it was captured")
                files[name] = b"".join(chunks)
                signatures[name] = signature
            except OSError as error:
                unsafe_entries.add(name)
                errors.append(f"cannot safely capture run file {name}: {error}")
            finally:
                if file_descriptor is not None:
                    os.close(file_descriptor)
        if unsafe_entries:
            errors.append(
                "run files must be bounded regular non-symlinks with one link: "
                + ", ".join(sorted(unsafe_entries))
            )
        if frozenset(os.listdir(descriptor)) != entry_names:
            errors.append("run directory entries changed during capture")
        if _stat_signature(os.fstat(descriptor)) != _stat_signature(opened_root):
            errors.append("run directory changed during capture")
        return (
            _CapturedBundle(
                root_signature=_stat_signature(opened_root),
                entry_names=entry_names,
                files=files,
                file_signatures=signatures,
            ),
            errors,
        )
    finally:
        os.close(descriptor)


def _captured_bundle_is_current(root: Path, capture: _CapturedBundle) -> bool:
    try:
        linked_root = os.lstat(root)
        if _stat_signature(linked_root) != capture.root_signature:
            return False
        flags = (
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
        )
        descriptor = os.open(root, flags)
        try:
            if _stat_signature(os.fstat(descriptor)) != capture.root_signature:
                return False
            if frozenset(os.listdir(descriptor)) != capture.entry_names:
                return False
            return all(
                _stat_signature(os.stat(name, dir_fd=descriptor, follow_symlinks=False))
                == signature
                for name, signature in capture.file_signatures.items()
            )
        finally:
            os.close(descriptor)
    except OSError:
        return False


class AuditStatus(StrEnum):
    """Unambiguous disposition of a run-bundle audit."""

    INTEGRITY_FAILED = "integrity_failed"
    DEVELOPMENT_ONLY = "integrity_valid_development_only"
    RELEASE_IDENTITY_VERIFIED = "release_identity_verified"


@dataclass(frozen=True, slots=True)
class AuditReport:
    """Machine-readable integrity and release-identity decisions.

    ``audit-run`` does not receive a study protocol or typed upstream verifier
    receipts, so it can never authorize formal reporting.  The legacy ``valid``
    property remains as an in-process alias for bundle integrity only; it is
    deliberately absent from the serialized CLI schema.
    """

    run_directory: str
    bundle_integrity_valid: bool
    release_identity_valid: bool
    development_only: bool
    status: AuditStatus
    errors: tuple[str, ...]
    warnings: tuple[str, ...]
    release_blockers: tuple[str, ...]
    checked_files: tuple[str, ...]
    formal_reporting_eligible: Literal[False] = False

    def __post_init__(self) -> None:
        if not isinstance(self.status, AuditStatus):
            raise TypeError("audit status must be an AuditStatus")
        if self.release_identity_valid and not self.bundle_integrity_valid:
            raise ValueError("release identity cannot pass an invalid bundle")
        if self.development_only != (
            self.bundle_integrity_valid and not self.release_identity_valid
        ):
            raise ValueError("development_only is inconsistent with audit gates")
        expected_status = (
            AuditStatus.INTEGRITY_FAILED
            if not self.bundle_integrity_valid
            else (
                AuditStatus.RELEASE_IDENTITY_VERIFIED
                if self.release_identity_valid
                else AuditStatus.DEVELOPMENT_ONLY
            )
        )
        if self.status is not expected_status:
            raise ValueError("audit status is inconsistent with audit gates")
        if self.formal_reporting_eligible is not False:
            raise ValueError("audit-run cannot authorize formal reporting")
        if self.release_identity_valid and any(
            blocker != "formal_reporting_requires_study_protocol_gate"
            for blocker in self.release_blockers
        ):
            raise ValueError("verified release identity carries identity blockers")

    @property
    def valid(self) -> bool:
        """Return bundle integrity only; prefer ``bundle_integrity_valid``."""
        return self.bundle_integrity_valid

    def to_mapping(self) -> dict[str, object]:
        """Return a stable JSON representation for CLI output."""
        return {
            "bundle_integrity_valid": self.bundle_integrity_valid,
            "checked_files": list(self.checked_files),
            "development_only": self.development_only,
            "errors": list(self.errors),
            "formal_reporting_eligible": self.formal_reporting_eligible,
            "release_blockers": list(self.release_blockers),
            "release_identity_valid": self.release_identity_valid,
            "run_directory": self.run_directory,
            "schema_version": 2,
            "status": self.status.value,
            "warnings": list(self.warnings),
        }


def _build_report(
    *,
    run_directory: Path,
    errors: list[str],
    warnings: list[str],
    checked_files: list[str],
    git_commit: str | None = None,
    git_dirty: bool = True,
    live_git_commit_matches: bool = False,
    live_git_clean: bool = False,
    environment_lock_present: bool = False,
) -> AuditReport:
    bundle_integrity_valid = not errors
    release_blockers: list[str] = []
    if not bundle_integrity_valid:
        release_blockers.append("bundle_integrity_failed")
    if git_commit is None:
        release_blockers.append("git_commit_missing")
    if git_dirty:
        release_blockers.append("git_worktree_dirty_at_run")
    if not live_git_commit_matches:
        release_blockers.append("live_git_commit_mismatch")
    if not live_git_clean:
        release_blockers.append("live_git_worktree_dirty_or_unavailable")
    if not environment_lock_present:
        release_blockers.append("environment_lock_missing")
    release_identity_valid = not any(
        blocker
        in {
            "bundle_integrity_failed",
            "git_commit_missing",
            "git_worktree_dirty_at_run",
            "live_git_commit_mismatch",
            "live_git_worktree_dirty_or_unavailable",
            "environment_lock_missing",
        }
        for blocker in release_blockers
    )
    release_blockers.append("formal_reporting_requires_study_protocol_gate")
    status = (
        AuditStatus.INTEGRITY_FAILED
        if not bundle_integrity_valid
        else (
            AuditStatus.RELEASE_IDENTITY_VERIFIED
            if release_identity_valid
            else AuditStatus.DEVELOPMENT_ONLY
        )
    )
    return AuditReport(
        run_directory=str(run_directory),
        bundle_integrity_valid=bundle_integrity_valid,
        release_identity_valid=release_identity_valid,
        development_only=bundle_integrity_valid and not release_identity_valid,
        status=status,
        errors=tuple(errors),
        warnings=tuple(warnings),
        release_blockers=tuple(release_blockers),
        checked_files=tuple(sorted(set(checked_files))),
    )


def _audit_timestamp(value: str, field: str, errors: list[str]) -> None:
    try:
        timestamp = datetime.fromisoformat(value)
    except ValueError:
        errors.append(f"metadata {field} is not ISO 8601")
        return
    if timestamp.tzinfo is None or timestamp.utcoffset() is None:
        errors.append(f"metadata {field} has no timezone")


def _audit_source(
    label: str,
    path_text: str,
    expected_source_hash: str,
    trusted_root: Path,
    warnings: list[str],
    errors: list[str],
) -> None:
    path = Path(path_text).expanduser().resolve()
    try:
        path.relative_to(trusted_root)
    except ValueError:
        warnings.append(f"original {label} is outside the trusted project root")
        return
    if not path.is_file():
        warnings.append(f"original {label} source is unavailable: {path}")
        return
    if sha256_file(path) != expected_source_hash:
        errors.append(f"original {label} source hash changed: {path}")


def audit_run(run_directory: Path) -> AuditReport:
    """Audit hashes, strict schemas, event keys, sources, and result linkage."""
    run_dir = Path(os.path.abspath(run_directory.expanduser()))
    warnings: list[str] = []
    checked: list[str] = []
    trusted_root = find_project_root(Path(__file__))
    expected_files = frozenset(RUN_BUNDLE_FILES)
    try:
        capture, errors = _capture_run_bundle(run_dir, expected_files)
    except ValueError as error:
        return _build_report(
            run_directory=run_dir,
            errors=[str(error)],
            warnings=[],
            checked_files=[],
        )

    actual_files = capture.entry_names
    missing_files = sorted(expected_files - actual_files)
    unexpected_files = sorted(actual_files - expected_files)
    if missing_files:
        errors.append(f"missing run files: {', '.join(missing_files)}")
    if unexpected_files:
        errors.append(f"unexpected run files: {', '.join(unexpected_files)}")

    metadata_bytes = capture.files.get("metadata.json")
    seal_bytes = capture.files.get("metadata.sha256")
    completion_bytes = capture.files.get("completion.json")
    metadata: RunMetadata | None = None

    if metadata_bytes is not None and seal_bytes is not None:
        checked.extend(["metadata.json", "metadata.sha256"])
        try:
            seal_parts = seal_bytes.decode("utf-8").strip().split()
        except UnicodeError as error:
            errors.append(f"cannot read metadata seal: {error}")
            seal_parts = []
        if len(seal_parts) != 2 or seal_parts[1] != "metadata.json":
            errors.append("metadata.sha256 has an invalid format")
        elif seal_parts[0] != sha256_bytes(metadata_bytes):
            errors.append("metadata.json hash does not match metadata.sha256")
        try:
            metadata = RunMetadata.from_mapping(
                read_json_bytes_object(metadata_bytes, source="metadata.json")
            )
        except (MetadataValidationError, KeyError, TypeError, ValueError) as error:
            errors.append(f"metadata schema is invalid: {error}")

    if metadata is None:
        return _build_report(
            run_directory=run_dir,
            errors=errors or ["metadata is unavailable"],
            warnings=warnings,
            checked_files=checked,
        )
    assert metadata_bytes is not None

    if completion_bytes is not None:
        checked.append("completion.json")
        try:
            completion = RunCompletion.from_mapping(
                read_json_bytes_object(completion_bytes, source="completion.json")
            )
        except (
            MetadataValidationError,
            RunCreationError,
            TypeError,
            ValueError,
        ) as error:
            errors.append(f"completion marker is invalid: {error}")
        else:
            if completion.run_id != metadata.run_id:
                errors.append("completion marker run ID differs from metadata")
            if completion.metadata_sha256 != sha256_bytes(metadata_bytes):
                errors.append("completion marker metadata hash differs from metadata")

    _audit_timestamp(metadata.started_at, "started_at", errors)
    _audit_timestamp(metadata.finished_at, "finished_at", errors)
    if metadata.schema_version != 1:
        errors.append("unsupported run metadata schema version")
    if metadata.claim_scope != CLAIM_SCOPE:
        errors.append("metadata claim_scope is not the synthetic smoke scope")
    if metadata.model_checkpoint_sha256 is not None:
        errors.append("synthetic smoke metadata must not claim a model checkpoint")
    if metadata.git.commit is None:
        warnings.append(
            "run is not traceable to a Git commit; formal reporting is blocked"
        )
    if metadata.git.dirty:
        warnings.append("run was created from a dirty Git worktree")
    if metadata.event_keys.schema_version != EVENT_KEY_SCHEMA_VERSION:
        errors.append("unsupported event-key schema version")

    lock_file = metadata.environment.lock_file
    lock_hash = metadata.environment.lock_sha256
    environment_lock_present = False
    if lock_file is not None and lock_hash is not None:
        current_lock = trusted_root / lock_file
        if not current_lock.is_file():
            warnings.append(f"recorded environment lock is unavailable: {current_lock}")
        elif sha256_file(current_lock) != lock_hash:
            errors.append(f"recorded environment lock changed: {current_lock}")
        else:
            environment_lock_present = True
    else:
        warnings.append("run has no committed environment lock identity")

    live_git = collect_git_metadata(trusted_root)
    live_git_commit_matches = (
        metadata.git.commit is not None and live_git.commit == metadata.git.commit
    )
    live_git_clean = live_git.commit is not None and not live_git.dirty
    if metadata.git.commit is not None and not live_git_commit_matches:
        warnings.append(
            "live Git HEAD does not match the commit recorded by the run; "
            "release identity is blocked"
        )
    if not live_git_clean:
        warnings.append(
            "live Git checkout is dirty or has no HEAD; release identity is blocked"
        )

    declared_paths = {artifact.path for artifact in metadata.artifacts}
    required_artifacts = {
        "config.json",
        "data_manifest.json",
        "events.jsonl",
        "result.json",
    }
    if declared_paths != required_artifacts:
        errors.append("metadata artifact set is incomplete or contains extra entries")

    for artifact in metadata.artifacts:
        relative = Path(artifact.path)
        if (
            relative.is_absolute()
            or len(relative.parts) != 1
            or relative.name != artifact.path
        ):
            errors.append(f"unsafe artifact path: {artifact.path}")
            continue
        artifact_bytes = capture.files.get(artifact.path)
        if artifact_bytes is None:
            errors.append(f"declared artifact is missing: {artifact.path}")
            continue
        checked.append(artifact.path)
        if len(artifact_bytes) != artifact.size_bytes:
            errors.append(f"artifact size changed: {artifact.path}")
        if sha256_bytes(artifact_bytes) != artifact.sha256:
            errors.append(f"artifact hash changed: {artifact.path}")

    config: SmokeExperimentConfig | None = None
    manifest: DataManifest | None = None
    config_hash: str | None = None
    manifest_hash: str | None = None
    try:
        config_bytes = capture.files["config.json"]
        config = SmokeExperimentConfig.from_mapping(
            read_json_bytes_object(config_bytes, source="config.json")
        )
        config_hash = sha256_bytes(canonical_json_bytes(config.to_mapping()))
        if config_hash != metadata.config_sha256:
            errors.append("config semantic hash does not match metadata")
        if config_hash != metadata.config_source.semantic_sha256:
            errors.append("config source semantic hash does not match metadata")
    except (MetadataValidationError, KeyError, TypeError, ValueError) as error:
        errors.append(f"config snapshot is invalid: {error}")

    try:
        manifest_bytes = capture.files["data_manifest.json"]
        manifest = DataManifest.from_mapping(
            read_json_bytes_object(manifest_bytes, source="data_manifest.json")
        )
        manifest_hash = sha256_bytes(canonical_json_bytes(manifest.to_mapping()))
        if manifest_hash != metadata.data_manifest.semantic_sha256:
            errors.append("data manifest semantic hash does not match metadata")
        if manifest.manifest_id != metadata.data_manifest_id:
            errors.append("data manifest ID does not match metadata")
    except (MetadataValidationError, KeyError, TypeError, ValueError) as error:
        errors.append(f"data manifest snapshot is invalid: {error}")

    _audit_source(
        "config",
        metadata.config_source.path,
        metadata.config_source.source_sha256,
        trusted_root,
        warnings,
        errors,
    )
    _audit_source(
        "data manifest",
        metadata.data_manifest.path,
        metadata.data_manifest.source_sha256,
        trusted_root,
        warnings,
        errors,
    )

    events: tuple[dict[str, object], ...] = ()
    try:
        events = read_jsonl_bytes_objects(
            capture.files["events.jsonl"], source="events.jsonl"
        )
    except (MetadataValidationError, KeyError) as error:
        errors.append(f"event log is invalid: {error}")
    observed_keys: list[str] = []
    for expected_sequence, event in enumerate(events):
        if event.get("sequence") != expected_sequence:
            errors.append(f"event sequence mismatch at index {expected_sequence}")
        key = event.get("event_key")
        if not isinstance(key, str) or not key:
            errors.append(f"event key missing at index {expected_sequence}")
        else:
            observed_keys.append(key)
    try:
        observed_key_metadata = build_event_key_metadata(
            schema_version=metadata.event_keys.schema_version,
            root_key=metadata.event_keys.root_key,
            keys=observed_keys,
        )
        if observed_key_metadata.keys != metadata.event_keys.keys:
            errors.append("event keys in JSONL do not match metadata")
        if observed_key_metadata.sha256 != metadata.event_keys.sha256:
            errors.append("event-key digest does not match metadata")
    except MetadataValidationError as error:
        errors.append(f"observed event-key manifest is invalid: {error}")

    result: dict[str, object] | None = None
    try:
        result = read_json_bytes_object(
            capture.files["result.json"], source="result.json"
        )
        if result.get("run_id") != metadata.run_id:
            errors.append("result run_id does not match metadata")
        if result.get("experiment_id") != metadata.experiment_id:
            errors.append("result experiment_id does not match metadata")
        if result.get("claim_scope") != metadata.claim_scope:
            errors.append("result claim_scope does not match metadata")
        if result.get("empirical_status") != EMPIRICAL_STATUS:
            errors.append("result does not preserve the blocked empirical status")
    except (MetadataValidationError, KeyError) as error:
        errors.append(f"result artifact is invalid: {error}")

    if config is not None:
        expected_event_count = config.replicates * config.horizon_steps
        if len(events) != expected_event_count:
            errors.append("event count does not match validated config")
    if manifest is not None and manifest.record_count != 0:
        errors.append("smoke run manifest unexpectedly contains records")

    if (
        config is not None
        and manifest is not None
        and config_hash is not None
        and manifest_hash is not None
    ):
        expected_run_id, expected_root_key = derive_run_identity(
            config_sha256=config_hash,
            manifest_sha256=manifest_hash,
            experiment_id=config.experiment_id,
            seed=config.seed,
        )
        if metadata.run_id != expected_run_id:
            errors.append("run_id cannot be derived from the frozen inputs")
        if metadata.event_keys.root_key != expected_root_key:
            errors.append("event-key root cannot be derived from the frozen inputs")

        expected_result, expected_events, expected_keys = run_smoke_experiment(config)
        expected_event_mappings = tuple(event.to_mapping() for event in expected_events)
        if events != expected_event_mappings:
            errors.append("event log does not match deterministic replay")
        if result != expected_result.to_mapping(expected_run_id):
            errors.append("result artifact does not match deterministic replay")
        try:
            expected_key_metadata = build_event_key_metadata(
                schema_version=EVENT_KEY_SCHEMA_VERSION,
                root_key=expected_root_key,
                keys=expected_keys,
            )
            if metadata.event_keys != expected_key_metadata:
                errors.append("event-key metadata does not match deterministic replay")
        except MetadataValidationError as error:
            errors.append(f"expected event-key metadata is invalid: {error}")

    if not _captured_bundle_is_current(run_dir, capture):
        errors.append("run bundle changed while it was being audited")

    return _build_report(
        run_directory=run_dir,
        errors=errors,
        warnings=warnings,
        checked_files=checked,
        git_commit=metadata.git.commit,
        git_dirty=metadata.git.dirty,
        live_git_commit_matches=live_git_commit_matches,
        live_git_clean=live_git_clean,
        environment_lock_present=environment_lock_present,
    )


__all__ = ["AuditReport", "AuditStatus", "audit_run"]
