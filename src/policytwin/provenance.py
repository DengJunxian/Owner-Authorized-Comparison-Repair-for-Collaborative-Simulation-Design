"""Deterministic artifacts, run metadata, and structured JSONL logging."""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import re
import tempfile
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from types import TracebackType
from typing import NoReturn, Self, TextIO, cast

from policytwin import __version__
from policytwin.git_boundary import (
    GitBoundaryError,
    discover_repository_root,
    require_same_git_cli_identity,
    run_git_command,
    verify_git_cli_identity,
)

_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


class MetadataValidationError(ValueError):
    """Raised when metadata or another machine-readable artifact is invalid."""


def _fail(message: str) -> NoReturn:
    raise MetadataValidationError(message)


def _validate_json_value(value: object, path: str = "$") -> None:
    if value is None or isinstance(value, str | bool | int):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            _fail(f"non-finite float at {path}")
        return
    if isinstance(value, Mapping):
        for key, child in value.items():
            if not isinstance(key, str):
                _fail(f"non-string object key at {path}")
            _validate_json_value(child, f"{path}.{key}")
        return
    if isinstance(value, list | tuple):
        for index, child in enumerate(value):
            _validate_json_value(child, f"{path}[{index}]")
        return
    _fail(f"unsupported JSON value at {path}: {type(value).__name__}")


def canonical_json_bytes(value: object) -> bytes:
    """Serialize a JSON-compatible value deterministically for hashing."""
    _validate_json_value(value)
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    """Return the hexadecimal SHA-256 digest of bytes."""
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    """Hash a file without loading it entirely into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _strict_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            _fail(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> NoReturn:
    _fail(f"non-finite JSON number is forbidden: {value}")


def read_json_bytes_object(
    content: bytes, *, source: str = "<memory>"
) -> dict[str, object]:
    """Parse one strict UTF-8 JSON object from already committed bytes."""
    try:
        value = json.loads(
            content.decode("utf-8"),
            object_pairs_hook=_strict_object,
            parse_constant=_reject_constant,
        )
    except (UnicodeError, json.JSONDecodeError) as error:
        raise MetadataValidationError(
            f"cannot parse JSON object {source}: {error}"
        ) from error
    if not isinstance(value, dict):
        _fail(f"top-level JSON value in {source} must be an object")
    return cast(dict[str, object], value)


def read_json_object(path: Path) -> dict[str, object]:
    """Read a strict JSON object, rejecting duplicates and non-finite values."""
    try:
        content = path.read_bytes()
    except OSError as error:
        raise MetadataValidationError(
            f"cannot read JSON object {path}: {error}"
        ) from error
    return read_json_bytes_object(content, source=str(path))


def read_jsonl_bytes_objects(
    content: bytes, *, source: str = "<memory>"
) -> tuple[dict[str, object], ...]:
    """Parse strict JSONL from already committed immutable bytes."""
    try:
        lines = content.decode("utf-8").splitlines()
    except UnicodeError as error:
        raise MetadataValidationError(
            f"cannot decode JSONL {source}: {error}"
        ) from error
    if not lines:
        _fail(f"JSONL file is empty: {source}")
    events: list[dict[str, object]] = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            _fail(f"blank JSONL line at {source}:{line_number}")
        try:
            value = json.loads(
                line,
                object_pairs_hook=_strict_object,
                parse_constant=_reject_constant,
            )
        except json.JSONDecodeError as error:
            raise MetadataValidationError(
                f"invalid JSONL at {source}:{line_number}: {error}"
            ) from error
        if not isinstance(value, dict):
            _fail(f"JSONL value at {source}:{line_number} must be an object")
        events.append(cast(dict[str, object], value))
    return tuple(events)


def read_jsonl_objects(path: Path) -> tuple[dict[str, object], ...]:
    """Read strict JSONL, requiring one JSON object on every non-empty line."""
    try:
        content = path.read_bytes()
    except OSError as error:
        raise MetadataValidationError(f"cannot read JSONL {path}: {error}") from error
    return read_jsonl_bytes_objects(content, source=str(path))


def _pretty_json_bytes(value: object) -> bytes:
    _validate_json_value(value)
    rendered = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        indent=2,
        sort_keys=True,
    )
    return f"{rendered}\n".encode()


def _atomic_write_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=path.parent, prefix=f".{path.name}.", delete=False
        ) as stream:
            temporary_name = stream.name
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
        directory_flags = os.O_RDONLY
        directory_flag = getattr(os, "O_DIRECTORY", None)
        if isinstance(directory_flag, int):
            directory_flags |= directory_flag
        parent_fd = os.open(path.parent, directory_flags)
        try:
            os.fsync(parent_fd)
        finally:
            os.close(parent_fd)
    finally:
        if temporary_name is not None:
            temporary_path = Path(temporary_name)
            if temporary_path.exists():
                temporary_path.unlink()


def write_json_file(path: Path, value: object) -> None:
    """Atomically write deterministic, human-readable JSON."""
    _atomic_write_bytes(path, _pretty_json_bytes(value))


def write_text_file(path: Path, value: str) -> None:
    """Atomically write UTF-8 text."""
    _atomic_write_bytes(path, value.encode("utf-8"))


def utc_now_text() -> str:
    """Return a timezone-aware UTC timestamp for non-scientific metadata."""
    return datetime.now(UTC).isoformat()


def _fields(
    mapping: Mapping[str, object], expected: frozenset[str], context: str
) -> None:
    actual = frozenset(mapping)
    missing = sorted(expected - actual)
    unknown = sorted(actual - expected)
    if missing:
        _fail(f"{context} missing fields: {', '.join(missing)}")
    if unknown:
        _fail(f"{context} has unknown fields: {', '.join(unknown)}")


def _str(mapping: Mapping[str, object], field: str) -> str:
    value = mapping[field]
    if not isinstance(value, str) or not value:
        _fail(f"{field} must be a non-empty string")
    return value


def _optional_str(mapping: Mapping[str, object], field: str) -> str | None:
    value = mapping[field]
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        _fail(f"{field} must be null or a non-empty string")
    return value


def _int(mapping: Mapping[str, object], field: str, *, minimum: int = 0) -> int:
    value = mapping[field]
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        _fail(f"{field} must be an integer greater than or equal to {minimum}")
    return value


def _bool(mapping: Mapping[str, object], field: str) -> bool:
    value = mapping[field]
    if not isinstance(value, bool):
        _fail(f"{field} must be a boolean")
    return value


def _validate_sha256(value: object, field: str) -> str:
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        _fail(f"{field} must be a lowercase SHA-256 digest")
    return value


def _parse_aware_timestamp(value: object, field: str) -> datetime:
    if not isinstance(value, str) or not value:
        _fail(f"{field} must be a non-empty timestamp")
    try:
        timestamp = datetime.fromisoformat(value)
    except ValueError as error:
        raise MetadataValidationError(f"{field} must be ISO 8601") from error
    if timestamp.tzinfo is None or timestamp.utcoffset() is None:
        _fail(f"{field} must include a timezone")
    return timestamp


def _mapping(mapping: Mapping[str, object], field: str) -> Mapping[str, object]:
    value = mapping[field]
    if not isinstance(value, Mapping):
        _fail(f"{field} must be an object")
    if not all(isinstance(key, str) for key in value):
        _fail(f"{field} contains a non-string key")
    return cast(Mapping[str, object], value)


@dataclass(frozen=True, slots=True)
class SourceArtifact:
    """Original source path and its byte-level and semantic hashes."""

    path: str
    source_sha256: str
    semantic_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.path, str) or not self.path:
            _fail("source artifact path must be a non-empty string")
        _validate_sha256(self.source_sha256, "source_sha256")
        _validate_sha256(self.semantic_sha256, "semantic_sha256")

    def to_mapping(self) -> dict[str, object]:
        return {
            "path": self.path,
            "semantic_sha256": self.semantic_sha256,
            "source_sha256": self.source_sha256,
        }

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, object]) -> SourceArtifact:
        _fields(
            mapping,
            frozenset({"path", "source_sha256", "semantic_sha256"}),
            "source artifact",
        )
        return cls(
            path=_str(mapping, "path"),
            source_sha256=_str(mapping, "source_sha256"),
            semantic_sha256=_str(mapping, "semantic_sha256"),
        )


@dataclass(frozen=True, slots=True)
class ArtifactDigest:
    """Digest of one file inside a run bundle."""

    path: str
    sha256: str
    size_bytes: int

    def __post_init__(self) -> None:
        if (
            not isinstance(self.path, str)
            or not self.path
            or Path(self.path).is_absolute()
            or ".." in Path(self.path).parts
        ):
            _fail("artifact path must be a safe relative path")
        _validate_sha256(self.sha256, "artifact sha256")
        if isinstance(self.size_bytes, bool) or not isinstance(self.size_bytes, int):
            _fail("artifact size must be an integer")
        if self.size_bytes < 0:
            _fail("artifact size must not be negative")

    def to_mapping(self) -> dict[str, object]:
        return {
            "path": self.path,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
        }

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, object]) -> ArtifactDigest:
        _fields(
            mapping,
            frozenset({"path", "sha256", "size_bytes"}),
            "artifact digest",
        )
        return cls(
            path=_str(mapping, "path"),
            sha256=_str(mapping, "sha256"),
            size_bytes=_int(mapping, "size_bytes"),
        )


@dataclass(frozen=True, slots=True)
class GitMetadata:
    """Git identity recorded without inventing a commit for an empty repository."""

    commit: str | None
    dirty: bool
    root: str | None

    def __post_init__(self) -> None:
        if self.commit is not None and (
            not isinstance(self.commit, str)
            or re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", self.commit) is None
        ):
            _fail("git commit must be null or a full hexadecimal object ID")
        if not isinstance(self.dirty, bool):
            _fail("git dirty flag must be a boolean")
        if self.root is not None and (not isinstance(self.root, str) or not self.root):
            _fail("git root must be null or a non-empty string")

    def to_mapping(self) -> dict[str, object]:
        return {"commit": self.commit, "dirty": self.dirty, "root": self.root}

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, object]) -> GitMetadata:
        _fields(mapping, frozenset({"commit", "dirty", "root"}), "git metadata")
        return cls(
            commit=_optional_str(mapping, "commit"),
            dirty=_bool(mapping, "dirty"),
            root=_optional_str(mapping, "root"),
        )


@dataclass(frozen=True, slots=True)
class EnvironmentMetadata:
    """Interpreter, platform, package, and environment-lock identity."""

    python_version: str
    python_implementation: str
    platform: str
    machine: str
    policytwin_version: str
    lock_file: str | None
    lock_sha256: str | None

    def __post_init__(self) -> None:
        for field, value in (
            ("python_version", self.python_version),
            ("python_implementation", self.python_implementation),
            ("platform", self.platform),
            ("machine", self.machine),
            ("policytwin_version", self.policytwin_version),
        ):
            if not isinstance(value, str) or not value:
                _fail(f"{field} must be a non-empty string")
        if (self.lock_file is None) != (self.lock_sha256 is None):
            _fail("environment lock path and hash must be present together")
        if self.lock_file is not None and (
            not isinstance(self.lock_file, str)
            or not self.lock_file
            or Path(self.lock_file).is_absolute()
            or ".." in Path(self.lock_file).parts
        ):
            _fail("environment lock path must be safe and relative")
        if self.lock_sha256 is not None:
            _validate_sha256(self.lock_sha256, "environment lock sha256")

    def to_mapping(self) -> dict[str, object]:
        return {
            "lock_file": self.lock_file,
            "lock_sha256": self.lock_sha256,
            "machine": self.machine,
            "platform": self.platform,
            "policytwin_version": self.policytwin_version,
            "python_implementation": self.python_implementation,
            "python_version": self.python_version,
        }

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, object]) -> EnvironmentMetadata:
        _fields(
            mapping,
            frozenset(
                {
                    "python_version",
                    "python_implementation",
                    "platform",
                    "machine",
                    "policytwin_version",
                    "lock_file",
                    "lock_sha256",
                }
            ),
            "environment metadata",
        )
        return cls(
            python_version=_str(mapping, "python_version"),
            python_implementation=_str(mapping, "python_implementation"),
            platform=_str(mapping, "platform"),
            machine=_str(mapping, "machine"),
            policytwin_version=_str(mapping, "policytwin_version"),
            lock_file=_optional_str(mapping, "lock_file"),
            lock_sha256=_optional_str(mapping, "lock_sha256"),
        )


@dataclass(frozen=True, slots=True)
class EventKeyMetadata:
    """Complete event-key manifest for the deliberately small smoke run."""

    schema_version: str
    root_key: str
    count: int
    sha256: str
    keys: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.schema_version, str) or not self.schema_version:
            _fail("event-key schema version must be a non-empty string")
        if not isinstance(self.root_key, str) or not self.root_key:
            _fail("event-key root must be a non-empty string")
        if not isinstance(self.keys, tuple) or not all(
            isinstance(key, str) and key for key in self.keys
        ):
            _fail("event keys must be an immutable tuple of non-empty strings")
        if len(set(self.keys)) != len(self.keys):
            _fail("event keys must be unique")
        if isinstance(self.count, bool) or self.count != len(self.keys):
            _fail("event-key count does not match keys")
        expected_sha256 = sha256_bytes(canonical_json_bytes(list(self.keys)))
        if self.sha256 != expected_sha256:
            _fail("event-key digest does not match keys")

    def to_mapping(self) -> dict[str, object]:
        return {
            "count": self.count,
            "keys": list(self.keys),
            "root_key": self.root_key,
            "schema_version": self.schema_version,
            "sha256": self.sha256,
        }

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, object]) -> EventKeyMetadata:
        _fields(
            mapping,
            frozenset({"schema_version", "root_key", "count", "sha256", "keys"}),
            "event-key metadata",
        )
        raw_keys = mapping["keys"]
        if not isinstance(raw_keys, list) or not all(
            isinstance(key, str) and key for key in raw_keys
        ):
            _fail("event keys must be an array of non-empty strings")
        keys = tuple(cast(list[str], raw_keys))
        count = _int(mapping, "count")
        if count != len(keys):
            _fail("event-key count does not match keys")
        return cls(
            schema_version=_str(mapping, "schema_version"),
            root_key=_str(mapping, "root_key"),
            count=count,
            sha256=_str(mapping, "sha256"),
            keys=keys,
        )


@dataclass(frozen=True, slots=True)
class RunMetadata:
    """Unified machine-readable provenance for one PolicyTwin run."""

    schema_version: int
    run_id: str
    experiment_id: str
    claim_scope: str
    started_at: str
    finished_at: str
    git: GitMetadata
    config_sha256: str
    config_source: SourceArtifact
    data_manifest_id: str
    data_manifest: SourceArtifact
    model_checkpoint_sha256: str | None
    environment: EnvironmentMetadata
    event_keys: EventKeyMetadata
    artifacts: tuple[ArtifactDigest, ...]

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            _fail("unsupported run metadata schema version")
        for field, value in (
            ("run_id", self.run_id),
            ("experiment_id", self.experiment_id),
            ("claim_scope", self.claim_scope),
            ("data_manifest_id", self.data_manifest_id),
        ):
            if not isinstance(value, str) or not value:
                _fail(f"{field} must be a non-empty string")
        started_at = _parse_aware_timestamp(self.started_at, "started_at")
        finished_at = _parse_aware_timestamp(self.finished_at, "finished_at")
        if finished_at < started_at:
            _fail("finished_at must not precede started_at")
        _validate_sha256(self.config_sha256, "config_sha256")
        if self.config_source.semantic_sha256 != self.config_sha256:
            _fail("config source semantic hash must match config_sha256")
        if self.model_checkpoint_sha256 is not None:
            _validate_sha256(self.model_checkpoint_sha256, "model_checkpoint_sha256")
        if not isinstance(self.artifacts, tuple):
            _fail("artifacts must be an immutable tuple")
        paths = tuple(artifact.path for artifact in self.artifacts)
        if len(set(paths)) != len(paths):
            _fail("artifact paths must be unique")

    def to_mapping(self) -> dict[str, object]:
        return {
            "artifacts": [artifact.to_mapping() for artifact in self.artifacts],
            "claim_scope": self.claim_scope,
            "config_sha256": self.config_sha256,
            "config_source": self.config_source.to_mapping(),
            "data_manifest": self.data_manifest.to_mapping(),
            "data_manifest_id": self.data_manifest_id,
            "environment": self.environment.to_mapping(),
            "event_keys": self.event_keys.to_mapping(),
            "experiment_id": self.experiment_id,
            "finished_at": self.finished_at,
            "git": self.git.to_mapping(),
            "model_checkpoint_sha256": self.model_checkpoint_sha256,
            "run_id": self.run_id,
            "schema_version": self.schema_version,
            "started_at": self.started_at,
        }

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, object]) -> RunMetadata:
        _fields(
            mapping,
            frozenset(
                {
                    "schema_version",
                    "run_id",
                    "experiment_id",
                    "claim_scope",
                    "started_at",
                    "finished_at",
                    "git",
                    "config_sha256",
                    "config_source",
                    "data_manifest_id",
                    "data_manifest",
                    "model_checkpoint_sha256",
                    "environment",
                    "event_keys",
                    "artifacts",
                }
            ),
            "run metadata",
        )
        raw_artifacts = mapping["artifacts"]
        if not isinstance(raw_artifacts, list):
            _fail("artifacts must be an array")
        artifacts: list[ArtifactDigest] = []
        for raw_artifact in raw_artifacts:
            if not isinstance(raw_artifact, Mapping):
                _fail("each artifact must be an object")
            artifacts.append(
                ArtifactDigest.from_mapping(cast(Mapping[str, object], raw_artifact))
            )
        return cls(
            schema_version=_int(mapping, "schema_version", minimum=1),
            run_id=_str(mapping, "run_id"),
            experiment_id=_str(mapping, "experiment_id"),
            claim_scope=_str(mapping, "claim_scope"),
            started_at=_str(mapping, "started_at"),
            finished_at=_str(mapping, "finished_at"),
            git=GitMetadata.from_mapping(_mapping(mapping, "git")),
            config_sha256=_str(mapping, "config_sha256"),
            config_source=SourceArtifact.from_mapping(
                _mapping(mapping, "config_source")
            ),
            data_manifest_id=_str(mapping, "data_manifest_id"),
            data_manifest=SourceArtifact.from_mapping(
                _mapping(mapping, "data_manifest")
            ),
            model_checkpoint_sha256=_optional_str(mapping, "model_checkpoint_sha256"),
            environment=EnvironmentMetadata.from_mapping(
                _mapping(mapping, "environment")
            ),
            event_keys=EventKeyMetadata.from_mapping(_mapping(mapping, "event_keys")),
            artifacts=tuple(artifacts),
        )


def collect_git_metadata(start: Path) -> GitMetadata:
    """Collect Git state through the fixed, non-inherited Git boundary."""
    try:
        cli_identity = verify_git_cli_identity()
        root_path = discover_repository_root(start)
    except GitBoundaryError as error:
        raise MetadataValidationError(f"Git trust boundary failed: {error}") from error
    if root_path is None:
        return GitMetadata(commit=None, dirty=True, root=None)
    try:
        replace_result = run_git_command(
            root_path,
            ("for-each-ref", "--format=%(refname)", "refs/replace/"),
            expected_cli_identity=cli_identity,
        )
        root_result = run_git_command(
            root_path,
            ("rev-parse", "--show-toplevel"),
            expected_cli_identity=cli_identity,
        )
        commit_result = run_git_command(
            root_path,
            ("rev-parse", "--verify", "HEAD^{commit}"),
            expected_cli_identity=cli_identity,
        )
    except GitBoundaryError as error:
        raise MetadataValidationError(f"Git trust boundary failed: {error}") from error
    if replace_result.returncode != 0 or replace_result.stdout.strip():
        _fail("Git replacement refs are forbidden")
    try:
        reported_root = root_result.stdout.decode("utf-8", errors="strict").strip()
    except UnicodeError as error:
        raise MetadataValidationError("Git repository root is not UTF-8") from error
    if root_result.returncode != 0 or Path(reported_root).resolve() != root_path:
        _fail("Git-reported repository root differs from the filesystem trust root")
    commit: str | None = None
    if commit_result.returncode == 0:
        try:
            candidate = commit_result.stdout.decode("ascii", errors="strict").strip()
        except UnicodeError as error:
            raise MetadataValidationError("Git commit ID is not ASCII") from error
        if re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", candidate) is None:
            _fail("Git returned an invalid commit ID")
        commit = candidate
        try:
            fsck_result = run_git_command(
                root_path,
                (
                    "fsck",
                    "--full",
                    "--strict",
                    "--no-reflogs",
                    "--no-dangling",
                    commit,
                ),
                expected_cli_identity=cli_identity,
            )
        except GitBoundaryError as error:
            raise MetadataValidationError(
                f"Git trust boundary failed: {error}"
            ) from error
        if fsck_result.returncode != 0:
            _fail("Git object database failed strict content verification")
    dirty = commit is None
    if commit is not None:
        try:
            working = run_git_command(
                root_path,
                (
                    "diff",
                    "--no-ext-diff",
                    "--no-textconv",
                    "--quiet",
                    "HEAD",
                    "--",
                ),
                expected_cli_identity=cli_identity,
            )
            staged = run_git_command(
                root_path,
                (
                    "diff",
                    "--cached",
                    "--no-ext-diff",
                    "--no-textconv",
                    "--quiet",
                    "HEAD",
                    "--",
                ),
                expected_cli_identity=cli_identity,
            )
            untracked = run_git_command(
                root_path,
                ("ls-files", "--others", "--exclude-standard", "-z"),
                expected_cli_identity=cli_identity,
            )
        except GitBoundaryError as error:
            raise MetadataValidationError(
                f"Git trust boundary failed: {error}"
            ) from error
        if working.returncode not in {0, 1} or staged.returncode not in {0, 1}:
            _fail("cannot determine tracked Git state")
        if untracked.returncode != 0:
            _fail("cannot determine untracked Git state")
        dirty = (
            working.returncode == 1 or staged.returncode == 1 or bool(untracked.stdout)
        )
        try:
            final_fsck = run_git_command(
                root_path,
                (
                    "fsck",
                    "--full",
                    "--strict",
                    "--no-reflogs",
                    "--no-dangling",
                    commit,
                ),
                expected_cli_identity=cli_identity,
            )
            require_same_git_cli_identity(
                cli_identity, context="after Git metadata collection"
            )
        except GitBoundaryError as error:
            raise MetadataValidationError(
                f"Git trust boundary failed: {error}"
            ) from error
        if final_fsck.returncode != 0:
            _fail("Git object database changed during metadata collection")
    return GitMetadata(commit=commit, dirty=dirty, root=str(root_path))


def collect_environment_metadata(project_root: Path) -> EnvironmentMetadata:
    """Collect interpreter and environment-lock identity."""
    lock_path = project_root / "uv.lock"
    return EnvironmentMetadata(
        python_version=platform.python_version(),
        python_implementation=platform.python_implementation(),
        platform=platform.system(),
        machine=platform.machine(),
        policytwin_version=__version__,
        lock_file="uv.lock" if lock_path.is_file() else None,
        lock_sha256=sha256_file(lock_path) if lock_path.is_file() else None,
    )


def build_event_key_metadata(
    *, schema_version: str, root_key: str, keys: Iterable[str]
) -> EventKeyMetadata:
    """Build a deterministic digest and explicit key list for a smoke run."""
    key_tuple = tuple(keys)
    return EventKeyMetadata(
        schema_version=schema_version,
        root_key=root_key,
        count=len(key_tuple),
        sha256=sha256_bytes(canonical_json_bytes(list(key_tuple))),
        keys=key_tuple,
    )


def digest_artifact(run_directory: Path, relative_path: str) -> ArtifactDigest:
    """Describe a file inside a run bundle."""
    path = run_directory / relative_path
    return ArtifactDigest(
        path=relative_path,
        sha256=sha256_file(path),
        size_bytes=path.stat().st_size,
    )


class JsonlEventWriter:
    """Write canonical JSON objects as deterministic JSONL events."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._stream: TextIO | None = None

    def __enter__(self) -> Self:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._stream = self._path.open("x", encoding="utf-8", newline="\n")
        return self

    def write(self, event: Mapping[str, object]) -> None:
        """Append one event without adding wall-clock or process-derived fields."""
        if self._stream is None:
            _fail("JSONL writer is not open")
        rendered = canonical_json_bytes(event).decode("utf-8")
        self._stream.write(f"{rendered}\n")

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        if self._stream is not None:
            self._stream.flush()
            os.fsync(self._stream.fileno())
            self._stream.close()
            self._stream = None


__all__ = [
    "ArtifactDigest",
    "EnvironmentMetadata",
    "EventKeyMetadata",
    "GitMetadata",
    "JsonlEventWriter",
    "MetadataValidationError",
    "RunMetadata",
    "SourceArtifact",
    "build_event_key_metadata",
    "canonical_json_bytes",
    "collect_environment_metadata",
    "collect_git_metadata",
    "digest_artifact",
    "read_json_bytes_object",
    "read_json_object",
    "read_jsonl_bytes_objects",
    "read_jsonl_objects",
    "sha256_bytes",
    "sha256_file",
    "utc_now_text",
    "write_json_file",
    "write_text_file",
]
