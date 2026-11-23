"""Fail-closed OS network isolation for frozen evaluation subprocesses.

The Python temporal-firewall guard remains useful defense in depth, but it is
not the trust boundary implemented here.  This module launches the complete
evaluation command under either a macOS sandbox policy or Docker's network-none
namespace, probes that boundary *inside* it, and writes a sealed receipt.
"""

from __future__ import annotations

import argparse
import errno
import hashlib
import json
import os
import platform
import re
import shutil
import socket
import stat
import subprocess
import sys
import tempfile
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path, PurePosixPath
from typing import NoReturn, Self, cast

from policytwin.git_boundary import GitBoundaryError, GitCliIdentity
from policytwin.provenance import (
    MetadataValidationError,
    canonical_json_bytes,
    read_json_bytes_object,
    sha256_bytes,
    sha256_file,
)
from policytwin.release_preflight import (
    SourcePreflightError,
    SourceSnapshotManifest,
    export_source_snapshot,
    extract_source_snapshot,
    read_source_snapshot_manifest,
    run_source_preflight,
    verify_source_snapshot_directory,
)

ISOLATION_RECEIPT_SCHEMA_VERSION = 6
DEFAULT_RECEIPT_NAME = "os-network-isolation.json"
MACOS_SANDBOX_PROFILE = "(version 1)\n(allow default)\n(deny network*)\n"
MACOS_SANDBOX_RUNTIME_IDENTITY = (
    "macos-sandbox-profile-v1@sha256:"
    f"{sha256_bytes(MACOS_SANDBOX_PROFILE.encode('utf-8'))}"
)
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_IMMUTABLE_IMAGE_PATTERN = re.compile(
    r"^(?:[a-z0-9][a-z0-9._/-]*@)?sha256:[0-9a-f]{64}$"
)
_IMAGE_ID_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")
_RECEIPT_NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_WORKER_MARKER = "POLICYTWIN_OS_NETWORK_ISOLATION"
_MACOS_UNIX_CANARY_ENV = "POLICYTWIN_MACOS_UNIX_CANARY"
_SANDBOX_MARKER = "macos-sandbox-deny-network-v1"
_DOCKER_MARKER = "docker-network-none-v1"
_DOCKER_BOOTSTRAP_SOURCE = r"""import hashlib,json,os,stat,sys
from pathlib import Path,PurePosixPath

def fail(message):
    print("POLICYTWIN_DOCKER_BOOTSTRAP: " + message, file=sys.stderr)
    raise SystemExit(124)

def pairs(items):
    result = {}
    for key, value in items:
        if not isinstance(key, str) or key in result:
            fail("manifest contains a duplicate or non-string key")
        result[key] = value
    return result

def safe_path(value):
    if not isinstance(value, str) or not value or "\\" in value:
        return False
    if any(ord(character) < 32 for character in value):
        return False
    path = PurePosixPath(value)
    return (not path.is_absolute() and path.as_posix() == value and
            all(part not in ("", ".", "..", ".git") for part in path.parts))

def canonical(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False,
                      sort_keys=True, separators=(",", ":")).encode("utf-8")

arguments = sys.argv[1:]
try:
    separator = arguments.index("--")
except ValueError:
    fail("target separator is missing")
bootstrap = arguments[:separator]
target = arguments[separator + 1:]
if len(bootstrap) != 6 or not target:
    fail("bootstrap arguments are malformed")
(manifest_path, source_root, manifest_sha256, tree_sha256,
 file_count, bootstrap_sha256) = bootstrap
if (len(bootstrap_sha256) != 64 or
    any(character not in "0123456789abcdef" for character in bootstrap_sha256)):
    fail("bootstrap identity is malformed")
try:
    raw = Path(manifest_path).read_bytes()
except OSError:
    fail("source manifest is unavailable")
if hashlib.sha256(raw).hexdigest() != manifest_sha256:
    fail("source manifest hash mismatch")
try:
    manifest = json.loads(raw, object_pairs_hook=pairs)
except (UnicodeError, json.JSONDecodeError, ValueError):
    fail("source manifest is not strict JSON")
if canonical(manifest) != raw or not isinstance(manifest, dict):
    fail("source manifest is not canonical JSON")
required = {"commit", "file_count", "files", "git_object_format",
            "schema_version", "total_size_bytes", "tree_sha256"}
if (set(manifest) != required or
    manifest.get("schema_version") != "policytwin.source-snapshot.v1"):
    fail("source manifest schema differs")
files = manifest.get("files")
if not isinstance(files, list) or manifest.get("file_count") != len(files):
    fail("source manifest file count differs")
if str(len(files)) != file_count or manifest.get("tree_sha256") != tree_sha256:
    fail("source identity differs from manifest")
if hashlib.sha256(canonical(files)).hexdigest() != tree_sha256:
    fail("source tree commitment is invalid")
expected = {}
expected_directories = set()
total_size = 0
previous = None
for entry in files:
    if (not isinstance(entry, dict) or
        set(entry) != {"mode", "path", "sha256", "size_bytes"}):
        fail("source file schema differs")
    path = entry.get("path")
    mode = entry.get("mode")
    size = entry.get("size_bytes")
    digest = entry.get("sha256")
    if (not safe_path(path) or mode not in ("100644", "100755") or
        isinstance(size, bool) or not isinstance(size, int) or size < 0 or
        not isinstance(digest, str) or len(digest) != 64 or
        any(character not in "0123456789abcdef" for character in digest)):
        fail("source file identity is invalid")
    if previous is not None and path <= previous:
        fail("source files are not strictly sorted")
    previous = path
    expected[path] = entry
    total_size += size
    parent = PurePosixPath(path).parent
    while parent.as_posix() != ".":
        expected_directories.add(parent.as_posix())
        parent = parent.parent
if manifest.get("total_size_bytes") != total_size:
    fail("source total size differs")
root = Path(source_root)
try:
    root_info = os.lstat(root)
except OSError:
    fail("private source root is unavailable")
if not stat.S_ISDIR(root_info.st_mode) or stat.S_ISLNK(root_info.st_mode):
    fail("private source root is not a real directory")
actual = {}
actual_directories = set()
try:
    for current, directory_names, file_names in os.walk(root, followlinks=False):
        current_path = Path(current)
        for name in directory_names:
            candidate = current_path / name
            info = os.lstat(candidate)
            relative = candidate.relative_to(root).as_posix()
            if (not stat.S_ISDIR(info.st_mode) or stat.S_ISLNK(info.st_mode) or
                not safe_path(relative) or relative in actual_directories):
                fail("private source contains an unsafe directory")
            actual_directories.add(relative)
        for name in file_names:
            candidate = current_path / name
            info = os.lstat(candidate)
            relative = candidate.relative_to(root).as_posix()
            if (not stat.S_ISREG(info.st_mode) or stat.S_ISLNK(info.st_mode) or
                not safe_path(relative) or relative in actual):
                fail("private source contains a special, linked, or duplicate file")
            actual[relative] = candidate
except OSError:
    fail("private source traversal failed")
if set(actual) != set(expected) or actual_directories != expected_directories:
    fail("private source path set differs from manifest")
for relative, entry in expected.items():
    path = actual[relative]
    info = os.lstat(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    executable = bool(info.st_mode & 0o111)
    if (info.st_size != entry["size_bytes"] or digest != entry["sha256"] or
        executable != (entry["mode"] == "100755") or bool(info.st_mode & 0o7000)):
        fail("private source bytes or mode differ from manifest")
environment = {key: value for key, value in os.environ.items()
               if key in {"POLICYTWIN_OS_NETWORK_ISOLATION", "HOME", "LANG", "PATH",
                          "PYTHONDONTWRITEBYTECODE", "PYTHONHASHSEED",
                          "TMPDIR", "TZ"}}
environment["PYTHONPATH"] = "/workspace/src"
os.execve(target[0], target, environment)
"""
DOCKER_BOOTSTRAP_SHA256 = sha256_bytes(_DOCKER_BOOTSTRAP_SOURCE.encode("utf-8"))
_MAX_DOCKER_OUTPUT_FILES = 100_000
_MAX_DOCKER_OUTPUT_DIRECTORIES = 100_000
_MAX_DOCKER_OUTPUT_BYTES = 1024 * 1024 * 1024
_OUTPUT_TREE_SCHEMA_VERSION = "policytwin.isolation-output-tree.v1"

_DOCKER_CONTROLS = (
    "--network=none",
    "--pull=never",
    "--read-only",
    "--cap-drop=ALL",
    "--privileged=false",
    "--ipc=private",
    "--security-opt=no-new-privileges:true",
    "--pids-limit=128",
    "no-host-bind-mounts",
    "source-copied-to-private-container-layer-before-start",
    "private-source-under-read-only-rootfs",
    "job-unique-docker-output-volume",
    "post-stop-docker-cp-export",
    f"trusted-stdlib-bootstrap@sha256:{DOCKER_BOOTSTRAP_SHA256}",
    "tmpfs-noexec-nosuid-nodev",
    "close-inherited-file-descriptors",
    "working-directory=/workspace",
    "no-added-capabilities-or-devices",
    "private-pid-and-ipc-namespaces",
    f"output-export-max-files={_MAX_DOCKER_OUTPUT_FILES}",
    f"output-export-max-directories={_MAX_DOCKER_OUTPUT_DIRECTORIES}",
    f"output-export-max-bytes={_MAX_DOCKER_OUTPUT_BYTES}",
)
_MAX_RECEIPT_BYTES = 64 * 1024
_SOURCE_MANIFEST_SUFFIX = ".source-manifest.json"
_DOCKER_RUNTIME_SUFFIX = ".docker-runtime.json"
_MACOS_RUNTIME_SUFFIX = ".macos-runtime.json"
_DOCKER_SOURCE_MANIFEST_PATH = "/policytwin-source-manifest.json"
_DOCKER_SOURCE_ROOT = "/workspace"
_DOCKER_FINAL_WORKDIR = "/workspace"
_DOCKER_OUTPUT_ROOT = "/output"
_DOCKER_FINAL_UID = 65532
_DOCKER_FINAL_GID = 65532
_DOCKER_FINAL_USER = f"{_DOCKER_FINAL_UID}:{_DOCKER_FINAL_GID}"
_DOCKER_DAEMON_SOCKET = Path("/var/run/docker.sock")
_DOCKER_DAEMON_ENDPOINT = "unix:///var/run/docker.sock"
_DOCKER_CLI_PATH = Path("/usr/bin/docker")
_MACOS_SANDBOX_EXEC_PATH = Path("/usr/bin/sandbox-exec")
_MACOS_ENV_PATH = Path("/usr/bin/env")
_DOCKER_AMBIENT_OVERRIDES = (
    "DOCKER_CERT_PATH",
    "DOCKER_CONFIG",
    "DOCKER_CONTEXT",
    "DOCKER_HOST",
    "DOCKER_TLS",
    "DOCKER_TLS_VERIFY",
)
_OBJECT_ID_PATTERN = re.compile(r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")
_DOCKER_ID_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_DOCKER_VOLUME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")


class IsolationError(RuntimeError):
    """Base error for the frozen evaluation isolation boundary."""


class IsolationUnavailableError(IsolationError):
    """Raised when the explicitly selected OS isolation backend is unavailable."""


class IsolationIntegrityError(IsolationError):
    """Raised when an isolation receipt is malformed or has been modified."""


class IsolationExecutionError(IsolationError):
    """Raised when the isolation launcher cannot produce an auditable receipt."""


class IsolationBackend(StrEnum):
    """Supported operating-system isolation mechanisms."""

    MACOS_SANDBOX = "macos_sandbox"
    DOCKER_NONE = "docker_none"


class IsolationJobStatus(StrEnum):
    """Terminal state of one isolated command."""

    COMPLETED = "completed"
    INPUT_FAILED = "input_failed"
    PROBE_FAILED = "probe_failed"
    TARGET_FAILED = "target_failed"


class IsolationInputMode(StrEnum):
    """How source bytes enter an isolated evaluation job."""

    LIVE_CHECKOUT_NETWORK_ONLY = "live_checkout_network_only"
    COMMITTED_SOURCE_SNAPSHOT = "committed_source_snapshot"


def _fail(error_type: type[IsolationError], message: str) -> NoReturn:
    raise error_type(message)


def _text(value: object, name: str, *, maximum: int = 256) -> str:
    if not isinstance(value, str) or not value or len(value) > maximum:
        _fail(IsolationIntegrityError, f"{name} must be a non-empty bounded string")
    if "\x00" in value:
        _fail(IsolationIntegrityError, f"{name} must not contain NUL")
    return value


def _sha256(value: object, name: str) -> str:
    text = _text(value, name, maximum=64)
    if not _SHA256_PATTERN.fullmatch(text):
        _fail(IsolationIntegrityError, f"{name} must be a lowercase SHA-256 digest")
    return text


def _strict_fields(
    value: Mapping[str, object], expected: frozenset[str], name: str
) -> None:
    fields = frozenset(value)
    if fields != expected:
        missing = sorted(expected - fields)
        extra = sorted(fields - expected)
        _fail(
            IsolationIntegrityError,
            f"{name} fields differ from the closed schema; "
            f"missing={missing}, extra={extra}",
        )


def _enum_value(enum_type: type[StrEnum], value: object, name: str) -> StrEnum:
    if not isinstance(value, str):
        _fail(IsolationIntegrityError, f"{name} must be a string")
    try:
        return enum_type(value)
    except ValueError as error:
        raise IsolationIntegrityError(f"unsupported {name}: {value}") from error


def _timestamp(value: object, name: str) -> str:
    text = _text(value, name, maximum=64)
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError as error:
        raise IsolationIntegrityError(f"{name} must be ISO-8601") from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        _fail(IsolationIntegrityError, f"{name} must be timezone-aware")
    return text


@dataclass(frozen=True)
class IsolationProbeEvidence:
    """Machine-readable evidence collected inside the selected OS boundary."""

    passed: bool
    operation: str
    errno_name: str
    interfaces: tuple[str, ...]
    non_loopback_route_count: int

    def __post_init__(self) -> None:
        if not isinstance(self.passed, bool):
            _fail(IsolationIntegrityError, "probe passed must be a boolean")
        _text(self.operation, "probe operation", maximum=128)
        _text(self.errno_name, "probe errno_name", maximum=64)
        if (
            not isinstance(self.interfaces, tuple)
            or not all(isinstance(item, str) and item for item in self.interfaces)
            or tuple(sorted(set(self.interfaces))) != self.interfaces
        ):
            _fail(
                IsolationIntegrityError,
                "probe interfaces must be a sorted immutable unique tuple",
            )
        if (
            isinstance(self.non_loopback_route_count, bool)
            or not isinstance(self.non_loopback_route_count, int)
            or self.non_loopback_route_count < 0
        ):
            _fail(
                IsolationIntegrityError,
                "probe non_loopback_route_count must be a non-negative integer",
            )

    def to_mapping(self) -> dict[str, object]:
        """Return the canonical JSON representation."""
        return {
            "errno_name": self.errno_name,
            "interfaces": list(self.interfaces),
            "non_loopback_route_count": self.non_loopback_route_count,
            "operation": self.operation,
            "passed": self.passed,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> Self:
        """Strictly parse probe evidence."""
        _strict_fields(
            value,
            frozenset(
                {
                    "errno_name",
                    "interfaces",
                    "non_loopback_route_count",
                    "operation",
                    "passed",
                }
            ),
            "isolation probe",
        )
        raw_interfaces = value["interfaces"]
        if not isinstance(raw_interfaces, list) or not all(
            isinstance(item, str) for item in raw_interfaces
        ):
            _fail(IsolationIntegrityError, "probe interfaces must be a string list")
        passed = value["passed"]
        route_count = value["non_loopback_route_count"]
        if not isinstance(passed, bool):
            _fail(IsolationIntegrityError, "probe passed must be a boolean")
        if isinstance(route_count, bool) or not isinstance(route_count, int):
            _fail(IsolationIntegrityError, "probe route count must be an integer")
        return cls(
            passed=passed,
            operation=_text(value["operation"], "probe operation", maximum=128),
            errno_name=_text(value["errno_name"], "probe errno_name", maximum=64),
            interfaces=tuple(cast(list[str], raw_interfaces)),
            non_loopback_route_count=route_count,
        )


@dataclass(frozen=True)
class IsolationSourceIdentity:
    """Closed identity of the bytes visible at the evaluation source root."""

    mode: IsolationInputMode
    commit: str | None
    tree_sha256: str | None
    manifest_sha256: str | None
    file_count: int | None
    archive_sha256: str | None
    archive_size_bytes: int | None
    bootstrap_sha256: str | None
    git_cli_identity: GitCliIdentity | None

    def __post_init__(self) -> None:
        if not isinstance(self.mode, IsolationInputMode):
            _fail(IsolationIntegrityError, "source identity mode is invalid")
        if self.mode is IsolationInputMode.LIVE_CHECKOUT_NETWORK_ONLY:
            if any(
                value is not None
                for value in (
                    self.commit,
                    self.tree_sha256,
                    self.manifest_sha256,
                    self.file_count,
                    self.archive_sha256,
                    self.archive_size_bytes,
                    self.bootstrap_sha256,
                    self.git_cli_identity,
                )
            ):
                _fail(
                    IsolationIntegrityError,
                    "live-checkout source identity cannot claim committed provenance",
                )
            return
        if (
            not isinstance(self.commit, str)
            or _OBJECT_ID_PATTERN.fullmatch(self.commit) is None
        ):
            _fail(IsolationIntegrityError, "snapshot commit identity is invalid")
        _sha256(self.tree_sha256, "source tree_sha256")
        _sha256(self.manifest_sha256, "source manifest_sha256")
        _sha256(self.archive_sha256, "source archive_sha256")
        _sha256(self.bootstrap_sha256, "source bootstrap_sha256")
        if (
            isinstance(self.file_count, bool)
            or not isinstance(self.file_count, int)
            or self.file_count <= 0
        ):
            _fail(
                IsolationIntegrityError,
                "snapshot file_count must be a positive integer",
            )
        if (
            isinstance(self.archive_size_bytes, bool)
            or not isinstance(self.archive_size_bytes, int)
            or self.archive_size_bytes <= 0
        ):
            _fail(
                IsolationIntegrityError,
                "snapshot archive_size_bytes must be a positive integer",
            )
        if self.bootstrap_sha256 != DOCKER_BOOTSTRAP_SHA256:
            _fail(
                IsolationIntegrityError,
                "snapshot bootstrap identity differs from the fixed bootstrap",
            )
        if not isinstance(self.git_cli_identity, GitCliIdentity):
            _fail(
                IsolationIntegrityError,
                "snapshot Git CLI identity is missing or invalid",
            )

    def to_mapping(self) -> dict[str, object]:
        """Return the canonical source identity."""
        return {
            "archive_sha256": self.archive_sha256,
            "archive_size_bytes": self.archive_size_bytes,
            "bootstrap_sha256": self.bootstrap_sha256,
            "commit": self.commit,
            "file_count": self.file_count,
            "git_cli_identity": (
                None
                if self.git_cli_identity is None
                else self.git_cli_identity.to_mapping()
            ),
            "manifest_sha256": self.manifest_sha256,
            "mode": self.mode.value,
            "tree_sha256": self.tree_sha256,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> Self:
        """Strictly parse an isolation source identity."""
        _strict_fields(
            value,
            frozenset(
                {
                    "archive_sha256",
                    "archive_size_bytes",
                    "bootstrap_sha256",
                    "commit",
                    "file_count",
                    "git_cli_identity",
                    "manifest_sha256",
                    "mode",
                    "tree_sha256",
                }
            ),
            "isolation source identity",
        )
        commit = value["commit"]
        tree_sha256 = value["tree_sha256"]
        manifest_sha256 = value["manifest_sha256"]
        file_count = value["file_count"]
        archive_size_bytes = value["archive_size_bytes"]
        for raw, name in (
            (value["archive_sha256"], "source archive_sha256"),
            (value["bootstrap_sha256"], "source bootstrap_sha256"),
            (commit, "source commit"),
            (tree_sha256, "source tree_sha256"),
            (manifest_sha256, "source manifest_sha256"),
        ):
            if raw is not None and not isinstance(raw, str):
                _fail(IsolationIntegrityError, f"{name} must be a string or null")
        if file_count is not None and (
            isinstance(file_count, bool) or not isinstance(file_count, int)
        ):
            _fail(
                IsolationIntegrityError, "source file_count must be an integer or null"
            )
        if archive_size_bytes is not None and (
            isinstance(archive_size_bytes, bool)
            or not isinstance(archive_size_bytes, int)
        ):
            _fail(
                IsolationIntegrityError,
                "source archive_size_bytes must be an integer or null",
            )
        raw_git_cli_identity = value["git_cli_identity"]
        if raw_git_cli_identity is None:
            git_cli_identity = None
        else:
            try:
                git_cli_identity = GitCliIdentity.from_mapping(raw_git_cli_identity)
            except GitBoundaryError as error:
                raise IsolationIntegrityError(
                    "source Git CLI identity is invalid"
                ) from error
        return cls(
            mode=cast(
                IsolationInputMode,
                _enum_value(IsolationInputMode, value["mode"], "source input mode"),
            ),
            commit=cast(str | None, commit),
            tree_sha256=cast(str | None, tree_sha256),
            manifest_sha256=cast(str | None, manifest_sha256),
            file_count=file_count,
            archive_sha256=cast(str | None, value["archive_sha256"]),
            archive_size_bytes=archive_size_bytes,
            bootstrap_sha256=cast(str | None, value["bootstrap_sha256"]),
            git_cli_identity=git_cli_identity,
        )


@dataclass(frozen=True)
class DockerSourceSnapshot:
    """Prepared committed source tree copied into a stopped container layer."""

    root: Path
    archive_path: Path
    archive_sha256: str
    archive_size_bytes: int
    manifest_path: Path
    manifest: SourceSnapshotManifest
    git_cli_identity: GitCliIdentity
    identity: IsolationSourceIdentity


@dataclass(frozen=True)
class DockerCliIdentity:
    """Pinned host Docker CLI identity used for one local lifecycle."""

    path: str
    size_bytes: int
    sha256: str

    def __post_init__(self) -> None:
        if self.path != str(_DOCKER_CLI_PATH):
            _fail(IsolationIntegrityError, "Docker CLI identity path is not fixed")
        if (
            isinstance(self.size_bytes, bool)
            or not isinstance(self.size_bytes, int)
            or self.size_bytes <= 0
        ):
            _fail(IsolationIntegrityError, "Docker CLI size must be positive")
        _sha256(self.sha256, "Docker CLI sha256")


@dataclass(frozen=True)
class MacosSandboxIdentity:
    """Pinned host ``sandbox-exec`` identity used for one local lifecycle."""

    path: str
    size_bytes: int
    sha256: str
    environment_launcher_path: str
    environment_launcher_size_bytes: int
    environment_launcher_sha256: str

    def __post_init__(self) -> None:
        if self.path != str(_MACOS_SANDBOX_EXEC_PATH):
            _fail(IsolationIntegrityError, "macOS sandbox identity path is not fixed")
        if (
            isinstance(self.size_bytes, bool)
            or not isinstance(self.size_bytes, int)
            or self.size_bytes <= 0
        ):
            _fail(IsolationIntegrityError, "macOS sandbox executable size is invalid")
        _sha256(self.sha256, "macOS sandbox executable sha256")
        if self.environment_launcher_path != str(_MACOS_ENV_PATH):
            _fail(
                IsolationIntegrityError,
                "macOS environment launcher identity path is not fixed",
            )
        if (
            isinstance(self.environment_launcher_size_bytes, bool)
            or not isinstance(self.environment_launcher_size_bytes, int)
            or self.environment_launcher_size_bytes <= 0
        ):
            _fail(
                IsolationIntegrityError,
                "macOS environment launcher executable size is invalid",
            )
        _sha256(
            self.environment_launcher_sha256,
            "macOS environment launcher executable sha256",
        )


@dataclass(frozen=True)
class IsolationOutputTreeIdentity:
    """Semantic identity of every non-companion output below a receipt root."""

    tree_sha256: str
    file_count: int
    directory_count: int
    total_size_bytes: int

    def __post_init__(self) -> None:
        _sha256(self.tree_sha256, "output tree_sha256")
        for value, name in (
            (self.file_count, "output file_count"),
            (self.directory_count, "output directory_count"),
            (self.total_size_bytes, "output total_size_bytes"),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                _fail(
                    IsolationIntegrityError,
                    f"{name} must be a non-negative integer",
                )
        if self.file_count > _MAX_DOCKER_OUTPUT_FILES:
            _fail(IsolationIntegrityError, "output file_count exceeds safety limit")
        if self.directory_count > _MAX_DOCKER_OUTPUT_DIRECTORIES:
            _fail(
                IsolationIntegrityError,
                "output directory_count exceeds safety limit",
            )
        if self.total_size_bytes > _MAX_DOCKER_OUTPUT_BYTES:
            _fail(
                IsolationIntegrityError,
                "output total_size_bytes exceeds safety limit",
            )

    def to_mapping(self) -> dict[str, object]:
        """Return the closed canonical output-tree summary."""
        return {
            "directory_count": self.directory_count,
            "file_count": self.file_count,
            "total_size_bytes": self.total_size_bytes,
            "tree_sha256": self.tree_sha256,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> Self:
        """Strictly parse one output-tree identity."""
        _strict_fields(
            value,
            frozenset(
                {
                    "directory_count",
                    "file_count",
                    "total_size_bytes",
                    "tree_sha256",
                }
            ),
            "isolation output tree",
        )
        counts: dict[str, int] = {}
        for name in ("file_count", "directory_count", "total_size_bytes"):
            raw = value[name]
            if isinstance(raw, bool) or not isinstance(raw, int):
                _fail(IsolationIntegrityError, f"output {name} must be an integer")
            counts[name] = raw
        return cls(
            tree_sha256=_sha256(value["tree_sha256"], "output tree_sha256"),
            file_count=counts["file_count"],
            directory_count=counts["directory_count"],
            total_size_bytes=counts["total_size_bytes"],
        )


@dataclass(frozen=True)
class IsolationReceipt:
    """Sealed terminal receipt for one OS-isolated frozen evaluation command."""

    schema_version: int
    backend: IsolationBackend
    runtime_identity: str
    source_identity: IsolationSourceIdentity
    runtime_controls_sha256: str
    isolation_config_sha256: str
    command_sha256: str
    command_name: str
    output_tree: IsolationOutputTreeIdentity
    probe: IsolationProbeEvidence
    status: IsolationJobStatus
    target_exit_code: int | None
    started_at: str
    finished_at: str

    def __post_init__(self) -> None:
        if self.schema_version != ISOLATION_RECEIPT_SCHEMA_VERSION:
            _fail(IsolationIntegrityError, "unsupported isolation receipt schema")
        if not isinstance(self.backend, IsolationBackend):
            _fail(IsolationIntegrityError, "receipt backend is invalid")
        if not isinstance(self.source_identity, IsolationSourceIdentity):
            _fail(IsolationIntegrityError, "receipt source identity is invalid")
        _sha256(self.runtime_controls_sha256, "runtime_controls_sha256")
        if self.backend is IsolationBackend.MACOS_SANDBOX:
            if (
                self.source_identity.mode
                is not IsolationInputMode.LIVE_CHECKOUT_NETWORK_ONLY
            ):
                _fail(
                    IsolationIntegrityError,
                    "macOS isolation receipt must declare "
                    "live-checkout network-only input",
                )
            if self.runtime_identity != MACOS_SANDBOX_RUNTIME_IDENTITY:
                _fail(
                    IsolationIntegrityError,
                    "macOS receipt runtime identity does not match the fixed profile",
                )
            expected_config = _isolation_config_sha256(
                self.backend,
                source_identity=self.source_identity,
                runtime_controls_sha256=self.runtime_controls_sha256,
            )
        else:
            if (
                self.source_identity.mode
                is not IsolationInputMode.COMMITTED_SOURCE_SNAPSHOT
            ):
                _fail(
                    IsolationIntegrityError,
                    "Docker isolation receipt requires a committed source snapshot",
                )
            if not _IMMUTABLE_IMAGE_PATTERN.fullmatch(self.runtime_identity):
                _fail(
                    IsolationIntegrityError,
                    "Docker receipt runtime identity must be an immutable image digest",
                )
            expected_config = _isolation_config_sha256(
                self.backend,
                docker_image=self.runtime_identity,
                source_identity=self.source_identity,
                runtime_controls_sha256=self.runtime_controls_sha256,
            )
        _sha256(self.isolation_config_sha256, "isolation_config_sha256")
        if self.isolation_config_sha256 != expected_config:
            _fail(
                IsolationIntegrityError,
                "runtime identity does not reproduce isolation_config_sha256",
            )
        _sha256(self.command_sha256, "command_sha256")
        _text(self.command_name, "command_name", maximum=128)
        if not isinstance(self.output_tree, IsolationOutputTreeIdentity):
            _fail(IsolationIntegrityError, "receipt output_tree is invalid")
        if not isinstance(self.probe, IsolationProbeEvidence):
            _fail(IsolationIntegrityError, "receipt probe is invalid")
        if not isinstance(self.status, IsolationJobStatus):
            _fail(IsolationIntegrityError, "receipt status is invalid")
        if self.target_exit_code is not None and (
            isinstance(self.target_exit_code, bool)
            or not isinstance(self.target_exit_code, int)
            or not 0 <= self.target_exit_code <= 255
        ):
            _fail(IsolationIntegrityError, "target_exit_code is invalid")
        started = datetime.fromisoformat(_timestamp(self.started_at, "started_at"))
        finished = datetime.fromisoformat(_timestamp(self.finished_at, "finished_at"))
        if finished < started:
            _fail(IsolationIntegrityError, "receipt finished before it started")
        if self.status in {
            IsolationJobStatus.INPUT_FAILED,
            IsolationJobStatus.PROBE_FAILED,
        }:
            if self.probe.passed or self.target_exit_code is not None:
                _fail(
                    IsolationIntegrityError,
                    "pre-target failure cannot claim a passed probe "
                    "or target execution",
                )
            if self.output_tree != _empty_output_tree_identity():
                _fail(
                    IsolationIntegrityError,
                    "pre-target failure must bind the canonical empty output tree",
                )
        elif not self.probe.passed or self.target_exit_code is None:
            _fail(
                IsolationIntegrityError,
                "target terminal status requires a passed isolation probe",
            )
        elif (self.status is IsolationJobStatus.COMPLETED) != (
            self.target_exit_code == 0
        ):
            _fail(
                IsolationIntegrityError,
                "completed status and zero target exit code must agree",
            )

    @property
    def sha256(self) -> str:
        """Return the canonical receipt commitment."""
        return sha256_bytes(canonical_json_bytes(self.to_mapping()))

    @property
    def successful(self) -> bool:
        """Return whether both the OS probe and target command completed."""
        return self.status is IsolationJobStatus.COMPLETED

    def to_mapping(self) -> dict[str, object]:
        """Return the canonical JSON representation."""
        return {
            "backend": self.backend.value,
            "command_name": self.command_name,
            "command_sha256": self.command_sha256,
            "finished_at": self.finished_at,
            "isolation_config_sha256": self.isolation_config_sha256,
            "output_tree": self.output_tree.to_mapping(),
            "probe": self.probe.to_mapping(),
            "runtime_identity": self.runtime_identity,
            "runtime_controls_sha256": self.runtime_controls_sha256,
            "schema_version": self.schema_version,
            "source_identity": self.source_identity.to_mapping(),
            "started_at": self.started_at,
            "status": self.status.value,
            "target_exit_code": self.target_exit_code,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> Self:
        """Strictly parse one receipt."""
        _strict_fields(
            value,
            frozenset(
                {
                    "backend",
                    "command_name",
                    "command_sha256",
                    "finished_at",
                    "isolation_config_sha256",
                    "output_tree",
                    "probe",
                    "runtime_identity",
                    "runtime_controls_sha256",
                    "schema_version",
                    "source_identity",
                    "started_at",
                    "status",
                    "target_exit_code",
                }
            ),
            "isolation receipt",
        )
        version = value["schema_version"]
        exit_code = value["target_exit_code"]
        if isinstance(version, bool) or not isinstance(version, int):
            _fail(IsolationIntegrityError, "receipt schema_version must be an integer")
        if exit_code is not None and (
            isinstance(exit_code, bool) or not isinstance(exit_code, int)
        ):
            _fail(
                IsolationIntegrityError, "target_exit_code must be an integer or null"
            )
        raw_probe = value["probe"]
        raw_output_tree = value["output_tree"]
        raw_source_identity = value["source_identity"]
        if not isinstance(raw_probe, Mapping):
            _fail(IsolationIntegrityError, "receipt probe must be an object")
        if not isinstance(raw_output_tree, Mapping):
            _fail(IsolationIntegrityError, "receipt output_tree must be an object")
        if not isinstance(raw_source_identity, Mapping):
            _fail(IsolationIntegrityError, "receipt source_identity must be an object")
        return cls(
            schema_version=version,
            backend=cast(
                IsolationBackend,
                _enum_value(IsolationBackend, value["backend"], "isolation backend"),
            ),
            runtime_identity=_text(
                value["runtime_identity"], "runtime_identity", maximum=512
            ),
            source_identity=IsolationSourceIdentity.from_mapping(
                cast(Mapping[str, object], raw_source_identity)
            ),
            runtime_controls_sha256=_sha256(
                value["runtime_controls_sha256"], "runtime_controls_sha256"
            ),
            isolation_config_sha256=_sha256(
                value["isolation_config_sha256"], "isolation_config_sha256"
            ),
            command_sha256=_sha256(value["command_sha256"], "command_sha256"),
            command_name=_text(value["command_name"], "command_name", maximum=128),
            output_tree=IsolationOutputTreeIdentity.from_mapping(
                cast(Mapping[str, object], raw_output_tree)
            ),
            probe=IsolationProbeEvidence.from_mapping(
                cast(Mapping[str, object], raw_probe)
            ),
            status=cast(
                IsolationJobStatus,
                _enum_value(IsolationJobStatus, value["status"], "receipt status"),
            ),
            target_exit_code=exit_code,
            started_at=_timestamp(value["started_at"], "started_at"),
            finished_at=_timestamp(value["finished_at"], "finished_at"),
        )


def _real_directory(path: Path, name: str) -> Path:
    absolute = Path(os.path.abspath(path.expanduser()))
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current /= part
        try:
            info = os.lstat(current)
        except OSError as error:
            raise IsolationUnavailableError(f"{name} does not exist: {path}") from error
        if stat.S_ISLNK(info.st_mode):
            _fail(IsolationUnavailableError, f"{name} contains a symbolic link")
    if not stat.S_ISDIR(os.lstat(absolute).st_mode):
        _fail(IsolationUnavailableError, f"{name} must be a real directory")
    return absolute


def _validated_command(command: Sequence[str]) -> tuple[str, ...]:
    if isinstance(command, str) or not command:
        _fail(IsolationUnavailableError, "frozen evaluation command must not be empty")
    normalized = tuple(command)
    if not all(
        isinstance(item, str) and item and "\x00" not in item for item in normalized
    ):
        _fail(
            IsolationUnavailableError,
            "frozen evaluation command arguments must be non-empty NUL-free strings",
        )
    if sum(len(item) for item in normalized) > 128 * 1024:
        _fail(IsolationUnavailableError, "frozen evaluation command is too large")
    return normalized


def _validated_receipt_name(value: str) -> str:
    if not _RECEIPT_NAME_PATTERN.fullmatch(value) or value.endswith(".sha256"):
        _fail(
            IsolationUnavailableError,
            "receipt name must be one bounded plain filename not ending in .sha256",
        )
    return value


def _immutable_image(value: str | None) -> str:
    if value is None or not _IMMUTABLE_IMAGE_PATTERN.fullmatch(value):
        _fail(
            IsolationUnavailableError,
            "Docker isolation requires an immutable sha256 image ID or digest; "
            "mutable tags are forbidden",
        )
    return value


def _validated_image_id(value: str | None) -> str:
    if value is None or _IMAGE_ID_PATTERN.fullmatch(value) is None:
        _fail(
            IsolationUnavailableError,
            "Docker isolation requires the inspected local sha256 image ID",
        )
    return value


def _docker_cli_environment() -> dict[str, str]:
    return {
        "DOCKER_CONFIG": "/var/empty/policytwin-docker-config",
        "DOCKER_HOST": _DOCKER_DAEMON_ENDPOINT,
        "HOME": "/var/empty",
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "PATH": "/usr/local/bin:/usr/bin:/bin",
        "TZ": "UTC",
    }


def _docker_cli_environment_sha256() -> str:
    return sha256_bytes(canonical_json_bytes(_docker_cli_environment()))


def _macos_launcher_environment() -> dict[str, str]:
    return {
        "HOME": "/var/empty",
        "LANG": "C",
        "LC_ALL": "C",
        "PATH": "/usr/bin:/bin",
        "TZ": "UTC",
    }


def _macos_launcher_environment_sha256() -> str:
    return sha256_bytes(canonical_json_bytes(_macos_launcher_environment()))


def _verified_fixed_executable_bytes(path: Path, *, name: str) -> tuple[int, str]:
    try:
        before = os.lstat(path)
    except OSError as error:
        raise IsolationUnavailableError(f"the fixed {name} is unavailable") from error
    if (
        stat.S_ISLNK(before.st_mode)
        or not stat.S_ISREG(before.st_mode)
        or before.st_uid != 0
        or not before.st_mode & 0o111
        or bool(before.st_mode & 0o022)
        or before.st_size <= 0
    ):
        _fail(
            IsolationUnavailableError,
            f"the fixed {name} must be root-owned, executable, non-writable, "
            "and regular",
        )
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise IsolationUnavailableError(f"cannot open the fixed {name}") from error
    digest = hashlib.sha256()
    try:
        opened = os.fstat(descriptor)
        if (
            opened.st_dev != before.st_dev
            or opened.st_ino != before.st_ino
            or opened.st_size != before.st_size
            or opened.st_uid != before.st_uid
            or opened.st_mode != before.st_mode
        ):
            _fail(IsolationUnavailableError, f"{name} changed while being opened")
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
        after_open = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    try:
        after_path = os.lstat(path)
    except OSError as error:
        raise IsolationUnavailableError(
            f"{name} disappeared during identity verification"
        ) from error
    stable_fields = ("st_dev", "st_ino", "st_mode", "st_uid", "st_size", "st_mtime_ns")
    if any(
        getattr(before, field) != getattr(after_open, field)
        or getattr(before, field) != getattr(after_path, field)
        for field in stable_fields
    ):
        _fail(IsolationUnavailableError, f"{name} changed during hashing")
    return before.st_size, digest.hexdigest()


def _verified_docker_cli_identity() -> DockerCliIdentity:
    size_bytes, digest = _verified_fixed_executable_bytes(
        _DOCKER_CLI_PATH, name="/usr/bin/docker CLI"
    )
    return DockerCliIdentity(
        path=str(_DOCKER_CLI_PATH),
        size_bytes=size_bytes,
        sha256=digest,
    )


def _verified_macos_sandbox_identity() -> MacosSandboxIdentity:
    size_bytes, digest = _verified_fixed_executable_bytes(
        _MACOS_SANDBOX_EXEC_PATH, name="/usr/bin/sandbox-exec"
    )
    # ``env -i`` is part of the fixed launcher chain.  It is a trusted OS path,
    # not a PATH-resolved helper; verify its filesystem trust properties too.
    environment_size_bytes, environment_digest = _verified_fixed_executable_bytes(
        _MACOS_ENV_PATH, name="/usr/bin/env"
    )
    return MacosSandboxIdentity(
        path=str(_MACOS_SANDBOX_EXEC_PATH),
        size_bytes=size_bytes,
        sha256=digest,
        environment_launcher_path=str(_MACOS_ENV_PATH),
        environment_launcher_size_bytes=environment_size_bytes,
        environment_launcher_sha256=environment_digest,
    )


def _require_local_docker_daemon() -> DockerCliIdentity:
    ambient = sorted(key for key in _DOCKER_AMBIENT_OVERRIDES if key in os.environ)
    if ambient:
        _fail(
            IsolationUnavailableError,
            f"ambient Docker endpoint/configuration overrides are forbidden: {ambient}",
        )
    resolved = shutil.which("docker")
    if resolved is None or Path(resolved) != _DOCKER_CLI_PATH:
        _fail(
            IsolationUnavailableError,
            "PATH must resolve docker exactly to the fixed /usr/bin/docker CLI",
        )
    docker_cli_identity = _verified_docker_cli_identity()
    try:
        info = os.lstat(_DOCKER_DAEMON_SOCKET)
    except OSError as error:
        raise IsolationUnavailableError(
            "the fixed local Docker AF_UNIX endpoint is unavailable"
        ) from error
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISSOCK(info.st_mode):
        _fail(
            IsolationUnavailableError,
            "the fixed Docker endpoint must be a real local AF_UNIX socket",
        )
    return docker_cli_identity


def _require_local_macos_sandbox() -> MacosSandboxIdentity:
    if platform.system() != "Darwin":
        _fail(
            IsolationUnavailableError,
            "macOS sandbox backend was requested on a non-macOS host",
        )
    resolved = shutil.which("sandbox-exec")
    if resolved is None or Path(resolved) != _MACOS_SANDBOX_EXEC_PATH:
        _fail(
            IsolationUnavailableError,
            "PATH must resolve sandbox-exec exactly to /usr/bin/sandbox-exec; "
            "no Python-only fallback is allowed",
        )
    return _verified_macos_sandbox_identity()


def _require_same_docker_cli_identity(
    expected: DockerCliIdentity,
    *,
    context: str,
    primary_error: BaseException | None = None,
) -> None:
    try:
        actual = _verified_docker_cli_identity()
    except IsolationError:
        if primary_error is not None:
            raise IsolationUnavailableError(
                f"Docker CLI identity verification failed {context}"
            ) from primary_error
        raise
    if actual != expected:
        mismatch = IsolationIntegrityError(f"Docker CLI identity changed {context}")
        if primary_error is not None:
            raise mismatch from primary_error
        raise mismatch


def _require_same_macos_sandbox_identity(
    expected: MacosSandboxIdentity,
    *,
    context: str,
    primary_error: BaseException | None = None,
) -> None:
    try:
        actual = _verified_macos_sandbox_identity()
    except IsolationError:
        if primary_error is not None:
            raise IsolationUnavailableError(
                f"macOS sandbox executable verification failed {context}"
            ) from primary_error
        raise
    if actual != expected:
        mismatch = IsolationIntegrityError(
            f"macOS sandbox executable identity changed {context}"
        )
        if primary_error is not None:
            raise mismatch from primary_error
        raise mismatch


def _live_checkout_source_identity() -> IsolationSourceIdentity:
    return IsolationSourceIdentity(
        mode=IsolationInputMode.LIVE_CHECKOUT_NETWORK_ONLY,
        commit=None,
        tree_sha256=None,
        manifest_sha256=None,
        file_count=None,
        archive_sha256=None,
        archive_size_bytes=None,
        bootstrap_sha256=None,
        git_cli_identity=None,
    )


def _committed_source_identity(
    manifest: SourceSnapshotManifest,
    *,
    archive_sha256: str,
    archive_size_bytes: int,
    git_cli_identity: GitCliIdentity,
) -> IsolationSourceIdentity:
    return IsolationSourceIdentity(
        mode=IsolationInputMode.COMMITTED_SOURCE_SNAPSHOT,
        commit=manifest.commit,
        tree_sha256=manifest.tree_sha256,
        manifest_sha256=sha256_bytes(canonical_json_bytes(manifest.to_mapping())),
        file_count=manifest.file_count,
        archive_sha256=archive_sha256,
        archive_size_bytes=archive_size_bytes,
        bootstrap_sha256=DOCKER_BOOTSTRAP_SHA256,
        git_cli_identity=git_cli_identity,
    )


def _manifest_matches_source_identity(
    manifest: SourceSnapshotManifest,
    identity: IsolationSourceIdentity,
) -> bool:
    return (
        identity.mode is IsolationInputMode.COMMITTED_SOURCE_SNAPSHOT
        and manifest.commit == identity.commit
        and manifest.tree_sha256 == identity.tree_sha256
        and sha256_bytes(canonical_json_bytes(manifest.to_mapping()))
        == identity.manifest_sha256
        and manifest.file_count == identity.file_count
        and identity.archive_sha256 is not None
        and identity.archive_size_bytes is not None
        and identity.bootstrap_sha256 == DOCKER_BOOTSTRAP_SHA256
        and isinstance(identity.git_cli_identity, GitCliIdentity)
    )


def _expected_macos_runtime_controls(
    *,
    sandbox_exec_sha256: str,
    sandbox_exec_size_bytes: int,
    environment_launcher_sha256: str,
    environment_launcher_size_bytes: int,
) -> dict[str, object]:
    if (
        isinstance(sandbox_exec_size_bytes, bool)
        or not isinstance(sandbox_exec_size_bytes, int)
        or sandbox_exec_size_bytes <= 0
    ):
        _fail(IsolationIntegrityError, "macOS sandbox executable size is invalid")
    if (
        isinstance(environment_launcher_size_bytes, bool)
        or not isinstance(environment_launcher_size_bytes, int)
        or environment_launcher_size_bytes <= 0
    ):
        _fail(
            IsolationIntegrityError,
            "macOS environment launcher executable size is invalid",
        )
    return {
        "backend": IsolationBackend.MACOS_SANDBOX.value,
        "close_fds": True,
        "environment_launcher_path": str(_MACOS_ENV_PATH),
        "environment_launcher_sha256": _sha256(
            environment_launcher_sha256,
            "macOS environment launcher executable sha256",
        ),
        "environment_launcher_size_bytes": environment_launcher_size_bytes,
        "launcher_environment_sha256": _macos_launcher_environment_sha256(),
        "probe": "inet-and-host-unix-canary-denied",
        "profile_sha256": sha256_bytes(MACOS_SANDBOX_PROFILE.encode("utf-8")),
        "sandbox_exec_path": str(_MACOS_SANDBOX_EXEC_PATH),
        "sandbox_exec_sha256": _sha256(
            sandbox_exec_sha256, "macOS sandbox executable sha256"
        ),
        "sandbox_exec_size_bytes": sandbox_exec_size_bytes,
        "schema_version": "policytwin.runtime-controls.v1",
    }


def _macos_runtime_controls_sha256(
    *,
    sandbox_exec_sha256: str,
    sandbox_exec_size_bytes: int,
    environment_launcher_sha256: str,
    environment_launcher_size_bytes: int,
) -> str:
    return _runtime_controls_sha256(
        _expected_macos_runtime_controls(
            sandbox_exec_sha256=sandbox_exec_sha256,
            sandbox_exec_size_bytes=sandbox_exec_size_bytes,
            environment_launcher_sha256=environment_launcher_sha256,
            environment_launcher_size_bytes=environment_launcher_size_bytes,
        )
    )


def _expected_docker_runtime_controls(
    *,
    docker_image: str,
    actual_image_id: str,
    docker_cli_sha256: str,
    docker_cli_size_bytes: int,
    user: str,
) -> dict[str, object]:
    if _IMAGE_ID_PATTERN.fullmatch(actual_image_id) is None:
        _fail(IsolationIntegrityError, "Docker actual image ID is invalid")
    if (
        isinstance(docker_cli_size_bytes, bool)
        or not isinstance(docker_cli_size_bytes, int)
        or docker_cli_size_bytes <= 0
    ):
        _fail(IsolationIntegrityError, "Docker CLI size must be positive")
    return {
        "actual_image_id": actual_image_id,
        "auto_remove": False,
        "backend": IsolationBackend.DOCKER_NONE.value,
        "binds": [],
        "bootstrap_sha256": DOCKER_BOOTSTRAP_SHA256,
        "cap_add": [],
        "cap_drop": ["ALL"],
        "dangerous_environment_keys": [],
        "daemon_endpoint": _DOCKER_DAEMON_ENDPOINT,
        "docker_cli_environment_sha256": _docker_cli_environment_sha256(),
        "docker_cli_path": str(_DOCKER_CLI_PATH),
        "docker_cli_sha256": _sha256(docker_cli_sha256, "docker_cli_sha256"),
        "docker_cli_size_bytes": docker_cli_size_bytes,
        "entrypoint": ["/usr/local/bin/python"],
        "devices": [],
        "device_requests": [],
        "healthcheck_disabled": True,
        "image": _immutable_image(docker_image),
        "mounts": [{"destination": _DOCKER_OUTPUT_ROOT, "rw": True, "type": "volume"}],
        "network_mode": "none",
        "no_new_privileges": True,
        "ipc_mode": "private",
        "pid_mode": "",
        "pids_limit": 128,
        "privileged": False,
        "readonly_rootfs": True,
        "schema_version": "policytwin.runtime-controls.v1",
        "source_transport": "private_container_layer",
        "security_opt": ["no-new-privileges:true"],
        "tmpfs": {
            "/tmp": "rw,noexec,nosuid,nodev,size=64m,mode=1777",
        },
        "user": user,
        "working_dir": _DOCKER_FINAL_WORKDIR,
    }


def _runtime_controls_sha256(value: Mapping[str, object]) -> str:
    return sha256_bytes(canonical_json_bytes(value))


def _isolation_config_sha256(
    backend: IsolationBackend,
    *,
    docker_image: str | None = None,
    source_identity: IsolationSourceIdentity | None = None,
    runtime_controls_sha256: str | None = None,
) -> str:
    selected_source = source_identity
    if selected_source is None:
        if backend is IsolationBackend.MACOS_SANDBOX:
            selected_source = _live_checkout_source_identity()
        else:
            _fail(
                IsolationUnavailableError,
                "Docker isolation configuration requires a committed source identity",
            )
    if backend is IsolationBackend.MACOS_SANDBOX:
        if runtime_controls_sha256 is None:
            _fail(
                IsolationUnavailableError,
                "macOS isolation configuration requires pinned launcher controls",
            )
        selected_runtime_controls = _sha256(
            runtime_controls_sha256, "runtime_controls_sha256"
        )
        if selected_source.mode is not IsolationInputMode.LIVE_CHECKOUT_NETWORK_ONLY:
            _fail(
                IsolationUnavailableError,
                "macOS isolation is network-only and must declare live checkout input",
            )
    elif selected_source.mode is not IsolationInputMode.COMMITTED_SOURCE_SNAPSHOT:
        _fail(
            IsolationUnavailableError,
            "Docker isolation requires committed source snapshot input",
        )
    if backend is IsolationBackend.MACOS_SANDBOX:
        material: dict[str, object] = {
            "backend": backend.value,
            "profile": MACOS_SANDBOX_PROFILE,
            "profile_sha256": sha256_bytes(MACOS_SANDBOX_PROFILE.encode("utf-8")),
            "subprocess_close_fds": True,
        }
    else:
        if runtime_controls_sha256 is None:
            _fail(
                IsolationUnavailableError,
                "Docker isolation configuration requires runtime controls evidence",
            )
        selected_runtime_controls = _sha256(
            runtime_controls_sha256, "runtime_controls_sha256"
        )
        material = {
            "backend": backend.value,
            "controls": list(_DOCKER_CONTROLS),
            "image": _immutable_image(docker_image),
        }
    material["source_identity"] = selected_source.to_mapping()
    material["runtime_controls_sha256"] = selected_runtime_controls
    return sha256_bytes(canonical_json_bytes(material))


def _runtime_identity(
    backend: IsolationBackend, *, docker_image: str | None = None
) -> str:
    if backend is IsolationBackend.MACOS_SANDBOX:
        return MACOS_SANDBOX_RUNTIME_IDENTITY
    return _immutable_image(docker_image)


def _worker_environment(
    project_root: Path, marker: str, *, macos_unix_canary_path: Path | None = None
) -> tuple[str, ...]:
    python_directory = str(Path(sys.executable).resolve().parent)
    environment = (
        f"{_WORKER_MARKER}={marker}",
        "HOME=/tmp",
        "LANG=C.UTF-8",
        f"PATH={python_directory}:/usr/local/bin:/usr/bin:/bin",
        "PYTHONDONTWRITEBYTECODE=1",
        "PYTHONHASHSEED=0",
        f"PYTHONPATH={project_root / 'src'}",
        "TMPDIR=/tmp",
        "TZ=UTC",
    )
    if macos_unix_canary_path is None:
        return environment
    return (*environment, f"{_MACOS_UNIX_CANARY_ENV}={macos_unix_canary_path}")


def _source_worker_arguments(identity: IsolationSourceIdentity) -> tuple[str, ...]:
    arguments = ("--input-mode", identity.mode.value)
    if identity.mode is IsolationInputMode.LIVE_CHECKOUT_NETWORK_ONLY:
        return arguments
    assert identity.commit is not None
    assert identity.tree_sha256 is not None
    assert identity.manifest_sha256 is not None
    assert identity.file_count is not None
    assert identity.archive_sha256 is not None
    assert identity.archive_size_bytes is not None
    assert identity.bootstrap_sha256 is not None
    assert identity.git_cli_identity is not None
    return (
        *arguments,
        "--source-commit",
        identity.commit,
        "--source-tree-sha256",
        identity.tree_sha256,
        "--source-manifest-sha256",
        identity.manifest_sha256,
        "--source-file-count",
        str(identity.file_count),
        "--source-archive-sha256",
        identity.archive_sha256,
        "--source-archive-size-bytes",
        str(identity.archive_size_bytes),
        "--source-bootstrap-sha256",
        identity.bootstrap_sha256,
        "--source-git-cli-path",
        identity.git_cli_identity.path,
        "--source-git-cli-platform-system",
        identity.git_cli_identity.platform_system,
        "--source-git-cli-platform-machine",
        identity.git_cli_identity.platform_machine,
        "--source-git-cli-sha256",
        identity.git_cli_identity.sha256,
        "--source-git-cli-size-bytes",
        str(identity.git_cli_identity.size_bytes),
    )


def _verify_prepared_docker_snapshot(
    snapshot: DockerSourceSnapshot,
) -> DockerSourceSnapshot:
    if not isinstance(snapshot, DockerSourceSnapshot):
        _fail(
            IsolationUnavailableError,
            "Docker launch requires a prepared committed source snapshot",
        )
    if not isinstance(snapshot.git_cli_identity, GitCliIdentity):
        _fail(
            IsolationUnavailableError,
            "Docker source snapshot lacks a valid Git CLI identity",
        )
    root = _real_directory(snapshot.root, "Docker source snapshot")
    archive_parent = _real_directory(
        snapshot.archive_path.parent, "Docker source archive parent"
    )
    archive_path = archive_parent / snapshot.archive_path.name
    manifest_parent = _real_directory(
        snapshot.manifest_path.parent, "Docker source manifest parent"
    )
    manifest_path = manifest_parent / snapshot.manifest_path.name
    try:
        manifest = read_source_snapshot_manifest(manifest_path)
        verify_source_snapshot_directory(root, manifest)
    except SourcePreflightError as error:
        raise IsolationUnavailableError(
            f"Docker source snapshot verification failed: {error}"
        ) from error
    try:
        archive_info = os.lstat(archive_path)
    except OSError as error:
        raise IsolationUnavailableError("Docker source archive is missing") from error
    if (
        not stat.S_ISREG(archive_info.st_mode)
        or stat.S_ISLNK(archive_info.st_mode)
        or archive_info.st_size != snapshot.archive_size_bytes
        or sha256_file(archive_path) != snapshot.archive_sha256
    ):
        _fail(
            IsolationUnavailableError,
            "Docker source archive differs from its prepared identity",
        )
    identity = _committed_source_identity(
        manifest,
        archive_sha256=snapshot.archive_sha256,
        archive_size_bytes=snapshot.archive_size_bytes,
        git_cli_identity=snapshot.git_cli_identity,
    )
    if manifest != snapshot.manifest or identity != snapshot.identity:
        _fail(
            IsolationUnavailableError,
            "prepared Docker snapshot identity differs from its verified manifest",
        )
    return DockerSourceSnapshot(
        root=root,
        archive_path=archive_path,
        archive_sha256=snapshot.archive_sha256,
        archive_size_bytes=snapshot.archive_size_bytes,
        manifest_path=manifest_path,
        manifest=manifest,
        git_cli_identity=snapshot.git_cli_identity,
        identity=identity,
    )


def _require_clean_matching_head(
    project_root: Path, snapshot: DockerSourceSnapshot
) -> None:
    report = run_source_preflight(project_root)
    if (
        not report.source_snapshot_ready
        or report.manifest is None
        or report.manifest != snapshot.manifest
        or report.git_cli_identity != snapshot.git_cli_identity
    ):
        _fail(
            IsolationUnavailableError,
            "Docker source snapshot no longer matches a clean project HEAD",
        )


def build_isolation_command(
    *,
    backend: IsolationBackend,
    command: Sequence[str],
    project_root: Path,
    output_directory: Path,
    receipt_name: str = DEFAULT_RECEIPT_NAME,
    docker_image: str | None = None,
    docker_source_snapshot: DockerSourceSnapshot | None = None,
    docker_local_image_id: str | None = None,
    docker_cli_sha256: str | None = None,
    docker_cli_size_bytes: int | None = None,
    macos_sandbox_exec_sha256: str | None = None,
    macos_sandbox_exec_size_bytes: int | None = None,
    macos_environment_launcher_sha256: str | None = None,
    macos_environment_launcher_size_bytes: int | None = None,
    macos_unix_canary_path: Path | None = None,
) -> tuple[str, ...]:
    """Build the exact fail-closed OS launcher command without executing it."""
    if not isinstance(backend, IsolationBackend):
        _fail(IsolationUnavailableError, "backend must be an IsolationBackend")
    target = _validated_command(command)
    root = _real_directory(project_root, "project root")
    output = _real_directory(output_directory, "output directory")
    name = _validated_receipt_name(receipt_name)
    if (output / name).exists() or (output / f"{name}.sha256").exists():
        _fail(IsolationUnavailableError, "isolation receipt path already exists")
    if output == root or output.is_relative_to(root):
        _fail(
            IsolationUnavailableError,
            "isolation output directory must be outside the project input tree",
        )
    try:
        output_entries = tuple(output.iterdir())
    except OSError as error:
        raise IsolationUnavailableError(
            "isolation output directory cannot be inspected"
        ) from error
    if output_entries:
        _fail(
            IsolationUnavailableError,
            "isolation output directory must be empty before launch",
        )
    worker_base = (
        "-m",
        "policytwin.evaluation.os_isolation",
        "_worker",
        "--backend",
        backend.value,
    )
    if backend is IsolationBackend.MACOS_SANDBOX:
        if docker_source_snapshot is not None:
            _fail(
                IsolationUnavailableError,
                "macOS network-only backend cannot claim a Docker source snapshot",
            )
        if platform.system() != "Darwin":
            _fail(
                IsolationUnavailableError,
                "macOS sandbox backend was requested on a non-macOS host",
            )
        if (
            macos_sandbox_exec_sha256 is None
            or macos_environment_launcher_sha256 is None
            or isinstance(macos_environment_launcher_size_bytes, bool)
            or not isinstance(macos_environment_launcher_size_bytes, int)
            or macos_environment_launcher_size_bytes <= 0
            or (
                isinstance(macos_sandbox_exec_size_bytes, bool)
                or not isinstance(macos_sandbox_exec_size_bytes, int)
                or macos_sandbox_exec_size_bytes <= 0
            )
        ):
            sandbox_identity = _require_local_macos_sandbox()
        else:
            sandbox_identity = MacosSandboxIdentity(
                path=str(_MACOS_SANDBOX_EXEC_PATH),
                size_bytes=macos_sandbox_exec_size_bytes,
                sha256=macos_sandbox_exec_sha256,
                environment_launcher_path=str(_MACOS_ENV_PATH),
                environment_launcher_size_bytes=(macos_environment_launcher_size_bytes),
                environment_launcher_sha256=macos_environment_launcher_sha256,
            )
        if macos_unix_canary_path is None:
            _fail(
                IsolationUnavailableError,
                "macOS sandbox launch requires a live host AF_UNIX canary",
            )
        canary_parent = _real_directory(
            macos_unix_canary_path.parent, "macOS AF_UNIX canary parent"
        )
        canary_path = canary_parent / macos_unix_canary_path.name
        try:
            canary_info = os.lstat(canary_path)
        except OSError as error:
            raise IsolationUnavailableError(
                "macOS AF_UNIX canary is unavailable"
            ) from error
        if not stat.S_ISSOCK(canary_info.st_mode):
            _fail(
                IsolationUnavailableError,
                "macOS AF_UNIX canary must be a live socket",
            )
        source_identity = _live_checkout_source_identity()
        runtime_controls_sha256 = _macos_runtime_controls_sha256(
            sandbox_exec_sha256=sandbox_identity.sha256,
            sandbox_exec_size_bytes=sandbox_identity.size_bytes,
            environment_launcher_sha256=(sandbox_identity.environment_launcher_sha256),
            environment_launcher_size_bytes=(
                sandbox_identity.environment_launcher_size_bytes
            ),
        )
        config_sha256 = _isolation_config_sha256(
            backend,
            source_identity=source_identity,
            runtime_controls_sha256=runtime_controls_sha256,
        )
        worker = (
            sys.executable,
            *worker_base,
            *_source_worker_arguments(source_identity),
            "--isolation-config-sha256",
            config_sha256,
            "--runtime-controls-sha256",
            runtime_controls_sha256,
            "--receipt",
            str(output / name),
            "--",
            *target,
        )
        return (
            str(_MACOS_SANDBOX_EXEC_PATH),
            "-p",
            MACOS_SANDBOX_PROFILE,
            "/usr/bin/env",
            "-i",
            *_worker_environment(
                root,
                _SANDBOX_MARKER,
                macos_unix_canary_path=canary_path,
            ),
            *worker,
        )
    docker = str(_DOCKER_CLI_PATH)
    image = _immutable_image(docker_image)
    local_image_id = _validated_image_id(docker_local_image_id)
    cli_sha256 = _sha256(docker_cli_sha256, "docker_cli_sha256")
    if (
        isinstance(docker_cli_size_bytes, bool)
        or not isinstance(docker_cli_size_bytes, int)
        or docker_cli_size_bytes <= 0
    ):
        _fail(IsolationUnavailableError, "Docker CLI size identity is invalid")
    if docker_source_snapshot is None:
        _fail(
            IsolationUnavailableError,
            "Docker launch refuses the live checkout; prepare a committed snapshot",
        )
    source_snapshot = _verify_prepared_docker_snapshot(docker_source_snapshot)
    _require_clean_matching_head(root, source_snapshot)
    source_identity = source_snapshot.identity
    user = _DOCKER_FINAL_USER
    runtime_controls = _expected_docker_runtime_controls(
        docker_image=image,
        actual_image_id=local_image_id,
        docker_cli_sha256=cli_sha256,
        docker_cli_size_bytes=docker_cli_size_bytes,
        user=user,
    )
    runtime_controls_sha256 = _runtime_controls_sha256(runtime_controls)
    config_sha256 = _isolation_config_sha256(
        backend,
        docker_image=image,
        source_identity=source_identity,
        runtime_controls_sha256=runtime_controls_sha256,
    )
    worker = (
        "/usr/local/bin/python",
        *worker_base,
        *_source_worker_arguments(source_identity),
        "--isolation-config-sha256",
        config_sha256,
        "--runtime-controls-sha256",
        runtime_controls_sha256,
        "--docker-image",
        image,
        "--receipt",
        f"{_DOCKER_OUTPUT_ROOT}/{name}",
        "--source-root",
        _DOCKER_SOURCE_ROOT,
        "--source-manifest",
        _DOCKER_SOURCE_MANIFEST_PATH,
        "--",
        *target,
    )
    bootstrap = (
        "-I",
        "-S",
        "-c",
        _DOCKER_BOOTSTRAP_SOURCE,
        _DOCKER_SOURCE_MANIFEST_PATH,
        _DOCKER_SOURCE_ROOT,
        cast(str, source_identity.manifest_sha256),
        cast(str, source_identity.tree_sha256),
        str(source_identity.file_count),
        DOCKER_BOOTSTRAP_SHA256,
        "--",
        *worker,
    )
    return (
        docker,
        "create",
        "--pull=never",
        "--network=none",
        "--read-only",
        "--cap-drop=ALL",
        "--privileged=false",
        "--ipc=private",
        "--security-opt=no-new-privileges:true",
        "--pids-limit=128",
        "--no-healthcheck",
        f"--user={user}",
        "--entrypoint=/usr/local/bin/python",
        f"--workdir={_DOCKER_SOURCE_ROOT}",
        f"--mount=type=volume,dst={_DOCKER_OUTPUT_ROOT},volume-nocopy",
        "--tmpfs=/tmp:rw,noexec,nosuid,nodev,size=64m,mode=1777",
        f"--env={_WORKER_MARKER}={_DOCKER_MARKER}",
        "--env=HOME=/tmp",
        "--env=LANG=C.UTF-8",
        "--env=PATH=/usr/local/bin:/usr/bin:/bin",
        "--env=PYTHONDONTWRITEBYTECODE=1",
        "--env=PYTHONHASHSEED=0",
        "--env=TMPDIR=/tmp",
        "--env=TZ=UTC",
        image,
        *bootstrap,
    )


def _linux_non_loopback_routes() -> tuple[str, ...]:
    route_path = Path("/proc/net/route")
    try:
        lines = route_path.read_text(encoding="ascii").splitlines()
    except OSError:
        return ("route_table_unavailable",)
    routes: list[str] = []
    for line in lines[1:]:
        fields = line.split()
        if fields and fields[0] != "lo":
            routes.append(fields[0])
    return tuple(sorted(routes))


def _probe_os_isolation(backend: IsolationBackend) -> IsolationProbeEvidence:
    marker = os.environ.get(_WORKER_MARKER)
    if backend is IsolationBackend.MACOS_SANDBOX:
        if marker != _SANDBOX_MARKER:
            return IsolationProbeEvidence(False, "sandbox_marker", "MISSING", (), 0)
        operation = "connect_inet_and_host_unix_denied_by_sandbox"
        canary_path = os.environ.get(_MACOS_UNIX_CANARY_ENV)
        if not canary_path:
            return IsolationProbeEvidence(
                False, operation, "UNIX_CANARY_MISSING", (), 0
            )
        unix_socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        unix_socket.settimeout(0.2)
        try:
            unix_socket.connect(canary_path)
        except OSError as error:
            if error.errno not in {errno.EPERM, errno.EACCES}:
                return IsolationProbeEvidence(
                    False,
                    operation,
                    f"UNIX_{errno.errorcode.get(error.errno or -1, 'UNKNOWN')}",
                    (),
                    0,
                )
            unix_error_name = errno.errorcode.get(error.errno or -1, "UNKNOWN")
        else:
            return IsolationProbeEvidence(
                False, operation, "UNIX_CONNECTION_ALLOWED", (), 0
            )
        finally:
            unix_socket.close()
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(0.2)
        try:
            sock.connect(("127.0.0.1", 9))
        except OSError as error:
            error_name = errno.errorcode.get(error.errno or -1, "UNKNOWN")
            return IsolationProbeEvidence(
                error.errno in {errno.EPERM, errno.EACCES},
                operation,
                f"UNIX_{unix_error_name}+INET_{error_name}",
                (),
                0,
            )
        finally:
            sock.close()
        return IsolationProbeEvidence(False, operation, "CONNECTION_ALLOWED", (), 0)
    if marker != _DOCKER_MARKER or platform.system() != "Linux":
        return IsolationProbeEvidence(False, "docker_marker", "MISSING", (), 1)
    try:
        interfaces = tuple(
            sorted(item.name for item in Path("/sys/class/net").iterdir())
        )
    except OSError:
        interfaces = ()
    routes = _linux_non_loopback_routes()
    topology_passed = interfaces == ("lo",) and not routes
    operation = "network_namespace_has_loopback_only"
    if not topology_passed:
        return IsolationProbeEvidence(
            False,
            operation,
            "NON_ISOLATED_TOPOLOGY",
            interfaces,
            len(routes),
        )
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(0.2)
    try:
        sock.connect(("192.0.2.1", 9))
    except OSError as error:
        error_name = errno.errorcode.get(error.errno or -1, "UNKNOWN")
        return IsolationProbeEvidence(
            error.errno in {errno.ENETUNREACH, errno.EHOSTUNREACH},
            operation,
            error_name,
            interfaces,
            0,
        )
    finally:
        sock.close()
    return IsolationProbeEvidence(
        False,
        operation,
        "CONNECTION_ALLOWED",
        interfaces,
        0,
    )


def _write_new(path: Path, content: bytes) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags, 0o600)
    except OSError as error:
        raise IsolationExecutionError(
            f"cannot create isolation artifact: {path}"
        ) from error
    try:
        written = 0
        while written < len(content):
            count = os.write(descriptor, content[written:])
            if count <= 0:
                _fail(IsolationExecutionError, "short write in isolation receipt")
            written += count
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _make_staging_tree_writable(root: Path) -> None:
    try:
        root_info = os.lstat(root)
    except OSError:
        return
    if stat.S_ISLNK(root_info.st_mode) or not stat.S_ISDIR(root_info.st_mode):
        return
    for current, directory_names, file_names in os.walk(root, topdown=False):
        current_path = Path(current)
        for name in file_names:
            with suppress(OSError):
                path = current_path / name
                info = os.lstat(path)
                if stat.S_ISREG(info.st_mode) and not stat.S_ISLNK(info.st_mode):
                    os.chmod(path, 0o600, follow_symlinks=False)
        for name in directory_names:
            with suppress(OSError):
                path = current_path / name
                info = os.lstat(path)
                if stat.S_ISDIR(info.st_mode) and not stat.S_ISLNK(info.st_mode):
                    os.chmod(path, 0o700, follow_symlinks=False)
        with suppress(OSError):
            info = os.lstat(current_path)
            if stat.S_ISDIR(info.st_mode) and not stat.S_ISLNK(info.st_mode):
                os.chmod(current_path, 0o700, follow_symlinks=False)


def _checked_remove_private_temporary_tree(
    root: Path,
    *,
    expected_parent: Path,
    expected_prefix: str,
    context: str,
    primary_error: BaseException | None = None,
) -> None:
    absolute = Path(os.path.abspath(root))
    parent = Path(os.path.abspath(expected_parent))
    try:
        parent_info = os.lstat(parent)
    except OSError as error:
        raise IsolationExecutionError(
            f"cannot inspect the expected {context} parent"
        ) from error
    if (
        not expected_prefix
        or stat.S_ISLNK(parent_info.st_mode)
        or not stat.S_ISDIR(parent_info.st_mode)
        or absolute.parent != parent
        or not absolute.name.startswith(expected_prefix)
        or absolute.parent == absolute
    ):
        _fail(
            IsolationExecutionError,
            f"refusing to remove an unrecognized {context} path",
        )
    try:
        info = os.lstat(absolute)
    except FileNotFoundError:
        return
    except OSError as error:
        cleanup_error = IsolationExecutionError(f"cannot inspect {context} for cleanup")
        if primary_error is not None:
            raise cleanup_error from primary_error
        raise cleanup_error from error
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
        _fail(
            IsolationExecutionError,
            f"{context} cleanup requires a real directory",
        )
    last_error: OSError | None = None
    for attempt in range(2):
        if attempt:
            try:
                current_info = os.lstat(absolute)
            except FileNotFoundError:
                return
            except OSError as error:
                last_error = error
                continue
            if stat.S_ISLNK(current_info.st_mode) or not stat.S_ISDIR(
                current_info.st_mode
            ):
                cleanup_error = IsolationExecutionError(
                    f"{context} changed type during cleanup"
                )
                if primary_error is not None:
                    raise cleanup_error from primary_error
                raise cleanup_error
            _make_staging_tree_writable(absolute)
        try:
            shutil.rmtree(absolute)
        except OSError as error:
            last_error = error
            continue
        try:
            os.lstat(absolute)
        except FileNotFoundError:
            return
        except OSError as error:
            last_error = error
            continue
    cleanup_error = IsolationExecutionError(f"{context} cleanup failed after retry")
    if primary_error is not None:
        raise cleanup_error from primary_error
    if last_error is not None:
        raise cleanup_error from last_error
    raise cleanup_error


@contextmanager
def _macos_unix_canary() -> Iterator[Path]:
    temporary_parent = Path("/private/tmp")
    if not temporary_parent.is_dir():
        temporary_parent = Path(tempfile.gettempdir()).resolve(strict=True)
    staging = Path(
        tempfile.mkdtemp(prefix="policytwin-unix-canary-", dir=temporary_parent)
    ).resolve(strict=True)
    path = staging / "probe.sock"
    listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        listener.bind(str(path))
        listener.listen(1)
        yield path
    finally:
        listener.close()
        _checked_remove_private_temporary_tree(
            staging,
            expected_parent=temporary_parent,
            expected_prefix="policytwin-unix-canary-",
            context="macOS network canary staging",
            primary_error=sys.exc_info()[1],
        )


@contextmanager
def prepare_docker_source_snapshot(
    project_root: Path,
) -> Iterator[DockerSourceSnapshot]:
    """Export clean HEAD, safely extract it, and yield a verified read-only tree."""
    # macOS commonly spells its temporary root through the ``/var`` symlink.
    # Resolve the freshly-created private directory before it is ever used as a
    # Docker-copy source; the later no-symlink validation then sees the real
    # ``/private/var/...`` ancestry instead of rejecting a safe OS temp path.
    temporary_parent = Path(tempfile.gettempdir()).resolve(strict=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix="policytwin-frozen-source-",
            dir=temporary_parent,
        )
    ).resolve(strict=True)
    try:
        archive_path = staging / "source.tar"
        try:
            archive_receipt = export_source_snapshot(project_root, archive_path)
            snapshot_root = extract_source_snapshot(
                archive_path, staging / "workspace", archive_receipt
            )
        except SourcePreflightError as error:
            raise IsolationUnavailableError(
                f"committed source snapshot preflight failed: {error}"
            ) from error
        manifest = archive_receipt.source_manifest
        manifest_path = staging / "source-manifest.json"
        _write_new(manifest_path, canonical_json_bytes(manifest.to_mapping()))
        # ``docker cp`` preserves this mode and writes the private-layer copy
        # as root.  World-readable, host-private staging lets the configured
        # non-root final UID verify it without granting any write permission.
        manifest_path.chmod(0o444)
        prepared = DockerSourceSnapshot(
            root=snapshot_root,
            archive_path=archive_path,
            archive_sha256=archive_receipt.archive_sha256,
            archive_size_bytes=archive_receipt.archive_size_bytes,
            manifest_path=manifest_path,
            manifest=manifest,
            git_cli_identity=archive_receipt.git_cli_identity,
            identity=_committed_source_identity(
                manifest,
                archive_sha256=archive_receipt.archive_sha256,
                archive_size_bytes=archive_receipt.archive_size_bytes,
                git_cli_identity=archive_receipt.git_cli_identity,
            ),
        )
        yield _verify_prepared_docker_snapshot(prepared)
    finally:
        _checked_remove_private_temporary_tree(
            staging,
            expected_parent=temporary_parent,
            expected_prefix="policytwin-frozen-source-",
            context="Docker source snapshot staging",
            primary_error=sys.exc_info()[1],
        )


def _receipt_source_manifest_path(path: Path) -> Path:
    return path.with_name(f"{path.name}{_SOURCE_MANIFEST_SUFFIX}")


def _receipt_docker_runtime_path(path: Path) -> Path:
    return path.with_name(f"{path.name}{_DOCKER_RUNTIME_SUFFIX}")


def _receipt_macos_runtime_path(path: Path) -> Path:
    return path.with_name(f"{path.name}{_MACOS_RUNTIME_SUFFIX}")


def _all_reserved_output_names(receipt_path: Path) -> frozenset[str]:
    return frozenset(
        {
            receipt_path.name,
            f"{receipt_path.name}.sha256",
            _receipt_source_manifest_path(receipt_path).name,
            _receipt_docker_runtime_path(receipt_path).name,
            _receipt_macos_runtime_path(receipt_path).name,
        }
    )


def _expected_companion_names(
    receipt_path: Path, receipt: IsolationReceipt
) -> frozenset[str]:
    names = {
        receipt_path.name,
        f"{receipt_path.name}.sha256",
        (
            _receipt_docker_runtime_path(receipt_path).name
            if receipt.backend is IsolationBackend.DOCKER_NONE
            else _receipt_macos_runtime_path(receipt_path).name
        ),
    }
    if (
        receipt.backend is IsolationBackend.DOCKER_NONE
        and receipt.status is not IsolationJobStatus.INPUT_FAILED
    ):
        names.add(_receipt_source_manifest_path(receipt_path).name)
    return frozenset(names)


def _require_reserved_outputs_absent(receipt_path: Path) -> None:
    for name in sorted(_all_reserved_output_names(receipt_path)):
        candidate = receipt_path.parent / name
        try:
            os.lstat(candidate)
        except FileNotFoundError:
            continue
        except OSError as error:
            raise IsolationExecutionError(
                "cannot inspect a reserved isolation companion path"
            ) from error
        _fail(
            IsolationExecutionError,
            f"target output pre-created reserved isolation companion: {name}",
        )


def _validated_macos_runtime_controls(
    value: Mapping[str, object],
) -> dict[str, object]:
    _strict_fields(
        value,
        frozenset(
            {
                "backend",
                "close_fds",
                "environment_launcher_path",
                "environment_launcher_sha256",
                "environment_launcher_size_bytes",
                "launcher_environment_sha256",
                "probe",
                "profile_sha256",
                "sandbox_exec_path",
                "sandbox_exec_sha256",
                "sandbox_exec_size_bytes",
                "schema_version",
            }
        ),
        "macOS runtime controls",
    )
    if value["sandbox_exec_path"] != str(_MACOS_SANDBOX_EXEC_PATH):
        _fail(IsolationIntegrityError, "macOS sandbox executable path differs")
    if value["environment_launcher_path"] != str(_MACOS_ENV_PATH):
        _fail(IsolationIntegrityError, "macOS environment launcher path differs")
    sha256 = _sha256(value["sandbox_exec_sha256"], "macOS sandbox executable sha256")
    size_bytes = value["sandbox_exec_size_bytes"]
    if isinstance(size_bytes, bool) or not isinstance(size_bytes, int):
        _fail(
            IsolationIntegrityError,
            "macOS sandbox executable size must be an integer",
        )
    environment_sha256 = _sha256(
        value["environment_launcher_sha256"],
        "macOS environment launcher executable sha256",
    )
    environment_size_bytes = value["environment_launcher_size_bytes"]
    if isinstance(environment_size_bytes, bool) or not isinstance(
        environment_size_bytes, int
    ):
        _fail(
            IsolationIntegrityError,
            "macOS environment launcher executable size must be an integer",
        )
    expected = _expected_macos_runtime_controls(
        sandbox_exec_sha256=sha256,
        sandbox_exec_size_bytes=size_bytes,
        environment_launcher_sha256=environment_sha256,
        environment_launcher_size_bytes=environment_size_bytes,
    )
    if dict(value) != expected:
        _fail(
            IsolationIntegrityError,
            "macOS runtime controls are not the closed launcher profile",
        )
    return expected


def _validated_docker_runtime_controls(
    value: Mapping[str, object],
) -> dict[str, object]:
    expected_fields = frozenset(
        {
            "actual_image_id",
            "auto_remove",
            "backend",
            "binds",
            "bootstrap_sha256",
            "cap_add",
            "cap_drop",
            "dangerous_environment_keys",
            "daemon_endpoint",
            "docker_cli_environment_sha256",
            "docker_cli_path",
            "docker_cli_sha256",
            "docker_cli_size_bytes",
            "device_requests",
            "devices",
            "entrypoint",
            "healthcheck_disabled",
            "image",
            "mounts",
            "network_mode",
            "no_new_privileges",
            "ipc_mode",
            "pid_mode",
            "pids_limit",
            "privileged",
            "readonly_rootfs",
            "schema_version",
            "source_transport",
            "security_opt",
            "tmpfs",
            "user",
            "working_dir",
        }
    )
    _strict_fields(value, expected_fields, "Docker runtime controls")
    user = _text(value["user"], "Docker runtime user", maximum=64)
    image = _immutable_image(_text(value["image"], "Docker runtime image", maximum=512))
    actual_image_id = _text(
        value["actual_image_id"], "Docker actual image ID", maximum=71
    )
    if value["docker_cli_path"] != str(_DOCKER_CLI_PATH):
        _fail(IsolationIntegrityError, "Docker CLI path differs from the fixed path")
    docker_cli_sha256 = _sha256(value["docker_cli_sha256"], "docker_cli_sha256")
    docker_cli_size_bytes = value["docker_cli_size_bytes"]
    if isinstance(docker_cli_size_bytes, bool) or not isinstance(
        docker_cli_size_bytes, int
    ):
        _fail(IsolationIntegrityError, "Docker CLI size must be an integer")
    expected = _expected_docker_runtime_controls(
        docker_image=image,
        actual_image_id=actual_image_id,
        docker_cli_sha256=docker_cli_sha256,
        docker_cli_size_bytes=docker_cli_size_bytes,
        user=user,
    )
    if dict(value) != expected:
        _fail(
            IsolationIntegrityError,
            "Docker runtime controls are not the closed no-bind profile",
        )
    return expected


def write_docker_runtime_attestation(
    receipt_path: Path, controls: Mapping[str, object]
) -> Path:
    """Attach canonical Docker inspect evidence after the container has stopped."""
    validated = _validated_docker_runtime_controls(controls)
    path = _receipt_docker_runtime_path(receipt_path)
    parent = _real_directory(path.parent, "Docker runtime attestation parent")
    destination = parent / path.name
    _write_new(destination, canonical_json_bytes(validated))
    return destination


def write_macos_runtime_attestation(
    receipt_path: Path, controls: Mapping[str, object]
) -> Path:
    """Attach canonical fixed-launcher evidence after ``sandbox-exec`` stops."""
    validated = _validated_macos_runtime_controls(controls)
    path = _receipt_macos_runtime_path(receipt_path)
    parent = _real_directory(path.parent, "macOS runtime attestation parent")
    destination = parent / path.name
    _write_new(destination, canonical_json_bytes(validated))
    return destination


def write_isolation_receipt(
    receipt: IsolationReceipt,
    path: Path,
    *,
    source_manifest_bytes: bytes | None = None,
) -> None:
    """Create a canonical receipt and SHA-256 sidecar without overwriting files."""
    if not isinstance(receipt, IsolationReceipt):
        _fail(IsolationExecutionError, "receipt must be an IsolationReceipt")
    parent = _real_directory(path.parent, "receipt parent")
    if path.parent != parent:
        path = parent / path.name
    source_manifest_path = _receipt_source_manifest_path(path)
    if receipt.backend is IsolationBackend.MACOS_SANDBOX:
        if source_manifest_bytes is not None:
            _fail(
                IsolationExecutionError,
                "macOS live-checkout receipt cannot attach a committed manifest",
            )
    elif source_manifest_bytes is not None:
        try:
            mapping = read_json_bytes_object(
                source_manifest_bytes, source=source_manifest_path.name
            )
            manifest = SourceSnapshotManifest.from_mapping(mapping)
        except (MetadataValidationError, SourcePreflightError) as error:
            raise IsolationExecutionError(
                "cannot attach an invalid committed source manifest"
            ) from error
        if canonical_json_bytes(
            mapping
        ) != source_manifest_bytes or not _manifest_matches_source_identity(
            manifest, receipt.source_identity
        ):
            _fail(
                IsolationExecutionError,
                "attached source manifest differs from the receipt identity",
            )
        _write_new(source_manifest_path, source_manifest_bytes)
    elif receipt.status is not IsolationJobStatus.INPUT_FAILED:
        _fail(
            IsolationExecutionError,
            "Docker receipt requires its complete committed source manifest",
        )
    content = canonical_json_bytes(receipt.to_mapping())
    _write_new(path, content)
    _write_new(path.with_name(f"{path.name}.sha256"), f"{receipt.sha256}\n".encode())
    descriptor = os.open(parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _read_runtime_controls_attestation(
    content: bytes, *, source: str, name: str
) -> Mapping[str, object]:
    try:
        mapping = read_json_bytes_object(content, source=source)
    except MetadataValidationError as error:
        raise IsolationIntegrityError(str(error)) from error
    if canonical_json_bytes(mapping) != content:
        _fail(IsolationIntegrityError, f"{name} is not canonical JSON")
    return mapping


@dataclass(frozen=True)
class _StableDirectoryMember:
    content: bytes
    stat_identity: tuple[int, ...]


def _directory_binding_identity(info: os.stat_result) -> tuple[int, int, int]:
    return (info.st_dev, info.st_ino, stat.S_IFMT(info.st_mode))


@contextmanager
def _held_root_anchored_directory(
    path: Path, *, name: str, error_type: type[IsolationError]
) -> Iterator[int]:
    """Pin a symlink-free absolute directory path for one complete audit.

    Every component is opened relative to a descriptor for its already pinned
    parent.  All descriptors remain open until the caller finishes, and each
    parent/name binding is checked again before success is returned.
    """
    absolute = Path(os.path.abspath(path.expanduser()))
    if absolute.anchor != os.sep:
        _fail(error_type, f"{name} must have an absolute POSIX path")
    directory_flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )
    descriptors: list[int] = []
    bindings: list[tuple[int, str, int, tuple[int, int, int]]] = []
    completed = False
    try:
        try:
            current_descriptor = os.open(absolute.anchor, directory_flags)
        except OSError as error:
            raise error_type(f"cannot open the root anchor for {name}") from error
        descriptors.append(current_descriptor)
        try:
            root_info = os.fstat(current_descriptor)
        except OSError as error:
            raise error_type(f"cannot inspect the root anchor for {name}") from error
        if not stat.S_ISDIR(root_info.st_mode) or stat.S_ISLNK(root_info.st_mode):
            _fail(error_type, f"the root anchor for {name} is not a directory")

        for part in absolute.parts[1:]:
            try:
                before = os.stat(
                    part,
                    dir_fd=current_descriptor,
                    follow_symlinks=False,
                )
            except OSError as error:
                raise error_type(f"{name} does not exist") from error
            if stat.S_ISLNK(before.st_mode) or not stat.S_ISDIR(before.st_mode):
                _fail(error_type, f"{name} contains a symbolic link or non-directory")
            try:
                child_descriptor = os.open(
                    part, directory_flags, dir_fd=current_descriptor
                )
            except OSError as error:
                raise error_type(f"cannot safely open {name}") from error
            descriptors.append(child_descriptor)
            try:
                opened = os.fstat(child_descriptor)
                after_path = os.stat(
                    part,
                    dir_fd=current_descriptor,
                    follow_symlinks=False,
                )
            except OSError as error:
                raise error_type(f"{name} changed while being opened") from error
            expected = _directory_binding_identity(before)
            if (
                _directory_binding_identity(opened) != expected
                or _directory_binding_identity(after_path) != expected
            ):
                _fail(error_type, f"{name} changed while being opened")
            bindings.append((current_descriptor, part, child_descriptor, expected))
            current_descriptor = child_descriptor

        yield current_descriptor
        completed = True
    finally:
        validation_error: IsolationError | None = None
        if completed:
            for parent_descriptor, part, child_descriptor, expected in bindings:
                try:
                    opened = os.fstat(child_descriptor)
                    at_path = os.stat(
                        part,
                        dir_fd=parent_descriptor,
                        follow_symlinks=False,
                    )
                except OSError as error:
                    validation_error = error_type(
                        f"{name} path changed during the audit"
                    )
                    validation_error.__cause__ = error
                    break
                if (
                    _directory_binding_identity(opened) != expected
                    or _directory_binding_identity(at_path) != expected
                ):
                    validation_error = error_type(
                        f"{name} path changed during the audit"
                    )
                    break
        for descriptor in reversed(descriptors):
            with suppress(OSError):
                os.close(descriptor)
        if validation_error is not None:
            raise validation_error


def _read_stable_directory_member(
    parent_descriptor: int,
    name: str,
    *,
    display_name: str,
    maximum_bytes: int,
) -> _StableDirectoryMember:
    try:
        before = os.stat(name, dir_fd=parent_descriptor, follow_symlinks=False)
    except OSError as error:
        raise IsolationIntegrityError(f"{display_name} is missing") from error
    if (
        stat.S_ISLNK(before.st_mode)
        or not stat.S_ISREG(before.st_mode)
        or before.st_nlink != 1
        or bool(before.st_mode & 0o7000)
        or before.st_size > maximum_bytes
    ):
        _fail(
            IsolationIntegrityError,
            f"{display_name} must be a bounded singly-linked regular file "
            "without special permissions",
        )
    flags = (
        os.O_RDONLY
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NONBLOCK", 0)
    )
    try:
        descriptor = os.open(name, flags, dir_fd=parent_descriptor)
    except OSError as error:
        raise IsolationIntegrityError(f"cannot safely open {display_name}") from error
    content = bytearray()
    try:
        opened = os.fstat(descriptor)
        expected = _output_stat_identity(before)
        if _output_stat_identity(opened) != expected:
            _fail(IsolationIntegrityError, f"{display_name} changed while being opened")
        while True:
            try:
                chunk = os.read(descriptor, min(1024 * 1024, maximum_bytes + 1))
            except OSError as error:
                raise IsolationIntegrityError(f"cannot read {display_name}") from error
            if not chunk:
                break
            content.extend(chunk)
            if len(content) > maximum_bytes or len(content) > before.st_size:
                _fail(
                    IsolationIntegrityError, f"{display_name} changed while being read"
                )
        after_open = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    try:
        after_path = os.stat(name, dir_fd=parent_descriptor, follow_symlinks=False)
    except OSError as error:
        raise IsolationIntegrityError(
            f"{display_name} disappeared while being read"
        ) from error
    if (
        len(content) != before.st_size
        or _output_stat_identity(after_open) != expected
        or _output_stat_identity(after_path) != expected
    ):
        _fail(IsolationIntegrityError, f"{display_name} changed while being read")
    return _StableDirectoryMember(bytes(content), expected)


def _revalidate_directory_member(
    parent_descriptor: int,
    name: str,
    member: _StableDirectoryMember,
    *,
    display_name: str,
) -> None:
    try:
        current = os.stat(name, dir_fd=parent_descriptor, follow_symlinks=False)
    except OSError as error:
        raise IsolationIntegrityError(
            f"{display_name} changed during receipt audit"
        ) from error
    if _output_stat_identity(current) != member.stat_identity:
        _fail(IsolationIntegrityError, f"{display_name} changed during receipt audit")


def _directory_member_exists(parent_descriptor: int, name: str) -> bool:
    try:
        os.stat(name, dir_fd=parent_descriptor, follow_symlinks=False)
    except FileNotFoundError:
        return False
    except OSError as error:
        raise IsolationIntegrityError(
            "cannot inspect an isolation receipt companion"
        ) from error
    return True


def _source_manifest_from_bytes(
    content: bytes, *, source: str
) -> SourceSnapshotManifest:
    try:
        mapping = read_json_bytes_object(content, source=source)
        if canonical_json_bytes(mapping) != content:
            _fail(
                IsolationIntegrityError,
                "attached source manifest is not canonical JSON",
            )
        return SourceSnapshotManifest.from_mapping(mapping)
    except (MetadataValidationError, SourcePreflightError) as error:
        raise IsolationIntegrityError(
            "cannot verify attached committed source manifest"
        ) from error


def audit_isolation_receipt(
    path: Path, *, require_success: bool = True
) -> IsolationReceipt:
    """Verify canonical bytes, the sidecar seal, schema, and optional success state."""
    receipt_name = path.name
    sidecar_name = f"{receipt_name}.sha256"
    source_manifest_name = f"{receipt_name}{_SOURCE_MANIFEST_SUFFIX}"
    docker_runtime_name = f"{receipt_name}{_DOCKER_RUNTIME_SUFFIX}"
    macos_runtime_name = f"{receipt_name}{_MACOS_RUNTIME_SUFFIX}"
    with _held_root_anchored_directory(
        path.parent,
        name="receipt parent",
        error_type=IsolationIntegrityError,
    ) as parent_descriptor:
        receipt_member = _read_stable_directory_member(
            parent_descriptor,
            receipt_name,
            display_name="isolation receipt",
            maximum_bytes=_MAX_RECEIPT_BYTES,
        )
        sidecar_member = _read_stable_directory_member(
            parent_descriptor,
            sidecar_name,
            display_name="isolation receipt SHA-256 seal",
            maximum_bytes=_MAX_RECEIPT_BYTES,
        )
        content = receipt_member.content
        try:
            mapping = read_json_bytes_object(content, source=receipt_name)
        except MetadataValidationError as error:
            raise IsolationIntegrityError(str(error)) from error
        if canonical_json_bytes(mapping) != content:
            _fail(IsolationIntegrityError, "isolation receipt is not canonical JSON")
        receipt = IsolationReceipt.from_mapping(mapping)
        try:
            seal = sidecar_member.content.decode("ascii")
        except UnicodeError as error:
            raise IsolationIntegrityError(
                "cannot read isolation receipt seal"
            ) from error
        if seal != f"{receipt.sha256}\n":
            _fail(
                IsolationIntegrityError,
                "isolation receipt SHA-256 seal does not match",
            )

        members = {
            receipt_name: (receipt_member, "isolation receipt"),
            sidecar_name: (sidecar_member, "isolation receipt SHA-256 seal"),
        }
        absent_members: dict[str, str] = {}

        def require_absent(name: str, message: str) -> None:
            if _directory_member_exists(parent_descriptor, name):
                _fail(IsolationIntegrityError, message)
            absent_members[name] = message

        if receipt.backend is IsolationBackend.MACOS_SANDBOX:
            require_absent(
                source_manifest_name,
                "macOS network-only receipt must not claim a committed source manifest",
            )
            require_absent(
                docker_runtime_name,
                "macOS receipt must not claim Docker runtime controls",
            )
        elif receipt.status is IsolationJobStatus.INPUT_FAILED:
            require_absent(
                source_manifest_name,
                "Docker input failure must not claim a source manifest companion",
            )
        else:
            source_member = _read_stable_directory_member(
                parent_descriptor,
                source_manifest_name,
                display_name="attached committed source manifest",
                maximum_bytes=8 * 1024 * 1024,
            )
            members[source_manifest_name] = (
                source_member,
                "attached committed source manifest",
            )
            source_manifest = _source_manifest_from_bytes(
                source_member.content, source=source_manifest_name
            )
            if not _manifest_matches_source_identity(
                source_manifest, receipt.source_identity
            ):
                _fail(
                    IsolationIntegrityError,
                    "attached source manifest differs from receipt source identity",
                )

        if receipt.backend is IsolationBackend.DOCKER_NONE:
            require_absent(
                macos_runtime_name,
                "Docker receipt must not claim macOS runtime controls",
            )
            runtime_name = docker_runtime_name
            runtime_display_name = "Docker runtime controls attestation"
            runtime_member = _read_stable_directory_member(
                parent_descriptor,
                runtime_name,
                display_name=runtime_display_name,
                maximum_bytes=_MAX_RECEIPT_BYTES,
            )
            runtime_mapping = _read_runtime_controls_attestation(
                runtime_member.content,
                source=runtime_name,
                name=runtime_display_name,
            )
            controls = _validated_docker_runtime_controls(runtime_mapping)
            mismatch_message = (
                "Docker runtime controls differ from the receipt commitment"
            )
        else:
            runtime_name = macos_runtime_name
            runtime_display_name = "macOS runtime controls attestation"
            runtime_member = _read_stable_directory_member(
                parent_descriptor,
                runtime_name,
                display_name=runtime_display_name,
                maximum_bytes=_MAX_RECEIPT_BYTES,
            )
            runtime_mapping = _read_runtime_controls_attestation(
                runtime_member.content,
                source=runtime_name,
                name=runtime_display_name,
            )
            controls = _validated_macos_runtime_controls(runtime_mapping)
            mismatch_message = (
                "macOS runtime controls differ from the receipt commitment"
            )
        members[runtime_name] = (runtime_member, runtime_display_name)
        if _runtime_controls_sha256(controls) != receipt.runtime_controls_sha256:
            _fail(IsolationIntegrityError, mismatch_message)

        receipt_path_for_names = Path(receipt_name)
        observed_output_tree = _compute_output_tree_identity_from_descriptor(
            parent_descriptor,
            excluded_root_names=_expected_companion_names(
                receipt_path_for_names, receipt
            ),
            error_type=IsolationIntegrityError,
        )
        if observed_output_tree != receipt.output_tree:
            _fail(
                IsolationIntegrityError,
                "isolated output tree differs from the receipt commitment",
            )

        for member_name, (member, display_name) in members.items():
            _revalidate_directory_member(
                parent_descriptor,
                member_name,
                member,
                display_name=display_name,
            )
        for member_name, message in absent_members.items():
            if _directory_member_exists(parent_descriptor, member_name):
                _fail(IsolationIntegrityError, message)

        if require_success and not receipt.successful:
            _fail(
                IsolationIntegrityError,
                f"isolated evaluation did not complete: {receipt.status.value}",
            )
        return receipt


def _target_environment() -> dict[str, str]:
    allowed = {
        _WORKER_MARKER,
        "HOME",
        "LANG",
        "LC_ALL",
        "PATH",
        "PYTHONDONTWRITEBYTECODE",
        "PYTHONHASHSEED",
        "PYTHONPATH",
        "TMPDIR",
        "TZ",
    }
    return {key: value for key, value in os.environ.items() if key in allowed}


def run_isolation_worker(
    *,
    backend: IsolationBackend,
    isolation_config_sha256: str,
    runtime_controls_sha256: str | None = None,
    receipt_path: Path,
    command: Sequence[str],
    docker_image: str | None = None,
    source_identity: IsolationSourceIdentity | None = None,
    source_root: Path | None = None,
    source_manifest_path: Path | None = None,
) -> IsolationReceipt:
    """Probe the already-entered OS boundary, then run and audit the target."""
    target = _validated_command(command)
    _validated_receipt_name(receipt_path.name)
    receipt_parent = _real_directory(receipt_path.parent, "receipt parent")
    receipt_path = receipt_parent / receipt_path.name
    reserved_output_names = _all_reserved_output_names(receipt_path)
    _require_reserved_outputs_absent(receipt_path)
    initial_output_tree = _compute_output_tree_identity(
        receipt_parent,
        excluded_root_names=reserved_output_names,
        error_type=IsolationExecutionError,
    )
    if initial_output_tree != _empty_output_tree_identity():
        _fail(
            IsolationExecutionError,
            "isolated output directory must be empty before target execution",
        )
    selected_source = source_identity
    if selected_source is None:
        if backend is IsolationBackend.MACOS_SANDBOX:
            selected_source = _live_checkout_source_identity()
        else:
            _fail(
                IsolationExecutionError,
                "Docker worker requires a committed source identity",
            )
    selected_runtime_controls = runtime_controls_sha256
    if selected_runtime_controls is None:
        _fail(
            IsolationExecutionError,
            f"{backend.value} worker requires runtime controls evidence",
        )
    _sha256(selected_runtime_controls, "runtime_controls_sha256")
    expected_config = _isolation_config_sha256(
        backend,
        docker_image=docker_image,
        source_identity=selected_source,
        runtime_controls_sha256=selected_runtime_controls,
    )
    if isolation_config_sha256 != expected_config:
        _fail(
            IsolationExecutionError,
            "worker isolation configuration commitment does not match the backend",
        )
    started = datetime.now(UTC).isoformat()
    source_manifest_bytes: bytes | None = None
    input_valid = True
    if backend is IsolationBackend.DOCKER_NONE:
        if source_root is None or source_manifest_path is None:
            input_valid = False
        else:
            try:
                source_manifest = read_source_snapshot_manifest(source_manifest_path)
                verify_source_snapshot_directory(source_root, source_manifest)
                if not _manifest_matches_source_identity(
                    source_manifest, selected_source
                ):
                    input_valid = False
                else:
                    source_manifest_bytes = canonical_json_bytes(
                        source_manifest.to_mapping()
                    )
            except (OSError, SourcePreflightError):
                input_valid = False
    elif source_root is not None or source_manifest_path is not None:
        input_valid = False
    probe = (
        _probe_os_isolation(backend)
        if input_valid
        else IsolationProbeEvidence(
            False,
            "source_snapshot_verification",
            "INPUT_INVALID",
            (),
            0,
        )
    )
    exit_code: int | None = None
    if not input_valid:
        status = IsolationJobStatus.INPUT_FAILED
    elif probe.passed:
        try:
            completed = subprocess.run(
                target,
                check=False,
                close_fds=True,
                env=_target_environment(),
            )
            exit_code = completed.returncode if completed.returncode >= 0 else 128
            exit_code = min(exit_code, 255)
        except OSError:
            exit_code = 126
        status = (
            IsolationJobStatus.COMPLETED
            if exit_code == 0
            else IsolationJobStatus.TARGET_FAILED
        )
    else:
        status = IsolationJobStatus.PROBE_FAILED
    _require_reserved_outputs_absent(receipt_path)
    output_tree = _compute_output_tree_identity(
        receipt_parent,
        excluded_root_names=reserved_output_names,
        error_type=IsolationExecutionError,
    )
    if (
        status
        in {
            IsolationJobStatus.INPUT_FAILED,
            IsolationJobStatus.PROBE_FAILED,
        }
        and output_tree != _empty_output_tree_identity()
    ):
        _fail(
            IsolationExecutionError,
            "pre-target failure output tree is not empty",
        )
    receipt = IsolationReceipt(
        schema_version=ISOLATION_RECEIPT_SCHEMA_VERSION,
        backend=backend,
        runtime_identity=_runtime_identity(backend, docker_image=docker_image),
        source_identity=selected_source,
        runtime_controls_sha256=selected_runtime_controls,
        isolation_config_sha256=expected_config,
        command_sha256=sha256_bytes(canonical_json_bytes(list(target))),
        command_name=Path(target[0]).name,
        output_tree=output_tree,
        probe=probe,
        status=status,
        target_exit_code=exit_code,
        started_at=started,
        finished_at=datetime.now(UTC).isoformat(),
    )
    write_isolation_receipt(
        receipt,
        receipt_path,
        source_manifest_bytes=source_manifest_bytes,
    )
    return receipt


def _docker_subprocess(
    command: Sequence[str], *, capture_output: bool = True
) -> subprocess.CompletedProcess[bytes]:
    if not command or command[0] != str(_DOCKER_CLI_PATH):
        _fail(
            IsolationUnavailableError,
            "Docker lifecycle commands must use the fixed /usr/bin/docker CLI",
        )
    try:
        return subprocess.run(
            tuple(command),
            check=False,
            close_fds=True,
            capture_output=capture_output,
            env=_docker_cli_environment(),
        )
    except OSError as error:
        raise IsolationUnavailableError(
            "Docker lifecycle command could not start; no fallback was used"
        ) from error


def _docker_local_image_id(docker: str, image: str) -> str:
    completed = _docker_subprocess(
        (docker, "image", "inspect", "--format={{.Id}}", image)
    )
    if completed.returncode != 0:
        _fail(
            IsolationUnavailableError,
            "immutable Docker image is not staged locally; no pull was attempted",
        )
    try:
        value = completed.stdout.decode("ascii").strip()
    except UnicodeError as error:
        raise IsolationUnavailableError("Docker image ID is not ASCII") from error
    return _validated_image_id(value)


def _checked_docker_cleanup(
    command: Sequence[str],
    *,
    context: str,
    primary_error: BaseException | None = None,
) -> None:
    last_error: BaseException | None = None
    for _attempt in range(2):
        try:
            completed = _docker_subprocess(command)
        except IsolationError as error:
            last_error = error
            continue
        if completed.returncode == 0:
            return
        last_error = IsolationExecutionError(
            f"{context} returned exit code {completed.returncode}"
        )
    cleanup_error = IsolationExecutionError(f"{context} failed after retry")
    if primary_error is not None:
        raise cleanup_error from primary_error
    if last_error is not None:
        raise cleanup_error from last_error
    raise cleanup_error


def _docker_container_id(completed: subprocess.CompletedProcess[bytes]) -> str:
    if completed.returncode != 0:
        _fail(IsolationExecutionError, "Docker container creation failed")
    try:
        identifier = completed.stdout.decode("ascii").strip()
    except UnicodeError as error:
        raise IsolationExecutionError(
            "Docker returned a non-ASCII container ID"
        ) from error
    if _DOCKER_ID_PATTERN.fullmatch(identifier) is None:
        _fail(IsolationExecutionError, "Docker returned an invalid container ID")
    return identifier


def _docker_inspect(docker: str, container_id: str) -> Mapping[str, object]:
    completed = _docker_subprocess(
        (docker, "inspect", "--type=container", container_id)
    )
    if completed.returncode != 0 or len(completed.stdout) > 1024 * 1024:
        _fail(IsolationExecutionError, "Docker container inspection failed")
    try:
        value = json.loads(completed.stdout)
    except (UnicodeError, json.JSONDecodeError) as error:
        raise IsolationExecutionError(
            "Docker inspect output is invalid JSON"
        ) from error
    if (
        not isinstance(value, list)
        or len(value) != 1
        or not isinstance(value[0], Mapping)
    ):
        _fail(IsolationExecutionError, "Docker inspect returned an invalid document")
    return cast(Mapping[str, object], value[0])


def _inspect_mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        _fail(IsolationExecutionError, f"Docker inspect {name} must be an object")
    return cast(Mapping[str, object], value)


def _inspect_list(value: object, name: str) -> list[object]:
    if not isinstance(value, list):
        _fail(IsolationExecutionError, f"Docker inspect {name} must be an array")
    return value


def _inspect_optional_list(value: object, name: str) -> list[object]:
    return [] if value is None else _inspect_list(value, name)


def _inspect_final_container(
    inspect: Mapping[str, object],
    *,
    expected_image: str,
    expected_image_id: str,
    expected_docker_cli_sha256: str,
    expected_docker_cli_size_bytes: int,
    expected_user: str,
    expected_command: Sequence[str],
) -> tuple[dict[str, object], str, int | None, str]:
    identifier = inspect.get("Id")
    if (
        not isinstance(identifier, str)
        or _DOCKER_ID_PATTERN.fullmatch(identifier) is None
    ):
        _fail(IsolationExecutionError, "Docker inspect container ID is invalid")
    config = _inspect_mapping(inspect.get("Config"), "Config")
    host = _inspect_mapping(inspect.get("HostConfig"), "HostConfig")
    state = _inspect_mapping(inspect.get("State"), "State")
    if config.get("Image") != expected_image or config.get("User") != expected_user:
        _fail(IsolationExecutionError, "Docker image or user differs from the request")
    if config.get("Entrypoint") != ["/usr/local/bin/python"]:
        _fail(
            IsolationExecutionError,
            "Docker final container entrypoint is not the fixed bootstrap interpreter",
        )
    if config.get("WorkingDir") != _DOCKER_FINAL_WORKDIR:
        _fail(
            IsolationExecutionError,
            "Docker final container working directory differs from /workspace",
        )
    healthcheck = _inspect_mapping(config.get("Healthcheck"), "Config.Healthcheck")
    if healthcheck.get("Test") != ["NONE"] or set(healthcheck) != {"Test"}:
        _fail(IsolationExecutionError, "Docker image healthcheck is not disabled")
    raw_environment = _inspect_list(config.get("Env"), "Config.Env")
    dangerous_environment_keys: list[str] = []
    for item in raw_environment:
        if not isinstance(item, str) or "=" not in item:
            _fail(IsolationExecutionError, "Docker environment entry is invalid")
        key = item.split("=", 1)[0]
        if (
            key in {"PYTHONPATH", "PYTHONHOME"}
            or key.startswith("LD_")
            or key.startswith("DYLD_")
        ):
            dangerous_environment_keys.append(key)
    if dangerous_environment_keys:
        _fail(
            IsolationExecutionError,
            "Docker pre-bootstrap environment contains startup injection variables",
        )
    if inspect.get("Image") != expected_image_id:
        _fail(
            IsolationExecutionError,
            "Docker actual image ID differs from its pre-start inspection",
        )
    raw_command = config.get("Cmd")
    if not isinstance(raw_command, list) or raw_command != list(expected_command):
        _fail(
            IsolationExecutionError,
            "Docker container command differs from the hash-bound bootstrap command",
        )
    if raw_command[:3] != ["-I", "-S", "-c"]:
        _fail(
            IsolationExecutionError,
            "Docker bootstrap interpreter is not isolated from unverified source",
        )
    if sha256_bytes(str(raw_command[3]).encode("utf-8")) != DOCKER_BOOTSTRAP_SHA256:
        _fail(IsolationExecutionError, "Docker bootstrap source hash differs")
    binds = host.get("Binds")
    normalized_binds: list[object] = (
        [] if binds is None else _inspect_list(binds, "HostConfig.Binds")
    )
    if normalized_binds:
        _fail(IsolationExecutionError, "Docker final container contains host binds")
    host_mounts = host.get("Mounts")
    if host_mounts is not None:
        for raw_mount in _inspect_list(host_mounts, "HostConfig.Mounts"):
            mount = _inspect_mapping(raw_mount, "HostConfig.Mounts entry")
            if mount.get("Type") == "bind":
                _fail(
                    IsolationExecutionError,
                    "Docker final container contains a host bind mount",
                )
    raw_mounts = _inspect_list(inspect.get("Mounts"), "Mounts")
    mounts: list[dict[str, object]] = []
    output_volume: str | None = None
    for raw_mount in raw_mounts:
        mount = _inspect_mapping(raw_mount, "Mounts entry")
        mount_type = mount.get("Type")
        destination = mount.get("Destination")
        rw = mount.get("RW")
        if mount_type == "bind":
            _fail(IsolationExecutionError, "Docker runtime contains a bind mount")
        if (
            mount_type != "volume"
            or destination != _DOCKER_OUTPUT_ROOT
            or rw is not True
        ):
            _fail(
                IsolationExecutionError,
                "Docker runtime mount set differs from its private output volume",
            )
        name = mount.get("Name")
        if not isinstance(name, str) or _DOCKER_VOLUME_PATTERN.fullmatch(name) is None:
            _fail(IsolationExecutionError, "Docker output volume name is invalid")
        output_volume = name
        mounts.append({"destination": destination, "rw": rw, "type": mount_type})
    if len(mounts) != 1 or output_volume is None:
        _fail(IsolationExecutionError, "Docker output volume is missing or ambiguous")
    cap_add = _inspect_optional_list(host.get("CapAdd"), "HostConfig.CapAdd")
    if cap_add:
        _fail(IsolationExecutionError, "Docker final container adds capabilities")
    devices = _inspect_optional_list(host.get("Devices"), "HostConfig.Devices")
    device_requests = _inspect_optional_list(
        host.get("DeviceRequests"), "HostConfig.DeviceRequests"
    )
    if devices or device_requests:
        _fail(IsolationExecutionError, "Docker final container exposes host devices")
    if host.get("Privileged") is not False:
        _fail(IsolationExecutionError, "Docker final container is privileged")
    if host.get("PidMode") != "" or host.get("IpcMode") != "private":
        _fail(
            IsolationExecutionError,
            "Docker final container PID/IPC namespaces are not private",
        )
    cap_drop = host.get("CapDrop")
    normalized_cap_drop = sorted(
        str(item).upper() for item in _inspect_list(cap_drop, "HostConfig.CapDrop")
    )
    security = _inspect_list(host.get("SecurityOpt"), "HostConfig.SecurityOpt")
    if security != ["no-new-privileges:true"]:
        _fail(
            IsolationExecutionError,
            "Docker final container security options differ from the exact profile",
        )
    tmpfs = _inspect_mapping(host.get("Tmpfs"), "HostConfig.Tmpfs")
    expected_tmpfs = cast(
        Mapping[str, object],
        _expected_docker_runtime_controls(
            docker_image=expected_image,
            actual_image_id=expected_image_id,
            docker_cli_sha256=expected_docker_cli_sha256,
            docker_cli_size_bytes=expected_docker_cli_size_bytes,
            user=expected_user,
        )["tmpfs"],
    )
    raw_tmpfs_options = tmpfs.get("/tmp")
    expected_tmpfs_options = expected_tmpfs.get("/tmp")
    if (
        set(tmpfs) != {"/tmp"}
        or not isinstance(raw_tmpfs_options, str)
        or not isinstance(expected_tmpfs_options, str)
        or set(raw_tmpfs_options.split(",")) != set(expected_tmpfs_options.split(","))
    ):
        _fail(IsolationExecutionError, "Docker tmpfs controls differ")
    controls: dict[str, object] = {
        "actual_image_id": expected_image_id,
        "auto_remove": host.get("AutoRemove"),
        "backend": IsolationBackend.DOCKER_NONE.value,
        "binds": normalized_binds,
        "bootstrap_sha256": DOCKER_BOOTSTRAP_SHA256,
        "cap_add": cap_add,
        "cap_drop": normalized_cap_drop,
        "dangerous_environment_keys": dangerous_environment_keys,
        "daemon_endpoint": _DOCKER_DAEMON_ENDPOINT,
        "docker_cli_environment_sha256": _docker_cli_environment_sha256(),
        "docker_cli_path": str(_DOCKER_CLI_PATH),
        "docker_cli_sha256": expected_docker_cli_sha256,
        "docker_cli_size_bytes": expected_docker_cli_size_bytes,
        "device_requests": device_requests,
        "devices": devices,
        "entrypoint": ["/usr/local/bin/python"],
        "healthcheck_disabled": True,
        "image": expected_image,
        "mounts": mounts,
        "network_mode": host.get("NetworkMode"),
        "no_new_privileges": "no-new-privileges:true" in security,
        "ipc_mode": host.get("IpcMode"),
        "pid_mode": host.get("PidMode"),
        "pids_limit": host.get("PidsLimit"),
        "privileged": host.get("Privileged"),
        "readonly_rootfs": host.get("ReadonlyRootfs"),
        "schema_version": "policytwin.runtime-controls.v1",
        "source_transport": "private_container_layer",
        "security_opt": security,
        "tmpfs": dict(expected_tmpfs),
        "user": expected_user,
        "working_dir": config.get("WorkingDir"),
    }
    controls = _validated_docker_runtime_controls(controls)
    raw_exit_code = state.get("ExitCode")
    exit_code = (
        raw_exit_code
        if isinstance(raw_exit_code, int) and not isinstance(raw_exit_code, bool)
        else None
    )
    status = state.get("Status")
    if not isinstance(status, str):
        _fail(IsolationExecutionError, "Docker container state is invalid")
    return controls, output_volume, exit_code, status


_DOCKER_VOLUME_INIT_SOURCE = (
    "import os,sys; p='/output'; "
    "assert not os.listdir(p), 'output volume is not empty'; "
    "os.chown(p,int(sys.argv[1]),int(sys.argv[2])); os.chmod(p,0o700)"
)


def _docker_export_source(
    *,
    source_path: str,
    target_path: str,
    max_files: int,
    max_directories: int,
    max_bytes: int,
) -> str:
    if min(max_files, max_directories, max_bytes) < 1:
        _fail(IsolationExecutionError, "Docker output limits must be positive")
    return rf"""import os,shutil,stat,sys
from pathlib import Path,PurePosixPath
def safe(value):
    path = PurePosixPath(value)
    return (value and "\\" not in value and not path.is_absolute() and
            path.as_posix() == value and
            all(part not in ("", ".", "..") and
                not any(ord(character) < 32 for character in part)
                for part in path.parts))
source = Path({source_path!r})
target = Path({target_path!r})
target.mkdir(mode=0o700)
file_count = 0
directory_count = 0
total_size = 0
for current, directory_names, file_names in os.walk(source, followlinks=False):
    current_path = Path(current)
    relative_parent = current_path.relative_to(source)
    destination_parent = target / relative_parent
    destination_parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    directory_count += len(directory_names)
    if directory_count > {max_directories}:
        raise SystemExit("output directory safety limit exceeded")
    for name in directory_names:
        path = current_path / name
        info = os.lstat(path)
        relative = path.relative_to(source).as_posix()
        if (not safe(relative) or not stat.S_ISDIR(info.st_mode) or
            stat.S_ISLNK(info.st_mode)):
            raise SystemExit("unsafe output directory")
    for name in file_names:
        path = current_path / name
        info = os.lstat(path)
        relative = path.relative_to(source).as_posix()
        if (not safe(relative) or not stat.S_ISREG(info.st_mode) or
            stat.S_ISLNK(info.st_mode) or info.st_nlink != 1):
            raise SystemExit("unsafe output file")
        file_count += 1
        total_size += info.st_size
        if file_count > {max_files} or total_size > {max_bytes}:
            raise SystemExit("output safety limit exceeded")
        destination = destination_parent / name
        with path.open("rb") as reader, destination.open("xb") as writer:
            shutil.copyfileobj(reader, writer, length=1024 * 1024)
        destination.chmod(0o600)
"""


_DOCKER_EXPORT_SOURCE = _docker_export_source(
    source_path="/export",
    target_path="/policytwin-export",
    max_files=_MAX_DOCKER_OUTPUT_FILES,
    max_directories=_MAX_DOCKER_OUTPUT_DIRECTORIES,
    max_bytes=_MAX_DOCKER_OUTPUT_BYTES,
)


def _safe_output_relative(value: str) -> bool:
    if not value or "\\" in value or any(ord(character) < 32 for character in value):
        return False
    try:
        encoded = value.encode("utf-8")
    except UnicodeError:
        return False
    path = PurePosixPath(value)
    return (
        len(encoded) <= 16 * 1024
        and len(path.parts) <= 256
        and not path.is_absolute()
        and path.as_posix() == value
        and all(part not in {"", ".", ".."} for part in path.parts)
    )


@dataclass(frozen=True)
class _OutputTreeSnapshot:
    entries: tuple[bytes, ...]
    identity: IsolationOutputTreeIdentity


def _output_stat_identity(info: os.stat_result) -> tuple[int, ...]:
    """Return mutation-sensitive metadata which is deliberately not committed."""
    return (
        info.st_dev,
        info.st_ino,
        info.st_mode,
        info.st_nlink,
        info.st_uid,
        info.st_gid,
        info.st_size,
        info.st_mtime_ns,
        info.st_ctime_ns,
    )


def _output_tree_identity(
    entries: Sequence[bytes],
    *,
    file_count: int,
    directory_count: int,
    total_size_bytes: int,
) -> IsolationOutputTreeIdentity:
    decoded_entries: list[object] = []
    for entry in entries:
        try:
            decoded_entries.append(json.loads(entry))
        except (UnicodeError, json.JSONDecodeError) as error:  # pragma: no cover
            raise AssertionError("internal output-tree entry is invalid") from error
    tree_sha256 = sha256_bytes(
        canonical_json_bytes(
            {
                "entries": decoded_entries,
                "schema_version": _OUTPUT_TREE_SCHEMA_VERSION,
            }
        )
    )
    return IsolationOutputTreeIdentity(
        tree_sha256=tree_sha256,
        file_count=file_count,
        directory_count=directory_count,
        total_size_bytes=total_size_bytes,
    )


def _empty_output_tree_identity() -> IsolationOutputTreeIdentity:
    return _output_tree_identity(
        (), file_count=0, directory_count=0, total_size_bytes=0
    )


def _hash_stable_output_file(
    name: str,
    before: os.stat_result,
    *,
    parent_descriptor: int,
    error_type: type[IsolationError],
) -> str:
    if (
        stat.S_ISLNK(before.st_mode)
        or not stat.S_ISREG(before.st_mode)
        or before.st_nlink != 1
        or bool(before.st_mode & 0o7000)
    ):
        _fail(
            error_type,
            "output tree contains a linked, special, or specially-permissioned file",
        )
    flags = (
        os.O_RDONLY
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NONBLOCK", 0)
    )
    try:
        descriptor = os.open(name, flags, dir_fd=parent_descriptor)
    except OSError as error:
        raise error_type("cannot safely open an output file") from error
    digest = hashlib.sha256()
    bytes_read = 0
    try:
        opened = os.fstat(descriptor)
        if _output_stat_identity(opened) != _output_stat_identity(before):
            _fail(error_type, "output file changed while being opened")
        while True:
            try:
                chunk = os.read(descriptor, 1024 * 1024)
            except OSError as error:
                raise error_type("cannot read an output file") from error
            if not chunk:
                break
            bytes_read += len(chunk)
            if bytes_read > before.st_size:
                _fail(error_type, "output file grew while being hashed")
            digest.update(chunk)
        after_open = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    try:
        after_path = os.stat(name, dir_fd=parent_descriptor, follow_symlinks=False)
    except OSError as error:
        raise error_type("output file disappeared while being hashed") from error
    expected = _output_stat_identity(before)
    if (
        bytes_read != before.st_size
        or _output_stat_identity(after_open) != expected
        or _output_stat_identity(after_path) != expected
    ):
        _fail(error_type, "output file changed while being hashed")
    return digest.hexdigest()


def _capture_output_tree_from_descriptor_once(
    root_descriptor: int,
    *,
    excluded_root_names: frozenset[str],
    error_type: type[IsolationError],
) -> _OutputTreeSnapshot:
    try:
        root_before = os.fstat(root_descriptor)
    except OSError as error:
        raise error_type("output root is unavailable") from error
    if (
        stat.S_ISLNK(root_before.st_mode)
        or not stat.S_ISDIR(root_before.st_mode)
        or bool(root_before.st_mode & 0o7000)
    ):
        _fail(
            error_type,
            "output root must be a real directory without special permissions",
        )

    entries: list[bytes] = []
    file_count = 0
    directory_count = 0
    total_size_bytes = 0

    directory_flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )

    def visit(
        descriptor: int,
        relative_parent: PurePosixPath,
        directory_before: os.stat_result,
    ) -> None:
        nonlocal file_count, directory_count, total_size_bytes
        try:
            names = sorted(os.listdir(descriptor))
        except OSError as error:
            raise error_type("cannot traverse output tree") from error
        if (
            stat.S_ISLNK(directory_before.st_mode)
            or not stat.S_ISDIR(directory_before.st_mode)
            or bool(directory_before.st_mode & 0o7000)
        ):
            _fail(
                error_type,
                "output tree contains a linked, special, or "
                "specially-permissioned directory",
            )
        for name in names:
            if relative_parent == PurePosixPath(".") and name in excluded_root_names:
                continue
            relative = PurePosixPath(name)
            if relative_parent != PurePosixPath("."):
                relative = relative_parent / name
            relative_text = relative.as_posix()
            if not _safe_output_relative(relative_text):
                _fail(error_type, "output tree contains an unsafe path")
            try:
                before = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
            except OSError as error:
                raise error_type("output path disappeared during traversal") from error
            if stat.S_ISDIR(before.st_mode) and not stat.S_ISLNK(before.st_mode):
                if bool(before.st_mode & 0o7000):
                    _fail(
                        error_type,
                        "output tree contains a specially-permissioned directory",
                    )
                directory_count += 1
                if directory_count > _MAX_DOCKER_OUTPUT_DIRECTORIES:
                    _fail(error_type, "output tree exceeds directory safety limit")
                entries.append(
                    canonical_json_bytes(
                        {
                            "kind": "directory",
                            "path": relative_text,
                            "sha256": None,
                            "size_bytes": None,
                        }
                    )
                )
                try:
                    child_descriptor = os.open(name, directory_flags, dir_fd=descriptor)
                except OSError as error:
                    raise error_type(
                        "cannot safely open an output directory"
                    ) from error
                try:
                    opened = os.fstat(child_descriptor)
                    if _output_stat_identity(opened) != _output_stat_identity(before):
                        _fail(
                            error_type,
                            "output directory changed while being opened",
                        )
                    visit(child_descriptor, relative, opened)
                    after_open = os.fstat(child_descriptor)
                finally:
                    os.close(child_descriptor)
                try:
                    after_path = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
                except OSError as error:
                    raise error_type(
                        "output directory disappeared during traversal"
                    ) from error
                expected = _output_stat_identity(before)
                if (
                    _output_stat_identity(after_open) != expected
                    or _output_stat_identity(after_path) != expected
                ):
                    _fail(error_type, "output directory changed during traversal")
                continue
            if (
                stat.S_ISLNK(before.st_mode)
                or not stat.S_ISREG(before.st_mode)
                or before.st_nlink != 1
                or bool(before.st_mode & 0o7000)
            ):
                _fail(
                    error_type,
                    "output tree contains a linked, special, or "
                    "specially-permissioned file",
                )
            file_count += 1
            total_size_bytes += before.st_size
            if (
                file_count > _MAX_DOCKER_OUTPUT_FILES
                or total_size_bytes > _MAX_DOCKER_OUTPUT_BYTES
            ):
                _fail(error_type, "output tree exceeds file or byte safety limit")
            digest = _hash_stable_output_file(
                name,
                before,
                parent_descriptor=descriptor,
                error_type=error_type,
            )
            entries.append(
                canonical_json_bytes(
                    {
                        "kind": "file",
                        "path": relative_text,
                        "sha256": digest,
                        "size_bytes": before.st_size,
                    }
                )
            )
        try:
            directory_after = os.fstat(descriptor)
        except OSError as error:
            raise error_type("output directory disappeared during traversal") from error
        if _output_stat_identity(directory_after) != _output_stat_identity(
            directory_before
        ):
            _fail(error_type, "output directory changed during traversal")

    visit(root_descriptor, PurePosixPath("."), root_before)
    try:
        root_after = os.fstat(root_descriptor)
    except OSError as error:
        raise error_type("output root disappeared during traversal") from error
    expected_root = _output_stat_identity(root_before)
    if _output_stat_identity(root_after) != expected_root:
        _fail(error_type, "output root changed during traversal")
    entries.sort()
    return _OutputTreeSnapshot(
        entries=tuple(entries),
        identity=_output_tree_identity(
            entries,
            file_count=file_count,
            directory_count=directory_count,
            total_size_bytes=total_size_bytes,
        ),
    )


def _capture_output_tree_once(
    root: Path,
    *,
    excluded_root_names: frozenset[str],
    error_type: type[IsolationError],
) -> _OutputTreeSnapshot:
    try:
        root_before = os.lstat(root)
    except OSError as error:
        raise error_type("output root is unavailable") from error
    if (
        stat.S_ISLNK(root_before.st_mode)
        or not stat.S_ISDIR(root_before.st_mode)
        or bool(root_before.st_mode & 0o7000)
    ):
        _fail(
            error_type,
            "output root must be a real directory without special permissions",
        )
    directory_flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )
    try:
        root_descriptor = os.open(root, directory_flags)
    except OSError as error:
        raise error_type("cannot safely open output root") from error
    try:
        opened_root = os.fstat(root_descriptor)
        if _output_stat_identity(opened_root) != _output_stat_identity(root_before):
            _fail(error_type, "output root changed while being opened")
        snapshot = _capture_output_tree_from_descriptor_once(
            root_descriptor,
            excluded_root_names=excluded_root_names,
            error_type=error_type,
        )
        after_open_root = os.fstat(root_descriptor)
    finally:
        os.close(root_descriptor)
    try:
        root_after = os.lstat(root)
    except OSError as error:
        raise error_type("output root disappeared during traversal") from error
    expected_root = _output_stat_identity(root_before)
    if (
        _output_stat_identity(after_open_root) != expected_root
        or _output_stat_identity(root_after) != expected_root
    ):
        _fail(error_type, "output root changed during traversal")
    return snapshot


def _compute_output_tree_identity(
    root: Path,
    *,
    excluded_root_names: frozenset[str],
    error_type: type[IsolationError],
) -> IsolationOutputTreeIdentity:
    """Read the semantic tree twice and reject any unstable observation."""
    first = _capture_output_tree_once(
        root,
        excluded_root_names=excluded_root_names,
        error_type=error_type,
    )
    second = _capture_output_tree_once(
        root,
        excluded_root_names=excluded_root_names,
        error_type=error_type,
    )
    if first != second:
        _fail(error_type, "output tree changed between verification passes")
    return first.identity


def _compute_output_tree_identity_from_descriptor(
    root_descriptor: int,
    *,
    excluded_root_names: frozenset[str],
    error_type: type[IsolationError],
) -> IsolationOutputTreeIdentity:
    """Read one pinned semantic output tree twice and reject instability."""
    first = _capture_output_tree_from_descriptor_once(
        root_descriptor,
        excluded_root_names=excluded_root_names,
        error_type=error_type,
    )
    second = _capture_output_tree_from_descriptor_once(
        root_descriptor,
        excluded_root_names=excluded_root_names,
        error_type=error_type,
    )
    if first != second:
        _fail(error_type, "output tree changed between verification passes")
    return first.identity


def _initialize_docker_output_volume(
    *, docker: str, image: str, volume: str, uid: int, gid: int
) -> None:
    command = (
        docker,
        "create",
        "--pull=never",
        "--network=none",
        "--read-only",
        "--cap-drop=ALL",
        "--cap-add=CHOWN",
        "--cap-add=FOWNER",
        "--privileged=false",
        "--ipc=private",
        "--security-opt=no-new-privileges:true",
        "--pids-limit=32",
        "--no-healthcheck",
        "--user=0:0",
        "--entrypoint=/usr/local/bin/python",
        f"--mount=type=volume,src={volume},dst={_DOCKER_OUTPUT_ROOT}",
        "--tmpfs=/tmp:rw,noexec,nosuid,nodev,size=8m,mode=1777",
        image,
        "-I",
        "-S",
        "-c",
        _DOCKER_VOLUME_INIT_SOURCE,
        str(uid),
        str(gid),
    )
    if any("type=bind" in argument for argument in command):
        _fail(IsolationExecutionError, "Docker volume initializer contains a bind")
    container_id = _docker_container_id(_docker_subprocess(command))
    try:
        started = _docker_subprocess((docker, "start", "--attach", container_id))
        inspected = _docker_inspect(docker, container_id)
        state = _inspect_mapping(inspected.get("State"), "initializer State")
        if started.returncode != 0 or state.get("ExitCode") != 0:
            _fail(IsolationExecutionError, "Docker output volume initialization failed")
    finally:
        _checked_docker_cleanup(
            (docker, "rm", "--force", container_id),
            context="Docker output-volume initializer cleanup",
            primary_error=sys.exc_info()[1],
        )


def _validate_copied_output_tree(root: Path) -> None:
    file_count = 0
    directory_count = 0
    total_size = 0
    for current, directory_names, file_names in os.walk(root, followlinks=False):
        current_path = Path(current)
        directory_count += len(directory_names)
        if directory_count > _MAX_DOCKER_OUTPUT_DIRECTORIES:
            _fail(
                IsolationExecutionError,
                "Docker output exceeds directory safety limit",
            )
        for name in directory_names:
            path = current_path / name
            info = os.lstat(path)
            relative = path.relative_to(root).as_posix()
            if (
                not _safe_output_relative(relative)
                or not stat.S_ISDIR(info.st_mode)
                or stat.S_ISLNK(info.st_mode)
            ):
                _fail(
                    IsolationExecutionError,
                    "Docker output contains a linked or special directory",
                )
        for name in file_names:
            path = current_path / name
            info = os.lstat(path)
            relative = path.relative_to(root).as_posix()
            if (
                not _safe_output_relative(relative)
                or not stat.S_ISREG(info.st_mode)
                or stat.S_ISLNK(info.st_mode)
                or info.st_nlink != 1
            ):
                _fail(
                    IsolationExecutionError,
                    "Docker output contains a linked or special file",
                )
            file_count += 1
            total_size += info.st_size
            if (
                file_count > _MAX_DOCKER_OUTPUT_FILES
                or total_size > _MAX_DOCKER_OUTPUT_BYTES
            ):
                _fail(IsolationExecutionError, "Docker output exceeds safety limits")


def _make_private_output_staging_writable(root: Path) -> None:
    for current, directory_names, file_names in os.walk(root, topdown=False):
        current_path = Path(current)
        for name in file_names:
            path = current_path / name
            with suppress(OSError):
                info = os.lstat(path)
                if stat.S_ISREG(info.st_mode) and not stat.S_ISLNK(info.st_mode):
                    os.chmod(path, 0o600, follow_symlinks=False)
        for name in directory_names:
            path = current_path / name
            with suppress(OSError):
                info = os.lstat(path)
                if stat.S_ISDIR(info.st_mode) and not stat.S_ISLNK(info.st_mode):
                    os.chmod(path, 0o700, follow_symlinks=False)
        with suppress(OSError):
            os.chmod(current_path, 0o700, follow_symlinks=False)


def _checked_remove_private_output_staging(
    root: Path,
    *,
    output_parent: Path,
    primary_error: BaseException | None = None,
) -> None:
    parent = _real_directory(output_parent, "output parent")
    absolute = Path(os.path.abspath(root))
    if absolute.parent != parent or not absolute.name.startswith(
        ".policytwin-docker-output-"
    ):
        _fail(
            IsolationExecutionError,
            "refusing to remove an unrecognized Docker output staging path",
        )
    try:
        info = os.lstat(absolute)
    except FileNotFoundError:
        return
    except OSError as error:
        cleanup_error = IsolationExecutionError(
            "cannot inspect private Docker output staging for cleanup"
        )
        if primary_error is not None:
            raise cleanup_error from primary_error
        raise cleanup_error from error
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
        _fail(
            IsolationExecutionError,
            "private Docker output staging cleanup requires a real directory",
        )
    last_error: OSError | None = None
    for attempt in range(2):
        if attempt:
            _make_private_output_staging_writable(absolute)
        try:
            shutil.rmtree(absolute)
        except OSError as error:
            last_error = error
            continue
        if not absolute.exists() and not absolute.is_symlink():
            return
    cleanup_error = IsolationExecutionError(
        "private Docker output staging cleanup failed after retry"
    )
    if primary_error is not None:
        raise cleanup_error from primary_error
    if last_error is not None:
        raise cleanup_error from last_error
    raise cleanup_error


def _copy_docker_output_after_stop(
    *,
    docker: str,
    image: str,
    volume: str,
    output_directory: Path,
) -> Path:
    parent = _real_directory(output_directory.parent, "output parent")
    staging = Path(
        tempfile.mkdtemp(prefix=".policytwin-docker-output-", dir=parent)
    ).resolve(strict=True)
    exporter_id: str | None = None
    try:
        exporter = (
            docker,
            "create",
            "--pull=never",
            "--network=none",
            "--cap-drop=ALL",
            "--privileged=false",
            "--ipc=private",
            "--security-opt=no-new-privileges:true",
            "--pids-limit=16",
            "--no-healthcheck",
            "--user=0:0",
            "--entrypoint=/usr/local/bin/python",
            f"--mount=type=volume,src={volume},dst=/export,readonly",
            "--tmpfs=/tmp:rw,noexec,nosuid,nodev,size=8m,mode=1777",
            image,
            "-I",
            "-S",
            "-c",
            _DOCKER_EXPORT_SOURCE,
        )
        if any("type=bind" in argument for argument in exporter):
            _fail(IsolationExecutionError, "Docker output exporter contains a bind")
        exporter_id = _docker_container_id(_docker_subprocess(exporter))
        started = _docker_subprocess(
            (docker, "start", "--attach", exporter_id), capture_output=False
        )
        inspected = _docker_inspect(docker, exporter_id)
        state = _inspect_mapping(inspected.get("State"), "exporter State")
        if started.returncode != 0 or state.get("ExitCode") != 0:
            _fail(IsolationExecutionError, "Docker output exporter failed")
        copied = _docker_subprocess(
            (
                docker,
                "cp",
                f"{exporter_id}:/policytwin-export/.",
                str(staging),
            )
        )
        if copied.returncode != 0:
            _fail(IsolationExecutionError, "Docker output copy failed")
        _validate_copied_output_tree(staging)
        return staging
    except Exception:
        _checked_remove_private_output_staging(
            staging,
            output_parent=parent,
            primary_error=sys.exc_info()[1],
        )
        raise
    finally:
        if exporter_id is not None:
            try:
                _checked_docker_cleanup(
                    (docker, "rm", "--force", exporter_id),
                    context="Docker private output exporter cleanup",
                    primary_error=sys.exc_info()[1],
                )
            except IsolationError as cleanup_error:
                _checked_remove_private_output_staging(
                    staging,
                    output_parent=parent,
                    primary_error=cleanup_error,
                )
                raise


def _publish_validated_output(staging: Path, output_directory: Path) -> None:
    output = _real_directory(output_directory, "output directory")
    if any(output.iterdir()):
        _fail(IsolationExecutionError, "output directory changed during Docker job")
    if staging.parent != output.parent:
        _fail(IsolationExecutionError, "Docker output staging is not co-located")
    try:
        os.replace(staging, output)
    except OSError as error:
        raise IsolationExecutionError(
            "validated Docker output could not be atomically published"
        ) from error


def _launch_private_docker_job(
    *,
    launcher: Sequence[str],
    target: tuple[str, ...],
    source_snapshot: DockerSourceSnapshot,
    project_root: Path,
    output_directory: Path,
    receipt_name: str,
    docker_image: str,
    docker_local_image_id: str,
    docker_cli_sha256: str,
    docker_cli_size_bytes: int,
) -> IsolationReceipt:
    docker = launcher[0]
    expected_cli_identity = DockerCliIdentity(
        path=str(_DOCKER_CLI_PATH),
        size_bytes=docker_cli_size_bytes,
        sha256=docker_cli_sha256,
    )
    _require_same_docker_cli_identity(
        expected_cli_identity, context="before the private Docker lifecycle"
    )
    try:
        image_index = tuple(launcher).index(docker_image)
    except ValueError as error:
        raise IsolationExecutionError(
            "Docker launcher does not contain its immutable image"
        ) from error
    expected_container_command = tuple(launcher[image_index + 1 :])
    container_id: str | None = None
    output_volume: str | None = None
    staging: Path | None = None
    final_cleanup_attempted = False
    final_identity_verified = False
    try:
        container_id = _docker_container_id(_docker_subprocess(launcher))
        initial_inspect = _docker_inspect(docker, container_id)
        controls, output_volume, _, initial_status = _inspect_final_container(
            initial_inspect,
            expected_image=docker_image,
            expected_image_id=docker_local_image_id,
            expected_docker_cli_sha256=docker_cli_sha256,
            expected_docker_cli_size_bytes=docker_cli_size_bytes,
            expected_user=_DOCKER_FINAL_USER,
            expected_command=expected_container_command,
        )
        if initial_status != "created":
            _fail(
                IsolationExecutionError,
                "Docker final container started before source staging completed",
            )
        if _runtime_controls_sha256(controls) != _runtime_controls_sha256(
            _expected_docker_runtime_controls(
                docker_image=docker_image,
                actual_image_id=docker_local_image_id,
                docker_cli_sha256=docker_cli_sha256,
                docker_cli_size_bytes=docker_cli_size_bytes,
                user=_DOCKER_FINAL_USER,
            )
        ):
            _fail(IsolationExecutionError, "Docker runtime controls hash differs")
        _initialize_docker_output_volume(
            docker=docker,
            image=docker_image,
            volume=output_volume,
            uid=_DOCKER_FINAL_UID,
            gid=_DOCKER_FINAL_GID,
        )
        source_snapshot = _verify_prepared_docker_snapshot(source_snapshot)
        source_copy = _docker_subprocess(
            (
                docker,
                "cp",
                f"{source_snapshot.root}/.",
                f"{container_id}:{_DOCKER_SOURCE_ROOT}",
            )
        )
        manifest_copy = _docker_subprocess(
            (
                docker,
                "cp",
                str(source_snapshot.manifest_path),
                f"{container_id}:{_DOCKER_SOURCE_MANIFEST_PATH}",
            )
        )
        if source_copy.returncode != 0 or manifest_copy.returncode != 0:
            _fail(
                IsolationExecutionError,
                "Docker private source staging failed before container start",
            )
        # The staged bytes are now in Docker's private container layer.  A
        # second host-side verification closes the mutable-staging window;
        # the fixed bootstrap independently rehashes the private copy at start.
        _verify_prepared_docker_snapshot(source_snapshot)
        _require_clean_matching_head(project_root, source_snapshot)
        staged_inspect = _docker_inspect(docker, container_id)
        staged_controls, staged_volume, _, staged_status = _inspect_final_container(
            staged_inspect,
            expected_image=docker_image,
            expected_image_id=docker_local_image_id,
            expected_docker_cli_sha256=docker_cli_sha256,
            expected_docker_cli_size_bytes=docker_cli_size_bytes,
            expected_user=_DOCKER_FINAL_USER,
            expected_command=expected_container_command,
        )
        if (
            staged_status != "created"
            or staged_volume != output_volume
            or staged_controls != controls
        ):
            _fail(
                IsolationExecutionError,
                "Docker controls changed during private source staging",
            )
        _docker_subprocess(
            (docker, "start", "--attach", container_id), capture_output=False
        )
        final_inspect = _docker_inspect(docker, container_id)
        final_controls, final_volume, container_exit, final_status = (
            _inspect_final_container(
                final_inspect,
                expected_image=docker_image,
                expected_image_id=docker_local_image_id,
                expected_docker_cli_sha256=docker_cli_sha256,
                expected_docker_cli_size_bytes=docker_cli_size_bytes,
                expected_user=_DOCKER_FINAL_USER,
                expected_command=expected_container_command,
            )
        )
        if (
            final_status != "exited"
            or final_volume != output_volume
            or final_controls != controls
            or container_exit is None
        ):
            _fail(
                IsolationExecutionError,
                "Docker final container did not stop under its inspected controls",
            )
        staging = _copy_docker_output_after_stop(
            docker=docker,
            image=docker_image,
            volume=output_volume,
            output_directory=output_directory,
        )
        receipt_path = staging / receipt_name
        write_docker_runtime_attestation(receipt_path, final_controls)
        receipt = audit_isolation_receipt(receipt_path, require_success=False)
        expected_source_identity = source_snapshot.identity
        expected_runtime_controls_sha256 = _runtime_controls_sha256(final_controls)
        expected_config = _isolation_config_sha256(
            IsolationBackend.DOCKER_NONE,
            docker_image=docker_image,
            source_identity=expected_source_identity,
            runtime_controls_sha256=expected_runtime_controls_sha256,
        )
        expected_command_sha256 = sha256_bytes(canonical_json_bytes(list(target)))
        if (
            receipt.backend is not IsolationBackend.DOCKER_NONE
            or receipt.runtime_identity != docker_image
            or receipt.source_identity != expected_source_identity
            or receipt.runtime_controls_sha256 != expected_runtime_controls_sha256
            or receipt.isolation_config_sha256 != expected_config
            or receipt.command_sha256 != expected_command_sha256
        ):
            _fail(
                IsolationExecutionError,
                "Docker receipt does not bind the staged source, controls, and target",
            )
        expected_exit = (
            125
            if receipt.status
            in {IsolationJobStatus.INPUT_FAILED, IsolationJobStatus.PROBE_FAILED}
            else (receipt.target_exit_code or 0)
        )
        if container_exit != expected_exit:
            _fail(
                IsolationExecutionError,
                "Docker exit code differs from the sealed isolation receipt",
            )
        # The private container and its anonymous output volume are part of the
        # trusted lifecycle, not published result artifacts.  Their checked
        # cleanup therefore has to succeed before validated bytes become
        # visible in the caller-owned output directory.
        final_cleanup_attempted = True
        _checked_docker_cleanup(
            (docker, "rm", "--force", "--volumes", container_id),
            context="Docker final container and anonymous-volume cleanup",
        )
        container_id = None
        output_volume = None
        _require_same_docker_cli_identity(
            expected_cli_identity,
            context="after the private Docker lifecycle cleanup",
        )
        final_identity_verified = True
        _publish_validated_output(staging, output_directory)
        staging = None
        return audit_isolation_receipt(
            Path(os.path.abspath(output_directory.expanduser())) / receipt_name,
            require_success=False,
        )
    finally:
        primary_error = sys.exc_info()[1]
        staging_cleanup_error: IsolationError | None = None
        if staging is not None:
            try:
                _checked_remove_private_output_staging(
                    staging,
                    output_parent=output_directory.parent,
                    primary_error=primary_error,
                )
            except IsolationError as error:
                staging_cleanup_error = error
        docker_cleanup_error: IsolationError | None = None
        if not final_cleanup_attempted:
            try:
                if container_id is not None:
                    _checked_docker_cleanup(
                        (docker, "rm", "--force", "--volumes", container_id),
                        context=("Docker final container and anonymous-volume cleanup"),
                        primary_error=staging_cleanup_error or primary_error,
                    )
                elif output_volume is not None:
                    _checked_docker_cleanup(
                        (docker, "volume", "rm", output_volume),
                        context="Docker orphan output-volume cleanup",
                        primary_error=staging_cleanup_error or primary_error,
                    )
            except IsolationError as error:
                docker_cleanup_error = error
        if not final_identity_verified:
            _require_same_docker_cli_identity(
                expected_cli_identity,
                context="after the private Docker lifecycle cleanup",
                primary_error=docker_cleanup_error
                or staging_cleanup_error
                or primary_error,
            )
        if docker_cleanup_error is not None:
            raise docker_cleanup_error
        if staging_cleanup_error is not None:
            raise staging_cleanup_error


def _launch_prepared_frozen_evaluation_job(
    *,
    backend: IsolationBackend,
    command: Sequence[str],
    project_root: Path,
    output_directory: Path,
    receipt_name: str = DEFAULT_RECEIPT_NAME,
    docker_image: str | None = None,
    docker_source_snapshot: DockerSourceSnapshot | None = None,
    macos_sandbox_identity: MacosSandboxIdentity | None = None,
    macos_unix_canary_path: Path | None = None,
) -> IsolationReceipt:
    """Launch, then independently audit, one complete OS-isolated job."""
    target = _validated_command(command)
    local_image_id: str | None = None
    docker_cli_identity: DockerCliIdentity | None = None
    if backend is IsolationBackend.DOCKER_NONE:
        docker_cli_identity = _require_local_docker_daemon()
        docker = str(_DOCKER_CLI_PATH)
        local_image_id = _docker_local_image_id(docker, _immutable_image(docker_image))
    elif macos_sandbox_identity is None:
        macos_sandbox_identity = _require_local_macos_sandbox()
    launcher = build_isolation_command(
        backend=backend,
        command=target,
        project_root=project_root,
        output_directory=output_directory,
        receipt_name=receipt_name,
        docker_image=docker_image,
        docker_source_snapshot=docker_source_snapshot,
        docker_local_image_id=local_image_id,
        docker_cli_sha256=(
            None if docker_cli_identity is None else docker_cli_identity.sha256
        ),
        docker_cli_size_bytes=(
            None if docker_cli_identity is None else docker_cli_identity.size_bytes
        ),
        macos_sandbox_exec_sha256=(
            None if macos_sandbox_identity is None else macos_sandbox_identity.sha256
        ),
        macos_sandbox_exec_size_bytes=(
            None
            if macos_sandbox_identity is None
            else macos_sandbox_identity.size_bytes
        ),
        macos_environment_launcher_sha256=(
            None
            if macos_sandbox_identity is None
            else macos_sandbox_identity.environment_launcher_sha256
        ),
        macos_environment_launcher_size_bytes=(
            None
            if macos_sandbox_identity is None
            else macos_sandbox_identity.environment_launcher_size_bytes
        ),
        macos_unix_canary_path=macos_unix_canary_path,
    )
    if backend is IsolationBackend.DOCKER_NONE:
        if docker_source_snapshot is None:
            _fail(
                IsolationUnavailableError,
                "Docker launch requires its prepared private source snapshot",
            )
        return _launch_private_docker_job(
            launcher=launcher,
            target=target,
            source_snapshot=docker_source_snapshot,
            project_root=project_root,
            output_directory=output_directory,
            receipt_name=receipt_name,
            docker_image=_immutable_image(docker_image),
            docker_local_image_id=_validated_image_id(local_image_id),
            docker_cli_sha256=_sha256(
                None if docker_cli_identity is None else docker_cli_identity.sha256,
                "docker_cli_sha256",
            ),
            docker_cli_size_bytes=(
                0 if docker_cli_identity is None else docker_cli_identity.size_bytes
            ),
        )
    if macos_sandbox_identity is None:
        _fail(IsolationUnavailableError, "macOS sandbox identity is unavailable")
    try:
        completed = subprocess.run(
            launcher,
            check=False,
            close_fds=True,
            env=_macos_launcher_environment(),
        )
    except OSError as error:
        raise IsolationUnavailableError(
            "selected OS isolation launcher could not start; no fallback was used"
        ) from error
    finally:
        _require_same_macos_sandbox_identity(
            macos_sandbox_identity,
            context="after the macOS sandbox lifecycle",
            primary_error=sys.exc_info()[1],
        )
    receipt_path = Path(os.path.abspath(output_directory.expanduser())) / receipt_name
    macos_controls = _expected_macos_runtime_controls(
        sandbox_exec_sha256=macos_sandbox_identity.sha256,
        sandbox_exec_size_bytes=macos_sandbox_identity.size_bytes,
        environment_launcher_sha256=(
            macos_sandbox_identity.environment_launcher_sha256
        ),
        environment_launcher_size_bytes=(
            macos_sandbox_identity.environment_launcher_size_bytes
        ),
    )
    write_macos_runtime_attestation(receipt_path, macos_controls)
    try:
        receipt = audit_isolation_receipt(receipt_path, require_success=False)
    except IsolationIntegrityError as error:
        raise IsolationExecutionError(
            "OS isolation launcher did not produce a valid receipt"
        ) from error
    expected_source_identity = (
        _live_checkout_source_identity()
        if docker_source_snapshot is None
        else docker_source_snapshot.identity
    )
    expected_config = _isolation_config_sha256(
        backend,
        docker_image=docker_image,
        source_identity=expected_source_identity,
        runtime_controls_sha256=receipt.runtime_controls_sha256,
    )
    expected_runtime_identity = _runtime_identity(backend, docker_image=docker_image)
    expected_command = sha256_bytes(canonical_json_bytes(list(target)))
    if (
        receipt.backend is not backend
        or receipt.runtime_identity != expected_runtime_identity
        or receipt.source_identity != expected_source_identity
        or receipt.runtime_controls_sha256 != _runtime_controls_sha256(macos_controls)
        or receipt.isolation_config_sha256 != expected_config
        or receipt.command_sha256 != expected_command
    ):
        _fail(
            IsolationExecutionError,
            "isolation receipt does not bind the requested backend and command",
        )
    expected_exit = (
        125
        if receipt.status
        in {IsolationJobStatus.INPUT_FAILED, IsolationJobStatus.PROBE_FAILED}
        else (receipt.target_exit_code or 0)
    )
    if completed.returncode != expected_exit:
        _fail(
            IsolationExecutionError,
            "launcher exit code does not match the sealed isolation receipt",
        )
    return receipt


def launch_frozen_evaluation_job(
    *,
    backend: IsolationBackend,
    command: Sequence[str],
    project_root: Path,
    output_directory: Path,
    receipt_name: str = DEFAULT_RECEIPT_NAME,
    docker_image: str | None = None,
) -> IsolationReceipt:
    """Launch one complete job with the backend's mandatory source boundary."""
    if backend is IsolationBackend.DOCKER_NONE:
        _require_local_docker_daemon()
        with prepare_docker_source_snapshot(project_root) as source_snapshot:
            return _launch_prepared_frozen_evaluation_job(
                backend=backend,
                command=command,
                project_root=project_root,
                output_directory=output_directory,
                receipt_name=receipt_name,
                docker_image=docker_image,
                docker_source_snapshot=source_snapshot,
            )
    macos_sandbox_identity = _require_local_macos_sandbox()
    with _macos_unix_canary() as canary_path:
        return _launch_prepared_frozen_evaluation_job(
            backend=backend,
            command=command,
            project_root=project_root,
            output_directory=output_directory,
            receipt_name=receipt_name,
            docker_image=docker_image,
            macos_sandbox_identity=macos_sandbox_identity,
            macos_unix_canary_path=canary_path,
        )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="policytwin-frozen-eval")
    subparsers = parser.add_subparsers(dest="mode", required=True)
    launch = subparsers.add_parser("launch")
    launch.add_argument("--backend", choices=tuple(IsolationBackend), required=True)
    launch.add_argument("--project-root", type=Path, required=True)
    launch.add_argument("--output-directory", type=Path, required=True)
    launch.add_argument("--receipt-name", default=DEFAULT_RECEIPT_NAME)
    launch.add_argument("--docker-image")
    launch.add_argument("command", nargs=argparse.REMAINDER)
    worker = subparsers.add_parser("_worker", help=argparse.SUPPRESS)
    worker.add_argument("--backend", choices=tuple(IsolationBackend), required=True)
    worker.add_argument(
        "--input-mode", choices=tuple(IsolationInputMode), required=True
    )
    worker.add_argument("--source-commit")
    worker.add_argument("--source-tree-sha256")
    worker.add_argument("--source-manifest-sha256")
    worker.add_argument("--source-file-count", type=int)
    worker.add_argument("--source-archive-sha256")
    worker.add_argument("--source-archive-size-bytes", type=int)
    worker.add_argument("--source-bootstrap-sha256")
    worker.add_argument("--source-git-cli-path")
    worker.add_argument("--source-git-cli-platform-system")
    worker.add_argument("--source-git-cli-platform-machine")
    worker.add_argument("--source-git-cli-sha256")
    worker.add_argument("--source-git-cli-size-bytes", type=int)
    worker.add_argument("--source-root", type=Path)
    worker.add_argument("--source-manifest", type=Path)
    worker.add_argument("--isolation-config-sha256", required=True)
    worker.add_argument("--runtime-controls-sha256", required=True)
    worker.add_argument("--receipt", type=Path, required=True)
    worker.add_argument("--docker-image")
    worker.add_argument("command", nargs=argparse.REMAINDER)
    audit = subparsers.add_parser("audit")
    audit.add_argument("receipt", type=Path)
    return parser


def _without_separator(command: Sequence[str]) -> tuple[str, ...]:
    normalized = tuple(command)
    if normalized and normalized[0] == "--":
        normalized = normalized[1:]
    return _validated_command(normalized)


def _source_identity_from_arguments(
    arguments: argparse.Namespace,
) -> IsolationSourceIdentity:
    raw_git_identity = (
        arguments.source_git_cli_path,
        arguments.source_git_cli_platform_system,
        arguments.source_git_cli_platform_machine,
        arguments.source_git_cli_sha256,
        arguments.source_git_cli_size_bytes,
    )
    if all(value is None for value in raw_git_identity):
        git_cli_identity = None
    elif any(value is None for value in raw_git_identity):
        _fail(IsolationIntegrityError, "source Git CLI identity is incomplete")
    else:
        try:
            git_cli_identity = GitCliIdentity(
                path=arguments.source_git_cli_path,
                platform_system=arguments.source_git_cli_platform_system,
                platform_machine=arguments.source_git_cli_platform_machine,
                sha256=arguments.source_git_cli_sha256,
                size_bytes=arguments.source_git_cli_size_bytes,
            )
        except GitBoundaryError as error:
            raise IsolationIntegrityError(
                "source Git CLI identity is invalid"
            ) from error
    return IsolationSourceIdentity(
        mode=IsolationInputMode(arguments.input_mode),
        commit=arguments.source_commit,
        tree_sha256=arguments.source_tree_sha256,
        manifest_sha256=arguments.source_manifest_sha256,
        file_count=arguments.source_file_count,
        archive_sha256=arguments.source_archive_sha256,
        archive_size_bytes=arguments.source_archive_size_bytes,
        bootstrap_sha256=arguments.source_bootstrap_sha256,
        git_cli_identity=git_cli_identity,
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Run the isolation launcher, worker, or receipt auditor CLI."""
    arguments = _parser().parse_args(argv)
    try:
        if arguments.mode == "launch":
            receipt = launch_frozen_evaluation_job(
                backend=IsolationBackend(arguments.backend),
                command=_without_separator(arguments.command),
                project_root=arguments.project_root,
                output_directory=arguments.output_directory,
                receipt_name=arguments.receipt_name,
                docker_image=arguments.docker_image,
            )
            print(canonical_json_bytes(receipt.to_mapping()).decode("utf-8"))
            if receipt.status in {
                IsolationJobStatus.INPUT_FAILED,
                IsolationJobStatus.PROBE_FAILED,
            }:
                return 125
            return receipt.target_exit_code or 0
        if arguments.mode == "_worker":
            receipt = run_isolation_worker(
                backend=IsolationBackend(arguments.backend),
                isolation_config_sha256=arguments.isolation_config_sha256,
                runtime_controls_sha256=arguments.runtime_controls_sha256,
                receipt_path=arguments.receipt,
                command=_without_separator(arguments.command),
                docker_image=arguments.docker_image,
                source_identity=_source_identity_from_arguments(arguments),
                source_root=arguments.source_root,
                source_manifest_path=arguments.source_manifest,
            )
            if receipt.status in {
                IsolationJobStatus.INPUT_FAILED,
                IsolationJobStatus.PROBE_FAILED,
            }:
                return 125
            return receipt.target_exit_code or 0
        receipt = audit_isolation_receipt(arguments.receipt)
        print(canonical_json_bytes(receipt.to_mapping()).decode("utf-8"))
        return 0
    except IsolationError as error:
        payload = {
            "error": str(error),
            "error_type": type(error).__name__,
            "schema_version": 1,
            "status": "error",
        }
        print(canonical_json_bytes(payload).decode("utf-8"), file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover - console script is the main entry.
    raise SystemExit(main())


__all__ = [
    "DEFAULT_RECEIPT_NAME",
    "DOCKER_BOOTSTRAP_SHA256",
    "DockerCliIdentity",
    "DockerSourceSnapshot",
    "ISOLATION_RECEIPT_SCHEMA_VERSION",
    "MACOS_SANDBOX_RUNTIME_IDENTITY",
    "IsolationBackend",
    "IsolationError",
    "IsolationExecutionError",
    "IsolationIntegrityError",
    "IsolationInputMode",
    "IsolationJobStatus",
    "IsolationOutputTreeIdentity",
    "IsolationProbeEvidence",
    "IsolationReceipt",
    "IsolationSourceIdentity",
    "IsolationUnavailableError",
    "MacosSandboxIdentity",
    "audit_isolation_receipt",
    "build_isolation_command",
    "launch_frozen_evaluation_job",
    "main",
    "prepare_docker_source_snapshot",
    "run_isolation_worker",
    "write_isolation_receipt",
    "write_docker_runtime_attestation",
    "write_macos_runtime_attestation",
]
