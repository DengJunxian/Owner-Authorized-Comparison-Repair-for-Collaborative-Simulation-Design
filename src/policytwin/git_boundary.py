"""Hardened local Git command boundary for reproducibility metadata.

This module deliberately supports one fixed system Git executable and ordinary
standalone repositories.  It does not treat local root ownership or SHA-256
commitments as an external host attestation.
"""

from __future__ import annotations

import hashlib
import os
import platform
import re
import shutil
import stat
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, cast

_DARWIN_GIT_SHIM_PATH = Path("/usr/bin/git")
_GIT_CLI_PATHS = {
    "Darwin": Path("/Library/Developer/CommandLineTools/usr/bin/git"),
    "Linux": Path("/usr/bin/git"),
}
_SUPPORTED_MACHINES = {
    "Darwin": frozenset({"arm64", "x86_64"}),
    "Linux": frozenset({"aarch64", "x86_64"}),
}

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_MAX_GIT_CLI_BYTES = 256 * 1024 * 1024
_MAX_GIT_CONTROL_FILE_BYTES = 8 * 1024 * 1024
_HIGH_RISK_GIT_ENVIRONMENT = frozenset(
    {
        "GIT_ALTERNATE_OBJECT_DIRECTORIES",
        "GIT_CEILING_DIRECTORIES",
        "GIT_COMMON_DIR",
        "GIT_CONFIG",
        "GIT_CONFIG_COUNT",
        "GIT_CONFIG_GLOBAL",
        "GIT_CONFIG_NOSYSTEM",
        "GIT_CONFIG_SYSTEM",
        "GIT_DIR",
        "GIT_DISCOVERY_ACROSS_FILESYSTEM",
        "GIT_EXEC_PATH",
        "GIT_INDEX_FILE",
        "GIT_NAMESPACE",
        "GIT_OBJECT_DIRECTORY",
        "GIT_REPLACE_REF_BASE",
        "GIT_SHALLOW_FILE",
        "GIT_WORK_TREE",
    }
)


class GitBoundaryError(RuntimeError):
    """Raised when the fixed Git trust boundary cannot be established."""


@dataclass(frozen=True, slots=True)
class GitCliIdentity:
    """Byte identity of the fixed local Git executable."""

    path: str
    platform_system: str
    platform_machine: str
    sha256: str
    size_bytes: int

    def __post_init__(self) -> None:
        expected_path = _GIT_CLI_PATHS.get(self.platform_system)
        allowed_machines = _SUPPORTED_MACHINES.get(self.platform_system)
        if (
            expected_path is None
            or allowed_machines is None
            or self.platform_machine not in allowed_machines
            or self.path != str(expected_path)
        ):
            raise GitBoundaryError("Git CLI identity path is not fixed")
        if not isinstance(self.sha256, str) or _SHA256.fullmatch(self.sha256) is None:
            raise GitBoundaryError("Git CLI identity SHA-256 is invalid")
        if (
            isinstance(self.size_bytes, bool)
            or not isinstance(self.size_bytes, int)
            or self.size_bytes <= 0
            or self.size_bytes > _MAX_GIT_CLI_BYTES
        ):
            raise GitBoundaryError("Git CLI identity size is invalid")

    def to_mapping(self) -> dict[str, object]:
        """Return the closed machine-readable identity."""
        return {
            "path": self.path,
            "platform_machine": self.platform_machine,
            "platform_system": self.platform_system,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
        }

    @classmethod
    def from_mapping(cls, value: object) -> GitCliIdentity:
        """Strictly parse a persisted Git executable identity."""
        if not isinstance(value, dict) or set(value) != {
            "path",
            "platform_machine",
            "platform_system",
            "sha256",
            "size_bytes",
        }:
            raise GitBoundaryError("Git CLI identity has an invalid schema")
        path = value["path"]
        platform_system = value["platform_system"]
        platform_machine = value["platform_machine"]
        digest = value["sha256"]
        size = value["size_bytes"]
        if (
            not isinstance(path, str)
            or not isinstance(platform_system, str)
            or not isinstance(platform_machine, str)
            or not isinstance(digest, str)
            or isinstance(size, bool)
            or not isinstance(size, int)
        ):
            raise GitBoundaryError("Git CLI identity fields have invalid types")
        return cls(
            path=path,
            platform_system=platform_system,
            platform_machine=platform_machine,
            sha256=digest,
            size_bytes=cast(int, size),
        )


@dataclass(frozen=True, slots=True)
class GitCommandResult:
    """Bounded result of one command executed by the fixed Git CLI."""

    returncode: int
    stdout: bytes
    stderr: bytes


def git_cli_environment(repository_root: Path) -> dict[str, str]:
    """Return the complete non-inherited environment for local Git commands."""
    root = repository_root.resolve(strict=True)
    return {
        "GIT_ATTR_NOSYSTEM": "1",
        "GIT_CONFIG_GLOBAL": "/dev/null",
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_COMMON_DIR": str(root / ".git"),
        "GIT_INDEX_FILE": str(root / ".git" / "index"),
        "GIT_NO_REPLACE_OBJECTS": "1",
        "GIT_NO_LAZY_FETCH": "1",
        "GIT_OBJECT_DIRECTORY": str(root / ".git" / "objects"),
        "GIT_OPTIONAL_LOCKS": "0",
        "GIT_PAGER": "cat",
        "GIT_TERMINAL_PROMPT": "0",
        "GIT_DIR": str(root / ".git"),
        "GIT_WORK_TREE": str(root),
        "HOME": "/var/empty",
        "LANG": "C",
        "LC_ALL": "C",
        "PATH": "/usr/bin:/bin",
        "TZ": "UTC",
        "XDG_CONFIG_HOME": "/var/empty",
    }


def reject_ambient_git_overrides() -> None:
    """Reject caller variables able to redirect repository or object identity."""
    dangerous = sorted(
        key
        for key in os.environ
        if key in _HIGH_RISK_GIT_ENVIRONMENT
        or key.startswith("GIT_CONFIG_KEY_")
        or key.startswith("GIT_CONFIG_VALUE_")
    )
    if dangerous:
        raise GitBoundaryError(
            f"ambient Git trust-root overrides are forbidden: {dangerous}"
        )


def _validate_git_cli_info(info: os.stat_result) -> None:
    if (
        stat.S_ISLNK(info.st_mode)
        or not stat.S_ISREG(info.st_mode)
        or info.st_uid != 0
        or not bool(info.st_mode & 0o111)
        or bool(info.st_mode & 0o022)
        or bool(info.st_mode & 0o7000)
        or info.st_size <= 0
        or info.st_size > _MAX_GIT_CLI_BYTES
    ):
        raise GitBoundaryError(
            "fixed Git CLI must be a bounded root-owned executable regular "
            "file without group/world write or special mode bits"
        )


def git_cli_path() -> Path:
    """Return the fixed real Git executable for the supported host platform."""
    system = platform.system()
    machine = platform.machine()
    path = _GIT_CLI_PATHS.get(system)
    machines = _SUPPORTED_MACHINES.get(system)
    if path is None or machines is None or machine not in machines:
        raise GitBoundaryError(
            f"unsupported Git host platform: system={system}, machine={machine}"
        )
    return path


def verify_git_cli_identity() -> GitCliIdentity:
    """Open, hash, and stability-check the fixed real Git executable."""
    reject_ambient_git_overrides()
    system = platform.system()
    machine = platform.machine()
    actual_path = git_cli_path()
    resolved = shutil.which("git")
    allowed_path_entries = {str(actual_path)}
    if system == "Darwin":
        allowed_path_entries.add(str(_DARWIN_GIT_SHIM_PATH))
    if resolved not in allowed_path_entries:
        raise GitBoundaryError(
            "caller PATH does not resolve to the fixed Git CLI or recognized "
            "root-owned Darwin shim"
        )
    if resolved == str(_DARWIN_GIT_SHIM_PATH):
        try:
            _validate_git_cli_info(os.lstat(_DARWIN_GIT_SHIM_PATH))
        except OSError as error:
            raise GitBoundaryError("Darwin Git shim is unavailable") from error
    try:
        before = os.lstat(actual_path)
        _validate_git_cli_info(before)
        descriptor = os.open(
            actual_path,
            os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
        )
    except (OSError, GitBoundaryError) as error:
        if isinstance(error, GitBoundaryError):
            raise
        raise GitBoundaryError("fixed real Git CLI is unavailable") from error
    digest = hashlib.sha256()
    try:
        opened = os.fstat(descriptor)
        _validate_git_cli_info(opened)
        if (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino):
            raise GitBoundaryError("Git CLI changed while being opened")
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
        after_open = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    try:
        after_path = os.lstat(actual_path)
    except OSError as error:
        raise GitBoundaryError("Git CLI disappeared during hashing") from error
    stable_fields = (
        "st_dev",
        "st_ino",
        "st_mode",
        "st_uid",
        "st_gid",
        "st_size",
        "st_mtime_ns",
        "st_ctime_ns",
    )
    if any(
        getattr(before, field) != getattr(after_open, field)
        or getattr(before, field) != getattr(after_path, field)
        for field in stable_fields
    ):
        raise GitBoundaryError("Git CLI changed during hashing")
    return GitCliIdentity(
        path=str(actual_path),
        platform_system=system,
        platform_machine=machine,
        sha256=digest.hexdigest(),
        size_bytes=before.st_size,
    )


def require_same_git_cli_identity(expected: GitCliIdentity, *, context: str) -> None:
    """Fail if the fixed Git executable differs from an earlier identity."""
    if not isinstance(expected, GitCliIdentity):
        raise GitBoundaryError("expected Git CLI identity is invalid")
    observed = verify_git_cli_identity()
    if observed != expected:
        raise GitBoundaryError(f"Git CLI identity changed {context}")


def _read_bounded_control_file(path: Path) -> bytes:
    try:
        descriptor = os.open(
            path,
            os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
        )
    except OSError as error:
        raise GitBoundaryError(
            f"cannot safely read Git control file: {path}"
        ) from error
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_size < 0
            or before.st_size > _MAX_GIT_CONTROL_FILE_BYTES
        ):
            raise GitBoundaryError(f"Git control file is unsafe: {path}")
        chunks: list[bytes] = []
        remaining = before.st_size + 1
        while remaining > 0:
            chunk = os.read(descriptor, min(1024 * 1024, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    ) or sum(len(chunk) for chunk in chunks) != before.st_size:
        raise GitBoundaryError(f"Git control file changed while reading: {path}")
    return b"".join(chunks)


def discover_repository_root(start: Path) -> Path | None:
    """Discover a standalone repository without invoking or trusting Git."""
    try:
        absolute = start.expanduser().resolve(strict=True)
    except OSError as error:
        raise GitBoundaryError("Git metadata start path is unavailable") from error
    candidate = absolute if absolute.is_dir() else absolute.parent
    for root in (candidate, *candidate.parents):
        git_directory = root / ".git"
        try:
            info = os.lstat(git_directory)
        except FileNotFoundError:
            continue
        except OSError as error:
            raise GitBoundaryError("cannot inspect repository trust root") from error
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
            raise GitBoundaryError(
                "only a standalone repository with a real .git directory is supported"
            )
        assert_trusted_repository_layout(root)
        return root.resolve(strict=True)
    return None


def assert_trusted_repository_layout(repository_root: Path) -> None:
    """Reject replace refs, grafts, alternates, and linked trust-root storage."""
    root = repository_root.resolve(strict=True)
    git_directory = root / ".git"
    try:
        git_info = os.lstat(git_directory)
    except OSError as error:
        raise GitBoundaryError(
            "repository has no inspectable .git directory"
        ) from error
    if stat.S_ISLNK(git_info.st_mode) or not stat.S_ISDIR(git_info.st_mode):
        raise GitBoundaryError("repository .git path must be a real directory")

    forbidden = (
        git_directory / "commondir",
        git_directory / "info" / "grafts",
        git_directory / "objects" / "info" / "alternates",
        git_directory / "objects" / "info" / "http-alternates",
        git_directory / "refs" / "replace",
        git_directory / "reftable",
        git_directory / "info" / "attributes",
    )
    if any(path.exists() or path.is_symlink() for path in forbidden):
        raise GitBoundaryError(
            "repository uses forbidden replace refs, grafts, alternates, "
            "or shared storage"
        )
    for required_directory in (git_directory / "objects", git_directory / "refs"):
        try:
            info = os.lstat(required_directory)
        except FileNotFoundError:
            continue
        except OSError as error:
            raise GitBoundaryError("cannot inspect Git object/ref storage") from error
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
            raise GitBoundaryError("Git object/ref storage must be a real directory")
        entry_count = 0
        for current, directory_names, file_names in os.walk(
            required_directory, followlinks=False
        ):
            current_path = Path(current)
            for name in (*directory_names, *file_names):
                entry_count += 1
                if entry_count > 1_000_000:
                    raise GitBoundaryError("Git object/ref storage is unbounded")
                entry = current_path / name
                try:
                    entry_info = os.lstat(entry)
                except OSError as error:
                    raise GitBoundaryError(
                        "cannot inspect Git object/ref storage entry"
                    ) from error
                if stat.S_ISLNK(entry_info.st_mode) or not (
                    stat.S_ISDIR(entry_info.st_mode) or stat.S_ISREG(entry_info.st_mode)
                ):
                    raise GitBoundaryError(
                        "Git object/ref storage contains a linked or special entry"
                    )

    for control_name in ("HEAD", "config", "index"):
        control_path = git_directory / control_name
        if control_path.exists() or control_path.is_symlink():
            _read_bounded_control_file(control_path)

    exclude_path = git_directory / "info" / "exclude"
    if exclude_path.exists() or exclude_path.is_symlink():
        exclude_bytes = _read_bounded_control_file(exclude_path)
        try:
            exclude_lines = exclude_bytes.decode("utf-8", errors="strict").splitlines()
        except UnicodeError as error:
            raise GitBoundaryError("Git info/exclude is not canonical UTF-8") from error
        if any(
            line.strip() and not line.lstrip().startswith("#") for line in exclude_lines
        ):
            raise GitBoundaryError("Git info/exclude contains unversioned ignore rules")

    packed_refs = git_directory / "packed-refs"
    if packed_refs.exists() or packed_refs.is_symlink():
        content = _read_bounded_control_file(packed_refs)
        try:
            lines = content.decode("ascii", errors="strict").splitlines()
        except UnicodeError as error:
            raise GitBoundaryError("packed-refs is not canonical ASCII") from error
        if any(
            line.split(" ", maxsplit=1)[-1].startswith("refs/replace/")
            for line in lines
            if line and not line.startswith(("#", "^"))
        ):
            raise GitBoundaryError("repository contains packed replace refs")


def run_git_command(
    repository_root: Path,
    arguments: Sequence[str],
    *,
    expected_cli_identity: GitCliIdentity,
    stdout: BinaryIO | None = None,
) -> GitCommandResult:
    """Run one absolute-path Git command under the closed local boundary."""
    if (
        isinstance(arguments, str)
        or not arguments
        or any(
            not isinstance(argument, str) or not argument or "\x00" in argument
            for argument in arguments
        )
    ):
        raise GitBoundaryError("Git command arguments are invalid")
    reject_ambient_git_overrides()
    assert_trusted_repository_layout(repository_root)
    require_same_git_cli_identity(expected_cli_identity, context="before command")
    command = (
        expected_cli_identity.path,
        "--no-replace-objects",
        "-c",
        "core.fsmonitor=false",
        "-c",
        "core.untrackedCache=false",
        "-c",
        "core.excludesFile=/dev/null",
        "-c",
        "core.attributesFile=/dev/null",
        *arguments,
    )
    try:
        if stdout is None:
            completed = subprocess.run(
                command,
                cwd=repository_root,
                check=False,
                close_fds=True,
                stdin=subprocess.DEVNULL,
                capture_output=True,
                env=git_cli_environment(repository_root),
            )
        else:
            completed = subprocess.run(
                command,
                cwd=repository_root,
                check=False,
                close_fds=True,
                stdin=subprocess.DEVNULL,
                stdout=stdout,
                stderr=subprocess.PIPE,
                env=git_cli_environment(repository_root),
            )
    except OSError as error:
        raise GitBoundaryError("fixed Git command could not start") from error
    require_same_git_cli_identity(expected_cli_identity, context="after command")
    return GitCommandResult(
        returncode=completed.returncode,
        stdout=completed.stdout if isinstance(completed.stdout, bytes) else b"",
        stderr=completed.stderr if isinstance(completed.stderr, bytes) else b"",
    )


__all__ = [
    "GitBoundaryError",
    "GitCliIdentity",
    "GitCommandResult",
    "assert_trusted_repository_layout",
    "discover_repository_root",
    "git_cli_path",
    "git_cli_environment",
    "reject_ambient_git_overrides",
    "require_same_git_cli_identity",
    "run_git_command",
    "verify_git_cli_identity",
]
