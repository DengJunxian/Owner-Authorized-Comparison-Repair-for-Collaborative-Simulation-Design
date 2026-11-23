"""Clean committed-source reproduction with a fresh, staged uv cache.

The online phase is limited to staging the environment named by ``uv.lock``
into a newly created job-scoped cache.  A second extraction of the exact same
``git archive HEAD`` is then synchronized and checked with uv offline mode
forced in both the command line and environment. This is a local candidate;
it is never production evidence and does not claim that the cache is a
portable wheelhouse or an independently retained release bundle.
"""

from __future__ import annotations

import argparse
import os
import platform
import re
import shutil
import stat
import subprocess
import sys
import tarfile
import tomllib
from collections.abc import Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path, PurePosixPath
from typing import NoReturn, Protocol

from policytwin.git_boundary import GitBoundaryError, GitCliIdentity
from policytwin.provenance import (
    MetadataValidationError,
    canonical_json_bytes,
    read_json_bytes_object,
    read_json_object,
    sha256_bytes,
    sha256_file,
    write_json_file,
    write_text_file,
)
from policytwin.release_preflight import (
    SOURCE_ARCHIVE_SCHEMA_VERSION,
    SourceArchiveReceipt,
    SourceFileDigest,
    SourcePreflightError,
    SourceSnapshotManifest,
    export_source_snapshot,
)

CLEAN_REPRODUCTION_SCHEMA_VERSION = "policytwin.clean-reproduction.v4"
TRUSTED_UV_MANIFEST_SCHEMA_VERSION = "policytwin.trusted-uv-tools.v1"
TRUSTED_UV_MANIFEST_RELATIVE_PATH = "configs/reproducibility/trusted_uv_tools.json"
_ARCHIVE_PREFIX = "policytwin-source"
_UV_VERSION = re.compile(r"^uv ([0-9]+(?:\.[0-9]+){2})(?:\s|$)")
_EXACT_VERSION = re.compile(r"^==([0-9]+(?:\.[0-9]+){2})$")
_SAFE_COMMAND_ID = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_HTTPS_URL = re.compile(r"^https://[^\s]+$")
_CACHE_COPY_CHUNK_BYTES = 1024 * 1024
_MAX_MATERIALIZED_CACHE_ENTRIES = 500_000
_MAX_MATERIALIZED_CACHE_BYTES = 8 * 1024 * 1024 * 1024
_INVENTORY_PROGRAM = """\
import hashlib
import importlib.metadata as metadata
import json
import platform
import sys


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

packages = sorted(
    (
        {
            "name": distribution.metadata["Name"].lower().replace("_", "-"),
            "version": distribution.version,
        }
        for distribution in metadata.distributions()
    ),
    key=lambda item: (item["name"], item["version"]),
)
print(
    json.dumps(
        {
            "packages": packages,
            "python_executable_sha256": sha256_file(sys.executable),
            "python_version": platform.python_version(),
        },
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )
)
"""


class CleanReproductionRuleCode(StrEnum):
    """Stable fail-closed outcomes for the clean reproduction boundary."""

    INVALID_CONFIGURATION = "INVALID_CONFIGURATION"
    WORK_ROOT_EXISTS = "WORK_ROOT_EXISTS"
    RECEIPT_EXISTS = "RECEIPT_EXISTS"
    UV_NOT_FOUND = "UV_NOT_FOUND"
    UV_VERSION_MISMATCH = "UV_VERSION_MISMATCH"
    SOURCE_PREFLIGHT_FAILED = "SOURCE_PREFLIGHT_FAILED"
    ARCHIVE_INVALID = "ARCHIVE_INVALID"
    LOCK_CONFIGURATION_INVALID = "LOCK_CONFIGURATION_INVALID"
    COMMAND_FAILED = "COMMAND_FAILED"
    CACHE_MISSING = "CACHE_MISSING"
    CACHE_UNSAFE_ENTRY = "CACHE_UNSAFE_ENTRY"
    CACHE_LIMIT_EXCEEDED = "CACHE_LIMIT_EXCEEDED"
    INVENTORY_INVALID = "INVENTORY_INVALID"
    TRUSTED_TOOL_MANIFEST_INVALID = "TRUSTED_TOOL_MANIFEST_INVALID"
    TOOL_PLATFORM_UNTRUSTED = "TOOL_PLATFORM_UNTRUSTED"
    TOOL_BINARY_UNTRUSTED = "TOOL_BINARY_UNTRUSTED"
    POST_RUN_AUDIT_FAILED = "POST_RUN_AUDIT_FAILED"
    RECEIPT_INVALID = "RECEIPT_INVALID"
    NON_PRODUCTION_EVIDENCE = "NON_PRODUCTION_EVIDENCE"
    IO_ERROR = "IO_ERROR"


class ExecutionAuthority(StrEnum):
    """Authority that actually executed receipt-producing commands."""

    TRUSTED_LOCAL_SUBPROCESS = "policytwin_trusted_local_subprocess_v1"
    INJECTED_TEST_EXECUTOR = "policytwin_injected_test_executor_v1"


class ReproductionEvidenceClass(StrEnum):
    """Non-interchangeable local-candidate and test evidence classes."""

    TRUSTED_LOCAL_CANDIDATE = "trusted_local_subprocess_candidate"
    SYNTHETIC_TEST_ONLY = "synthetic_test_only"


class CleanReproductionError(RuntimeError):
    """Raised after a clean reproduction violation has failed closed."""

    def __init__(
        self,
        message: str,
        rule_code: CleanReproductionRuleCode,
        *,
        command_id: str | None = None,
    ) -> None:
        super().__init__(message)
        self.rule_code = rule_code
        self.command_id = command_id


def _fail(
    rule_code: CleanReproductionRuleCode,
    message: str,
    *,
    command_id: str | None = None,
) -> NoReturn:
    raise CleanReproductionError(message, rule_code, command_id=command_id)


class _TestCommandExecutor(Protocol):
    """Test-only callable used by the permanently synthetic boundary."""

    def __call__(
        self,
        argv: Sequence[str],
        *,
        cwd: Path,
        env: Mapping[str, str],
    ) -> subprocess.CompletedProcess[bytes]: ...


@dataclass(frozen=True, slots=True)
class CleanReproductionConfig:
    """Immutable locations and executable identity for one clean job."""

    project_root: Path
    work_root: Path
    receipt_path: Path
    uv_executable: str = "uv"

    def __post_init__(self) -> None:
        if not isinstance(self.project_root, Path):
            _fail(
                CleanReproductionRuleCode.INVALID_CONFIGURATION,
                "project_root must be a Path",
            )
        if not isinstance(self.work_root, Path):
            _fail(
                CleanReproductionRuleCode.INVALID_CONFIGURATION,
                "work_root must be a Path",
            )
        if not isinstance(self.receipt_path, Path):
            _fail(
                CleanReproductionRuleCode.INVALID_CONFIGURATION,
                "receipt_path must be a Path",
            )
        if not isinstance(self.uv_executable, str) or not self.uv_executable:
            _fail(
                CleanReproductionRuleCode.INVALID_CONFIGURATION,
                "uv_executable must be a non-empty string",
            )


@dataclass(frozen=True, slots=True)
class _ExecutionEvidence:
    authority: ExecutionAuthority
    evidence_class: ReproductionEvidenceClass
    production_evidence: bool
    external_attestation_required: bool

    def __post_init__(self) -> None:
        local_candidate_pair = (
            self.authority is ExecutionAuthority.TRUSTED_LOCAL_SUBPROCESS
            and self.evidence_class is ReproductionEvidenceClass.TRUSTED_LOCAL_CANDIDATE
            and self.external_attestation_required is True
        )
        test_pair = (
            self.authority is ExecutionAuthority.INJECTED_TEST_EXECUTOR
            and self.evidence_class is ReproductionEvidenceClass.SYNTHETIC_TEST_ONLY
            and self.external_attestation_required is False
        )
        if (
            not (local_candidate_pair or test_pair)
            or self.production_evidence is not False
        ):
            _fail(
                CleanReproductionRuleCode.INVALID_CONFIGURATION,
                "execution authority and evidence class are inconsistent",
            )


_TRUSTED_LOCAL_CANDIDATE_EVIDENCE = _ExecutionEvidence(
    authority=ExecutionAuthority.TRUSTED_LOCAL_SUBPROCESS,
    evidence_class=ReproductionEvidenceClass.TRUSTED_LOCAL_CANDIDATE,
    production_evidence=False,
    external_attestation_required=True,
)
_SYNTHETIC_TEST_EVIDENCE = _ExecutionEvidence(
    authority=ExecutionAuthority.INJECTED_TEST_EXECUTOR,
    evidence_class=ReproductionEvidenceClass.SYNTHETIC_TEST_ONLY,
    production_evidence=False,
    external_attestation_required=False,
)


@dataclass(frozen=True, slots=True)
class TrustedUvToolEntry:
    """One exact upstream uv archive and executable identity."""

    version: str
    platform_system: str
    platform_machine: str
    archive_url: str
    archive_sha256: str
    checksum_url: str
    executable_archive_path: str
    executable_sha256: str

    def __post_init__(self) -> None:
        if (
            _EXACT_VERSION.fullmatch(f"=={self.version}") is None
            or not self.platform_system
            or not self.platform_machine
            or _HTTPS_URL.fullmatch(self.archive_url) is None
            or _SHA256.fullmatch(self.archive_sha256) is None
            or self.checksum_url != f"{self.archive_url}.sha256"
            or PurePosixPath(self.executable_archive_path).is_absolute()
            or any(
                part in {"", ".", ".."}
                for part in PurePosixPath(self.executable_archive_path).parts
            )
            or _SHA256.fullmatch(self.executable_sha256) is None
        ):
            _fail(
                CleanReproductionRuleCode.TRUSTED_TOOL_MANIFEST_INVALID,
                "trusted uv entry is not canonical",
            )

    def to_mapping(self) -> dict[str, str]:
        return {
            "archive_sha256": self.archive_sha256,
            "archive_url": self.archive_url,
            "checksum_url": self.checksum_url,
            "executable_archive_path": self.executable_archive_path,
            "executable_sha256": self.executable_sha256,
            "platform_machine": self.platform_machine,
            "platform_system": self.platform_system,
            "version": self.version,
        }


@dataclass(frozen=True, slots=True)
class TrustedUvToolManifest:
    """Strict committed trust candidates derived from upstream sidecars."""

    release_page_url: str
    trust_basis: str
    entries: tuple[TrustedUvToolEntry, ...]

    def __post_init__(self) -> None:
        identities = [
            (entry.platform_system, entry.platform_machine, entry.version)
            for entry in self.entries
        ]
        if (
            _HTTPS_URL.fullmatch(self.release_page_url) is None
            or self.trust_basis != "upstream_release_sha256_sidecar"
            or not self.entries
            or identities != sorted(identities)
            or len(identities) != len(set(identities))
        ):
            _fail(
                CleanReproductionRuleCode.TRUSTED_TOOL_MANIFEST_INVALID,
                "trusted uv manifest identities are not canonical",
            )


@dataclass(frozen=True, slots=True)
class _TrustedToolVerification:
    manifest_sha256: str
    entry_sha256: str


@dataclass(frozen=True, slots=True)
class CommandReceipt:
    """Hash-only command record with logs retained under the work root."""

    command_id: str
    phase: str
    argv: tuple[str, ...]
    cwd_role: str
    offline_forced: bool
    returncode: int
    stdout_log: str
    stdout_sha256: str
    stderr_log: str
    stderr_sha256: str

    def __post_init__(self) -> None:
        if _SAFE_COMMAND_ID.fullmatch(self.command_id) is None:
            _fail(
                CleanReproductionRuleCode.INVALID_CONFIGURATION,
                "command_id is not canonical",
            )
        if self.phase not in {"preflight", "online_staging", "offline_check"}:
            _fail(
                CleanReproductionRuleCode.INVALID_CONFIGURATION,
                "unknown command phase",
            )
        if not self.argv or any(not value for value in self.argv):
            _fail(
                CleanReproductionRuleCode.INVALID_CONFIGURATION,
                "command argv must be non-empty",
            )
        if self.returncode < 0:
            _fail(
                CleanReproductionRuleCode.INVALID_CONFIGURATION,
                "negative command return code is not supported",
            )

    def to_mapping(self) -> dict[str, object]:
        return {
            "argv": list(self.argv),
            "command_id": self.command_id,
            "cwd_role": self.cwd_role,
            "offline_forced": self.offline_forced,
            "phase": self.phase,
            "returncode": self.returncode,
            "stderr_log": self.stderr_log,
            "stderr_sha256": self.stderr_sha256,
            "stdout_log": self.stdout_log,
            "stdout_sha256": self.stdout_sha256,
        }


@dataclass(frozen=True, slots=True)
class FileTreeDigest:
    """Content identity for the isolated cache at one completed phase."""

    sha256: str
    entry_count: int
    total_regular_bytes: int

    def to_mapping(self) -> dict[str, object]:
        return {
            "entry_count": self.entry_count,
            "sha256": self.sha256,
            "total_regular_bytes": self.total_regular_bytes,
        }


@dataclass(frozen=True, slots=True)
class CleanReproductionReceipt:
    """Successful local candidate or synthetic receipt with closed provenance."""

    source_commit: str
    source_git_cli_path: str
    source_git_cli_platform_system: str
    source_git_cli_platform_machine: str
    source_git_cli_sha256: str
    source_git_cli_size_bytes: int
    execution_authority: ExecutionAuthority
    execution_authority_receipt_sha256: str
    evidence_class: ReproductionEvidenceClass
    production_evidence: bool
    external_attestation_required: bool
    trusted_tool_identity_verified: bool
    trusted_tool_manifest_sha256: str | None
    trusted_tool_entry_sha256: str | None
    source_tree_sha256: str
    source_archive_sha256: str
    source_archive_size_bytes: int
    source_archive_receipt_sha256: str
    lock_sha256: str
    required_python_version: str
    uv_version: str
    uv_binary_sha256: str
    verifier_python: str
    platform_system: str
    platform_machine: str
    staged_cache: FileTreeDigest
    final_cache: FileTreeDigest
    environment_inventory: Mapping[str, object]
    environment_inventory_sha256: str
    commands: tuple[CommandReceipt, ...]

    def __post_init__(self) -> None:
        if (
            self.source_git_cli_platform_system != self.platform_system
            or self.source_git_cli_platform_machine != self.platform_machine
        ):
            _fail(
                CleanReproductionRuleCode.INVALID_CONFIGURATION,
                "source Git CLI platform differs from reproduction platform",
            )
        try:
            GitCliIdentity(
                path=self.source_git_cli_path,
                platform_system=self.source_git_cli_platform_system,
                platform_machine=self.source_git_cli_platform_machine,
                sha256=self.source_git_cli_sha256,
                size_bytes=self.source_git_cli_size_bytes,
            )
        except GitBoundaryError as error:
            raise CleanReproductionError(
                "source Git CLI identity is invalid",
                CleanReproductionRuleCode.INVALID_CONFIGURATION,
            ) from error
        _ExecutionEvidence(
            authority=self.execution_authority,
            evidence_class=self.evidence_class,
            production_evidence=self.production_evidence,
            external_attestation_required=self.external_attestation_required,
        )
        local_candidate = (
            self.evidence_class is ReproductionEvidenceClass.TRUSTED_LOCAL_CANDIDATE
        )
        if (
            self.trusted_tool_identity_verified is not local_candidate
            or (
                local_candidate
                and (
                    self.trusted_tool_manifest_sha256 is None
                    or _SHA256.fullmatch(self.trusted_tool_manifest_sha256) is None
                    or self.trusted_tool_entry_sha256 is None
                    or _SHA256.fullmatch(self.trusted_tool_entry_sha256) is None
                )
            )
            or (
                not local_candidate
                and (
                    self.trusted_tool_manifest_sha256 is not None
                    or self.trusted_tool_entry_sha256 is not None
                )
            )
        ):
            _fail(
                CleanReproductionRuleCode.INVALID_CONFIGURATION,
                "trusted tool receipt fields do not match the evidence class",
            )

    def to_mapping(self) -> dict[str, object]:
        return {
            "cache_claim": "fresh_job_scoped_cache_not_portable_dependency_bundle",
            "commands": [command.to_mapping() for command in self.commands],
            "evidence_class": self.evidence_class.value,
            "environment_inventory": dict(self.environment_inventory),
            "environment_inventory_sha256": self.environment_inventory_sha256,
            "external_attestation_required": self.external_attestation_required,
            "final_cache": self.final_cache.to_mapping(),
            "lock_sha256": self.lock_sha256,
            "offline_boundary": "uv_offline_flag_and_environment",
            "execution_authority": self.execution_authority.value,
            "execution_authority_receipt_sha256": (
                self.execution_authority_receipt_sha256
            ),
            "platform_machine": self.platform_machine,
            "platform_system": self.platform_system,
            "portable_release_proven": False,
            "production_evidence": self.production_evidence,
            "required_python_version": self.required_python_version,
            "schema_version": CLEAN_REPRODUCTION_SCHEMA_VERSION,
            "source_archive_receipt_sha256": self.source_archive_receipt_sha256,
            "source_archive_sha256": self.source_archive_sha256,
            "source_archive_size_bytes": self.source_archive_size_bytes,
            "source_commit": self.source_commit,
            "source_git_cli_path": self.source_git_cli_path,
            "source_git_cli_platform_machine": self.source_git_cli_platform_machine,
            "source_git_cli_platform_system": self.source_git_cli_platform_system,
            "source_git_cli_sha256": self.source_git_cli_sha256,
            "source_git_cli_size_bytes": self.source_git_cli_size_bytes,
            "source_tree_sha256": self.source_tree_sha256,
            "staged_cache": self.staged_cache.to_mapping(),
            "status": "passed",
            "trusted_tool_entry_sha256": self.trusted_tool_entry_sha256,
            "trusted_tool_identity_verified": self.trusted_tool_identity_verified,
            "trusted_tool_manifest_path": TRUSTED_UV_MANIFEST_RELATIVE_PATH,
            "trusted_tool_manifest_sha256": self.trusted_tool_manifest_sha256,
            "uv_binary_sha256": self.uv_binary_sha256,
            "uv_version": self.uv_version,
            "verifier_python": self.verifier_python,
        }


@dataclass(frozen=True, slots=True)
class CleanReproductionAudit:
    """Result of rehashing a retained reproduction receipt and all logs."""

    receipt_status: str
    evidence_class: ReproductionEvidenceClass
    production_evidence: bool
    external_attestation_required: bool
    receipt_sha256: str
    receipt_canonical_sha256: str
    verified_command_count: int

    def to_mapping(self) -> dict[str, object]:
        return {
            "evidence_class": self.evidence_class.value,
            "external_attestation_required": self.external_attestation_required,
            "production_evidence": self.production_evidence,
            "receipt_canonical_sha256": self.receipt_canonical_sha256,
            "receipt_sha256": self.receipt_sha256,
            "receipt_status": self.receipt_status,
            "schema_version": "policytwin.clean-reproduction-audit.v2",
            "status": "verified",
            "verified_command_count": self.verified_command_count,
        }


class _CommandExecutionBoundary(Protocol):
    """Internal command boundary whose exact implementation fixes evidence class."""

    def execute(
        self,
        argv: Sequence[str],
        *,
        cwd: Path,
        env: Mapping[str, str],
    ) -> subprocess.CompletedProcess[bytes]: ...


class _TrustedLocalSubprocessBoundary:
    """Non-configurable local-candidate boundary backed by subprocess.run."""

    __slots__ = ()

    def execute(
        self,
        argv: Sequence[str],
        *,
        cwd: Path,
        env: Mapping[str, str],
    ) -> subprocess.CompletedProcess[bytes]:
        return subprocess.run(
            list(argv),
            cwd=cwd,
            env=dict(env),
            check=False,
            capture_output=True,
        )


class _InjectedTestBoundary:
    """Injectable boundary whose exact type is always classified as synthetic."""

    __slots__ = ("_executor",)

    def __init__(self, executor: _TestCommandExecutor) -> None:
        self._executor = executor

    def execute(
        self,
        argv: Sequence[str],
        *,
        cwd: Path,
        env: Mapping[str, str],
    ) -> subprocess.CompletedProcess[bytes]:
        return self._executor(argv, cwd=cwd, env=env)


def _evidence_for_boundary(
    boundary: _CommandExecutionBoundary,
) -> _ExecutionEvidence:
    """Map an exact built-in boundary type to its non-selectable evidence class."""
    if type(boundary) is _TrustedLocalSubprocessBoundary:
        return _TRUSTED_LOCAL_CANDIDATE_EVIDENCE
    if type(boundary) is _InjectedTestBoundary:
        return _SYNTHETIC_TEST_EVIDENCE
    _fail(
        CleanReproductionRuleCode.INVALID_CONFIGURATION,
        "unknown or subclassed command execution boundary",
    )


def _manifest_mapping(
    value: object, *, fields: frozenset[str], context: str
) -> Mapping[str, object]:
    if (
        not isinstance(value, Mapping)
        or not all(isinstance(key, str) for key in value)
        or frozenset(value) != fields
    ):
        _fail(
            CleanReproductionRuleCode.TRUSTED_TOOL_MANIFEST_INVALID,
            f"{context} does not match the closed schema",
        )
    return value


def _manifest_text(value: object, *, context: str) -> str:
    if not isinstance(value, str) or not value:
        _fail(
            CleanReproductionRuleCode.TRUSTED_TOOL_MANIFEST_INVALID,
            f"{context} must be a non-empty string",
        )
    return value


def _read_trusted_uv_manifest(source_root: Path) -> TrustedUvToolManifest:
    path = source_root / TRUSTED_UV_MANIFEST_RELATIVE_PATH
    if not path.is_file() or path.is_symlink():
        _fail(
            CleanReproductionRuleCode.TRUSTED_TOOL_MANIFEST_INVALID,
            "committed trusted uv manifest is missing or not a regular file",
        )
    try:
        document = read_json_object(path)
    except MetadataValidationError as error:
        raise CleanReproductionError(
            "committed trusted uv manifest is not strict JSON",
            CleanReproductionRuleCode.TRUSTED_TOOL_MANIFEST_INVALID,
        ) from error
    root = _manifest_mapping(
        document,
        fields=frozenset(
            {
                "entries",
                "release_page_url",
                "schema_version",
                "tool",
                "trust_basis",
            }
        ),
        context="trusted uv manifest",
    )
    if (
        root["schema_version"] != TRUSTED_UV_MANIFEST_SCHEMA_VERSION
        or root["tool"] != "uv"
    ):
        _fail(
            CleanReproductionRuleCode.TRUSTED_TOOL_MANIFEST_INVALID,
            "trusted uv manifest schema or tool is unsupported",
        )
    raw_entries = root["entries"]
    if not isinstance(raw_entries, list):
        _fail(
            CleanReproductionRuleCode.TRUSTED_TOOL_MANIFEST_INVALID,
            "trusted uv entries must be an array",
        )
    entries: list[TrustedUvToolEntry] = []
    entry_fields = frozenset(
        {
            "archive_sha256",
            "archive_url",
            "checksum_url",
            "executable_archive_path",
            "executable_sha256",
            "platform_machine",
            "platform_system",
            "version",
        }
    )
    for index, raw_entry in enumerate(raw_entries):
        entry = _manifest_mapping(
            raw_entry,
            fields=entry_fields,
            context=f"trusted uv entry {index}",
        )
        entries.append(
            TrustedUvToolEntry(
                version=_manifest_text(entry["version"], context="trusted uv version"),
                platform_system=_manifest_text(
                    entry["platform_system"], context="trusted platform system"
                ),
                platform_machine=_manifest_text(
                    entry["platform_machine"], context="trusted platform machine"
                ),
                archive_url=_manifest_text(
                    entry["archive_url"], context="trusted archive URL"
                ),
                archive_sha256=_manifest_text(
                    entry["archive_sha256"], context="trusted archive SHA-256"
                ),
                checksum_url=_manifest_text(
                    entry["checksum_url"], context="trusted checksum URL"
                ),
                executable_archive_path=_manifest_text(
                    entry["executable_archive_path"],
                    context="trusted executable archive path",
                ),
                executable_sha256=_manifest_text(
                    entry["executable_sha256"],
                    context="trusted executable SHA-256",
                ),
            )
        )
    manifest = TrustedUvToolManifest(
        release_page_url=_manifest_text(
            root["release_page_url"], context="trusted release page URL"
        ),
        trust_basis=_manifest_text(
            root["trust_basis"], context="trusted uv trust basis"
        ),
        entries=tuple(entries),
    )
    versions = {entry.version for entry in manifest.entries}
    if (
        len(versions) != 1
        or manifest.release_page_url
        != f"https://github.com/astral-sh/uv/releases/tag/{next(iter(versions))}"
        or any(
            not entry.archive_url.startswith(
                f"https://github.com/astral-sh/uv/releases/download/{entry.version}/uv-"
            )
            for entry in manifest.entries
        )
    ):
        _fail(
            CleanReproductionRuleCode.TRUSTED_TOOL_MANIFEST_INVALID,
            "trusted uv provenance URLs do not match the declared release",
        )
    return manifest


def _verify_trusted_uv_tool(
    source_root: Path,
    executable: Path,
    *,
    version: str,
    platform_system: str,
    platform_machine: str,
) -> _TrustedToolVerification:
    manifest = _read_trusted_uv_manifest(source_root)
    matching = tuple(
        entry
        for entry in manifest.entries
        if entry.version == version
        and entry.platform_system == platform_system
        and entry.platform_machine == platform_machine
    )
    if len(matching) != 1:
        _fail(
            CleanReproductionRuleCode.TOOL_PLATFORM_UNTRUSTED,
            "trusted uv manifest has no exact entry for this version and platform",
        )
    entry = matching[0]
    if (
        not executable.is_file()
        or executable.is_symlink()
        or sha256_file(executable) != entry.executable_sha256
    ):
        _fail(
            CleanReproductionRuleCode.TOOL_BINARY_UNTRUSTED,
            "uv executable bytes do not match the committed trusted-tool entry",
        )
    manifest_path = source_root / TRUSTED_UV_MANIFEST_RELATIVE_PATH
    return _TrustedToolVerification(
        manifest_sha256=sha256_file(manifest_path),
        entry_sha256=sha256_bytes(canonical_json_bytes(entry.to_mapping())),
    )


def _normalize_paths(config: CleanReproductionConfig) -> tuple[Path, Path, Path]:
    project_root = config.project_root.expanduser().resolve()
    if not project_root.is_dir():
        _fail(
            CleanReproductionRuleCode.INVALID_CONFIGURATION,
            "project_root must be an existing directory",
        )
    work_parent = config.work_root.expanduser().absolute().parent.resolve()
    if not work_parent.is_dir():
        _fail(
            CleanReproductionRuleCode.INVALID_CONFIGURATION,
            "work_root parent must exist",
        )
    work_root = work_parent / config.work_root.name
    if work_root.exists() or work_root.is_symlink():
        _fail(
            CleanReproductionRuleCode.WORK_ROOT_EXISTS,
            "work_root must be absent so no prior cache can be reused",
        )
    if project_root == work_root or project_root in work_root.parents:
        _fail(
            CleanReproductionRuleCode.INVALID_CONFIGURATION,
            "work_root must be outside the source repository",
        )

    raw_receipt = config.receipt_path.expanduser().absolute()
    receipt_parent = raw_receipt.parent
    if receipt_parent == work_root or work_root in receipt_parent.parents:
        normalized_parent = receipt_parent
    else:
        normalized_parent = receipt_parent.resolve()
        if not normalized_parent.is_dir():
            _fail(
                CleanReproductionRuleCode.INVALID_CONFIGURATION,
                "receipt parent must exist or be under work_root",
            )
    receipt_path = normalized_parent / raw_receipt.name
    if receipt_path.exists() or receipt_path.is_symlink():
        _fail(
            CleanReproductionRuleCode.RECEIPT_EXISTS,
            "receipt_path must not already exist",
        )
    seal_path = receipt_path.with_name(f"{receipt_path.name}.sha256")
    if seal_path.exists() or seal_path.is_symlink():
        _fail(
            CleanReproductionRuleCode.RECEIPT_EXISTS,
            "receipt seal path must not already exist",
        )
    if project_root == receipt_path or project_root in receipt_path.parents:
        _fail(
            CleanReproductionRuleCode.INVALID_CONFIGURATION,
            "receipt_path must be outside the source repository",
        )
    return project_root, work_root, receipt_path


def _resolve_uv(executable: str) -> Path:
    candidate = shutil.which(executable)
    if candidate is None:
        path = Path(executable).expanduser()
        candidate = str(path) if path.is_file() else None
    if candidate is None:
        _fail(
            CleanReproductionRuleCode.UV_NOT_FOUND,
            f"cannot locate uv executable: {executable}",
        )
    resolved = Path(candidate).resolve()
    if not resolved.is_file():
        _fail(
            CleanReproductionRuleCode.UV_NOT_FOUND,
            "resolved uv executable is not a regular file",
        )
    return resolved


def _safe_archive_relative(value: str) -> str | None:
    if "\\" in value or any(ord(character) < 32 for character in value):
        return None
    path = PurePosixPath(value)
    if path.is_absolute() or not path.parts or path.parts[0] != _ARCHIVE_PREFIX:
        return None
    relative = PurePosixPath(*path.parts[1:])
    if any(part in {"", ".", ".."} for part in relative.parts):
        return None
    return relative.as_posix() if relative.parts else ""


def _extract_verified_archive(
    archive_path: Path,
    destination: Path,
    source_receipt: SourceArchiveReceipt,
) -> None:
    if destination.exists() or destination.is_symlink():
        _fail(
            CleanReproductionRuleCode.ARCHIVE_INVALID,
            "archive destination must be absent",
        )
    destination.mkdir()
    expected = {entry.path: entry for entry in source_receipt.source_manifest.files}
    seen: set[str] = set()
    try:
        with tarfile.open(archive_path, mode="r:") as archive:
            for member in archive:
                relative = _safe_archive_relative(member.name)
                if relative is None:
                    _fail(
                        CleanReproductionRuleCode.ARCHIVE_INVALID,
                        "archive contains an unsafe path",
                    )
                if member.isdir():
                    if relative:
                        (destination / relative).mkdir(parents=True, exist_ok=True)
                    continue
                if not member.isfile() or not relative or relative in seen:
                    _fail(
                        CleanReproductionRuleCode.ARCHIVE_INVALID,
                        "archive contains a non-regular or duplicate entry",
                    )
                expected_entry = expected.get(relative)
                if expected_entry is None:
                    _fail(
                        CleanReproductionRuleCode.ARCHIVE_INVALID,
                        "archive contains a file absent from the source manifest",
                    )
                stream = archive.extractfile(member)
                if stream is None:
                    _fail(
                        CleanReproductionRuleCode.ARCHIVE_INVALID,
                        "cannot read an archived regular file",
                    )
                content = stream.read()
                if (
                    len(content) != expected_entry.size_bytes
                    or sha256_bytes(content) != expected_entry.sha256
                ):
                    _fail(
                        CleanReproductionRuleCode.ARCHIVE_INVALID,
                        "archived bytes differ from the committed source manifest",
                    )
                output = destination / relative
                output.parent.mkdir(parents=True, exist_ok=True)
                with output.open("xb") as target:
                    target.write(content)
                output.chmod(0o755 if expected_entry.mode == "100755" else 0o644)
                seen.add(relative)
    except (OSError, tarfile.TarError) as error:
        raise CleanReproductionError(
            f"cannot extract source archive: {error}",
            CleanReproductionRuleCode.ARCHIVE_INVALID,
        ) from error
    if seen != set(expected):
        _fail(
            CleanReproductionRuleCode.ARCHIVE_INVALID,
            "archive is missing committed source files",
        )


def _path_within(root: Path, candidate: Path) -> bool:
    return candidate == root or root in candidate.parents


def _resolve_internal_cache_link(path: Path, root: Path) -> Path:
    try:
        resolved = path.resolve(strict=True)
    except (OSError, RuntimeError) as error:
        raise CleanReproductionError(
            "uv cache contains a broken or cyclic symbolic link",
            CleanReproductionRuleCode.CACHE_UNSAFE_ENTRY,
        ) from error
    if not _path_within(root, resolved):
        _fail(
            CleanReproductionRuleCode.CACHE_UNSAFE_ENTRY,
            "uv cache symbolic link resolves outside the staged cache",
        )
    try:
        mode = resolved.stat().st_mode
    except OSError as error:
        raise CleanReproductionError(
            "cannot inspect a uv cache symbolic-link target",
            CleanReproductionRuleCode.CACHE_UNSAFE_ENTRY,
        ) from error
    if not (stat.S_ISREG(mode) or stat.S_ISDIR(mode)):
        _fail(
            CleanReproductionRuleCode.CACHE_UNSAFE_ENTRY,
            "uv cache symbolic link targets a special filesystem entry",
        )
    return resolved


def _tree_digest(root: Path, *, allow_internal_symlinks: bool) -> FileTreeDigest:
    if not root.is_dir() or root.is_symlink():
        _fail(
            CleanReproductionRuleCode.CACHE_MISSING,
            "the isolated uv cache was not created as a regular directory",
        )
    resolved_root = root.resolve(strict=True)
    entries: list[dict[str, object]] = []
    total_regular_bytes = 0
    for path in sorted(resolved_root.rglob("*"), key=lambda item: item.as_posix()):
        relative = path.relative_to(resolved_root).as_posix()
        if path.is_symlink():
            if not allow_internal_symlinks:
                _fail(
                    CleanReproductionRuleCode.CACHE_UNSAFE_ENTRY,
                    "materialized offline uv cache contains a symbolic link",
                )
            target = _resolve_internal_cache_link(path, resolved_root)
            entries.append(
                {
                    "kind": "internal_symlink",
                    "path": relative,
                    "target": target.relative_to(resolved_root).as_posix(),
                    "target_kind": "directory" if target.is_dir() else "file",
                }
            )
        elif path.is_file():
            size = path.stat().st_size
            total_regular_bytes += size
            entries.append(
                {
                    "kind": "file",
                    "path": relative,
                    "sha256": sha256_file(path),
                    "size_bytes": size,
                }
            )
        elif path.is_dir():
            entries.append({"kind": "directory", "path": relative})
        else:
            _fail(
                CleanReproductionRuleCode.CACHE_UNSAFE_ENTRY,
                "uv cache contains a special filesystem entry",
            )
    if not entries:
        _fail(
            CleanReproductionRuleCode.CACHE_MISSING,
            "the isolated uv cache is empty after online staging",
        )
    return FileTreeDigest(
        sha256=sha256_bytes(canonical_json_bytes(entries)),
        entry_count=len(entries),
        total_regular_bytes=total_regular_bytes,
    )


@dataclass(slots=True)
class _CacheCopyBudget:
    entries: int = 0
    regular_bytes: int = 0

    def add_entry(self) -> None:
        self.entries += 1
        if self.entries > _MAX_MATERIALIZED_CACHE_ENTRIES:
            _fail(
                CleanReproductionRuleCode.CACHE_LIMIT_EXCEEDED,
                "materialized uv cache exceeds the entry limit",
            )

    def add_bytes(self, count: int) -> None:
        self.regular_bytes += count
        if self.regular_bytes > _MAX_MATERIALIZED_CACHE_BYTES:
            _fail(
                CleanReproductionRuleCode.CACHE_LIMIT_EXCEEDED,
                "materialized uv cache exceeds the byte limit",
            )


def _materialize_cache_file(
    source: Path,
    destination: Path,
    *,
    source_stat: os.stat_result,
    budget: _CacheCopyBudget,
) -> None:
    budget.add_entry()
    budget.add_bytes(source_stat.st_size)
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor: int | None = None
    try:
        descriptor = os.open(source, flags)
        opened_stat = os.fstat(descriptor)
        if not stat.S_ISREG(opened_stat.st_mode) or (
            opened_stat.st_dev,
            opened_stat.st_ino,
        ) != (source_stat.st_dev, source_stat.st_ino):
            _fail(
                CleanReproductionRuleCode.CACHE_UNSAFE_ENTRY,
                "uv cache file changed while it was being materialized",
            )
        copied = 0
        input_stream = os.fdopen(descriptor, "rb")
        descriptor = None
        with input_stream, destination.open("xb") as output_stream:
            for chunk in iter(lambda: input_stream.read(_CACHE_COPY_CHUNK_BYTES), b""):
                copied += len(chunk)
                if copied > source_stat.st_size:
                    _fail(
                        CleanReproductionRuleCode.CACHE_UNSAFE_ENTRY,
                        "uv cache file grew while it was being materialized",
                    )
                output_stream.write(chunk)
        if copied != source_stat.st_size:
            _fail(
                CleanReproductionRuleCode.CACHE_UNSAFE_ENTRY,
                "uv cache file size changed while it was being materialized",
            )
        destination.chmod(0o755 if source_stat.st_mode & 0o111 else 0o644)
    except CleanReproductionError:
        raise
    except OSError as error:
        raise CleanReproductionError(
            f"cannot materialize uv cache file: {error}",
            CleanReproductionRuleCode.CACHE_UNSAFE_ENTRY,
        ) from error
    finally:
        if descriptor is not None:
            with suppress(OSError):
                os.close(descriptor)


def _materialize_cache_entry(
    source: Path,
    destination: Path,
    *,
    source_root: Path,
    ancestor_directories: frozenset[tuple[int, int]],
    budget: _CacheCopyBudget,
) -> None:
    try:
        source_stat = source.lstat()
    except OSError as error:
        raise CleanReproductionError(
            "cannot inspect staged uv cache entry",
            CleanReproductionRuleCode.CACHE_UNSAFE_ENTRY,
        ) from error
    if stat.S_ISLNK(source_stat.st_mode):
        resolved = _resolve_internal_cache_link(source, source_root)
        _materialize_cache_entry(
            resolved,
            destination,
            source_root=source_root,
            ancestor_directories=ancestor_directories,
            budget=budget,
        )
        return
    if stat.S_ISREG(source_stat.st_mode):
        _materialize_cache_file(
            source,
            destination,
            source_stat=source_stat,
            budget=budget,
        )
        return
    if stat.S_ISDIR(source_stat.st_mode):
        identity = (source_stat.st_dev, source_stat.st_ino)
        if identity in ancestor_directories:
            _fail(
                CleanReproductionRuleCode.CACHE_UNSAFE_ENTRY,
                "uv cache contains a recursive directory link",
            )
        budget.add_entry()
        try:
            destination.mkdir(mode=0o755)
            children = sorted(source.iterdir(), key=lambda item: item.name)
        except OSError as error:
            raise CleanReproductionError(
                "cannot materialize uv cache directory",
                CleanReproductionRuleCode.CACHE_UNSAFE_ENTRY,
            ) from error
        next_ancestors = ancestor_directories | {identity}
        for child in children:
            _materialize_cache_entry(
                child,
                destination / child.name,
                source_root=source_root,
                ancestor_directories=next_ancestors,
                budget=budget,
            )
        return
    _fail(
        CleanReproductionRuleCode.CACHE_UNSAFE_ENTRY,
        "uv cache contains a special filesystem entry",
    )


def _materialize_uv_cache(source_root: Path, destination_root: Path) -> None:
    if (
        not source_root.is_dir()
        or source_root.is_symlink()
        or destination_root.exists()
        or destination_root.is_symlink()
    ):
        _fail(
            CleanReproductionRuleCode.CACHE_UNSAFE_ENTRY,
            "cache materialization requires a regular source and absent destination",
        )
    resolved_source = source_root.resolve(strict=True)
    try:
        resolved_destination = (
            destination_root.parent.resolve(strict=True) / destination_root.name
        )
    except OSError as error:
        raise CleanReproductionError(
            "cache materialization destination parent is unavailable",
            CleanReproductionRuleCode.CACHE_UNSAFE_ENTRY,
        ) from error
    if _path_within(resolved_source, resolved_destination) or _path_within(
        resolved_destination, resolved_source
    ):
        _fail(
            CleanReproductionRuleCode.CACHE_UNSAFE_ENTRY,
            "staged and offline cache locations must not overlap",
        )
    try:
        _materialize_cache_entry(
            resolved_source,
            resolved_destination,
            source_root=resolved_source,
            ancestor_directories=frozenset(),
            budget=_CacheCopyBudget(),
        )
        _tree_digest(resolved_destination, allow_internal_symlinks=False)
    except CleanReproductionError as error:
        _cleanup_failed_cache_materialization(resolved_destination, primary_error=error)
        raise
    except OSError as error:
        wrapped = CleanReproductionError(
            "uv cache materialization failed during final verification",
            CleanReproductionRuleCode.CACHE_UNSAFE_ENTRY,
        )
        try:
            raise wrapped from error
        except CleanReproductionError as primary_error:
            _cleanup_failed_cache_materialization(
                resolved_destination, primary_error=primary_error
            )
            raise


def _cleanup_failed_cache_materialization(
    root: Path, *, primary_error: CleanReproductionError
) -> None:
    """Remove a partial cache or replace the primary failure with a cleanup failure."""
    try:
        root_info = os.lstat(root)
    except FileNotFoundError:
        return
    except OSError as error:
        raise CleanReproductionError(
            "cannot inspect partial uv cache during cleanup after "
            f"{primary_error.rule_code.value}: {error}",
            CleanReproductionRuleCode.IO_ERROR,
        ) from primary_error
    if stat.S_ISLNK(root_info.st_mode) or not stat.S_ISDIR(root_info.st_mode):
        raise CleanReproductionError(
            "partial uv cache cleanup root is linked or not a directory after "
            f"{primary_error.rule_code.value}",
            CleanReproductionRuleCode.IO_ERROR,
        ) from primary_error
    try:
        shutil.rmtree(root)
    except OSError as error:
        raise CleanReproductionError(
            "partial uv cache cleanup failed after "
            f"{primary_error.rule_code.value}: {error}",
            CleanReproductionRuleCode.IO_ERROR,
        ) from primary_error
    try:
        os.lstat(root)
    except FileNotFoundError:
        return
    except OSError as error:
        raise CleanReproductionError(
            "cannot prove partial uv cache absence after cleanup following "
            f"{primary_error.rule_code.value}: {error}",
            CleanReproductionRuleCode.IO_ERROR,
        ) from primary_error
    else:
        raise CleanReproductionError(
            "partial uv cache remains after cleanup following "
            f"{primary_error.rule_code.value}",
            CleanReproductionRuleCode.IO_ERROR,
        ) from primary_error


def _remove_disposable_uv_cache(root: Path) -> None:
    """Remove a used cache only after verifying its root cannot redirect cleanup."""
    try:
        root_info = os.lstat(root)
    except OSError as error:
        raise CleanReproductionError(
            "cannot inspect disposable uv cache root",
            CleanReproductionRuleCode.CACHE_UNSAFE_ENTRY,
        ) from error
    if stat.S_ISLNK(root_info.st_mode) or not stat.S_ISDIR(root_info.st_mode):
        _fail(
            CleanReproductionRuleCode.CACHE_UNSAFE_ENTRY,
            "disposable uv cache root is missing, linked, or not a directory",
        )
    try:
        shutil.rmtree(root)
    except OSError as error:
        raise CleanReproductionError(
            "disposable uv cache cleanup failed",
            CleanReproductionRuleCode.IO_ERROR,
        ) from error
    try:
        os.lstat(root)
    except FileNotFoundError:
        return
    except OSError as error:
        raise CleanReproductionError(
            "cannot prove disposable uv cache absence after cleanup",
            CleanReproductionRuleCode.IO_ERROR,
        ) from error
    else:
        _fail(
            CleanReproductionRuleCode.IO_ERROR,
            "disposable uv cache remains after cleanup",
        )


def _uv_required_version(source_root: Path) -> str:
    try:
        with (source_root / "pyproject.toml").open("rb") as stream:
            document = tomllib.load(stream)
        tool = document["tool"]
        uv = tool["uv"]
        required = uv["required-version"]
    except (OSError, KeyError, TypeError, tomllib.TOMLDecodeError) as error:
        raise CleanReproductionError(
            "pyproject.toml lacks a readable exact tool.uv required-version",
            CleanReproductionRuleCode.LOCK_CONFIGURATION_INVALID,
        ) from error
    if not isinstance(required, str):
        _fail(
            CleanReproductionRuleCode.LOCK_CONFIGURATION_INVALID,
            "tool.uv required-version must be a string",
        )
    match = _EXACT_VERSION.fullmatch(required)
    if match is None:
        _fail(
            CleanReproductionRuleCode.LOCK_CONFIGURATION_INVALID,
            "tool.uv required-version must be an exact three-part version",
        )
    return match.group(1)


def _required_python_version(source_root: Path) -> str:
    try:
        declaration = (source_root / ".python-version").read_text(
            encoding="utf-8", errors="strict"
        )
    except (OSError, UnicodeError) as error:
        raise CleanReproductionError(
            ".python-version is not readable UTF-8",
            CleanReproductionRuleCode.LOCK_CONFIGURATION_INVALID,
        ) from error
    if declaration not in {declaration.strip(), f"{declaration.strip()}\n"}:
        _fail(
            CleanReproductionRuleCode.LOCK_CONFIGURATION_INVALID,
            ".python-version must contain one canonical version line",
        )
    version = declaration.strip()
    if _EXACT_VERSION.fullmatch(f"=={version}") is None:
        _fail(
            CleanReproductionRuleCode.LOCK_CONFIGURATION_INVALID,
            ".python-version must pin an exact three-part version",
        )
    return version


def _subprocess_environment(
    *,
    runtime_root: Path,
    cache_root: Path | None = None,
    environment_root: Path | None = None,
    offline: bool,
) -> dict[str, str]:
    home = runtime_root / "home"
    temporary = runtime_root / "tmp"
    xdg_cache = runtime_root / "xdg-cache"
    xdg_config = runtime_root / "xdg-config"
    xdg_data = runtime_root / "xdg-data"
    for directory in (home, temporary, xdg_cache, xdg_config, xdg_data):
        directory.mkdir(parents=True, exist_ok=True)
    environment = {
        "HOME": str(home),
        "LANG": "C",
        "LC_ALL": "C",
        "PATH": os.defpath,
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONHASHSEED": "0",
        "PYTHONNOUSERSITE": "1",
        "TEMP": str(temporary),
        "TMP": str(temporary),
        "TMPDIR": str(temporary),
        "TZ": "UTC",
        "UV_NO_CONFIG": "1",
        "UV_NO_PROGRESS": "1",
        "UV_PYTHON_DOWNLOADS": "never",
        "XDG_CACHE_HOME": str(xdg_cache),
        "XDG_CONFIG_HOME": str(xdg_config),
        "XDG_DATA_HOME": str(xdg_data),
    }
    if cache_root is not None or environment_root is not None:
        if cache_root is None or environment_root is None:
            _fail(
                CleanReproductionRuleCode.INVALID_CONFIGURATION,
                "uv cache and environment roots must be configured together",
            )
        environment.update(
            {
                "UV_CACHE_DIR": str(cache_root),
                "UV_LINK_MODE": "copy",
                "UV_PYTHON": str(Path(sys.executable).resolve()),
                "UV_PROJECT_ENVIRONMENT": str(environment_root),
            }
        )
    if offline:
        environment["UV_OFFLINE"] = "1"
    return environment


class _CommandRecorder:
    def __init__(self, work_root: Path, boundary: _CommandExecutionBoundary) -> None:
        self._work_root = work_root
        self._boundary = boundary
        self.receipts: list[CommandReceipt] = []

    def run(
        self,
        command_id: str,
        *,
        phase: str,
        argv: Sequence[str],
        cwd: Path,
        cwd_role: str,
        env: Mapping[str, str],
        offline_forced: bool,
        recorded_argv: Sequence[str] | None = None,
    ) -> bytes:
        try:
            completed = self._boundary.execute(tuple(argv), cwd=cwd, env=env)
        except OSError as error:
            _fail(
                CleanReproductionRuleCode.COMMAND_FAILED,
                f"cannot execute {command_id}: {error}",
                command_id=command_id,
            )
        sequence = len(self.receipts) + 1
        log_root = self._work_root / "logs"
        log_root.mkdir(exist_ok=True)
        stdout_name = f"logs/{sequence:02d}-{command_id}.stdout.log"
        stderr_name = f"logs/{sequence:02d}-{command_id}.stderr.log"
        (self._work_root / stdout_name).write_bytes(completed.stdout)
        (self._work_root / stderr_name).write_bytes(completed.stderr)
        receipt = CommandReceipt(
            command_id=command_id,
            phase=phase,
            argv=tuple(
                recorded_argv if recorded_argv is not None else ("uv", *tuple(argv)[1:])
            ),
            cwd_role=cwd_role,
            offline_forced=offline_forced,
            returncode=completed.returncode,
            stdout_log=stdout_name,
            stdout_sha256=sha256_bytes(completed.stdout),
            stderr_log=stderr_name,
            stderr_sha256=sha256_bytes(completed.stderr),
        )
        self.receipts.append(receipt)
        if completed.returncode != 0:
            _fail(
                CleanReproductionRuleCode.COMMAND_FAILED,
                f"{command_id} exited with status {completed.returncode}",
                command_id=command_id,
            )
        return completed.stdout


def _validate_inventory(
    content: bytes, *, required_python_version: str
) -> dict[str, object]:
    try:
        inventory = read_json_bytes_object(content, source="environment inventory")
    except MetadataValidationError as error:
        raise CleanReproductionError(
            "offline environment inventory is not strict JSON",
            CleanReproductionRuleCode.INVENTORY_INVALID,
        ) from error
    if set(inventory) != {
        "packages",
        "python_executable_sha256",
        "python_version",
    }:
        _fail(
            CleanReproductionRuleCode.INVENTORY_INVALID,
            "offline environment inventory has unexpected fields",
        )
    packages = inventory["packages"]
    python_executable_sha256 = inventory["python_executable_sha256"]
    python_version = inventory["python_version"]
    if (
        not isinstance(packages, list)
        or not packages
        or not isinstance(python_version, str)
        or python_version != required_python_version
        or not isinstance(python_executable_sha256, str)
        or re.fullmatch(r"[0-9a-f]{64}", python_executable_sha256) is None
    ):
        _fail(
            CleanReproductionRuleCode.INVENTORY_INVALID,
            "offline environment inventory does not match the pinned Python runtime",
        )
    normalized: list[tuple[str, str]] = []
    for package in packages:
        if not isinstance(package, Mapping):
            _fail(
                CleanReproductionRuleCode.INVENTORY_INVALID,
                "environment package entry is not an object",
            )
        name = package.get("name")
        version = package.get("version")
        if (
            set(package) != {"name", "version"}
            or not isinstance(name, str)
            or not name
            or not isinstance(version, str)
            or not version
        ):
            _fail(
                CleanReproductionRuleCode.INVENTORY_INVALID,
                "environment package entry has invalid fields",
            )
        normalized.append((name, version))
    if normalized != sorted(normalized) or len(set(normalized)) != len(normalized):
        _fail(
            CleanReproductionRuleCode.INVENTORY_INVALID,
            "environment package inventory must be sorted and unique",
        )
    if not any(name == "policytwin" for name, _ in normalized):
        _fail(
            CleanReproductionRuleCode.INVENTORY_INVALID,
            "offline environment does not contain PolicyTwin",
        )
    return inventory


def _write_execution_authority_record(
    work_root: Path, evidence: _ExecutionEvidence
) -> str:
    path = work_root / "execution-authority.json"
    boundary = (
        "subprocess_run_shell_false_v1"
        if evidence.authority is ExecutionAuthority.TRUSTED_LOCAL_SUBPROCESS
        else "injected_test_callable_v1"
    )
    write_json_file(
        path,
        {
            "evidence_class": evidence.evidence_class.value,
            "execution_authority": evidence.authority.value,
            "executor_boundary": boundary,
            "external_attestation_required": evidence.external_attestation_required,
            "production_evidence": evidence.production_evidence,
            "schema_version": "policytwin.clean-execution-authority.v2",
        },
    )
    return sha256_file(path)


def _failure_mapping(
    error: CleanReproductionError,
    commands: Sequence[CommandReceipt],
    evidence: _ExecutionEvidence,
    authority_receipt_sha256: str,
) -> dict[str, object]:
    return {
        "cache_claim": "no_success_claim",
        "command_id": error.command_id,
        "commands": [command.to_mapping() for command in commands],
        "evidence_class": evidence.evidence_class.value,
        "error": str(error),
        "execution_authority": evidence.authority.value,
        "execution_authority_receipt_sha256": authority_receipt_sha256,
        "external_attestation_required": evidence.external_attestation_required,
        "portable_release_proven": False,
        "production_evidence": evidence.production_evidence,
        "rule_code": error.rule_code.value,
        "schema_version": CLEAN_REPRODUCTION_SCHEMA_VERSION,
        "status": "failed",
    }


def _write_sealed_receipt(path: Path, value: Mapping[str, object]) -> None:
    """Write strict JSON and an adjacent hash of the exact persisted bytes."""
    seal_path = path.with_name(f"{path.name}.sha256")
    if (
        path.exists()
        or path.is_symlink()
        or seal_path.exists()
        or seal_path.is_symlink()
    ):
        _fail(
            CleanReproductionRuleCode.RECEIPT_EXISTS,
            "receipt or its adjacent seal already exists",
        )
    write_json_file(path, value)
    try:
        write_text_file(seal_path, f"{sha256_file(path)}\n")
    except OSError:
        path.unlink(missing_ok=True)
        raise


def _safe_log_path(work_root: Path, logical_path: str) -> Path:
    path = PurePosixPath(logical_path)
    if (
        path.is_absolute()
        or path.as_posix() != logical_path
        or len(path.parts) != 2
        or path.parts[0] != "logs"
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        _fail(
            CleanReproductionRuleCode.RECEIPT_INVALID,
            "receipt contains an unsafe command-log path",
        )
    candidate = work_root
    for part in path.parts:
        candidate = candidate / part
        if candidate.is_symlink():
            _fail(
                CleanReproductionRuleCode.RECEIPT_INVALID,
                "command-log path crosses a symlink",
            )
    if not candidate.is_file():
        _fail(
            CleanReproductionRuleCode.RECEIPT_INVALID,
            "receipt command log is missing",
        )
    return candidate


def _audit_commands(raw_commands: object, *, work_root: Path, passed: bool) -> int:
    if not isinstance(raw_commands, list) or (passed and not raw_commands):
        _fail(
            CleanReproductionRuleCode.RECEIPT_INVALID,
            "receipt commands must be an array and passed receipts cannot be empty",
        )
    expected_success_ids = (
        "uv_version",
        "stage_locked_dependencies",
        "offline_sync",
        "offline_lock_check",
        "ruff_format_check",
        "ruff_lint",
        "mypy",
        "pytest",
        "environment_inventory",
    )
    command_ids: list[str] = []
    for index, raw_command in enumerate(raw_commands, start=1):
        if not isinstance(raw_command, Mapping):
            _fail(
                CleanReproductionRuleCode.RECEIPT_INVALID,
                "receipt command entry is not an object",
            )
        expected_fields = {
            "argv",
            "command_id",
            "cwd_role",
            "offline_forced",
            "phase",
            "returncode",
            "stderr_log",
            "stderr_sha256",
            "stdout_log",
            "stdout_sha256",
        }
        if set(raw_command) != expected_fields:
            _fail(
                CleanReproductionRuleCode.RECEIPT_INVALID,
                "receipt command entry has unexpected fields",
            )
        command_id = raw_command["command_id"]
        argv = raw_command["argv"]
        phase = raw_command["phase"]
        offline_forced = raw_command["offline_forced"]
        returncode = raw_command["returncode"]
        if (
            not isinstance(command_id, str)
            or _SAFE_COMMAND_ID.fullmatch(command_id) is None
            or not isinstance(argv, list)
            or not argv
            or any(not isinstance(item, str) or not item for item in argv)
            or argv[0] != "uv"
            or phase not in {"preflight", "online_staging", "offline_check"}
            or not isinstance(offline_forced, bool)
            or isinstance(returncode, bool)
            or not isinstance(returncode, int)
            or returncode < 0
        ):
            _fail(
                CleanReproductionRuleCode.RECEIPT_INVALID,
                "receipt command identity or status is invalid",
            )
        command_ids.append(command_id)
        if phase == "offline_check" and (not offline_forced or "--offline" not in argv):
            _fail(
                CleanReproductionRuleCode.RECEIPT_INVALID,
                "offline command lacks the recorded uv offline boundary",
            )
        if phase != "offline_check" and offline_forced:
            _fail(
                CleanReproductionRuleCode.RECEIPT_INVALID,
                "non-offline command is mislabeled as offline",
            )
        for stream_name in ("stdout", "stderr"):
            log_field = f"{stream_name}_log"
            hash_field = f"{stream_name}_sha256"
            log_name = raw_command[log_field]
            expected_hash = raw_command[hash_field]
            canonical_name = f"logs/{index:02d}-{command_id}.{stream_name}.log"
            if (
                not isinstance(log_name, str)
                or log_name != canonical_name
                or not isinstance(expected_hash, str)
                or re.fullmatch(r"[0-9a-f]{64}", expected_hash) is None
            ):
                _fail(
                    CleanReproductionRuleCode.RECEIPT_INVALID,
                    "receipt command-log commitment is invalid",
                )
            if sha256_file(_safe_log_path(work_root, log_name)) != expected_hash:
                _fail(
                    CleanReproductionRuleCode.RECEIPT_INVALID,
                    "retained command log differs from its receipt hash",
                )
    if len(set(command_ids)) != len(command_ids):
        _fail(
            CleanReproductionRuleCode.RECEIPT_INVALID,
            "receipt repeats a command identity",
        )
    if passed:
        if tuple(command_ids) != expected_success_ids or any(
            command["returncode"] != 0 for command in raw_commands
        ):
            _fail(
                CleanReproductionRuleCode.RECEIPT_INVALID,
                "passed receipt lacks the exact successful verification sequence",
            )
    else:
        if tuple(command_ids) != expected_success_ids[: len(command_ids)]:
            _fail(
                CleanReproductionRuleCode.RECEIPT_INVALID,
                "failed receipt command history is not a valid prefix",
            )
        if all(command["returncode"] == 0 for command in raw_commands):
            # Some pre-command failures (for example a version mismatch) happen
            # after successful uv-version capture and remain valid failures.
            pass
    return len(raw_commands)


def _source_archive_receipt_from_mapping(
    document: Mapping[str, object],
) -> SourceArchiveReceipt:
    if (
        set(document)
        != {
            "archive_sha256",
            "archive_size_bytes",
            "git_cli_identity",
            "manifest_path",
            "schema_version",
            "source_manifest",
        }
        or document.get("schema_version") != SOURCE_ARCHIVE_SCHEMA_VERSION
    ):
        _fail(
            CleanReproductionRuleCode.RECEIPT_INVALID,
            "retained source-archive receipt has an invalid schema",
        )
    raw_manifest = document.get("source_manifest")
    raw_git_cli_identity = document.get("git_cli_identity")
    try:
        git_cli_identity = GitCliIdentity.from_mapping(raw_git_cli_identity)
    except GitBoundaryError as error:
        raise CleanReproductionError(
            "retained source archive Git CLI identity is invalid",
            CleanReproductionRuleCode.RECEIPT_INVALID,
        ) from error
    if not isinstance(raw_manifest, Mapping) or set(raw_manifest) != {
        "commit",
        "file_count",
        "files",
        "git_object_format",
        "schema_version",
        "total_size_bytes",
        "tree_sha256",
    }:
        _fail(
            CleanReproductionRuleCode.RECEIPT_INVALID,
            "retained source manifest has unexpected fields",
        )
    raw_files = raw_manifest.get("files")
    if not isinstance(raw_files, list):
        _fail(
            CleanReproductionRuleCode.RECEIPT_INVALID,
            "retained source manifest files are invalid",
        )
    files: list[SourceFileDigest] = []
    try:
        for raw_file in raw_files:
            if not isinstance(raw_file, Mapping) or set(raw_file) != {
                "mode",
                "path",
                "sha256",
                "size_bytes",
            }:
                _fail(
                    CleanReproductionRuleCode.RECEIPT_INVALID,
                    "retained source file commitment is invalid",
                )
            path = raw_file["path"]
            mode = raw_file["mode"]
            size_bytes = raw_file["size_bytes"]
            digest = raw_file["sha256"]
            if (
                not isinstance(path, str)
                or not isinstance(mode, str)
                or isinstance(size_bytes, bool)
                or not isinstance(size_bytes, int)
                or not isinstance(digest, str)
            ):
                _fail(
                    CleanReproductionRuleCode.RECEIPT_INVALID,
                    "retained source file field types are invalid",
                )
            files.append(
                SourceFileDigest(
                    path=path,
                    mode=mode,
                    size_bytes=size_bytes,
                    sha256=digest,
                )
            )
        commit = raw_manifest["commit"]
        object_format = raw_manifest["git_object_format"]
        file_count = raw_manifest["file_count"]
        total_size = raw_manifest["total_size_bytes"]
        tree_digest = raw_manifest["tree_sha256"]
        archive_digest = document["archive_sha256"]
        archive_size = document["archive_size_bytes"]
        manifest_path = document["manifest_path"]
        if (
            not isinstance(commit, str)
            or not isinstance(object_format, str)
            or isinstance(file_count, bool)
            or not isinstance(file_count, int)
            or isinstance(total_size, bool)
            or not isinstance(total_size, int)
            or not isinstance(tree_digest, str)
            or not isinstance(archive_digest, str)
            or isinstance(archive_size, bool)
            or not isinstance(archive_size, int)
            or not isinstance(manifest_path, str)
        ):
            _fail(
                CleanReproductionRuleCode.RECEIPT_INVALID,
                "retained source/archive receipt field types are invalid",
            )
        manifest = SourceSnapshotManifest(
            schema_version=str(raw_manifest["schema_version"]),
            commit=commit,
            git_object_format=object_format,
            files=tuple(files),
            file_count=file_count,
            total_size_bytes=total_size,
            tree_sha256=tree_digest,
        )
        return SourceArchiveReceipt(
            archive_sha256=archive_digest,
            archive_size_bytes=archive_size,
            manifest_path=manifest_path,
            git_cli_identity=git_cli_identity,
            source_manifest=manifest,
        )
    except SourcePreflightError as error:
        raise CleanReproductionError(
            "retained source archive receipt fails its source schema",
            CleanReproductionRuleCode.RECEIPT_INVALID,
        ) from error


def _audit_execution_authority_record(
    document: Mapping[str, object],
    *,
    work_root: Path,
    expected_evidence: _ExecutionEvidence,
) -> None:
    expected_hash = document.get("execution_authority_receipt_sha256")
    path = work_root / "execution-authority.json"
    if (
        not isinstance(expected_hash, str)
        or re.fullmatch(r"[0-9a-f]{64}", expected_hash) is None
        or not path.is_file()
        or path.is_symlink()
        or sha256_file(path) != expected_hash
    ):
        _fail(
            CleanReproductionRuleCode.RECEIPT_INVALID,
            "execution-authority record is missing or differs from its hash",
        )
    try:
        record = read_json_object(path)
    except MetadataValidationError as error:
        raise CleanReproductionError(
            "execution-authority record is not strict JSON",
            CleanReproductionRuleCode.RECEIPT_INVALID,
        ) from error
    expected_boundary = (
        "subprocess_run_shell_false_v1"
        if expected_evidence.authority is ExecutionAuthority.TRUSTED_LOCAL_SUBPROCESS
        else "injected_test_callable_v1"
    )
    expected = {
        "evidence_class": expected_evidence.evidence_class.value,
        "execution_authority": expected_evidence.authority.value,
        "executor_boundary": expected_boundary,
        "external_attestation_required": (
            expected_evidence.external_attestation_required
        ),
        "production_evidence": expected_evidence.production_evidence,
        "schema_version": "policytwin.clean-execution-authority.v2",
    }
    if record != expected:
        _fail(
            CleanReproductionRuleCode.NON_PRODUCTION_EVIDENCE
            if expected_evidence.evidence_class
            is ReproductionEvidenceClass.TRUSTED_LOCAL_CANDIDATE
            else CleanReproductionRuleCode.RECEIPT_INVALID,
            "execution-authority record does not match the accepted boundary",
        )


def _audit_clean_reproduction_receipt(
    receipt_path: Path,
    *,
    work_root: Path,
    expected_evidence: _ExecutionEvidence,
) -> CleanReproductionAudit:
    """Verify strict JSON, adjacent seal, log hashes, and success artifacts."""
    receipt = receipt_path.expanduser().resolve()
    root = work_root.expanduser().resolve()
    if not receipt.is_file() or not root.is_dir():
        _fail(
            CleanReproductionRuleCode.RECEIPT_INVALID,
            "receipt and work_root must be existing regular locations",
        )
    seal_path = receipt.with_name(f"{receipt.name}.sha256")
    try:
        seal = seal_path.read_text(encoding="ascii", errors="strict")
    except (OSError, UnicodeError) as error:
        raise CleanReproductionError(
            "cannot read the adjacent reproduction receipt seal",
            CleanReproductionRuleCode.RECEIPT_INVALID,
        ) from error
    receipt_sha256 = sha256_file(receipt)
    if seal != f"{receipt_sha256}\n":
        _fail(
            CleanReproductionRuleCode.RECEIPT_INVALID,
            "reproduction receipt seal does not match exact JSON bytes",
        )
    try:
        document = read_json_object(receipt)
    except MetadataValidationError as error:
        raise CleanReproductionError(
            "reproduction receipt is not strict JSON",
            CleanReproductionRuleCode.RECEIPT_INVALID,
        ) from error
    if document.get("schema_version") != CLEAN_REPRODUCTION_SCHEMA_VERSION:
        _fail(
            CleanReproductionRuleCode.RECEIPT_INVALID,
            "reproduction receipt schema is unsupported",
        )
    status = document.get("status")
    if not isinstance(status, str) or status not in {"passed", "failed"}:
        _fail(
            CleanReproductionRuleCode.RECEIPT_INVALID,
            "reproduction receipt status is invalid",
        )
    success_fields = {
        "cache_claim",
        "commands",
        "evidence_class",
        "environment_inventory",
        "environment_inventory_sha256",
        "external_attestation_required",
        "final_cache",
        "execution_authority",
        "execution_authority_receipt_sha256",
        "lock_sha256",
        "offline_boundary",
        "platform_machine",
        "platform_system",
        "portable_release_proven",
        "production_evidence",
        "required_python_version",
        "schema_version",
        "source_archive_receipt_sha256",
        "source_archive_sha256",
        "source_archive_size_bytes",
        "source_commit",
        "source_git_cli_path",
        "source_git_cli_platform_machine",
        "source_git_cli_platform_system",
        "source_git_cli_sha256",
        "source_git_cli_size_bytes",
        "source_tree_sha256",
        "staged_cache",
        "status",
        "trusted_tool_entry_sha256",
        "trusted_tool_identity_verified",
        "trusted_tool_manifest_path",
        "trusted_tool_manifest_sha256",
        "uv_binary_sha256",
        "uv_version",
        "verifier_python",
    }
    failure_fields = {
        "cache_claim",
        "command_id",
        "commands",
        "evidence_class",
        "error",
        "execution_authority",
        "execution_authority_receipt_sha256",
        "external_attestation_required",
        "portable_release_proven",
        "production_evidence",
        "rule_code",
        "schema_version",
        "status",
    }
    if set(document) != (success_fields if status == "passed" else failure_fields):
        _fail(
            CleanReproductionRuleCode.RECEIPT_INVALID,
            "reproduction receipt has missing or unknown top-level fields",
        )
    if (
        document.get("execution_authority") != expected_evidence.authority.value
        or document.get("evidence_class") != expected_evidence.evidence_class.value
        or document.get("production_evidence")
        is not expected_evidence.production_evidence
        or document.get("external_attestation_required")
        is not expected_evidence.external_attestation_required
    ):
        _fail(
            CleanReproductionRuleCode.NON_PRODUCTION_EVIDENCE
            if expected_evidence.evidence_class
            is ReproductionEvidenceClass.TRUSTED_LOCAL_CANDIDATE
            else CleanReproductionRuleCode.RECEIPT_INVALID,
            "receipt execution authority or evidence class is not accepted",
        )
    _audit_execution_authority_record(
        document,
        work_root=root,
        expected_evidence=expected_evidence,
    )
    if document.get("portable_release_proven") is not False:
        _fail(
            CleanReproductionRuleCode.RECEIPT_INVALID,
            "reproduction receipt overstates portable-release evidence",
        )
    if document.get("production_evidence") is not False:
        _fail(
            CleanReproductionRuleCode.RECEIPT_INVALID,
            "repository-local receipts cannot claim production evidence",
        )
    command_count = _audit_commands(
        document.get("commands"), work_root=root, passed=status == "passed"
    )
    if status == "passed":
        archive_hash = document.get("source_archive_sha256")
        archive_receipt_hash = document.get("source_archive_receipt_sha256")
        archive_size = document.get("source_archive_size_bytes")
        lock_hash = document.get("lock_sha256")
        inventory = document.get("environment_inventory")
        inventory_hash = document.get("environment_inventory_sha256")
        staged_cache = document.get("staged_cache")
        final_cache = document.get("final_cache")
        required_python = document.get("required_python_version")
        uv_version = document.get("uv_version")
        uv_binary_hash = document.get("uv_binary_sha256")
        tool_manifest_hash = document.get("trusted_tool_manifest_sha256")
        tool_entry_hash = document.get("trusted_tool_entry_sha256")
        source_commit = document.get("source_commit")
        source_git_cli_path = document.get("source_git_cli_path")
        source_git_cli_platform_machine = document.get(
            "source_git_cli_platform_machine"
        )
        source_git_cli_platform_system = document.get("source_git_cli_platform_system")
        source_git_cli_sha256 = document.get("source_git_cli_sha256")
        source_git_cli_size_bytes = document.get("source_git_cli_size_bytes")
        source_tree_hash = document.get("source_tree_sha256")
        local_candidate = (
            expected_evidence.evidence_class
            is ReproductionEvidenceClass.TRUSTED_LOCAL_CANDIDATE
        )
        if (
            not isinstance(archive_hash, str)
            or not isinstance(archive_receipt_hash, str)
            or isinstance(archive_size, bool)
            or not isinstance(archive_size, int)
            or not isinstance(lock_hash, str)
            or not isinstance(inventory, Mapping)
            or not isinstance(inventory_hash, str)
            or not isinstance(staged_cache, Mapping)
            or not isinstance(final_cache, Mapping)
            or not isinstance(required_python, str)
            or not isinstance(uv_version, str)
            or not isinstance(uv_binary_hash, str)
            or not isinstance(source_commit, str)
            or not isinstance(source_git_cli_path, str)
            or not isinstance(source_git_cli_platform_machine, str)
            or not isinstance(source_git_cli_platform_system, str)
            or not isinstance(source_git_cli_sha256, str)
            or isinstance(source_git_cli_size_bytes, bool)
            or not isinstance(source_git_cli_size_bytes, int)
            or not isinstance(source_tree_hash, str)
            or (
                local_candidate
                and (
                    not isinstance(tool_manifest_hash, str)
                    or not isinstance(tool_entry_hash, str)
                )
            )
        ):
            _fail(
                CleanReproductionRuleCode.RECEIPT_INVALID,
                "passed receipt lacks required artifact commitments",
            )
        if (
            document.get("cache_claim")
            != "fresh_job_scoped_cache_not_portable_dependency_bundle"
            or document.get("offline_boundary") != "uv_offline_flag_and_environment"
            or document.get("verifier_python") != required_python
            or document.get("platform_system") != source_git_cli_platform_system
            or document.get("platform_machine") != source_git_cli_platform_machine
            or not isinstance(document.get("platform_machine"), str)
            or not document.get("platform_machine")
            or not isinstance(document.get("platform_system"), str)
            or not document.get("platform_system")
            or document.get("trusted_tool_manifest_path")
            != TRUSTED_UV_MANIFEST_RELATIVE_PATH
            or document.get("trusted_tool_identity_verified") is not local_candidate
            or (
                not local_candidate
                and (tool_manifest_hash is not None or tool_entry_hash is not None)
            )
        ):
            _fail(
                CleanReproductionRuleCode.RECEIPT_INVALID,
                "passed receipt has invalid fixed evidence declarations",
            )
        committed_files = (
            (root / "policytwin-source.tar", archive_hash),
            (
                root / "policytwin-source.tar.manifest.json",
                archive_receipt_hash,
            ),
            (root / "offline-source" / "uv.lock", lock_hash),
            (root / "tools" / "uv", uv_binary_hash),
        )
        if any(
            re.fullmatch(r"[0-9a-f]{64}", expected) is None
            or not path.is_file()
            or path.is_symlink()
            or sha256_file(path) != expected
            for path, expected in committed_files
        ):
            _fail(
                CleanReproductionRuleCode.RECEIPT_INVALID,
                "retained source archive or lock differs from the receipt",
            )
        if (root / "policytwin-source.tar").stat().st_size != archive_size:
            _fail(
                CleanReproductionRuleCode.RECEIPT_INVALID,
                "retained source archive size differs from the receipt",
            )
        try:
            archive_receipt_document = read_json_object(
                root / "policytwin-source.tar.manifest.json"
            )
        except MetadataValidationError as error:
            raise CleanReproductionError(
                "retained source archive receipt is not strict JSON",
                CleanReproductionRuleCode.RECEIPT_INVALID,
            ) from error
        source_receipt = _source_archive_receipt_from_mapping(archive_receipt_document)
        if (
            source_receipt.archive_sha256 != archive_hash
            or source_receipt.archive_size_bytes != archive_size
            or source_receipt.manifest_path != "policytwin-source.tar.manifest.json"
            or source_receipt.source_manifest.commit != source_commit
            or source_receipt.source_manifest.tree_sha256 != source_tree_hash
            or source_receipt.git_cli_identity.path != source_git_cli_path
            or source_receipt.git_cli_identity.platform_machine
            != source_git_cli_platform_machine
            or source_receipt.git_cli_identity.platform_system
            != source_git_cli_platform_system
            or source_receipt.git_cli_identity.sha256 != source_git_cli_sha256
            or source_receipt.git_cli_identity.size_bytes != source_git_cli_size_bytes
        ):
            _fail(
                CleanReproductionRuleCode.RECEIPT_INVALID,
                "source archive receipt does not match the reproduction receipt",
            )
        if _required_python_version(root / "offline-source") != required_python:
            _fail(
                CleanReproductionRuleCode.RECEIPT_INVALID,
                "retained Python declaration differs from the receipt",
            )
        if _uv_required_version(root / "offline-source") != uv_version:
            _fail(
                CleanReproductionRuleCode.RECEIPT_INVALID,
                "retained uv declaration differs from the receipt",
            )
        if local_candidate:
            platform_system = document["platform_system"]
            platform_machine = document["platform_machine"]
            assert isinstance(platform_system, str)
            assert isinstance(platform_machine, str)
            try:
                trusted_tool = _verify_trusted_uv_tool(
                    root / "offline-source",
                    root / "tools" / "uv",
                    version=uv_version,
                    platform_system=platform_system,
                    platform_machine=platform_machine,
                )
            except CleanReproductionError as error:
                raise CleanReproductionError(
                    "retained trusted uv evidence fails verification",
                    CleanReproductionRuleCode.RECEIPT_INVALID,
                ) from error
            if (
                trusted_tool.manifest_sha256 != tool_manifest_hash
                or trusted_tool.entry_sha256 != tool_entry_hash
            ):
                _fail(
                    CleanReproductionRuleCode.RECEIPT_INVALID,
                    "retained trusted uv manifest differs from the receipt",
                )
        validated_inventory = _validate_inventory(
            canonical_json_bytes(inventory),
            required_python_version=required_python,
        )
        if sha256_bytes(canonical_json_bytes(inventory)) != inventory_hash:
            _fail(
                CleanReproductionRuleCode.RECEIPT_INVALID,
                "environment inventory differs from its canonical hash",
            )
        python_binary_hash = validated_inventory["python_executable_sha256"]
        if (
            not isinstance(python_binary_hash, str)
            or not (root / "tools" / "python").is_file()
            or (root / "tools" / "python").is_symlink()
            or sha256_file(root / "tools" / "python") != python_binary_hash
        ):
            _fail(
                CleanReproductionRuleCode.RECEIPT_INVALID,
                "retained Python executable differs from the inventory",
            )
        disposable_cache = root / "offline-work-uv-cache"
        if disposable_cache.exists() or disposable_cache.is_symlink():
            _fail(
                CleanReproductionRuleCode.RECEIPT_INVALID,
                "disposable offline uv cache exists beside a successful receipt",
            )
        observed_stage_cache = _tree_digest(
            root / "staged-uv-cache", allow_internal_symlinks=True
        ).to_mapping()
        observed_final_cache = _tree_digest(
            root / "offline-uv-cache", allow_internal_symlinks=False
        ).to_mapping()
        if (
            dict(staged_cache) != observed_stage_cache
            or dict(final_cache) != observed_final_cache
        ):
            _fail(
                CleanReproductionRuleCode.RECEIPT_INVALID,
                "retained staged or final uv cache differs from the receipt",
            )
    else:
        rule_code = document.get("rule_code")
        command_id = document.get("command_id")
        if not isinstance(rule_code, str):
            _fail(
                CleanReproductionRuleCode.RECEIPT_INVALID,
                "failed receipt has a non-string rule code",
            )
        try:
            parsed_rule_code = CleanReproductionRuleCode(rule_code)
        except (TypeError, ValueError) as error:
            raise CleanReproductionError(
                "failed receipt has an unknown rule code",
                CleanReproductionRuleCode.RECEIPT_INVALID,
            ) from error
        if (
            document.get("cache_claim") != "no_success_claim"
            or not isinstance(document.get("error"), str)
            or not document.get("error")
            or parsed_rule_code is CleanReproductionRuleCode.RECEIPT_INVALID
            or (
                command_id is not None
                and (
                    not isinstance(command_id, str)
                    or _SAFE_COMMAND_ID.fullmatch(command_id) is None
                )
            )
        ):
            _fail(
                CleanReproductionRuleCode.RECEIPT_INVALID,
                "failed receipt has invalid fixed evidence declarations",
            )
    return CleanReproductionAudit(
        receipt_status=status,
        evidence_class=expected_evidence.evidence_class,
        production_evidence=expected_evidence.production_evidence,
        external_attestation_required=(expected_evidence.external_attestation_required),
        receipt_sha256=receipt_sha256,
        receipt_canonical_sha256=sha256_bytes(canonical_json_bytes(document)),
        verified_command_count=command_count,
    )


def audit_clean_reproduction_receipt(
    receipt_path: Path, *, work_root: Path
) -> CleanReproductionAudit:
    """Audit a trusted local candidate without upgrading it to production."""
    return _audit_clean_reproduction_receipt(
        receipt_path,
        work_root=work_root,
        expected_evidence=_TRUSTED_LOCAL_CANDIDATE_EVIDENCE,
    )


def _audit_synthetic_clean_reproduction_test_receipt(
    receipt_path: Path, *, work_root: Path
) -> CleanReproductionAudit:
    """Mechanically audit a permanently non-production test receipt."""
    return _audit_clean_reproduction_receipt(
        receipt_path,
        work_root=work_root,
        expected_evidence=_SYNTHETIC_TEST_EVIDENCE,
    )


def _execute_clean_reproduction(
    config: CleanReproductionConfig,
    *,
    boundary: _CommandExecutionBoundary,
) -> CleanReproductionReceipt:
    """Internal engine deriving evidence from an exact, closed boundary type."""
    evidence = _evidence_for_boundary(boundary)
    project_root, work_root, receipt_path = _normalize_paths(config)
    work_root.mkdir(mode=0o700)
    authority_receipt_sha256 = _write_execution_authority_record(work_root, evidence)
    recorder = _CommandRecorder(work_root, boundary)
    try:
        discovered_uv_path = _resolve_uv(config.uv_executable)
        archive_path = work_root / "policytwin-source.tar"
        try:
            source_receipt = export_source_snapshot(project_root, archive_path)
        except SourcePreflightError as error:
            raise CleanReproductionError(
                f"source snapshot preflight failed: {error}",
                CleanReproductionRuleCode.SOURCE_PREFLIGHT_FAILED,
            ) from error
        archive_receipt_path = Path(f"{archive_path}.manifest.json")

        stage_source = work_root / "stage-source"
        offline_source = work_root / "offline-source"
        _extract_verified_archive(archive_path, stage_source, source_receipt)
        required_uv_version = _uv_required_version(stage_source)
        required_python_version = _required_python_version(stage_source)
        platform_system = platform.system()
        platform_machine = platform.machine()
        if platform.python_version() != required_python_version:
            _fail(
                CleanReproductionRuleCode.LOCK_CONFIGURATION_INVALID,
                "verifier Python does not match the exact .python-version declaration",
            )
        trusted_tool = (
            _verify_trusted_uv_tool(
                stage_source,
                discovered_uv_path,
                version=required_uv_version,
                platform_system=platform_system,
                platform_machine=platform_machine,
            )
            if evidence.evidence_class
            is ReproductionEvidenceClass.TRUSTED_LOCAL_CANDIDATE
            else None
        )

        retained_uv_path = work_root / "tools" / "uv"
        retained_uv_path.parent.mkdir()
        shutil.copyfile(discovered_uv_path, retained_uv_path)
        retained_uv_path.chmod(0o755)
        if trusted_tool is not None:
            retained_tool = _verify_trusted_uv_tool(
                stage_source,
                retained_uv_path,
                version=required_uv_version,
                platform_system=platform_system,
                platform_machine=platform_machine,
            )
            if retained_tool != trusted_tool:
                _fail(
                    CleanReproductionRuleCode.TOOL_BINARY_UNTRUSTED,
                    "retained uv identity differs from the pre-copy commitment",
                )
            trusted_tool = retained_tool
        retained_python_path = work_root / "tools" / "python"
        shutil.copyfile(Path(sys.executable).resolve(), retained_python_path)
        retained_python_path.chmod(0o755)
        uv_path = retained_uv_path
        runtime_root = work_root / "runtime"
        base_environment = _subprocess_environment(
            runtime_root=runtime_root,
            offline=False,
        )
        version_output = recorder.run(
            "uv_version",
            phase="preflight",
            argv=(str(uv_path), "--version"),
            cwd=stage_source,
            cwd_role="stage_archive",
            env=base_environment,
            offline_forced=False,
        )
        try:
            version_text = version_output.decode("utf-8", errors="strict").strip()
        except UnicodeError as error:
            raise CleanReproductionError(
                "uv version output is not UTF-8",
                CleanReproductionRuleCode.UV_VERSION_MISMATCH,
            ) from error
        version_match = _UV_VERSION.match(version_text)
        if version_match is None or version_match.group(1) != required_uv_version:
            _fail(
                CleanReproductionRuleCode.UV_VERSION_MISMATCH,
                "uv executable version does not match pyproject.toml",
            )

        staged_cache_root = work_root / "staged-uv-cache"
        stage_environment_root = work_root / "stage-environment"
        online_environment = _subprocess_environment(
            runtime_root=runtime_root,
            cache_root=staged_cache_root,
            environment_root=stage_environment_root,
            offline=False,
        )
        recorder.run(
            "stage_locked_dependencies",
            phase="online_staging",
            argv=(str(uv_path), "sync", "--frozen"),
            cwd=stage_source,
            cwd_role="stage_archive",
            env=online_environment,
            offline_forced=False,
        )
        staged_cache = _tree_digest(staged_cache_root, allow_internal_symlinks=True)
        offline_work_cache_root = work_root / "offline-work-uv-cache"
        _materialize_uv_cache(staged_cache_root, offline_work_cache_root)

        _extract_verified_archive(archive_path, offline_source, source_receipt)
        lock_sha256 = sha256_file(offline_source / "uv.lock")
        offline_environment_root = offline_source / ".venv"
        offline_environment = _subprocess_environment(
            runtime_root=runtime_root,
            cache_root=offline_work_cache_root,
            environment_root=offline_environment_root,
            offline=True,
        )
        recorder.run(
            "offline_sync",
            phase="offline_check",
            argv=(str(uv_path), "sync", "--frozen", "--offline"),
            cwd=offline_source,
            cwd_role="fresh_offline_archive",
            env=offline_environment,
            offline_forced=True,
        )
        recorder.run(
            "offline_lock_check",
            phase="offline_check",
            argv=(str(uv_path), "lock", "--check", "--offline"),
            cwd=offline_source,
            cwd_role="fresh_offline_archive",
            env=offline_environment,
            offline_forced=True,
        )
        recorder.run(
            "ruff_format_check",
            phase="offline_check",
            argv=(
                str(uv_path),
                "run",
                "--frozen",
                "--offline",
                "ruff",
                "format",
                "--check",
                ".",
            ),
            cwd=offline_source,
            cwd_role="fresh_offline_archive",
            env=offline_environment,
            offline_forced=True,
        )
        recorder.run(
            "ruff_lint",
            phase="offline_check",
            argv=(
                str(uv_path),
                "run",
                "--frozen",
                "--offline",
                "ruff",
                "check",
                ".",
            ),
            cwd=offline_source,
            cwd_role="fresh_offline_archive",
            env=offline_environment,
            offline_forced=True,
        )
        recorder.run(
            "mypy",
            phase="offline_check",
            argv=(
                str(uv_path),
                "run",
                "--frozen",
                "--offline",
                "mypy",
                "src",
                "tests",
            ),
            cwd=offline_source,
            cwd_role="fresh_offline_archive",
            env=offline_environment,
            offline_forced=True,
        )
        recorder.run(
            "pytest",
            phase="offline_check",
            argv=(
                str(uv_path),
                "run",
                "--frozen",
                "--offline",
                "pytest",
                "-q",
            ),
            cwd=offline_source,
            cwd_role="fresh_offline_archive",
            env=offline_environment,
            offline_forced=True,
        )
        inventory_output = recorder.run(
            "environment_inventory",
            phase="offline_check",
            argv=(
                str(uv_path),
                "run",
                "--frozen",
                "--offline",
                str(offline_environment_root / "bin" / "python"),
                "-c",
                _INVENTORY_PROGRAM,
            ),
            cwd=offline_source,
            cwd_role="fresh_offline_archive",
            env=offline_environment,
            offline_forced=True,
            recorded_argv=(
                "uv",
                "run",
                "--frozen",
                "--offline",
                "python",
                "-c",
                "<policytwin-environment-inventory-v1>",
            ),
        )
        inventory = _validate_inventory(
            inventory_output, required_python_version=required_python_version
        )
        offline_cache_root = work_root / "offline-uv-cache"
        _materialize_uv_cache(offline_work_cache_root, offline_cache_root)
        _remove_disposable_uv_cache(offline_work_cache_root)
        final_cache = _tree_digest(offline_cache_root, allow_internal_symlinks=False)

        receipt = CleanReproductionReceipt(
            source_commit=source_receipt.source_manifest.commit,
            source_git_cli_path=source_receipt.git_cli_identity.path,
            source_git_cli_platform_system=(
                source_receipt.git_cli_identity.platform_system
            ),
            source_git_cli_platform_machine=(
                source_receipt.git_cli_identity.platform_machine
            ),
            source_git_cli_sha256=source_receipt.git_cli_identity.sha256,
            source_git_cli_size_bytes=source_receipt.git_cli_identity.size_bytes,
            execution_authority=evidence.authority,
            execution_authority_receipt_sha256=authority_receipt_sha256,
            evidence_class=evidence.evidence_class,
            production_evidence=evidence.production_evidence,
            external_attestation_required=evidence.external_attestation_required,
            trusted_tool_identity_verified=trusted_tool is not None,
            trusted_tool_manifest_sha256=(
                trusted_tool.manifest_sha256 if trusted_tool is not None else None
            ),
            trusted_tool_entry_sha256=(
                trusted_tool.entry_sha256 if trusted_tool is not None else None
            ),
            source_tree_sha256=source_receipt.source_manifest.tree_sha256,
            source_archive_sha256=source_receipt.archive_sha256,
            source_archive_size_bytes=source_receipt.archive_size_bytes,
            source_archive_receipt_sha256=sha256_file(archive_receipt_path),
            lock_sha256=lock_sha256,
            required_python_version=required_python_version,
            uv_version=required_uv_version,
            uv_binary_sha256=sha256_file(uv_path),
            verifier_python=platform.python_version(),
            platform_system=platform_system,
            platform_machine=platform_machine,
            staged_cache=staged_cache,
            final_cache=final_cache,
            environment_inventory=inventory,
            environment_inventory_sha256=sha256_bytes(canonical_json_bytes(inventory)),
            commands=tuple(recorder.receipts),
        )
        _write_sealed_receipt(receipt_path, receipt.to_mapping())
        try:
            _audit_clean_reproduction_receipt(
                receipt_path,
                work_root=work_root,
                expected_evidence=evidence,
            )
        except CleanReproductionError as audit_error:
            receipt_path.unlink(missing_ok=True)
            receipt_path.with_name(f"{receipt_path.name}.sha256").unlink(
                missing_ok=True
            )
            raise CleanReproductionError(
                "sealed success receipt failed the immediate post-run audit",
                CleanReproductionRuleCode.POST_RUN_AUDIT_FAILED,
            ) from audit_error
        return receipt
    except CleanReproductionError as error:
        try:
            _write_sealed_receipt(
                receipt_path,
                _failure_mapping(
                    error,
                    recorder.receipts,
                    evidence,
                    authority_receipt_sha256,
                ),
            )
        except OSError as receipt_error:
            raise CleanReproductionError(
                "reproduction failed and its receipt could not be written: "
                f"{receipt_error}",
                CleanReproductionRuleCode.IO_ERROR,
                command_id=error.command_id,
            ) from receipt_error
        raise
    except OSError as error:
        wrapped = CleanReproductionError(
            f"clean reproduction I/O failed: {error}",
            CleanReproductionRuleCode.IO_ERROR,
        )
        with suppress(OSError):
            _write_sealed_receipt(
                receipt_path,
                _failure_mapping(
                    wrapped,
                    recorder.receipts,
                    evidence,
                    authority_receipt_sha256,
                ),
            )
        raise wrapped from error


def run_clean_reproduction(
    config: CleanReproductionConfig,
) -> CleanReproductionReceipt:
    """Run a non-production candidate through the fixed trusted-tool boundary."""
    return _execute_clean_reproduction(
        config,
        boundary=_TrustedLocalSubprocessBoundary(),
    )


def _run_synthetic_clean_reproduction_test(
    config: CleanReproductionConfig,
    *,
    executor: _TestCommandExecutor,
) -> CleanReproductionReceipt:
    """Run injected test mechanics that can never become production evidence."""
    return _execute_clean_reproduction(
        config,
        boundary=_InjectedTestBoundary(executor),
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m policytwin.clean_reproduction",
        description=(
            "Stage locked dependencies in a fresh cache and reproduce an exact "
            "committed source archive as a non-production local candidate."
        ),
    )
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--receipt", type=Path)
    parser.add_argument("--audit-receipt", type=Path)
    parser.add_argument("--uv", default="uv")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Execute the clean reproduction CLI and emit a strict JSON decision."""
    args = _parser().parse_args(argv)
    receipt_path = (
        args.receipt
        if args.receipt is not None
        else args.work_root / "clean-reproduction-receipt.json"
    )
    try:
        if args.audit_receipt is not None:
            audit = audit_clean_reproduction_receipt(
                args.audit_receipt, work_root=args.work_root
            )
            sys.stdout.buffer.write(canonical_json_bytes(audit.to_mapping()) + b"\n")
            return 0
        receipt = run_clean_reproduction(
            CleanReproductionConfig(
                project_root=args.project_root,
                work_root=args.work_root,
                receipt_path=receipt_path,
                uv_executable=args.uv,
            )
        )
    except CleanReproductionError as error:
        sys.stderr.buffer.write(
            canonical_json_bytes(
                {
                    "command_id": error.command_id,
                    "error": str(error),
                    "receipt_path": str(receipt_path),
                    "rule_code": error.rule_code.value,
                    "schema_version": CLEAN_REPRODUCTION_SCHEMA_VERSION,
                    "status": "failed",
                }
            )
            + b"\n"
        )
        return 1
    sys.stdout.buffer.write(canonical_json_bytes(receipt.to_mapping()) + b"\n")
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through public API.
    raise SystemExit(main())


__all__ = [
    "CLEAN_REPRODUCTION_SCHEMA_VERSION",
    "TRUSTED_UV_MANIFEST_RELATIVE_PATH",
    "TRUSTED_UV_MANIFEST_SCHEMA_VERSION",
    "CleanReproductionConfig",
    "CleanReproductionAudit",
    "CleanReproductionError",
    "CleanReproductionReceipt",
    "CleanReproductionRuleCode",
    "CommandReceipt",
    "ExecutionAuthority",
    "FileTreeDigest",
    "ReproductionEvidenceClass",
    "TrustedUvToolEntry",
    "TrustedUvToolManifest",
    "audit_clean_reproduction_receipt",
    "main",
    "run_clean_reproduction",
]
