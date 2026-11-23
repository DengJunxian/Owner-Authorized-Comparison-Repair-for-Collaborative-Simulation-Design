"""Fail-closed data-rights, privacy, and redistribution release screening.

This module performs mechanical checks over exact local declarations.  It does
not interpret law, establish ownership, or turn a caller-provided digest into
legal authority.  In particular, external or real data remain blocked until a
separate trusted rights-review issuer exists.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path, PurePosixPath
from typing import NoReturn, Self, cast

from policytwin.provenance import (
    MetadataValidationError,
    canonical_json_bytes,
    read_json_bytes_object,
    sha256_bytes,
    sha256_file,
)

RELEASE_RIGHTS_SCHEMA_VERSION = "policytwin.release-rights.v1"

_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class RightsRuleCode(StrEnum):
    """Stable machine-readable outcomes for the release-rights gate."""

    INVALID_SCHEMA = "INVALID_SCHEMA"
    MANIFEST_HASH_MISMATCH = "MANIFEST_HASH_MISMATCH"
    PATH_VIOLATION = "PATH_VIOLATION"
    FILE_HASH_MISMATCH = "FILE_HASH_MISMATCH"
    UNKNOWN_ASSET = "UNKNOWN_ASSET"
    USE_NOT_GRANTED = "USE_NOT_GRANTED"
    PROJECT_LICENSE_UNRESOLVED = "PROJECT_LICENSE_UNRESOLVED"
    SOURCE_BUNDLE_AUTHORITY_MISSING = "SOURCE_BUNDLE_AUTHORITY_MISSING"
    EXTERNAL_RIGHTS_AUTHORITY_MISSING = "EXTERNAL_RIGHTS_AUTHORITY_MISSING"
    PRIVACY_AUTHORITY_MISSING = "PRIVACY_AUTHORITY_MISSING"
    RAW_TEXT_RELEASE_DENIED = "RAW_TEXT_RELEASE_DENIED"
    DIRECT_IDENTIFIER_RELEASE_DENIED = "DIRECT_IDENTIFIER_RELEASE_DENIED"


class ReleaseRightsError(ValueError):
    """Base class carrying a stable rights-gate rule code."""

    def __init__(self, message: str, rule_code: RightsRuleCode) -> None:
        super().__init__(message)
        self.rule_code = rule_code


class ReleaseRightsSchemaError(ReleaseRightsError):
    """Raised when an immutable rights declaration has an invalid schema."""


class ReleaseRightsIntegrityError(ReleaseRightsError):
    """Raised when committed manifest, license, or data-manifest bytes changed."""


def _raise[T: ReleaseRightsError](
    error_type: type[T], rule_code: RightsRuleCode, message: str
) -> NoReturn:
    raise error_type(message, rule_code)


def _schema(message: str) -> NoReturn:
    _raise(ReleaseRightsSchemaError, RightsRuleCode.INVALID_SCHEMA, message)


def _identifier(value: object, field: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER.fullmatch(value) is None:
        _schema(f"{field} must be a stable ASCII identifier")
    return value


def _digest(value: object, field: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        _schema(f"{field} must be a lowercase SHA-256 digest")
    return value


def _optional_digest(value: object, field: str) -> str | None:
    if value is None:
        return None
    return _digest(value, field)


def _optional_text(value: object, field: str) -> str | None:
    if value is None:
        return None
    if (
        not isinstance(value, str)
        or not value.strip()
        or value != value.strip()
        or len(value) > 256
        or any(ord(character) < 32 for character in value)
    ):
        _schema(f"{field} must be null or bounded printable text")
    return value


def _relative_path(value: object, field: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value or "://" in value:
        _schema(f"{field} must be a portable relative path")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or path.as_posix() != value
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        _schema(f"{field} must not escape the project root")
    return value


def _fields(
    mapping: Mapping[str, object], expected: frozenset[str], context: str
) -> None:
    actual = frozenset(mapping)
    if actual != expected:
        _schema(
            f"{context} fields differ; missing={sorted(expected - actual)!r}, "
            f"unknown={sorted(actual - expected)!r}"
        )


def _mapping(value: object, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        _schema(f"{field} must be an object with string keys")
    return cast(Mapping[str, object], value)


def _array(value: object, field: str) -> Sequence[object]:
    if not isinstance(value, Sequence) or isinstance(value, str | bytes | bytearray):
        _schema(f"{field} must be an array")
    return cast(Sequence[object], value)


def _enum[T: StrEnum](value: object, enum_type: type[T], field: str) -> T:
    if not isinstance(value, str):
        _schema(f"{field} must be a string")
    try:
        return enum_type(value)
    except ValueError as error:
        raise ReleaseRightsSchemaError(
            f"unsupported {field}: {value}", RightsRuleCode.INVALID_SCHEMA
        ) from error


def _boolean(value: object, field: str) -> bool:
    if not isinstance(value, bool):
        _schema(f"{field} must be boolean")
    return value


class ProjectLicenseStatus(StrEnum):
    """Whether the repository owner has selected a source-code license."""

    UNRESOLVED_OWNER_CHOICE = "unresolved_owner_choice"
    OWNER_SELECTED = "owner_selected"


class DataOrigin(StrEnum):
    """Closed provenance classes relevant to local rights authority."""

    PROJECT_GENERATED_SYNTHETIC = "project_generated_synthetic"
    EXTERNAL_OR_REAL = "external_or_real"


class PrivacyClass(StrEnum):
    """Conservative privacy classification for one declared data asset."""

    SYNTHETIC_NON_PERSONAL = "synthetic_non_personal"
    PUBLIC_NON_PERSONAL = "public_non_personal"
    DEIDENTIFIED_PERSONAL = "deidentified_personal"
    PERSONAL = "personal"
    SENSITIVE_PERSONAL = "sensitive_personal"
    UNKNOWN = "unknown"


class DataUse(StrEnum):
    """Purposes that must be granted independently; a tag grants none."""

    INTERNAL_RESEARCH = "internal_research"
    PUBLISH_AGGREGATES = "publish_aggregates"
    REDISTRIBUTE_DERIVED = "redistribute_derived"
    REDISTRIBUTE_SOURCE = "redistribute_source"


class ReleaseTarget(StrEnum):
    """Closed release surfaces evaluated by the mechanical gate."""

    INTERNAL_SYNTHETIC_TEST = "internal_synthetic_test"
    PAPER_AGGREGATES = "paper_aggregates"
    DERIVED_ARTIFACT_BUNDLE = "derived_artifact_bundle"
    SOURCE_DATA_BUNDLE = "source_data_bundle"
    PROJECT_SOURCE_TREE = "project_source_tree"


_REQUIRED_USE = {
    ReleaseTarget.INTERNAL_SYNTHETIC_TEST: DataUse.INTERNAL_RESEARCH,
    ReleaseTarget.PAPER_AGGREGATES: DataUse.PUBLISH_AGGREGATES,
    ReleaseTarget.DERIVED_ARTIFACT_BUNDLE: DataUse.REDISTRIBUTE_DERIVED,
    ReleaseTarget.SOURCE_DATA_BUNDLE: DataUse.REDISTRIBUTE_SOURCE,
}

_PUBLIC_TARGETS = frozenset(
    {
        ReleaseTarget.PAPER_AGGREGATES,
        ReleaseTarget.DERIVED_ARTIFACT_BUNDLE,
        ReleaseTarget.SOURCE_DATA_BUNDLE,
    }
)


@dataclass(frozen=True, slots=True)
class ProjectLicenseDeclaration:
    """Owner-controlled project-license choice and exact local license bytes."""

    status: ProjectLicenseStatus
    spdx_expression: str | None
    license_path: str | None
    license_file_sha256: str | None

    def __post_init__(self) -> None:
        if not isinstance(self.status, ProjectLicenseStatus):
            _schema("project license status is invalid")
        expression = _optional_text(self.spdx_expression, "spdx_expression")
        path = (
            None
            if self.license_path is None
            else _relative_path(self.license_path, "license_path")
        )
        digest = _optional_digest(self.license_file_sha256, "license_file_sha256")
        selected_values = (expression, path, digest)
        if self.status is ProjectLicenseStatus.UNRESOLVED_OWNER_CHOICE:
            if any(value is not None for value in selected_values):
                _schema("an unresolved project license must not claim selected terms")
        elif any(value is None for value in selected_values):
            _schema(
                "an owner-selected project license requires expression, path, and hash"
            )

    def to_mapping(self) -> dict[str, object]:
        return {
            "license_file_sha256": self.license_file_sha256,
            "license_path": self.license_path,
            "spdx_expression": self.spdx_expression,
            "status": self.status.value,
        }

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, object]) -> Self:
        _fields(
            mapping,
            frozenset(
                {"status", "spdx_expression", "license_path", "license_file_sha256"}
            ),
            "project license declaration",
        )
        return cls(
            status=_enum(mapping["status"], ProjectLicenseStatus, "license status"),
            spdx_expression=_optional_text(
                mapping["spdx_expression"], "spdx_expression"
            ),
            license_path=(
                None
                if mapping["license_path"] is None
                else _relative_path(mapping["license_path"], "license_path")
            ),
            license_file_sha256=_optional_digest(
                mapping["license_file_sha256"], "license_file_sha256"
            ),
        )


@dataclass(frozen=True, slots=True)
class DataRightsDeclaration:
    """Exact per-asset permissions, privacy class, and source commitment."""

    asset_id: str
    manifest_path: str
    manifest_sha256: str
    origin: DataOrigin
    license_tag: str
    privacy_class: PrivacyClass
    contains_raw_text: bool
    contains_direct_identifiers: bool
    permitted_uses: tuple[DataUse, ...]
    external_rights_review_sha256: str | None
    privacy_review_sha256: str | None

    def __post_init__(self) -> None:
        _identifier(self.asset_id, "asset_id")
        _relative_path(self.manifest_path, "manifest_path")
        _digest(self.manifest_sha256, "manifest_sha256")
        if not isinstance(self.origin, DataOrigin):
            _schema("data origin is invalid")
        _identifier(self.license_tag, "license_tag")
        if not isinstance(self.privacy_class, PrivacyClass):
            _schema("privacy_class is invalid")
        _boolean(self.contains_raw_text, "contains_raw_text")
        _boolean(self.contains_direct_identifiers, "contains_direct_identifiers")
        if not isinstance(self.permitted_uses, tuple) or not self.permitted_uses:
            _schema("permitted_uses must be a non-empty immutable tuple")
        if not all(isinstance(item, DataUse) for item in self.permitted_uses):
            _schema("permitted_uses contains an invalid use")
        ordered_uses = tuple(sorted(self.permitted_uses, key=str))
        if len(set(ordered_uses)) != len(ordered_uses):
            _schema("permitted_uses must not contain duplicates")
        object.__setattr__(self, "permitted_uses", ordered_uses)
        _optional_digest(
            self.external_rights_review_sha256, "external_rights_review_sha256"
        )
        _optional_digest(self.privacy_review_sha256, "privacy_review_sha256")
        if self.origin is DataOrigin.PROJECT_GENERATED_SYNTHETIC:
            if self.privacy_class is not PrivacyClass.SYNTHETIC_NON_PERSONAL:
                _schema(
                    "project-generated synthetic data must be synthetic non-personal"
                )
            if self.contains_raw_text or self.contains_direct_identifiers:
                _schema(
                    "synthetic non-personal data cannot claim sensitive payload fields"
                )
            if self.external_rights_review_sha256 is not None:
                _schema(
                    "synthetic project data must not claim an external rights review"
                )
        elif self.privacy_class is PrivacyClass.SYNTHETIC_NON_PERSONAL:
            _schema("external or real data cannot be classified as project synthetic")

    def to_mapping(self) -> dict[str, object]:
        return {
            "asset_id": self.asset_id,
            "contains_direct_identifiers": self.contains_direct_identifiers,
            "contains_raw_text": self.contains_raw_text,
            "external_rights_review_sha256": self.external_rights_review_sha256,
            "license_tag": self.license_tag,
            "manifest_path": self.manifest_path,
            "manifest_sha256": self.manifest_sha256,
            "origin": self.origin.value,
            "permitted_uses": [item.value for item in self.permitted_uses],
            "privacy_class": self.privacy_class.value,
            "privacy_review_sha256": self.privacy_review_sha256,
        }

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, object]) -> Self:
        _fields(
            mapping,
            frozenset(
                {
                    "asset_id",
                    "manifest_path",
                    "manifest_sha256",
                    "origin",
                    "license_tag",
                    "privacy_class",
                    "contains_raw_text",
                    "contains_direct_identifiers",
                    "permitted_uses",
                    "external_rights_review_sha256",
                    "privacy_review_sha256",
                }
            ),
            "data rights declaration",
        )
        return cls(
            asset_id=_identifier(mapping["asset_id"], "asset_id"),
            manifest_path=_relative_path(mapping["manifest_path"], "manifest_path"),
            manifest_sha256=_digest(mapping["manifest_sha256"], "manifest_sha256"),
            origin=_enum(mapping["origin"], DataOrigin, "origin"),
            license_tag=_identifier(mapping["license_tag"], "license_tag"),
            privacy_class=_enum(
                mapping["privacy_class"], PrivacyClass, "privacy_class"
            ),
            contains_raw_text=_boolean(
                mapping["contains_raw_text"], "contains_raw_text"
            ),
            contains_direct_identifiers=_boolean(
                mapping["contains_direct_identifiers"],
                "contains_direct_identifiers",
            ),
            permitted_uses=tuple(
                _enum(item, DataUse, "permitted use")
                for item in _array(mapping["permitted_uses"], "permitted_uses")
            ),
            external_rights_review_sha256=_optional_digest(
                mapping["external_rights_review_sha256"],
                "external_rights_review_sha256",
            ),
            privacy_review_sha256=_optional_digest(
                mapping["privacy_review_sha256"], "privacy_review_sha256"
            ),
        )


@dataclass(frozen=True, slots=True)
class ReleaseRightsManifest:
    """Immutable project/data rights inventory; declarations are not authority."""

    schema_version: str
    manifest_id: str
    project_license: ProjectLicenseDeclaration
    assets: tuple[DataRightsDeclaration, ...]

    def __post_init__(self) -> None:
        if self.schema_version != RELEASE_RIGHTS_SCHEMA_VERSION:
            _schema("unsupported release-rights schema version")
        _identifier(self.manifest_id, "manifest_id")
        if not isinstance(self.project_license, ProjectLicenseDeclaration):
            _schema("project_license is invalid")
        if not isinstance(self.assets, tuple) or not all(
            isinstance(item, DataRightsDeclaration) for item in self.assets
        ):
            _schema("assets must be an immutable data-rights tuple")
        ordered = tuple(sorted(self.assets, key=lambda item: item.asset_id))
        if len({item.asset_id for item in ordered}) != len(ordered):
            _schema("asset IDs must be unique")
        if len({item.manifest_path for item in ordered}) != len(ordered):
            _schema("asset manifest paths must be unique")
        object.__setattr__(self, "assets", ordered)

    @property
    def sha256(self) -> str:
        return sha256_bytes(canonical_json_bytes(self.to_mapping()))

    def to_mapping(self) -> dict[str, object]:
        return {
            "assets": [item.to_mapping() for item in self.assets],
            "manifest_id": self.manifest_id,
            "project_license": self.project_license.to_mapping(),
            "schema_version": self.schema_version,
        }

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, object]) -> Self:
        _fields(
            mapping,
            frozenset({"schema_version", "manifest_id", "project_license", "assets"}),
            "release rights manifest",
        )
        return cls(
            schema_version=cast(str, mapping["schema_version"]),
            manifest_id=_identifier(mapping["manifest_id"], "manifest_id"),
            project_license=ProjectLicenseDeclaration.from_mapping(
                _mapping(mapping["project_license"], "project_license")
            ),
            assets=tuple(
                DataRightsDeclaration.from_mapping(_mapping(item, "asset"))
                for item in _array(mapping["assets"], "assets")
            ),
        )


@dataclass(frozen=True, slots=True)
class RightsFinding:
    """One deterministic blocker emitted by the mechanical gate."""

    rule_code: RightsRuleCode
    asset_id: str | None
    message: str

    def __post_init__(self) -> None:
        if not isinstance(self.rule_code, RightsRuleCode):
            _schema("finding rule_code is invalid")
        if self.asset_id is not None:
            _identifier(self.asset_id, "finding asset_id")
        if not isinstance(self.message, str) or not self.message:
            _schema("finding message must be non-empty")

    def to_mapping(self) -> dict[str, object]:
        return {
            "asset_id": self.asset_id,
            "message": self.message,
            "rule_code": self.rule_code.value,
        }


@dataclass(frozen=True, slots=True)
class ReleaseRightsDecision:
    """Machine-readable rights result, separate from scientific eligibility."""

    manifest_sha256: str
    target: ReleaseTarget
    asset_ids: tuple[str, ...]
    rights_gate_eligible: bool
    blockers: tuple[RightsFinding, ...]

    def __post_init__(self) -> None:
        _digest(self.manifest_sha256, "manifest_sha256")
        if not isinstance(self.target, ReleaseTarget):
            _schema("release target is invalid")
        if not isinstance(self.asset_ids, tuple):
            _schema("asset_ids must be an immutable tuple")
        if tuple(sorted(set(self.asset_ids))) != self.asset_ids:
            _schema("asset_ids must be sorted and unique")
        if not all(_IDENTIFIER.fullmatch(item) is not None for item in self.asset_ids):
            _schema("asset_ids contains an invalid identifier")
        if not isinstance(self.rights_gate_eligible, bool):
            _schema("rights_gate_eligible must be boolean")
        if not isinstance(self.blockers, tuple) or not all(
            isinstance(item, RightsFinding) for item in self.blockers
        ):
            _schema("blockers must be an immutable findings tuple")
        if self.rights_gate_eligible != (not self.blockers):
            _schema("rights eligibility must be false exactly when blockers exist")

    def to_mapping(self) -> dict[str, object]:
        return {
            "asset_ids": list(self.asset_ids),
            "blockers": [item.to_mapping() for item in self.blockers],
            "manifest_sha256": self.manifest_sha256,
            "rights_gate_eligible": self.rights_gate_eligible,
            "schema_version": 1,
            "target": self.target.value,
        }


def _verified_local_file(root: Path, relative_path: str, expected_sha256: str) -> Path:
    candidate = root / relative_path
    try:
        root_resolved = root.resolve(strict=True)
        candidate_resolved = candidate.resolve(strict=True)
    except OSError as error:
        raise ReleaseRightsIntegrityError(
            f"cannot resolve committed local file {relative_path}: {error}",
            RightsRuleCode.PATH_VIOLATION,
        ) from error
    try:
        candidate_resolved.relative_to(root_resolved)
    except ValueError as error:
        raise ReleaseRightsIntegrityError(
            f"committed local file escapes project root: {relative_path}",
            RightsRuleCode.PATH_VIOLATION,
        ) from error
    current = root_resolved
    for part in PurePosixPath(relative_path).parts:
        current = current / part
        if current.is_symlink():
            _raise(
                ReleaseRightsIntegrityError,
                RightsRuleCode.PATH_VIOLATION,
                f"committed local path contains a symbolic link: {relative_path}",
            )
    if not candidate_resolved.is_file():
        _raise(
            ReleaseRightsIntegrityError,
            RightsRuleCode.PATH_VIOLATION,
            f"committed local path is not a regular file: {relative_path}",
        )
    if sha256_file(candidate_resolved) != expected_sha256:
        _raise(
            ReleaseRightsIntegrityError,
            RightsRuleCode.FILE_HASH_MISMATCH,
            f"committed local file hash changed: {relative_path}",
        )
    return candidate_resolved


class LoadedReleaseRightsManifest:
    """Locally hash-verified declaration; still not external legal authority."""

    __slots__ = ("_manifest", "_project_root", "_source_path", "_source_sha256")

    def __init__(
        self,
        *,
        manifest: ReleaseRightsManifest,
        project_root: Path,
        source_path: Path,
        source_sha256: str,
        _token: object,
    ) -> None:
        if _token is not _LOAD_TOKEN:
            _schema(
                "loaded rights manifests may only be created by the verified loader"
            )
        self._manifest = manifest
        self._project_root = project_root
        self._source_path = source_path
        self._source_sha256 = source_sha256

    @property
    def manifest(self) -> ReleaseRightsManifest:
        return self._manifest

    @property
    def source_sha256(self) -> str:
        return self._source_sha256

    def reverify(self) -> None:
        """Rehash every committed local input immediately before a decision."""
        try:
            source_changed = (
                self._source_path.is_symlink()
                or sha256_file(self._source_path) != self._source_sha256
            )
        except OSError as error:
            raise ReleaseRightsIntegrityError(
                f"cannot reverify release-rights manifest: {error}",
                RightsRuleCode.MANIFEST_HASH_MISMATCH,
            ) from error
        if source_changed:
            _raise(
                ReleaseRightsIntegrityError,
                RightsRuleCode.MANIFEST_HASH_MISMATCH,
                "release-rights manifest changed after loading",
            )
        for asset in self._manifest.assets:
            _verified_local_file(
                self._project_root, asset.manifest_path, asset.manifest_sha256
            )
        project = self._manifest.project_license
        if project.status is ProjectLicenseStatus.OWNER_SELECTED:
            assert project.license_path is not None
            assert project.license_file_sha256 is not None
            _verified_local_file(
                self._project_root,
                project.license_path,
                project.license_file_sha256,
            )


_LOAD_TOKEN = object()


def load_release_rights_manifest(
    source_path: Path, *, project_root: Path, expected_source_sha256: str
) -> LoadedReleaseRightsManifest:
    """Load exact bytes and verify every referenced local manifest or license."""
    _digest(expected_source_sha256, "expected_source_sha256")
    try:
        root = project_root.resolve(strict=True)
        source = source_path.resolve(strict=True)
    except OSError as error:
        raise ReleaseRightsIntegrityError(
            f"cannot resolve release-rights input: {error}",
            RightsRuleCode.PATH_VIOLATION,
        ) from error
    try:
        source.relative_to(root)
    except ValueError as error:
        raise ReleaseRightsIntegrityError(
            "release-rights manifest is outside the project root",
            RightsRuleCode.PATH_VIOLATION,
        ) from error
    if source_path.is_symlink() or not source.is_file():
        _raise(
            ReleaseRightsIntegrityError,
            RightsRuleCode.PATH_VIOLATION,
            "release-rights manifest must be a regular non-symlink file",
        )
    actual_sha256 = sha256_file(source)
    if actual_sha256 != expected_source_sha256:
        _raise(
            ReleaseRightsIntegrityError,
            RightsRuleCode.MANIFEST_HASH_MISMATCH,
            "release-rights manifest hash does not match the external commitment",
        )
    try:
        mapping = read_json_bytes_object(source.read_bytes(), source=str(source))
        manifest = ReleaseRightsManifest.from_mapping(mapping)
    except (OSError, MetadataValidationError) as error:
        raise ReleaseRightsSchemaError(
            f"cannot parse release-rights manifest: {error}",
            RightsRuleCode.INVALID_SCHEMA,
        ) from error
    for asset in manifest.assets:
        _verified_local_file(root, asset.manifest_path, asset.manifest_sha256)
    project = manifest.project_license
    if project.status is ProjectLicenseStatus.OWNER_SELECTED:
        assert project.license_path is not None
        assert project.license_file_sha256 is not None
        _verified_local_file(root, project.license_path, project.license_file_sha256)
    return LoadedReleaseRightsManifest(
        manifest=manifest,
        project_root=root,
        source_path=source,
        source_sha256=actual_sha256,
        _token=_LOAD_TOKEN,
    )


def evaluate_release_rights(
    loaded: LoadedReleaseRightsManifest,
    *,
    target: ReleaseTarget,
    asset_ids: tuple[str, ...],
) -> ReleaseRightsDecision:
    """Evaluate exact use permissions without treating declarations as law."""
    if not isinstance(loaded, LoadedReleaseRightsManifest):
        _schema("release-rights evaluation requires a verified loaded manifest")
    if not isinstance(target, ReleaseTarget):
        _schema("target must be a ReleaseTarget")
    if not isinstance(asset_ids, tuple):
        _schema("asset_ids must be an immutable tuple")
    ordered_ids = tuple(sorted(_identifier(item, "asset_id") for item in asset_ids))
    if len(set(ordered_ids)) != len(ordered_ids):
        _schema("asset_ids must not contain duplicates")
    loaded.reverify()
    manifest = loaded.manifest
    by_id = {item.asset_id: item for item in manifest.assets}
    findings: list[RightsFinding] = []

    if target is ReleaseTarget.PROJECT_SOURCE_TREE:
        if ordered_ids:
            findings.append(
                RightsFinding(
                    rule_code=RightsRuleCode.USE_NOT_GRANTED,
                    asset_id=None,
                    message=(
                        "project source release cannot implicitly include data assets; "
                        "evaluate a data-bundle target separately"
                    ),
                )
            )
        if (
            manifest.project_license.status
            is ProjectLicenseStatus.UNRESOLVED_OWNER_CHOICE
        ):
            findings.append(
                RightsFinding(
                    rule_code=RightsRuleCode.PROJECT_LICENSE_UNRESOLVED,
                    asset_id=None,
                    message=(
                        "repository owner has not selected a project source license"
                    ),
                )
            )
        findings.append(
            RightsFinding(
                rule_code=RightsRuleCode.SOURCE_BUNDLE_AUTHORITY_MISSING,
                asset_id=None,
                message=(
                    "project source release requires an unimplemented independent "
                    "authority binding the exact SourceSnapshotManifest to every "
                    "bundled data and generated-result rights declaration; a code "
                    "license alone is insufficient"
                ),
            )
        )
        return ReleaseRightsDecision(
            manifest_sha256=loaded.source_sha256,
            target=target,
            asset_ids=ordered_ids,
            rights_gate_eligible=not findings,
            blockers=tuple(findings),
        )

    required_use = _REQUIRED_USE[target]
    if not ordered_ids:
        findings.append(
            RightsFinding(
                rule_code=RightsRuleCode.UNKNOWN_ASSET,
                asset_id=None,
                message="a data release target requires at least one explicit asset",
            )
        )
    for asset_id in ordered_ids:
        asset = by_id.get(asset_id)
        if asset is None:
            findings.append(
                RightsFinding(
                    rule_code=RightsRuleCode.UNKNOWN_ASSET,
                    asset_id=asset_id,
                    message="asset is absent from the frozen rights manifest",
                )
            )
            continue
        if required_use not in asset.permitted_uses:
            findings.append(
                RightsFinding(
                    rule_code=RightsRuleCode.USE_NOT_GRANTED,
                    asset_id=asset_id,
                    message=f"required use is not granted: {required_use.value}",
                )
            )
        if asset.origin is DataOrigin.EXTERNAL_OR_REAL:
            findings.append(
                RightsFinding(
                    rule_code=RightsRuleCode.EXTERNAL_RIGHTS_AUTHORITY_MISSING,
                    asset_id=asset_id,
                    message=(
                        "external or real assets require an unimplemented trusted "
                        "rights-review issuer; a declared receipt hash is not authority"
                    ),
                )
            )
        if target in _PUBLIC_TARGETS:
            if asset.contains_raw_text:
                findings.append(
                    RightsFinding(
                        rule_code=RightsRuleCode.RAW_TEXT_RELEASE_DENIED,
                        asset_id=asset_id,
                        message=(
                            "public release is blocked for an asset containing raw text"
                        ),
                    )
                )
            if asset.contains_direct_identifiers:
                findings.append(
                    RightsFinding(
                        rule_code=RightsRuleCode.DIRECT_IDENTIFIER_RELEASE_DENIED,
                        asset_id=asset_id,
                        message=(
                            "public release is blocked for an asset containing direct "
                            "identifiers"
                        ),
                    )
                )
            if asset.privacy_class not in {
                PrivacyClass.SYNTHETIC_NON_PERSONAL,
                PrivacyClass.PUBLIC_NON_PERSONAL,
            }:
                findings.append(
                    RightsFinding(
                        rule_code=RightsRuleCode.PRIVACY_AUTHORITY_MISSING,
                        asset_id=asset_id,
                        message=(
                            "personal, deidentified, sensitive, or unknown data "
                            "require "
                            "an unimplemented independent privacy review"
                        ),
                    )
                )

    return ReleaseRightsDecision(
        manifest_sha256=loaded.source_sha256,
        target=target,
        asset_ids=ordered_ids,
        rights_gate_eligible=not findings,
        blockers=tuple(findings),
    )


__all__ = [
    "RELEASE_RIGHTS_SCHEMA_VERSION",
    "DataOrigin",
    "DataRightsDeclaration",
    "DataUse",
    "LoadedReleaseRightsManifest",
    "PrivacyClass",
    "ProjectLicenseDeclaration",
    "ProjectLicenseStatus",
    "ReleaseRightsDecision",
    "ReleaseRightsError",
    "ReleaseRightsIntegrityError",
    "ReleaseRightsManifest",
    "ReleaseRightsSchemaError",
    "ReleaseTarget",
    "RightsFinding",
    "RightsRuleCode",
    "evaluate_release_rights",
    "load_release_rights_manifest",
]
