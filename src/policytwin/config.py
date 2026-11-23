"""Strict immutable configuration schemas for PolicyTwin experiments."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import NoReturn

_IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_MAX_SMOKE_EVENTS = 1_000_000


class ConfigValidationError(ValueError):
    """Raised when an experiment configuration cannot be trusted."""


def _fail(message: str) -> NoReturn:
    raise ConfigValidationError(message)


def _strict_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            _fail(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> NoReturn:
    _fail(f"non-finite JSON number is forbidden: {value}")


def _load_json_document(path: Path) -> tuple[dict[str, object], str]:
    try:
        source = path.read_bytes()
    except OSError as error:
        raise ConfigValidationError(f"cannot read {path}: {error}") from error
    source_sha256 = hashlib.sha256(source).hexdigest()
    try:
        raw = source.decode("utf-8")
        value = json.loads(
            raw,
            object_pairs_hook=_strict_object,
            parse_constant=_reject_json_constant,
        )
    except (json.JSONDecodeError, UnicodeError) as error:
        raise ConfigValidationError(f"invalid JSON in {path}: {error}") from error
    if not isinstance(value, dict):
        _fail(f"top-level JSON value in {path} must be an object")
    return value, source_sha256


def _validate_fields(
    mapping: Mapping[str, object], expected: frozenset[str], context: str
) -> None:
    actual = frozenset(mapping)
    missing = sorted(expected - actual)
    unknown = sorted(actual - expected)
    if missing:
        _fail(f"{context} is missing required fields: {', '.join(missing)}")
    if unknown:
        _fail(f"{context} has unknown fields: {', '.join(unknown)}")


def _require_str(mapping: Mapping[str, object], field: str) -> str:
    value = mapping[field]
    if not isinstance(value, str) or not value.strip():
        _fail(f"{field} must be a non-empty string")
    return value


def _require_identifier(mapping: Mapping[str, object], field: str) -> str:
    value = _require_str(mapping, field)
    if _IDENTIFIER_PATTERN.fullmatch(value) is None:
        _fail(f"{field} must be a stable ASCII identifier")
    return value


def _require_int(
    mapping: Mapping[str, object], field: str, *, minimum: int, maximum: int
) -> int:
    value = mapping[field]
    if isinstance(value, bool) or not isinstance(value, int):
        _fail(f"{field} must be an integer")
    if not minimum <= value <= maximum:
        _fail(f"{field} must be between {minimum} and {maximum}")
    return value


def _validate_int_value(
    field: str, value: object, *, minimum: int, maximum: int
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        _fail(f"{field} must be an integer")
    if not minimum <= value <= maximum:
        _fail(f"{field} must be between {minimum} and {maximum}")
    return value


def _validate_identifier_value(field: str, value: object) -> str:
    if not isinstance(value, str) or _IDENTIFIER_PATTERN.fullmatch(value) is None:
        _fail(f"{field} must be a stable ASCII identifier")
    return value


def _require_aware_datetime(mapping: Mapping[str, object], field: str) -> datetime:
    raw = _require_str(mapping, field)
    try:
        value = datetime.fromisoformat(raw)
    except ValueError as error:
        raise ConfigValidationError(f"{field} must be an ISO 8601 datetime") from error
    if value.tzinfo is None or value.utcoffset() is None:
        _fail(f"{field} must include an explicit timezone offset")
    return value.astimezone(UTC)


def _canonical_bytes(mapping: Mapping[str, object]) -> bytes:
    return json.dumps(
        mapping,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _semantic_hash(mapping: Mapping[str, object]) -> str:
    return hashlib.sha256(_canonical_bytes(mapping)).hexdigest()


@dataclass(frozen=True, slots=True)
class DataManifest:
    """Manifest for the deliberately data-free synthetic smoke experiment."""

    schema_version: int
    manifest_id: str
    data_kind: str
    description: str
    record_count: int
    files: tuple[str, ...]

    def __post_init__(self) -> None:
        """Enforce invariants even when constructed outside the JSON loader."""
        _validate_int_value("schema_version", self.schema_version, minimum=1, maximum=1)
        _validate_identifier_value("manifest_id", self.manifest_id)
        if not isinstance(self.data_kind, str) or self.data_kind != "synthetic":
            _fail("this engineering skeleton only permits synthetic manifests")
        if not isinstance(self.description, str) or not self.description.strip():
            _fail("description must be a non-empty string")
        _validate_int_value("record_count", self.record_count, minimum=0, maximum=0)
        if not isinstance(self.files, tuple):
            _fail("files must be an immutable tuple")
        if self.files:
            _fail("the synthetic smoke manifest must not reference data files")

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, object]) -> DataManifest:
        """Validate a mapping without accepting defaults or unknown fields."""
        _validate_fields(
            mapping,
            frozenset(
                {
                    "schema_version",
                    "manifest_id",
                    "data_kind",
                    "description",
                    "record_count",
                    "files",
                }
            ),
            "data manifest",
        )
        schema_version = _require_int(mapping, "schema_version", minimum=1, maximum=1)
        manifest_id = _require_identifier(mapping, "manifest_id")
        data_kind = _require_str(mapping, "data_kind")
        if data_kind != "synthetic":
            _fail("this engineering skeleton only permits synthetic manifests")
        description = _require_str(mapping, "description")
        record_count = _require_int(mapping, "record_count", minimum=0, maximum=0)
        raw_files = mapping["files"]
        if not isinstance(raw_files, Sequence) or isinstance(raw_files, str | bytes):
            _fail("files must be a JSON array")
        if len(raw_files) != 0:
            _fail("the synthetic smoke manifest must not reference data files")
        return cls(
            schema_version=schema_version,
            manifest_id=manifest_id,
            data_kind=data_kind,
            description=description,
            record_count=record_count,
            files=(),
        )

    def to_mapping(self) -> dict[str, object]:
        """Return the canonical machine-readable representation."""
        return {
            "data_kind": self.data_kind,
            "description": self.description,
            "files": list(self.files),
            "manifest_id": self.manifest_id,
            "record_count": self.record_count,
            "schema_version": self.schema_version,
        }


@dataclass(frozen=True, slots=True)
class SmokeExperimentConfig:
    """Immutable schema for the fully synthetic smoke experiment."""

    schema_version: int
    experiment_type: str
    experiment_id: str
    knowledge_cutoff: datetime
    data_manifest: str
    replicates: int
    horizon_steps: int
    initial_value_ticks: int
    shock_radius_ticks: int
    policy_shift_ticks: int
    seed: str

    def __post_init__(self) -> None:
        """Enforce invariants for both parsed and direct construction."""
        _validate_int_value("schema_version", self.schema_version, minimum=1, maximum=1)
        if (
            not isinstance(self.experiment_type, str)
            or self.experiment_type != "synthetic_smoke"
        ):
            _fail("only experiment_type='synthetic_smoke' is implemented")
        _validate_identifier_value("experiment_id", self.experiment_id)
        if not isinstance(self.knowledge_cutoff, datetime) or (
            self.knowledge_cutoff.tzinfo is None
            or self.knowledge_cutoff.utcoffset() is None
        ):
            _fail("knowledge_cutoff must include an explicit timezone offset")
        try:
            normalized_cutoff = self.knowledge_cutoff.astimezone(UTC)
        except (OverflowError, ValueError) as error:
            raise ConfigValidationError(
                "knowledge_cutoff is outside UTC range"
            ) from error
        object.__setattr__(self, "knowledge_cutoff", normalized_cutoff)
        if (
            not isinstance(self.data_manifest, str)
            or not self.data_manifest.strip()
            or Path(self.data_manifest).is_absolute()
            or "://" in self.data_manifest
        ):
            _fail("data_manifest must be a local relative path")
        _validate_int_value("replicates", self.replicates, minimum=1, maximum=100_000)
        _validate_int_value(
            "horizon_steps", self.horizon_steps, minimum=1, maximum=1_000_000
        )
        _validate_int_value(
            "initial_value_ticks",
            self.initial_value_ticks,
            minimum=1,
            maximum=1_000_000_000,
        )
        _validate_int_value(
            "shock_radius_ticks",
            self.shock_radius_ticks,
            minimum=0,
            maximum=1_000_000,
        )
        _validate_int_value(
            "policy_shift_ticks",
            self.policy_shift_ticks,
            minimum=-1_000_000,
            maximum=1_000_000,
        )
        if self.replicates * self.horizon_steps > _MAX_SMOKE_EVENTS:
            _fail(f"smoke experiment may contain at most {_MAX_SMOKE_EVENTS} events")
        if (
            not isinstance(self.seed, str)
            or not self.seed.strip()
            or len(self.seed) > 512
        ):
            _fail("seed must contain between 1 and 512 characters")

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, object]) -> SmokeExperimentConfig:
        """Validate a mapping strictly and fail closed on ambiguity."""
        _validate_fields(
            mapping,
            frozenset(
                {
                    "schema_version",
                    "experiment_type",
                    "experiment_id",
                    "knowledge_cutoff",
                    "data_manifest",
                    "replicates",
                    "horizon_steps",
                    "initial_value_ticks",
                    "shock_radius_ticks",
                    "policy_shift_ticks",
                    "seed",
                }
            ),
            "experiment config",
        )
        schema_version = _require_int(mapping, "schema_version", minimum=1, maximum=1)
        experiment_type = _require_str(mapping, "experiment_type")
        if experiment_type != "synthetic_smoke":
            _fail("only experiment_type='synthetic_smoke' is implemented")
        experiment_id = _require_identifier(mapping, "experiment_id")
        knowledge_cutoff = _require_aware_datetime(mapping, "knowledge_cutoff")
        data_manifest = _require_str(mapping, "data_manifest")
        if Path(data_manifest).is_absolute() or "://" in data_manifest:
            _fail("data_manifest must be a local relative path")
        replicates = _require_int(mapping, "replicates", minimum=1, maximum=100_000)
        horizon_steps = _require_int(
            mapping, "horizon_steps", minimum=1, maximum=1_000_000
        )
        initial_value_ticks = _require_int(
            mapping, "initial_value_ticks", minimum=1, maximum=1_000_000_000
        )
        shock_radius_ticks = _require_int(
            mapping, "shock_radius_ticks", minimum=0, maximum=1_000_000
        )
        policy_shift_ticks = _require_int(
            mapping,
            "policy_shift_ticks",
            minimum=-1_000_000,
            maximum=1_000_000,
        )
        seed = _require_str(mapping, "seed")
        if len(seed) > 512:
            _fail("seed must contain at most 512 characters")
        return cls(
            schema_version=schema_version,
            experiment_type=experiment_type,
            experiment_id=experiment_id,
            knowledge_cutoff=knowledge_cutoff,
            data_manifest=data_manifest,
            replicates=replicates,
            horizon_steps=horizon_steps,
            initial_value_ticks=initial_value_ticks,
            shock_radius_ticks=shock_radius_ticks,
            policy_shift_ticks=policy_shift_ticks,
            seed=seed,
        )

    def to_mapping(self) -> dict[str, object]:
        """Return the canonical machine-readable representation."""
        return {
            "data_manifest": self.data_manifest,
            "experiment_id": self.experiment_id,
            "experiment_type": self.experiment_type,
            "horizon_steps": self.horizon_steps,
            "initial_value_ticks": self.initial_value_ticks,
            "knowledge_cutoff": self.knowledge_cutoff.isoformat(),
            "policy_shift_ticks": self.policy_shift_ticks,
            "replicates": self.replicates,
            "schema_version": self.schema_version,
            "seed": self.seed,
            "shock_radius_ticks": self.shock_radius_ticks,
        }


@dataclass(frozen=True, slots=True)
class LoadedExperimentConfig:
    """Validated config, manifest, locations, and semantic/source hashes."""

    config_path: Path
    manifest_path: Path
    config: SmokeExperimentConfig
    manifest: DataManifest
    config_sha256: str
    manifest_sha256: str
    config_source_sha256: str
    manifest_source_sha256: str

    def __post_init__(self) -> None:
        """Reject forged paths, semantic hashes, and malformed source digests."""
        if not isinstance(self.config, SmokeExperimentConfig) or not isinstance(
            self.manifest, DataManifest
        ):
            _fail("loaded config contains an invalid schema object")
        if not self.config_path.is_absolute() or not self.manifest_path.is_absolute():
            _fail("loaded source paths must be absolute")
        if self.config_sha256 != _semantic_hash(self.config.to_mapping()):
            _fail("loaded config semantic hash is invalid")
        if self.manifest_sha256 != _semantic_hash(self.manifest.to_mapping()):
            _fail("loaded manifest semantic hash is invalid")
        for field, value in (
            ("config_source_sha256", self.config_source_sha256),
            ("manifest_source_sha256", self.manifest_source_sha256),
        ):
            if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
                _fail(f"{field} must be a lowercase SHA-256 digest")


def _source_root(config_path: Path) -> Path:
    for candidate in (config_path.parent, *config_path.parents):
        if (candidate / "pyproject.toml").is_file():
            return candidate.resolve()
    return config_path.parent.resolve()


def load_experiment_config(path: Path) -> LoadedExperimentConfig:
    """Load and validate a smoke configuration and its local manifest."""
    config_path = path.expanduser().resolve()
    config_mapping, config_source_sha256 = _load_json_document(config_path)
    config = SmokeExperimentConfig.from_mapping(config_mapping)
    manifest_path = (config_path.parent / config.data_manifest).resolve()
    allowed_root = _source_root(config_path)
    try:
        manifest_path.relative_to(allowed_root)
    except ValueError as error:
        raise ConfigValidationError(
            f"data manifest escapes the allowed source root: {allowed_root}"
        ) from error
    manifest_mapping, manifest_source_sha256 = _load_json_document(manifest_path)
    manifest = DataManifest.from_mapping(manifest_mapping)
    return LoadedExperimentConfig(
        config_path=config_path,
        manifest_path=manifest_path,
        config=config,
        manifest=manifest,
        config_sha256=_semantic_hash(config.to_mapping()),
        manifest_sha256=_semantic_hash(manifest.to_mapping()),
        config_source_sha256=config_source_sha256,
        manifest_source_sha256=manifest_source_sha256,
    )


__all__ = [
    "ConfigValidationError",
    "DataManifest",
    "LoadedExperimentConfig",
    "SmokeExperimentConfig",
    "load_experiment_config",
]
