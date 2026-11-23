"""Fixed-layout rooted bundle for synthetic ledger posterior evidence.

The root-manifest digest supplied by a caller commits to one captured directory
tree.  Loading replays calibration, plan construction, every historical source,
and the final aggregation.  It does not prove external custody, registration,
historical execution, or independent reproduction.
"""

from __future__ import annotations

import os
import re
import stat
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from threading import RLock
from typing import Literal, NoReturn, Self, cast
from weakref import WeakKeyDictionary, WeakValueDictionary, finalize

from policytwin.calibration.artifacts import (
    CalibrationArtifactLoadResult,
    build_calibration_artifact_payloads,
    verify_calibration_artifact_snapshot,
)
from policytwin.calibration.workflow import CalibrationRun
from policytwin.counterfactual.integrated_market import IntegratedSyntheticMarketConfig
from policytwin.evaluation.ledger_bundle import (
    LedgerPosteriorPlanConstructionSpec,
    verify_ledger_posterior_plan_construction_bytes,
)
from policytwin.evaluation.ledger_posterior import (
    LedgerPosteriorEvaluation,
    LedgerPosteriorStudyPlan,
    verify_ledger_posterior_evaluation_bytes,
)
from policytwin.experiments.integrated_synthetic import (
    IntegratedProtocolExecutionError,
    ProtocolBoundIntegratedSyntheticRunArtifact,
    integrated_synthetic_forward_market_config,
    require_ledger_derivation_source_artifact,
    verify_protocol_bound_integrated_synthetic_run_artifact_bytes,
)
from policytwin.protocol import (
    ProtocolExecutionConsumer,
    ProtocolExecutionPurpose,
    ProtocolExecutionReceipt,
    ProtocolValidationError,
    StudyProtocol,
    validate_protocol_execution_receipt_binding,
)
from policytwin.provenance import (
    MetadataValidationError,
    canonical_json_bytes,
    read_json_bytes_object,
    sha256_bytes,
)

LEDGER_EVIDENCE_BUNDLE_SCHEMA_VERSION = "policytwin.ledger-evidence-rooted-bundle.v1"
LEDGER_EVIDENCE_BUNDLE_CLAIM_SCOPE: Literal[
    "synthetic_same_environment_reconstruction_mechanics_only"
] = "synthetic_same_environment_reconstruction_mechanics_only"

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SOURCE_NAME = re.compile(r"^source-([0-9]{6})\.json$")
_MAX_SOURCE_COUNT = 128
_MAX_FILE_BYTES = 32 * 1024 * 1024
_MAX_TOTAL_BYTES = 64 * 1024 * 1024
_MAX_MANIFEST_BYTES = 2 * 1024 * 1024
_MAX_PLAN_BYTES = 1024 * 1024
_MAX_SMALL_JSON_BYTES = 4 * 1024 * 1024
_INCOMPLETE_MARKER = ".policytwin-incomplete"

_CALIBRATION_FILES = frozenset(
    {
        "audit.jsonl",
        "calibration_report.json",
        "config.json",
        "event_keys.json",
        "input_receipt.json",
        "manifest.json",
        "posterior_predictive_checks.json",
        "posterior_samples.json",
        "protocol_execution_receipt.json",
        "provenance.json",
        "rank_histogram.json",
        "simulator_config.json",
        "true_parameters.json",
    }
)
_ROOT_FIXED_FILES = frozenset(
    {
        "batch-receipt.json",
        "bundle-manifest.json",
        "calibration-protocol.json",
        "evaluation-config.json",
        "plan-construction.json",
        "protocol.json",
        "result.json",
    }
)


class LedgerEvidenceBundleError(RuntimeError):
    """Raised when a rooted evidence bundle is incomplete or inconsistent."""


def _fail(message: str) -> NoReturn:
    raise LedgerEvidenceBundleError(message)


def _sha256(value: object, field: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        _fail(f"{field} must be a lowercase SHA-256 digest")
    return value


def _positive_int(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        _fail(f"{field} must be a positive integer")
    return value


def _strict_fields(
    value: Mapping[str, object], expected: frozenset[str], field: str
) -> None:
    if not all(isinstance(key, str) for key in value):
        _fail(f"{field} must have string keys")
    actual = frozenset(value)
    if actual != expected:
        _fail(
            f"{field} fields changed: missing={sorted(expected - actual)!r}, "
            f"extra={sorted(actual - expected)!r}"
        )


def _exact_boolean(value: object, expected: bool, field: str) -> bool:
    if not isinstance(value, bool) or value is not expected:
        _fail(f"{field} must remain {expected!r}")
    return value


@dataclass(frozen=True, slots=True)
class LedgerEvidenceBundleFile:
    """Hash-and-size commitment for one non-manifest leaf file."""

    path: str
    sha256: str
    size_bytes: int

    def __post_init__(self) -> None:
        if (
            not isinstance(self.path, str)
            or not self.path
            or self.path.startswith("/")
            or ".." in Path(self.path).parts
            or "\\" in self.path
        ):
            _fail("bundle file path must be a normalized relative POSIX path")
        _sha256(self.sha256, "bundle file sha256")
        _positive_int(self.size_bytes, "bundle file size_bytes")

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> Self:
        _strict_fields(value, frozenset({"path", "sha256", "size_bytes"}), "file")
        path = value["path"]
        if not isinstance(path, str):
            _fail("bundle file path must be text")
        return cls(
            path=path,
            sha256=_sha256(value["sha256"], "bundle file sha256"),
            size_bytes=_positive_int(value["size_bytes"], "bundle file size_bytes"),
        )

    def to_mapping(self) -> dict[str, object]:
        return {
            "path": self.path,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
        }


@dataclass(frozen=True, slots=True)
class LedgerEvidenceBundleManifest:
    """Closed root manifest; its own raw digest must come from the caller."""

    files: tuple[LedgerEvidenceBundleFile, ...]
    source_count: int
    calibration_manifest_raw_sha256: str
    calibration_protocol_sha256: str
    evaluation_config_sha256: str
    plan_construction_sha256: str
    expected_plan_sha256: str
    protocol_sha256: str
    batch_receipt_sha256: str
    result_sha256: str
    claim_scope: Literal["synthetic_same_environment_reconstruction_mechanics_only"] = (
        LEDGER_EVIDENCE_BUNDLE_CLAIM_SCOPE
    )
    synthetic_only: Literal[True] = True
    caller_commitment_only: Literal[True] = True
    external_custody_verified: Literal[False] = False
    externally_registered: Literal[False] = False
    independent_reproduction: Literal[False] = False
    historical_execution_attested: Literal[False] = False
    loaded_sources_replay_only: Literal[True] = True
    schema_version: str = LEDGER_EVIDENCE_BUNDLE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if (
            not isinstance(self.files, tuple)
            or not self.files
            or not all(type(item) is LedgerEvidenceBundleFile for item in self.files)
        ):
            _fail("bundle manifest files must be a non-empty typed tuple")
        paths = tuple(item.path for item in self.files)
        if paths != tuple(sorted(paths)) or len(paths) != len(set(paths)):
            _fail("bundle manifest files must use unique canonical path order")
        count = _positive_int(self.source_count, "source_count")
        if count > _MAX_SOURCE_COUNT:
            _fail("bundle source_count exceeds its fixed bound")
        expected_sources = {f"source-{ordinal:06d}.json" for ordinal in range(count)}
        observed_sources = {path for path in paths if _SOURCE_NAME.fullmatch(path)}
        if observed_sources != expected_sources:
            _fail("bundle manifest source paths are incomplete or non-canonical")
        for field in (
            "calibration_manifest_raw_sha256",
            "calibration_protocol_sha256",
            "evaluation_config_sha256",
            "plan_construction_sha256",
            "expected_plan_sha256",
            "protocol_sha256",
            "batch_receipt_sha256",
            "result_sha256",
        ):
            _sha256(getattr(self, field), field)
        if (
            self.claim_scope != LEDGER_EVIDENCE_BUNDLE_CLAIM_SCOPE
            or self.synthetic_only is not True
            or self.caller_commitment_only is not True
            or self.external_custody_verified is not False
            or self.externally_registered is not False
            or self.independent_reproduction is not False
            or self.historical_execution_attested is not False
            or self.loaded_sources_replay_only is not True
            or self.schema_version != LEDGER_EVIDENCE_BUNDLE_SCHEMA_VERSION
        ):
            _fail("bundle manifest schema or evidence-claim boundary changed")

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> Self:
        expected = frozenset(
            {
                "batch_receipt_sha256",
                "calibration_manifest_raw_sha256",
                "calibration_protocol_sha256",
                "caller_commitment_only",
                "claim_scope",
                "evaluation_config_sha256",
                "expected_plan_sha256",
                "external_custody_verified",
                "externally_registered",
                "files",
                "historical_execution_attested",
                "independent_reproduction",
                "loaded_sources_replay_only",
                "plan_construction_sha256",
                "protocol_sha256",
                "result_sha256",
                "schema_version",
                "source_count",
                "synthetic_only",
            }
        )
        _strict_fields(value, expected, "bundle manifest")
        raw_files = value["files"]
        if not isinstance(raw_files, list) or not raw_files:
            _fail("bundle manifest files must be a non-empty list")
        files: list[LedgerEvidenceBundleFile] = []
        for raw in raw_files:
            if not isinstance(raw, Mapping):
                _fail("bundle manifest file entry must be an object")
            files.append(LedgerEvidenceBundleFile.from_mapping(raw))
        claim_scope = value["claim_scope"]
        schema_version = value["schema_version"]
        if not isinstance(claim_scope, str) or not isinstance(schema_version, str):
            _fail("bundle manifest scope and schema must be text")
        return cls(
            files=tuple(files),
            source_count=_positive_int(value["source_count"], "source_count"),
            calibration_manifest_raw_sha256=_sha256(
                value["calibration_manifest_raw_sha256"],
                "calibration_manifest_raw_sha256",
            ),
            calibration_protocol_sha256=_sha256(
                value["calibration_protocol_sha256"],
                "calibration_protocol_sha256",
            ),
            evaluation_config_sha256=_sha256(
                value["evaluation_config_sha256"], "evaluation_config_sha256"
            ),
            plan_construction_sha256=_sha256(
                value["plan_construction_sha256"], "plan_construction_sha256"
            ),
            expected_plan_sha256=_sha256(
                value["expected_plan_sha256"], "expected_plan_sha256"
            ),
            protocol_sha256=_sha256(value["protocol_sha256"], "protocol_sha256"),
            batch_receipt_sha256=_sha256(
                value["batch_receipt_sha256"], "batch_receipt_sha256"
            ),
            result_sha256=_sha256(value["result_sha256"], "result_sha256"),
            claim_scope=cast(
                Literal["synthetic_same_environment_reconstruction_mechanics_only"],
                claim_scope,
            ),
            synthetic_only=cast(
                Literal[True],
                _exact_boolean(value["synthetic_only"], True, "synthetic_only"),
            ),
            caller_commitment_only=cast(
                Literal[True],
                _exact_boolean(
                    value["caller_commitment_only"], True, "caller_commitment_only"
                ),
            ),
            external_custody_verified=cast(
                Literal[False],
                _exact_boolean(
                    value["external_custody_verified"],
                    False,
                    "external_custody_verified",
                ),
            ),
            externally_registered=cast(
                Literal[False],
                _exact_boolean(
                    value["externally_registered"], False, "externally_registered"
                ),
            ),
            independent_reproduction=cast(
                Literal[False],
                _exact_boolean(
                    value["independent_reproduction"],
                    False,
                    "independent_reproduction",
                ),
            ),
            historical_execution_attested=cast(
                Literal[False],
                _exact_boolean(
                    value["historical_execution_attested"],
                    False,
                    "historical_execution_attested",
                ),
            ),
            loaded_sources_replay_only=cast(
                Literal[True],
                _exact_boolean(
                    value["loaded_sources_replay_only"],
                    True,
                    "loaded_sources_replay_only",
                ),
            ),
            schema_version=schema_version,
        )

    @property
    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.to_mapping())

    @property
    def sha256(self) -> str:
        return sha256_bytes(self.canonical_bytes)

    def to_mapping(self) -> dict[str, object]:
        return {
            "batch_receipt_sha256": self.batch_receipt_sha256,
            "calibration_manifest_raw_sha256": (self.calibration_manifest_raw_sha256),
            "calibration_protocol_sha256": self.calibration_protocol_sha256,
            "caller_commitment_only": self.caller_commitment_only,
            "claim_scope": self.claim_scope,
            "evaluation_config_sha256": self.evaluation_config_sha256,
            "expected_plan_sha256": self.expected_plan_sha256,
            "external_custody_verified": self.external_custody_verified,
            "externally_registered": self.externally_registered,
            "files": [item.to_mapping() for item in self.files],
            "historical_execution_attested": self.historical_execution_attested,
            "independent_reproduction": self.independent_reproduction,
            "loaded_sources_replay_only": self.loaded_sources_replay_only,
            "plan_construction_sha256": self.plan_construction_sha256,
            "protocol_sha256": self.protocol_sha256,
            "result_sha256": self.result_sha256,
            "schema_version": self.schema_version,
            "source_count": self.source_count,
            "synthetic_only": self.synthetic_only,
        }


class _LedgerEvidenceBundleLoadMintTicket:
    """Opaque one-use authority for one verified byte snapshot."""

    __slots__ = ("__weakref__",)

    def __init__(self) -> None:
        raise TypeError("ledger evidence load mint tickets are verifier-issued")

    def __copy__(self) -> NoReturn:
        raise TypeError("ledger evidence load mint tickets cannot be copied")

    def __deepcopy__(self, memo: object) -> NoReturn:
        del memo
        raise TypeError("ledger evidence load mint tickets cannot be copied")

    def __reduce__(self) -> NoReturn:
        raise TypeError("ledger evidence load mint tickets cannot be serialized")


@dataclass(frozen=True, slots=True)
class _LedgerEvidenceBundleLoadMintBinding:
    root: str
    root_snapshot_fingerprint: tuple[int, ...]
    manifest_raw_sha256: str


@dataclass(frozen=True, slots=True, init=False, weakref_slot=True)
class LedgerEvidenceBundleLoadResult:
    """Verifier-produced byte-snapshot replay with deliberately narrow claims.

    ``root`` records where the snapshot was captured; it is not a promise that
    the mutable path remains current after the loader returns.  Authority users
    must call :func:`require_verified_ledger_evidence_bundle_result` rather than
    trusting the type or its booleans alone.  Nested calibration and aggregate
    results are canonical value evidence, so an exact value copy is equivalent;
    nested source artifacts remain process-local origin capabilities and are
    rechecked individually.
    """

    root: Path
    root_snapshot_fingerprint: tuple[int, ...]
    manifest_raw_sha256: str
    manifest: LedgerEvidenceBundleManifest
    calibration: CalibrationArtifactLoadResult
    calibration_protocol: StudyProtocol
    plan_construction: LedgerPosteriorPlanConstructionSpec
    plan: LedgerPosteriorStudyPlan
    protocol: StudyProtocol
    batch_receipt: ProtocolExecutionReceipt
    source_artifacts: tuple[ProtocolBoundIntegratedSyntheticRunArtifact, ...]
    result: LedgerPosteriorEvaluation
    caller_manifest_commitment_verified: Literal[True] = True
    caller_commitment_only: Literal[True] = True
    external_custody_verified: Literal[False] = False
    externally_registered: Literal[False] = False
    independent_reproduction: Literal[False] = False
    historical_execution_attested: Literal[False] = False
    loaded_sources_replay_only: Literal[True] = True

    def __init__(self) -> None:
        raise TypeError(
            "LedgerEvidenceBundleLoadResult is verifier-produced; use "
            "load_ledger_evidence_bundle"
        )

    def __copy__(self) -> NoReturn:
        raise TypeError("verified ledger evidence results cannot be copied")

    def __deepcopy__(self, memo: object) -> NoReturn:
        del memo
        raise TypeError("verified ledger evidence results cannot be copied")

    def __reduce__(self) -> NoReturn:
        raise TypeError("verified ledger evidence results cannot be serialized")

    def _fresh_validate(self) -> None:
        if not isinstance(self.root, Path) or not self.root.is_absolute():
            _fail("loaded bundle root must be an absolute Path")
        if (
            type(self.root_snapshot_fingerprint) is not tuple
            or len(self.root_snapshot_fingerprint) != 9
            or not all(type(value) is int for value in self.root_snapshot_fingerprint)
        ):
            _fail("loaded bundle root snapshot fingerprint changed")
        _sha256(self.manifest_raw_sha256, "manifest_raw_sha256")
        exact_values: tuple[tuple[object, type[object], str], ...] = (
            (self.manifest, LedgerEvidenceBundleManifest, "manifest"),
            (self.calibration, CalibrationArtifactLoadResult, "calibration"),
            (self.calibration_protocol, StudyProtocol, "calibration_protocol"),
            (
                self.plan_construction,
                LedgerPosteriorPlanConstructionSpec,
                "plan_construction",
            ),
            (self.plan, LedgerPosteriorStudyPlan, "plan"),
            (self.protocol, StudyProtocol, "protocol"),
            (self.batch_receipt, ProtocolExecutionReceipt, "batch_receipt"),
            (self.result, LedgerPosteriorEvaluation, "result"),
        )
        for value, expected, field in exact_values:
            if type(value) is not expected:
                _fail(f"loaded bundle {field} has the wrong exact type")
        if (
            not isinstance(self.source_artifacts, tuple)
            or not self.source_artifacts
            or not all(
                type(item) is ProtocolBoundIntegratedSyntheticRunArtifact
                for item in self.source_artifacts
            )
        ):
            _fail("loaded bundle sources have the wrong exact tuple type")
        for source in self.source_artifacts:
            try:
                require_ledger_derivation_source_artifact(source)
            except IntegratedProtocolExecutionError as error:
                raise LedgerEvidenceBundleError(
                    "loaded bundle source lacks verifier-owned replay origin"
                ) from error
        if (
            self.caller_manifest_commitment_verified is not True
            or self.caller_commitment_only is not True
            or self.external_custody_verified is not False
            or self.externally_registered is not False
            or self.independent_reproduction is not False
            or self.historical_execution_attested is not False
            or self.loaded_sources_replay_only is not True
        ):
            _fail("loaded bundle claim flags cannot be widened")
        if (
            self.manifest.sha256 != self.manifest_raw_sha256
            or self.calibration.manifest_raw_sha256
            != self.manifest.calibration_manifest_raw_sha256
            or self.calibration_protocol.sha256
            != self.manifest.calibration_protocol_sha256
            or self.plan_construction.sha256 != self.manifest.plan_construction_sha256
            or self.plan.sha256 != self.manifest.expected_plan_sha256
            or self.protocol.sha256 != self.manifest.protocol_sha256
            or self.batch_receipt.sha256 != self.manifest.batch_receipt_sha256
            or self.result.sha256 != self.manifest.result_sha256
            or len(self.source_artifacts) != self.manifest.source_count
        ):
            _fail("loaded bundle cross-artifact hash binding changed")
        _validate_calibration_protocol(self.calibration_protocol, self.calibration.run)
        config = integrated_synthetic_forward_market_config()
        rebuilt = self.plan_construction.rebuild(
            calibration_run=self.calibration.run,
            config=config,
        )
        try:
            validate_protocol_execution_receipt_binding(
                self.protocol, self.batch_receipt
            )
        except ProtocolValidationError as error:
            raise LedgerEvidenceBundleError(
                f"loaded bundle batch receipt binding changed: {error}"
            ) from error
        if (
            rebuilt.canonical_bytes != self.plan.canonical_bytes
            or self.result.plan.canonical_bytes != self.plan.canonical_bytes
            or self.result.protocol_sha256 != self.protocol.sha256
            or self.result.batch_protocol_execution_receipt != self.batch_receipt
            or tuple(
                item.source_artifact_sha256 for item in self.result.draw_executions
            )
            != tuple(item.sha256 for item in self.source_artifacts)
            or tuple(item.input_sha256 for item in self.source_artifacts)
            != tuple(item.pair_input_sha256 for item in self.plan.draw_entries)
        ):
            _fail("loaded bundle typed artifacts no longer share one derivation")


_VERIFIED_LEDGER_EVIDENCE_RESULTS: WeakValueDictionary[
    int, LedgerEvidenceBundleLoadResult
] = WeakValueDictionary()
_VERIFIED_LEDGER_EVIDENCE_RESULT_SEALS: dict[int, str] = {}
_LEDGER_EVIDENCE_LOAD_MINT_TICKETS: WeakKeyDictionary[
    _LedgerEvidenceBundleLoadMintTicket, _LedgerEvidenceBundleLoadMintBinding
] = WeakKeyDictionary()
_VERIFIED_LEDGER_EVIDENCE_RESULTS_LOCK = RLock()


def _load_result_seal(loaded: LedgerEvidenceBundleLoadResult) -> str:
    calibration_run = loaded.calibration.run
    return sha256_bytes(
        canonical_json_bytes(
            {
                "batch_receipt_sha256": loaded.batch_receipt.sha256,
                "calibration_audit": loaded.calibration.audit.to_mapping(),
                "calibration_config_sha256": calibration_run.config.sha256,
                "calibration_input_sha256": calibration_run.observed.sha256,
                "calibration_manifest_raw_sha256": (
                    loaded.calibration.manifest_raw_sha256
                ),
                "calibration_posterior_sha256": calibration_run.posterior.sha256,
                "calibration_protocol_sha256": loaded.calibration_protocol.sha256,
                "calibration_report_sha256": calibration_run.report.sha256,
                "calibration_sbc_sha256": calibration_run.sbc.sha256,
                "caller_commitment_only": loaded.caller_commitment_only,
                "caller_manifest_commitment_verified": (
                    loaded.caller_manifest_commitment_verified
                ),
                "external_custody_verified": loaded.external_custody_verified,
                "externally_registered": loaded.externally_registered,
                "historical_execution_attested": (loaded.historical_execution_attested),
                "independent_reproduction": loaded.independent_reproduction,
                "loaded_sources_replay_only": loaded.loaded_sources_replay_only,
                "manifest_raw_sha256": loaded.manifest_raw_sha256,
                "manifest_sha256": loaded.manifest.sha256,
                "plan_construction_sha256": loaded.plan_construction.sha256,
                "plan_sha256": loaded.plan.sha256,
                "protocol_sha256": loaded.protocol.sha256,
                "result_sha256": loaded.result.sha256,
                "root": str(loaded.root),
                "root_snapshot_fingerprint": list(loaded.root_snapshot_fingerprint),
                "source_artifact_sha256": [
                    item.sha256 for item in loaded.source_artifacts
                ],
            }
        )
    )


def _forget_verified_ledger_evidence_result(identity: int) -> None:
    with _VERIFIED_LEDGER_EVIDENCE_RESULTS_LOCK:
        if _VERIFIED_LEDGER_EVIDENCE_RESULTS.get(identity) is None:
            _VERIFIED_LEDGER_EVIDENCE_RESULT_SEALS.pop(identity, None)


def require_verified_ledger_evidence_bundle_result(
    loaded: LedgerEvidenceBundleLoadResult,
) -> LedgerEvidenceBundleLoadResult:
    """Require a sealed same-process result from the strict snapshot loader."""
    if type(loaded) is not LedgerEvidenceBundleLoadResult:
        _fail("ledger evidence load result has the wrong concrete type")
    try:
        before_seal = _load_result_seal(loaded)
    except Exception as error:
        raise LedgerEvidenceBundleError(
            "ledger evidence load result cannot reproduce its mint-time seal"
        ) from error
    with _VERIFIED_LEDGER_EVIDENCE_RESULTS_LOCK:
        registered = _VERIFIED_LEDGER_EVIDENCE_RESULTS.get(id(loaded))
        mint_seal = _VERIFIED_LEDGER_EVIDENCE_RESULT_SEALS.get(id(loaded))
    if registered is not loaded or mint_seal != before_seal:
        _fail(
            "ledger evidence load result was not minted by the strict loader "
            "or differs from its mint-time seal"
        )
    loaded._fresh_validate()
    try:
        after_seal = _load_result_seal(loaded)
    except Exception as error:
        raise LedgerEvidenceBundleError(
            "ledger evidence load result changed during seal validation"
        ) from error
    with _VERIFIED_LEDGER_EVIDENCE_RESULTS_LOCK:
        still_registered = _VERIFIED_LEDGER_EVIDENCE_RESULTS.get(id(loaded))
        current_seal = _VERIFIED_LEDGER_EVIDENCE_RESULT_SEALS.get(id(loaded))
    if (
        still_registered is not loaded
        or before_seal != after_seal
        or current_seal != after_seal
    ):
        _fail("ledger evidence load result changed after its verifier-owned mint")
    return loaded


def reverify_current_ledger_evidence_bundle(
    loaded: LedgerEvidenceBundleLoadResult,
) -> LedgerEvidenceBundleLoadResult:
    """Reopen the recorded locator and mint a new verified byte-snapshot handle."""
    verified = require_verified_ledger_evidence_bundle_result(loaded)
    return load_ledger_evidence_bundle(
        verified.root,
        expected_manifest_sha256=verified.manifest_raw_sha256,
    )


def _mint_load_result(
    *,
    mint_ticket: _LedgerEvidenceBundleLoadMintTicket,
    root: Path,
    root_snapshot_fingerprint: tuple[int, ...],
    manifest_raw_sha256: str,
    manifest: LedgerEvidenceBundleManifest,
    calibration: CalibrationArtifactLoadResult,
    calibration_protocol: StudyProtocol,
    plan_construction: LedgerPosteriorPlanConstructionSpec,
    plan: LedgerPosteriorStudyPlan,
    protocol: StudyProtocol,
    batch_receipt: ProtocolExecutionReceipt,
    source_artifacts: tuple[ProtocolBoundIntegratedSyntheticRunArtifact, ...],
    result: LedgerPosteriorEvaluation,
) -> LedgerEvidenceBundleLoadResult:
    if type(mint_ticket) is not _LedgerEvidenceBundleLoadMintTicket:
        _fail("ledger evidence result mint requires an exact verifier ticket")
    with _VERIFIED_LEDGER_EVIDENCE_RESULTS_LOCK:
        binding = _LEDGER_EVIDENCE_LOAD_MINT_TICKETS.pop(mint_ticket, None)
    if binding is None:
        _fail("ledger evidence result mint ticket is invalid or already consumed")
    if (
        binding.root != str(root)
        or binding.root_snapshot_fingerprint != root_snapshot_fingerprint
        or binding.manifest_raw_sha256 != manifest_raw_sha256
    ):
        _fail("ledger evidence result fields differ from the mint ticket binding")
    loaded = object.__new__(LedgerEvidenceBundleLoadResult)
    fields: tuple[tuple[str, object], ...] = (
        ("root", root),
        ("root_snapshot_fingerprint", root_snapshot_fingerprint),
        ("manifest_raw_sha256", manifest_raw_sha256),
        ("manifest", manifest),
        ("calibration", calibration),
        ("calibration_protocol", calibration_protocol),
        ("plan_construction", plan_construction),
        ("plan", plan),
        ("protocol", protocol),
        ("batch_receipt", batch_receipt),
        ("source_artifacts", source_artifacts),
        ("result", result),
        ("caller_manifest_commitment_verified", True),
        ("caller_commitment_only", True),
        ("external_custody_verified", False),
        ("externally_registered", False),
        ("independent_reproduction", False),
        ("historical_execution_attested", False),
        ("loaded_sources_replay_only", True),
    )
    for field, value in fields:
        object.__setattr__(loaded, field, value)
    loaded._fresh_validate()
    identity = id(loaded)
    mint_seal = _load_result_seal(loaded)
    with _VERIFIED_LEDGER_EVIDENCE_RESULTS_LOCK:
        _VERIFIED_LEDGER_EVIDENCE_RESULTS[identity] = loaded
        _VERIFIED_LEDGER_EVIDENCE_RESULT_SEALS[identity] = mint_seal
    finalize(loaded, _forget_verified_ledger_evidence_result, identity)
    return loaded


def _validate_calibration_protocol(
    protocol: StudyProtocol, run: CalibrationRun
) -> None:
    try:
        reconstructed = StudyProtocol.from_mapping(protocol.to_mapping())
        validate_protocol_execution_receipt_binding(
            reconstructed, run.protocol_execution_receipt
        )
    except ProtocolValidationError as error:
        raise LedgerEvidenceBundleError(
            f"calibration protocol binding is invalid: {error}"
        ) from error
    receipt = run.protocol_execution_receipt
    calibration_statistics = tuple(
        item.name for item in run.config.summaries.calibration
    )
    heldout_statistics = tuple(item.name for item in run.config.summaries.held_out)
    if (
        reconstructed != protocol
        or receipt.consumer is not ProtocolExecutionConsumer.CALIBRATION_WORKFLOW
        or receipt.purpose is not ProtocolExecutionPurpose.PRE_POLICY_CALIBRATION
        or receipt.config_sha256 != run.config.sha256
        or receipt.input_sha256 != run.observed.sha256
        or protocol.study_id != run.config.study_id
        or protocol.knowledge_cutoff != run.config.cutoff
        or protocol.calibration_statistics != calibration_statistics
        or protocol.heldout_validation_statistics != heldout_statistics
    ):
        _fail("calibration protocol does not bind the complete calibration run")


def _validate_writer_inputs(
    *,
    calibration_run: CalibrationRun,
    calibration_protocol: StudyProtocol,
    plan_construction: LedgerPosteriorPlanConstructionSpec,
    protocol: StudyProtocol,
    evaluation: LedgerPosteriorEvaluation,
    source_artifacts: tuple[ProtocolBoundIntegratedSyntheticRunArtifact, ...],
) -> tuple[
    IntegratedSyntheticMarketConfig,
    LedgerPosteriorStudyPlan,
    tuple[ProtocolBoundIntegratedSyntheticRunArtifact, ...],
    LedgerPosteriorEvaluation,
]:
    exact_values: tuple[tuple[object, type[object], str], ...] = (
        (calibration_run, CalibrationRun, "calibration_run"),
        (calibration_protocol, StudyProtocol, "calibration_protocol"),
        (
            plan_construction,
            LedgerPosteriorPlanConstructionSpec,
            "plan_construction",
        ),
        (protocol, StudyProtocol, "protocol"),
        (evaluation, LedgerPosteriorEvaluation, "evaluation"),
    )
    for value, expected, field in exact_values:
        if type(value) is not expected:
            raise TypeError(f"{field} has the wrong exact type")
    if not isinstance(source_artifacts, tuple) or not source_artifacts:
        raise TypeError("source_artifacts must be a non-empty tuple")
    if not all(
        type(item) is ProtocolBoundIntegratedSyntheticRunArtifact
        for item in source_artifacts
    ):
        raise TypeError("source_artifacts contain a wrong exact type")
    _validate_calibration_protocol(calibration_protocol, calibration_run)
    config = integrated_synthetic_forward_market_config()
    plan = plan_construction.rebuild(
        calibration_run=calibration_run,
        config=config,
    )
    if evaluation.plan.canonical_bytes != plan.canonical_bytes:
        _fail("evaluation does not retain the reconstructed plan")
    batch_receipt = evaluation.batch_protocol_execution_receipt
    try:
        strict_protocol = StudyProtocol.from_mapping(protocol.to_mapping())
        strict_receipt = ProtocolExecutionReceipt.from_mapping(
            batch_receipt.to_mapping()
        )
        fresh = LedgerPosteriorEvaluation.from_mapping(
            evaluation.to_mapping(),
            plan=plan,
            protocol=strict_protocol,
            batch_receipt=strict_receipt,
            source_artifacts=source_artifacts,
        )
    except Exception as error:
        raise LedgerEvidenceBundleError(
            f"ledger evaluation cannot be fully rederived: {error}"
        ) from error
    if strict_protocol != protocol or strict_receipt != batch_receipt:
        _fail("protocol or batch receipt does not survive strict reconstruction")
    if fresh.canonical_bytes != evaluation.canonical_bytes:
        _fail("evaluation differs from full source-bound rederivation")
    by_input = {item.input_sha256: item for item in source_artifacts}
    if len(by_input) != len(source_artifacts):
        _fail("source tuple reuses a pair input")
    try:
        ordered = tuple(
            by_input[entry.pair_input_sha256] for entry in plan.draw_entries
        )
    except KeyError as error:
        raise LedgerEvidenceBundleError(
            "source tuple does not cover every plan draw"
        ) from error
    if len(ordered) != len(source_artifacts):
        _fail("source tuple contains an extra draw")
    return config, plan, ordered, fresh


@dataclass(slots=True)
class _WrittenLeaf:
    parent_fd: int
    name: str
    fd: int
    fingerprint: tuple[int, ...]
    expected_sha256: str
    size_bytes: int

    def close(self) -> None:
        os.close(self.fd)


def _write_new_file_at(parent_fd: int, name: str, content: bytes) -> _WrittenLeaf:
    if (
        not isinstance(name, str)
        or not name
        or name in {".", ".."}
        or "/" in name
        or "\\" in name
    ):
        _fail("bundle leaf name must be one safe path component")
    descriptor: int | None = None
    try:
        descriptor = os.open(
            name,
            os.O_RDWR
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            0o600,
            dir_fd=parent_fd,
        )
        view = memoryview(content)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                _fail(f"cannot complete bundle file write: {name}")
            view = view[written:]
        os.fsync(descriptor)
        written_info = os.fstat(descriptor)
        linked_info = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        if (
            not stat.S_ISREG(written_info.st_mode)
            or written_info.st_nlink != 1
            or written_info.st_size != len(content)
            or _fingerprint(linked_info) != _fingerprint(written_info)
        ):
            _fail(f"new bundle leaf identity changed while written: {name}")
        return _WrittenLeaf(
            parent_fd=parent_fd,
            name=name,
            fd=descriptor,
            fingerprint=_fingerprint(written_info),
            expected_sha256=sha256_bytes(content),
            size_bytes=len(content),
        )
    except OSError as error:
        if descriptor is not None:
            os.close(descriptor)
        raise LedgerEvidenceBundleError(f"cannot create bundle file {name}") from error
    except Exception:
        if descriptor is not None:
            os.close(descriptor)
        raise


def _require_written_leaf_current(leaf: _WrittenLeaf, path: str) -> None:
    try:
        current = os.fstat(leaf.fd)
        linked = os.stat(
            leaf.name,
            dir_fd=leaf.parent_fd,
            follow_symlinks=False,
        )
        payload = _read_exact_file(leaf.fd, leaf.size_bytes, path)
    except OSError as error:
        raise LedgerEvidenceBundleError(
            f"cannot recheck newly written bundle leaf: {path}"
        ) from error
    if (
        _fingerprint(current) != leaf.fingerprint
        or _fingerprint(linked) != leaf.fingerprint
        or current.st_nlink != 1
        or len(payload) != leaf.size_bytes
        or sha256_bytes(payload) != leaf.expected_sha256
    ):
        _fail(f"new bundle leaf changed before publication: {path}")


def write_ledger_evidence_bundle(
    directory: Path,
    *,
    calibration_run: CalibrationRun,
    calibration_protocol: StudyProtocol,
    plan_construction: LedgerPosteriorPlanConstructionSpec,
    protocol: StudyProtocol,
    evaluation: LedgerPosteriorEvaluation,
    source_artifacts: tuple[ProtocolBoundIntegratedSyntheticRunArtifact, ...],
    project_root: Path | None = None,
) -> Path:
    """Write a new descriptor-relative bundle and never overwrite an old one.

    The destination parent must already exist and every ancestor must be a real
    directory rather than a symbolic link.  A crash can leave a partial root,
    but ``bundle-manifest.json`` is written last, so that root is not loadable.
    """
    if not isinstance(directory, Path):
        raise TypeError("directory must be a Path")
    candidate = Path(os.path.abspath(directory.expanduser()))
    if candidate.name in {"", ".", ".."}:
        _fail("bundle destination must name one new directory")
    parent_chain = _open_absolute_directory_chain(candidate.parent)
    root_fd: int | None = None
    calibration_fd: int | None = None
    incomplete_leaf: _WrittenLeaf | None = None
    written_leaves: list[tuple[str, _WrittenLeaf]] = []
    try:
        try:
            os.stat(
                candidate.name,
                dir_fd=parent_chain.final_fd,
                follow_symlinks=False,
            )
        except FileNotFoundError:
            pass
        else:
            _fail("ledger evidence bundles are immutable and cannot be overwritten")

        config, plan, ordered_sources, fresh = _validate_writer_inputs(
            calibration_run=calibration_run,
            calibration_protocol=calibration_protocol,
            plan_construction=plan_construction,
            protocol=protocol,
            evaluation=evaluation,
            source_artifacts=source_artifacts,
        )
        calibration_payloads = build_calibration_artifact_payloads(
            run=calibration_run,
            project_root=project_root,
        )
        if frozenset(calibration_payloads) != _CALIBRATION_FILES:
            _fail("calibration payload builder changed its fixed file set")
        payloads: dict[str, bytes] = {
            "batch-receipt.json": canonical_json_bytes(
                fresh.batch_protocol_execution_receipt.to_mapping()
            ),
            "calibration-protocol.json": canonical_json_bytes(
                calibration_protocol.to_mapping()
            ),
            "evaluation-config.json": canonical_json_bytes(config.to_mapping()),
            "plan-construction.json": plan_construction.canonical_bytes,
            "protocol.json": canonical_json_bytes(protocol.to_mapping()),
            "result.json": fresh.canonical_bytes,
        }
        payloads.update(
            {
                f"source-{ordinal:06d}.json": artifact.canonical_bytes
                for ordinal, artifact in enumerate(ordered_sources)
            }
        )
        leaf_payloads = dict(payloads)
        leaf_payloads.update(
            {
                f"calibration/{name}": content
                for name, content in calibration_payloads.items()
            }
        )
        files = tuple(
            LedgerEvidenceBundleFile(
                path=name,
                sha256=sha256_bytes(content),
                size_bytes=len(content),
            )
            for name, content in sorted(leaf_payloads.items())
        )
        calibration_manifest = calibration_payloads["manifest.json"]
        manifest = LedgerEvidenceBundleManifest(
            files=files,
            source_count=len(ordered_sources),
            calibration_manifest_raw_sha256=sha256_bytes(calibration_manifest),
            calibration_protocol_sha256=calibration_protocol.sha256,
            evaluation_config_sha256=config.sha256,
            plan_construction_sha256=plan_construction.sha256,
            expected_plan_sha256=plan.sha256,
            protocol_sha256=protocol.sha256,
            batch_receipt_sha256=fresh.batch_protocol_execution_receipt.sha256,
            result_sha256=fresh.sha256,
        )
        all_bytes = len(manifest.canonical_bytes) + sum(
            len(content) for content in leaf_payloads.values()
        )
        if all_bytes > _MAX_TOTAL_BYTES:
            _fail("generated bundle exceeds its fixed total byte limit")
        for path, content in leaf_payloads.items():
            if len(content) <= 0 or len(content) > _file_limit(path):
                _fail(f"generated bundle leaf exceeds its byte limit: {path}")
        if len(manifest.canonical_bytes) > _MAX_MANIFEST_BYTES:
            _fail("generated bundle manifest exceeds its byte limit")

        try:
            os.mkdir(candidate.name, 0o700, dir_fd=parent_chain.final_fd)
            root_fd = os.open(
                candidate.name,
                os.O_RDONLY
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NOFOLLOW", 0)
                | getattr(os, "O_DIRECTORY", 0),
                dir_fd=parent_chain.final_fd,
            )
            root_identity = _directory_identity(os.fstat(root_fd))
            os.mkdir("calibration", 0o700, dir_fd=root_fd)
            calibration_fd = os.open(
                "calibration",
                os.O_RDONLY
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NOFOLLOW", 0)
                | getattr(os, "O_DIRECTORY", 0),
                dir_fd=root_fd,
            )
            calibration_identity = _directory_identity(os.fstat(calibration_fd))
            incomplete_leaf = _write_new_file_at(
                root_fd, _INCOMPLETE_MARKER, b"incomplete\n"
            )
            os.fsync(root_fd)
            os.fsync(parent_chain.final_fd)
        except OSError as error:
            raise LedgerEvidenceBundleError(
                "cannot create descriptor-relative ledger evidence bundle"
            ) from error

        for name, content in sorted(calibration_payloads.items()):
            written_leaves.append(
                (
                    f"calibration/{name}",
                    _write_new_file_at(calibration_fd, name, content),
                )
            )
        for name, content in sorted(payloads.items()):
            written_leaves.append((name, _write_new_file_at(root_fd, name, content)))
        written_leaves.append(
            (
                "bundle-manifest.json",
                _write_new_file_at(
                    root_fd, "bundle-manifest.json", manifest.canonical_bytes
                ),
            )
        )
        os.fsync(calibration_fd)
        os.fsync(root_fd)
        _require_directory_chain_current(parent_chain)
        if incomplete_leaf is None:
            _fail("bundle incomplete marker was not retained")
        expected_root_names = frozenset(payloads) | {
            "bundle-manifest.json",
            "calibration",
            _INCOMPLETE_MARKER,
        }
        if frozenset(os.listdir(root_fd)) != expected_root_names:
            _fail("new bundle root membership changed before publication")
        if frozenset(os.listdir(calibration_fd)) != frozenset(calibration_payloads):
            _fail("new calibration membership changed before publication")
        _require_written_leaf_current(incomplete_leaf, _INCOMPLETE_MARKER)
        for path, leaf in written_leaves:
            _require_written_leaf_current(leaf, path)
        linked_root = os.stat(
            candidate.name,
            dir_fd=parent_chain.final_fd,
            follow_symlinks=False,
        )
        linked_calibration = os.stat(
            "calibration", dir_fd=root_fd, follow_symlinks=False
        )
        if (
            _directory_identity(linked_root) != root_identity
            or _directory_identity(os.fstat(root_fd)) != root_identity
            or _directory_identity(linked_calibration) != calibration_identity
            or _directory_identity(os.fstat(calibration_fd)) != calibration_identity
        ):
            _fail("bundle directory identity changed while it was written")
        os.unlink(_INCOMPLETE_MARKER, dir_fd=root_fd)
        try:
            os.fsync(root_fd)
            os.fsync(parent_chain.final_fd)
            _require_directory_chain_current(parent_chain)
            linked_root = os.stat(
                candidate.name,
                dir_fd=parent_chain.final_fd,
                follow_symlinks=False,
            )
            if (
                _directory_identity(linked_root) != root_identity
                or _directory_identity(os.fstat(root_fd)) != root_identity
                or _INCOMPLETE_MARKER in os.listdir(root_fd)
            ):
                _fail("bundle publication identity changed at final commit")
            if frozenset(os.listdir(root_fd)) != (
                expected_root_names - {_INCOMPLETE_MARKER}
            ) or frozenset(os.listdir(calibration_fd)) != frozenset(
                calibration_payloads
            ):
                _fail("bundle membership changed at final commit")
            for path, leaf in written_leaves:
                _require_written_leaf_current(leaf, path)
        except Exception as error:
            restored_marker: _WrittenLeaf | None = None
            try:
                restored_marker = _write_new_file_at(
                    root_fd, _INCOMPLETE_MARKER, b"incomplete\n"
                )
                os.fsync(root_fd)
            except Exception as restore_error:
                raise LedgerEvidenceBundleError(
                    "bundle publication state is indeterminate after commit failure"
                ) from restore_error
            finally:
                if restored_marker is not None:
                    restored_marker.close()
            raise LedgerEvidenceBundleError(
                "bundle final commit failed and was returned to incomplete state"
            ) from error
        return candidate / "bundle-manifest.json"
    finally:
        for _, leaf in written_leaves:
            leaf.close()
        if incomplete_leaf is not None:
            incomplete_leaf.close()
        if calibration_fd is not None:
            os.close(calibration_fd)
        if root_fd is not None:
            os.close(root_fd)
        parent_chain.close()


def _fingerprint(value: os.stat_result) -> tuple[int, ...]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_nlink,
        value.st_uid,
        value.st_gid,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _directory_identity(value: os.stat_result) -> tuple[int, ...]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_uid,
        value.st_gid,
    )


@dataclass(slots=True)
class _DirectoryChainEntry:
    path: Path
    name: str
    fd: int
    fingerprint: tuple[int, ...]


@dataclass(slots=True)
class _DirectoryChain:
    absolute: Path
    entries: tuple[_DirectoryChainEntry, ...]

    @property
    def final_fd(self) -> int:
        return self.entries[-1].fd

    @property
    def final_fingerprint(self) -> tuple[int, ...]:
        return self.entries[-1].fingerprint

    def close(self) -> None:
        for entry in reversed(self.entries):
            os.close(entry.fd)


def _open_absolute_directory_chain(path: Path) -> _DirectoryChain:
    candidate = Path(os.path.abspath(path.expanduser()))
    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_DIRECTORY", 0)
    )
    entries: list[_DirectoryChainEntry] = []
    try:
        anchor_fd = os.open(candidate.anchor, flags)
        anchor_info = os.fstat(anchor_fd)
        if not stat.S_ISDIR(anchor_info.st_mode):
            _fail("absolute path anchor must be a directory")
        current_path = Path(candidate.anchor)
        entries.append(
            _DirectoryChainEntry(
                path=current_path,
                name=candidate.anchor,
                fd=anchor_fd,
                fingerprint=_directory_identity(anchor_info),
            )
        )
        for part in candidate.parts[1:]:
            child = os.open(part, flags, dir_fd=entries[-1].fd)
            child_info = os.fstat(child)
            if not stat.S_ISDIR(child_info.st_mode):
                os.close(child)
                _fail("absolute path component must be a directory")
            current_path = current_path / part
            entries.append(
                _DirectoryChainEntry(
                    path=current_path,
                    name=part,
                    fd=child,
                    fingerprint=_directory_identity(child_info),
                )
            )
        return _DirectoryChain(candidate, tuple(entries))
    except OSError as error:
        for entry in reversed(entries):
            os.close(entry.fd)
        raise LedgerEvidenceBundleError(
            "cannot open directory path without following links"
        ) from error
    except Exception:
        for entry in reversed(entries):
            os.close(entry.fd)
        raise


def _require_directory_chain_current(chain: _DirectoryChain) -> None:
    for index, entry in enumerate(chain.entries):
        try:
            current = os.fstat(entry.fd)
            path_info = os.stat(entry.path, follow_symlinks=False)
            if index == 0:
                linked_info = path_info
            else:
                linked_info = os.stat(
                    entry.name,
                    dir_fd=chain.entries[index - 1].fd,
                    follow_symlinks=False,
                )
        except OSError as error:
            raise LedgerEvidenceBundleError(
                "cannot recheck descriptor-held directory chain"
            ) from error
        if (
            _directory_identity(current) != entry.fingerprint
            or _directory_identity(path_info) != entry.fingerprint
            or _directory_identity(linked_info) != entry.fingerprint
        ):
            _fail("directory ancestor identity changed during bundle operation")


def _file_limit(path: str) -> int:
    if path == "bundle-manifest.json":
        return _MAX_MANIFEST_BYTES
    if path == "plan-construction.json":
        return _MAX_PLAN_BYTES
    if path in {
        "batch-receipt.json",
        "calibration-protocol.json",
        "evaluation-config.json",
        "protocol.json",
    }:
        return _MAX_SMALL_JSON_BYTES
    return _MAX_FILE_BYTES


def _read_exact_file(
    descriptor: int,
    size: int,
    path: str,
    *,
    remaining_bundle_bytes: int | None = None,
) -> bytes:
    limit = _file_limit(path)
    if remaining_bundle_bytes is not None:
        limit = min(limit, remaining_bundle_bytes)
    if size <= 0 or size > limit:
        _fail(f"bundle leaf is empty or exceeds its byte limit: {path}")
    os.lseek(descriptor, 0, os.SEEK_SET)
    chunks: list[bytes] = []
    observed = 0
    while observed <= limit:
        chunk = os.read(descriptor, min(1024 * 1024, limit + 1 - observed))
        if not chunk:
            break
        chunks.append(chunk)
        observed += len(chunk)
    if observed != size or observed > limit:
        _fail(f"bundle leaf changed or exceeds its byte limit: {path}")
    return b"".join(chunks)


@dataclass(slots=True)
class _CapturedLeaf:
    path: str
    name: str
    parent_fd: int
    fd: int
    payload: bytes | None
    expected_sha256: str
    size_bytes: int
    fingerprint: tuple[int, ...]


@dataclass(slots=True)
class _CapturedTree:
    root: Path
    root_chain: _DirectoryChain
    root_fingerprint: tuple[int, ...]
    root_names: frozenset[str]
    calibration_fd: int
    calibration_fingerprint: tuple[int, ...]
    leaves: dict[str, _CapturedLeaf]
    source_count: int

    @property
    def root_fd(self) -> int:
        return self.root_chain.final_fd

    def close(self) -> None:
        for leaf in self.leaves.values():
            os.close(leaf.fd)
        os.close(self.calibration_fd)
        self.root_chain.close()


def _capture_leaf(
    parent_fd: int,
    name: str,
    path: str,
    *,
    remaining_bundle_bytes: int,
    expected_size: int | None = None,
    expected_sha256: str,
    retain_payload: bool,
) -> _CapturedLeaf:
    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_NONBLOCK", 0)
    )
    descriptor: int | None = None
    try:
        descriptor = os.open(name, flags, dir_fd=parent_fd)
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            _fail(f"bundle leaf must be a singly linked regular file: {path}")
        if expected_size is not None and before.st_size != expected_size:
            _fail(f"bundle leaf size differs from the root manifest: {path}")
        if (
            before.st_size <= 0
            or before.st_size > _file_limit(path)
            or before.st_size > remaining_bundle_bytes
        ):
            _fail(f"bundle leaf is empty or exceeds its byte limit: {path}")
        fingerprint = _fingerprint(before)
        payload = (
            _read_exact_file(
                descriptor,
                before.st_size,
                path,
                remaining_bundle_bytes=remaining_bundle_bytes,
            )
            if retain_payload
            else None
        )
        if payload is not None and sha256_bytes(payload) != expected_sha256:
            if path == "bundle-manifest.json":
                _fail("bundle manifest bytes differ from the caller commitment")
            _fail(f"bundle leaf digest differs from its commitment: {path}")
        linked = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        if (
            _fingerprint(os.fstat(descriptor)) != fingerprint
            or _fingerprint(linked) != fingerprint
        ):
            _fail(f"bundle leaf changed while captured: {path}")
        return _CapturedLeaf(
            path=path,
            name=name,
            parent_fd=parent_fd,
            fd=descriptor,
            payload=payload,
            expected_sha256=expected_sha256,
            size_bytes=before.st_size,
            fingerprint=fingerprint,
        )
    except OSError as error:
        if descriptor is not None:
            os.close(descriptor)
        raise LedgerEvidenceBundleError(
            f"bundle leaf must be a singly linked regular file: {path}"
        ) from error
    except Exception:
        if descriptor is not None:
            os.close(descriptor)
        raise


def _capture_tree(
    directory: Path,
    *,
    expected_manifest_sha256: str,
) -> _CapturedTree:
    if not isinstance(directory, Path):
        raise TypeError("directory must be a Path")
    root_chain = _open_absolute_directory_chain(directory)
    root = root_chain.absolute
    root_fd = root_chain.final_fd
    calibration_fd: int | None = None
    leaves: dict[str, _CapturedLeaf] = {}
    try:
        root_info = os.fstat(root_fd)
        root_names = frozenset(os.listdir(root_fd))
        if _INCOMPLETE_MARKER in root_names:
            _fail("bundle publication is incomplete and cannot be loaded")
        source_ordinals: list[int] = []
        for name in root_names:
            match = _SOURCE_NAME.fullmatch(name)
            if match is not None:
                source_ordinals.append(int(match.group(1)))
        source_ordinals.sort()
        if (
            not source_ordinals
            or len(source_ordinals) > _MAX_SOURCE_COUNT
            or source_ordinals != list(range(len(source_ordinals)))
        ):
            _fail("bundle source files are empty, excessive, or non-contiguous")
        expected_root = (
            _ROOT_FIXED_FILES
            | {"calibration"}
            | {f"source-{ordinal:06d}.json" for ordinal in source_ordinals}
        )
        if root_names != expected_root:
            _fail(
                "bundle root file set changed: "
                f"missing={sorted(expected_root - root_names)!r}, "
                f"extra={sorted(root_names - expected_root)!r}"
            )
        manifest_leaf = _capture_leaf(
            root_fd,
            "bundle-manifest.json",
            "bundle-manifest.json",
            remaining_bundle_bytes=_MAX_MANIFEST_BYTES,
            expected_sha256=expected_manifest_sha256,
            retain_payload=True,
        )
        leaves["bundle-manifest.json"] = manifest_leaf
        manifest_payload = manifest_leaf.payload
        if manifest_payload is None:
            _fail("bundle manifest capture did not retain its bytes")
        manifest_raw_sha256 = sha256_bytes(manifest_payload)
        if manifest_raw_sha256 != expected_manifest_sha256:
            _fail("bundle manifest bytes differ from the caller commitment")
        manifest_mapping = _canonical_object(manifest_payload, "bundle-manifest.json")
        manifest = LedgerEvidenceBundleManifest.from_mapping(manifest_mapping)
        if manifest.canonical_bytes != manifest_payload:
            _fail("bundle manifest does not survive strict reconstruction")
        declared_entries = {item.path: item for item in manifest.files}
        declared_paths = (root_names - {"bundle-manifest.json", "calibration"}) | {
            f"calibration/{name}" for name in _CALIBRATION_FILES
        }
        if set(declared_entries) != declared_paths:
            _fail("bundle manifest does not describe the fixed root membership")
        if manifest.source_count != len(source_ordinals):
            _fail("bundle manifest source count differs from root membership")
        declared_total = len(manifest_payload) + sum(
            item.size_bytes for item in manifest.files
        )
        if declared_total > _MAX_TOTAL_BYTES:
            _fail("bundle manifest declares excessive total bytes")
        for path, entry in declared_entries.items():
            if entry.size_bytes > _file_limit(path):
                _fail(
                    "bundle manifest declares a leaf that exceeds its byte limit: "
                    f"{path}"
                )
        directory_flags = (
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_DIRECTORY", 0)
        )
        calibration_fd = os.open("calibration", directory_flags, dir_fd=root_fd)
        calibration_info = os.fstat(calibration_fd)
        if not stat.S_ISDIR(calibration_info.st_mode):
            _fail("calibration bundle member must be a directory")
        calibration_names = frozenset(os.listdir(calibration_fd))
        if calibration_names != _CALIBRATION_FILES:
            _fail("nested calibration bundle file set changed")
        remaining_bytes = _MAX_TOTAL_BYTES - len(manifest_payload)
        for name in sorted(root_names - {"bundle-manifest.json", "calibration"}):
            entry = declared_entries[name]
            leaf = _capture_leaf(
                root_fd,
                name,
                name,
                remaining_bundle_bytes=remaining_bytes,
                expected_size=entry.size_bytes,
                expected_sha256=entry.sha256,
                retain_payload=False,
            )
            leaves[name] = leaf
            remaining_bytes -= leaf.size_bytes
        for name in sorted(calibration_names):
            path = f"calibration/{name}"
            entry = declared_entries[path]
            leaf = _capture_leaf(
                calibration_fd,
                name,
                path,
                remaining_bundle_bytes=remaining_bytes,
                expected_size=entry.size_bytes,
                expected_sha256=entry.sha256,
                retain_payload=False,
            )
            leaves[path] = leaf
            remaining_bytes -= leaf.size_bytes
        return _CapturedTree(
            root=root,
            root_chain=root_chain,
            root_fingerprint=_fingerprint(root_info),
            root_names=root_names,
            calibration_fd=calibration_fd,
            calibration_fingerprint=_fingerprint(calibration_info),
            leaves=leaves,
            source_count=len(source_ordinals),
        )
    except OSError as error:
        for leaf in leaves.values():
            os.close(leaf.fd)
        if calibration_fd is not None:
            os.close(calibration_fd)
        root_chain.close()
        raise LedgerEvidenceBundleError(
            "cannot capture fixed bundle directory layout"
        ) from error
    except Exception:
        for leaf in leaves.values():
            os.close(leaf.fd)
        if calibration_fd is not None:
            os.close(calibration_fd)
        root_chain.close()
        raise


def _read_captured_leaf(captured: _CapturedTree, path: str) -> bytes:
    leaf = captured.leaves[path]
    try:
        linked = os.stat(leaf.name, dir_fd=leaf.parent_fd, follow_symlinks=False)
        current = os.fstat(leaf.fd)
        payload = (
            leaf.payload
            if leaf.payload is not None
            else _read_exact_file(leaf.fd, leaf.size_bytes, path)
        )
    except OSError as error:
        raise LedgerEvidenceBundleError(
            f"cannot read captured bundle leaf: {path}"
        ) from error
    if (
        _fingerprint(linked) != leaf.fingerprint
        or _fingerprint(current) != leaf.fingerprint
        or current.st_size != leaf.size_bytes
        or len(payload) != leaf.size_bytes
        or sha256_bytes(payload) != leaf.expected_sha256
    ):
        _fail(
            "bundle leaf changed during verification or differs from its "
            f"commitment: {path}"
        )
    return payload


def _require_current_tree(captured: _CapturedTree) -> None:
    _require_directory_chain_current(captured.root_chain)
    if frozenset(os.listdir(captured.root_fd)) != captured.root_names:
        _fail("bundle root membership changed during verification")
    if frozenset(os.listdir(captured.calibration_fd)) != _CALIBRATION_FILES:
        _fail("calibration membership changed during verification")
    for path in captured.leaves:
        _read_captured_leaf(captured, path)
    try:
        calibration_path_info = os.stat(
            "calibration", dir_fd=captured.root_fd, follow_symlinks=False
        )
    except OSError as error:
        raise LedgerEvidenceBundleError(
            "cannot finalize bundle directory identity"
        ) from error
    if (
        _fingerprint(os.fstat(captured.root_fd)) != captured.root_fingerprint
        or _fingerprint(calibration_path_info) != captured.calibration_fingerprint
        or _fingerprint(os.fstat(captured.calibration_fd))
        != captured.calibration_fingerprint
    ):
        _fail("bundle directory identity changed during verification")


def _canonical_object(content: bytes, source: str) -> dict[str, object]:
    try:
        value = read_json_bytes_object(content, source=source)
    except MetadataValidationError as error:
        raise LedgerEvidenceBundleError(str(error)) from error
    if canonical_json_bytes(value) != content:
        _fail(f"{source} must use canonical duplicate-free JSON")
    return value


def _parse_protocol(content: bytes, source: str) -> StudyProtocol:
    mapping = _canonical_object(content, source)
    try:
        protocol = StudyProtocol.from_mapping(mapping)
    except ProtocolValidationError as error:
        raise LedgerEvidenceBundleError(f"invalid {source}: {error}") from error
    if canonical_json_bytes(protocol.to_mapping()) != content:
        _fail(f"{source} does not survive strict reconstruction")
    return protocol


def _parse_receipt(content: bytes) -> ProtocolExecutionReceipt:
    mapping = _canonical_object(content, "batch-receipt.json")
    try:
        receipt = ProtocolExecutionReceipt.from_mapping(mapping)
    except ProtocolValidationError as error:
        raise LedgerEvidenceBundleError(f"invalid batch receipt: {error}") from error
    if canonical_json_bytes(receipt.to_mapping()) != content:
        _fail("batch receipt does not survive strict reconstruction")
    return receipt


def _verify_manifest_closure(
    captured: _CapturedTree,
    manifest: LedgerEvidenceBundleManifest,
) -> None:
    expected_paths = set(captured.leaves) - {"bundle-manifest.json"}
    entries = {item.path: item for item in manifest.files}
    if set(entries) != expected_paths:
        _fail("bundle manifest does not close over the exact captured leaf set")
    for path, entry in entries.items():
        leaf = captured.leaves[path]
        if entry.size_bytes != leaf.size_bytes or entry.sha256 != leaf.expected_sha256:
            _fail(f"bundle manifest hash or size mismatch: {path}")
    if manifest.source_count != captured.source_count:
        _fail("bundle manifest source count differs from root membership")


def load_ledger_evidence_bundle(
    directory: Path,
    *,
    expected_manifest_sha256: str,
) -> LedgerEvidenceBundleLoadResult:
    """Capture and replay a rooted bundle using only its path and caller digest."""
    expected = _sha256(expected_manifest_sha256, "expected_manifest_sha256")
    captured = _capture_tree(
        directory,
        expected_manifest_sha256=expected,
    )
    try:
        manifest_bytes = _read_captured_leaf(captured, "bundle-manifest.json")
        manifest_raw_sha256 = sha256_bytes(manifest_bytes)
        manifest_mapping = _canonical_object(manifest_bytes, "bundle-manifest.json")
        manifest = LedgerEvidenceBundleManifest.from_mapping(manifest_mapping)
        if manifest.canonical_bytes != manifest_bytes:
            _fail("bundle manifest does not survive strict reconstruction")
        _verify_manifest_closure(captured, manifest)

        config = integrated_synthetic_forward_market_config()
        config_bytes = _read_captured_leaf(captured, "evaluation-config.json")
        if (
            config_bytes != canonical_json_bytes(config.to_mapping())
            or manifest.evaluation_config_sha256 != config.sha256
        ):
            _fail("evaluation config differs from the allowlisted frozen factory")
        calibration_manifest_bytes = _read_captured_leaf(
            captured, "calibration/manifest.json"
        )
        if (
            sha256_bytes(calibration_manifest_bytes)
            != manifest.calibration_manifest_raw_sha256
        ):
            _fail("nested calibration manifest commitment differs")
        try:
            calibration = verify_calibration_artifact_snapshot(
                {
                    name.removeprefix("calibration/"): _read_captured_leaf(
                        captured, name
                    )
                    for name in captured.leaves
                    if name.startswith("calibration/")
                },
                expected_manifest_sha256=manifest.calibration_manifest_raw_sha256,
            )
        except Exception as error:
            raise LedgerEvidenceBundleError(
                f"calibration artifact replay failed: {error}"
            ) from error

        calibration_protocol = _parse_protocol(
            _read_captured_leaf(captured, "calibration-protocol.json"),
            "calibration-protocol.json",
        )
        if calibration_protocol.sha256 != manifest.calibration_protocol_sha256:
            _fail("calibration protocol hash differs from the root manifest")
        _validate_calibration_protocol(calibration_protocol, calibration.run)

        construction_bytes = _read_captured_leaf(captured, "plan-construction.json")
        if sha256_bytes(construction_bytes) != manifest.plan_construction_sha256:
            _fail("plan construction hash differs from the root manifest")
        try:
            construction = verify_ledger_posterior_plan_construction_bytes(
                construction_bytes,
                expected_sha256=manifest.plan_construction_sha256,
                calibration_run=calibration.run,
                config=config,
            )
            plan = construction.rebuild(
                calibration_run=calibration.run,
                config=config,
            )
        except Exception as error:
            raise LedgerEvidenceBundleError(
                f"plan construction replay failed: {error}"
            ) from error
        if plan.sha256 != manifest.expected_plan_sha256:
            _fail("reconstructed plan differs from the root manifest")
        if not (
            captured.source_count
            == manifest.source_count
            == plan.fixed_draw_count
            == len(plan.draw_entries)
            == len(plan.pairs)
        ):
            _fail("root, manifest, and reconstructed plan source counts differ")

        protocol = _parse_protocol(
            _read_captured_leaf(captured, "protocol.json"), "protocol.json"
        )
        if protocol.sha256 != manifest.protocol_sha256:
            _fail("evaluation protocol hash differs from the root manifest")
        batch_receipt = _parse_receipt(
            _read_captured_leaf(captured, "batch-receipt.json")
        )
        if batch_receipt.sha256 != manifest.batch_receipt_sha256:
            _fail("batch receipt hash differs from the root manifest")
        try:
            validate_protocol_execution_receipt_binding(protocol, batch_receipt)
        except ProtocolValidationError as error:
            raise LedgerEvidenceBundleError(
                f"batch receipt protocol binding failed: {error}"
            ) from error

        loaded_sources: list[ProtocolBoundIntegratedSyntheticRunArtifact] = []
        manifest_entries = {item.path: item for item in manifest.files}
        for ordinal, (entry, pair) in enumerate(
            zip(plan.draw_entries, plan.pairs, strict=True)
        ):
            path = f"source-{ordinal:06d}.json"
            source_entry = manifest_entries[path]
            try:
                source = verify_protocol_bound_integrated_synthetic_run_artifact_bytes(
                    _read_captured_leaf(captured, path),
                    expected_sha256=source_entry.sha256,
                    expected_protocol=protocol,
                    expected_config=config,
                    expected_pair=pair,
                )
            except Exception as error:
                raise LedgerEvidenceBundleError(
                    f"historical source replay failed for draw {ordinal}: {error}"
                ) from error
            if source.input_sha256 != entry.pair_input_sha256:
                _fail("loaded source differs from its reconstructed plan slot")
            loaded_sources.append(source)
        sources = tuple(loaded_sources)
        result_entry = manifest_entries["result.json"]
        try:
            result = verify_ledger_posterior_evaluation_bytes(
                _read_captured_leaf(captured, "result.json"),
                expected_sha256=result_entry.sha256,
                plan=plan,
                protocol=protocol,
                batch_receipt=batch_receipt,
                source_artifacts=sources,
            )
        except Exception as error:
            raise LedgerEvidenceBundleError(
                f"ledger result rederivation failed: {error}"
            ) from error
        if result.sha256 != manifest.result_sha256:
            _fail("rederived result differs from the root manifest")
        _require_current_tree(captured)
        mint_ticket = object.__new__(_LedgerEvidenceBundleLoadMintTicket)
        mint_binding = _LedgerEvidenceBundleLoadMintBinding(
            root=str(captured.root),
            root_snapshot_fingerprint=captured.root_fingerprint,
            manifest_raw_sha256=manifest_raw_sha256,
        )
        with _VERIFIED_LEDGER_EVIDENCE_RESULTS_LOCK:
            _LEDGER_EVIDENCE_LOAD_MINT_TICKETS[mint_ticket] = mint_binding
        loaded = _mint_load_result(
            mint_ticket=mint_ticket,
            root=captured.root,
            root_snapshot_fingerprint=captured.root_fingerprint,
            manifest_raw_sha256=manifest_raw_sha256,
            manifest=manifest,
            calibration=calibration,
            calibration_protocol=calibration_protocol,
            plan_construction=construction,
            plan=plan,
            protocol=protocol,
            batch_receipt=batch_receipt,
            source_artifacts=sources,
            result=result,
        )
        _require_current_tree(captured)
        return require_verified_ledger_evidence_bundle_result(loaded)
    finally:
        captured.close()


__all__ = [
    "LEDGER_EVIDENCE_BUNDLE_CLAIM_SCOPE",
    "LEDGER_EVIDENCE_BUNDLE_SCHEMA_VERSION",
    "LedgerEvidenceBundleError",
    "LedgerEvidenceBundleFile",
    "LedgerEvidenceBundleLoadResult",
    "LedgerEvidenceBundleManifest",
    "load_ledger_evidence_bundle",
    "require_verified_ledger_evidence_bundle_result",
    "reverify_current_ledger_evidence_bundle",
    "write_ledger_evidence_bundle",
]
