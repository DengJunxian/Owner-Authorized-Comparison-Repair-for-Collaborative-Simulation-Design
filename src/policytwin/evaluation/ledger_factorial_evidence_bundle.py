"""Fixed-layout rooted bundle for synthetic factorial ledger evidence.

The caller-supplied root-manifest digest commits to one captured byte tree.
Loading deterministically replays calibration, plan construction, every source,
the complete source batch, and the source-bound factorial result.  It does not
establish external custody, historical execution, preregistration, independent
reproduction, or any real-world causal or interval claim.
"""

from __future__ import annotations

import os
import re
import stat
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from threading import RLock
from typing import Literal, NoReturn, Self, cast
from weakref import WeakKeyDictionary, WeakValueDictionary, finalize

from policytwin.calibration.artifacts import (
    CalibrationArtifactAuditReport,
    CalibrationArtifactLoadResult,
    build_calibration_artifact_payloads,
    verify_calibration_artifact_snapshot,
)
from policytwin.calibration.workflow import CalibrationRun
from policytwin.evaluation.ledger_evidence_bundle import (
    LedgerEvidenceBundleError,
    _capture_leaf,
    _CapturedLeaf,
    _directory_identity,
    _DirectoryChain,
    _fingerprint,
    _open_absolute_directory_chain,
    _read_exact_file,
    _require_directory_chain_current,
    _require_written_leaf_current,
    _validate_calibration_protocol,
    _write_new_file_at,
    _WrittenLeaf,
)
from policytwin.evaluation.ledger_factorial_bundle import (
    LedgerFactorialPlanConstructionSpec,
    verify_ledger_factorial_plan_construction_bytes,
)
from policytwin.evaluation.ledger_factorial_study import (
    LEDGER_FACTORIAL_MAX_SOURCE_COUNT,
    LedgerFactorialSourceBatch,
    LedgerFactorialStudyPlan,
    SourceBoundLedgerFactorialEvaluation,
    derive_source_bound_ledger_factorial_evaluation,
    require_executed_ledger_factorial_source_batch,
    require_executed_source_bound_ledger_factorial_evaluation,
    require_ledger_factorial_derivation_source_batch,
    require_source_bound_ledger_factorial_evaluation,
    verify_ledger_factorial_source_batch_bytes,
    verify_source_bound_ledger_factorial_evaluation_bytes,
)
from policytwin.experiments.integrated_synthetic import (
    IntegratedProtocolExecutionError,
    ProtocolBoundIntegratedSyntheticRunArtifact,
    integrated_synthetic_forward_market_config,
    require_ledger_derivation_source_artifact,
    require_verifier_owned_protocol_bound_integrated_artifact,
    verify_protocol_bound_integrated_synthetic_run_artifact_bytes,
)
from policytwin.protocol import (
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

LEDGER_FACTORIAL_EVIDENCE_BUNDLE_SCHEMA_VERSION = (
    "policytwin.ledger-factorial-evidence-rooted-bundle.v1"
)
LEDGER_FACTORIAL_EVIDENCE_BUNDLE_CLAIM_SCOPE: Literal[
    "synthetic_factorial_rooted_replay_mechanics_only"
] = "synthetic_factorial_rooted_replay_mechanics_only"

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SOURCE_NAME = re.compile(r"^source-([0-9]{6})\.json$")
_MAX_TOTAL_BYTES = 64 * 1024 * 1024
_MAX_MANIFEST_BYTES = 2 * 1024 * 1024
_MAX_FILE_BYTES = 32 * 1024 * 1024
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
        "source-batch.json",
    }
)


class LedgerFactorialEvidenceBundleError(RuntimeError):
    """Raised when a factorial rooted bundle is incomplete or inconsistent."""


def _fail(message: str) -> NoReturn:
    raise LedgerFactorialEvidenceBundleError(message)


def _sha256(value: object, field: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        _fail(f"{field} must be a lowercase SHA-256 digest")
    return value


def _positive_int(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        _fail(f"{field} must be a positive integer")
    return value


def _at_least_two(value: object, field: str) -> int:
    result = _positive_int(value, field)
    if result < 2:
        _fail(f"{field} must be at least two")
    return result


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


def _source_manifest_sha256(
    sources: Sequence[ProtocolBoundIntegratedSyntheticRunArtifact],
) -> str:
    return sha256_bytes(canonical_json_bytes([item.sha256 for item in sources]))


def _grid_sha256(result: SourceBoundLedgerFactorialEvaluation) -> str:
    return sha256_bytes(
        canonical_json_bytes([item.to_mapping() for item in result.grid])
    )


def _validate_loaded_calibration(
    calibration: CalibrationArtifactLoadResult,
) -> None:
    if type(calibration) is not CalibrationArtifactLoadResult:
        _fail("loaded calibration has the wrong exact type")
    if type(calibration.run) is not CalibrationRun:
        _fail("loaded calibration run has the wrong exact type")
    if type(calibration.audit) is not CalibrationArtifactAuditReport:
        _fail("loaded calibration audit has the wrong exact type")
    run = calibration.run
    expected_event_keys_sha256 = sha256_bytes(
        canonical_json_bytes(list(run.event_keys))
    )
    expected_audit: dict[str, object] = {
        "checked_files": sorted(_CALIBRATION_FILES),
        "config_sha256": run.config.sha256,
        "event_key_count": len(run.event_keys),
        "event_keys_sha256": expected_event_keys_sha256,
        # The calibration loader has already checked this value against the
        # closed provenance payload and its finite attestation vocabulary.  It
        # is part of CalibrationArtifactAuditReport.to_mapping(), so omitting
        # it here makes every current, otherwise valid bundle fail equality.
        "git_identity_attestation": calibration.audit.git_identity_attestation,
        "input_sha256": run.observed.sha256,
        "protocol_execution_receipt_sha256": (run.protocol_execution_receipt.sha256),
        "report_sha256": run.report.sha256,
        "simulator_config_sha256": sha256_bytes(run.simulator_config_json),
        "study_protocol_sha256": run.protocol_execution_receipt.protocol_sha256,
        "true_parameters_sha256": run.true_parameters_sha256,
        "valid": True,
    }
    if calibration.audit.to_mapping() != expected_audit:
        _fail("loaded calibration audit differs from its complete run closure")


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


@dataclass(frozen=True, slots=True)
class LedgerFactorialEvidenceBundleFile:
    """Hash-and-size commitment for one non-manifest leaf."""

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
            _fail("factorial bundle path must be normalized relative POSIX text")
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
class LedgerFactorialEvidenceBundleManifest:
    """Closed commitment for one complete ``M x R`` rooted replay tree."""

    files: tuple[LedgerFactorialEvidenceBundleFile, ...]
    posterior_draw_count: int
    scenario_replicate_count: int
    source_count: int
    calibration_manifest_raw_sha256: str
    calibration_protocol_sha256: str
    evaluation_config_sha256: str
    plan_construction_sha256: str
    expected_plan_sha256: str
    expected_slot_manifest_sha256: str
    expected_output_manifest_sha256: str
    expected_analysis_spec_sha256: str
    protocol_sha256: str
    batch_receipt_sha256: str
    source_batch_sha256: str
    source_artifact_manifest_sha256: str
    grid_sha256: str
    analysis_sha256: str
    result_sha256: str
    claim_scope: Literal["synthetic_factorial_rooted_replay_mechanics_only"] = (
        LEDGER_FACTORIAL_EVIDENCE_BUNDLE_CLAIM_SCOPE
    )
    synthetic_only: Literal[True] = True
    caller_commitment_only: Literal[True] = True
    loaded_sources_replay_only: Literal[True] = True
    loaded_source_batch_replay_only: Literal[True] = True
    external_custody_verified: Literal[False] = False
    externally_registered: Literal[False] = False
    historical_execution_attested: Literal[False] = False
    independent_reproduction: Literal[False] = False
    analysis_chronology_externally_verified: Literal[False] = False
    source_execution_identity_externally_attested: Literal[False] = False
    real_world_uncertainty_components_identified: Literal[False] = False
    real_world_causal_claim: Literal[False] = False
    credible_interval_claim: Literal[False] = False
    confidence_interval_claim: Literal[False] = False
    iid_mcse_reported: Literal[False] = False
    publication_eligible: Literal[False] = False
    schema_version: str = LEDGER_FACTORIAL_EVIDENCE_BUNDLE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if (
            not isinstance(self.files, tuple)
            or not self.files
            or not all(
                type(item) is LedgerFactorialEvidenceBundleFile for item in self.files
            )
        ):
            _fail("factorial manifest files must be a non-empty typed tuple")
        paths = tuple(item.path for item in self.files)
        if paths != tuple(sorted(paths)) or len(paths) != len(set(paths)):
            _fail("factorial manifest files must use unique canonical path order")
        posterior_count = _at_least_two(
            self.posterior_draw_count, "posterior_draw_count"
        )
        scenario_count = _at_least_two(
            self.scenario_replicate_count, "scenario_replicate_count"
        )
        count = _positive_int(self.source_count, "source_count")
        if count != posterior_count * scenario_count:
            _fail("source_count must equal posterior_draw_count times scenario count")
        if count > LEDGER_FACTORIAL_MAX_SOURCE_COUNT:
            _fail("factorial bundle source_count exceeds the shared fixed bound")
        expected_sources = {f"source-{ordinal:06d}.json" for ordinal in range(count)}
        observed_sources = {path for path in paths if _SOURCE_NAME.fullmatch(path)}
        if observed_sources != expected_sources:
            _fail("factorial manifest source paths are incomplete or non-canonical")
        expected_paths = (
            (_ROOT_FIXED_FILES - {"bundle-manifest.json"})
            | {f"calibration/{name}" for name in _CALIBRATION_FILES}
            | expected_sources
        )
        if set(paths) != expected_paths or len(paths) != len(expected_paths):
            _fail("factorial manifest files do not close over the fixed layout")
        for field in (
            "calibration_manifest_raw_sha256",
            "calibration_protocol_sha256",
            "evaluation_config_sha256",
            "plan_construction_sha256",
            "expected_plan_sha256",
            "expected_slot_manifest_sha256",
            "expected_output_manifest_sha256",
            "expected_analysis_spec_sha256",
            "protocol_sha256",
            "batch_receipt_sha256",
            "source_batch_sha256",
            "source_artifact_manifest_sha256",
            "grid_sha256",
            "analysis_sha256",
            "result_sha256",
        ):
            _sha256(getattr(self, field), field)
        if (
            self.claim_scope != LEDGER_FACTORIAL_EVIDENCE_BUNDLE_CLAIM_SCOPE
            or self.synthetic_only is not True
            or self.caller_commitment_only is not True
            or self.loaded_sources_replay_only is not True
            or self.loaded_source_batch_replay_only is not True
            or self.schema_version != LEDGER_FACTORIAL_EVIDENCE_BUNDLE_SCHEMA_VERSION
            or any(
                value is not False
                for value in (
                    self.external_custody_verified,
                    self.externally_registered,
                    self.historical_execution_attested,
                    self.independent_reproduction,
                    self.analysis_chronology_externally_verified,
                    self.source_execution_identity_externally_attested,
                    self.real_world_uncertainty_components_identified,
                    self.real_world_causal_claim,
                    self.credible_interval_claim,
                    self.confidence_interval_claim,
                    self.iid_mcse_reported,
                    self.publication_eligible,
                )
            )
        ):
            _fail("factorial manifest schema or scientific claim boundary changed")

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> Self:
        expected = frozenset(
            {
                "analysis_chronology_externally_verified",
                "analysis_sha256",
                "batch_receipt_sha256",
                "calibration_manifest_raw_sha256",
                "calibration_protocol_sha256",
                "caller_commitment_only",
                "claim_scope",
                "confidence_interval_claim",
                "credible_interval_claim",
                "evaluation_config_sha256",
                "expected_analysis_spec_sha256",
                "expected_output_manifest_sha256",
                "expected_plan_sha256",
                "expected_slot_manifest_sha256",
                "external_custody_verified",
                "externally_registered",
                "files",
                "grid_sha256",
                "historical_execution_attested",
                "iid_mcse_reported",
                "independent_reproduction",
                "loaded_source_batch_replay_only",
                "loaded_sources_replay_only",
                "plan_construction_sha256",
                "posterior_draw_count",
                "protocol_sha256",
                "publication_eligible",
                "real_world_causal_claim",
                "real_world_uncertainty_components_identified",
                "result_sha256",
                "scenario_replicate_count",
                "schema_version",
                "source_artifact_manifest_sha256",
                "source_batch_sha256",
                "source_count",
                "source_execution_identity_externally_attested",
                "synthetic_only",
            }
        )
        _strict_fields(value, expected, "factorial bundle manifest")
        raw_files = value["files"]
        if not isinstance(raw_files, list):
            _fail("factorial manifest files must be a list")
        files: list[LedgerFactorialEvidenceBundleFile] = []
        for raw_file in raw_files:
            if not isinstance(raw_file, Mapping):
                _fail("factorial manifest file entry must be an object")
            files.append(LedgerFactorialEvidenceBundleFile.from_mapping(raw_file))
        return cls(
            files=tuple(files),
            posterior_draw_count=_at_least_two(
                value["posterior_draw_count"], "posterior_draw_count"
            ),
            scenario_replicate_count=_at_least_two(
                value["scenario_replicate_count"], "scenario_replicate_count"
            ),
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
            expected_slot_manifest_sha256=_sha256(
                value["expected_slot_manifest_sha256"],
                "expected_slot_manifest_sha256",
            ),
            expected_output_manifest_sha256=_sha256(
                value["expected_output_manifest_sha256"],
                "expected_output_manifest_sha256",
            ),
            expected_analysis_spec_sha256=_sha256(
                value["expected_analysis_spec_sha256"],
                "expected_analysis_spec_sha256",
            ),
            protocol_sha256=_sha256(value["protocol_sha256"], "protocol_sha256"),
            batch_receipt_sha256=_sha256(
                value["batch_receipt_sha256"], "batch_receipt_sha256"
            ),
            source_batch_sha256=_sha256(
                value["source_batch_sha256"], "source_batch_sha256"
            ),
            source_artifact_manifest_sha256=_sha256(
                value["source_artifact_manifest_sha256"],
                "source_artifact_manifest_sha256",
            ),
            grid_sha256=_sha256(value["grid_sha256"], "grid_sha256"),
            analysis_sha256=_sha256(value["analysis_sha256"], "analysis_sha256"),
            result_sha256=_sha256(value["result_sha256"], "result_sha256"),
            claim_scope=cast(
                Literal["synthetic_factorial_rooted_replay_mechanics_only"],
                value["claim_scope"],
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
            loaded_sources_replay_only=cast(
                Literal[True],
                _exact_boolean(
                    value["loaded_sources_replay_only"],
                    True,
                    "loaded_sources_replay_only",
                ),
            ),
            loaded_source_batch_replay_only=cast(
                Literal[True],
                _exact_boolean(
                    value["loaded_source_batch_replay_only"],
                    True,
                    "loaded_source_batch_replay_only",
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
            historical_execution_attested=cast(
                Literal[False],
                _exact_boolean(
                    value["historical_execution_attested"],
                    False,
                    "historical_execution_attested",
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
            analysis_chronology_externally_verified=cast(
                Literal[False],
                _exact_boolean(
                    value["analysis_chronology_externally_verified"],
                    False,
                    "analysis_chronology_externally_verified",
                ),
            ),
            source_execution_identity_externally_attested=cast(
                Literal[False],
                _exact_boolean(
                    value["source_execution_identity_externally_attested"],
                    False,
                    "source_execution_identity_externally_attested",
                ),
            ),
            real_world_uncertainty_components_identified=cast(
                Literal[False],
                _exact_boolean(
                    value["real_world_uncertainty_components_identified"],
                    False,
                    "real_world_uncertainty_components_identified",
                ),
            ),
            real_world_causal_claim=cast(
                Literal[False],
                _exact_boolean(
                    value["real_world_causal_claim"], False, "real_world_causal_claim"
                ),
            ),
            credible_interval_claim=cast(
                Literal[False],
                _exact_boolean(
                    value["credible_interval_claim"],
                    False,
                    "credible_interval_claim",
                ),
            ),
            confidence_interval_claim=cast(
                Literal[False],
                _exact_boolean(
                    value["confidence_interval_claim"],
                    False,
                    "confidence_interval_claim",
                ),
            ),
            iid_mcse_reported=cast(
                Literal[False],
                _exact_boolean(value["iid_mcse_reported"], False, "iid_mcse_reported"),
            ),
            publication_eligible=cast(
                Literal[False],
                _exact_boolean(
                    value["publication_eligible"], False, "publication_eligible"
                ),
            ),
            schema_version=cast(str, value["schema_version"]),
        )

    @property
    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.to_mapping())

    @property
    def sha256(self) -> str:
        return sha256_bytes(self.canonical_bytes)

    def to_mapping(self) -> dict[str, object]:
        return {
            "analysis_chronology_externally_verified": (
                self.analysis_chronology_externally_verified
            ),
            "analysis_sha256": self.analysis_sha256,
            "batch_receipt_sha256": self.batch_receipt_sha256,
            "calibration_manifest_raw_sha256": (self.calibration_manifest_raw_sha256),
            "calibration_protocol_sha256": self.calibration_protocol_sha256,
            "caller_commitment_only": self.caller_commitment_only,
            "claim_scope": self.claim_scope,
            "confidence_interval_claim": self.confidence_interval_claim,
            "credible_interval_claim": self.credible_interval_claim,
            "evaluation_config_sha256": self.evaluation_config_sha256,
            "expected_analysis_spec_sha256": self.expected_analysis_spec_sha256,
            "expected_output_manifest_sha256": self.expected_output_manifest_sha256,
            "expected_plan_sha256": self.expected_plan_sha256,
            "expected_slot_manifest_sha256": self.expected_slot_manifest_sha256,
            "external_custody_verified": self.external_custody_verified,
            "externally_registered": self.externally_registered,
            "files": [item.to_mapping() for item in self.files],
            "grid_sha256": self.grid_sha256,
            "historical_execution_attested": self.historical_execution_attested,
            "iid_mcse_reported": self.iid_mcse_reported,
            "independent_reproduction": self.independent_reproduction,
            "loaded_source_batch_replay_only": self.loaded_source_batch_replay_only,
            "loaded_sources_replay_only": self.loaded_sources_replay_only,
            "plan_construction_sha256": self.plan_construction_sha256,
            "posterior_draw_count": self.posterior_draw_count,
            "protocol_sha256": self.protocol_sha256,
            "publication_eligible": self.publication_eligible,
            "real_world_causal_claim": self.real_world_causal_claim,
            "real_world_uncertainty_components_identified": (
                self.real_world_uncertainty_components_identified
            ),
            "result_sha256": self.result_sha256,
            "scenario_replicate_count": self.scenario_replicate_count,
            "schema_version": self.schema_version,
            "source_artifact_manifest_sha256": (self.source_artifact_manifest_sha256),
            "source_batch_sha256": self.source_batch_sha256,
            "source_count": self.source_count,
            "source_execution_identity_externally_attested": (
                self.source_execution_identity_externally_attested
            ),
            "synthetic_only": self.synthetic_only,
        }


class _LedgerFactorialEvidenceLoadMintTicket:
    """Opaque one-use authority for one descriptor-captured byte snapshot."""

    __slots__ = ("__weakref__",)

    def __init__(self) -> None:
        raise TypeError("factorial evidence load mint tickets are verifier-issued")

    def __copy__(self) -> NoReturn:
        raise TypeError("factorial evidence load mint tickets cannot be copied")

    def __deepcopy__(self, memo: object) -> NoReturn:
        del memo
        raise TypeError("factorial evidence load mint tickets cannot be copied")

    def __reduce__(self) -> NoReturn:
        raise TypeError("factorial evidence load mint tickets cannot be serialized")


@dataclass(frozen=True, slots=True)
class _LedgerFactorialEvidenceLoadMintBinding:
    root: str
    root_snapshot_fingerprint: tuple[int, ...]
    manifest_raw_sha256: str


@dataclass(frozen=True, slots=True, init=False, weakref_slot=True)
class LedgerFactorialEvidenceBundleLoadResult:
    """Verifier-owned replay snapshot with deliberately narrow claims.

    Nested sources and the source batch retain their loader-issued replay-only
    origins.  This outer type never remints them as current-process executions.
    """

    root: Path
    root_snapshot_fingerprint: tuple[int, ...]
    manifest_raw_sha256: str
    manifest: LedgerFactorialEvidenceBundleManifest
    calibration: CalibrationArtifactLoadResult
    calibration_protocol: StudyProtocol
    plan_construction: LedgerFactorialPlanConstructionSpec
    plan: LedgerFactorialStudyPlan
    protocol: StudyProtocol
    batch_receipt: ProtocolExecutionReceipt
    source_artifacts: tuple[ProtocolBoundIntegratedSyntheticRunArtifact, ...]
    source_batch: LedgerFactorialSourceBatch
    result: SourceBoundLedgerFactorialEvaluation
    caller_manifest_commitment_verified: Literal[True] = True
    caller_commitment_only: Literal[True] = True
    loaded_sources_replay_only: Literal[True] = True
    loaded_source_batch_replay_only: Literal[True] = True
    external_custody_verified: Literal[False] = False
    externally_registered: Literal[False] = False
    historical_execution_attested: Literal[False] = False
    independent_reproduction: Literal[False] = False
    analysis_chronology_externally_verified: Literal[False] = False
    source_execution_identity_externally_attested: Literal[False] = False
    real_world_uncertainty_components_identified: Literal[False] = False
    real_world_causal_claim: Literal[False] = False
    credible_interval_claim: Literal[False] = False
    confidence_interval_claim: Literal[False] = False
    iid_mcse_reported: Literal[False] = False
    publication_eligible: Literal[False] = False

    def __init__(self) -> None:
        raise TypeError(
            "LedgerFactorialEvidenceBundleLoadResult is verifier-produced; "
            "use load_ledger_factorial_evidence_bundle"
        )

    def __copy__(self) -> NoReturn:
        raise TypeError("verified factorial evidence results cannot be copied")

    def __deepcopy__(self, memo: object) -> NoReturn:
        del memo
        raise TypeError("verified factorial evidence results cannot be copied")

    def __reduce__(self) -> NoReturn:
        raise TypeError("verified factorial evidence results cannot be serialized")

    def _fresh_validate(self) -> None:
        if not isinstance(self.root, Path) or not self.root.is_absolute():
            _fail("loaded factorial bundle root must be an absolute Path")
        if (
            type(self.root_snapshot_fingerprint) is not tuple
            or len(self.root_snapshot_fingerprint) != 9
            or not all(type(value) is int for value in self.root_snapshot_fingerprint)
        ):
            _fail("loaded factorial bundle root snapshot fingerprint changed")
        _sha256(self.manifest_raw_sha256, "manifest_raw_sha256")
        exact_values: tuple[tuple[object, type[object], str], ...] = (
            (self.manifest, LedgerFactorialEvidenceBundleManifest, "manifest"),
            (self.calibration, CalibrationArtifactLoadResult, "calibration"),
            (self.calibration_protocol, StudyProtocol, "calibration_protocol"),
            (
                self.plan_construction,
                LedgerFactorialPlanConstructionSpec,
                "plan_construction",
            ),
            (self.plan, LedgerFactorialStudyPlan, "plan"),
            (self.protocol, StudyProtocol, "protocol"),
            (self.batch_receipt, ProtocolExecutionReceipt, "batch_receipt"),
            (self.source_batch, LedgerFactorialSourceBatch, "source_batch"),
            (self.result, SourceBoundLedgerFactorialEvaluation, "result"),
        )
        for value, expected, field in exact_values:
            if type(value) is not expected:
                _fail(f"loaded factorial bundle {field} has the wrong exact type")
        _validate_loaded_calibration(self.calibration)
        if (
            not isinstance(self.source_artifacts, tuple)
            or not self.source_artifacts
            or not all(
                type(item) is ProtocolBoundIntegratedSyntheticRunArtifact
                for item in self.source_artifacts
            )
        ):
            _fail("loaded factorial sources have the wrong exact tuple type")
        for source in self.source_artifacts:
            try:
                require_ledger_derivation_source_artifact(source)
            except IntegratedProtocolExecutionError as error:
                raise LedgerFactorialEvidenceBundleError(
                    "loaded factorial source lacks verifier-owned replay origin"
                ) from error
        try:
            require_ledger_factorial_derivation_source_batch(self.source_batch)
            require_source_bound_ledger_factorial_evaluation(self.result)
        except Exception as error:
            raise LedgerFactorialEvidenceBundleError(
                "loaded factorial batch or result lacks strict replay origin"
            ) from error
        if (
            self.caller_manifest_commitment_verified is not True
            or self.caller_commitment_only is not True
            or self.loaded_sources_replay_only is not True
            or self.loaded_source_batch_replay_only is not True
            or any(
                value is not False
                for value in (
                    self.external_custody_verified,
                    self.externally_registered,
                    self.historical_execution_attested,
                    self.independent_reproduction,
                    self.analysis_chronology_externally_verified,
                    self.source_execution_identity_externally_attested,
                    self.real_world_uncertainty_components_identified,
                    self.real_world_causal_claim,
                    self.credible_interval_claim,
                    self.confidence_interval_claim,
                    self.iid_mcse_reported,
                    self.publication_eligible,
                )
            )
        ):
            _fail("loaded factorial bundle claim flags cannot be widened")
        manifest = self.manifest
        if (
            manifest.sha256 != self.manifest_raw_sha256
            or self.calibration.manifest_raw_sha256
            != manifest.calibration_manifest_raw_sha256
            or self.calibration_protocol.sha256 != manifest.calibration_protocol_sha256
            or self.plan_construction.sha256 != manifest.plan_construction_sha256
            or self.plan.sha256 != manifest.expected_plan_sha256
            or self.plan.slot_manifest_sha256 != manifest.expected_slot_manifest_sha256
            or self.plan.output_manifest_sha256
            != manifest.expected_output_manifest_sha256
            or self.plan.analysis_spec.sha256 != manifest.expected_analysis_spec_sha256
            or self.protocol.sha256 != manifest.protocol_sha256
            or self.batch_receipt.sha256 != manifest.batch_receipt_sha256
            or self.source_batch.sha256 != manifest.source_batch_sha256
            or self.result.sha256 != manifest.result_sha256
            or self.result.analysis.sha256 != manifest.analysis_sha256
            or _grid_sha256(self.result) != manifest.grid_sha256
            or _source_manifest_sha256(self.source_artifacts)
            != manifest.source_artifact_manifest_sha256
        ):
            _fail("loaded factorial bundle cross-artifact hash binding changed")
        config = integrated_synthetic_forward_market_config()
        try:
            _validate_calibration_protocol(
                self.calibration_protocol, self.calibration.run
            )
            rebuilt = self.plan_construction.rebuild(
                calibration_run=self.calibration.run,
                config=config,
            )
            validate_protocol_execution_receipt_binding(
                self.protocol, self.batch_receipt
            )
        except Exception as error:
            raise LedgerFactorialEvidenceBundleError(
                "loaded factorial plan or protocol binding changed"
            ) from error
        source_hashes = tuple(item.sha256 for item in self.source_artifacts)
        if (
            rebuilt.canonical_bytes != self.plan.canonical_bytes
            or manifest.posterior_draw_count != self.plan.posterior_draw_count
            or manifest.scenario_replicate_count != self.plan.scenario_replicate_count
            or manifest.source_count != len(self.plan.slots)
            or len(self.source_artifacts) != len(self.plan.slots)
            or self.source_batch.plan_sha256 != self.plan.sha256
            or self.source_batch.protocol_sha256 != self.protocol.sha256
            or self.source_batch.batch_receipt != self.batch_receipt
            or self.result.plan.canonical_bytes != self.plan.canonical_bytes
            or self.result.protocol.sha256 != self.protocol.sha256
            or self.result.source_batch is not self.source_batch
            or self.result.source_artifact_sha256s != source_hashes
            or tuple(
                item.source_artifact_sha256 for item in self.source_batch.executions
            )
            != source_hashes
            or tuple(item.input_sha256 for item in self.source_artifacts)
            != tuple(item.pair_input_sha256 for item in self.plan.slots)
        ):
            _fail("loaded factorial artifacts no longer share one row-major derivation")


_VERIFIED_FACTORIAL_EVIDENCE_RESULTS: WeakValueDictionary[
    int, LedgerFactorialEvidenceBundleLoadResult
] = WeakValueDictionary()
_VERIFIED_FACTORIAL_EVIDENCE_RESULT_SEALS: dict[int, str] = {}
_FACTORIAL_EVIDENCE_LOAD_MINT_TICKETS: WeakKeyDictionary[
    _LedgerFactorialEvidenceLoadMintTicket, _LedgerFactorialEvidenceLoadMintBinding
] = WeakKeyDictionary()
_VERIFIED_FACTORIAL_EVIDENCE_RESULTS_LOCK = RLock()


def _load_result_seal(loaded: LedgerFactorialEvidenceBundleLoadResult) -> str:
    calibration_run = loaded.calibration.run
    return sha256_bytes(
        canonical_json_bytes(
            {
                "analysis_sha256": loaded.result.analysis.sha256,
                "batch_receipt_sha256": loaded.batch_receipt.sha256,
                "calibration_audit": loaded.calibration.audit.to_mapping(),
                "calibration_config_sha256": calibration_run.config.sha256,
                "calibration_event_keys_sha256": sha256_bytes(
                    canonical_json_bytes(list(calibration_run.event_keys))
                ),
                "calibration_input_sha256": calibration_run.observed.sha256,
                "calibration_manifest_raw_sha256": (
                    loaded.calibration.manifest_raw_sha256
                ),
                "calibration_posterior_sha256": calibration_run.posterior.sha256,
                "calibration_protocol_sha256": loaded.calibration_protocol.sha256,
                "calibration_report_sha256": calibration_run.report.sha256,
                "calibration_sbc_sha256": calibration_run.sbc.sha256,
                "calibration_simulator_config_sha256": sha256_bytes(
                    calibration_run.simulator_config_json
                ),
                "calibration_true_parameters_sha256": (
                    calibration_run.true_parameters_sha256
                ),
                "caller_commitment_only": loaded.caller_commitment_only,
                "caller_manifest_commitment_verified": (
                    loaded.caller_manifest_commitment_verified
                ),
                "claims": {
                    "analysis_chronology_externally_verified": (
                        loaded.analysis_chronology_externally_verified
                    ),
                    "confidence_interval_claim": loaded.confidence_interval_claim,
                    "credible_interval_claim": loaded.credible_interval_claim,
                    "external_custody_verified": loaded.external_custody_verified,
                    "externally_registered": loaded.externally_registered,
                    "historical_execution_attested": (
                        loaded.historical_execution_attested
                    ),
                    "iid_mcse_reported": loaded.iid_mcse_reported,
                    "independent_reproduction": loaded.independent_reproduction,
                    "loaded_source_batch_replay_only": (
                        loaded.loaded_source_batch_replay_only
                    ),
                    "loaded_sources_replay_only": loaded.loaded_sources_replay_only,
                    "publication_eligible": loaded.publication_eligible,
                    "real_world_causal_claim": loaded.real_world_causal_claim,
                    "real_world_uncertainty_components_identified": (
                        loaded.real_world_uncertainty_components_identified
                    ),
                    "source_execution_identity_externally_attested": (
                        loaded.source_execution_identity_externally_attested
                    ),
                },
                "manifest_raw_sha256": loaded.manifest_raw_sha256,
                "manifest_sha256": loaded.manifest.sha256,
                "plan_construction_sha256": loaded.plan_construction.sha256,
                "plan_sha256": loaded.plan.sha256,
                "protocol_sha256": loaded.protocol.sha256,
                "result_sha256": loaded.result.sha256,
                "root": str(loaded.root),
                "root_snapshot_fingerprint": list(loaded.root_snapshot_fingerprint),
                "source_artifact_sha256s": [
                    item.sha256 for item in loaded.source_artifacts
                ],
                "source_batch_sha256": loaded.source_batch.sha256,
            }
        )
    )


def _forget_verified_factorial_evidence_result(identity: int) -> None:
    with _VERIFIED_FACTORIAL_EVIDENCE_RESULTS_LOCK:
        if _VERIFIED_FACTORIAL_EVIDENCE_RESULTS.get(identity) is None:
            _VERIFIED_FACTORIAL_EVIDENCE_RESULT_SEALS.pop(identity, None)


def require_verified_ledger_factorial_evidence_bundle_result(
    loaded: LedgerFactorialEvidenceBundleLoadResult,
) -> LedgerFactorialEvidenceBundleLoadResult:
    """Require a sealed same-process result from the strict rooted loader."""
    if type(loaded) is not LedgerFactorialEvidenceBundleLoadResult:
        _fail("factorial evidence load result has the wrong concrete type")
    try:
        before_seal = _load_result_seal(loaded)
    except Exception as error:
        raise LedgerFactorialEvidenceBundleError(
            "factorial evidence load result cannot reproduce its mint-time seal"
        ) from error
    with _VERIFIED_FACTORIAL_EVIDENCE_RESULTS_LOCK:
        registered = _VERIFIED_FACTORIAL_EVIDENCE_RESULTS.get(id(loaded))
        mint_seal = _VERIFIED_FACTORIAL_EVIDENCE_RESULT_SEALS.get(id(loaded))
    if registered is not loaded or mint_seal != before_seal:
        _fail(
            "factorial evidence load result was not minted by the strict loader "
            "or differs from its mint-time seal"
        )
    loaded._fresh_validate()
    after_seal = _load_result_seal(loaded)
    with _VERIFIED_FACTORIAL_EVIDENCE_RESULTS_LOCK:
        still_registered = _VERIFIED_FACTORIAL_EVIDENCE_RESULTS.get(id(loaded))
        current_seal = _VERIFIED_FACTORIAL_EVIDENCE_RESULT_SEALS.get(id(loaded))
    if (
        still_registered is not loaded
        or before_seal != after_seal
        or current_seal != after_seal
    ):
        _fail("factorial evidence load result changed after verifier-owned mint")
    return loaded


def _mint_load_result(
    *,
    mint_ticket: _LedgerFactorialEvidenceLoadMintTicket,
    root: Path,
    root_snapshot_fingerprint: tuple[int, ...],
    manifest_raw_sha256: str,
    manifest: LedgerFactorialEvidenceBundleManifest,
    calibration: CalibrationArtifactLoadResult,
    calibration_protocol: StudyProtocol,
    plan_construction: LedgerFactorialPlanConstructionSpec,
    plan: LedgerFactorialStudyPlan,
    protocol: StudyProtocol,
    batch_receipt: ProtocolExecutionReceipt,
    source_artifacts: tuple[ProtocolBoundIntegratedSyntheticRunArtifact, ...],
    source_batch: LedgerFactorialSourceBatch,
    result: SourceBoundLedgerFactorialEvaluation,
) -> LedgerFactorialEvidenceBundleLoadResult:
    if type(mint_ticket) is not _LedgerFactorialEvidenceLoadMintTicket:
        _fail("factorial evidence result mint requires an exact verifier ticket")
    with _VERIFIED_FACTORIAL_EVIDENCE_RESULTS_LOCK:
        binding = _FACTORIAL_EVIDENCE_LOAD_MINT_TICKETS.pop(mint_ticket, None)
    if binding is None:
        _fail("factorial evidence result mint ticket is invalid or consumed")
    if (
        binding.root != str(root)
        or binding.root_snapshot_fingerprint != root_snapshot_fingerprint
        or binding.manifest_raw_sha256 != manifest_raw_sha256
    ):
        _fail("factorial evidence mint fields differ from ticket binding")
    loaded = object.__new__(LedgerFactorialEvidenceBundleLoadResult)
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
        ("source_batch", source_batch),
        ("result", result),
        ("caller_manifest_commitment_verified", True),
        ("caller_commitment_only", True),
        ("loaded_sources_replay_only", True),
        ("loaded_source_batch_replay_only", True),
        ("external_custody_verified", False),
        ("externally_registered", False),
        ("historical_execution_attested", False),
        ("independent_reproduction", False),
        ("analysis_chronology_externally_verified", False),
        ("source_execution_identity_externally_attested", False),
        ("real_world_uncertainty_components_identified", False),
        ("real_world_causal_claim", False),
        ("credible_interval_claim", False),
        ("confidence_interval_claim", False),
        ("iid_mcse_reported", False),
        ("publication_eligible", False),
    )
    for field, value in fields:
        object.__setattr__(loaded, field, value)
    loaded._fresh_validate()
    identity = id(loaded)
    seal = _load_result_seal(loaded)
    with _VERIFIED_FACTORIAL_EVIDENCE_RESULTS_LOCK:
        _VERIFIED_FACTORIAL_EVIDENCE_RESULTS[identity] = loaded
        _VERIFIED_FACTORIAL_EVIDENCE_RESULT_SEALS[identity] = seal
    finalize(loaded, _forget_verified_factorial_evidence_result, identity)
    return loaded


def _validate_writer_inputs(
    *,
    calibration_run: CalibrationRun,
    calibration_protocol: StudyProtocol,
    plan: LedgerFactorialStudyPlan,
    protocol: StudyProtocol,
    source_batch: LedgerFactorialSourceBatch,
    source_artifacts: tuple[ProtocolBoundIntegratedSyntheticRunArtifact, ...],
) -> tuple[
    LedgerFactorialPlanConstructionSpec,
    tuple[ProtocolBoundIntegratedSyntheticRunArtifact, ...],
    SourceBoundLedgerFactorialEvaluation,
]:
    exact_values: tuple[tuple[object, type[object], str], ...] = (
        (calibration_run, CalibrationRun, "calibration_run"),
        (calibration_protocol, StudyProtocol, "calibration_protocol"),
        (plan, LedgerFactorialStudyPlan, "plan"),
        (protocol, StudyProtocol, "protocol"),
        (source_batch, LedgerFactorialSourceBatch, "source_batch"),
    )
    for value, expected, field in exact_values:
        if type(value) is not expected:
            raise TypeError(f"{field} has the wrong exact type")
    if not isinstance(source_artifacts, tuple) or not source_artifacts:
        raise TypeError("source_artifacts must be a non-empty tuple")
    expected_source_count = len(plan.slots)
    if (
        expected_source_count <= 0
        or expected_source_count > LEDGER_FACTORIAL_MAX_SOURCE_COUNT
        or len(source_artifacts) != expected_source_count
    ):
        _fail(
            "writer source count must equal the bounded frozen plan before "
            "source-origin validation"
        )
    if not all(
        type(item) is ProtocolBoundIntegratedSyntheticRunArtifact
        for item in source_artifacts
    ):
        raise TypeError("source_artifacts contain a wrong exact type")
    try:
        _validate_calibration_protocol(calibration_protocol, calibration_run)
        require_executed_ledger_factorial_source_batch(source_batch)
        for source in source_artifacts:
            require_verifier_owned_protocol_bound_integrated_artifact(source)
    except Exception as error:
        raise LedgerFactorialEvidenceBundleError(
            "writer requires bound calibration and current-process executed sources"
        ) from error
    config = integrated_synthetic_forward_market_config()
    if (
        plan.config != config
        or plan.config.sha256 != config.sha256
        or plan.calibration_run is not calibration_run
    ):
        _fail("writer plan differs from supplied calibration or allowlisted config")
    try:
        construction = LedgerFactorialPlanConstructionSpec.from_plan(plan)
        rebuilt = construction.rebuild(
            calibration_run=calibration_run,
            config=config,
        )
    except Exception as error:
        raise LedgerFactorialEvidenceBundleError(
            "writer cannot reconstruct the supplied factorial plan"
        ) from error
    if rebuilt.canonical_bytes != plan.canonical_bytes:
        _fail("writer construction does not reproduce the supplied factorial plan")
    by_input = {item.input_sha256: item for item in source_artifacts}
    if len(by_input) != len(source_artifacts):
        _fail("writer source tuple reuses a pair input")
    try:
        ordered = tuple(by_input[item.pair_input_sha256] for item in plan.slots)
    except KeyError as error:
        raise LedgerFactorialEvidenceBundleError(
            "writer source tuple does not cover every row-major slot"
        ) from error
    if len(ordered) != len(source_artifacts):
        _fail("writer source tuple contains an extra grid artifact")
    source_hashes = tuple(item.sha256 for item in ordered)
    if (
        len(plan.slots) != plan.posterior_draw_count * plan.scenario_replicate_count
        or source_batch.plan_sha256 != plan.sha256
        or source_batch.protocol_sha256 != protocol.sha256
        or tuple(item.source_artifact_sha256 for item in source_batch.executions)
        != source_hashes
        or tuple(item.pair_input_sha256 for item in source_batch.executions)
        != tuple(item.pair_input_sha256 for item in plan.slots)
    ):
        _fail("executed source batch differs from the supplied row-major grid")
    try:
        fresh = derive_source_bound_ledger_factorial_evaluation(
            plan=plan,
            protocol=protocol,
            source_batch=source_batch,
            source_artifacts=ordered,
        )
        require_executed_source_bound_ledger_factorial_evaluation(fresh)
    except Exception as error:
        raise LedgerFactorialEvidenceBundleError(
            "writer cannot freshly derive the complete source-bound result"
        ) from error
    return construction, ordered, fresh


def _write_ledger_factorial_evidence_bundle(
    directory: Path,
    *,
    calibration_run: CalibrationRun,
    calibration_protocol: StudyProtocol,
    plan: LedgerFactorialStudyPlan,
    protocol: StudyProtocol,
    source_batch: LedgerFactorialSourceBatch,
    source_artifacts: tuple[ProtocolBoundIntegratedSyntheticRunArtifact, ...],
    project_root: Path | None,
) -> Path:
    if not isinstance(directory, Path):
        raise TypeError("directory must be a Path")
    candidate = Path(os.path.abspath(directory.expanduser()))
    if candidate.name in {"", ".", ".."}:
        _fail("factorial bundle destination must name one new directory")
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
            _fail("factorial evidence bundles are immutable and cannot be overwritten")

        construction, ordered_sources, result = _validate_writer_inputs(
            calibration_run=calibration_run,
            calibration_protocol=calibration_protocol,
            plan=plan,
            protocol=protocol,
            source_batch=source_batch,
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
                source_batch.batch_receipt.to_mapping()
            ),
            "calibration-protocol.json": canonical_json_bytes(
                calibration_protocol.to_mapping()
            ),
            "evaluation-config.json": canonical_json_bytes(plan.config.to_mapping()),
            "plan-construction.json": construction.canonical_bytes,
            "protocol.json": canonical_json_bytes(protocol.to_mapping()),
            "result.json": result.canonical_bytes,
            "source-batch.json": source_batch.canonical_bytes,
        }
        payloads.update(
            {
                f"source-{ordinal:06d}.json": source.canonical_bytes
                for ordinal, source in enumerate(ordered_sources)
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
            LedgerFactorialEvidenceBundleFile(
                path=path,
                sha256=sha256_bytes(content),
                size_bytes=len(content),
            )
            for path, content in sorted(leaf_payloads.items())
        )
        manifest = LedgerFactorialEvidenceBundleManifest(
            files=files,
            posterior_draw_count=plan.posterior_draw_count,
            scenario_replicate_count=plan.scenario_replicate_count,
            source_count=len(ordered_sources),
            calibration_manifest_raw_sha256=sha256_bytes(
                calibration_payloads["manifest.json"]
            ),
            calibration_protocol_sha256=calibration_protocol.sha256,
            evaluation_config_sha256=plan.config.sha256,
            plan_construction_sha256=construction.sha256,
            expected_plan_sha256=plan.sha256,
            expected_slot_manifest_sha256=plan.slot_manifest_sha256,
            expected_output_manifest_sha256=plan.output_manifest_sha256,
            expected_analysis_spec_sha256=plan.analysis_spec.sha256,
            protocol_sha256=protocol.sha256,
            batch_receipt_sha256=source_batch.batch_receipt.sha256,
            source_batch_sha256=source_batch.sha256,
            source_artifact_manifest_sha256=_source_manifest_sha256(ordered_sources),
            grid_sha256=_grid_sha256(result),
            analysis_sha256=result.analysis.sha256,
            result_sha256=result.sha256,
        )
        manifest_bytes = manifest.canonical_bytes
        all_bytes = len(manifest_bytes) + sum(
            len(content) for content in leaf_payloads.values()
        )
        if all_bytes > _MAX_TOTAL_BYTES:
            _fail("generated factorial bundle exceeds its total byte limit")
        for path, content in leaf_payloads.items():
            if len(content) <= 0 or len(content) > _file_limit(path):
                _fail(f"generated factorial bundle leaf exceeds limit: {path}")
        if len(manifest_bytes) > _MAX_MANIFEST_BYTES:
            _fail("generated factorial bundle manifest exceeds its byte limit")

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
            raise LedgerFactorialEvidenceBundleError(
                "cannot create descriptor-relative factorial evidence bundle"
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
                _write_new_file_at(root_fd, "bundle-manifest.json", manifest_bytes),
            )
        )
        os.fsync(calibration_fd)
        os.fsync(root_fd)
        _require_directory_chain_current(parent_chain)
        if incomplete_leaf is None:
            _fail("factorial bundle incomplete marker was not retained")
        expected_root_names = frozenset(payloads) | {
            "bundle-manifest.json",
            "calibration",
            _INCOMPLETE_MARKER,
        }
        if (
            _bounded_directory_names(
                root_fd,
                maximum_entries=len(expected_root_names),
                label="new factorial bundle root",
            )
            != expected_root_names
        ):
            _fail("new factorial bundle membership changed before publication")
        if (
            _bounded_directory_names(
                calibration_fd,
                maximum_entries=len(_CALIBRATION_FILES),
                label="new factorial calibration directory",
            )
            != _CALIBRATION_FILES
        ):
            _fail("new factorial calibration membership changed before publication")
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
            _fail("factorial bundle directory identity changed while written")
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
            committed_root_names = expected_root_names - {_INCOMPLETE_MARKER}
            current_root_names = _bounded_directory_names(
                root_fd,
                maximum_entries=len(committed_root_names),
                label="committed factorial bundle root",
            )
            if (
                _directory_identity(linked_root) != root_identity
                or _directory_identity(os.fstat(root_fd)) != root_identity
                or _INCOMPLETE_MARKER in current_root_names
            ):
                _fail("factorial bundle identity changed at final commit")
            if (
                current_root_names != committed_root_names
                or _bounded_directory_names(
                    calibration_fd,
                    maximum_entries=len(_CALIBRATION_FILES),
                    label="committed factorial calibration directory",
                )
                != _CALIBRATION_FILES
            ):
                _fail("factorial bundle membership changed at final commit")
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
                raise LedgerFactorialEvidenceBundleError(
                    "factorial publication state is indeterminate after failure"
                ) from restore_error
            finally:
                if restored_marker is not None:
                    restored_marker.close()
            raise LedgerFactorialEvidenceBundleError(
                "factorial final commit failed and was returned to incomplete state"
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


def write_ledger_factorial_evidence_bundle(
    directory: Path,
    *,
    calibration_run: CalibrationRun,
    calibration_protocol: StudyProtocol,
    plan: LedgerFactorialStudyPlan,
    protocol: StudyProtocol,
    source_batch: LedgerFactorialSourceBatch,
    source_artifacts: tuple[ProtocolBoundIntegratedSyntheticRunArtifact, ...],
    project_root: Path | None = None,
) -> Path:
    """Publish one new immutable bundle; the root manifest is written last."""
    try:
        return _write_ledger_factorial_evidence_bundle(
            directory,
            calibration_run=calibration_run,
            calibration_protocol=calibration_protocol,
            plan=plan,
            protocol=protocol,
            source_batch=source_batch,
            source_artifacts=source_artifacts,
            project_root=project_root,
        )
    except LedgerFactorialEvidenceBundleError:
        raise
    except LedgerEvidenceBundleError as error:
        raise LedgerFactorialEvidenceBundleError(str(error)) from error


@dataclass(slots=True)
class _CapturedFactorialTree:
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


def _canonical_object(content: bytes, source: str) -> dict[str, object]:
    try:
        value = read_json_bytes_object(content, source=source)
    except MetadataValidationError as error:
        raise LedgerFactorialEvidenceBundleError(str(error)) from error
    if canonical_json_bytes(value) != content:
        _fail(f"{source} must use canonical duplicate-free JSON")
    return value


def _parse_protocol(content: bytes, source: str) -> StudyProtocol:
    mapping = _canonical_object(content, source)
    try:
        protocol = StudyProtocol.from_mapping(mapping)
    except ProtocolValidationError as error:
        raise LedgerFactorialEvidenceBundleError(
            f"invalid {source}: {error}"
        ) from error
    if canonical_json_bytes(protocol.to_mapping()) != content:
        _fail(f"{source} does not survive strict reconstruction")
    return protocol


def _parse_receipt(content: bytes) -> ProtocolExecutionReceipt:
    mapping = _canonical_object(content, "batch-receipt.json")
    try:
        receipt = ProtocolExecutionReceipt.from_mapping(mapping)
    except ProtocolValidationError as error:
        raise LedgerFactorialEvidenceBundleError(
            f"invalid batch receipt: {error}"
        ) from error
    if canonical_json_bytes(receipt.to_mapping()) != content:
        _fail("batch receipt does not survive strict reconstruction")
    return receipt


def _bounded_directory_names(
    descriptor: int,
    *,
    maximum_entries: int,
    label: str,
) -> frozenset[str]:
    """Read at most ``maximum_entries + 1`` names from a held directory.

    The extra observation exists only to fail closed on overfull directories;
    no unbounded name list or attacker-controlled diagnostic sample is retained.
    """
    if (
        isinstance(maximum_entries, bool)
        or not isinstance(maximum_entries, int)
        or maximum_entries <= 0
    ):
        raise TypeError("maximum_entries must be a positive integer")
    names: set[str] = set()
    try:
        with os.scandir(descriptor) as entries:
            for entry in entries:
                if len(names) >= maximum_entries:
                    _fail(
                        f"{label} exceeds its bounded membership count "
                        f"of {maximum_entries}"
                    )
                names.add(entry.name)
    except OSError as error:
        raise LedgerFactorialEvidenceBundleError(
            f"cannot enumerate bounded {label} membership"
        ) from error
    return frozenset(names)


def _capture_factorial_tree(
    directory: Path,
    *,
    expected_manifest_sha256: str,
) -> _CapturedFactorialTree:
    if not isinstance(directory, Path):
        raise TypeError("directory must be a Path")
    root_chain = _open_absolute_directory_chain(directory)
    root = root_chain.absolute
    root_fd = root_chain.final_fd
    calibration_fd: int | None = None
    leaves: dict[str, _CapturedLeaf] = {}
    try:
        root_info = os.fstat(root_fd)
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
            _fail("factorial manifest capture did not retain bytes")
        if sha256_bytes(manifest_payload) != expected_manifest_sha256:
            _fail("factorial manifest differs from caller commitment")
        manifest = LedgerFactorialEvidenceBundleManifest.from_mapping(
            _canonical_object(manifest_payload, "bundle-manifest.json")
        )
        if manifest.canonical_bytes != manifest_payload:
            _fail("factorial manifest does not survive strict reconstruction")
        try:
            os.stat(
                _INCOMPLETE_MARKER,
                dir_fd=root_fd,
                follow_symlinks=False,
            )
        except FileNotFoundError:
            pass
        except OSError as error:
            raise LedgerFactorialEvidenceBundleError(
                "cannot inspect factorial bundle publication marker"
            ) from error
        else:
            _fail("factorial bundle publication is incomplete")
        expected_root = (
            _ROOT_FIXED_FILES
            | {"calibration"}
            | {f"source-{ordinal:06d}.json" for ordinal in range(manifest.source_count)}
        )
        root_names = _bounded_directory_names(
            root_fd,
            maximum_entries=len(expected_root),
            label="factorial bundle root",
        )
        if _INCOMPLETE_MARKER in root_names:
            _fail("factorial bundle publication is incomplete")
        if root_names != expected_root:
            _fail("factorial bundle root file set differs from committed layout")
        declared_entries = {item.path: item for item in manifest.files}
        declared_paths = (root_names - {"bundle-manifest.json", "calibration"}) | {
            f"calibration/{name}" for name in _CALIBRATION_FILES
        }
        if set(declared_entries) != declared_paths:
            _fail("factorial manifest does not describe fixed root membership")
        declared_total = len(manifest_payload) + sum(
            item.size_bytes for item in manifest.files
        )
        if declared_total > _MAX_TOTAL_BYTES:
            _fail("factorial manifest declares excessive total bytes")
        for path, entry in declared_entries.items():
            if entry.size_bytes > _file_limit(path):
                _fail(f"factorial manifest leaf exceeds its fixed limit: {path}")
        directory_flags = (
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_DIRECTORY", 0)
        )
        calibration_fd = os.open("calibration", directory_flags, dir_fd=root_fd)
        calibration_info = os.fstat(calibration_fd)
        if not stat.S_ISDIR(calibration_info.st_mode):
            _fail("factorial calibration member must be a directory")
        calibration_names = _bounded_directory_names(
            calibration_fd,
            maximum_entries=len(_CALIBRATION_FILES),
            label="factorial calibration directory",
        )
        if calibration_names != _CALIBRATION_FILES:
            _fail("nested factorial calibration file set changed")
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
        return _CapturedFactorialTree(
            root=root,
            root_chain=root_chain,
            root_fingerprint=_fingerprint(root_info),
            root_names=root_names,
            calibration_fd=calibration_fd,
            calibration_fingerprint=_fingerprint(calibration_info),
            leaves=leaves,
            source_count=manifest.source_count,
        )
    except OSError as error:
        for leaf in leaves.values():
            os.close(leaf.fd)
        if calibration_fd is not None:
            os.close(calibration_fd)
        root_chain.close()
        raise LedgerFactorialEvidenceBundleError(
            "cannot capture fixed factorial bundle directory layout"
        ) from error
    except Exception:
        for leaf in leaves.values():
            os.close(leaf.fd)
        if calibration_fd is not None:
            os.close(calibration_fd)
        root_chain.close()
        raise


def _read_captured_leaf(captured: _CapturedFactorialTree, path: str) -> bytes:
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
        raise LedgerFactorialEvidenceBundleError(
            f"cannot read captured factorial bundle leaf: {path}"
        ) from error
    if (
        _fingerprint(linked) != leaf.fingerprint
        or _fingerprint(current) != leaf.fingerprint
        or current.st_size != leaf.size_bytes
        or len(payload) != leaf.size_bytes
        or sha256_bytes(payload) != leaf.expected_sha256
    ):
        _fail(f"factorial bundle leaf changed or differs from commitment: {path}")
    return payload


def _require_current_tree(captured: _CapturedFactorialTree) -> None:
    _require_directory_chain_current(captured.root_chain)
    if (
        _bounded_directory_names(
            captured.root_fd,
            maximum_entries=len(captured.root_names),
            label="factorial bundle root recheck",
        )
        != captured.root_names
    ):
        _fail("factorial bundle root membership changed during replay")
    if (
        _bounded_directory_names(
            captured.calibration_fd,
            maximum_entries=len(_CALIBRATION_FILES),
            label="factorial calibration recheck",
        )
        != _CALIBRATION_FILES
    ):
        _fail("factorial calibration membership changed during replay")
    for path in captured.leaves:
        _read_captured_leaf(captured, path)
    try:
        calibration_path_info = os.stat(
            "calibration", dir_fd=captured.root_fd, follow_symlinks=False
        )
    except OSError as error:
        raise LedgerFactorialEvidenceBundleError(
            "cannot finalize factorial bundle directory identity"
        ) from error
    if (
        _fingerprint(os.fstat(captured.root_fd)) != captured.root_fingerprint
        or _fingerprint(calibration_path_info) != captured.calibration_fingerprint
        or _fingerprint(os.fstat(captured.calibration_fd))
        != captured.calibration_fingerprint
    ):
        _fail("factorial bundle directory identity changed during replay")


def _verify_manifest_closure(
    captured: _CapturedFactorialTree,
    manifest: LedgerFactorialEvidenceBundleManifest,
) -> None:
    expected_paths = set(captured.leaves) - {"bundle-manifest.json"}
    entries = {item.path: item for item in manifest.files}
    if set(entries) != expected_paths:
        _fail("factorial manifest does not close over captured leaf set")
    for path, entry in entries.items():
        leaf = captured.leaves[path]
        if entry.size_bytes != leaf.size_bytes or entry.sha256 != leaf.expected_sha256:
            _fail(f"factorial manifest hash or size mismatch: {path}")
    if manifest.source_count != captured.source_count:
        _fail("factorial manifest source count differs from captured root")


def _load_ledger_factorial_evidence_bundle(
    directory: Path,
    *,
    expected_manifest_sha256: str,
) -> LedgerFactorialEvidenceBundleLoadResult:
    expected = _sha256(expected_manifest_sha256, "expected_manifest_sha256")
    captured = _capture_factorial_tree(
        directory,
        expected_manifest_sha256=expected,
    )
    try:
        manifest_bytes = _read_captured_leaf(captured, "bundle-manifest.json")
        manifest_raw_sha256 = sha256_bytes(manifest_bytes)
        manifest = LedgerFactorialEvidenceBundleManifest.from_mapping(
            _canonical_object(manifest_bytes, "bundle-manifest.json")
        )
        if manifest.canonical_bytes != manifest_bytes:
            _fail("factorial manifest does not survive strict reconstruction")
        _verify_manifest_closure(captured, manifest)
        entries = {item.path: item for item in manifest.files}

        config = integrated_synthetic_forward_market_config()
        config_bytes = _read_captured_leaf(captured, "evaluation-config.json")
        if (
            config_bytes != canonical_json_bytes(config.to_mapping())
            or entries["evaluation-config.json"].sha256 != config.sha256
            or manifest.evaluation_config_sha256 != config.sha256
        ):
            _fail("factorial config differs from allowlisted frozen factory")

        calibration_manifest_bytes = _read_captured_leaf(
            captured, "calibration/manifest.json"
        )
        if (
            sha256_bytes(calibration_manifest_bytes)
            != manifest.calibration_manifest_raw_sha256
        ):
            _fail("nested factorial calibration manifest commitment differs")
        try:
            calibration = verify_calibration_artifact_snapshot(
                {
                    name: _read_captured_leaf(captured, f"calibration/{name}")
                    for name in _CALIBRATION_FILES
                },
                expected_manifest_sha256=(manifest.calibration_manifest_raw_sha256),
            )
        except Exception as error:
            raise LedgerFactorialEvidenceBundleError(
                f"factorial calibration replay failed: {error}"
            ) from error

        calibration_protocol = _parse_protocol(
            _read_captured_leaf(captured, "calibration-protocol.json"),
            "calibration-protocol.json",
        )
        if (
            calibration_protocol.sha256 != manifest.calibration_protocol_sha256
            or entries["calibration-protocol.json"].sha256
            != manifest.calibration_protocol_sha256
        ):
            _fail("factorial calibration protocol hash differs from manifest")
        try:
            _validate_calibration_protocol(calibration_protocol, calibration.run)
        except Exception as error:
            raise LedgerFactorialEvidenceBundleError(
                "factorial calibration protocol binding is invalid"
            ) from error

        construction_bytes = _read_captured_leaf(captured, "plan-construction.json")
        if (
            sha256_bytes(construction_bytes) != manifest.plan_construction_sha256
            or entries["plan-construction.json"].sha256
            != manifest.plan_construction_sha256
        ):
            _fail("factorial construction hash differs from root manifest")
        try:
            construction = verify_ledger_factorial_plan_construction_bytes(
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
            raise LedgerFactorialEvidenceBundleError(
                f"factorial plan construction replay failed: {error}"
            ) from error
        if (
            plan.sha256 != manifest.expected_plan_sha256
            or plan.slot_manifest_sha256 != manifest.expected_slot_manifest_sha256
            or plan.output_manifest_sha256 != manifest.expected_output_manifest_sha256
            or plan.analysis_spec.sha256 != manifest.expected_analysis_spec_sha256
            or plan.posterior_draw_count != manifest.posterior_draw_count
            or plan.scenario_replicate_count != manifest.scenario_replicate_count
            or len(plan.slots) != manifest.source_count
            or len(plan.pairs) != manifest.source_count
            or captured.source_count != manifest.source_count
        ):
            _fail("reconstructed factorial plan differs from root commitments")

        protocol = _parse_protocol(
            _read_captured_leaf(captured, "protocol.json"), "protocol.json"
        )
        if (
            protocol.sha256 != manifest.protocol_sha256
            or entries["protocol.json"].sha256 != manifest.protocol_sha256
        ):
            _fail("factorial evaluation protocol hash differs from manifest")
        batch_receipt = _parse_receipt(
            _read_captured_leaf(captured, "batch-receipt.json")
        )
        if (
            batch_receipt.sha256 != manifest.batch_receipt_sha256
            or entries["batch-receipt.json"].sha256 != manifest.batch_receipt_sha256
        ):
            _fail("factorial batch receipt hash differs from manifest")
        try:
            validate_protocol_execution_receipt_binding(protocol, batch_receipt)
        except ProtocolValidationError as error:
            raise LedgerFactorialEvidenceBundleError(
                f"factorial batch receipt protocol binding failed: {error}"
            ) from error

        loaded_sources: list[ProtocolBoundIntegratedSyntheticRunArtifact] = []
        for ordinal, (slot, pair) in enumerate(
            zip(plan.slots, plan.pairs, strict=True)
        ):
            path = f"source-{ordinal:06d}.json"
            entry = entries[path]
            try:
                source = verify_protocol_bound_integrated_synthetic_run_artifact_bytes(
                    _read_captured_leaf(captured, path),
                    expected_sha256=entry.sha256,
                    expected_protocol=protocol,
                    expected_config=config,
                    expected_pair=pair,
                )
            except Exception as error:
                raise LedgerFactorialEvidenceBundleError(
                    f"factorial historical source replay failed at slot {ordinal}: "
                    f"{error}"
                ) from error
            if source.input_sha256 != slot.pair_input_sha256:
                _fail("loaded factorial source differs from reconstructed slot")
            loaded_sources.append(source)
        sources = tuple(loaded_sources)
        if _source_manifest_sha256(sources) != (
            manifest.source_artifact_manifest_sha256
        ):
            _fail("loaded factorial source-artifact manifest differs")

        source_batch_entry = entries["source-batch.json"]
        if source_batch_entry.sha256 != manifest.source_batch_sha256:
            _fail("factorial source-batch entry differs from root commitment")
        try:
            source_batch = verify_ledger_factorial_source_batch_bytes(
                _read_captured_leaf(captured, "source-batch.json"),
                expected_sha256=manifest.source_batch_sha256,
                plan=plan,
                protocol=protocol,
                batch_receipt=batch_receipt,
                source_artifacts=sources,
            )
            require_ledger_factorial_derivation_source_batch(source_batch)
        except Exception as error:
            raise LedgerFactorialEvidenceBundleError(
                f"factorial source-batch replay failed: {error}"
            ) from error

        result_entry = entries["result.json"]
        if result_entry.sha256 != manifest.result_sha256:
            _fail("factorial result entry differs from root commitment")
        try:
            result = verify_source_bound_ledger_factorial_evaluation_bytes(
                _read_captured_leaf(captured, "result.json"),
                expected_sha256=manifest.result_sha256,
                plan=plan,
                protocol=protocol,
                source_batch=source_batch,
                source_artifacts=sources,
            )
            require_source_bound_ledger_factorial_evaluation(result)
        except Exception as error:
            raise LedgerFactorialEvidenceBundleError(
                f"factorial source-bound result replay failed: {error}"
            ) from error
        if (
            result.source_batch is not source_batch
            or result.analysis.sha256 != manifest.analysis_sha256
            or _grid_sha256(result) != manifest.grid_sha256
            or result.source_artifact_sha256s != tuple(item.sha256 for item in sources)
        ):
            _fail("factorial result differs from nested root commitments")

        _require_current_tree(captured)
        mint_ticket = object.__new__(_LedgerFactorialEvidenceLoadMintTicket)
        binding = _LedgerFactorialEvidenceLoadMintBinding(
            root=str(captured.root),
            root_snapshot_fingerprint=captured.root_fingerprint,
            manifest_raw_sha256=manifest_raw_sha256,
        )
        with _VERIFIED_FACTORIAL_EVIDENCE_RESULTS_LOCK:
            _FACTORIAL_EVIDENCE_LOAD_MINT_TICKETS[mint_ticket] = binding
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
            source_batch=source_batch,
            result=result,
        )
        _require_current_tree(captured)
        return require_verified_ledger_factorial_evidence_bundle_result(loaded)
    finally:
        captured.close()


def load_ledger_factorial_evidence_bundle(
    directory: Path,
    *,
    expected_manifest_sha256: str,
) -> LedgerFactorialEvidenceBundleLoadResult:
    """Capture and replay a bundle using only a path and caller root digest."""
    try:
        return _load_ledger_factorial_evidence_bundle(
            directory,
            expected_manifest_sha256=expected_manifest_sha256,
        )
    except LedgerFactorialEvidenceBundleError:
        raise
    except LedgerEvidenceBundleError as error:
        raise LedgerFactorialEvidenceBundleError(str(error)) from error


def reverify_current_ledger_factorial_evidence_bundle(
    loaded: LedgerFactorialEvidenceBundleLoadResult,
) -> LedgerFactorialEvidenceBundleLoadResult:
    """Reopen the recorded locator and mint a new verified replay snapshot."""
    verified = require_verified_ledger_factorial_evidence_bundle_result(loaded)
    return load_ledger_factorial_evidence_bundle(
        verified.root,
        expected_manifest_sha256=verified.manifest_raw_sha256,
    )


__all__ = [
    "LEDGER_FACTORIAL_EVIDENCE_BUNDLE_CLAIM_SCOPE",
    "LEDGER_FACTORIAL_EVIDENCE_BUNDLE_SCHEMA_VERSION",
    "LedgerFactorialEvidenceBundleError",
    "LedgerFactorialEvidenceBundleFile",
    "LedgerFactorialEvidenceBundleLoadResult",
    "LedgerFactorialEvidenceBundleManifest",
    "load_ledger_factorial_evidence_bundle",
    "require_verified_ledger_factorial_evidence_bundle_result",
    "reverify_current_ledger_factorial_evidence_bundle",
    "write_ledger_factorial_evidence_bundle",
]
