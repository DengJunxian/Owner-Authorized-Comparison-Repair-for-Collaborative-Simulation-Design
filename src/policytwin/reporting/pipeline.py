"""One-way, hash-bound rendering of frozen result statistics for papers.

This module renders only statistics already present in a validated result JSON.
It deliberately contains no estimator, calibration, simulation, or data-query
entry point.
"""

from __future__ import annotations

import html
import math
import os
import re
import stat
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import NoReturn, Self, cast

from policytwin.evaluation import (
    IntervalKind,
    PolicyEvaluationSchemaError,
    ValidityCertificate,
    ValidityCheck,
    ValidityCheckStatus,
    ValidityEvidence,
    ValidityEvidenceAuthority,
    ValidityLevel,
    ValidityThresholds,
)
from policytwin.experiments import (
    ExperimentMatrix,
    ExperimentMatrixError,
    ExperimentResult,
    MatrixRunManifest,
    RegistryStatus,
    SyntheticExecutionAuthority,
    audit_matrix_run,
)
from policytwin.outcome_knowledge import DesignTemporality
from policytwin.protocol import (
    ClaimScope,
    CohortBasis,
    OutputRole,
    ProtocolValidationError,
    ReleaseEvidence,
    StudyMode,
    StudyProtocol,
    evaluate_release_eligibility,
    load_study_protocol,
)
from policytwin.provenance import (
    MetadataValidationError,
    canonical_json_bytes,
    collect_git_metadata,
    read_json_bytes_object,
    read_json_object,
    sha256_bytes,
    sha256_file,
    write_json_file,
    write_text_file,
)
from policytwin.reporting.derivation import (
    StatisticsDerivationArtifact,
    StatisticsDerivationIntegrityError,
    StatisticsDerivationSchemaError,
    derive_reporting_statistics,
)

REPORT_RESULT_SCHEMA = "policytwin.validated-reporting-result.v2"
SYNTHETIC_UNVERIFIED_RESULT_SCHEMA = "policytwin.validated-reporting-result.v1"
REPORT_MANIFEST_SCHEMA = "policytwin.paper-artifact-manifest.v5"
REQUIRED_REPORT_OUTPUTS: tuple[str, ...] = (
    "paired_distribution",
    "predicted_vs_empirical",
    "coverage_width",
    "group_benefit_harm",
    "ablation_table",
    "calibration_diagnostics",
    "risk_coverage",
    "certificate_heatmap",
    "compute_scaling_table",
)
_OUTPUT_KINDS: dict[str, str] = {
    "paired_distribution": "svg",
    "predicted_vs_empirical": "svg",
    "coverage_width": "svg",
    "group_benefit_harm": "svg",
    "ablation_table": "latex",
    "calibration_diagnostics": "svg",
    "risk_coverage": "svg",
    "certificate_heatmap": "svg",
    "compute_scaling_table": "latex",
}
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_GIT_COMMIT = re.compile(r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")
_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")
_COVERAGE_WIDTH_FIELDS = frozenset(
    {
        "cohort_id",
        "coverage",
        "estimand_id",
        "horizon_id",
        "interval_kind",
        "interval_method_id",
        "label",
        "level",
        "metric_id",
        "registered_output_id",
        "sampling_unit",
        "valid_draw_count",
        "width",
    }
)
_INTERVAL_METHOD_KIND: dict[str, IntervalKind] = {
    "central_empirical_quantile": IntervalKind.PREDICTIVE,
    "equal_tailed_posterior": IntervalKind.POSTERIOR_CREDIBLE,
    "highest_density_posterior": IntervalKind.POSTERIOR_CREDIBLE,
    "percentile_cluster_bootstrap": IntervalKind.BOOTSTRAP_CONFIDENCE,
    "studentized_cluster_bootstrap": IntervalKind.BOOTSTRAP_CONFIDENCE,
    "inverted_randomization_test": IntervalKind.RANDOMIZATION,
}
_COVERAGE_SERIES_IDENTITY_FIELDS = (
    "registered_output_id",
    "metric_id",
    "cohort_id",
    "horizon_id",
    "interval_kind",
    "interval_method_id",
    "estimand_id",
    "sampling_unit",
)
_GROUP_BENEFIT_HARM_FIELDS = frozenset(
    {
        "benefit_probability",
        "cohort_basis",
        "cohort_definition_sha256",
        "group",
        "harm_probability",
    }
)
_ABLATION_FIELDS = frozenset(
    {
        "estimate",
        "estimand_id",
        "level",
        "mechanism",
        "mcse_spec_id",
        "method_id",
        "sampling_unit",
        "uncertainty",
        "uncertainty_kind",
        "valid_draw_count",
    }
)
_ABLATION_METHOD_SEMANTICS: dict[str, tuple[str, bool]] = {
    "equal_tailed_posterior_half_width": (
        "posterior_credible_interval_half_width",
        True,
    ),
    "event_keyed_bootstrap_mcse": ("monte_carlo_standard_error", False),
    "highest_density_posterior_half_width": (
        "posterior_credible_interval_half_width",
        True,
    ),
    "inverted_randomization_half_width": (
        "randomization_interval_half_width",
        True,
    ),
    "percentile_cluster_bootstrap_half_width": (
        "bootstrap_confidence_interval_half_width",
        True,
    ),
    "studentized_cluster_bootstrap_half_width": (
        "bootstrap_confidence_interval_half_width",
        True,
    ),
}


class ReportingPipelineError(RuntimeError):
    """Base class for fail-closed reporting errors."""


class ReportingSchemaError(ReportingPipelineError, ValueError):
    """Raised when a reporting input is incomplete or ambiguous."""


class ReportingIntegrityError(ReportingPipelineError):
    """Raised when a source does not match an expected commitment."""


def _schema_fail(message: str) -> NoReturn:
    raise ReportingSchemaError(message)


def _integrity_fail(message: str) -> NoReturn:
    raise ReportingIntegrityError(message)


def _fields(
    value: Mapping[str, object], expected: frozenset[str], context: str
) -> None:
    actual = frozenset(value)
    missing = sorted(expected - actual)
    unknown = sorted(actual - expected)
    if missing:
        _schema_fail(f"{context} is missing fields: {', '.join(missing)}")
    if unknown:
        _schema_fail(f"{context} has unknown fields: {', '.join(unknown)}")


def _mapping(value: object, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        _schema_fail(f"{field} must be an object with string keys")
    return cast(Mapping[str, object], value)


def _array(value: object, field: str, *, nonempty: bool = True) -> list[object]:
    if not isinstance(value, list):
        _schema_fail(f"{field} must be an array")
    if nonempty and not value:
        _schema_fail(f"{field} must not be empty")
    return cast(list[object], value)


def _string(value: object, field: str, *, maximum: int = 512) -> str:
    if (
        not isinstance(value, str)
        or not value.strip()
        or len(value) > maximum
        or "\x00" in value
    ):
        _schema_fail(f"{field} must be a non-empty string of at most {maximum} chars")
    return value


def _bundle_logical_path(value: object, field: str) -> str:
    """Validate one portable path relative to a frozen source bundle."""
    text = _string(value, field)
    candidate = PurePosixPath(text)
    if (
        candidate.is_absolute()
        or not candidate.parts
        or candidate.as_posix() != text
        or any(part in {".", ".."} for part in candidate.parts)
        or "\\" in text
    ):
        _schema_fail(f"{field} must be a safe bundle-relative logical path")
    return text


def _lexical_absolute_path(path: Path) -> Path:
    """Expand a caller path without resolving or erasing traversal components."""
    expanded = path.expanduser()
    return expanded if expanded.is_absolute() else Path.cwd() / expanded


def _reject_lexical_path_aliases(path: Path, *, label: str) -> None:
    """Reject aliases visible in the caller-supplied lexical path chain.

    Only components in the expanded absolute lexical path are inspected.  We do
    not walk a second, resolved path and reject aliases that the caller did not
    name; callers that use a platform alias such as macOS ``/tmp`` must instead
    supply its canonical lexical location (for example ``/private/tmp``).
    """
    if any(part in {".", ".."} for part in path.parts):
        _integrity_fail(f"{label} path must not use aliases or traversal")
    cursor = Path(path.anchor)
    for part in path.parts[1:]:
        cursor /= part
        if cursor.is_symlink():
            _integrity_fail(f"{label} path must not contain symbolic-link components")


def _source_bundle_entry_logical_path(
    path: Path,
    *,
    bundle_root: Path,
    label: str,
    require_directory: bool,
) -> str:
    """Validate one source-bundle entry before any source bytes are read."""
    lexical_root = _lexical_absolute_path(bundle_root)
    lexical_path = _lexical_absolute_path(path)
    _reject_lexical_path_aliases(lexical_root, label="reporting source bundle root")
    if not lexical_root.is_dir():
        _integrity_fail("reporting source bundle root must be a regular directory")
    try:
        relative = lexical_path.relative_to(lexical_root)
    except ValueError:
        _integrity_fail(f"{label} is outside the reporting source bundle")
    if not relative.parts:
        _integrity_fail(f"{label} must be inside the reporting source bundle")
    _reject_lexical_path_aliases(lexical_path, label=label)
    if require_directory:
        if not lexical_path.is_dir():
            _integrity_fail(f"{label} must be a regular directory")
    elif not lexical_path.is_file():
        _integrity_fail(f"{label} must be a regular file")
    try:
        resolved_root = lexical_root.resolve(strict=True)
        resolved_path = lexical_path.resolve(strict=True)
        resolved_relative = resolved_path.relative_to(resolved_root)
    except (OSError, ValueError) as error:
        raise ReportingIntegrityError(
            f"{label} cannot be resolved inside the reporting source bundle"
        ) from error
    if resolved_root != lexical_root:
        _integrity_fail(
            "reporting source bundle root path must not use aliases or traversal"
        )
    if resolved_path != lexical_path or resolved_relative.parts != relative.parts:
        _integrity_fail(f"{label} path must not use aliases or traversal")
    logical = PurePosixPath(*relative.parts).as_posix()
    return _bundle_logical_path(logical, f"{label} logical path")


def _source_bundle_logical_path(
    path: Path,
    *,
    bundle_root: Path,
    label: str,
) -> str:
    """Return a path commitment that is stable when a whole bundle moves."""
    return _source_bundle_entry_logical_path(
        path,
        bundle_root=bundle_root,
        label=label,
        require_directory=False,
    )


def _source_bundle_directory_logical_path(
    path: Path,
    *,
    bundle_root: Path,
    label: str,
) -> str:
    """Validate a directory entry in a frozen reporting source bundle."""
    return _source_bundle_entry_logical_path(
        path,
        bundle_root=bundle_root,
        label=label,
        require_directory=True,
    )


def _validate_reporting_result_format(path: Path) -> None:
    """Reject unsupported result containers without opening the source file."""
    if path.suffix.lower() == ".parquet":
        _schema_fail(
            "Parquet input is blocked: no pinned Parquet parser is available in the "
            "environment lock"
        )
    if path.suffix.lower() != ".json":
        _schema_fail("validated reporting results must use strict JSON")


def _identifier(value: object, field: str) -> str:
    text = _string(value, field, maximum=128)
    if _IDENTIFIER.fullmatch(text) is None:
        _schema_fail(f"{field} must be a stable identifier")
    return text


def _sha256(value: object, field: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        _schema_fail(f"{field} must be a lowercase SHA-256 digest")
    return value


def _optional_sha256(value: object, field: str) -> str | None:
    if value is None:
        return None
    return _sha256(value, field)


def _bool(value: object, field: str) -> bool:
    if not isinstance(value, bool):
        _schema_fail(f"{field} must be a boolean")
    return value


def _optional_bool(value: object, field: str) -> bool | None:
    if value is None:
        return None
    return _bool(value, field)


def _number(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        _schema_fail(f"{field} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        _schema_fail(f"{field} must be finite")
    return result


def _nonnegative(value: object, field: str) -> float:
    result = _number(value, field)
    if result < 0.0:
        _schema_fail(f"{field} must be non-negative")
    return result


def _fraction(value: object, field: str) -> float:
    result = _number(value, field)
    if not 0.0 <= result <= 1.0:
        _schema_fail(f"{field} must lie in [0, 1]")
    return result


def _open_fraction(value: object, field: str) -> float:
    result = _fraction(value, field)
    if result <= 0.0 or result >= 1.0:
        _schema_fail(f"{field} must lie strictly inside (0, 1)")
    return result


def _positive_int(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        _schema_fail(f"{field} must be a positive integer")
    return value


def _strict_rows(
    value: object,
    *,
    section: str,
    fields: frozenset[str],
) -> tuple[Mapping[str, object], ...]:
    rows = _array(value, section)
    parsed: list[Mapping[str, object]] = []
    for index, raw in enumerate(rows):
        row = _mapping(raw, f"{section}[{index}]")
        _fields(row, fields, f"{section}[{index}]")
        parsed.append(row)
    return tuple(parsed)


def _coverage_width_rows(
    value: object, *, allow_multiple_series: bool = False
) -> tuple[Mapping[str, object], ...]:
    rows = _strict_rows(
        value,
        section="coverage_width",
        fields=_COVERAGE_WIDTH_FIELDS,
    )
    first_identity: tuple[str, ...] | None = None
    levels_by_identity: dict[tuple[str, ...], list[float]] = {}
    for index, row in enumerate(rows):
        prefix = f"coverage_width[{index}]"
        _string(row["label"], f"{prefix}.label")
        identity = tuple(
            _identifier(row[field], f"{prefix}.{field}")
            for field in _COVERAGE_SERIES_IDENTITY_FIELDS
        )
        if first_identity is None:
            first_identity = identity
        elif identity != first_identity and not allow_multiple_series:
            _schema_fail(
                "coverage_width rows must describe one registered interval series"
            )
        try:
            interval_kind = IntervalKind(identity[4])
        except ValueError as error:
            raise ReportingSchemaError(
                f"{prefix}.interval_kind is unsupported"
            ) from error
        interval_method_id = identity[5]
        expected_kind = _INTERVAL_METHOD_KIND.get(interval_method_id)
        if expected_kind is None:
            _schema_fail(f"{prefix}.interval_method_id is unsupported")
        if interval_kind is not expected_kind:
            _schema_fail(f"{prefix} interval method is incompatible with interval kind")
        level = _open_fraction(row["level"], f"{prefix}.level")
        _fraction(row["coverage"], f"{prefix}.coverage")
        _nonnegative(row["width"], f"{prefix}.width")
        _positive_int(row["valid_draw_count"], f"{prefix}.valid_draw_count")
        levels = levels_by_identity.setdefault(identity, [])
        if level in levels:
            _schema_fail("coverage_width levels must be unique within each series")
        if levels and level <= levels[-1]:
            _schema_fail(
                "coverage_width levels must be strictly increasing within each series"
            )
        levels.append(level)
    return rows


def _validate_formal_coverage_width(
    statistics: Mapping[str, object],
    protocol: StudyProtocol,
    *,
    expected_output_ids: tuple[str, ...],
) -> None:
    rows = _coverage_width_rows(
        statistics["coverage_width"], allow_multiple_series=True
    )
    registered = {output.output_id: output for output in protocol.outputs}
    required_primary_ids = tuple(
        output.output_id
        for output in protocol.outputs
        if output.role is OutputRole.PRIMARY and output.interval_method != "none"
    )
    if expected_output_ids != required_primary_ids:
        _integrity_fail(
            "coverage_width figure specification must freeze every primary "
            "registered output with intervals in protocol order"
        )
    observed_output_ids = tuple(
        dict.fromkeys(cast(str, row["registered_output_id"]) for row in rows)
    )
    if observed_output_ids != expected_output_ids:
        _integrity_fail(
            "coverage_width registered output set does not match the frozen "
            "figure specification"
        )
    observed_levels: dict[str, list[float]] = {}
    for index, row in enumerate(rows):
        prefix = f"coverage_width[{index}]"
        output_id = _identifier(
            row["registered_output_id"], f"{prefix}.registered_output_id"
        )
        output = registered.get(output_id)
        if output is None:
            _integrity_fail(
                f"{prefix} cannot be matched to a registered protocol output"
            )
        bindings = (
            ("metric_id", output.metric_id),
            ("cohort_id", output.cohort_id),
            ("horizon_id", output.horizon_id),
            ("interval_method_id", output.interval_method),
            ("estimand_id", protocol.estimand.definition_id),
        )
        for field, expected in bindings:
            if row[field] != expected:
                _integrity_fail(
                    f"{prefix}.{field} does not match RegisteredOutput {output_id}"
                )
        level = _open_fraction(row["level"], f"{prefix}.level")
        if level not in output.interval_levels:
            _integrity_fail(f"{prefix}.level is not registered for output {output_id}")
        observed_levels.setdefault(output_id, []).append(level)
    for output_id, levels in observed_levels.items():
        expected_levels = registered[output_id].interval_levels
        if tuple(levels) != expected_levels:
            _integrity_fail(
                f"coverage_width levels do not exhaust RegisteredOutput {output_id}"
            )


def _ablation_rows(value: object) -> tuple[Mapping[str, object], ...]:
    rows = _strict_rows(
        value,
        section="ablation_table",
        fields=_ABLATION_FIELDS,
    )
    mechanisms: list[str] = []
    first_identity: tuple[object, ...] | None = None
    for index, row in enumerate(rows):
        prefix = f"ablation_table[{index}]"
        mechanisms.append(_identifier(row["mechanism"], f"{prefix}.mechanism"))
        _number(row["estimate"], f"{prefix}.estimate")
        _nonnegative(row["uncertainty"], f"{prefix}.uncertainty")
        kind = _identifier(row["uncertainty_kind"], f"{prefix}.uncertainty_kind")
        method_id = _identifier(row["method_id"], f"{prefix}.method_id")
        _identifier(row["mcse_spec_id"], f"{prefix}.mcse_spec_id")
        semantics = _ABLATION_METHOD_SEMANTICS.get(method_id)
        if semantics is None:
            _schema_fail(f"{prefix}.method_id is unsupported")
        expected_kind, requires_level = semantics
        if kind != expected_kind:
            _schema_fail(f"{prefix} uncertainty method is incompatible with its kind")
        raw_level = row["level"]
        if requires_level:
            if raw_level is None:
                _schema_fail(f"{prefix}.level is required for interval uncertainty")
            level: float | None = _open_fraction(raw_level, f"{prefix}.level")
        else:
            if raw_level is not None:
                _schema_fail(
                    f"{prefix}.level must be null for Monte Carlo standard error"
                )
            level = None
        sampling_unit = _identifier(row["sampling_unit"], f"{prefix}.sampling_unit")
        estimand_id = _identifier(row["estimand_id"], f"{prefix}.estimand_id")
        _positive_int(row["valid_draw_count"], f"{prefix}.valid_draw_count")
        identity = (kind, method_id, level, sampling_unit, estimand_id)
        if first_identity is None:
            first_identity = identity
        elif identity != first_identity:
            _schema_fail(
                "ablation_table rows must use one comparable uncertainty definition"
            )
    if len(set(mechanisms)) != len(mechanisms):
        _schema_fail("ablation_table mechanism IDs must be unique")
    return rows


def _group_benefit_harm_rows(
    value: object,
) -> tuple[Mapping[str, object], ...]:
    rows = _strict_rows(
        value,
        section="group_benefit_harm",
        fields=_GROUP_BENEFIT_HARM_FIELDS,
    )
    group_ids: list[str] = []
    for index, row in enumerate(rows):
        prefix = f"group_benefit_harm[{index}]"
        group_id = _identifier(row["group"], f"{prefix}.group")
        group_ids.append(group_id)
        raw_basis = _string(row["cohort_basis"], f"{prefix}.cohort_basis")
        try:
            basis = CohortBasis(raw_basis)
        except ValueError as error:
            raise ReportingSchemaError(
                f"{prefix}.cohort_basis is unsupported"
            ) from error
        if basis is CohortBasis.POST_TREATMENT_DESCRIPTIVE:
            _schema_fail(
                f"{prefix} cannot present a post-treatment descriptive cohort "
                "as a benefit/harm group"
            )
        _sha256(
            row["cohort_definition_sha256"],
            f"{prefix}.cohort_definition_sha256",
        )
        _fraction(
            row["benefit_probability"],
            f"{prefix}.benefit_probability",
        )
        _fraction(
            row["harm_probability"],
            f"{prefix}.harm_probability",
        )
    if len(set(group_ids)) != len(group_ids):
        _schema_fail("group_benefit_harm group IDs must be unique")
    return rows


def _validate_formal_group_cohorts(
    statistics: Mapping[str, object], protocol: StudyProtocol
) -> None:
    rows = _group_benefit_harm_rows(statistics["group_benefit_harm"])
    registered = {cohort.cohort_id: cohort for cohort in protocol.cohorts}
    for index, row in enumerate(rows):
        group_id = cast(str, row["group"])
        cohort = registered.get(group_id)
        if cohort is None:
            _integrity_fail(
                f"group_benefit_harm[{index}] references an unregistered cohort"
            )
        if row["cohort_basis"] != cohort.basis.value:
            _integrity_fail(
                f"group_benefit_harm[{index}] cohort basis commitment mismatch"
            )
        if row["cohort_definition_sha256"] != cohort.definition_sha256:
            _integrity_fail(
                f"group_benefit_harm[{index}] cohort definition commitment mismatch"
            )


def _validate_result_statistics(
    statistics: Mapping[str, object], *, allow_multiple_coverage_series: bool = False
) -> None:
    _fields(statistics, frozenset(REQUIRED_REPORT_OUTPUTS), "result statistics")

    paired = _strict_rows(
        statistics["paired_distribution"],
        section="paired_distribution",
        fields=frozenset({"difference", "density"}),
    )
    for index, row in enumerate(paired):
        _number(row["difference"], f"paired_distribution[{index}].difference")
        _nonnegative(row["density"], f"paired_distribution[{index}].density")

    predicted = _strict_rows(
        statistics["predicted_vs_empirical"],
        section="predicted_vs_empirical",
        fields=frozenset({"label", "predicted", "empirical"}),
    )
    for index, row in enumerate(predicted):
        _string(row["label"], f"predicted_vs_empirical[{index}].label")
        _number(row["predicted"], f"predicted_vs_empirical[{index}].predicted")
        _number(row["empirical"], f"predicted_vs_empirical[{index}].empirical")

    _coverage_width_rows(
        statistics["coverage_width"],
        allow_multiple_series=allow_multiple_coverage_series,
    )

    _group_benefit_harm_rows(statistics["group_benefit_harm"])

    _ablation_rows(statistics["ablation_table"])

    calibration = _strict_rows(
        statistics["calibration_diagnostics"],
        section="calibration_diagnostics",
        fields=frozenset({"statistic", "observed", "predicted"}),
    )
    for index, row in enumerate(calibration):
        _string(row["statistic"], f"calibration_diagnostics[{index}].statistic")
        _number(row["observed"], f"calibration_diagnostics[{index}].observed")
        _number(row["predicted"], f"calibration_diagnostics[{index}].predicted")

    risk = _strict_rows(
        statistics["risk_coverage"],
        section="risk_coverage",
        fields=frozenset({"coverage", "risk"}),
    )
    for index, row in enumerate(risk):
        _fraction(row["coverage"], f"risk_coverage[{index}].coverage")
        _nonnegative(row["risk"], f"risk_coverage[{index}].risk")

    heatmap = _strict_rows(
        statistics["certificate_heatmap"],
        section="certificate_heatmap",
        fields=frozenset({"row", "column", "value"}),
    )
    seen_cells: set[tuple[str, str]] = set()
    for index, item in enumerate(heatmap):
        row_name = _string(item["row"], f"certificate_heatmap[{index}].row")
        column = _string(item["column"], f"certificate_heatmap[{index}].column")
        _fraction(item["value"], f"certificate_heatmap[{index}].value")
        if (row_name, column) in seen_cells:
            _schema_fail("certificate_heatmap contains a duplicate cell")
        seen_cells.add((row_name, column))
    rows = {row for row, _ in seen_cells}
    columns = {column for _, column in seen_cells}
    if len(seen_cells) != len(rows) * len(columns):
        _schema_fail("certificate_heatmap must provide a complete rectangular grid")

    scaling = _strict_rows(
        statistics["compute_scaling_table"],
        section="compute_scaling_table",
        fields=frozenset({"workers", "events", "wall_seconds"}),
    )
    for index, row in enumerate(scaling):
        _positive_int(row["workers"], f"compute_scaling_table[{index}].workers")
        _positive_int(row["events"], f"compute_scaling_table[{index}].events")
        _nonnegative(
            row["wall_seconds"], f"compute_scaling_table[{index}].wall_seconds"
        )


@dataclass(frozen=True, slots=True)
class FigureOutputSpec:
    """One frozen output name, type, and human-readable title."""

    output_id: str
    filename: str
    kind: str
    title: str
    registered_output_ids: tuple[str, ...] = ()

    @classmethod
    def from_mapping(
        cls, mapping: Mapping[str, object], *, formal: bool = False
    ) -> Self:
        output_id = _identifier(mapping.get("output_id"), "output_id")
        expected_fields = {"output_id", "filename", "kind", "title"}
        if formal and output_id == "coverage_width":
            expected_fields.add("registered_output_ids")
        _fields(
            mapping,
            frozenset(expected_fields),
            "figure output specification",
        )
        if output_id not in _OUTPUT_KINDS:
            _schema_fail(f"unsupported reporting output_id: {output_id}")
        kind = _string(mapping["kind"], "output kind", maximum=16)
        if kind != _OUTPUT_KINDS[output_id]:
            _schema_fail(
                f"output kind for {output_id} must be {_OUTPUT_KINDS[output_id]}"
            )
        filename = _string(mapping["filename"], "output filename", maximum=128)
        path = PurePosixPath(filename)
        extension = ".svg" if kind == "svg" else ".tex"
        if (
            path.is_absolute()
            or len(path.parts) != 1
            or path.as_posix() != filename
            or path.suffix != extension
            or filename in {".", "..", "artifact_manifest.json"}
        ):
            _schema_fail(
                "output filename must be a safe basename with the expected suffix"
            )
        registered_output_ids: tuple[str, ...] = ()
        if formal and output_id == "coverage_width":
            registered_output_ids = tuple(
                _identifier(value, "coverage registered_output_ids item")
                for value in _array(
                    mapping["registered_output_ids"],
                    "coverage registered_output_ids",
                )
            )
            if len(set(registered_output_ids)) != len(registered_output_ids):
                _schema_fail(
                    "coverage registered_output_ids must not contain duplicates"
                )
        return cls(
            output_id=output_id,
            filename=filename,
            kind=kind,
            title=_string(mapping["title"], "output title", maximum=256),
            registered_output_ids=registered_output_ids,
        )


@dataclass(frozen=True, slots=True)
class GeneratedPaperArtifact:
    """One rendered output and all of its upstream commitments."""

    output_id: str
    path: str
    media_type: str
    sha256: str
    source_artifact_id: str
    source_artifact_sha256: str
    experiment_registry_sha256: str
    figure_spec_sha256: str
    config_sha256: str
    study_protocol_sha256: str | None
    matrix_run_manifest_sha256: str | None
    statistics_derivation_sha256: str | None
    validity_certificate_sha256: str | None

    def to_mapping(self) -> dict[str, object]:
        return {
            "config_sha256": self.config_sha256,
            "experiment_registry_sha256": self.experiment_registry_sha256,
            "figure_spec_sha256": self.figure_spec_sha256,
            "matrix_run_manifest_sha256": self.matrix_run_manifest_sha256,
            "media_type": self.media_type,
            "output_id": self.output_id,
            "path": self.path,
            "sha256": self.sha256,
            "source_artifact_id": self.source_artifact_id,
            "source_artifact_sha256": self.source_artifact_sha256,
            "study_protocol_sha256": self.study_protocol_sha256,
            "statistics_derivation_sha256": self.statistics_derivation_sha256,
            "validity_certificate_sha256": self.validity_certificate_sha256,
        }

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, object]) -> Self:
        """Parse an exact-schema generated-artifact commitment."""
        _fields(
            mapping,
            frozenset(
                {
                    "config_sha256",
                    "experiment_registry_sha256",
                    "figure_spec_sha256",
                    "matrix_run_manifest_sha256",
                    "media_type",
                    "output_id",
                    "path",
                    "sha256",
                    "source_artifact_id",
                    "source_artifact_sha256",
                    "study_protocol_sha256",
                    "statistics_derivation_sha256",
                    "validity_certificate_sha256",
                }
            ),
            "generated paper artifact",
        )
        output_id = _identifier(mapping["output_id"], "generated output_id")
        if output_id not in _OUTPUT_KINDS:
            _schema_fail("generated paper artifact output_id is unsupported")
        path = _string(mapping["path"], "generated artifact path", maximum=128)
        candidate = PurePosixPath(path)
        expected_suffix = ".svg" if _OUTPUT_KINDS[output_id] == "svg" else ".tex"
        if (
            candidate.is_absolute()
            or len(candidate.parts) != 1
            or candidate.as_posix() != path
            or path in {".", "..", "artifact_manifest.json"}
            or candidate.suffix != expected_suffix
        ):
            _schema_fail("generated artifact path must be a safe basename")
        media_type = _string(mapping["media_type"], "generated media_type")
        expected_media_type = (
            "image/svg+xml"
            if _OUTPUT_KINDS[output_id] == "svg"
            else "application/x-tex"
        )
        if media_type != expected_media_type:
            _schema_fail("generated artifact media type disagrees with output kind")
        return cls(
            output_id=output_id,
            path=path,
            media_type=media_type,
            sha256=_sha256(mapping["sha256"], "generated artifact sha256"),
            source_artifact_id=_identifier(
                mapping["source_artifact_id"], "generated source_artifact_id"
            ),
            source_artifact_sha256=_sha256(
                mapping["source_artifact_sha256"],
                "generated source_artifact_sha256",
            ),
            experiment_registry_sha256=_sha256(
                mapping["experiment_registry_sha256"],
                "generated experiment_registry_sha256",
            ),
            figure_spec_sha256=_sha256(
                mapping["figure_spec_sha256"], "generated figure_spec_sha256"
            ),
            config_sha256=_sha256(mapping["config_sha256"], "generated config_sha256"),
            study_protocol_sha256=_optional_sha256(
                mapping["study_protocol_sha256"],
                "generated study_protocol_sha256",
            ),
            matrix_run_manifest_sha256=_optional_sha256(
                mapping["matrix_run_manifest_sha256"],
                "generated matrix_run_manifest_sha256",
            ),
            statistics_derivation_sha256=_optional_sha256(
                mapping["statistics_derivation_sha256"],
                "generated statistics_derivation_sha256",
            ),
            validity_certificate_sha256=_optional_sha256(
                mapping["validity_certificate_sha256"],
                "generated validity_certificate_sha256",
            ),
        )


@dataclass(frozen=True, slots=True)
class PaperArtifactManifest:
    """Complete traceability record for one reporting-only render."""

    synthetic_only: bool
    config_sha256: str
    result_path: str
    registry_path: str
    figure_spec_path: str
    result_sha256: str
    registry_sha256: str
    figure_spec_sha256: str
    reporting_eligibility: str
    formal_reporting_eligible: bool
    allowed_claim_scope: str | None
    study_protocol_path: str | None
    study_protocol_source_sha256: str | None
    study_protocol_sha256: str | None
    outcome_knowledge_manifest_path: str | None
    outcome_knowledge_manifest_source_sha256: str | None
    outcome_knowledge_manifest_semantic_sha256: str | None
    design_temporality_decision_sha256: str | None
    derived_design_temporality: str | None
    statistics_derivation_path: str | None
    statistics_derivation_sha256: str | None
    matrix_sha256: str | None
    matrix_run_manifest_sha256: str | None
    validity_certificate_sha256: str | None
    git_commit: str | None
    environment_lock_sha256: str | None
    data_manifest_sha256: str | None
    model_checkpoint_sha256: str | None
    event_key_manifest_sha256: str | None
    external_preregistration_receipt_sha256: str | None
    preregistration_timestamp_verified: bool | None
    primary_policy_sources_verified: bool | None
    data_licenses_verified: bool | None
    post_policy_inputs_frozen: bool | None
    empirical_identification_audit_sha256: str | None
    empirical_identification_passed: bool | None
    outputs: tuple[GeneratedPaperArtifact, ...]

    def __post_init__(self) -> None:
        for field in ("result_path", "registry_path", "figure_spec_path"):
            _bundle_logical_path(getattr(self, field), field)
        for field in (
            "study_protocol_path",
            "outcome_knowledge_manifest_path",
            "statistics_derivation_path",
        ):
            value = getattr(self, field)
            if value is not None:
                _bundle_logical_path(value, field)
        if self.reporting_eligibility not in {
            "formal_audited",
            "synthetic_unverified",
        }:
            _schema_fail("paper manifest reporting_eligibility is invalid")
        if self.formal_reporting_eligible != (
            self.reporting_eligibility == "formal_audited"
        ):
            _schema_fail("paper manifest formal eligibility is inconsistent")
        if self.reporting_eligibility == "synthetic_unverified":
            if not self.synthetic_only:
                _schema_fail("an unverified reporting manifest must be synthetic")
            if self.allowed_claim_scope is not None:
                _schema_fail(
                    "unverified synthetic output cannot claim an eligible scope"
                )
            provenance = (
                self.study_protocol_path,
                self.study_protocol_source_sha256,
                self.study_protocol_sha256,
                self.outcome_knowledge_manifest_path,
                self.outcome_knowledge_manifest_source_sha256,
                self.outcome_knowledge_manifest_semantic_sha256,
                self.design_temporality_decision_sha256,
                self.derived_design_temporality,
                self.matrix_sha256,
                self.matrix_run_manifest_sha256,
                self.statistics_derivation_path,
                self.statistics_derivation_sha256,
                self.validity_certificate_sha256,
                self.git_commit,
                self.environment_lock_sha256,
                self.data_manifest_sha256,
                self.model_checkpoint_sha256,
                self.event_key_manifest_sha256,
                self.external_preregistration_receipt_sha256,
                self.preregistration_timestamp_verified,
                self.primary_policy_sources_verified,
                self.data_licenses_verified,
                self.post_policy_inputs_frozen,
                self.empirical_identification_audit_sha256,
                self.empirical_identification_passed,
            )
            if any(value is not None for value in provenance):
                _schema_fail(
                    "unverified synthetic output cannot carry formal provenance"
                )
        else:
            required = (
                self.allowed_claim_scope,
                self.study_protocol_path,
                self.study_protocol_source_sha256,
                self.study_protocol_sha256,
                self.matrix_sha256,
                self.matrix_run_manifest_sha256,
                self.statistics_derivation_path,
                self.statistics_derivation_sha256,
                self.validity_certificate_sha256,
                self.git_commit,
                self.environment_lock_sha256,
                self.data_manifest_sha256,
                self.model_checkpoint_sha256,
                self.event_key_manifest_sha256,
                self.preregistration_timestamp_verified,
                self.primary_policy_sources_verified,
                self.data_licenses_verified,
                self.post_policy_inputs_frozen,
                self.empirical_identification_passed,
            )
            if any(value is None for value in required):
                _schema_fail("formal paper manifest is missing required provenance")
            if self.allowed_claim_scope not in {item.value for item in ClaimScope}:
                _schema_fail("formal paper manifest claim scope is invalid")
            if self.allowed_claim_scope == ClaimScope.SYNTHETIC_ONLY.value:
                if self.synthetic_only is not True:
                    _schema_fail(
                        "formal synthetic claim scope requires synthetic-only output"
                    )
            elif self.synthetic_only is not False:
                _schema_fail("formal real-study claim scope cannot be synthetic-only")
            if (
                self.git_commit is None
                or _GIT_COMMIT.fullmatch(self.git_commit) is None
            ):
                _schema_fail("formal paper manifest Git commit is invalid")
            if self.allowed_claim_scope != ClaimScope.SYNTHETIC_ONLY.value and (
                self.external_preregistration_receipt_sha256 is None
                or self.preregistration_timestamp_verified is not True
                or self.primary_policy_sources_verified is not True
                or self.data_licenses_verified is not True
                or self.post_policy_inputs_frozen is not True
            ):
                _schema_fail(
                    "formal real-study manifest lacks required verified evidence"
                )
            outcome_provenance = (
                self.outcome_knowledge_manifest_path,
                self.outcome_knowledge_manifest_source_sha256,
                self.outcome_knowledge_manifest_semantic_sha256,
                self.design_temporality_decision_sha256,
                self.derived_design_temporality,
            )
            if self.allowed_claim_scope == ClaimScope.SYNTHETIC_ONLY.value:
                if any(value is not None for value in outcome_provenance):
                    _schema_fail(
                        "formal synthetic manifest cannot carry real-event outcome "
                        "knowledge provenance"
                    )
            elif any(value is None for value in outcome_provenance):
                _schema_fail(
                    "formal real-study manifest lacks outcome-knowledge provenance"
                )
            elif (
                self.derived_design_temporality
                == DesignTemporality.UNVERIFIED_ABSTAIN.value
            ):
                _schema_fail(
                    "formal real-study manifest cannot authorize an unverified "
                    "outcome-knowledge chronology"
                )
            elif self.allowed_claim_scope != ClaimScope.SYNTHETIC_ONLY.value:
                _schema_fail(
                    "standalone formal real-study manifests are not supported "
                    "without a typed external verifier"
                )
            if self.allowed_claim_scope == ClaimScope.REAL_CAUSAL.value and (
                self.empirical_identification_audit_sha256 is None
                or self.empirical_identification_passed is not True
            ):
                _schema_fail("formal causal manifest lacks an identification audit")
        for field in (
            "result_sha256",
            "registry_sha256",
            "figure_spec_sha256",
            "config_sha256",
        ):
            _sha256(getattr(self, field), field)
        for field in (
            "study_protocol_source_sha256",
            "study_protocol_sha256",
            "outcome_knowledge_manifest_source_sha256",
            "outcome_knowledge_manifest_semantic_sha256",
            "design_temporality_decision_sha256",
            "matrix_sha256",
            "matrix_run_manifest_sha256",
            "statistics_derivation_sha256",
            "validity_certificate_sha256",
            "environment_lock_sha256",
            "data_manifest_sha256",
            "model_checkpoint_sha256",
            "event_key_manifest_sha256",
            "external_preregistration_receipt_sha256",
            "empirical_identification_audit_sha256",
        ):
            _optional_sha256(getattr(self, field), field)
        if self.derived_design_temporality is not None:
            try:
                DesignTemporality(self.derived_design_temporality)
            except ValueError as error:
                raise ReportingSchemaError(
                    "paper manifest derived design temporality is invalid"
                ) from error
        for field in (
            "preregistration_timestamp_verified",
            "primary_policy_sources_verified",
            "data_licenses_verified",
            "post_policy_inputs_frozen",
            "empirical_identification_passed",
        ):
            value = getattr(self, field)
            if value is not None and not isinstance(value, bool):
                _schema_fail(f"{field} must be boolean or null")
        if not isinstance(self.outputs, tuple) or len(self.outputs) != len(
            REQUIRED_REPORT_OUTPUTS
        ):
            _schema_fail("paper manifest must contain all nine outputs")
        if tuple(item.output_id for item in self.outputs) != REQUIRED_REPORT_OUTPUTS:
            _schema_fail("paper manifest output order is invalid")
        paths = tuple(item.path for item in self.outputs)
        if len(set(paths)) != len(paths):
            _schema_fail("paper manifest output paths must be unique")
        if len({item.source_artifact_id for item in self.outputs}) != 1:
            _schema_fail("paper manifest outputs reference different source artifacts")

    @property
    def schema_version(self) -> str:
        return REPORT_MANIFEST_SCHEMA

    def to_mapping(self) -> dict[str, object]:
        return {
            "config_sha256": self.config_sha256,
            "figure_spec_path": self.figure_spec_path,
            "figure_spec_sha256": self.figure_spec_sha256,
            "formal_reporting_eligible": self.formal_reporting_eligible,
            "allowed_claim_scope": self.allowed_claim_scope,
            "data_manifest_sha256": self.data_manifest_sha256,
            "data_licenses_verified": self.data_licenses_verified,
            "empirical_identification_audit_sha256": (
                self.empirical_identification_audit_sha256
            ),
            "empirical_identification_passed": self.empirical_identification_passed,
            "environment_lock_sha256": self.environment_lock_sha256,
            "event_key_manifest_sha256": self.event_key_manifest_sha256,
            "external_preregistration_receipt_sha256": (
                self.external_preregistration_receipt_sha256
            ),
            "git_commit": self.git_commit,
            "matrix_run_manifest_sha256": self.matrix_run_manifest_sha256,
            "matrix_sha256": self.matrix_sha256,
            "model_checkpoint_sha256": self.model_checkpoint_sha256,
            "post_policy_inputs_frozen": self.post_policy_inputs_frozen,
            "preregistration_timestamp_verified": (
                self.preregistration_timestamp_verified
            ),
            "primary_policy_sources_verified": self.primary_policy_sources_verified,
            "outputs": [item.to_mapping() for item in self.outputs],
            "reporting_eligibility": self.reporting_eligibility,
            "registry_path": self.registry_path,
            "registry_sha256": self.registry_sha256,
            "result_path": self.result_path,
            "result_sha256": self.result_sha256,
            "schema_version": self.schema_version,
            "status": "complete",
            "study_protocol_path": self.study_protocol_path,
            "study_protocol_sha256": self.study_protocol_sha256,
            "study_protocol_source_sha256": self.study_protocol_source_sha256,
            "outcome_knowledge_manifest_path": self.outcome_knowledge_manifest_path,
            "outcome_knowledge_manifest_source_sha256": (
                self.outcome_knowledge_manifest_source_sha256
            ),
            "outcome_knowledge_manifest_semantic_sha256": (
                self.outcome_knowledge_manifest_semantic_sha256
            ),
            "design_temporality_decision_sha256": (
                self.design_temporality_decision_sha256
            ),
            "derived_design_temporality": self.derived_design_temporality,
            "statistics_derivation_path": self.statistics_derivation_path,
            "statistics_derivation_sha256": self.statistics_derivation_sha256,
            "synthetic_only": self.synthetic_only,
            "validity_certificate_sha256": self.validity_certificate_sha256,
        }

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, object]) -> Self:
        """Parse a strict portable paper manifest for post-render auditing."""
        _fields(
            mapping,
            frozenset(
                {
                    "allowed_claim_scope",
                    "config_sha256",
                    "data_manifest_sha256",
                    "data_licenses_verified",
                    "empirical_identification_audit_sha256",
                    "empirical_identification_passed",
                    "environment_lock_sha256",
                    "event_key_manifest_sha256",
                    "external_preregistration_receipt_sha256",
                    "figure_spec_path",
                    "figure_spec_sha256",
                    "formal_reporting_eligible",
                    "git_commit",
                    "matrix_run_manifest_sha256",
                    "matrix_sha256",
                    "model_checkpoint_sha256",
                    "post_policy_inputs_frozen",
                    "preregistration_timestamp_verified",
                    "primary_policy_sources_verified",
                    "outputs",
                    "registry_path",
                    "registry_sha256",
                    "reporting_eligibility",
                    "result_path",
                    "result_sha256",
                    "schema_version",
                    "status",
                    "study_protocol_path",
                    "study_protocol_sha256",
                    "study_protocol_source_sha256",
                    "outcome_knowledge_manifest_path",
                    "outcome_knowledge_manifest_source_sha256",
                    "outcome_knowledge_manifest_semantic_sha256",
                    "design_temporality_decision_sha256",
                    "derived_design_temporality",
                    "statistics_derivation_path",
                    "statistics_derivation_sha256",
                    "synthetic_only",
                    "validity_certificate_sha256",
                }
            ),
            "paper artifact manifest",
        )
        if mapping["schema_version"] != REPORT_MANIFEST_SCHEMA:
            _schema_fail("paper artifact manifest schema version is unsupported")
        if mapping["status"] != "complete":
            _integrity_fail("paper artifact manifest is not complete")

        def optional_text(field: str) -> str | None:
            value = mapping[field]
            return None if value is None else _string(value, field)

        outputs = tuple(
            GeneratedPaperArtifact.from_mapping(_mapping(raw, "paper manifest output"))
            for raw in _array(mapping["outputs"], "paper manifest outputs")
        )
        return cls(
            synthetic_only=_bool(mapping["synthetic_only"], "synthetic_only"),
            config_sha256=_sha256(mapping["config_sha256"], "config_sha256"),
            result_path=_string(mapping["result_path"], "result_path"),
            registry_path=_string(mapping["registry_path"], "registry_path"),
            figure_spec_path=_string(mapping["figure_spec_path"], "figure_spec_path"),
            result_sha256=_sha256(mapping["result_sha256"], "result_sha256"),
            registry_sha256=_sha256(mapping["registry_sha256"], "registry_sha256"),
            figure_spec_sha256=_sha256(
                mapping["figure_spec_sha256"], "figure_spec_sha256"
            ),
            reporting_eligibility=_string(
                mapping["reporting_eligibility"], "reporting_eligibility"
            ),
            formal_reporting_eligible=_bool(
                mapping["formal_reporting_eligible"],
                "formal_reporting_eligible",
            ),
            allowed_claim_scope=optional_text("allowed_claim_scope"),
            study_protocol_path=optional_text("study_protocol_path"),
            study_protocol_source_sha256=_optional_sha256(
                mapping["study_protocol_source_sha256"],
                "study_protocol_source_sha256",
            ),
            study_protocol_sha256=_optional_sha256(
                mapping["study_protocol_sha256"], "study_protocol_sha256"
            ),
            outcome_knowledge_manifest_path=optional_text(
                "outcome_knowledge_manifest_path"
            ),
            outcome_knowledge_manifest_source_sha256=_optional_sha256(
                mapping["outcome_knowledge_manifest_source_sha256"],
                "outcome_knowledge_manifest_source_sha256",
            ),
            outcome_knowledge_manifest_semantic_sha256=_optional_sha256(
                mapping["outcome_knowledge_manifest_semantic_sha256"],
                "outcome_knowledge_manifest_semantic_sha256",
            ),
            design_temporality_decision_sha256=_optional_sha256(
                mapping["design_temporality_decision_sha256"],
                "design_temporality_decision_sha256",
            ),
            derived_design_temporality=optional_text("derived_design_temporality"),
            matrix_sha256=_optional_sha256(mapping["matrix_sha256"], "matrix_sha256"),
            matrix_run_manifest_sha256=_optional_sha256(
                mapping["matrix_run_manifest_sha256"],
                "matrix_run_manifest_sha256",
            ),
            statistics_derivation_path=optional_text("statistics_derivation_path"),
            statistics_derivation_sha256=_optional_sha256(
                mapping["statistics_derivation_sha256"],
                "statistics_derivation_sha256",
            ),
            validity_certificate_sha256=_optional_sha256(
                mapping["validity_certificate_sha256"],
                "validity_certificate_sha256",
            ),
            git_commit=optional_text("git_commit"),
            environment_lock_sha256=_optional_sha256(
                mapping["environment_lock_sha256"], "environment_lock_sha256"
            ),
            data_manifest_sha256=_optional_sha256(
                mapping["data_manifest_sha256"], "data_manifest_sha256"
            ),
            model_checkpoint_sha256=_optional_sha256(
                mapping["model_checkpoint_sha256"], "model_checkpoint_sha256"
            ),
            event_key_manifest_sha256=_optional_sha256(
                mapping["event_key_manifest_sha256"],
                "event_key_manifest_sha256",
            ),
            external_preregistration_receipt_sha256=_optional_sha256(
                mapping["external_preregistration_receipt_sha256"],
                "external_preregistration_receipt_sha256",
            ),
            preregistration_timestamp_verified=_optional_bool(
                mapping["preregistration_timestamp_verified"],
                "preregistration_timestamp_verified",
            ),
            primary_policy_sources_verified=_optional_bool(
                mapping["primary_policy_sources_verified"],
                "primary_policy_sources_verified",
            ),
            data_licenses_verified=_optional_bool(
                mapping["data_licenses_verified"], "data_licenses_verified"
            ),
            post_policy_inputs_frozen=_optional_bool(
                mapping["post_policy_inputs_frozen"], "post_policy_inputs_frozen"
            ),
            empirical_identification_audit_sha256=_optional_sha256(
                mapping["empirical_identification_audit_sha256"],
                "empirical_identification_audit_sha256",
            ),
            empirical_identification_passed=_optional_bool(
                mapping["empirical_identification_passed"],
                "empirical_identification_passed",
            ),
            outputs=outputs,
        )


@dataclass(frozen=True, slots=True)
class FormalReportingTrust:
    """Independent files and auditors required by the formal reporting gate.

    Every ``expected_*`` digest is an external anchor supplied by the caller.
    Recomputing a digest beside a changed file is not independent evidence.
    """

    study_protocol_path: Path
    expected_study_protocol_source_sha256: str
    release_evidence: ReleaseEvidence
    repository_root: Path
    config_path: Path
    expected_config_sha256: str
    environment_lock_path: Path
    expected_environment_lock_sha256: str
    data_manifest_path: Path
    expected_data_manifest_sha256: str
    model_checkpoint_path: Path
    expected_model_checkpoint_sha256: str
    event_key_manifest_path: Path
    expected_event_key_manifest_sha256: str
    matrix: ExperimentMatrix
    matrix_run_directory: Path
    expected_matrix_run_manifest_sha256: str
    validity_certificate_path: Path
    expected_validity_certificate_sha256: str
    statistics_derivation_path: Path | None = None
    expected_statistics_derivation_sha256: str | None = None
    execution_receipt_authority: SyntheticExecutionAuthority | None = None
    outcome_knowledge_manifest_path: Path | None = None
    expected_outcome_knowledge_manifest_source_sha256: str | None = None
    expected_outcome_knowledge_manifest_semantic_sha256: str | None = None

    def __post_init__(self) -> None:
        for field in (
            "expected_study_protocol_source_sha256",
            "expected_config_sha256",
            "expected_environment_lock_sha256",
            "expected_data_manifest_sha256",
            "expected_model_checkpoint_sha256",
            "expected_event_key_manifest_sha256",
            "expected_matrix_run_manifest_sha256",
            "expected_validity_certificate_sha256",
        ):
            _sha256(getattr(self, field), field)
        if (self.statistics_derivation_path is None) != (
            self.expected_statistics_derivation_sha256 is None
        ):
            _schema_fail(
                "formal trust requires both statistics derivation path and digest"
            )
        if self.expected_statistics_derivation_sha256 is not None:
            _sha256(
                self.expected_statistics_derivation_sha256,
                "expected_statistics_derivation_sha256",
            )
        outcome_values = (
            self.outcome_knowledge_manifest_path,
            self.expected_outcome_knowledge_manifest_source_sha256,
            self.expected_outcome_knowledge_manifest_semantic_sha256,
        )
        if any(value is None for value in outcome_values) and any(
            value is not None for value in outcome_values
        ):
            _schema_fail(
                "formal trust requires the outcome manifest path, source digest, "
                "and semantic digest together"
            )
        for field in (
            "expected_outcome_knowledge_manifest_source_sha256",
            "expected_outcome_knowledge_manifest_semantic_sha256",
        ):
            _optional_sha256(getattr(self, field), field)
        if not isinstance(self.release_evidence, ReleaseEvidence):
            _schema_fail("formal trust requires ReleaseEvidence")
        if not isinstance(self.matrix, ExperimentMatrix):
            _schema_fail("formal trust requires an ExperimentMatrix")
        if self.execution_receipt_authority is not None and not isinstance(
            self.execution_receipt_authority, SyntheticExecutionAuthority
        ):
            _schema_fail("formal trust execution_receipt_authority has an invalid type")


@dataclass(frozen=True, slots=True)
class PaperArtifactAuditReport:
    """Positive audit result for a complete, externally anchored render."""

    valid: bool
    manifest_sha256: str
    reporting_eligibility: str
    output_count: int

    def __post_init__(self) -> None:
        if self.valid is not True:
            _schema_fail("a positive paper-artifact audit must have valid=True")
        _sha256(self.manifest_sha256, "paper audit manifest_sha256")
        if self.output_count != len(REQUIRED_REPORT_OUTPUTS):
            _schema_fail("paper audit output_count is invalid")


@dataclass(frozen=True, slots=True)
class _ValidatedReportingInputs:
    result: Mapping[str, object]
    output_specs: tuple[FigureOutputSpec, ...]
    artifact_id: str
    config_sha256: str
    synthetic_only: bool
    result_sha256: str
    registry_sha256: str
    figure_spec_sha256: str
    reporting_eligibility: str
    formal_reporting_eligible: bool
    allowed_claim_scope: str | None
    study_protocol_path: str | None
    study_protocol_source_sha256: str | None
    study_protocol_sha256: str | None
    outcome_knowledge_manifest_path: str | None
    outcome_knowledge_manifest_source_sha256: str | None
    outcome_knowledge_manifest_semantic_sha256: str | None
    design_temporality_decision_sha256: str | None
    derived_design_temporality: str | None
    statistics_derivation_path: str | None
    statistics_derivation_sha256: str | None
    matrix_sha256: str | None
    matrix_run_manifest_sha256: str | None
    validity_certificate_sha256: str | None
    git_commit: str | None
    environment_lock_sha256: str | None
    data_manifest_sha256: str | None
    model_checkpoint_sha256: str | None
    event_key_manifest_sha256: str | None
    external_preregistration_receipt_sha256: str | None
    preregistration_timestamp_verified: bool | None
    primary_policy_sources_verified: bool | None
    data_licenses_verified: bool | None
    post_policy_inputs_frozen: bool | None
    empirical_identification_audit_sha256: str | None
    empirical_identification_passed: bool | None


def _read_committed_json(
    path: Path,
    expected_sha256: str,
    *,
    label: str,
) -> Mapping[str, object]:
    expected = _sha256(expected_sha256, f"expected {label} sha256")
    if path.is_symlink() or not path.is_file():
        _integrity_fail(f"{label} must be a regular non-symlink file")
    try:
        payload = path.read_bytes()
    except OSError as error:
        raise ReportingIntegrityError(f"cannot read {label}: {error}") from error
    observed = sha256_bytes(payload)
    if observed != expected:
        _integrity_fail(f"{label} hash mismatch")
    try:
        return read_json_bytes_object(payload, source=str(path))
    except MetadataValidationError as error:
        raise ReportingSchemaError(f"invalid {label} JSON: {error}") from error


def _validate_legacy_registry_entry(
    registry: Mapping[str, object],
    *,
    registry_path: Path,
    result_path: Path,
    artifact_id: str,
    result_sha256: str,
    config_sha256: str,
    synthetic_only: bool,
) -> None:
    _fields(
        registry,
        frozenset({"schema_version", "registry_id", "artifacts"}),
        "experiment registry",
    )
    if registry["schema_version"] != 1:
        _schema_fail("experiment registry schema_version must be 1")
    _identifier(registry["registry_id"], "registry_id")
    matches = 0
    for index, raw in enumerate(_array(registry["artifacts"], "registry artifacts")):
        entry = _mapping(raw, f"registry artifacts[{index}]")
        _fields(
            entry,
            frozenset(
                {
                    "artifact_id",
                    "relative_path",
                    "sha256",
                    "config_sha256",
                    "validation_status",
                    "synthetic_only",
                }
            ),
            f"registry artifacts[{index}]",
        )
        entry_id = _identifier(entry["artifact_id"], "registry artifact_id")
        relative_path = _string(entry["relative_path"], "registry relative_path")
        candidate = PurePosixPath(relative_path)
        if (
            candidate.is_absolute()
            or candidate.as_posix() != relative_path
            or any(part in {"", ".", ".."} for part in candidate.parts)
        ):
            _schema_fail("registry relative_path must be a safe local relative path")
        entry_sha = _sha256(entry["sha256"], "registry artifact sha256")
        entry_config = _sha256(entry["config_sha256"], "registry config_sha256")
        status = _string(entry["validation_status"], "registry validation_status")
        entry_synthetic = _bool(entry["synthetic_only"], "registry synthetic_only")
        if entry_id != artifact_id:
            continue
        matches += 1
        try:
            registered_path = (registry_path.parent / relative_path).resolve(
                strict=True
            )
            supplied_path = result_path.resolve(strict=True)
        except OSError as error:
            raise ReportingIntegrityError(
                f"cannot resolve registered result path: {error}"
            ) from error
        if registered_path != supplied_path:
            _integrity_fail("supplied result path differs from the experiment registry")
        if entry_sha != result_sha256:
            _integrity_fail("registry result hash differs from the supplied commitment")
        if entry_config != config_sha256:
            _integrity_fail("registry config hash differs from the validated result")
        if status != "validated":
            _integrity_fail("experiment registry has not marked the result validated")
        if entry_synthetic != synthetic_only:
            _integrity_fail("registry/result synthetic markers disagree")
    if matches != 1:
        _integrity_fail("experiment registry must contain exactly one result entry")


def _validate_legacy_figure_spec(
    spec: Mapping[str, object],
    *,
    artifact_id: str,
    result_sha256: str,
    registry_sha256: str,
    config_sha256: str,
    synthetic_only: bool,
) -> tuple[FigureOutputSpec, ...]:
    _fields(
        spec,
        frozenset(
            {
                "schema_version",
                "spec_id",
                "status",
                "synthetic_only",
                "result_artifact_id",
                "result_sha256",
                "experiment_registry_sha256",
                "config_sha256",
                "outputs",
            }
        ),
        "frozen figure specification",
    )
    if spec["schema_version"] != 1:
        _schema_fail("figure specification schema_version must be 1")
    _identifier(spec["spec_id"], "figure spec_id")
    status = _string(spec["status"], "figure status")
    if status != "frozen":
        _integrity_fail("figure specification is blocked or not frozen")
    if _identifier(spec["result_artifact_id"], "result_artifact_id") != artifact_id:
        _integrity_fail("figure specification references another result artifact")
    if _sha256(spec["result_sha256"], "figure result_sha256") != result_sha256:
        _integrity_fail("figure specification result hash mismatch")
    if (
        _sha256(
            spec["experiment_registry_sha256"],
            "figure experiment_registry_sha256",
        )
        != registry_sha256
    ):
        _integrity_fail("figure specification registry hash mismatch")
    if _sha256(spec["config_sha256"], "figure config_sha256") != config_sha256:
        _integrity_fail("figure specification config hash mismatch")
    if _bool(spec["synthetic_only"], "figure synthetic_only") != synthetic_only:
        _integrity_fail("figure/result synthetic markers disagree")
    outputs = tuple(
        FigureOutputSpec.from_mapping(_mapping(raw, "figure output"))
        for raw in _array(spec["outputs"], "figure outputs")
    )
    ids = tuple(item.output_id for item in outputs)
    filenames = tuple(item.filename for item in outputs)
    if ids != REQUIRED_REPORT_OUTPUTS:
        _schema_fail(
            "figure outputs must contain all nine required artifacts in canonical order"
        )
    if len(set(filenames)) != len(filenames):
        _schema_fail("figure output filenames must be unique")
    return outputs


def _verify_committed_file(path: Path, expected_sha256: str, *, label: str) -> str:
    expected = _sha256(expected_sha256, f"expected {label} sha256")
    if path.is_symlink() or not path.is_file():
        _integrity_fail(f"{label} must be a regular non-symlink file")
    try:
        observed = sha256_file(path)
    except OSError as error:
        raise ReportingIntegrityError(f"cannot hash {label}: {error}") from error
    if observed != expected:
        _integrity_fail(f"{label} hash mismatch")
    return observed


_VALIDITY_CHECK_IDS = frozenset(
    {
        "temporal_leakage",
        "pre_policy_posterior_predictive",
        "ood_policy_distance",
        "model_disagreement",
        "seed_instability",
        "prompt_instability",
        "parameter_support_failure",
        "empirical_coverage_history",
        "unsupported_group_labels",
        "missing_required_mechanisms",
    }
)


def _parse_validity_certificate(
    mapping: Mapping[str, object], *, expected_synthetic_only: bool
) -> ValidityCertificate:
    _fields(
        mapping,
        frozenset(
            {
                "checks",
                "diagnostic_level",
                "evidence",
                "evidence_authority",
                "evidence_sha256",
                "level",
                "schema_version",
                "synthetic_only",
                "thresholds",
                "thresholds_sha256",
            }
        ),
        "validity certificate",
    )
    if mapping["schema_version"] != 3:
        _schema_fail("validity certificate schema_version must equal 3")
    try:
        level = ValidityLevel(_string(mapping["level"], "validity level"))
        diagnostic_level = ValidityLevel(
            _string(mapping["diagnostic_level"], "validity diagnostic level")
        )
        evidence_authority = ValidityEvidenceAuthority(
            _string(mapping["evidence_authority"], "validity evidence authority")
        )
    except ValueError as error:
        raise ReportingSchemaError("validity certificate level is invalid") from error
    checks: list[ValidityCheck] = []
    for index, raw in enumerate(_array(mapping["checks"], "validity checks")):
        item = _mapping(raw, f"validity checks[{index}]")
        _fields(
            item,
            frozenset({"check_id", "observed", "reason", "status"}),
            f"validity checks[{index}]",
        )
        try:
            status = ValidityCheckStatus(
                _string(item["status"], f"validity checks[{index}].status")
            )
        except ValueError as error:
            raise ReportingSchemaError("validity check status is invalid") from error
        checks.append(
            ValidityCheck(
                check_id=_identifier(
                    item["check_id"], f"validity checks[{index}].check_id"
                ),
                status=status,
                observed=item["observed"],
                reason=_string(item["reason"], f"validity checks[{index}].reason"),
            )
        )
    if frozenset(item.check_id for item in checks) != _VALIDITY_CHECK_IDS:
        _schema_fail("validity certificate does not contain all required checks")
    synthetic_only = _bool(mapping["synthetic_only"], "validity synthetic_only")
    if synthetic_only != expected_synthetic_only:
        _integrity_fail("validity certificate/result synthetic markers disagree")
    try:
        thresholds = ValidityThresholds.from_mapping(
            _mapping(mapping["thresholds"], "validity thresholds")
        )
        evidence = ValidityEvidence.from_mapping(
            _mapping(mapping["evidence"], "validity evidence")
        )
    except PolicyEvaluationSchemaError as error:
        raise ReportingSchemaError(
            "validity certificate retained inputs are malformed"
        ) from error
    try:
        return ValidityCertificate(
            schema_version=3,
            level=level,
            diagnostic_level=diagnostic_level,
            evidence_authority=evidence_authority,
            thresholds=thresholds,
            evidence=evidence,
            thresholds_sha256=_sha256(
                mapping["thresholds_sha256"], "validity thresholds_sha256"
            ),
            evidence_sha256=_sha256(
                mapping["evidence_sha256"], "validity evidence_sha256"
            ),
            checks=tuple(checks),
            synthetic_only=synthetic_only,
        )
    except PolicyEvaluationSchemaError as error:
        raise ReportingIntegrityError(
            "validity certificate does not reproduce from retained inputs"
        ) from error


def _validate_source_rows(
    value: object, manifest: MatrixRunManifest
) -> tuple[tuple[str, str, str], ...]:
    rows: list[tuple[str, str, str]] = []
    for index, raw in enumerate(_array(value, "result source_rows")):
        item = _mapping(raw, f"result source_rows[{index}]")
        _fields(
            item,
            frozenset({"config_sha256", "result_sha256", "row_id"}),
            f"result source_rows[{index}]",
        )
        rows.append(
            (
                _identifier(item["row_id"], "source row_id"),
                _sha256(item["config_sha256"], "source row config_sha256"),
                _sha256(item["result_sha256"], "source row result_sha256"),
            )
        )
    expected = tuple(
        (entry.row_id, entry.config_sha256, cast(str, entry.result_sha256))
        for entry in manifest.final_entries
        if entry.status is RegistryStatus.COMPLETED
    )
    if tuple(rows) != expected:
        _integrity_fail(
            "reporting result source_rows omit or alter completed matrix rows"
        )
    return tuple(rows)


def _validate_completed_mechanism_execution(
    *, trust: FormalReportingTrust, manifest: MatrixRunManifest
) -> tuple[ExperimentResult, ...]:
    """Re-read audited row results and require real mechanism execution.

    ``audit_matrix_run`` establishes the complete run and result commitments.
    This reporting-specific pass then parses every completed ``ExperimentResult``
    and binds its semantic hash back to the audited manifest before inspecting
    ``mechanism_execution``.  A completed scheduler smoke row is therefore not
    paper evidence merely because its files are internally consistent.
    """
    completed = tuple(
        entry
        for entry in manifest.final_entries
        if entry.status is RegistryStatus.COMPLETED
    )
    if not completed:
        _integrity_fail("formal reporting requires at least one completed matrix row")
    results: list[ExperimentResult] = []
    for entry in completed:
        if entry.result_sha256 is None:
            _integrity_fail(
                f"completed matrix row {entry.row_id!r} has no result commitment"
            )
        result_path = trust.matrix_run_directory / "runs" / entry.row_id / "result.json"
        if result_path.is_symlink() or not result_path.is_file():
            _integrity_fail(
                f"completed matrix row {entry.row_id!r} has no regular result file"
            )
        try:
            result = ExperimentResult.from_mapping(read_json_object(result_path))
        except (ExperimentMatrixError, MetadataValidationError, OSError) as error:
            raise ReportingIntegrityError(
                f"cannot parse audited completed result {entry.row_id!r}: {error}"
            ) from error
        if result.sha256 != entry.result_sha256:
            _integrity_fail(
                f"completed matrix row {entry.row_id!r} changed after run audit"
            )
        if result.mechanism_execution is not True:
            _integrity_fail(
                "formal reporting rejected completed row "
                f"{entry.row_id!r}: mechanism_execution is not true"
            )
        results.append(result)
    return tuple(results)


def _validate_formal_figure_spec(
    spec: Mapping[str, object],
    *,
    artifact_id: str,
    result_sha256: str,
    registry_sha256: str,
    result: Mapping[str, object],
) -> tuple[FigureOutputSpec, ...]:
    provenance_fields = {
        "claim_scope",
        "config_sha256",
        "data_manifest_sha256",
        "environment_lock_sha256",
        "event_key_manifest_sha256",
        "external_preregistration_receipt_sha256",
        "preregistration_timestamp_verified",
        "primary_policy_sources_verified",
        "data_licenses_verified",
        "post_policy_inputs_frozen",
        "empirical_identification_audit_sha256",
        "empirical_identification_passed",
        "git_commit",
        "matrix_run_manifest_sha256",
        "matrix_sha256",
        "model_checkpoint_sha256",
        "study_protocol_sha256",
        "statistics_derivation_sha256",
        "validity_certificate_sha256",
    }
    _fields(
        spec,
        frozenset(
            {
                "experiment_registry_sha256",
                "formal_reporting_required",
                "outputs",
                "reporting_mode",
                "result_artifact_id",
                "result_sha256",
                "schema_version",
                "spec_id",
                "status",
                "synthetic_only",
                *provenance_fields,
            }
        ),
        "formal frozen figure specification",
    )
    if spec["schema_version"] != 2:
        _schema_fail("formal figure specification schema_version must equal 2")
    _identifier(spec["spec_id"], "figure spec_id")
    if spec["status"] != "frozen" or spec["reporting_mode"] != "formal":
        _integrity_fail("formal figure specification is not frozen for formal use")
    if (
        _bool(spec["formal_reporting_required"], "formal_reporting_required")
        is not True
    ):
        _integrity_fail("figure specification does not require formal reporting")
    if _identifier(spec["result_artifact_id"], "result_artifact_id") != artifact_id:
        _integrity_fail("figure specification references another result artifact")
    if _sha256(spec["result_sha256"], "figure result_sha256") != result_sha256:
        _integrity_fail("figure specification result hash mismatch")
    if (
        _sha256(
            spec["experiment_registry_sha256"],
            "figure experiment_registry_sha256",
        )
        != registry_sha256
    ):
        _integrity_fail("figure specification registry hash mismatch")
    if (
        _bool(spec["synthetic_only"], "figure synthetic_only")
        != result["synthetic_only"]
    ):
        _integrity_fail("figure/result synthetic markers disagree")
    for field in provenance_fields:
        expected = result[field]
        if spec[field] != expected:
            _integrity_fail(f"figure/result {field} commitments disagree")
    outputs = tuple(
        FigureOutputSpec.from_mapping(_mapping(raw, "figure output"), formal=True)
        for raw in _array(spec["outputs"], "figure outputs")
    )
    if tuple(item.output_id for item in outputs) != REQUIRED_REPORT_OUTPUTS:
        _schema_fail(
            "figure outputs must contain all nine required artifacts in canonical order"
        )
    if len({item.filename for item in outputs}) != len(outputs):
        _schema_fail("figure output filenames must be unique")
    return outputs


def _validate_formal_trust(
    *,
    trust: FormalReportingTrust,
    result: Mapping[str, object],
    result_path: Path,
    result_sha256: str,
    registry_path: Path,
    registry_sha256: str,
    spec: Mapping[str, object],
) -> tuple[tuple[FigureOutputSpec, ...], dict[str, object]]:
    repository_root = trust.repository_root.expanduser().resolve(strict=True)
    _verify_committed_file(
        trust.study_protocol_path,
        trust.expected_study_protocol_source_sha256,
        label="study protocol",
    )
    try:
        loaded_protocol = load_study_protocol(
            trust.study_protocol_path,
            expected_source_sha256=trust.expected_study_protocol_source_sha256,
        )
    except ProtocolValidationError as error:
        raise ReportingIntegrityError(str(error)) from error
    protocol = loaded_protocol.protocol
    outcome_manifest_path = trust.outcome_knowledge_manifest_path
    outcome_expected_source = trust.expected_outcome_knowledge_manifest_source_sha256
    outcome_expected_semantic = (
        trust.expected_outcome_knowledge_manifest_semantic_sha256
    )
    if protocol.mode is StudyMode.SYNTHETIC:
        if outcome_manifest_path is not None:
            _integrity_fail(
                "formal synthetic trust must not include a real-event outcome ledger"
            )
    else:
        if (
            outcome_manifest_path is None
            or outcome_expected_source is None
            or outcome_expected_semantic is None
        ):
            _integrity_fail(
                "formal real-study trust lacks an externally anchored outcome ledger"
            )
        if (
            outcome_expected_source != protocol.outcome_knowledge_manifest_source_sha256
            or outcome_expected_semantic
            != protocol.outcome_knowledge_manifest_semantic_sha256
        ):
            _integrity_fail(
                "formal outcome-ledger anchors differ from the study protocol"
            )
        _verify_committed_file(
            outcome_manifest_path,
            outcome_expected_source,
            label="outcome-knowledge manifest",
        )
    commitments = (
        (
            trust.config_path,
            trust.expected_config_sha256,
            "run config",
            trust.release_evidence.run_config_sha256,
        ),
        (
            trust.environment_lock_path,
            trust.expected_environment_lock_sha256,
            "environment lock",
            trust.release_evidence.environment_lock_sha256,
        ),
        (
            trust.data_manifest_path,
            trust.expected_data_manifest_sha256,
            "data manifest",
            trust.release_evidence.data_manifest_sha256,
        ),
        (
            trust.model_checkpoint_path,
            trust.expected_model_checkpoint_sha256,
            "model checkpoint",
            trust.release_evidence.model_checkpoint_sha256,
        ),
        (
            trust.event_key_manifest_path,
            trust.expected_event_key_manifest_sha256,
            "event-key manifest",
            trust.release_evidence.event_key_manifest_sha256,
        ),
    )
    for path, expected, label, release_value in commitments:
        observed = _verify_committed_file(path, expected, label=label)
        if release_value != observed:
            _integrity_fail(f"release evidence {label} commitment mismatch")
    receipt_authority = trust.execution_receipt_authority
    if receipt_authority is not None:
        if receipt_authority.repository_root != repository_root:
            _integrity_fail(
                "execution receipt authority is bound to another repository root"
            )
        receipt_inputs = receipt_authority.source_inputs
        if receipt_authority.protocol.sha256 != protocol.sha256:
            _integrity_fail(
                "execution receipt authority is bound to another study protocol"
            )
        receipt_bindings = (
            (
                receipt_inputs.run_config_path,
                trust.config_path,
                receipt_inputs.expected_run_config_sha256,
                trust.expected_config_sha256,
                "run config",
            ),
            (
                receipt_inputs.study_protocol_path,
                trust.study_protocol_path,
                receipt_inputs.expected_study_protocol_source_sha256,
                trust.expected_study_protocol_source_sha256,
                "study protocol source",
            ),
            (
                receipt_inputs.data_manifest_path,
                trust.data_manifest_path,
                None,
                None,
                "data manifest",
            ),
            (
                receipt_inputs.model_checkpoint_path,
                trust.model_checkpoint_path,
                None,
                None,
                "model checkpoint",
            ),
            (
                receipt_inputs.environment_lock_path,
                trust.environment_lock_path,
                None,
                None,
                "environment lock",
            ),
            (
                receipt_inputs.event_key_manifest_path,
                trust.event_key_manifest_path,
                receipt_inputs.expected_event_key_manifest_sha256,
                trust.expected_event_key_manifest_sha256,
                "event-key manifest",
            ),
        )
        for (
            receipt_path,
            trust_path,
            receipt_hash,
            trust_hash,
            label,
        ) in receipt_bindings:
            try:
                same_path = receipt_path.resolve(strict=True) == trust_path.resolve(
                    strict=True
                )
            except OSError as error:
                raise ReportingIntegrityError(
                    f"cannot resolve receipt-bound {label}: {error}"
                ) from error
            if not same_path or (
                receipt_hash is not None and receipt_hash != trust_hash
            ):
                _integrity_fail(
                    f"execution receipt authority {label} differs from formal trust"
                )
    synthetic_only = _bool(result["synthetic_only"], "result synthetic_only")
    run_manifest_path = trust.matrix_run_directory / "run-manifest.json"
    try:
        if run_manifest_path.is_symlink() or not run_manifest_path.is_file():
            _integrity_fail("matrix run manifest must be a regular non-symlink file")
        run_manifest_payload = run_manifest_path.read_bytes()
        run_manifest = MatrixRunManifest.from_mapping(
            read_json_bytes_object(
                run_manifest_payload,
                source=str(run_manifest_path),
            )
        )
        if run_manifest.sha256 != trust.expected_matrix_run_manifest_sha256:
            _integrity_fail("matrix run manifest semantic hash mismatch")
        run_manifest_file_sha256 = sha256_bytes(run_manifest_payload)
        run_audit = audit_matrix_run(
            trust.matrix_run_directory,
            trust.matrix,
            expected_manifest_sha256=trust.expected_matrix_run_manifest_sha256,
            execution_authority=trust.execution_receipt_authority,
        )
    except (ExperimentMatrixError, MetadataValidationError, OSError) as error:
        raise ReportingIntegrityError(f"matrix run audit failed: {error}") from error
    if run_audit.manifest_sha256 != trust.expected_matrix_run_manifest_sha256:
        _integrity_fail("matrix run audit returned another manifest commitment")
    _verify_committed_file(
        run_manifest_path,
        run_manifest_file_sha256,
        label="matrix run manifest after run audit",
    )
    for row in trust.matrix.rows:
        frozen = row.frozen_inputs
        if (
            frozen.data_manifest_sha256 != trust.expected_data_manifest_sha256
            or frozen.model_checkpoint_sha256 != trust.expected_model_checkpoint_sha256
            or frozen.environment_lock_sha256 != trust.expected_environment_lock_sha256
            or frozen.synthetic_only != synthetic_only
        ):
            _integrity_fail(
                f"matrix row {row.row_id!r} is bound to different frozen inputs"
            )
    expected_registry_path = (trust.matrix_run_directory / "matrix.json").resolve(
        strict=True
    )
    if registry_path.resolve(strict=True) != expected_registry_path:
        _integrity_fail("formal experiment registry must be the audited matrix.json")
    if registry_sha256 != sha256_file(expected_registry_path):
        _integrity_fail("formal registry hash differs from the audited matrix")
    validity_mapping = _read_committed_json(
        trust.validity_certificate_path,
        trust.expected_validity_certificate_sha256,
        label="validity certificate",
    )
    validity_sha256 = trust.expected_validity_certificate_sha256
    certificate = _parse_validity_certificate(
        validity_mapping, expected_synthetic_only=synthetic_only
    )
    if certificate.thresholds_sha256 != protocol.abstention_rule.definition_sha256:
        _integrity_fail(
            "validity thresholds are not bound to the preregistered abstention rule"
        )
    if trust.release_evidence.result_artifact_sha256 != result_sha256:
        _integrity_fail("release evidence result artifact commitment mismatch")
    if trust.release_evidence.run_manifest_sha256 != run_audit.manifest_sha256:
        _integrity_fail("release evidence matrix run manifest commitment mismatch")
    if trust.release_evidence.registered_rows_total != run_audit.row_count:
        _integrity_fail("release evidence registered row count mismatch")
    if trust.release_evidence.registered_rows_accounted != run_audit.row_count:
        _integrity_fail("release evidence does not account for the complete matrix")
    if not trust.release_evidence.failed_rows_reported:
        _integrity_fail("release evidence does not retain failed rows")
    decision = evaluate_release_eligibility(
        protocol,
        trust.release_evidence,
        outcome_knowledge_manifest_path=outcome_manifest_path,
    )
    if not decision.integrity_valid or not decision.formal_reporting_eligible:
        reasons = (*decision.integrity_errors, *decision.eligibility_errors)
        _integrity_fail(
            "release is not formal-reporting eligible: " + "; ".join(reasons)
        )
    observed_git = collect_git_metadata(Path(__file__).resolve().parent)
    if (
        observed_git.root is None
        or Path(observed_git.root).resolve() != repository_root
    ):
        _integrity_fail(
            "the audited Git repository does not own the executing PolicyTwin code"
        )
    if (
        observed_git.commit != trust.release_evidence.git_commit
        or observed_git.dirty != trust.release_evidence.git_dirty
        or observed_git.commit is None
        or observed_git.dirty
    ):
        _integrity_fail("live Git identity does not match clean release evidence")
    expected_synthetic = decision.allowed_claim_scope is ClaimScope.SYNTHETIC_ONLY
    if synthetic_only != expected_synthetic:
        _integrity_fail("protocol claim scope and result synthetic marker disagree")
    formal_fields = frozenset(
        {
            "artifact_id",
            "artifact_type",
            "claim_scope",
            "config_sha256",
            "data_manifest_sha256",
            "environment_lock_sha256",
            "event_key_manifest_sha256",
            "external_preregistration_receipt_sha256",
            "preregistration_timestamp_verified",
            "primary_policy_sources_verified",
            "data_licenses_verified",
            "post_policy_inputs_frozen",
            "empirical_identification_audit_sha256",
            "empirical_identification_passed",
            "formal_reporting_eligible",
            "git_commit",
            "git_dirty",
            "matrix_run_manifest_sha256",
            "matrix_sha256",
            "model_checkpoint_sha256",
            "schema_version",
            "source_rows",
            "statistics",
            "study_protocol_sha256",
            "statistics_derivation_sha256",
            "synthetic_only",
            "validation_status",
            "validity_certificate_sha256",
        }
    )
    _fields(result, formal_fields, "formal validated result")
    if result["schema_version"] != 2 or result["artifact_type"] != REPORT_RESULT_SCHEMA:
        _schema_fail("formal result does not use the v2 reporting schema")
    if result["validation_status"] != "formal_audited":
        _integrity_fail("a caller-declared validated flag is not formal audit evidence")
    if (
        _bool(result["formal_reporting_eligible"], "formal_reporting_eligible")
        is not True
    ):
        _integrity_fail("formal result does not declare its required reporting gate")
    cross_bindings: tuple[tuple[str, object], ...] = (
        ("study_protocol_sha256", protocol.sha256),
        (
            "statistics_derivation_sha256",
            trust.expected_statistics_derivation_sha256,
        ),
        ("config_sha256", trust.expected_config_sha256),
        ("matrix_sha256", trust.matrix.sha256),
        ("matrix_run_manifest_sha256", run_audit.manifest_sha256),
        ("validity_certificate_sha256", validity_sha256),
        ("environment_lock_sha256", trust.expected_environment_lock_sha256),
        ("data_manifest_sha256", trust.expected_data_manifest_sha256),
        ("model_checkpoint_sha256", trust.expected_model_checkpoint_sha256),
        ("event_key_manifest_sha256", trust.expected_event_key_manifest_sha256),
        (
            "external_preregistration_receipt_sha256",
            trust.release_evidence.external_preregistration_receipt_sha256,
        ),
        (
            "preregistration_timestamp_verified",
            trust.release_evidence.preregistration_timestamp_verified,
        ),
        (
            "primary_policy_sources_verified",
            trust.release_evidence.primary_policy_sources_verified,
        ),
        ("data_licenses_verified", trust.release_evidence.data_licenses_verified),
        (
            "post_policy_inputs_frozen",
            trust.release_evidence.post_policy_inputs_frozen,
        ),
        (
            "empirical_identification_audit_sha256",
            trust.release_evidence.empirical_identification_audit_sha256,
        ),
        (
            "empirical_identification_passed",
            trust.release_evidence.empirical_identification_passed,
        ),
        ("git_commit", observed_git.commit),
        ("git_dirty", False),
        ("claim_scope", cast(ClaimScope, decision.allowed_claim_scope).value),
    )
    for field, expected_value in cross_bindings:
        if result[field] != expected_value:
            _integrity_fail(f"formal result {field} commitment mismatch")
    _validate_source_rows(result["source_rows"], run_manifest)
    artifact_id = _identifier(result["artifact_id"], "result artifact_id")
    output_specs = _validate_formal_figure_spec(
        spec,
        artifact_id=artifact_id,
        result_sha256=result_sha256,
        registry_sha256=registry_sha256,
        result=result,
    )
    coverage_spec = next(
        item for item in output_specs if item.output_id == "coverage_width"
    )
    declared_statistics = _mapping(result["statistics"], "formal result statistics")
    # These semantic cross-bindings remain useful even before typed derivation
    # evidence is available.  Passing them never authorizes rendering.
    _validate_formal_coverage_width(
        declared_statistics,
        protocol,
        expected_output_ids=coverage_spec.registered_output_ids,
    )
    _validate_formal_group_cohorts(declared_statistics, protocol)
    completed_results = _validate_completed_mechanism_execution(
        trust=trust, manifest=run_manifest
    )
    if (
        trust.execution_receipt_authority is None
        or not run_audit.execution_receipts_verified
    ):
        _integrity_fail(
            "formal reporting blocked: every typed source result requires a "
            "run-manifest-bound, live-replayed source-input and "
            "runner-implementation receipt; runner_kind and caller hashes are "
            "not authority"
        )
    if not synthetic_only:
        _integrity_fail(
            "formal non-synthetic reporting requires an external execution trust "
            "root; the bundled replay verifier is synthetic-only"
        )
    if (
        trust.statistics_derivation_path is None
        or trust.expected_statistics_derivation_sha256 is None
    ):
        _integrity_fail(
            "formal reporting blocked: no audited statistics-derivation contract "
            "can reconstruct all nine reporting sections from the completed "
            "ExperimentResult artifacts"
        )
    try:
        derivation_mapping = _read_committed_json(
            trust.statistics_derivation_path,
            trust.expected_statistics_derivation_sha256,
            label="statistics derivation artifact",
        )
        derivation_sha256 = trust.expected_statistics_derivation_sha256
        derivation = StatisticsDerivationArtifact.from_mapping(derivation_mapping)
        if derivation.selection_rule.study_protocol_sha256 != protocol.sha256:
            _integrity_fail(
                "statistics derivation selection rule is bound to another "
                "study protocol"
            )
        derived = derive_reporting_statistics(
            derivation,
            completed_results,
            trust.matrix,
            matrix_run_manifest_sha256=run_audit.manifest_sha256,
            validity_certificate=certificate,
        )
    except StatisticsDerivationSchemaError as error:
        raise ReportingSchemaError(str(error)) from error
    except StatisticsDerivationIntegrityError as error:
        raise ReportingIntegrityError(str(error)) from error
    if sha256_bytes(canonical_json_bytes(declared_statistics)) != derived.sha256:
        _integrity_fail(
            "formal result statistics were altered or hand-filled instead of "
            "matching typed derivation"
        )
    recomputed_statistics = derived.to_mapping()
    _validate_result_statistics(
        recomputed_statistics, allow_multiple_coverage_series=True
    )
    _validate_formal_coverage_width(
        recomputed_statistics,
        protocol,
        expected_output_ids=coverage_spec.registered_output_ids,
    )
    _validate_formal_group_cohorts(recomputed_statistics, protocol)
    if (
        certificate.evidence_authority
        is ValidityEvidenceAuthority.SELF_REPORTED_SYNTHETIC_FIXTURE
    ):
        _integrity_fail(
            "self-reported synthetic validity evidence cannot authorize formal "
            "reporting; typed upstream verifier receipts are required"
        )
    if certificate.level in {ValidityLevel.RED, ValidityLevel.ABSTAIN}:
        _integrity_fail(
            "red or abstain validity certificates cannot authorize formal outputs"
        )
    if not derivation.selection_rule.has_independent_preregistration_authority:
        _integrity_fail(
            "synthetic threshold-source receipts prove mechanics only and "
            "cannot authorize formal preregistration"
        )
    provenance: dict[str, object] = {
        "allowed_claim_scope": cast(ClaimScope, decision.allowed_claim_scope).value,
        "study_protocol_path": str(trust.study_protocol_path.resolve(strict=True)),
        "study_protocol_source_sha256": loaded_protocol.source_sha256,
        "study_protocol_sha256": protocol.sha256,
        "outcome_knowledge_manifest_path": (
            None
            if outcome_manifest_path is None
            else str(outcome_manifest_path.resolve(strict=True))
        ),
        "outcome_knowledge_manifest_source_sha256": (
            decision.outcome_knowledge_manifest_source_sha256
        ),
        "outcome_knowledge_manifest_semantic_sha256": (
            decision.outcome_knowledge_manifest_semantic_sha256
        ),
        "design_temporality_decision_sha256": (
            decision.design_temporality_decision_sha256
        ),
        "derived_design_temporality": decision.derived_design_temporality,
        "statistics_derivation_path": str(
            trust.statistics_derivation_path.resolve(strict=True)
        ),
        "statistics_derivation_sha256": derivation_sha256,
        "matrix_sha256": trust.matrix.sha256,
        "matrix_run_manifest_sha256": run_audit.manifest_sha256,
        "validity_certificate_sha256": validity_sha256,
        "git_commit": observed_git.commit,
        "environment_lock_sha256": trust.expected_environment_lock_sha256,
        "data_manifest_sha256": trust.expected_data_manifest_sha256,
        "model_checkpoint_sha256": trust.expected_model_checkpoint_sha256,
        "event_key_manifest_sha256": trust.expected_event_key_manifest_sha256,
        "external_preregistration_receipt_sha256": (
            trust.release_evidence.external_preregistration_receipt_sha256
        ),
        "preregistration_timestamp_verified": (
            trust.release_evidence.preregistration_timestamp_verified
        ),
        "primary_policy_sources_verified": (
            trust.release_evidence.primary_policy_sources_verified
        ),
        "data_licenses_verified": trust.release_evidence.data_licenses_verified,
        "post_policy_inputs_frozen": trust.release_evidence.post_policy_inputs_frozen,
        "empirical_identification_audit_sha256": (
            trust.release_evidence.empirical_identification_audit_sha256
        ),
        "empirical_identification_passed": (
            trust.release_evidence.empirical_identification_passed
        ),
    }
    _verify_committed_file(
        run_manifest_path,
        run_manifest_file_sha256,
        label="matrix run manifest at reporting authorization",
    )
    _verify_committed_file(
        trust.validity_certificate_path,
        validity_sha256,
        label="validity certificate at reporting authorization",
    )
    _verify_committed_file(
        trust.statistics_derivation_path,
        derivation_sha256,
        label="statistics derivation artifact at reporting authorization",
    )
    for path, expected, label, _ in commitments:
        _verify_committed_file(
            path,
            expected,
            label=f"{label} at reporting authorization",
        )
    if outcome_manifest_path is not None and outcome_expected_source is not None:
        _verify_committed_file(
            outcome_manifest_path,
            outcome_expected_source,
            label="outcome-knowledge manifest at reporting authorization",
        )
    return output_specs, provenance


def _validated_inputs(
    *,
    result_path: Path,
    registry_path: Path,
    figure_spec_path: Path,
    expected_result_sha256: str,
    expected_registry_sha256: str,
    expected_figure_spec_sha256: str,
    formal_trust: FormalReportingTrust | None,
    allow_unverified_synthetic: bool,
) -> _ValidatedReportingInputs:
    result = _read_committed_json(
        result_path, expected_result_sha256, label="validated result"
    )
    registry = _read_committed_json(
        registry_path, expected_registry_sha256, label="experiment registry"
    )
    spec = _read_committed_json(
        figure_spec_path,
        expected_figure_spec_sha256,
        label="frozen figure specification",
    )
    required_base = {
        "artifact_id",
        "config_sha256",
        "statistics",
        "synthetic_only",
    }
    missing_base = sorted(required_base - set(result))
    if missing_base:
        _schema_fail("validated result is missing fields: " + ", ".join(missing_base))
    artifact_id = _identifier(result["artifact_id"], "result artifact_id")
    config_sha256 = _sha256(result["config_sha256"], "result config_sha256")
    synthetic_only = _bool(result["synthetic_only"], "result synthetic_only")
    statistics = _mapping(result["statistics"], "result statistics")
    _validate_result_statistics(
        statistics,
        allow_multiple_coverage_series=formal_trust is not None,
    )
    result_sha256 = _sha256(expected_result_sha256, "expected result sha256")
    registry_sha256 = _sha256(expected_registry_sha256, "expected registry sha256")
    figure_spec_sha256 = _sha256(
        expected_figure_spec_sha256, "expected figure spec sha256"
    )
    if formal_trust is None:
        if not allow_unverified_synthetic:
            _integrity_fail(
                "reporting requires formal trust or explicit "
                "allow_unverified_synthetic=True"
            )
        _fields(
            result,
            frozenset(
                {
                    "schema_version",
                    "artifact_id",
                    "artifact_type",
                    "validation_status",
                    "synthetic_only",
                    "config_sha256",
                    "statistics",
                }
            ),
            "unverified synthetic result",
        )
        if not synthetic_only:
            _integrity_fail("non-synthetic results require the formal reporting gate")
        if (
            result["schema_version"] != 1
            or result["artifact_type"] != SYNTHETIC_UNVERIFIED_RESULT_SCHEMA
        ):
            _schema_fail("unverified synthetic result must use the legacy v1 schema")
        if result["validation_status"] != "validated":
            _integrity_fail("unverified synthetic result is not marked validated")
        _validate_legacy_registry_entry(
            registry,
            registry_path=registry_path,
            result_path=result_path,
            artifact_id=artifact_id,
            result_sha256=result_sha256,
            config_sha256=config_sha256,
            synthetic_only=synthetic_only,
        )
        output_specs = _validate_legacy_figure_spec(
            spec,
            artifact_id=artifact_id,
            result_sha256=result_sha256,
            registry_sha256=registry_sha256,
            config_sha256=config_sha256,
            synthetic_only=synthetic_only,
        )
        provenance: dict[str, object] = {
            "allowed_claim_scope": None,
            "study_protocol_path": None,
            "study_protocol_source_sha256": None,
            "study_protocol_sha256": None,
            "outcome_knowledge_manifest_path": None,
            "outcome_knowledge_manifest_source_sha256": None,
            "outcome_knowledge_manifest_semantic_sha256": None,
            "design_temporality_decision_sha256": None,
            "derived_design_temporality": None,
            "statistics_derivation_path": None,
            "statistics_derivation_sha256": None,
            "matrix_sha256": None,
            "matrix_run_manifest_sha256": None,
            "validity_certificate_sha256": None,
            "git_commit": None,
            "environment_lock_sha256": None,
            "data_manifest_sha256": None,
            "model_checkpoint_sha256": None,
            "event_key_manifest_sha256": None,
            "external_preregistration_receipt_sha256": None,
            "preregistration_timestamp_verified": None,
            "primary_policy_sources_verified": None,
            "data_licenses_verified": None,
            "post_policy_inputs_frozen": None,
            "empirical_identification_audit_sha256": None,
            "empirical_identification_passed": None,
        }
        reporting_eligibility = "synthetic_unverified"
        formal_eligible = False
    else:
        output_specs, provenance = _validate_formal_trust(
            trust=formal_trust,
            result=result,
            result_path=result_path,
            result_sha256=result_sha256,
            registry_path=registry_path,
            registry_sha256=registry_sha256,
            spec=spec,
        )
        reporting_eligibility = "formal_audited"
        formal_eligible = True
    _verify_committed_file(
        result_path,
        result_sha256,
        label="validated result at rendering authorization",
    )
    _verify_committed_file(
        registry_path,
        registry_sha256,
        label="experiment registry at rendering authorization",
    )
    _verify_committed_file(
        figure_spec_path,
        figure_spec_sha256,
        label="frozen figure specification at rendering authorization",
    )
    return _ValidatedReportingInputs(
        result=result,
        output_specs=output_specs,
        artifact_id=artifact_id,
        config_sha256=config_sha256,
        synthetic_only=synthetic_only,
        result_sha256=result_sha256,
        registry_sha256=registry_sha256,
        figure_spec_sha256=figure_spec_sha256,
        reporting_eligibility=reporting_eligibility,
        formal_reporting_eligible=formal_eligible,
        allowed_claim_scope=cast(str | None, provenance["allowed_claim_scope"]),
        study_protocol_path=cast(str | None, provenance["study_protocol_path"]),
        study_protocol_source_sha256=cast(
            str | None, provenance["study_protocol_source_sha256"]
        ),
        study_protocol_sha256=cast(str | None, provenance["study_protocol_sha256"]),
        outcome_knowledge_manifest_path=cast(
            str | None, provenance["outcome_knowledge_manifest_path"]
        ),
        outcome_knowledge_manifest_source_sha256=cast(
            str | None,
            provenance["outcome_knowledge_manifest_source_sha256"],
        ),
        outcome_knowledge_manifest_semantic_sha256=cast(
            str | None,
            provenance["outcome_knowledge_manifest_semantic_sha256"],
        ),
        design_temporality_decision_sha256=cast(
            str | None,
            provenance["design_temporality_decision_sha256"],
        ),
        derived_design_temporality=cast(
            str | None, provenance["derived_design_temporality"]
        ),
        statistics_derivation_path=cast(
            str | None, provenance["statistics_derivation_path"]
        ),
        statistics_derivation_sha256=cast(
            str | None, provenance["statistics_derivation_sha256"]
        ),
        matrix_sha256=cast(str | None, provenance["matrix_sha256"]),
        matrix_run_manifest_sha256=cast(
            str | None, provenance["matrix_run_manifest_sha256"]
        ),
        validity_certificate_sha256=cast(
            str | None, provenance["validity_certificate_sha256"]
        ),
        git_commit=cast(str | None, provenance["git_commit"]),
        environment_lock_sha256=cast(str | None, provenance["environment_lock_sha256"]),
        data_manifest_sha256=cast(str | None, provenance["data_manifest_sha256"]),
        model_checkpoint_sha256=cast(str | None, provenance["model_checkpoint_sha256"]),
        event_key_manifest_sha256=cast(
            str | None, provenance["event_key_manifest_sha256"]
        ),
        external_preregistration_receipt_sha256=cast(
            str | None,
            provenance["external_preregistration_receipt_sha256"],
        ),
        preregistration_timestamp_verified=cast(
            bool | None, provenance["preregistration_timestamp_verified"]
        ),
        primary_policy_sources_verified=cast(
            bool | None, provenance["primary_policy_sources_verified"]
        ),
        data_licenses_verified=cast(bool | None, provenance["data_licenses_verified"]),
        post_policy_inputs_frozen=cast(
            bool | None, provenance["post_policy_inputs_frozen"]
        ),
        empirical_identification_audit_sha256=cast(
            str | None, provenance["empirical_identification_audit_sha256"]
        ),
        empirical_identification_passed=cast(
            bool | None, provenance["empirical_identification_passed"]
        ),
    )


def _format_number(value: object) -> str:
    return format(_number(value, "rendered statistic"), ".10g")


def _scale(values: tuple[float, ...], low: float, high: float) -> tuple[float, ...]:
    minimum = min(values)
    maximum = max(values)
    if maximum == minimum:
        return tuple((low + high) / 2.0 for _ in values)
    return tuple(
        low + (value - minimum) * (high - low) / (maximum - minimum) for value in values
    )


def _svg_document(
    *, title: str, config_sha256: str, synthetic_only: bool, body: str
) -> str:
    prefix = "SYNTHETIC — " if synthetic_only else ""
    complete_title = f"{prefix}{title} | config SHA-256: {config_sha256}"
    synthetic_banner = (
        '<text x="400" y="42" text-anchor="middle" class="synthetic">SYNTHETIC</text>'
        if synthetic_only
        else ""
    )
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<svg xmlns="http://www.w3.org/2000/svg" width="800" height="500" '
        'viewBox="0 0 800 500" role="img">\n'
        f"  <title>{html.escape(complete_title)}</title>\n"
        "  <desc>Rendered only from hash-validated, precomputed statistics.</desc>\n"
        "  <style>text{font-family:system-ui,sans-serif;fill:#172033}"
        ".axis{stroke:#667085;stroke-width:1}.series{fill:none;stroke:#2563eb;"
        "stroke-width:3}.point{fill:#2563eb}.secondary{fill:#f97316}"
        ".label{font-size:13px}.heading{font-size:18px;font-weight:600}"
        ".synthetic{font-size:20px;font-weight:800;fill:#b42318}</style>\n"
        '  <rect width="800" height="500" fill="#ffffff"/>\n'
        f"  {synthetic_banner}\n"
        f'  <text x="400" y="70" text-anchor="middle" class="heading">'
        f"{html.escape(title)}</text>\n"
        f"{body}"
        f'  <text x="400" y="486" text-anchor="middle" class="label">config '
        f"SHA-256: {config_sha256}</text>\n"
        "</svg>\n"
    )


def _line_chart(
    rows: tuple[Mapping[str, object], ...],
    *,
    x_field: str,
    y_field: str,
    x_label: str,
    y_label: str,
    title: str,
    config_sha256: str,
    synthetic_only: bool,
) -> str:
    x_values = tuple(_number(row[x_field], x_field) for row in rows)
    y_values = tuple(_number(row[y_field], y_field) for row in rows)
    x_coordinates = _scale(x_values, 110.0, 740.0)
    y_coordinates = tuple(
        420.0 - coordinate for coordinate in _scale(y_values, 0.0, 310.0)
    )
    points = " ".join(
        f"{x:.2f},{y:.2f}" for x, y in zip(x_coordinates, y_coordinates, strict=True)
    )
    circles = "".join(
        f'  <circle cx="{x:.2f}" cy="{y:.2f}" r="4" class="point"/>\n'
        for x, y in zip(x_coordinates, y_coordinates, strict=True)
    )
    body = (
        '  <line x1="90" y1="420" x2="750" y2="420" class="axis"/>\n'
        '  <line x1="90" y1="100" x2="90" y2="420" class="axis"/>\n'
        f'  <polyline points="{points}" class="series"/>\n'
        f"{circles}"
        f'  <text x="420" y="455" text-anchor="middle" class="label">'
        f"{html.escape(x_label)}</text>\n"
        f'  <text x="20" y="260" transform="rotate(-90 20 260)" '
        f'class="label">{html.escape(y_label)}</text>\n'
    )
    return _svg_document(
        title=title,
        config_sha256=config_sha256,
        synthetic_only=synthetic_only,
        body=body,
    )


def _group_bars(
    rows: tuple[Mapping[str, object], ...],
    *,
    title: str,
    config_sha256: str,
    synthetic_only: bool,
) -> str:
    width = 540.0 / len(rows)
    provenance = ";".join(
        (
            f"{_identifier(row['group'], 'group')}:"
            f"{_string(row['cohort_basis'], 'cohort_basis')}:"
            f"{_sha256(row['cohort_definition_sha256'], 'cohort_definition_sha256')}"
        )
        for row in rows
    )
    chunks = [
        f"  <desc>cohort_provenance={html.escape(provenance)}</desc>\n",
        '  <line x1="100" y1="420" x2="740" y2="420" class="axis"/>\n',
    ]
    for index, row in enumerate(rows):
        center = 140.0 + index * width
        benefit = _fraction(row["benefit_probability"], "benefit_probability")
        harm = _fraction(row["harm_probability"], "harm_probability")
        benefit_height = benefit * 280.0
        harm_height = harm * 280.0
        chunks.extend(
            (
                f'  <rect x="{center - 24:.2f}" y="{420 - benefit_height:.2f}" '
                f'width="22" height="{benefit_height:.2f}" class="point"/>\n',
                f'  <rect x="{center + 2:.2f}" y="{420 - harm_height:.2f}" '
                f'width="22" height="{harm_height:.2f}" class="secondary"/>\n',
                f'  <text x="{center:.2f}" y="445" text-anchor="middle" '
                f'class="label">{html.escape(_string(row["group"], "group"))} '
                f"[{html.escape(_string(row['cohort_basis'], 'cohort_basis'))}]"
                "</text>\n",
            )
        )
    chunks.append(
        '  <text x="650" y="105" class="label">blue=benefit, orange=harm</text>\n'
    )
    return _svg_document(
        title=title,
        config_sha256=config_sha256,
        synthetic_only=synthetic_only,
        body="".join(chunks),
    )


def _calibration_bars(
    rows: tuple[Mapping[str, object], ...],
    *,
    title: str,
    config_sha256: str,
    synthetic_only: bool,
) -> str:
    values = tuple(
        _number(row[field], field)
        for row in rows
        for field in ("observed", "predicted")
    )
    minimum = min((*values, 0.0))
    maximum = max((*values, 0.0))

    def y_coordinate(value: float) -> float:
        if maximum == minimum:
            return 280.0
        return 410.0 - (value - minimum) * 280.0 / (maximum - minimum)

    baseline_y = y_coordinate(0.0)
    width = 540.0 / len(rows)
    chunks = [
        f'  <line x1="100" y1="{baseline_y:.2f}" x2="740" '
        f'y2="{baseline_y:.2f}" class="axis"/>\n'
    ]
    for index, row in enumerate(rows):
        center = 140.0 + index * width
        observed_y = y_coordinate(_number(row["observed"], "observed"))
        predicted_y = y_coordinate(_number(row["predicted"], "predicted"))
        statistic = html.escape(_string(row["statistic"], "statistic"))
        chunks.extend(
            (
                f'  <rect x="{center - 24:.2f}" '
                f'y="{min(observed_y, baseline_y):.2f}" width="22" '
                f'height="{abs(observed_y - baseline_y):.2f}" class="point"/>\n',
                f'  <rect x="{center + 2:.2f}" '
                f'y="{min(predicted_y, baseline_y):.2f}" width="22" '
                f'height="{abs(predicted_y - baseline_y):.2f}" '
                f'class="secondary"/>\n',
                f'  <text x="{center:.2f}" y="445" text-anchor="middle" '
                f'class="label">{statistic}</text>\n',
            )
        )
    chunks.append(
        '  <text x="620" y="105" class="label">blue=observed, orange=predicted</text>\n'
    )
    return _svg_document(
        title=title,
        config_sha256=config_sha256,
        synthetic_only=synthetic_only,
        body="".join(chunks),
    )


def _heatmap(
    rows: tuple[Mapping[str, object], ...],
    *,
    title: str,
    config_sha256: str,
    synthetic_only: bool,
) -> str:
    row_names = tuple(sorted({_string(row["row"], "heatmap row") for row in rows}))
    column_names = tuple(
        sorted({_string(row["column"], "heatmap column") for row in rows})
    )
    values = {
        (
            _string(row["row"], "heatmap row"),
            _string(row["column"], "heatmap column"),
        ): _fraction(row["value"], "heatmap value")
        for row in rows
    }
    cell_width = 520.0 / len(column_names)
    cell_height = 280.0 / len(row_names)
    chunks: list[str] = []
    for row_index, row_name in enumerate(row_names):
        y = 120.0 + row_index * cell_height
        chunks.append(
            f'  <text x="105" y="{y + cell_height / 2:.2f}" text-anchor="end" '
            f'class="label">{html.escape(row_name)}</text>\n'
        )
        for column_index, column_name in enumerate(column_names):
            x = 120.0 + column_index * cell_width
            value = values[(row_name, column_name)]
            red = round(245 - 155 * value)
            green = round(250 - 70 * value)
            blue = round(255 - 25 * value)
            chunks.append(
                f'  <rect x="{x:.2f}" y="{y:.2f}" width="{cell_width:.2f}" '
                f'height="{cell_height:.2f}" fill="rgb({red},{green},{blue})"/>\n'
                f'  <text x="{x + cell_width / 2:.2f}" '
                f'y="{y + cell_height / 2:.2f}" text-anchor="middle" class="label">'
                f"{_format_number(value)}</text>\n"
            )
    for index, column_name in enumerate(column_names):
        x = 120.0 + (index + 0.5) * cell_width
        chunks.append(
            f'  <text x="{x:.2f}" y="415" text-anchor="middle" class="label">'
            f"{html.escape(column_name)}</text>\n"
        )
    return _svg_document(
        title=title,
        config_sha256=config_sha256,
        synthetic_only=synthetic_only,
        body="".join(chunks),
    )


def _latex_escape(value: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(character, character) for character in value)


def _latex_table(
    *,
    title: str,
    config_sha256: str,
    synthetic_only: bool,
    headers: tuple[str, ...],
    rows: tuple[tuple[str, ...], ...],
    label: str,
) -> str:
    prefix = "SYNTHETIC --- " if synthetic_only else ""
    caption = (
        f"{_latex_escape(f'{prefix}{title}')}. Config SHA-256: "
        rf"\texttt{{{_latex_escape(config_sha256)}}}."
    )
    rendered_rows = "".join(
        " & ".join(_latex_escape(cell) for cell in row) + r" \\" + "\n" for row in rows
    )
    return (
        "% Generated only from hash-validated precomputed statistics.\n"
        "\\begin{table}[htbp]\n"
        "\\centering\n"
        f"\\caption{{{caption}}}\n"
        f"\\label{{tab:{_latex_escape(label)}}}\n"
        f"\\begin{{tabular}}{{{'l' * len(headers)}}}\n"
        "\\hline\n"
        + " & ".join(_latex_escape(header) for header in headers)
        + r" \\"
        + "\n\\hline\n"
        + rendered_rows
        + "\\hline\n"
        "\\end{tabular}\n"
        "\\end{table}\n"
    )


def _coverage_width_title(title: str, rows: tuple[Mapping[str, object], ...]) -> str:
    first = rows[0]
    registered_output = _identifier(
        first["registered_output_id"], "registered_output_id"
    )
    levels = ",".join(_format_number(row["level"]) for row in rows)
    draw_counts = ",".join(
        str(_positive_int(row["valid_draw_count"], "valid_draw_count")) for row in rows
    )
    semantics = "; ".join(
        (
            f"kind={_identifier(first['interval_kind'], 'interval_kind')}",
            f"method={_identifier(first['interval_method_id'], 'interval_method_id')}",
            f"estimand={_identifier(first['estimand_id'], 'estimand_id')}",
            f"sampling_unit={_identifier(first['sampling_unit'], 'sampling_unit')}",
            f"registered_output={registered_output}",
            f"metric={_identifier(first['metric_id'], 'metric_id')}",
            f"cohort={_identifier(first['cohort_id'], 'cohort_id')}",
            f"horizon={_identifier(first['horizon_id'], 'horizon_id')}",
            f"levels={levels}",
            f"valid_draws={draw_counts}",
        )
    )
    return f"{title} [{semantics}]"


def _ablation_table_title(title: str, rows: tuple[Mapping[str, object], ...]) -> str:
    first = rows[0]
    level = "null" if first["level"] is None else _format_number(first["level"])
    uncertainty_kind = _identifier(first["uncertainty_kind"], "uncertainty_kind")
    semantics = "; ".join(
        (
            f"uncertainty_kind={uncertainty_kind}",
            f"method={_identifier(first['method_id'], 'method_id')}",
            f"level={level}",
            f"sampling_unit={_identifier(first['sampling_unit'], 'sampling_unit')}",
            f"estimand={_identifier(first['estimand_id'], 'estimand_id')}",
        )
    )
    return f"{title} [{semantics}]"


def _render_output(
    spec: FigureOutputSpec,
    statistics: Mapping[str, object],
    *,
    config_sha256: str,
    synthetic_only: bool,
) -> str:
    rows = tuple(
        _mapping(item, spec.output_id)
        for item in _array(statistics[spec.output_id], spec.output_id)
    )
    if spec.output_id == "paired_distribution":
        return _line_chart(
            rows,
            x_field="difference",
            y_field="density",
            x_label="Policy minus baseline difference",
            y_label="Provided density",
            title=spec.title,
            config_sha256=config_sha256,
            synthetic_only=synthetic_only,
        )
    if spec.output_id == "predicted_vs_empirical":
        return _line_chart(
            rows,
            x_field="predicted",
            y_field="empirical",
            x_label="Provided model prediction",
            y_label="Provided empirical estimate",
            title=spec.title,
            config_sha256=config_sha256,
            synthetic_only=synthetic_only,
        )
    if spec.output_id == "coverage_width":
        return _line_chart(
            rows,
            x_field="coverage",
            y_field="width",
            x_label="Provided empirical coverage",
            y_label="Provided interval width",
            title=_coverage_width_title(spec.title, rows),
            config_sha256=config_sha256,
            synthetic_only=synthetic_only,
        )
    if spec.output_id == "group_benefit_harm":
        return _group_bars(
            rows,
            title=spec.title,
            config_sha256=config_sha256,
            synthetic_only=synthetic_only,
        )
    if spec.output_id == "calibration_diagnostics":
        return _calibration_bars(
            rows,
            title=spec.title,
            config_sha256=config_sha256,
            synthetic_only=synthetic_only,
        )
    if spec.output_id == "risk_coverage":
        return _line_chart(
            rows,
            x_field="coverage",
            y_field="risk",
            x_label="Provided coverage",
            y_label="Provided risk",
            title=spec.title,
            config_sha256=config_sha256,
            synthetic_only=synthetic_only,
        )
    if spec.output_id == "certificate_heatmap":
        return _heatmap(
            rows,
            title=spec.title,
            config_sha256=config_sha256,
            synthetic_only=synthetic_only,
        )
    if spec.output_id == "ablation_table":
        ablation_rows = tuple(
            (
                _identifier(row["mechanism"], "mechanism"),
                _format_number(row["estimate"]),
                _format_number(row["uncertainty"]),
                _identifier(row["mcse_spec_id"], "mcse_spec_id"),
                str(_positive_int(row["valid_draw_count"], "valid_draw_count")),
            )
            for row in rows
        )
        return _latex_table(
            headers=(
                "Mechanism",
                "Mean estimate",
                "Event-keyed bootstrap MCSE",
                "MCSE spec ID",
                "Valid draws",
            ),
            rows=ablation_rows,
            label=spec.output_id,
            title=_ablation_table_title(spec.title, rows),
            config_sha256=config_sha256,
            synthetic_only=synthetic_only,
        )
    if spec.output_id == "compute_scaling_table":
        scaling_rows = tuple(
            (
                str(_positive_int(row["workers"], "workers")),
                str(_positive_int(row["events"], "events")),
                _format_number(row["wall_seconds"]),
            )
            for row in rows
        )
        return _latex_table(
            headers=("Workers", "Events", "Provided wall seconds"),
            rows=scaling_rows,
            label=spec.output_id,
            title=spec.title,
            config_sha256=config_sha256,
            synthetic_only=synthetic_only,
        )
    raise AssertionError("validated reporting output was not rendered")


@dataclass(frozen=True, slots=True)
class _ReportingSourceBundlePreflight:
    """Validated logical source paths computed before parsing source content."""

    root: Path
    result_path: str
    registry_path: str
    figure_spec_path: str
    study_protocol_path: str | None
    outcome_knowledge_manifest_path: str | None
    statistics_derivation_path: str | None


def _preflight_reporting_source_bundle(
    *,
    result_path: Path,
    registry_path: Path,
    figure_spec_path: Path,
    formal_trust: FormalReportingTrust | None,
) -> _ReportingSourceBundlePreflight:
    """Prove lexical containment and file kinds before reading source bytes."""
    _validate_reporting_result_format(result_path)
    source_bundle_root = (
        registry_path.parent
        if formal_trust is None
        else formal_trust.matrix_run_directory.parent
    )
    result_logical_path = _source_bundle_logical_path(
        result_path,
        bundle_root=source_bundle_root,
        label="validated result",
    )
    registry_logical_path = _source_bundle_logical_path(
        registry_path,
        bundle_root=source_bundle_root,
        label="experiment registry",
    )
    figure_spec_logical_path = _source_bundle_logical_path(
        figure_spec_path,
        bundle_root=source_bundle_root,
        label="frozen figure specification",
    )
    if formal_trust is None:
        return _ReportingSourceBundlePreflight(
            root=source_bundle_root,
            result_path=result_logical_path,
            registry_path=registry_logical_path,
            figure_spec_path=figure_spec_logical_path,
            study_protocol_path=None,
            outcome_knowledge_manifest_path=None,
            statistics_derivation_path=None,
        )

    _source_bundle_directory_logical_path(
        formal_trust.matrix_run_directory,
        bundle_root=source_bundle_root,
        label="matrix run directory",
    )
    formal_sources = (
        (formal_trust.config_path, "run config"),
        (formal_trust.environment_lock_path, "environment lock"),
        (formal_trust.data_manifest_path, "data manifest"),
        (formal_trust.model_checkpoint_path, "model checkpoint"),
        (formal_trust.event_key_manifest_path, "event-key manifest"),
        (formal_trust.validity_certificate_path, "validity certificate"),
    )
    for source_path, label in formal_sources:
        _source_bundle_logical_path(
            source_path,
            bundle_root=source_bundle_root,
            label=label,
        )
    study_protocol_logical_path = _source_bundle_logical_path(
        formal_trust.study_protocol_path,
        bundle_root=source_bundle_root,
        label="study protocol",
    )
    outcome_knowledge_manifest_logical_path = (
        None
        if formal_trust.outcome_knowledge_manifest_path is None
        else _source_bundle_logical_path(
            formal_trust.outcome_knowledge_manifest_path,
            bundle_root=source_bundle_root,
            label="outcome-knowledge manifest",
        )
    )
    statistics_derivation_logical_path = (
        None
        if formal_trust.statistics_derivation_path is None
        else _source_bundle_logical_path(
            formal_trust.statistics_derivation_path,
            bundle_root=source_bundle_root,
            label="statistics derivation artifact",
        )
    )
    return _ReportingSourceBundlePreflight(
        root=source_bundle_root,
        result_path=result_logical_path,
        registry_path=registry_logical_path,
        figure_spec_path=figure_spec_logical_path,
        study_protocol_path=study_protocol_logical_path,
        outcome_knowledge_manifest_path=outcome_knowledge_manifest_logical_path,
        statistics_derivation_path=statistics_derivation_logical_path,
    )


def generate_paper_artifacts(
    *,
    result_path: Path,
    registry_path: Path,
    figure_spec_path: Path,
    output_directory: Path,
    expected_result_sha256: str,
    expected_registry_sha256: str,
    expected_figure_spec_sha256: str,
    formal_trust: FormalReportingTrust | None = None,
    allow_unverified_synthetic: bool = False,
) -> PaperArtifactManifest:
    """Render all nine frozen outputs without recomputing any statistic.

    Formal rendering additionally requires a typed derivation artifact and a
    live-replayed execution receipt for every completed source row.  The bundled
    positive receipt authority is synthetic-only; real reporting remains closed
    until an external execution trust root exists.
    """
    source_bundle = _preflight_reporting_source_bundle(
        result_path=result_path,
        registry_path=registry_path,
        figure_spec_path=figure_spec_path,
        formal_trust=formal_trust,
    )
    inputs = _validated_inputs(
        result_path=result_path,
        registry_path=registry_path,
        figure_spec_path=figure_spec_path,
        expected_result_sha256=expected_result_sha256,
        expected_registry_sha256=expected_registry_sha256,
        expected_figure_spec_sha256=expected_figure_spec_sha256,
        formal_trust=formal_trust,
        allow_unverified_synthetic=allow_unverified_synthetic,
    )
    statistics = _mapping(inputs.result["statistics"], "result statistics")
    rendered = tuple(
        (
            spec,
            _render_output(
                spec,
                statistics,
                config_sha256=inputs.config_sha256,
                synthetic_only=inputs.synthetic_only,
            ),
        )
        for spec in inputs.output_specs
    )
    output = output_directory.expanduser().resolve()
    try:
        output.mkdir(parents=True, exist_ok=False)
    except FileExistsError as error:
        raise ReportingIntegrityError(
            f"reporting output directory already exists: {output}"
        ) from error
    generated: list[GeneratedPaperArtifact] = []
    for spec, content in rendered:
        path = output / spec.filename
        write_text_file(path, content)
        generated.append(
            GeneratedPaperArtifact(
                output_id=spec.output_id,
                path=spec.filename,
                media_type=(
                    "image/svg+xml" if spec.kind == "svg" else "application/x-tex"
                ),
                sha256=sha256_file(path),
                source_artifact_id=inputs.artifact_id,
                source_artifact_sha256=inputs.result_sha256,
                experiment_registry_sha256=inputs.registry_sha256,
                figure_spec_sha256=inputs.figure_spec_sha256,
                config_sha256=inputs.config_sha256,
                study_protocol_sha256=inputs.study_protocol_sha256,
                matrix_run_manifest_sha256=inputs.matrix_run_manifest_sha256,
                statistics_derivation_sha256=(inputs.statistics_derivation_sha256),
                validity_certificate_sha256=inputs.validity_certificate_sha256,
            )
        )
    manifest = PaperArtifactManifest(
        synthetic_only=inputs.synthetic_only,
        config_sha256=inputs.config_sha256,
        result_path=source_bundle.result_path,
        registry_path=source_bundle.registry_path,
        figure_spec_path=source_bundle.figure_spec_path,
        result_sha256=inputs.result_sha256,
        registry_sha256=inputs.registry_sha256,
        figure_spec_sha256=inputs.figure_spec_sha256,
        reporting_eligibility=inputs.reporting_eligibility,
        formal_reporting_eligible=inputs.formal_reporting_eligible,
        allowed_claim_scope=inputs.allowed_claim_scope,
        study_protocol_path=source_bundle.study_protocol_path,
        study_protocol_source_sha256=inputs.study_protocol_source_sha256,
        study_protocol_sha256=inputs.study_protocol_sha256,
        outcome_knowledge_manifest_path=(source_bundle.outcome_knowledge_manifest_path),
        outcome_knowledge_manifest_source_sha256=(
            inputs.outcome_knowledge_manifest_source_sha256
        ),
        outcome_knowledge_manifest_semantic_sha256=(
            inputs.outcome_knowledge_manifest_semantic_sha256
        ),
        design_temporality_decision_sha256=(inputs.design_temporality_decision_sha256),
        derived_design_temporality=inputs.derived_design_temporality,
        matrix_sha256=inputs.matrix_sha256,
        matrix_run_manifest_sha256=inputs.matrix_run_manifest_sha256,
        statistics_derivation_path=source_bundle.statistics_derivation_path,
        statistics_derivation_sha256=inputs.statistics_derivation_sha256,
        validity_certificate_sha256=inputs.validity_certificate_sha256,
        git_commit=inputs.git_commit,
        environment_lock_sha256=inputs.environment_lock_sha256,
        data_manifest_sha256=inputs.data_manifest_sha256,
        model_checkpoint_sha256=inputs.model_checkpoint_sha256,
        event_key_manifest_sha256=inputs.event_key_manifest_sha256,
        external_preregistration_receipt_sha256=(
            inputs.external_preregistration_receipt_sha256
        ),
        preregistration_timestamp_verified=(inputs.preregistration_timestamp_verified),
        primary_policy_sources_verified=inputs.primary_policy_sources_verified,
        data_licenses_verified=inputs.data_licenses_verified,
        post_policy_inputs_frozen=inputs.post_policy_inputs_frozen,
        empirical_identification_audit_sha256=(
            inputs.empirical_identification_audit_sha256
        ),
        empirical_identification_passed=inputs.empirical_identification_passed,
        outputs=tuple(generated),
    )
    manifest_path = output / "artifact_manifest.json"
    write_json_file(manifest_path, manifest.to_mapping())
    write_text_file(
        output / "artifact_manifest.sha256", f"{sha256_file(manifest_path)}\n"
    )
    return manifest


_MAX_CAPTURED_PAPER_FILE_BYTES = 64 * 1024 * 1024


@dataclass(slots=True)
class _CapturedPaperFile:
    name: str
    fd: int
    payload: bytes
    fingerprint: tuple[int, ...]


def _paper_stat_fingerprint(info: os.stat_result) -> tuple[int, ...]:
    return (
        info.st_dev,
        info.st_ino,
        info.st_mode,
        info.st_nlink,
        info.st_size,
        info.st_mtime_ns,
        info.st_ctime_ns,
    )


def _read_captured_fd(fd: int, *, expected_size: int, label: str) -> bytes:
    if not 0 <= expected_size <= _MAX_CAPTURED_PAPER_FILE_BYTES:
        _integrity_fail(f"{label} exceeds the paper artifact size limit")
    try:
        os.lseek(fd, 0, os.SEEK_SET)
        chunks: list[bytes] = []
        observed = 0
        while True:
            chunk = os.read(fd, min(1024 * 1024, expected_size + 1 - observed))
            if not chunk:
                break
            chunks.append(chunk)
            observed += len(chunk)
            if observed > expected_size:
                _integrity_fail(f"{label} changed size while being read")
    except OSError as error:
        raise ReportingIntegrityError(f"cannot read {label}: {error}") from error
    payload = b"".join(chunks)
    if len(payload) != expected_size:
        _integrity_fail(f"{label} changed size while being read")
    return payload


def _capture_paper_file(
    directory_fd: int, name: str, *, label: str
) -> _CapturedPaperFile:
    if not isinstance(name, str) or not name or Path(name).name != name:
        _integrity_fail(f"{label} has an unsafe filename")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(name, flags, dir_fd=directory_fd)
    except OSError as error:
        raise ReportingIntegrityError(f"cannot open {label}: {error}") from error
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            _integrity_fail(f"{label} must be a singly linked regular file")
        fingerprint = _paper_stat_fingerprint(before)
        payload = _read_captured_fd(fd, expected_size=before.st_size, label=label)
        after = os.fstat(fd)
        if _paper_stat_fingerprint(after) != fingerprint:
            _integrity_fail(f"{label} changed while being captured")
        return _CapturedPaperFile(name, fd, payload, fingerprint)
    except Exception:
        os.close(fd)
        raise


def _require_current_paper_file(
    directory_fd: int, captured: _CapturedPaperFile, *, label: str
) -> None:
    try:
        path_before = os.stat(captured.name, dir_fd=directory_fd, follow_symlinks=False)
        fd_before = os.fstat(captured.fd)
    except OSError as error:
        raise ReportingIntegrityError(f"cannot recheck {label}: {error}") from error
    if (
        _paper_stat_fingerprint(path_before) != captured.fingerprint
        or _paper_stat_fingerprint(fd_before) != captured.fingerprint
    ):
        _integrity_fail(f"{label} path changed during paper artifact audit")
    replay = _read_captured_fd(
        captured.fd,
        expected_size=fd_before.st_size,
        label=label,
    )
    try:
        path_after = os.stat(captured.name, dir_fd=directory_fd, follow_symlinks=False)
        fd_after = os.fstat(captured.fd)
    except OSError as error:
        raise ReportingIntegrityError(f"cannot finalize {label}: {error}") from error
    if (
        replay != captured.payload
        or _paper_stat_fingerprint(path_after) != captured.fingerprint
        or _paper_stat_fingerprint(fd_after) != captured.fingerprint
    ):
        _integrity_fail(f"{label} changed during paper artifact audit")


def audit_paper_artifacts(
    output_directory: Path, *, expected_manifest_sha256: str
) -> PaperArtifactAuditReport:
    """Audit the exact rendered file set against an external manifest anchor."""
    expected = _sha256(expected_manifest_sha256, "expected_manifest_sha256")
    directory = output_directory.expanduser()
    if directory.is_symlink() or not directory.is_dir():
        _integrity_fail("paper artifact output must be a regular directory")
    directory_flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_DIRECTORY", 0)
    )
    try:
        directory_fd = os.open(directory, directory_flags)
    except OSError as error:
        raise ReportingIntegrityError(
            f"cannot open paper artifact output: {error}"
        ) from error
    captured: list[tuple[_CapturedPaperFile, str]] = []
    try:
        directory_fingerprint = _paper_stat_fingerprint(os.fstat(directory_fd))
        manifest_file = _capture_paper_file(
            directory_fd,
            "artifact_manifest.json",
            label="paper artifact manifest",
        )
        captured.append((manifest_file, "paper artifact manifest"))
        observed = sha256_bytes(manifest_file.payload)
        if observed != expected:
            _integrity_fail("paper artifact manifest hash mismatch")
        seal_file = _capture_paper_file(
            directory_fd,
            "artifact_manifest.sha256",
            label="paper artifact manifest seal",
        )
        captured.append((seal_file, "paper artifact manifest seal"))
        try:
            seal = seal_file.payload.decode("ascii")
            mapping = read_json_bytes_object(
                manifest_file.payload,
                source=str(directory / manifest_file.name),
            )
        except (UnicodeError, MetadataValidationError) as error:
            raise ReportingIntegrityError(str(error)) from error
        if seal != f"{observed}\n":
            _integrity_fail("paper artifact manifest self-seal mismatch")
        manifest = PaperArtifactManifest.from_mapping(mapping)
        expected_names = {
            "artifact_manifest.json",
            "artifact_manifest.sha256",
            *(item.path for item in manifest.outputs),
        }
        observed_names = set(os.listdir(directory_fd))
        if observed_names != expected_names:
            _integrity_fail(
                "paper artifact file set changed: "
                f"missing={sorted(expected_names - observed_names)!r}, "
                f"extra={sorted(observed_names - expected_names)!r}"
            )
        for item in manifest.outputs:
            output_file = _capture_paper_file(
                directory_fd,
                item.path,
                label=f"paper output {item.path}",
            )
            captured.append((output_file, f"paper output {item.path}"))
            if sha256_bytes(output_file.payload) != item.sha256:
                _integrity_fail(f"paper output changed after rendering: {item.path}")
            if (
                item.source_artifact_sha256 != manifest.result_sha256
                or item.experiment_registry_sha256 != manifest.registry_sha256
                or item.figure_spec_sha256 != manifest.figure_spec_sha256
                or item.config_sha256 != manifest.config_sha256
                or item.study_protocol_sha256 != manifest.study_protocol_sha256
                or item.matrix_run_manifest_sha256
                != manifest.matrix_run_manifest_sha256
                or item.statistics_derivation_sha256
                != manifest.statistics_derivation_sha256
                or item.validity_certificate_sha256
                != manifest.validity_certificate_sha256
            ):
                _integrity_fail(
                    "generated output provenance disagrees with its manifest"
                )
        if set(os.listdir(directory_fd)) != expected_names:
            _integrity_fail("paper artifact file set changed during audit")
        for captured_file, captured_label in captured:
            _require_current_paper_file(
                directory_fd,
                captured_file,
                label=captured_label,
            )
        try:
            path_directory = os.stat(directory, follow_symlinks=False)
            fd_directory = os.fstat(directory_fd)
        except OSError as error:
            raise ReportingIntegrityError(
                f"cannot finalize paper artifact directory: {error}"
            ) from error
        if (
            _paper_stat_fingerprint(path_directory) != directory_fingerprint
            or _paper_stat_fingerprint(fd_directory) != directory_fingerprint
        ):
            _integrity_fail("paper artifact directory changed during audit")
        return PaperArtifactAuditReport(
            valid=True,
            manifest_sha256=observed,
            reporting_eligibility=manifest.reporting_eligibility,
            output_count=len(manifest.outputs),
        )
    finally:
        for captured_file, _ in captured:
            os.close(captured_file.fd)
        os.close(directory_fd)


__all__ = [
    "GeneratedPaperArtifact",
    "FormalReportingTrust",
    "PaperArtifactAuditReport",
    "PaperArtifactManifest",
    "REPORT_MANIFEST_SCHEMA",
    "REPORT_RESULT_SCHEMA",
    "SYNTHETIC_UNVERIFIED_RESULT_SCHEMA",
    "REQUIRED_REPORT_OUTPUTS",
    "ReportingIntegrityError",
    "ReportingPipelineError",
    "ReportingSchemaError",
    "audit_paper_artifacts",
    "generate_paper_artifacts",
]
