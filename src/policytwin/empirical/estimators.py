"""Deterministic, standard-library empirical estimators for frozen panels.

The estimators in this module implement a deliberately narrow common-timing
design.  Difference-in-differences and event-study regressions operate on
unit-level changes, optionally adjusted for characteristics measured strictly
before treatment.  The quantile estimand is the unconditional treated-minus-
control difference between change-score quantiles; it is not a quantile of
individual causal effects.

Every resampling decision is derived from an explicit :class:`EventKey` and is
independent of input row order or execution order.  Failed bootstrap and
randomization draws are retained as ``None`` rather than silently redrawn.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, replace
from statistics import fmean
from typing import NoReturn

from policytwin.counterfactual.randomness import EventKey, deterministic_int
from policytwin.empirical.schema import (
    BootstrapDraw,
    ClusterBootstrapAudit,
    ConfidenceInterval,
    DiagnosticStatus,
    DrawValue,
    EmpiricalEvaluationConfig,
    EmpiricalSchemaError,
    EmpiricalSpecification,
    EstimateStatus,
    FrozenPanel,
    NamedEstimate,
    ObservationStatus,
    PanelUnit,
    PreTrendDiagnostic,
    RandomizationDesign,
    RandomizationDraw,
    RandomizationInferenceResult,
    SubgroupDefinition,
)

_RANK_TOLERANCE = 1.0e-12


def _fail(message: str) -> NoReturn:
    raise EmpiricalSchemaError(message)


def _validate_positive_int(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        _fail(f"{field} must be a positive integer")
    return value


def _mean(values: Sequence[float], field: str) -> float:
    if not values:
        _fail(f"{field} must not be empty")
    try:
        result = fmean(values)
    except OverflowError as error:
        raise EmpiricalSchemaError(f"{field} overflowed") from error
    if not math.isfinite(result):
        _fail(f"{field} must have a finite mean")
    return result


def _quantile(values: Sequence[float], probability: float) -> float:
    """Return the deterministic linear empirical quantile (Hyndman-Fan 7)."""
    if not values:
        _fail("quantile input must not be empty")
    if not 0.0 < probability < 1.0 or not math.isfinite(probability):
        _fail("quantile probability must lie strictly between zero and one")
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower_index = math.floor(position)
    upper_index = math.ceil(position)
    if lower_index == upper_index:
        return ordered[lower_index]
    weight = position - lower_index
    result = ordered[lower_index] + weight * (
        ordered[upper_index] - ordered[lower_index]
    )
    if not math.isfinite(result):
        _fail("quantile calculation produced a non-finite value")
    return result


def _quantile_label(probability: float) -> str:
    return format(probability, ".15g")


def _event_name(period: int) -> str:
    return f"event_study[{period}]"


def _qte_name(probability: float) -> str:
    """Name the implemented estimand without implying individual effects."""
    return (
        "change_score_quantile_difference_treated_minus_control["
        f"{_quantile_label(probability)}]"
    )


def _subgroup_name(subgroup_id: str, estimate_name: str) -> str:
    return f"subgroup[{subgroup_id}].{estimate_name}"


@dataclass(frozen=True, slots=True)
class _UnitChange:
    unit_id: str
    cluster_id: str
    treated: bool
    pre_mean: float
    post_mean: float
    event_outcomes: tuple[tuple[int, float], ...]
    covariates: tuple[float, ...]
    characteristics: tuple[tuple[str, float], ...]

    @property
    def change(self) -> float:
        result = self.post_mean - self.pre_mean
        if not math.isfinite(result):
            _fail("unit change score is not finite")
        return result

    def event_outcome(self, period: int) -> float:
        for candidate, value in self.event_outcomes:
            if candidate == period:
                return value
        _fail(f"relative period {period} is unavailable for unit {self.unit_id!r}")

    def characteristic(self, name: str) -> float | None:
        for candidate, value in self.characteristics:
            if candidate == name:
                return value
        return None


@dataclass(frozen=True, slots=True)
class _RegressionEstimate:
    point: float | None
    status: EstimateStatus
    reason: str | None


def _validate_call(
    config: EmpiricalEvaluationConfig,
    panel: FrozenPanel,
    specification: EmpiricalSpecification,
    unit_ids: tuple[str, ...],
    subgroups: tuple[SubgroupDefinition, ...] | None = None,
) -> tuple[str, ...]:
    if not isinstance(config, EmpiricalEvaluationConfig):
        _fail("config must be an EmpiricalEvaluationConfig")
    if not isinstance(panel, FrozenPanel):
        _fail("panel must be a FrozenPanel")
    if not isinstance(specification, EmpiricalSpecification):
        _fail("specification must be an EmpiricalSpecification")
    frozen_specifications = {
        item.specification_id: item for item in config.specifications
    }
    if frozen_specifications.get(specification.specification_id) != specification:
        _fail("specification is not an unchanged member of the frozen config")
    if not isinstance(unit_ids, tuple):
        _fail("unit_ids must be an immutable tuple")
    if not all(isinstance(unit_id, str) for unit_id in unit_ids):
        _fail("unit_ids must contain strings")
    if len(set(unit_ids)) != len(unit_ids):
        _fail("unit_ids must not contain duplicates")
    frozen_cohort = set(config.treatment.unit_ids)
    panel_units = {unit.unit_id for unit in panel.units}
    if not frozen_cohort <= panel_units:
        _fail("the panel is missing members of the frozen treatment cohort")
    unknown_units = sorted(set(unit_ids) - frozen_cohort)
    if unknown_units:
        _fail(
            f"unit_ids contain units outside the frozen cohort: {unknown_units[:3]!r}"
        )
    missing_periods = sorted(
        set(specification.required_periods) - set(panel.expected_periods)
    )
    if missing_periods:
        _fail(f"panel does not contain frozen required periods: {missing_periods!r}")
    if subgroups is None:
        return tuple(sorted(unit_ids))
    if not isinstance(subgroups, tuple) or not all(
        isinstance(subgroup, SubgroupDefinition) for subgroup in subgroups
    ):
        _fail("subgroups must be an immutable tuple of SubgroupDefinition")
    config_subgroups = {item.subgroup_id: item for item in config.subgroups}
    passed_ids = tuple(item.subgroup_id for item in subgroups)
    if len(set(passed_ids)) != len(passed_ids):
        _fail("subgroups must not contain duplicates")
    for subgroup in subgroups:
        if config_subgroups.get(subgroup.subgroup_id) != subgroup:
            _fail("subgroup is not an unchanged member of the frozen config")
    if set(passed_ids) != set(specification.subgroup_ids):
        _fail("subgroups must exactly match the specification's frozen subgroup IDs")
    return tuple(sorted(unit_ids))


def _characteristics(unit: PanelUnit) -> tuple[tuple[str, float], ...]:
    return tuple(
        sorted(
            (characteristic.name, characteristic.value)
            for characteristic in unit.characteristics
        )
    )


def _prepare_units(
    config: EmpiricalEvaluationConfig,
    panel: FrozenPanel,
    specification: EmpiricalSpecification,
    unit_ids: tuple[str, ...],
) -> tuple[_UnitChange, ...]:
    """Build one order-independent complete-case sample for the full spec."""
    result: list[_UnitChange] = []
    required_periods = tuple(sorted(specification.required_periods))
    pre_periods = tuple(sorted(specification.pre_periods))
    post_periods = tuple(sorted(specification.post_periods))
    event_periods = tuple(sorted(specification.event_periods))
    for unit_id in unit_ids:
        unit = panel.unit(unit_id)
        outcomes: dict[int, float] = {}
        complete = True
        for period in required_periods:
            observation = panel.observation(unit_id, period)
            if (
                observation.status is not ObservationStatus.OBSERVED
                or observation.outcome is None
            ):
                complete = False
                break
            outcomes[period] = observation.outcome
        if not complete:
            continue
        covariates: list[float] = []
        for name in specification.covariates:
            value = unit.characteristic(name)
            if value is None:
                complete = False
                break
            covariates.append(value)
        if not complete:
            continue
        result.append(
            _UnitChange(
                unit_id=unit_id,
                cluster_id=unit.cluster_id,
                treated=config.treatment.is_treated(unit_id),
                pre_mean=_mean(
                    [outcomes[period] for period in pre_periods],
                    "pre-treatment outcomes",
                ),
                post_mean=_mean(
                    [outcomes[period] for period in post_periods],
                    "post-treatment outcomes",
                ),
                event_outcomes=tuple(
                    (period, outcomes[period]) for period in event_periods
                ),
                covariates=tuple(covariates),
                characteristics=_characteristics(unit),
            )
        )
    return tuple(result)


def _arm_counts(units: Sequence[_UnitChange]) -> tuple[int, int]:
    treated = sum(unit.treated for unit in units)
    return treated, len(units) - treated


def _dot(left: Sequence[float], right: Sequence[float]) -> float:
    return math.fsum(first * second for first, second in zip(left, right, strict=True))


def _ols_treatment_coefficient(
    outcomes: Sequence[float], units: Sequence[_UnitChange]
) -> _RegressionEstimate:
    """Fit ``outcome ~ intercept + treatment + frozen covariates`` by QR."""
    n_treated, n_control = _arm_counts(units)
    if n_treated == 0 or n_control == 0:
        return _RegressionEstimate(
            point=None,
            status=EstimateStatus.INSUFFICIENT_DATA,
            reason="both treatment arms require complete observations",
        )
    if len(outcomes) != len(units):
        _fail("outcome and unit vectors have different lengths")
    covariate_count = len(units[0].covariates)
    if any(len(unit.covariates) != covariate_count for unit in units):
        _fail("unit covariate vectors have inconsistent lengths")
    if covariate_count == 0:
        treated_values = [
            outcome
            for outcome, unit in zip(outcomes, units, strict=True)
            if unit.treated
        ]
        control_values = [
            outcome
            for outcome, unit in zip(outcomes, units, strict=True)
            if not unit.treated
        ]
        point = _mean(treated_values, "treated outcomes") - _mean(
            control_values, "control outcomes"
        )
        if not math.isfinite(point):
            return _RegressionEstimate(
                point=None,
                status=EstimateStatus.SINGULAR_DESIGN,
                reason="difference in means is not finite",
            )
        return _RegressionEstimate(point, EstimateStatus.ESTIMATED, None)

    design_rows = [(1.0, float(unit.treated), *unit.covariates) for unit in units]
    column_count = len(design_rows[0])
    if len(design_rows) < column_count:
        return _RegressionEstimate(
            point=None,
            status=EstimateStatus.INSUFFICIENT_DATA,
            reason="fewer complete units than regression columns",
        )
    columns = [
        [row[column_index] for row in design_rows]
        for column_index in range(column_count)
    ]
    q_columns: list[list[float]] = []
    upper = [[0.0] * column_count for _ in range(column_count)]
    maximum_norm = max(math.sqrt(_dot(column, column)) for column in columns)
    tolerance = _RANK_TOLERANCE * max(1.0, maximum_norm)
    for column_index, source in enumerate(columns):
        residual = list(source)
        for previous_index, q_column in enumerate(q_columns):
            projection = _dot(q_column, residual)
            upper[previous_index][column_index] += projection
            residual = [
                value - projection * basis
                for value, basis in zip(residual, q_column, strict=True)
            ]
        # One re-orthogonalization pass makes the small dense solver less
        # sensitive to near-collinearity without changing column semantics.
        for previous_index, q_column in enumerate(q_columns):
            correction = _dot(q_column, residual)
            upper[previous_index][column_index] += correction
            residual = [
                value - correction * basis
                for value, basis in zip(residual, q_column, strict=True)
            ]
        norm = math.sqrt(_dot(residual, residual))
        if not math.isfinite(norm) or norm <= tolerance:
            return _RegressionEstimate(
                point=None,
                status=EstimateStatus.SINGULAR_DESIGN,
                reason="frozen covariate design is rank deficient",
            )
        upper[column_index][column_index] = norm
        q_columns.append([value / norm for value in residual])
    projected_outcome = [_dot(column, outcomes) for column in q_columns]
    coefficients = [0.0] * column_count
    for row_index in range(column_count - 1, -1, -1):
        remainder = math.fsum(
            upper[row_index][column_index] * coefficients[column_index]
            for column_index in range(row_index + 1, column_count)
        )
        coefficient = (projected_outcome[row_index] - remainder) / upper[row_index][
            row_index
        ]
        if not math.isfinite(coefficient):
            return _RegressionEstimate(
                point=None,
                status=EstimateStatus.SINGULAR_DESIGN,
                reason="regression coefficient is not finite",
            )
        coefficients[row_index] = coefficient
    return _RegressionEstimate(coefficients[1], EstimateStatus.ESTIMATED, None)


def _named_regression(
    name: str, outcomes: Sequence[float], units: Sequence[_UnitChange]
) -> NamedEstimate:
    n_treated, n_control = _arm_counts(units)
    regression = _ols_treatment_coefficient(outcomes, units)
    return NamedEstimate(
        name=name,
        point=regression.point,
        status=regression.status,
        n_treated=n_treated,
        n_control=n_control,
        reason=regression.reason,
    )


def _estimate_set(
    units: Sequence[_UnitChange], specification: EmpiricalSpecification
) -> tuple[NamedEstimate, ...]:
    n_treated, n_control = _arm_counts(units)
    estimates: list[NamedEstimate] = [
        _named_regression("did", [unit.change for unit in units], units)
    ]
    reference = specification.reference_period
    for period in sorted(specification.event_periods):
        if period == reference:
            continue
        estimates.append(
            _named_regression(
                _event_name(period),
                [
                    unit.event_outcome(period) - unit.event_outcome(reference)
                    for unit in units
                ],
                units,
            )
        )
    treated_changes = [unit.change for unit in units if unit.treated]
    control_changes = [unit.change for unit in units if not unit.treated]
    for probability in specification.quantiles:
        name = _qte_name(probability)
        if not treated_changes or not control_changes:
            estimates.append(
                NamedEstimate(
                    name=name,
                    point=None,
                    status=EstimateStatus.INSUFFICIENT_DATA,
                    n_treated=n_treated,
                    n_control=n_control,
                    reason="both treatment arms require complete change scores",
                )
            )
            continue
        point = _quantile(treated_changes, probability) - _quantile(
            control_changes, probability
        )
        estimates.append(
            NamedEstimate(
                name=name,
                point=point,
                status=EstimateStatus.ESTIMATED,
                n_treated=n_treated,
                n_control=n_control,
            )
        )
    return tuple(estimates)


def _subgroup_units(
    units: Sequence[_UnitChange], subgroup: SubgroupDefinition
) -> tuple[_UnitChange, ...]:
    selected: list[_UnitChange] = []
    for unit in units:
        value = unit.characteristic(subgroup.characteristic_name)
        if value is not None and subgroup.contains(value):
            selected.append(unit)
    return tuple(selected)


def _all_estimates(
    units: Sequence[_UnitChange],
    specification: EmpiricalSpecification,
    subgroups: tuple[SubgroupDefinition, ...],
) -> tuple[NamedEstimate, ...]:
    results = list(_estimate_set(units, specification))
    for subgroup in sorted(subgroups, key=lambda value: value.subgroup_id):
        for estimate in _estimate_set(_subgroup_units(units, subgroup), specification):
            results.append(
                replace(
                    estimate,
                    name=_subgroup_name(subgroup.subgroup_id, estimate.name),
                )
            )
    return tuple(results)


def point_estimates(
    config: EmpiricalEvaluationConfig,
    panel: FrozenPanel,
    specification: EmpiricalSpecification,
    unit_ids: tuple[str, ...],
    subgroups: tuple[SubgroupDefinition, ...] = (),
) -> tuple[NamedEstimate, ...]:
    """Return DiD, event-study, quantile-contrast, and frozen subgroup estimates."""
    canonical_ids = _validate_call(config, panel, specification, unit_ids, subgroups)
    units = _prepare_units(config, panel, specification, canonical_ids)
    return _all_estimates(units, specification, subgroups)


def evaluate_estimates(
    config: EmpiricalEvaluationConfig,
    panel: FrozenPanel,
    specification: EmpiricalSpecification,
    unit_ids: tuple[str, ...],
    subgroups: tuple[SubgroupDefinition, ...] = (),
) -> tuple[NamedEstimate, ...]:
    """Compatibility alias for :func:`point_estimates`."""
    return point_estimates(config, panel, specification, unit_ids, subgroups)


def _resampling_key(
    *,
    config: EmpiricalEvaluationConfig,
    specification: EmpiricalSpecification,
    replicate: int,
    simulation_step: int,
    purpose: str,
) -> EventKey:
    return EventKey(
        experiment_id=config.study_id,
        seed=config.random_seed,
        replicate_id=replicate,
        simulation_step=simulation_step,
        purpose=f"empirical.{purpose}.{specification.specification_id}",
    )


def cluster_bootstrap(
    config: EmpiricalEvaluationConfig,
    panel: FrozenPanel,
    specification: EmpiricalSpecification,
    unit_ids: tuple[str, ...],
    subgroups: tuple[SubgroupDefinition, ...] = (),
) -> tuple[BootstrapDraw, ...]:
    """Resample arm-disjoint clusters and retain every requested full draw.

    Treatment and control clusters are sampled independently so a draw cannot
    lose an arm merely because of pooled-cluster sampling.  A cluster containing
    both arms is not silently split: every value is undefined and the separate
    inference audit records the support violation.
    """
    canonical_ids = _validate_call(config, panel, specification, unit_ids, subgroups)
    repetitions = _validate_positive_int(
        config.bootstrap_repetitions, "bootstrap_repetitions"
    )
    units = _prepare_units(config, panel, specification, canonical_ids)
    template = _all_estimates(units, specification, subgroups)
    treated_cluster_set = {unit.cluster_id for unit in units if unit.treated}
    control_cluster_set = {unit.cluster_id for unit in units if not unit.treated}
    mixed_clusters = treated_cluster_set & control_cluster_set
    treated_clusters = tuple(sorted(treated_cluster_set - mixed_clusters))
    control_clusters = tuple(sorted(control_cluster_set - mixed_clusters))
    clusters = tuple(sorted(treated_cluster_set | control_cluster_set))
    by_cluster = {
        cluster_id: tuple(unit for unit in units if unit.cluster_id == cluster_id)
        for cluster_id in clusters
    }
    draws: list[BootstrapDraw] = []
    for replicate in range(repetitions):
        root_key = _resampling_key(
            config=config,
            specification=specification,
            replicate=replicate,
            simulation_step=0,
            purpose="cluster_bootstrap.draw",
        )
        sampled_units: list[_UnitChange] = []
        if not mixed_clusters and treated_clusters and control_clusters:
            for arm, arm_clusters in (
                ("treated", treated_clusters),
                ("control", control_clusters),
            ):
                for pick_index in range(len(arm_clusters)):
                    pick_key = _resampling_key(
                        config=config,
                        specification=specification,
                        replicate=replicate,
                        simulation_step=pick_index,
                        purpose=f"cluster_bootstrap.{arm}.pick",
                    )
                    selected = deterministic_int(pick_key, 0, len(arm_clusters) - 1)
                    sampled_units.extend(by_cluster[arm_clusters[selected]])
        if sampled_units:
            sampled_estimates = _all_estimates(sampled_units, specification, subgroups)
            if tuple(item.name for item in sampled_estimates) != tuple(
                item.name for item in template
            ):
                _fail("bootstrap draw changed the frozen estimate set")
            values = tuple(
                DrawValue(name=item.name, value=item.point)
                for item in sampled_estimates
            )
        else:
            values = tuple(DrawValue(name=item.name, value=None) for item in template)
        draws.append(
            BootstrapDraw(
                replicate=replicate,
                event_key=root_key.value,
                values=values,
            )
        )
    return tuple(draws)


def cluster_bootstrap_audit(
    config: EmpiricalEvaluationConfig,
    panel: FrozenPanel,
    specification: EmpiricalSpecification,
    unit_ids: tuple[str, ...],
    draws: tuple[BootstrapDraw, ...],
    subgroups: tuple[SubgroupDefinition, ...] = (),
) -> ClusterBootstrapAudit:
    """Derive cluster support and draw accounting from frozen typed inputs."""
    canonical_ids = _validate_call(config, panel, specification, unit_ids, subgroups)
    units = _prepare_units(config, panel, specification, canonical_ids)
    treated_cluster_set = {unit.cluster_id for unit in units if unit.treated}
    control_cluster_set = {unit.cluster_id for unit in units if not unit.treated}
    mixed = tuple(sorted(treated_cluster_set & control_cluster_set))
    treated_only = treated_cluster_set - set(mixed)
    control_only = control_cluster_set - set(mixed)
    if not isinstance(draws, tuple):
        _fail("bootstrap draws must be an immutable tuple")
    if len(draws) != config.bootstrap_repetitions:
        _fail("retained bootstrap draws differ from bootstrap_repetitions")
    valid_draws = sum(
        bool(draw.values) and all(value.value is not None for value in draw.values)
        for draw in draws
    )
    return ClusterBootstrapAudit(
        total_clusters=len(treated_cluster_set | control_cluster_set),
        treated_clusters=len(treated_only),
        control_clusters=len(control_only),
        mixed_cluster_ids=mixed,
        minimum_total_clusters=config.minimum_total_clusters,
        minimum_treated_clusters=config.minimum_treated_clusters,
        minimum_control_clusters=config.minimum_control_clusters,
        requested_draws=len(draws),
        valid_draws=valid_draws,
        invalid_draws=len(draws) - valid_draws,
        minimum_valid_draws=config.minimum_valid_bootstrap_draws,
        maximum_invalid_fraction=config.maximum_invalid_bootstrap_fraction,
    )


def confidence_intervals(
    estimates: tuple[NamedEstimate, ...],
    draws: tuple[BootstrapDraw, ...],
    confidence_level: float,
    *,
    minimum_valid_draws: int = 1,
    inference_gate_passed: bool = True,
) -> tuple[ConfidenceInterval, ...]:
    """Return intervals only after the frozen valid bootstrap-draw floor is met."""
    if not isinstance(estimates, tuple) or not isinstance(draws, tuple):
        _fail("estimates and draws must be immutable tuples")
    if not 0.0 < confidence_level < 1.0 or not math.isfinite(confidence_level):
        _fail("confidence_level must lie strictly between zero and one")
    minimum_valid_draws = _validate_positive_int(
        minimum_valid_draws, "minimum_valid_draws"
    )
    if not isinstance(inference_gate_passed, bool):
        _fail("inference_gate_passed must be a boolean")
    if minimum_valid_draws > len(draws):
        _fail("minimum_valid_draws must not exceed the retained bootstrap draws")
    names = tuple(estimate.name for estimate in estimates)
    if len(set(names)) != len(names):
        _fail("estimate names must be unique")
    for draw in draws:
        if tuple(value.name for value in draw.values) != names:
            _fail("bootstrap draw names do not match the frozen estimate set")
    lower_probability = (1.0 - confidence_level) / 2.0
    upper_probability = 1.0 - lower_probability
    intervals: list[ConfidenceInterval] = []
    for index, estimate in enumerate(estimates):
        valid = tuple(
            draw.values[index].value
            for draw in draws
            if draw.values[index].value is not None
        )
        lower: float | None = None
        upper: float | None = None
        if (
            estimate.status is EstimateStatus.ESTIMATED
            and len(valid) >= minimum_valid_draws
            and inference_gate_passed
        ):
            normalized = tuple(value for value in valid if value is not None)
            lower = _quantile(normalized, lower_probability)
            upper = _quantile(normalized, upper_probability)
        intervals.append(
            ConfidenceInterval(
                name=estimate.name,
                confidence_level=confidence_level,
                lower=lower,
                upper=upper,
                valid_draws=len(valid),
                minimum_valid_draws=minimum_valid_draws,
                inference_gate_passed=inference_gate_passed,
            )
        )
    return tuple(intervals)


def _did_estimate(units: Sequence[_UnitChange]) -> _RegressionEstimate:
    return _ols_treatment_coefficient([unit.change for unit in units], units)


def _permuted_units(
    units: tuple[_UnitChange, ...],
    *,
    config: EmpiricalEvaluationConfig,
    specification: EmpiricalSpecification,
    replicate: int,
    treated_count: int,
) -> tuple[_UnitChange, ...]:
    identifiers = [unit.unit_id for unit in units]
    for step, upper in enumerate(range(len(identifiers) - 1, 0, -1)):
        key = _resampling_key(
            config=config,
            specification=specification,
            replicate=replicate,
            simulation_step=step,
            purpose="randomization.shuffle",
        )
        selected = deterministic_int(key, 0, upper)
        identifiers[upper], identifiers[selected] = (
            identifiers[selected],
            identifiers[upper],
        )
    treated_ids = set(identifiers[:treated_count])
    return tuple(replace(unit, treated=unit.unit_id in treated_ids) for unit in units)


def randomization_inference(
    config: EmpiricalEvaluationConfig,
    panel: FrozenPanel,
    specification: EmpiricalSpecification,
    unit_ids: tuple[str, ...],
) -> RandomizationInferenceResult:
    """Run unit-level complete-randomization inference for the DiD statistic."""
    if config.treatment.randomization_design is not RandomizationDesign.UNIT_COMPLETE:
        _fail("only explicitly configured unit-complete randomization is supported")
    canonical_ids = _validate_call(config, panel, specification, unit_ids)
    repetitions = _validate_positive_int(
        config.randomization_repetitions, "randomization_repetitions"
    )
    units = _prepare_units(config, panel, specification, canonical_ids)
    observed = _did_estimate(units)
    treated_count, control_count = _arm_counts(units)
    draws: list[RandomizationDraw] = []
    for replicate in range(repetitions):
        root_key = _resampling_key(
            config=config,
            specification=specification,
            replicate=replicate,
            simulation_step=0,
            purpose="randomization.draw",
        )
        statistic: float | None = None
        if treated_count > 0 and control_count > 0:
            permuted = _permuted_units(
                units,
                config=config,
                specification=specification,
                replicate=replicate,
                treated_count=treated_count,
            )
            estimate = _did_estimate(permuted)
            statistic = estimate.point
        draws.append(
            RandomizationDraw(
                replicate=replicate,
                event_key=root_key.value,
                statistic=statistic,
            )
        )
    valid_statistics = tuple(
        draw.statistic for draw in draws if draw.statistic is not None
    )
    p_value: float | None = None
    if (
        observed.status is EstimateStatus.ESTIMATED
        and observed.point is not None
        and len(valid_statistics) == len(draws)
        and len(valid_statistics) >= config.minimum_valid_randomization_draws
    ):
        numerical_tolerance = _RANK_TOLERANCE * max(1.0, abs(observed.point))
        extreme_threshold = max(0.0, abs(observed.point) - numerical_tolerance)
        extreme = sum(
            abs(statistic) >= extreme_threshold
            for statistic in valid_statistics
            if statistic is not None
        )
        p_value = (extreme + 1.0) / (len(valid_statistics) + 1.0)
    return RandomizationInferenceResult(
        observed_statistic=observed.point,
        p_value=p_value,
        draws=tuple(draws),
        requested_draws=repetitions,
        minimum_valid_draws=config.minimum_valid_randomization_draws,
    )


def _slope(coordinates: Sequence[float], outcomes: Sequence[float]) -> float | None:
    if len(coordinates) != len(outcomes) or len(coordinates) < 2:
        return None
    x_mean = _mean(coordinates, "pretrend time coordinates")
    y_mean = _mean(outcomes, "pretrend outcomes")
    denominator = math.fsum((coordinate - x_mean) ** 2 for coordinate in coordinates)
    if denominator <= 0.0 or not math.isfinite(denominator):
        return None
    numerator = math.fsum(
        (coordinate - x_mean) * (outcome - y_mean)
        for coordinate, outcome in zip(coordinates, outcomes, strict=True)
    )
    result = numerator / denominator
    return result if math.isfinite(result) else None


def pretrend_diagnostic(
    config: EmpiricalEvaluationConfig,
    panel: FrozenPanel,
    specification: EmpiricalSpecification,
    unit_ids: tuple[str, ...],
) -> PreTrendDiagnostic:
    """Evaluate all registered pre-treatment leads and the arm slope gap."""
    canonical_ids = _validate_call(config, panel, specification, unit_ids)
    units = _prepare_units(config, panel, specification, canonical_ids)
    pre_periods = tuple(
        period for period in sorted(specification.event_periods) if period < 0
    )
    lead_periods = tuple(
        period for period in pre_periods if period != specification.reference_period
    )
    lead_values: list[DrawValue] = []
    for period in lead_periods:
        estimate = _ols_treatment_coefficient(
            [
                unit.event_outcome(period)
                - unit.event_outcome(specification.reference_period)
                for unit in units
            ],
            units,
        )
        lead_values.append(DrawValue(_event_name(period), estimate.point))
    if not lead_values or any(value.value is None for value in lead_values):
        return PreTrendDiagnostic(
            status=DiagnosticStatus.INSUFFICIENT_DATA,
            lead_estimates=tuple(lead_values),
            max_absolute_lead=None,
            treated_control_slope_gap=None,
            max_absolute_lead_tolerance=(specification.max_absolute_lead_tolerance),
            max_slope_gap_per_day=specification.max_slope_gap_per_day,
            reason="registered pre-treatment leads are not jointly estimable",
        )
    treated_units = tuple(unit for unit in units if unit.treated)
    control_units = tuple(unit for unit in units if not unit.treated)
    treated_means = [
        _mean(
            [unit.event_outcome(period) for unit in treated_units],
            "treated pretrend outcomes",
        )
        for period in pre_periods
    ]
    control_means = [
        _mean(
            [unit.event_outcome(period) for unit in control_units],
            "control pretrend outcomes",
        )
        for period in pre_periods
    ]
    pretrend_coordinates = tuple(
        config.pretrend_coordinate(period) for period in pre_periods
    )
    treated_slope = _slope(pretrend_coordinates, treated_means)
    control_slope = _slope(pretrend_coordinates, control_means)
    slope_gap = (
        treated_slope - control_slope
        if treated_slope is not None and control_slope is not None
        else None
    )
    if slope_gap is None:
        return PreTrendDiagnostic(
            status=DiagnosticStatus.INSUFFICIENT_DATA,
            lead_estimates=tuple(lead_values),
            max_absolute_lead=max(
                abs(value.value) for value in lead_values if value.value is not None
            ),
            treated_control_slope_gap=None,
            max_absolute_lead_tolerance=(specification.max_absolute_lead_tolerance),
            max_slope_gap_per_day=specification.max_slope_gap_per_day,
            reason="at least two pre-treatment periods are required for a slope gap",
        )
    max_absolute = max(
        abs(value.value) for value in lead_values if value.value is not None
    )
    passed = (
        max_absolute <= specification.max_absolute_lead_tolerance
        and abs(slope_gap) <= specification.max_slope_gap_per_day
    )
    return PreTrendDiagnostic(
        status=DiagnosticStatus.PASS if passed else DiagnosticStatus.FAIL,
        lead_estimates=tuple(lead_values),
        max_absolute_lead=max_absolute,
        treated_control_slope_gap=slope_gap,
        max_absolute_lead_tolerance=specification.max_absolute_lead_tolerance,
        max_slope_gap_per_day=specification.max_slope_gap_per_day,
        reason=None if passed else "a frozen pretrend threshold was exceeded",
    )


__all__ = [
    "cluster_bootstrap",
    "cluster_bootstrap_audit",
    "confidence_intervals",
    "evaluate_estimates",
    "point_estimates",
    "pretrend_diagnostic",
    "randomization_inference",
]
