"""Fail-closed orchestration for the frozen empirical specification curve."""

from __future__ import annotations

from collections import Counter

from policytwin.empirical.authorization import FrozenEvaluationAuthorization
from policytwin.empirical.estimators import (
    cluster_bootstrap,
    cluster_bootstrap_audit,
    confidence_intervals,
    point_estimates,
    pretrend_diagnostic,
    randomization_inference,
)
from policytwin.empirical.schema import (
    CausalReportingDecision,
    CausalReportingStatus,
    ClusterBootstrapAudit,
    ConfidenceInterval,
    DerivationEvidenceClass,
    DiagnosticStatus,
    EmpiricalEvaluationArtifact,
    EmpiricalEvaluationConfig,
    EmpiricalEvaluationError,
    EmpiricalSpecification,
    EstimateStatus,
    FreezeMode,
    FrozenEvaluationInputs,
    FrozenInputError,
    FrozenPanel,
    MissingDataPolicy,
    NamedEstimate,
    ObservationStatus,
    PeriodCompositionAudit,
    PreTrendDiagnostic,
    RandomizationInferenceResult,
    SampleAudit,
    SpecificationResult,
    SubgroupDefinition,
)
from policytwin.provenance import canonical_json_bytes, sha256_bytes


def _validate_frozen_inputs(
    config: EmpiricalEvaluationConfig,
    panel: FrozenPanel,
    frozen_inputs: FrozenEvaluationInputs,
) -> None:
    if frozen_inputs.config_sha256 != config.sha256:
        raise FrozenInputError("empirical config changed after the freeze receipt")
    if frozen_inputs.panel_sha256 != panel.sha256:
        raise FrozenInputError("empirical panel changed after the freeze receipt")
    if frozen_inputs.policy_start != config.treatment.policy_start:
        raise FrozenInputError("policy start changed after the freeze receipt")
    if frozen_inputs.panel_artifact_sha256 != panel.sha256:
        raise FrozenInputError("frozen panel artifact commitment changed")
    if (
        frozen_inputs.panel_derivation_registry_sha256
        != panel.derivation_registry_sha256
    ):
        raise FrozenInputError("panel derivation registry changed after authorization")
    if frozen_inputs.panel_derivation_receipt_count != len(panel.derivation_receipts):
        raise FrozenInputError("panel derivation receipt count changed")
    if frozen_inputs.mode is FreezeMode.SYNTHETIC_TEST_FIXTURE:
        if not config.synthetic_test_only or not panel.synthetic_only:
            raise FrozenInputError(
                "synthetic receipt cannot authorize unmarked empirical inputs"
            )
        if panel.schema_version != 1 or panel.derivation_receipts:
            raise FrozenInputError(
                "direct synthetic evaluation requires an unverified v1 panel"
            )
    elif config.synthetic_test_only != panel.synthetic_only:
        raise FrozenInputError("config and panel synthetic-only markers must agree")
    elif panel.schema_version != 2 or not panel.derivation_receipts:
        raise FrozenInputError(
            "verified snapshot evaluation requires a receipt-backed v2 panel"
        )
    if panel.expected_periods != config.required_periods:
        raise FrozenInputError(
            "frozen panel period grid must equal the configured specification union"
        )
    cohort = tuple(sorted(unit.unit_id for unit in panel.units))
    if cohort != config.treatment.unit_ids:
        raise FrozenInputError(
            "frozen panel cohort differs from the configured treatment assignment"
        )

    policy_start = config.treatment.policy_start
    cutoff = frozen_inputs.knowledge_cutoff
    subgroup_characteristics = {
        subgroup.characteristic_name for subgroup in config.subgroups
    }
    for unit in panel.units:
        for characteristic in unit.characteristics:
            record = characteristic.record
            if record.event_time >= policy_start:
                raise FrozenInputError(
                    "subgroup/covariate characteristic was measured after treatment"
                )
            if record.available_time >= policy_start:
                raise FrozenInputError(
                    "subgroup/covariate characteristic was not available pre-treatment"
                )
            if record.available_time > cutoff or record.event_time > cutoff:
                raise FrozenInputError("characteristic crosses the knowledge cutoff")
    for observation in panel.observations:
        record = observation.record
        if record.available_time > cutoff:
            raise FrozenInputError("observation available_time is after the cutoff")
        if record.event_time > cutoff:
            raise FrozenInputError("observation event_time is after the cutoff")
        if observation.relative_period < 0 and record.event_time >= policy_start:
            raise FrozenInputError(
                "negative relative period has a non-pre-treatment event_time"
            )
        if observation.relative_period >= 0 and record.event_time < policy_start:
            raise FrozenInputError(
                "non-negative relative period has a pre-treatment event_time"
            )
        mapped_event_time = config.event_time_for_period(observation.relative_period)
        if mapped_event_time is not None and record.event_time != mapped_event_time:
            raise FrozenInputError(
                "observation event_time differs from the frozen period_time_mapping"
            )
        group = observation.post_treatment_group
        if group is not None and group.name in subgroup_characteristics:
            raise FrozenInputError(
                "a post-treatment group name collides with a causal subgroup "
                "characteristic"
            )


def _transition_counts(
    panel: FrozenPanel,
) -> tuple[
    dict[tuple[str, int], int], dict[tuple[str, int], int], dict[tuple[str, int], int]
]:
    entrants: dict[tuple[str, int], int] = {}
    exits: dict[tuple[str, int], int] = {}
    reentries: dict[tuple[str, int], int] = {}
    for unit in panel.units:
        prior_observed = False
        prior_status: ObservationStatus | None = None
        prior_not_in_sample = False
        for period in panel.expected_periods:
            status = panel.observation(unit.unit_id, period).status
            key = (unit.unit_id, period)
            if status is ObservationStatus.OBSERVED:
                if not prior_observed and prior_not_in_sample:
                    entrants[key] = 1
                elif prior_observed and prior_status is not ObservationStatus.OBSERVED:
                    reentries[key] = 1
                prior_observed = True
            if (
                status is ObservationStatus.EXITED
                and prior_status is not ObservationStatus.EXITED
            ):
                exits[key] = 1
            if status is ObservationStatus.NOT_IN_SAMPLE:
                prior_not_in_sample = True
            prior_status = status
    return entrants, exits, reentries


def _characteristic_names(
    specification: EmpiricalSpecification,
    subgroups: tuple[SubgroupDefinition, ...],
) -> tuple[str, ...]:
    return tuple(
        sorted(
            set(specification.covariates)
            | {subgroup.characteristic_name for subgroup in subgroups}
        )
    )


def _sample_audit(
    config: EmpiricalEvaluationConfig,
    panel: FrozenPanel,
    specification: EmpiricalSpecification,
    subgroups: tuple[SubgroupDefinition, ...],
) -> SampleAudit:
    required_periods = specification.required_periods
    names = _characteristic_names(specification, subgroups)
    excluded: list[tuple[str, str]] = []
    complete: list[str] = []
    post_policy_observation_exclusion_unit_ids: list[str] = []
    registered_covariate_exclusion_unit_ids: list[str] = []
    for unit in sorted(panel.units, key=lambda value: value.unit_id):
        unavailable = tuple(
            (period, panel.observation(unit.unit_id, period).status)
            for period in required_periods
            if panel.observation(unit.unit_id, period).status
            is not ObservationStatus.OBSERVED
        )
        missing_covariates = tuple(
            name
            for name in specification.covariates
            if unit.characteristic(name) is None
        )
        if missing_covariates:
            registered_covariate_exclusion_unit_ids.append(unit.unit_id)
        exclusion_reasons: list[str] = []
        if unavailable:
            if any(period >= 0 for period, _ in unavailable):
                post_policy_observation_exclusion_unit_ids.append(unit.unit_id)
            exclusion_reasons.append(
                "non_observed_required_period:"
                + ",".join(f"{period}={status.value}" for period, status in unavailable)
            )
        if missing_covariates:
            exclusion_reasons.append(
                "missing_covariates:" + ",".join(missing_covariates)
            )
        if exclusion_reasons:
            excluded.append((unit.unit_id, ";".join(exclusion_reasons)))
        else:
            complete.append(unit.unit_id)
    if config.missing_data_policy is MissingDataPolicy.ERROR and excluded:
        raise EmpiricalEvaluationError(
            "frozen missing_data_policy=error rejected non-complete observations; "
            f"first exclusion={excluded[0]!r}"
        )

    entrants, exits, reentries = _transition_counts(panel)
    period_audits: list[PeriodCompositionAudit] = []
    max_missing = 0.0
    max_attrition = 0.0
    max_shift = 0.0
    flags: set[str] = set()
    subgroup_by_id = {item.subgroup_id: item for item in subgroups}
    for subgroup_id, subgroup in subgroup_by_id.items():
        unclassified = sum(
            panel.unit(unit_id).characteristic(subgroup.characteristic_name) is None
            for unit_id in config.treatment.unit_ids
        )
        if unclassified:
            flags.add(f"unclassified_subgroup:{subgroup_id}:{unclassified}")

    for period in required_periods:
        for arm, arm_ids in (
            ("treated", config.treatment.treated_unit_ids),
            ("control", config.treatment.control_unit_ids),
        ):
            cells = tuple(panel.observation(unit_id, period) for unit_id in arm_ids)
            observed_ids = tuple(
                sorted(
                    cell.unit_id
                    for cell in cells
                    if cell.status is ObservationStatus.OBSERVED
                )
            )
            counts = Counter(cell.status for cell in cells)
            expected = len(cells)
            observed = counts[ObservationStatus.OBSERVED]
            missing_fraction = 1.0 - observed / expected
            attrition_fraction = counts[ObservationStatus.EXITED] / expected
            max_missing = max(max_missing, missing_fraction)
            max_attrition = max(max_attrition, attrition_fraction)
            characteristic_means: list[tuple[str, float]] = []
            for name in names:
                baseline_values = tuple(
                    value
                    for unit_id in arm_ids
                    if (value := panel.unit(unit_id).characteristic(name)) is not None
                )
                observed_values = tuple(
                    value
                    for unit_id in observed_ids
                    if (value := panel.unit(unit_id).characteristic(name)) is not None
                )
                if observed_values:
                    observed_mean = sum(observed_values) / len(observed_values)
                    characteristic_means.append((name, observed_mean))
                    if baseline_values:
                        baseline_mean = sum(baseline_values) / len(baseline_values)
                        max_shift = max(max_shift, abs(observed_mean - baseline_mean))
            group_counts: Counter[str] = Counter()
            for cell in cells:
                if (
                    cell.status is ObservationStatus.OBSERVED
                    and cell.post_treatment_group is not None
                ):
                    group_counts[
                        f"{cell.post_treatment_group.name}="
                        f"{cell.post_treatment_group.value}"
                    ] += 1
            period_audits.append(
                PeriodCompositionAudit(
                    relative_period=period,
                    arm=arm,
                    expected=expected,
                    observed=observed,
                    missing=counts[ObservationStatus.MISSING],
                    exited=counts[ObservationStatus.EXITED],
                    not_in_sample=counts[ObservationStatus.NOT_IN_SAMPLE],
                    entrants=sum(
                        entrants.get((unit_id, period), 0) for unit_id in arm_ids
                    ),
                    exits=sum(exits.get((unit_id, period), 0) for unit_id in arm_ids),
                    reentries=sum(
                        reentries.get((unit_id, period), 0) for unit_id in arm_ids
                    ),
                    observed_unit_ids_sha256=sha256_bytes(
                        canonical_json_bytes(list(observed_ids))
                    ),
                    characteristic_means=tuple(characteristic_means),
                    post_treatment_group_counts=tuple(sorted(group_counts.items())),
                )
            )
    if max_missing > config.max_missing_fraction:
        flags.add("missing_fraction_exceeds_frozen_threshold")
    if max_attrition > config.max_attrition_fraction:
        flags.add("attrition_fraction_exceeds_frozen_threshold")
    if max_shift > config.max_composition_shift:
        flags.add("composition_shift_exceeds_frozen_threshold")
    if (
        config.missing_data_policy is MissingDataPolicy.COMPLETE_CASE
        and post_policy_observation_exclusion_unit_ids
    ):
        flags.add("post_policy_observation_exclusion_under_complete_case")
    if (
        config.missing_data_policy is MissingDataPolicy.COMPLETE_CASE
        and registered_covariate_exclusion_unit_ids
    ):
        flags.add("registered_covariate_exclusion_under_complete_case")
    return SampleAudit(
        period_audits=tuple(period_audits),
        complete_case_unit_ids=tuple(complete),
        excluded_units=tuple(excluded),
        post_policy_observation_exclusion_unit_ids=tuple(
            post_policy_observation_exclusion_unit_ids
        ),
        registered_covariate_exclusion_unit_ids=tuple(
            registered_covariate_exclusion_unit_ids
        ),
        max_missing_fraction_observed=max_missing,
        max_attrition_fraction_observed=max_attrition,
        max_composition_shift_observed=max_shift,
        flags=tuple(sorted(flags)),
    )


def _causal_reporting_decision(
    config: EmpiricalEvaluationConfig,
    *,
    estimates: tuple[NamedEstimate, ...],
    confidence_intervals: tuple[ConfidenceInterval, ...],
    bootstrap_inference: ClusterBootstrapAudit,
    randomization: RandomizationInferenceResult,
    pretrend: PreTrendDiagnostic,
    sample_audit: SampleAudit,
) -> CausalReportingDecision:
    """Apply only frozen support/pre-policy criteria, never target-effect results."""
    reasons: set[str] = set()
    if pretrend.status is DiagnosticStatus.INSUFFICIENT_DATA:
        reasons.add("pretrend_insufficient_data")
    elif pretrend.status is DiagnosticStatus.FAIL:
        if (
            pretrend.max_absolute_lead is not None
            and pretrend.max_absolute_lead > pretrend.max_absolute_lead_tolerance
        ):
            reasons.add("pre_policy_placebo_failed")
        if (
            pretrend.treated_control_slope_gap is not None
            and abs(pretrend.treated_control_slope_gap) > pretrend.max_slope_gap_per_day
        ):
            reasons.add("parallel_trends_failed")
        if not reasons:
            reasons.add("pretrend_failed")
    if any(item.status is not EstimateStatus.ESTIMATED for item in estimates):
        reasons.add("estimate_support_failed")
    if any(not item.valid_draw_gate_passed for item in confidence_intervals):
        reasons.add("bootstrap_valid_draw_gate_failed")
    reasons.update(bootstrap_inference.reasons)
    if randomization.invalid_draws:
        reasons.add("randomization_invalid_draws")
    if not randomization.valid_draw_gate_passed:
        reasons.add("randomization_valid_draw_gate_failed")
    for flag in sample_audit.flags:
        if flag == "missing_fraction_exceeds_frozen_threshold":
            reasons.add("sample_missingness_threshold_failed")
        elif flag == "attrition_fraction_exceeds_frozen_threshold":
            reasons.add("sample_attrition_threshold_failed")
        elif flag == "composition_shift_exceeds_frozen_threshold":
            reasons.add("sample_composition_threshold_failed")
        elif flag == "post_policy_observation_exclusion_under_complete_case":
            reasons.add("post_policy_observation_exclusion")
        elif flag == "registered_covariate_exclusion_under_complete_case":
            reasons.add("registered_covariate_exclusion")
        elif flag.startswith("unclassified_subgroup:"):
            reasons.add("subgroup_classification_failed")
        else:
            reasons.add("sample_audit_failed")
    return CausalReportingDecision(
        status=(
            CausalReportingStatus.INVALID if reasons else CausalReportingStatus.VALID
        ),
        criteria_sha256=config.causal_reporting_criteria_sha256,
        reasons=tuple(sorted(reasons)),
    )


class EmpiricalEvaluator:
    """Evaluate every frozen specification after all inputs have been sealed."""

    def evaluate(
        self,
        *,
        config: EmpiricalEvaluationConfig,
        panel: FrozenPanel,
        frozen_inputs: FrozenEvaluationInputs | FrozenEvaluationAuthorization,
    ) -> EmpiricalEvaluationArtifact:
        """Return the complete curve; diagnostics never select or drop a spec."""
        if isinstance(frozen_inputs, FrozenEvaluationAuthorization):
            receipt = frozen_inputs._consume(config=config, panel=panel)
        else:
            receipt = frozen_inputs
            if receipt.mode is FreezeMode.VERIFIED_TEMPORAL_SNAPSHOT:
                raise FrozenInputError(
                    "a serialized real-data receipt is evidence, not authority; "
                    "use the TemporalFirewall frozen evaluation bridge"
                )
        _validate_frozen_inputs(config, panel, receipt)
        return self._evaluate_validated(
            config=config,
            panel=panel,
            receipt=receipt,
        )

    def _evaluate_validated(
        self,
        *,
        config: EmpiricalEvaluationConfig,
        panel: FrozenPanel,
        receipt: FrozenEvaluationInputs,
    ) -> EmpiricalEvaluationArtifact:
        """Recompute estimators after the caller has crossed an authority gate."""
        subgroup_index = {item.subgroup_id: item for item in config.subgroups}
        results: list[SpecificationResult] = []
        for specification in config.specifications:
            subgroups = tuple(
                subgroup_index[subgroup_id]
                for subgroup_id in specification.subgroup_ids
            )
            sample_audit = _sample_audit(config, panel, specification, subgroups)
            unit_ids = sample_audit.complete_case_unit_ids
            estimates = point_estimates(
                config, panel, specification, unit_ids, subgroups
            )
            draws = cluster_bootstrap(config, panel, specification, unit_ids, subgroups)
            bootstrap_inference = cluster_bootstrap_audit(
                config,
                panel,
                specification,
                unit_ids,
                draws,
                subgroups,
            )
            intervals = confidence_intervals(
                estimates,
                draws,
                config.confidence_level,
                minimum_valid_draws=config.minimum_valid_bootstrap_draws,
                inference_gate_passed=(bootstrap_inference.valid_draw_gate_passed),
            )
            randomization = randomization_inference(
                config, panel, specification, unit_ids
            )
            pretrend = pretrend_diagnostic(config, panel, specification, unit_ids)
            causal_reporting = _causal_reporting_decision(
                config,
                estimates=estimates,
                confidence_intervals=intervals,
                bootstrap_inference=bootstrap_inference,
                randomization=randomization,
                pretrend=pretrend,
                sample_audit=sample_audit,
            )
            results.append(
                SpecificationResult(
                    specification_id=specification.specification_id,
                    estimates=estimates,
                    confidence_intervals=intervals,
                    bootstrap_draws=draws,
                    bootstrap_inference=bootstrap_inference,
                    randomization_inference=randomization,
                    pretrend_diagnostic=pretrend,
                    sample_audit=sample_audit,
                    causal_reporting=causal_reporting,
                )
            )
        observed_spec_ids = tuple(result.specification_id for result in results)
        frozen_spec_ids = tuple(
            specification.specification_id for specification in config.specifications
        )
        if observed_spec_ids != frozen_spec_ids:
            raise EmpiricalEvaluationError(
                "internal error: full frozen specification curve was not retained"
            )
        return EmpiricalEvaluationArtifact(
            schema_version=7,
            study_id=config.study_id,
            synthetic_only=panel.synthetic_only,
            config_sha256=config.sha256,
            panel_sha256=panel.sha256,
            causal_reporting_criteria_sha256=(config.causal_reporting_criteria_sha256),
            registered_specification_ids=frozen_spec_ids,
            frozen_inputs=receipt,
            specification_results=tuple(results),
        )


def _replay_verified_synthetic_artifact(
    *,
    config: EmpiricalEvaluationConfig,
    panel: FrozenPanel,
    frozen_inputs: FrozenEvaluationInputs,
) -> EmpiricalEvaluationArtifact:
    """Recompute one retained synthetic artifact without issuing authority.

    This private verifier exists only so a bundle can prove deterministic
    estimator reproducibility from its already-retained synthetic panel and
    receipt.  It does not reopen or revalidate the source TABLE snapshot, does
    not consume or mint a frozen-evaluation authorization, and categorically
    rejects real-data receipts.
    """
    if frozen_inputs.mode is not FreezeMode.VERIFIED_TEMPORAL_SNAPSHOT:
        raise FrozenInputError(
            "retained synthetic replay requires verified-snapshot evidence"
        )
    if not config.synthetic_test_only or not panel.synthetic_only:
        raise FrozenInputError(
            "retained estimator replay cannot authorize real empirical inputs"
        )
    if panel.schema_version != 2 or not panel.derivation_receipts:
        raise FrozenInputError(
            "retained synthetic replay requires a receipt-backed v2 panel"
        )
    if (
        frozen_inputs.derivation_evidence_class
        is not DerivationEvidenceClass.SYNTHETIC_TABLE_REPLAY_ONLY
        or not frozen_inputs.panel_derivation_replay_verified
        or frozen_inputs.source_authority_verified
        or frozen_inputs.checkpoint_derivation_replay_verified
        or frozen_inputs.prediction_derivation_replay_verified
        or frozen_inputs.cohort_derivation_replay_verified
    ):
        raise FrozenInputError(
            "retained synthetic replay derivation authority boundary is invalid"
        )
    implementation_sha256s = {
        receipt.plan.implementation_source_sha256
        for receipt in panel.derivation_receipts
    }
    if implementation_sha256s != {frozen_inputs.transformation_implementation_sha256}:
        raise FrozenInputError(
            "retained panel transformation implementation commitment changed"
        )
    _validate_frozen_inputs(config, panel, frozen_inputs)
    return EmpiricalEvaluator()._evaluate_validated(
        config=config,
        panel=panel,
        receipt=frozen_inputs,
    )


__all__ = ["EmpiricalEvaluator"]
