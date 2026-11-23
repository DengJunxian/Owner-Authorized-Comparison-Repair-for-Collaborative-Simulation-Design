"""End-to-end synthetic calibration, PPC, recovery, and SBC workflow."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from statistics import fmean

from policytwin.calibration.diagnostics import (
    ParameterIdentifiability,
    PosteriorPredictiveChecks,
    RecoveryMetric,
    identifiability_diagnostics,
    posterior_predictive_checks,
    quantile,
    synthetic_recovery,
)
from policytwin.calibration.estimation import (
    ABCPosteriorEstimator,
    PosteriorEstimator,
    SimulatorAdapter,
    calibration_event_key,
    sample_posterior,
    sample_prior,
)
from policytwin.calibration.schema import (
    CalibrationConfig,
    CalibrationError,
    CalibrationInput,
    PosteriorSamples,
    SummaryVector,
    parameter_mapping,
)
from policytwin.calibration.synthetic import synthetic_calibration_input
from policytwin.protocol import (
    ProtocolExecutionAuthorization,
    ProtocolExecutionConsumer,
    ProtocolExecutionPurpose,
    ProtocolExecutionReceipt,
    ProtocolValidationError,
    StudyProtocol,
    consume_protocol_execution_authorization,
)
from policytwin.provenance import canonical_json_bytes, sha256_bytes


@dataclass(frozen=True, slots=True)
class SBCRankHistogram:
    """Ranks and 90% interval coverage for one parameter across SBC datasets."""

    parameter: str
    posterior_draws_per_replicate: int
    ranks: tuple[int, ...]
    histogram: tuple[int, ...]
    interval_coverage_90: float

    def __post_init__(self) -> None:
        if not isinstance(self.parameter, str) or not self.parameter:
            raise CalibrationError("SBC parameter name must be non-empty")
        if (
            isinstance(self.posterior_draws_per_replicate, bool)
            or not isinstance(self.posterior_draws_per_replicate, int)
            or self.posterior_draws_per_replicate < 2
        ):
            raise CalibrationError("SBC requires at least two posterior draws")
        if not self.ranks:
            raise CalibrationError("SBC must retain at least one rank")
        if len(self.histogram) != self.posterior_draws_per_replicate + 1:
            raise CalibrationError("SBC histogram has the wrong number of rank bins")
        if any(
            isinstance(count, bool) or not isinstance(count, int) or count < 0
            for count in self.histogram
        ):
            raise CalibrationError("SBC histogram counts must be non-negative integers")
        if sum(self.histogram) != len(self.ranks):
            raise CalibrationError("SBC histogram does not match retained ranks")
        if any(
            rank < 0 or rank > self.posterior_draws_per_replicate for rank in self.ranks
        ):
            raise CalibrationError("SBC rank lies outside its support")
        if not 0.0 <= self.interval_coverage_90 <= 1.0:
            raise CalibrationError("SBC interval coverage must lie in [0, 1]")

    def to_mapping(self) -> dict[str, object]:
        return {
            "histogram": list(self.histogram),
            "interval_coverage_90": self.interval_coverage_90,
            "parameter": self.parameter,
            "posterior_draws_per_replicate": self.posterior_draws_per_replicate,
            "ranks": list(self.ranks),
        }


@dataclass(frozen=True, slots=True)
class SimulationBasedCalibration:
    """Complete synthetic SBC outputs and their explicit random keys."""

    replicates: int
    parameters: tuple[SBCRankHistogram, ...]
    event_keys: tuple[str, ...]

    def __post_init__(self) -> None:
        if (
            isinstance(self.replicates, bool)
            or not isinstance(self.replicates, int)
            or self.replicates <= 0
        ):
            raise CalibrationError("SBC replicate count must be positive")
        if (
            not isinstance(self.parameters, tuple)
            or not self.parameters
            or not all(isinstance(item, SBCRankHistogram) for item in self.parameters)
        ):
            raise CalibrationError("SBC must retain typed parameter diagnostics")
        names = tuple(item.parameter for item in self.parameters)
        if len(set(names)) != len(names):
            raise CalibrationError("SBC parameter diagnostics must be unique")
        if any(len(item.ranks) != self.replicates for item in self.parameters):
            raise CalibrationError("SBC ranks do not match the replicate count")
        if (
            not isinstance(self.event_keys, tuple)
            or not self.event_keys
            or not all(isinstance(item, str) and item for item in self.event_keys)
            or len(set(self.event_keys)) != len(self.event_keys)
        ):
            raise CalibrationError("SBC event keys must be non-empty and unique")

    def to_mapping(self) -> dict[str, object]:
        return {
            "event_keys": list(self.event_keys),
            "parameters": [item.to_mapping() for item in self.parameters],
            "replicates": self.replicates,
            "synthetic_only": True,
        }

    @property
    def sha256(self) -> str:
        return sha256_bytes(canonical_json_bytes(self.to_mapping()))


def run_simulation_based_calibration(
    *,
    config: CalibrationConfig,
    simulator: SimulatorAdapter,
    estimator: PosteriorEstimator,
) -> SimulationBasedCalibration:
    """Run small prior-predictive SBC with no policy-post observations."""
    ranks: dict[str, list[int]] = {prior.name: [] for prior in config.priors}
    coverage: dict[str, list[bool]] = {prior.name: [] for prior in config.priors}
    all_keys: list[str] = []
    for replicate in range(config.sbc_replicates):
        namespace = f"sbc_{replicate}"
        truth, truth_keys = sample_prior(
            config,
            replicate=replicate,
            namespace=f"{namespace}:truth",
        )
        calibration_observation_key = calibration_event_key(
            config,
            replicate=replicate,
            step=0,
            purpose=f"{namespace}:observed_simulator:calibration",
        )
        held_out_observation_key = calibration_event_key(
            config,
            replicate=replicate,
            step=1,
            purpose=f"{namespace}:observed_simulator:held_out",
        )
        calibration_summary = simulator.simulate(
            truth, event_key=calibration_observation_key
        )
        held_out_summary = simulator.simulate(truth, event_key=held_out_observation_key)
        observed = synthetic_calibration_input(
            config,
            calibration_summary,
            held_out_summary,
            record_id=f"synthetic-sbc-{replicate}",
            calibration_generation_event_key=calibration_observation_key.value,
            held_out_generation_event_key=held_out_observation_key.value,
        )
        fit = estimator.fit(
            config=config,
            observed=observed,
            simulator=simulator,
            namespace=f"{namespace}:fit",
        )
        posterior, posterior_keys = sample_posterior(
            config=config,
            fit=fit,
            draw_count=config.sbc_posterior_draws,
            namespace=f"{namespace}:posterior",
        )
        for prior in config.priors:
            values = [dict(draw.parameters)[prior.name] for draw in posterior.draws]
            true_value = truth[prior.name]
            ranks[prior.name].append(sum(value < true_value for value in values))
            lower = quantile(values, 0.05)
            upper = quantile(values, 0.95)
            coverage[prior.name].append(lower <= true_value <= upper)
        all_keys.extend(truth_keys)
        all_keys.extend(
            (calibration_observation_key.value, held_out_observation_key.value)
        )
        all_keys.extend(fit.event_keys)
        all_keys.extend(posterior_keys)
    results: list[SBCRankHistogram] = []
    for prior in config.priors:
        parameter_ranks = tuple(ranks[prior.name])
        histogram = tuple(
            parameter_ranks.count(rank)
            for rank in range(config.sbc_posterior_draws + 1)
        )
        results.append(
            SBCRankHistogram(
                parameter=prior.name,
                posterior_draws_per_replicate=config.sbc_posterior_draws,
                ranks=parameter_ranks,
                histogram=histogram,
                interval_coverage_90=fmean(coverage[prior.name]),
            )
        )
    if len(set(all_keys)) != len(all_keys):
        raise CalibrationError("SBC reused a random event key")
    return SimulationBasedCalibration(
        replicates=config.sbc_replicates,
        parameters=tuple(results),
        event_keys=tuple(all_keys),
    )


@dataclass(frozen=True, slots=True)
class CalibrationReport:
    """Machine-readable scientific status without real-world success claims."""

    schema_version: int
    study_protocol_sha256: str
    study_claim_scope: str
    protocol_execution_receipt_sha256: str
    config_sha256: str
    input_sha256: str
    simulator_id: str
    simulator_config_sha256: str
    true_parameters_sha256: str
    estimator_backend: str
    completion_status: str
    claim_scope: str
    real_data_status: str
    calibration_statistics: tuple[str, ...]
    held_out_statistics: tuple[str, ...]
    posterior_particle_count: int
    posterior_draw_count: int
    posterior_sha256: str
    posterior_predictive_checks_sha256: str
    rank_histogram_sha256: str
    event_key_count: int
    event_keys_sha256: str
    recovery: tuple[RecoveryMetric, ...]
    ppc_held_out_coverage: float
    sbc_interval_coverage_90: tuple[tuple[str, float], ...]
    identifiability: tuple[ParameterIdentifiability, ...]
    warnings: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.schema_version != 3:
            raise CalibrationError("unsupported calibration report schema version")
        for field in (
            "study_protocol_sha256",
            "protocol_execution_receipt_sha256",
            "config_sha256",
            "input_sha256",
            "simulator_config_sha256",
            "true_parameters_sha256",
            "posterior_sha256",
            "posterior_predictive_checks_sha256",
            "rank_histogram_sha256",
            "event_keys_sha256",
        ):
            value = getattr(self, field)
            if (
                not isinstance(value, str)
                or len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)
            ):
                raise CalibrationError(f"{field} must be a lowercase SHA-256")
        if not self.simulator_id or not self.estimator_backend:
            raise CalibrationError("simulator and estimator identities must be present")
        if self.study_claim_scope != "synthetic_mechanics_only":
            raise CalibrationError("study protocol claim scope changed")
        if self.completion_status != "synthetic_method_check_completed":
            raise CalibrationError("calibration report completion scope changed")
        if self.claim_scope != "synthetic_calibration_mechanics_only":
            raise CalibrationError("calibration report claim scope changed")
        if self.real_data_status != "blocked_not_run":
            raise CalibrationError(
                "real-data calibration must remain explicitly blocked"
            )
        if (
            self.posterior_particle_count < 2
            or self.posterior_draw_count < 2
            or self.event_key_count <= 0
        ):
            raise CalibrationError("report does not retain a posterior and random keys")
        if set(self.calibration_statistics) & set(self.held_out_statistics):
            raise CalibrationError(
                "report reused calibration statistics for validation"
            )
        if not self.calibration_statistics or not self.held_out_statistics:
            raise CalibrationError("report must retain both statistic roles")
        if not 0.0 <= self.ppc_held_out_coverage <= 1.0:
            raise CalibrationError("PPC coverage must lie in [0, 1]")

    def to_mapping(self) -> dict[str, object]:
        return {
            "calibration_statistics": list(self.calibration_statistics),
            "claim_scope": self.claim_scope,
            "completion_status": self.completion_status,
            "config_sha256": self.config_sha256,
            "estimator_backend": self.estimator_backend,
            "event_key_count": self.event_key_count,
            "event_keys_sha256": self.event_keys_sha256,
            "held_out_statistics": list(self.held_out_statistics),
            "identifiability": [item.to_mapping() for item in self.identifiability],
            "input_sha256": self.input_sha256,
            "posterior_draw_count": self.posterior_draw_count,
            "posterior_particle_count": self.posterior_particle_count,
            "posterior_predictive_checks_sha256": (
                self.posterior_predictive_checks_sha256
            ),
            "posterior_sha256": self.posterior_sha256,
            "ppc_held_out_coverage": self.ppc_held_out_coverage,
            "real_data_status": self.real_data_status,
            "recovery": [item.to_mapping() for item in self.recovery],
            "rank_histogram_sha256": self.rank_histogram_sha256,
            "sbc_interval_coverage_90": dict(self.sbc_interval_coverage_90),
            "schema_version": self.schema_version,
            "simulator_id": self.simulator_id,
            "simulator_config_sha256": self.simulator_config_sha256,
            "true_parameters_sha256": self.true_parameters_sha256,
            "study_claim_scope": self.study_claim_scope,
            "study_protocol_sha256": self.study_protocol_sha256,
            "synthetic_only": True,
            "protocol_execution_receipt_sha256": (
                self.protocol_execution_receipt_sha256
            ),
            "warnings": list(self.warnings),
        }

    @property
    def sha256(self) -> str:
        return sha256_bytes(canonical_json_bytes(self.to_mapping()))


@dataclass(frozen=True, slots=True)
class CalibrationRun:
    """Complete in-memory run ready for a versioned artifact writer."""

    config: CalibrationConfig
    observed: CalibrationInput
    protocol_execution_receipt: ProtocolExecutionReceipt
    simulator_config_json: bytes
    true_parameters: tuple[tuple[str, float], ...]
    posterior: PosteriorSamples
    ppc: PosteriorPredictiveChecks
    recovery: tuple[RecoveryMetric, ...]
    sbc: SimulationBasedCalibration
    identifiability: tuple[ParameterIdentifiability, ...]
    report: CalibrationReport
    event_keys: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.true_parameters, tuple):
            raise CalibrationError("synthetic truth must be an immutable tuple")
        try:
            truth_mapping = dict(self.true_parameters)
        except (TypeError, ValueError) as error:
            raise CalibrationError("synthetic truth has an invalid shape") from error
        if len(truth_mapping) != len(self.true_parameters):
            raise CalibrationError("synthetic truth parameter names must be unique")
        normalized_truth = parameter_mapping(truth_mapping)
        object.__setattr__(self, "true_parameters", normalized_truth)
        if len(set(self.event_keys)) != len(self.event_keys):
            raise CalibrationError("calibration run reused a random event key")
        if self.report.config_sha256 != self.config.sha256:
            raise CalibrationError("report/config commitment mismatch")
        if (
            self.report.study_protocol_sha256
            != self.protocol_execution_receipt.protocol_sha256
            or self.report.protocol_execution_receipt_sha256
            != self.protocol_execution_receipt.sha256
            or self.report.study_claim_scope
            != self.protocol_execution_receipt.claim_scope.value
        ):
            raise CalibrationError("report/protocol authorization commitment mismatch")
        if self.report.input_sha256 != self.observed.sha256:
            raise CalibrationError("report/input commitment mismatch")
        if self.report.posterior_sha256 != self.posterior.sha256:
            raise CalibrationError("report/posterior commitment mismatch")
        if self.report.posterior_predictive_checks_sha256 != self.ppc.sha256:
            raise CalibrationError("report/PPC commitment mismatch")
        if self.report.rank_histogram_sha256 != self.sbc.sha256:
            raise CalibrationError("report/SBC commitment mismatch")
        if (
            len(self.posterior.particles) != self.config.accepted_particles
            or len(self.posterior.draws) != self.config.posterior_draws
            or self.report.posterior_particle_count != len(self.posterior.particles)
            or self.report.posterior_draw_count != len(self.posterior.draws)
        ):
            raise CalibrationError("posterior counts changed from the frozen config")
        expected_statistics = {
            item.name: item.role for item in self.config.summaries.statistics
        }
        observed_statistics = {item.name: item.role for item in self.ppc.statistics}
        if observed_statistics != expected_statistics or any(
            len(item.predictive_draws) != self.config.ppc_draws
            for item in self.ppc.statistics
        ):
            raise CalibrationError("PPC output changed the frozen statistic contract")
        if self.report.recovery != self.recovery:
            raise CalibrationError("report/recovery output mismatch")
        if self.report.identifiability != self.identifiability:
            raise CalibrationError("report/identifiability output mismatch")
        expected_calibration = tuple(
            item.name for item in self.config.summaries.calibration
        )
        expected_held_out = tuple(item.name for item in self.config.summaries.held_out)
        held_out_ppc = tuple(
            item
            for item in self.ppc.statistics
            if item.role.value == "held_out_validation"
        )
        if (
            self.report.calibration_statistics != expected_calibration
            or self.report.held_out_statistics != expected_held_out
            or self.report.ppc_held_out_coverage
            != fmean(item.covered for item in held_out_ppc)
        ):
            raise CalibrationError("report/statistic-role output mismatch")
        expected_sbc_coverage = tuple(
            (item.parameter, item.interval_coverage_90) for item in self.sbc.parameters
        )
        if (
            self.sbc.replicates != self.config.sbc_replicates
            or self.report.sbc_interval_coverage_90 != expected_sbc_coverage
        ):
            raise CalibrationError("report/SBC diagnostic mismatch")
        if not isinstance(self.simulator_config_json, bytes):
            raise CalibrationError("simulator configuration must be canonical bytes")
        if (
            sha256_bytes(self.simulator_config_json)
            != self.report.simulator_config_sha256
        ):
            raise CalibrationError("report/simulator-config commitment mismatch")
        if self.report.true_parameters_sha256 != self.true_parameters_sha256:
            raise CalibrationError("report/synthetic-truth commitment mismatch")
        expected_key_sha256 = sha256_bytes(canonical_json_bytes(list(self.event_keys)))
        if (
            self.report.event_key_count != len(self.event_keys)
            or self.report.event_keys_sha256 != expected_key_sha256
        ):
            raise CalibrationError("report/event-key commitment mismatch")
        self._verify_builtin_synthetic_replay()

    @property
    def true_parameters_mapping(self) -> dict[str, object]:
        """Return the complete synthetic generating truth as a versioned object."""
        return {
            "parameters": dict(self.true_parameters),
            "schema_version": 1,
            "synthetic_only": True,
        }

    @property
    def true_parameters_sha256(self) -> str:
        """Return the semantic commitment to the retained generating truth."""
        return sha256_bytes(canonical_json_bytes(self.true_parameters_mapping))

    def _verify_builtin_synthetic_replay(self) -> None:
        """Recompute the sealed built-in closure from frozen inputs and keys."""
        simulator = _trusted_simulator_from_json(self.simulator_config_json)
        truth = dict(self.true_parameters)
        replayed_observed = _replay_synthetic_observed(
            config=self.config,
            simulator=simulator,
            true_parameters=truth,
        )
        if replayed_observed != self.observed:
            raise CalibrationError(
                "observed summaries do not replay from the retained synthetic truth"
            )
        if self.posterior.backend_id != ABCPosteriorEstimator().backend_id:
            raise CalibrationError(
                "sealed calibration replay supports only the built-in ABC backend; "
                "external estimator outputs remain unverified"
            )
        fit = ABCPosteriorEstimator().fit(
            config=self.config,
            observed=self.observed,
            simulator=simulator,
        )
        posterior, posterior_keys = sample_posterior(config=self.config, fit=fit)
        if posterior != self.posterior:
            raise CalibrationError("posterior does not replay from frozen inputs")
        ppc = posterior_predictive_checks(
            config=self.config,
            observed=self.observed,
            posterior=posterior,
            simulator=simulator,
        )
        if ppc != self.ppc:
            raise CalibrationError("PPC does not replay from frozen inputs")
        recovery = synthetic_recovery(self.config, posterior, truth)
        if recovery != self.recovery:
            raise CalibrationError(
                "recovery does not rederive from posterior and truth"
            )
        identifiability = identifiability_diagnostics(self.config, posterior)
        if identifiability != self.identifiability:
            raise CalibrationError(
                "identifiability diagnostics do not rederive from posterior"
            )
        sbc = run_simulation_based_calibration(
            config=self.config,
            simulator=simulator,
            estimator=ABCPosteriorEstimator(),
        )
        if sbc != self.sbc:
            raise CalibrationError("SBC does not replay from frozen inputs")
        event_keys = (
            self.observed.event_keys
            + fit.event_keys
            + posterior_keys
            + ppc.event_keys
            + sbc.event_keys
        )
        if event_keys != self.event_keys:
            raise CalibrationError("complete event-key manifest does not replay")
        expected_report = _build_calibration_report(
            config=self.config,
            observed=self.observed,
            protocol_receipt=self.protocol_execution_receipt,
            simulator=simulator,
            simulator_config_sha256=self.report.simulator_config_sha256,
            true_parameters_sha256=self.true_parameters_sha256,
            posterior=posterior,
            ppc=ppc,
            recovery=recovery,
            sbc=sbc,
            identifiability=identifiability,
            event_keys=event_keys,
        )
        if expected_report != self.report:
            raise CalibrationError("calibration report does not rederive from outputs")


def _trusted_simulator_from_json(simulator_config_json: bytes) -> SimulatorAdapter:
    """Reconstruct only locally implemented, deterministic synthetic simulators."""
    from policytwin.calibration.integrated import (
        INTEGRATED_CALIBRATION_SIMULATOR_ID,
        IntegratedSyntheticCalibrationAdapter,
    )
    from policytwin.calibration.synthetic import SyntheticMarketSimulator
    from policytwin.provenance import MetadataValidationError, read_json_bytes_object

    try:
        mapping = read_json_bytes_object(
            simulator_config_json,
            source="sealed calibration simulator configuration",
        )
    except MetadataValidationError as error:
        raise CalibrationError(str(error)) from error
    simulator_id = mapping.get("simulator_id")
    if simulator_id == "synthetic_summary_simulator_v1":
        if (
            frozenset(mapping)
            != {
                "noise_scale",
                "schema_version",
                "simulator_id",
                "synthetic_only",
            }
            or isinstance(mapping.get("schema_version"), bool)
            or mapping.get("schema_version") != 1
            or mapping.get("synthetic_only") is not True
        ):
            raise CalibrationError("synthetic simulator configuration schema changed")
        noise_scale = mapping.get("noise_scale")
        if isinstance(noise_scale, bool) or not isinstance(noise_scale, int | float):
            raise CalibrationError("synthetic simulator noise_scale must be numeric")
        return SyntheticMarketSimulator(noise_scale=float(noise_scale))
    if simulator_id == INTEGRATED_CALIBRATION_SIMULATOR_ID:
        if (
            frozenset(mapping)
            != {
                "integrated_config_sha256",
                "schema_version",
                "simulator_id",
                "step_count",
                "synthetic_only",
            }
            or isinstance(mapping.get("schema_version"), bool)
            or mapping.get("schema_version") != 1
            or mapping.get("synthetic_only") is not True
        ):
            raise CalibrationError("integrated simulator configuration schema changed")
        step_count = mapping.get("step_count")
        if isinstance(step_count, bool) or not isinstance(step_count, int):
            raise CalibrationError("integrated simulator step_count must be an integer")
        simulator = IntegratedSyntheticCalibrationAdapter(step_count=step_count)
        if simulator.to_mapping() != mapping:
            raise CalibrationError("integrated simulator dependency commitment changed")
        return simulator
    raise CalibrationError(
        "sealed calibration artifacts support only trusted built-in synthetic "
        "simulators; external simulator behavior remains unverified"
    )


def _replay_synthetic_observed(
    *,
    config: CalibrationConfig,
    simulator: SimulatorAdapter,
    true_parameters: Mapping[str, float],
) -> CalibrationInput:
    """Regenerate the two frozen role trajectories from their standard keys."""
    calibration_key = calibration_event_key(
        config,
        replicate=0,
        step=0,
        purpose="synthetic_observed:calibration",
    )
    held_out_key = calibration_event_key(
        config,
        replicate=0,
        step=1,
        purpose="synthetic_observed:held_out",
    )
    calibration_summary: SummaryVector = simulator.simulate(
        true_parameters, event_key=calibration_key
    )
    held_out_summary: SummaryVector = simulator.simulate(
        true_parameters, event_key=held_out_key
    )
    return synthetic_calibration_input(
        config,
        calibration_summary,
        held_out_summary,
        calibration_generation_event_key=calibration_key.value,
        held_out_generation_event_key=held_out_key.value,
    )


def _calibration_warnings(
    identifiability: tuple[ParameterIdentifiability, ...],
    ppc: PosteriorPredictiveChecks,
) -> tuple[str, ...]:
    warnings = [
        "SYNTHETIC ONLY: recovery, coverage, and diagnostics are not "
        "real-market evidence.",
        "Real-data calibration is blocked pending a temporal-firewall-issued "
        "input capability.",
    ]
    for diagnostic in identifiability:
        if not diagnostic.estimable_for_reporting:
            warnings.append(
                f"Parameter {diagnostic.parameter} is not identifiable enough for a "
                "precise paper estimate."
            )
        if diagnostic.boundary_pileup:
            warnings.append(
                f"Parameter {diagnostic.parameter} has posterior boundary pile-up."
            )
        if diagnostic.multimodal:
            warnings.append(
                f"Parameter {diagnostic.parameter} has a multimodal posterior."
            )
    for item in ppc.statistics:
        if item.role.value == "held_out_validation" and not item.covered:
            warnings.append(f"Held-out PPC failed for statistic {item.name}.")
    return tuple(warnings)


def _build_calibration_report(
    *,
    config: CalibrationConfig,
    observed: CalibrationInput,
    protocol_receipt: ProtocolExecutionReceipt,
    simulator: SimulatorAdapter,
    simulator_config_sha256: str,
    true_parameters_sha256: str,
    posterior: PosteriorSamples,
    ppc: PosteriorPredictiveChecks,
    recovery: tuple[RecoveryMetric, ...],
    sbc: SimulationBasedCalibration,
    identifiability: tuple[ParameterIdentifiability, ...],
    event_keys: tuple[str, ...],
) -> CalibrationReport:
    held_out = tuple(
        item for item in ppc.statistics if item.role.value == "held_out_validation"
    )
    return CalibrationReport(
        schema_version=3,
        study_protocol_sha256=protocol_receipt.protocol_sha256,
        study_claim_scope=protocol_receipt.claim_scope.value,
        protocol_execution_receipt_sha256=protocol_receipt.sha256,
        config_sha256=config.sha256,
        input_sha256=observed.sha256,
        simulator_id=simulator.simulator_id,
        simulator_config_sha256=simulator_config_sha256,
        true_parameters_sha256=true_parameters_sha256,
        estimator_backend=posterior.backend_id,
        completion_status="synthetic_method_check_completed",
        claim_scope="synthetic_calibration_mechanics_only",
        real_data_status="blocked_not_run",
        calibration_statistics=tuple(
            item.name for item in config.summaries.calibration
        ),
        held_out_statistics=tuple(item.name for item in config.summaries.held_out),
        posterior_particle_count=len(posterior.particles),
        posterior_draw_count=len(posterior.draws),
        posterior_sha256=posterior.sha256,
        posterior_predictive_checks_sha256=ppc.sha256,
        rank_histogram_sha256=sbc.sha256,
        event_key_count=len(event_keys),
        event_keys_sha256=sha256_bytes(canonical_json_bytes(list(event_keys))),
        recovery=recovery,
        ppc_held_out_coverage=fmean(item.covered for item in held_out),
        sbc_interval_coverage_90=tuple(
            (item.parameter, item.interval_coverage_90) for item in sbc.parameters
        ),
        identifiability=identifiability,
        warnings=_calibration_warnings(identifiability, ppc),
    )


def run_synthetic_calibration(
    *,
    config: CalibrationConfig,
    observed: CalibrationInput,
    simulator: SimulatorAdapter,
    true_parameters: Mapping[str, float],
    study_protocol: StudyProtocol,
    protocol_authorization: ProtocolExecutionAuthorization,
    execution_started_at: datetime,
    estimator: PosteriorEstimator | None = None,
) -> CalibrationRun:
    """Execute a protocol-authorized synthetic calibration with explicit keys."""
    if config.synthetic_only is not True or observed.synthetic_only is not True:
        raise CalibrationError("the implemented workflow is synthetic-only")
    normalized_truth = parameter_mapping(true_parameters)
    trusted_simulator = _trusted_simulator_from_json(
        canonical_json_bytes(simulator.to_mapping())
    )
    if trusted_simulator.to_mapping() != simulator.to_mapping():
        raise CalibrationError("simulator cannot be reconstructed exactly")
    try:
        protocol_receipt = consume_protocol_execution_authorization(
            protocol_authorization,
            protocol=study_protocol,
            consumer=ProtocolExecutionConsumer.CALIBRATION_WORKFLOW,
            purpose=ProtocolExecutionPurpose.PRE_POLICY_CALIBRATION,
            config_sha256=config.sha256,
            input_sha256=observed.sha256,
            execution_started_at=execution_started_at,
        )
    except ProtocolValidationError as error:
        raise CalibrationError(
            f"study protocol authorization rejected: {error}"
        ) from error
    calibration_statistics = tuple(item.name for item in config.summaries.calibration)
    held_out_statistics = tuple(item.name for item in config.summaries.held_out)
    if study_protocol.study_id != config.study_id:
        raise CalibrationError(
            "calibration config belongs to a different study protocol"
        )
    if study_protocol.knowledge_cutoff != config.cutoff:
        raise CalibrationError("calibration cutoff differs from the study protocol")
    if study_protocol.calibration_statistics != calibration_statistics:
        raise CalibrationError("calibration statistics differ from the study protocol")
    if study_protocol.heldout_validation_statistics != held_out_statistics:
        raise CalibrationError("held-out statistics differ from the study protocol")
    observed.validate_for(config)
    simulator_config_json = canonical_json_bytes(simulator.to_mapping())
    simulator_config_sha256 = sha256_bytes(simulator_config_json)
    backend = ABCPosteriorEstimator() if estimator is None else estimator
    if type(backend) is not ABCPosteriorEstimator:
        raise CalibrationError(
            "sealed synthetic calibration currently verifies only the built-in "
            "ABC backend; external estimators remain interface-only"
        )
    fit = backend.fit(config=config, observed=observed, simulator=simulator)
    if fit.backend_id != backend.backend_id:
        raise CalibrationError("estimator fit backend identity changed")
    posterior, posterior_keys = sample_posterior(config=config, fit=fit)
    ppc = posterior_predictive_checks(
        config=config,
        observed=observed,
        posterior=posterior,
        simulator=simulator,
    )
    recovery = synthetic_recovery(config, posterior, dict(normalized_truth))
    diagnostics = identifiability_diagnostics(config, posterior)
    sbc = run_simulation_based_calibration(
        config=config,
        simulator=simulator,
        estimator=backend,
    )
    event_keys = (
        observed.event_keys
        + fit.event_keys
        + posterior_keys
        + ppc.event_keys
        + sbc.event_keys
    )
    if len(set(event_keys)) != len(event_keys):
        raise CalibrationError("calibration workflow reused a random event key")
    true_parameters_sha256 = sha256_bytes(
        canonical_json_bytes(
            {
                "parameters": dict(normalized_truth),
                "schema_version": 1,
                "synthetic_only": True,
            }
        )
    )
    report = _build_calibration_report(
        config=config,
        observed=observed,
        protocol_receipt=protocol_receipt,
        simulator=simulator,
        simulator_config_sha256=simulator_config_sha256,
        true_parameters_sha256=true_parameters_sha256,
        posterior=posterior,
        ppc=ppc,
        recovery=recovery,
        sbc=sbc,
        identifiability=diagnostics,
        event_keys=event_keys,
    )
    return CalibrationRun(
        config=config,
        observed=observed,
        protocol_execution_receipt=protocol_receipt,
        simulator_config_json=simulator_config_json,
        true_parameters=normalized_truth,
        posterior=posterior,
        ppc=ppc,
        recovery=recovery,
        sbc=sbc,
        identifiability=diagnostics,
        report=report,
        event_keys=event_keys,
    )


__all__ = [
    "CalibrationReport",
    "CalibrationRun",
    "SBCRankHistogram",
    "SimulationBasedCalibration",
    "run_simulation_based_calibration",
    "run_synthetic_calibration",
]
