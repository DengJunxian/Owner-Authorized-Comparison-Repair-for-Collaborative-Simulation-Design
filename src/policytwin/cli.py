"""Command-line interface for validation, synthetic runs, and audits."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from policytwin.audit import audit_run
from policytwin.config import ConfigValidationError, load_experiment_config
from policytwin.experiments.matrix import (
    ExperimentMatrixError,
    audit_matrix_run_from_directory,
)
from policytwin.experiments.smoke import EMPIRICAL_STATUS
from policytwin.provenance import MetadataValidationError
from policytwin.run import RunCreationError, create_run_bundle


def _emit(value: object, *, error: bool = False) -> None:
    rendered = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    stream = sys.stderr if error else sys.stdout
    stream.write(f"{rendered}\n")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="policytwin",
        description="Run and audit reproducible PolicyTwin engineering experiments.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    validate = commands.add_parser(
        "validate-config", help="strictly validate a local experiment config"
    )
    validate.add_argument("config", type=Path)

    run = commands.add_parser("run", help="run a fully synthetic experiment")
    run.add_argument("config", type=Path)
    run.add_argument("--output", type=Path, required=True)

    audit = commands.add_parser("audit-run", help="audit a run bundle")
    audit.add_argument("run_directory", type=Path)
    required_gate = audit.add_mutually_exclusive_group()
    required_gate.add_argument(
        "--require-release-identity",
        action="store_true",
        help="fail unless the intact bundle has a clean Git and lock identity",
    )
    required_gate.add_argument(
        "--require-formal-reporting-eligible",
        action="store_true",
        help="fail unless the study-wide formal release gate was established",
    )

    audit_matrix = commands.add_parser(
        "audit-matrix", help="audit a sealed synthetic experiment matrix directory"
    )
    audit_matrix.add_argument("matrix_directory", type=Path)
    audit_matrix.add_argument("--expected-matrix-sha256", required=True)
    audit_matrix.add_argument("--expected-manifest-sha256", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Execute the PolicyTwin CLI and return a process exit code."""
    args = _build_parser().parse_args(argv)
    try:
        if args.command == "validate-config":
            loaded = load_experiment_config(args.config)
            _emit(
                {
                    "config_sha256": loaded.config_sha256,
                    "data_manifest_id": loaded.manifest.manifest_id,
                    "data_manifest_sha256": loaded.manifest_sha256,
                    "experiment_id": loaded.config.experiment_id,
                    "experiment_type": loaded.config.experiment_type,
                    "schema_version": 1,
                    "status": "valid",
                }
            )
            return 0

        if args.command == "run":
            loaded = load_experiment_config(args.config)
            metadata = create_run_bundle(loaded, args.output)
            release_identity_valid = (
                metadata.git.commit is not None
                and not metadata.git.dirty
                and metadata.environment.lock_file is not None
                and metadata.environment.lock_sha256 is not None
            )
            _emit(
                {
                    "claim_scope": metadata.claim_scope,
                    "development_only": not release_identity_valid,
                    "empirical_status": EMPIRICAL_STATUS,
                    "formal_reporting_eligible": False,
                    "metadata": str(
                        args.output.expanduser().resolve() / "metadata.json"
                    ),
                    "output": str(args.output.expanduser().resolve()),
                    "release_identity_valid": release_identity_valid,
                    "run_id": metadata.run_id,
                    "schema_version": 2,
                    "status": (
                        "completed_release_identity"
                        if release_identity_valid
                        else "completed_development_only"
                    ),
                }
            )
            return 0

        if args.command == "audit-run":
            report = audit_run(args.run_directory)
            requested_gate: str
            requested_gate_passed: bool
            if args.require_formal_reporting_eligible:
                requested_gate = "formal_reporting_eligible"
                requested_gate_passed = report.formal_reporting_eligible
            elif args.require_release_identity:
                requested_gate = "release_identity_valid"
                requested_gate_passed = report.release_identity_valid
            else:
                requested_gate = "bundle_integrity_valid"
                requested_gate_passed = report.bundle_integrity_valid
            payload = {
                **report.to_mapping(),
                "requested_gate": requested_gate,
                "requested_gate_passed": requested_gate_passed,
            }
            _emit(payload, error=not requested_gate_passed)
            return 0 if requested_gate_passed else 1

        if args.command == "audit-matrix":
            matrix_report = audit_matrix_run_from_directory(
                args.matrix_directory,
                expected_matrix_sha256=args.expected_matrix_sha256,
                expected_manifest_sha256=args.expected_manifest_sha256,
            )
            _emit(
                {
                    "completed_count": matrix_report.completed_count,
                    "execution_receipts_verified": (
                        matrix_report.execution_receipts_verified
                    ),
                    "failure_count": matrix_report.failure_count,
                    "manifest_sha256": matrix_report.manifest_sha256,
                    "matrix_id": matrix_report.matrix_id,
                    "matrix_sha256": matrix_report.matrix_sha256,
                    "row_count": matrix_report.row_count,
                    "schema_version": 1,
                    "status": "valid",
                    "valid": matrix_report.valid,
                }
            )
            return 0

        _emit(
            {"error": "unsupported command", "schema_version": 1, "status": "error"},
            error=True,
        )
        return 2
    except (
        ConfigValidationError,
        ExperimentMatrixError,
        MetadataValidationError,
        RunCreationError,
        OSError,
        ValueError,
    ) as error:
        _emit(
            {
                "error": str(error),
                "error_type": type(error).__name__,
                "schema_version": 1,
                "status": "error",
            },
            error=True,
        )
        return 2


if __name__ == "__main__":  # pragma: no cover - console script is the main entry.
    raise SystemExit(main())
