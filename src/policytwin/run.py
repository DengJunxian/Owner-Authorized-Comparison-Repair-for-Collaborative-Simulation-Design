"""Fail-closed creation of versioned synthetic run bundles."""

from __future__ import annotations

import os
import re
import shutil
import stat
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Self

from policytwin.config import LoadedExperimentConfig, load_experiment_config
from policytwin.counterfactual.randomness import EVENT_KEY_SCHEMA_VERSION
from policytwin.experiments.smoke import CLAIM_SCOPE, run_smoke_experiment
from policytwin.provenance import (
    JsonlEventWriter,
    RunMetadata,
    SourceArtifact,
    build_event_key_metadata,
    canonical_json_bytes,
    collect_environment_metadata,
    collect_git_metadata,
    digest_artifact,
    sha256_bytes,
    sha256_file,
    utc_now_text,
    write_json_file,
    write_text_file,
)

RUN_BUNDLE_FILES = (
    "config.json",
    "data_manifest.json",
    "events.jsonl",
    "result.json",
    "metadata.json",
    "metadata.sha256",
    "completion.json",
)

_RUN_ID_PATTERN = re.compile(r"^run-[0-9a-f]{24}$")
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


class RunCreationError(RuntimeError):
    """Raised when a trustworthy run bundle cannot be produced."""


@dataclass(frozen=True, slots=True)
class RunCompletion:
    """Positive commit marker written as the final fallible publication step."""

    schema_version: int
    status: str
    run_id: str
    metadata_sha256: str

    def __post_init__(self) -> None:
        if self.schema_version != 1 or self.status != "completed":
            raise RunCreationError("run completion marker has invalid fixed fields")
        if _RUN_ID_PATTERN.fullmatch(self.run_id) is None:
            raise RunCreationError("run completion marker has an invalid run ID")
        if _SHA256_PATTERN.fullmatch(self.metadata_sha256) is None:
            raise RunCreationError("run completion marker has an invalid metadata hash")

    def to_mapping(self) -> dict[str, object]:
        """Return the strict machine-readable completion marker."""
        return {
            "metadata_sha256": self.metadata_sha256,
            "run_id": self.run_id,
            "schema_version": self.schema_version,
            "status": self.status,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> Self:
        """Strictly parse one completion marker."""
        if set(value) != {"metadata_sha256", "run_id", "schema_version", "status"}:
            raise RunCreationError("run completion marker fields differ from schema")
        schema_version = value["schema_version"]
        status = value["status"]
        run_id = value["run_id"]
        metadata_sha256 = value["metadata_sha256"]
        if (
            isinstance(schema_version, bool)
            or not isinstance(schema_version, int)
            or not isinstance(status, str)
            or not isinstance(run_id, str)
            or not isinstance(metadata_sha256, str)
        ):
            raise RunCreationError("run completion marker field types are invalid")
        return cls(
            schema_version=schema_version,
            status=status,
            run_id=run_id,
            metadata_sha256=metadata_sha256,
        )


def _fsync_directory(path: Path) -> None:
    directory_fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _checked_remove_failed_run_bundle(
    output: Path, *, primary_error: BaseException
) -> None:
    """Remove a failed bundle or raise an explicit cleanup failure."""
    try:
        info = os.lstat(output)
    except FileNotFoundError:
        return
    except OSError as error:
        raise RunCreationError(
            f"cannot inspect failed run bundle: {error}"
        ) from primary_error
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
        raise RunCreationError(
            "failed run bundle cleanup requires the real producer directory"
        ) from primary_error
    try:
        shutil.rmtree(output)
    except OSError as error:
        raise RunCreationError(
            f"failed run bundle cleanup failed: {error}"
        ) from primary_error
    if output.exists() or output.is_symlink():
        raise RunCreationError(
            "failed run bundle remains after cleanup"
        ) from primary_error


def find_project_root(start: Path) -> Path:
    """Find the nearest project root containing pyproject.toml and uv.lock."""
    resolved = start.resolve()
    first = resolved if resolved.is_dir() else resolved.parent
    for candidate in (first, *first.parents):
        if (candidate / "pyproject.toml").is_file() and (
            candidate / "uv.lock"
        ).is_file():
            return candidate
    raise RunCreationError(f"cannot locate project root above {start}")


def derive_run_identity(
    *,
    config_sha256: str,
    manifest_sha256: str,
    experiment_id: str,
    seed: str,
) -> tuple[str, str]:
    """Derive the stable run ID and event-key root from frozen inputs."""
    root_material = {
        "event_key_schema": EVENT_KEY_SCHEMA_VERSION,
        "experiment_id": experiment_id,
        "seed": seed,
    }
    root_key = f"ptr1:{sha256_bytes(canonical_json_bytes(root_material))}"
    run_material = {
        "config_sha256": config_sha256,
        "data_manifest_sha256": manifest_sha256,
        "event_key_root": root_key,
        "schema_version": 1,
    }
    run_id = f"run-{sha256_bytes(canonical_json_bytes(run_material))[:24]}"
    return run_id, root_key


def create_run_bundle(
    loaded: LoadedExperimentConfig, output_directory: Path
) -> RunMetadata:
    """Execute the synthetic smoke run and exclusively publish its bundle."""
    refreshed = load_experiment_config(loaded.config_path)
    if refreshed != loaded:
        raise RunCreationError("config or data manifest changed after validation")
    loaded = refreshed
    output = output_directory.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        project_root = find_project_root(loaded.config_path)
    except RunCreationError:
        project_root = find_project_root(Path(__file__))
    run_id, root_key = derive_run_identity(
        config_sha256=loaded.config_sha256,
        manifest_sha256=loaded.manifest_sha256,
        experiment_id=loaded.config.experiment_id,
        seed=loaded.config.seed,
    )
    git_metadata = collect_git_metadata(project_root)
    environment_metadata = collect_environment_metadata(project_root)
    started_at = utc_now_text()
    try:
        output.mkdir()
    except FileExistsError as error:
        raise RunCreationError(f"output path already exists: {output}") from error

    try:
        write_text_file(output / ".incomplete", "run bundle is not complete\n")
        result, events, event_keys = run_smoke_experiment(loaded.config)
        write_json_file(output / "config.json", loaded.config.to_mapping())
        write_json_file(output / "data_manifest.json", loaded.manifest.to_mapping())
        with JsonlEventWriter(output / "events.jsonl") as writer:
            for event in events:
                writer.write(event.to_mapping())
        write_json_file(output / "result.json", result.to_mapping(run_id))

        artifact_paths = (
            "config.json",
            "data_manifest.json",
            "events.jsonl",
            "result.json",
        )
        artifacts = tuple(
            digest_artifact(output, relative_path) for relative_path in artifact_paths
        )
        metadata = RunMetadata(
            schema_version=1,
            run_id=run_id,
            experiment_id=loaded.config.experiment_id,
            claim_scope=CLAIM_SCOPE,
            started_at=started_at,
            finished_at=utc_now_text(),
            git=git_metadata,
            config_sha256=loaded.config_sha256,
            config_source=SourceArtifact(
                path=str(loaded.config_path),
                source_sha256=loaded.config_source_sha256,
                semantic_sha256=loaded.config_sha256,
            ),
            data_manifest_id=loaded.manifest.manifest_id,
            data_manifest=SourceArtifact(
                path=str(loaded.manifest_path),
                source_sha256=loaded.manifest_source_sha256,
                semantic_sha256=loaded.manifest_sha256,
            ),
            model_checkpoint_sha256=None,
            environment=environment_metadata,
            event_keys=build_event_key_metadata(
                schema_version=EVENT_KEY_SCHEMA_VERSION,
                root_key=root_key,
                keys=event_keys,
            ),
            artifacts=artifacts,
        )
        metadata_path = output / "metadata.json"
        write_json_file(metadata_path, metadata.to_mapping())
        write_text_file(
            output / "metadata.sha256",
            f"{sha256_file(metadata_path)}  metadata.json\n",
        )
        _fsync_directory(output)
        (output / ".incomplete").unlink()
        write_json_file(
            output / "completion.json",
            RunCompletion(
                schema_version=1,
                status="completed",
                run_id=run_id,
                metadata_sha256=sha256_file(metadata_path),
            ).to_mapping(),
        )
        return metadata
    except Exception as error:
        _checked_remove_failed_run_bundle(output, primary_error=error)
        raise


__all__ = [
    "RUN_BUNDLE_FILES",
    "RunCompletion",
    "RunCreationError",
    "create_run_bundle",
    "derive_run_identity",
    "find_project_root",
]
