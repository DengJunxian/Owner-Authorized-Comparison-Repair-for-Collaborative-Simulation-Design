"""Replayable, synthetic-only receipts for strict temporal table transforms.

The receipt format deliberately supports only two closed operators:

``DIRECT``
    Select exactly one row and return its complete scalar ``values`` mapping.

``MEAN``
    Select one or more rows, compute ``math.fsum(values) / n`` for one numeric
    input column, and insert that value into a committed scalar output template.

Callers must supply every row from every table named by the plan.  Verification
re-runs selection over those complete tables and compares the selected rows,
derived output, and verifier-derived output record with the serialized receipt.
This closes a deterministic synthetic replay path; it does *not* establish that
a caller-supplied table is a complete or externally authoritative real source.
Accordingly, this version cannot issue a real-source authority claim.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path, PurePosixPath
from typing import Literal, NoReturn, Self, cast

from policytwin.data.temporal_firewall import (
    AccessPurpose,
    CutoffPolicy,
    DataPartition,
    RuleCode,
    TableScalar,
    TemporalFirewall,
    TemporalRecord,
    TemporalSchemaError,
    TemporalTableRow,
    TransformationLeakageError,
    WindowAggregationSpec,
    decode_temporal_table,
    table_values_sha256,
    temporal_records_sha256,
)
from policytwin.provenance import canonical_json_bytes, sha256_bytes

TRANSFORMATION_RECEIPT_SCHEMA_VERSION = 1
_OUTPUT_SOURCE = "policytwin.synthetic-transformation"
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


class TransformationReceiptError(TransformationLeakageError):
    """Raised when a transformation cannot be reproduced exactly."""


class TransformationReceiptSchemaError(TransformationReceiptError):
    """Raised when a transformation plan or receipt is malformed."""


class TransformationReceiptIntegrityError(TransformationReceiptError):
    """Raised when replay disagrees with a committed receipt."""


class TransformationOperator(StrEnum):
    """Closed transformation operator set for schema version one."""

    DIRECT = "direct"
    MEAN = "mean"


def _fail(message: str) -> NoReturn:
    raise TransformationReceiptSchemaError(message)


def _integrity_fail(message: str) -> NoReturn:
    raise TransformationReceiptIntegrityError(message)


def _text(value: object, field: str, *, maximum: int = 1024) -> str:
    if (
        not isinstance(value, str)
        or not value.strip()
        or len(value) > maximum
        or "\x00" in value
    ):
        _fail(f"{field} must be a non-empty string of at most {maximum} chars")
    return value


def _identifier(value: object, field: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER_PATTERN.fullmatch(value) is None:
        _fail(f"{field} must be a stable ASCII identifier")
    return value


def _sha256(value: object, field: str) -> str:
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        _fail(f"{field} must be a lowercase SHA-256 digest")
    return value


def _datetime(value: object, field: str) -> datetime:
    if not isinstance(value, datetime):
        _fail(f"{field} must be a datetime")
    if value.tzinfo is None or value.utcoffset() is None:
        _fail(f"{field} must have an explicit timezone")
    try:
        return value.astimezone(UTC)
    except (OverflowError, ValueError) as error:
        raise TransformationReceiptSchemaError(
            f"{field} is outside the UTC range"
        ) from error


def _parsed_datetime(value: object, field: str) -> datetime:
    if not isinstance(value, str) or not value:
        _fail(f"{field} must be a non-empty ISO 8601 timestamp")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as error:
        raise TransformationReceiptSchemaError(f"{field} must be ISO 8601") from error
    return _datetime(parsed, field)


def _relative_path(value: object) -> str:
    if not isinstance(value, str) or not value or "\\" in value or "://" in value:
        _fail("source path must be a non-empty portable relative path")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or path.as_posix() != value
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        _fail("source path must not escape its frozen snapshot")
    return value


def _fields(
    mapping: Mapping[str, object], expected: frozenset[str], context: str
) -> None:
    actual = frozenset(mapping)
    missing = sorted(expected - actual)
    unknown = sorted(actual - expected)
    if missing:
        _fail(f"{context} is missing fields: {', '.join(missing)}")
    if unknown:
        _fail(f"{context} has unknown fields: {', '.join(unknown)}")


def _mapping(value: object, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        _fail(f"{field} must be an object with string keys")
    return cast(Mapping[str, object], value)


def _array(value: object, field: str) -> Sequence[object]:
    if not isinstance(value, Sequence) or isinstance(value, str | bytes | bytearray):
        _fail(f"{field} must be an array")
    return cast(Sequence[object], value)


def _strict_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            _fail(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> NoReturn:
    _fail(f"non-finite JSON number is forbidden: {value}")


def _json_object(content: bytes, context: str) -> dict[str, object]:
    try:
        parsed = json.loads(
            content.decode("utf-8"),
            object_pairs_hook=_strict_pairs,
            parse_constant=_reject_constant,
        )
    except (UnicodeError, json.JSONDecodeError) as error:
        raise TransformationReceiptSchemaError(
            f"invalid JSON in {context}: {error}"
        ) from error
    if not isinstance(parsed, dict):
        _fail(f"top-level JSON in {context} must be an object")
    return cast(dict[str, object], parsed)


def _scalar(value: object, field: str) -> TableScalar:
    if value is not None and not isinstance(value, str | int | float | bool):
        _fail(f"{field} must be a JSON scalar")
    if isinstance(value, float) and not math.isfinite(value):
        _fail(f"{field} must not be non-finite")
    return value


def _scalar_items(
    values: Mapping[str, object] | Iterable[tuple[str, TableScalar]],
    field: str,
    *,
    allow_empty: bool,
) -> tuple[tuple[str, TableScalar], ...]:
    raw_items = tuple(values.items()) if isinstance(values, Mapping) else tuple(values)
    normalized: list[tuple[str, TableScalar]] = []
    for key, value in raw_items:
        normalized.append(
            (_text(key, f"{field} key", maximum=256), _scalar(value, field))
        )
    if not allow_empty and not normalized:
        _fail(f"{field} must not be empty")
    if len({key for key, _ in normalized}) != len(normalized):
        _fail(f"{field} contains duplicate keys")
    return tuple(sorted(normalized))


def _same_scalar(left: TableScalar, right: TableScalar) -> bool:
    """Compare JSON scalars without treating booleans as integers."""
    return type(left) is type(right) and left == right


def transformation_implementation_sha256() -> str:
    """Hash the complete source bytes implementing receipt replay.

    Persisted receipts intentionally stop verifying when this source file
    changes.  Historical replay therefore requires retention of these exact
    bytes; this digest is not an external custody or trusted-timestamp claim.
    """

    try:
        source = Path(__file__).read_bytes()
    except OSError as error:
        raise TransformationReceiptIntegrityError(
            f"cannot read transformation implementation source: {error}"
        ) from error
    return sha256_bytes(source)


@dataclass(frozen=True, slots=True)
class TransformationPlan:
    """Immutable query, temporal window, and output commitment."""

    schema_version: int
    transformation_id: str
    operator: TransformationOperator
    source_paths: tuple[str, ...]
    selection_equals: tuple[tuple[str, TableScalar], ...]
    window_start: datetime
    window_end: datetime
    output_time: datetime
    output_partition: DataPartition
    output_record_id: str
    input_column: str | None
    output_template: tuple[tuple[str, TableScalar], ...]
    output_field: str | None
    executed_at: datetime
    purpose: AccessPurpose
    construction_contract_sha256: str
    cutoff_policy_sha256: str
    implementation_source_sha256: str

    def __post_init__(self) -> None:
        if isinstance(self.schema_version, bool) or self.schema_version != 1:
            _fail("TransformationPlan schema_version must be 1")
        _identifier(self.transformation_id, "transformation_id")
        if not isinstance(self.operator, TransformationOperator):
            _fail("operator must be a TransformationOperator")
        if not isinstance(self.source_paths, tuple) or not self.source_paths:
            _fail("source_paths must be a non-empty immutable tuple")
        paths = tuple(sorted(_relative_path(path) for path in self.source_paths))
        if len(set(paths)) != len(paths):
            _fail("source_paths must be unique")
        object.__setattr__(self, "source_paths", paths)
        if not isinstance(self.selection_equals, tuple):
            _fail("selection_equals must be an immutable tuple")
        object.__setattr__(
            self,
            "selection_equals",
            _scalar_items(
                self.selection_equals,
                "selection_equals",
                allow_empty=False,
            ),
        )
        for field in ("window_start", "window_end", "output_time"):
            object.__setattr__(self, field, _datetime(getattr(self, field), field))
        if self.window_end < self.window_start:
            _fail("window_end must not precede window_start")
        if self.output_time < self.window_end:
            _fail("output_time must not precede window_end")
        if not isinstance(self.output_partition, DataPartition):
            _fail("output_partition must be a DataPartition")
        _identifier(self.output_record_id, "output_record_id")
        if not isinstance(self.output_template, tuple):
            _fail("output_template must be an immutable tuple")
        template = _scalar_items(
            self.output_template,
            "output_template",
            allow_empty=True,
        )
        object.__setattr__(self, "output_template", template)
        object.__setattr__(
            self, "executed_at", _datetime(self.executed_at, "executed_at")
        )
        if self.executed_at < self.output_time:
            _fail("executed_at must not precede output_time")
        if not isinstance(self.purpose, AccessPurpose):
            _fail("purpose must be an AccessPurpose")
        _sha256(self.construction_contract_sha256, "construction_contract_sha256")
        _sha256(self.cutoff_policy_sha256, "cutoff_policy_sha256")
        _sha256(self.implementation_source_sha256, "implementation_source_sha256")
        if self.operator is TransformationOperator.DIRECT:
            if self.input_column is not None or self.output_field is not None:
                _fail("DIRECT forbids input_column and output_field")
            if template:
                _fail("DIRECT returns the complete row and forbids output_template")
        else:
            if self.input_column is None or self.output_field is None:
                _fail("MEAN requires input_column and output_field")
            _text(self.input_column, "input_column", maximum=256)
            _text(self.output_field, "output_field", maximum=256)
            if self.output_field in dict(template):
                _fail("MEAN output_field must not already exist in output_template")

    def to_mapping(self) -> dict[str, object]:
        """Return the canonical plan representation."""
        return {
            "construction_contract_sha256": self.construction_contract_sha256,
            "cutoff_policy_sha256": self.cutoff_policy_sha256,
            "executed_at": self.executed_at.isoformat(),
            "implementation_source_sha256": self.implementation_source_sha256,
            "input_column": self.input_column,
            "operator": self.operator.value,
            "output_field": self.output_field,
            "output_partition": self.output_partition.value,
            "output_record_id": self.output_record_id,
            "output_template": dict(self.output_template),
            "output_time": self.output_time.isoformat(),
            "purpose": self.purpose.value,
            "schema_version": self.schema_version,
            "selection_equals": dict(self.selection_equals),
            "source_paths": list(self.source_paths),
            "transformation_id": self.transformation_id,
            "window_end": self.window_end.isoformat(),
            "window_start": self.window_start.isoformat(),
        }

    @property
    def sha256(self) -> str:
        """Return the canonical semantic plan commitment."""
        return sha256_bytes(canonical_json_bytes(self.to_mapping()))

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, object]) -> Self:
        """Strictly reconstruct a transformation plan."""
        _fields(
            mapping,
            frozenset(
                {
                    "construction_contract_sha256",
                    "cutoff_policy_sha256",
                    "executed_at",
                    "implementation_source_sha256",
                    "input_column",
                    "operator",
                    "output_field",
                    "output_partition",
                    "output_record_id",
                    "output_template",
                    "output_time",
                    "purpose",
                    "schema_version",
                    "selection_equals",
                    "source_paths",
                    "transformation_id",
                    "window_end",
                    "window_start",
                }
            ),
            "transformation plan",
        )
        schema_version = mapping["schema_version"]
        if not isinstance(schema_version, int) or isinstance(schema_version, bool):
            _fail("schema_version must be an integer")
        operator_value = mapping["operator"]
        partition_value = mapping["output_partition"]
        purpose_value = mapping["purpose"]
        if not all(
            isinstance(value, str)
            for value in (operator_value, partition_value, purpose_value)
        ):
            _fail("operator, output_partition, and purpose must be strings")
        try:
            operator = TransformationOperator(cast(str, operator_value))
            partition = DataPartition(cast(str, partition_value))
            purpose = AccessPurpose(cast(str, purpose_value))
        except ValueError as error:
            raise TransformationReceiptSchemaError(
                "plan contains an unsupported enum value"
            ) from error
        input_column = mapping["input_column"]
        output_field = mapping["output_field"]
        if input_column is not None and not isinstance(input_column, str):
            _fail("input_column must be null or a string")
        if output_field is not None and not isinstance(output_field, str):
            _fail("output_field must be null or a string")
        return cls(
            schema_version=schema_version,
            transformation_id=_identifier(
                mapping["transformation_id"], "transformation_id"
            ),
            operator=operator,
            source_paths=tuple(
                _relative_path(item)
                for item in _array(mapping["source_paths"], "source_paths")
            ),
            selection_equals=_scalar_items(
                _mapping(mapping["selection_equals"], "selection_equals"),
                "selection_equals",
                allow_empty=False,
            ),
            window_start=_parsed_datetime(mapping["window_start"], "window_start"),
            window_end=_parsed_datetime(mapping["window_end"], "window_end"),
            output_time=_parsed_datetime(mapping["output_time"], "output_time"),
            output_partition=partition,
            output_record_id=_identifier(
                mapping["output_record_id"], "output_record_id"
            ),
            input_column=input_column,
            output_template=_scalar_items(
                _mapping(mapping["output_template"], "output_template"),
                "output_template",
                allow_empty=True,
            ),
            output_field=output_field,
            executed_at=_parsed_datetime(mapping["executed_at"], "executed_at"),
            purpose=purpose,
            construction_contract_sha256=_sha256(
                mapping["construction_contract_sha256"],
                "construction_contract_sha256",
            ),
            cutoff_policy_sha256=_sha256(
                mapping["cutoff_policy_sha256"], "cutoff_policy_sha256"
            ),
            implementation_source_sha256=_sha256(
                mapping["implementation_source_sha256"],
                "implementation_source_sha256",
            ),
        )


@dataclass(frozen=True, slots=True)
class TransformationInputRow:
    """One exact source row selected from one committed source path."""

    source_path: str
    row: TemporalTableRow

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_path", _relative_path(self.source_path))
        if not isinstance(self.row, TemporalTableRow):
            _fail("input row must be a TemporalTableRow")

    def to_mapping(self) -> dict[str, object]:
        """Return the canonical selected-row representation."""
        return {"row": self.row.to_mapping(), "source_path": self.source_path}

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, object]) -> Self:
        """Strictly reconstruct a selected source row."""
        _fields(mapping, frozenset({"row", "source_path"}), "input row")
        try:
            row = TemporalTableRow.from_mapping(_mapping(mapping["row"], "row"))
        except TemporalSchemaError as error:
            raise TransformationReceiptSchemaError(str(error)) from error
        return cls(source_path=_relative_path(mapping["source_path"]), row=row)


def _derive_output(
    plan: TransformationPlan,
    input_rows: tuple[TransformationInputRow, ...],
) -> tuple[tuple[str, TableScalar], ...]:
    if not input_rows:
        _fail("transformation selection is empty")
    if plan.operator is TransformationOperator.DIRECT:
        if len(input_rows) != 1:
            _fail("DIRECT requires exactly one selected row")
        return input_rows[0].row.values
    assert plan.input_column is not None
    assert plan.output_field is not None
    numeric_values: list[int | float] = []
    for input_row in input_rows:
        values = dict(input_row.row.values)
        if plan.input_column not in values:
            _fail(
                f"MEAN input column {plan.input_column!r} is absent from "
                f"record {input_row.row.record.record_id!r}"
            )
        value = values[plan.input_column]
        if isinstance(value, bool) or not isinstance(value, int | float):
            _fail("MEAN inputs must be numeric and must not be bool, null, or text")
        if isinstance(value, float) and not math.isfinite(value):
            _fail("MEAN inputs must be finite")
        numeric_values.append(value)
    try:
        mean = math.fsum(numeric_values) / len(numeric_values)
    except (OverflowError, ValueError) as error:
        raise TransformationReceiptSchemaError(
            "MEAN could not produce a finite deterministic value"
        ) from error
    if not math.isfinite(mean):
        _fail("MEAN output must be finite")
    output = dict(plan.output_template)
    output[plan.output_field] = mean
    return _scalar_items(output, "derived output", allow_empty=False)


def _derive_output_record(
    plan: TransformationPlan,
    input_rows: tuple[TransformationInputRow, ...],
    output_values: tuple[tuple[str, TableScalar], ...],
) -> TemporalRecord:
    license_tags = {input_row.row.record.license_tag for input_row in input_rows}
    if len(license_tags) != 1:
        _fail("mixed input license tags cannot be collapsed into one output record")
    latest_input_ingestion = max(
        input_row.row.record.ingestion_time for input_row in input_rows
    )
    if plan.executed_at < latest_input_ingestion:
        _fail("executed_at must not precede any selected input ingestion_time")
    return TemporalRecord(
        record_id=plan.output_record_id,
        partition=plan.output_partition,
        event_time=plan.output_time,
        available_time=plan.output_time,
        ingestion_time=plan.executed_at,
        source=_OUTPUT_SOURCE,
        version=f"receipt-v1:{plan.implementation_source_sha256}",
        content_hash=table_values_sha256(dict(output_values)),
        license_tag=next(iter(license_tags)),
    )


@dataclass(frozen=True, slots=True)
class TransformationReceipt:
    """Serialized evidence for one exactly replayed synthetic transformation."""

    schema_version: int
    plan: TransformationPlan
    input_rows: tuple[TransformationInputRow, ...]
    output_values: tuple[tuple[str, TableScalar], ...]
    output_record: TemporalRecord
    synthetic_only: Literal[True] = True
    source_authority_verified: Literal[False] = False

    def __post_init__(self) -> None:
        if isinstance(self.schema_version, bool) or self.schema_version != 1:
            _fail("TransformationReceipt schema_version must be 1")
        if not isinstance(self.plan, TransformationPlan):
            _fail("receipt plan must be a TransformationPlan")
        if not isinstance(self.input_rows, tuple) or not self.input_rows:
            _fail("receipt input_rows must be a non-empty immutable tuple")
        if not all(
            isinstance(item, TransformationInputRow) for item in self.input_rows
        ):
            _fail("receipt input_rows contain an invalid value")
        sorted_rows = tuple(
            sorted(
                self.input_rows,
                key=lambda item: (item.source_path, item.row.record.record_id),
            )
        )
        keys = tuple(
            (item.source_path, item.row.record.record_id) for item in sorted_rows
        )
        if len(set(keys)) != len(keys):
            _fail("receipt input rows must be unique")
        if any(item.source_path not in self.plan.source_paths for item in sorted_rows):
            _fail("receipt input row uses a path absent from the plan")
        object.__setattr__(self, "input_rows", sorted_rows)
        if not isinstance(self.output_values, tuple):
            _fail("output_values must be an immutable tuple")
        output_values = _scalar_items(
            self.output_values,
            "output_values",
            allow_empty=False,
        )
        object.__setattr__(self, "output_values", output_values)
        if not isinstance(self.output_record, TemporalRecord):
            _fail("output_record must be a TemporalRecord")
        if self.synthetic_only is not True:
            _fail("transformation receipts must remain explicitly synthetic_only")
        if self.source_authority_verified is not False:
            _fail("real source authority is unsupported and must remain false")
        expected_values = _derive_output(self.plan, sorted_rows)
        if expected_values != output_values:
            _integrity_fail("receipt output does not match replayed input rows")
        expected_record = _derive_output_record(self.plan, sorted_rows, output_values)
        if expected_record != self.output_record:
            _integrity_fail("receipt output record was not verifier-derived")

    @property
    def output_mapping(self) -> dict[str, TableScalar]:
        """Return a fresh mapping of the complete derived scalar output."""
        return dict(self.output_values)

    @property
    def input_records(self) -> tuple[TemporalRecord, ...]:
        """Return exact temporal records used by the transformation."""
        return tuple(item.row.record for item in self.input_rows)

    @property
    def output_payload_sha256(self) -> str:
        """Return the canonical output payload commitment."""
        return table_values_sha256(self.output_mapping)

    def to_mapping(self) -> dict[str, object]:
        """Return the canonical receipt representation."""
        return {
            "input_rows": [item.to_mapping() for item in self.input_rows],
            "output_record": self.output_record.to_mapping(),
            "output_values": self.output_mapping,
            "plan": self.plan.to_mapping(),
            "schema_version": self.schema_version,
            "source_authority_verified": self.source_authority_verified,
            "synthetic_only": self.synthetic_only,
        }

    def to_json_bytes(self) -> bytes:
        """Encode this receipt as canonical JSON bytes."""
        return canonical_json_bytes(self.to_mapping())

    @property
    def sha256(self) -> str:
        """Return the canonical semantic receipt commitment."""
        return sha256_bytes(self.to_json_bytes())

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, object]) -> Self:
        """Strictly reconstruct a transformation receipt."""
        _fields(
            mapping,
            frozenset(
                {
                    "input_rows",
                    "output_record",
                    "output_values",
                    "plan",
                    "schema_version",
                    "source_authority_verified",
                    "synthetic_only",
                }
            ),
            "transformation receipt",
        )
        schema_version = mapping["schema_version"]
        if not isinstance(schema_version, int) or isinstance(schema_version, bool):
            _fail("schema_version must be an integer")
        synthetic = mapping["synthetic_only"]
        authority = mapping["source_authority_verified"]
        if synthetic is not True:
            _fail("transformation receipts must be explicitly synthetic_only")
        if authority is not False:
            _fail("real source authority is unsupported and must remain false")
        try:
            output_record = TemporalRecord.from_mapping(
                _mapping(mapping["output_record"], "output_record")
            )
        except TemporalSchemaError as error:
            raise TransformationReceiptSchemaError(str(error)) from error
        return cls(
            schema_version=schema_version,
            plan=TransformationPlan.from_mapping(_mapping(mapping["plan"], "plan")),
            input_rows=tuple(
                TransformationInputRow.from_mapping(_mapping(item, "input_rows item"))
                for item in _array(mapping["input_rows"], "input_rows")
            ),
            output_values=_scalar_items(
                _mapping(mapping["output_values"], "output_values"),
                "output_values",
                allow_empty=False,
            ),
            output_record=output_record,
            synthetic_only=True,
            source_authority_verified=False,
        )

    @classmethod
    def from_json_bytes(cls, content: bytes) -> Self:
        """Strictly decode canonical receipt JSON and reject alternate bytes."""
        if not isinstance(content, bytes):
            _fail("receipt content must be immutable bytes")
        mapping = _json_object(content, "transformation receipt")
        receipt = cls.from_mapping(mapping)
        if receipt.to_json_bytes() != content:
            _fail("transformation receipt JSON must use canonical encoding")
        return receipt


@dataclass(frozen=True, slots=True)
class VerifiedTransformation:
    """Immutable replay result returned only after complete-table verification."""

    receipt_sha256: str
    output_values: tuple[tuple[str, TableScalar], ...]
    output_record: TemporalRecord
    input_records: tuple[TemporalRecord, ...]

    def __post_init__(self) -> None:
        _sha256(self.receipt_sha256, "receipt_sha256")
        object.__setattr__(
            self,
            "output_values",
            _scalar_items(self.output_values, "output_values", allow_empty=False),
        )
        if not isinstance(self.output_record, TemporalRecord):
            _fail("verified output_record must be a TemporalRecord")
        if not isinstance(self.input_records, tuple) or not self.input_records:
            _fail("verified input_records must be a non-empty immutable tuple")
        if not all(isinstance(record, TemporalRecord) for record in self.input_records):
            _fail("verified input_records contain an invalid record")

    @property
    def output_mapping(self) -> dict[str, TableScalar]:
        """Return a fresh mapping of the replayed output."""
        return dict(self.output_values)


def _normalize_complete_tables(
    plan: TransformationPlan,
    complete_source_tables: Mapping[str, Iterable[TemporalTableRow]],
) -> tuple[TransformationInputRow, ...]:
    if not isinstance(complete_source_tables, Mapping) or not all(
        isinstance(path, str) for path in complete_source_tables
    ):
        _fail("complete_source_tables must map source paths to complete row iterables")
    actual_paths = {_relative_path(path) for path in complete_source_tables}
    if actual_paths != set(plan.source_paths):
        _fail("complete_source_tables paths must exactly equal plan source_paths")
    all_rows: list[TransformationInputRow] = []
    for path in sorted(actual_paths):
        try:
            rows = tuple(complete_source_tables[path])
        except TypeError as error:
            raise TransformationReceiptSchemaError(
                f"rows for {path!r} must be iterable"
            ) from error
        if not rows:
            _fail(f"complete source table {path!r} must not be empty")
        if not all(isinstance(row, TemporalTableRow) for row in rows):
            _fail(f"complete source table {path!r} contains a non-temporal row")
        all_rows.extend(
            TransformationInputRow(source_path=path, row=row) for row in rows
        )
    record_ids = tuple(item.row.record.record_id for item in all_rows)
    if len(set(record_ids)) != len(record_ids):
        _fail("complete source tables contain duplicate record IDs")
    return tuple(
        sorted(all_rows, key=lambda item: (item.source_path, item.row.record.record_id))
    )


def _select_rows(
    plan: TransformationPlan,
    all_rows: tuple[TransformationInputRow, ...],
) -> tuple[TransformationInputRow, ...]:
    predicate = dict(plan.selection_equals)

    def selected(item: TransformationInputRow) -> bool:
        record = item.row.record
        if not plan.window_start <= record.event_time <= plan.window_end:
            return False
        values = dict(item.row.values)
        return all(
            key in values and _same_scalar(values[key], expected)
            for key, expected in predicate.items()
        )

    result = tuple(item for item in all_rows if selected(item))
    if not result:
        _fail("transformation selection is empty")
    if plan.operator is TransformationOperator.DIRECT and len(result) != 1:
        _fail("DIRECT requires exactly one matching row across complete tables")
    return result


def _check_plan_authority(plan: TransformationPlan, policy: CutoffPolicy) -> None:
    if plan.cutoff_policy_sha256 != policy.sha256:
        _integrity_fail("plan cutoff policy hash does not match the active firewall")
    current_implementation = transformation_implementation_sha256()
    if plan.implementation_source_sha256 != current_implementation:
        _integrity_fail("plan implementation source hash does not match replay code")


def _replay(
    *,
    plan: TransformationPlan,
    complete_source_tables: Mapping[str, Iterable[TemporalTableRow]],
    firewall: TemporalFirewall,
) -> tuple[
    tuple[TransformationInputRow, ...],
    tuple[tuple[str, TableScalar], ...],
    TemporalRecord,
]:
    if not isinstance(plan, TransformationPlan):
        _fail("plan must be a TransformationPlan")
    if not isinstance(firewall, TemporalFirewall):
        _fail("firewall must be a TemporalFirewall")
    _check_plan_authority(plan, firewall.policy)
    all_rows = _normalize_complete_tables(plan, complete_source_tables)
    # Every row has already been exposed to selection, so fail closed if any
    # complete-table row violates the cutoff, partition, or license policy.
    firewall.validate_records(item.row.record for item in all_rows)
    input_rows = _select_rows(plan, all_rows)
    window = WindowAggregationSpec(
        records=tuple(item.row.record for item in input_rows),
        window_start=plan.window_start,
        window_end=plan.window_end,
        output_time=plan.output_time,
        output_partition=plan.output_partition,
    )
    firewall.check_window_aggregation(window, plan.purpose)
    output_values = _derive_output(plan, input_rows)
    output_record = _derive_output_record(plan, input_rows, output_values)
    firewall.validate_record(output_record)
    return input_rows, output_values, output_record


def _audit_receipt_outcome(
    *,
    firewall: TemporalFirewall,
    action: str,
    outcome: Literal["allowed", "rejected"],
    plan: TransformationPlan | None,
    error: Exception | None = None,
    receipt_sha256: str | None = None,
    input_records_sha256: str | None = None,
    output_payload_sha256: str | None = None,
) -> None:
    details: dict[str, object] = {
        "authority_boundary": "synthetic_table_replay_only",
        "source_authority_verified": False,
        "synthetic_only": True,
    }
    if plan is not None:
        details["construction_contract_sha256"] = plan.construction_contract_sha256
        details["executed_at"] = plan.executed_at.isoformat()
        details["implementation_source_sha256"] = plan.implementation_source_sha256
        details["operator"] = plan.operator.value
        details["plan_sha256"] = plan.sha256
        details["purpose"] = plan.purpose.value
        details["source_paths"] = list(plan.source_paths)
        details["transformation_id"] = plan.transformation_id
    if receipt_sha256 is not None:
        details["receipt_sha256"] = receipt_sha256
    if input_records_sha256 is not None:
        details["input_records_sha256"] = input_records_sha256
    if output_payload_sha256 is not None:
        details["output_payload_sha256"] = output_payload_sha256
    if error is not None:
        details["error"] = str(error)[:512]
        details["error_type"] = type(error).__name__
    rule = RuleCode.OK
    if outcome == "rejected":
        rule = (
            RuleCode.HASH_MISMATCH
            if isinstance(error, TransformationReceiptIntegrityError)
            else RuleCode.FEATURE_LINEAGE_INVALID
        )
    firewall.audit_log.record(
        action=action,
        outcome=outcome,
        rule_code=rule,
        policy_sha256=firewall.policy.sha256,
        cutoff=firewall.policy.cutoff.isoformat(),
        artifact_id=plan.transformation_id if plan is not None else None,
        record_id=plan.output_record_id if plan is not None else None,
        partition=(plan.output_partition.value if plan is not None else None),
        details=details,
    )


def issue_transformation_receipt(
    *,
    plan: TransformationPlan,
    complete_source_tables: Mapping[str, Iterable[TemporalTableRow]],
    firewall: TemporalFirewall,
) -> TransformationReceipt:
    """Replay complete synthetic tables and issue an immutable receipt.

    ``complete_source_tables`` is a caller obligation.  In a frozen evaluation
    it must come from full, already-validated TABLE artifact payloads rather
    than from a selected-row subset.
    """

    try:
        input_rows, output_values, output_record = _replay(
            plan=plan,
            complete_source_tables=complete_source_tables,
            firewall=firewall,
        )
        receipt = TransformationReceipt(
            schema_version=TRANSFORMATION_RECEIPT_SCHEMA_VERSION,
            plan=plan,
            input_rows=input_rows,
            output_values=output_values,
            output_record=output_record,
            synthetic_only=True,
            source_authority_verified=False,
        )
    except Exception as error:
        _audit_receipt_outcome(
            firewall=firewall,
            action="issue_transformation_receipt",
            outcome="rejected",
            plan=plan if isinstance(plan, TransformationPlan) else None,
            error=error,
        )
        raise
    _audit_receipt_outcome(
        firewall=firewall,
        action="issue_transformation_receipt",
        outcome="allowed",
        plan=plan,
        receipt_sha256=receipt.sha256,
        input_records_sha256=temporal_records_sha256(receipt.input_records),
        output_payload_sha256=receipt.output_payload_sha256,
    )
    return receipt


def verify_transformation_receipt(
    *,
    receipt: TransformationReceipt,
    complete_source_tables: Mapping[str, Iterable[TemporalTableRow]],
    firewall: TemporalFirewall,
) -> VerifiedTransformation:
    """Re-select from complete tables and verify every receipt commitment."""

    plan = receipt.plan if isinstance(receipt, TransformationReceipt) else None
    try:
        if not isinstance(receipt, TransformationReceipt):
            _fail("receipt must be a TransformationReceipt")
        input_rows, output_values, output_record = _replay(
            plan=receipt.plan,
            complete_source_tables=complete_source_tables,
            firewall=firewall,
        )
        if input_rows != receipt.input_rows:
            _integrity_fail(
                "receipt omits, adds, or changes rows selected from complete tables"
            )
        if output_values != receipt.output_values:
            _integrity_fail("replayed output differs from receipt output")
        if output_record != receipt.output_record:
            _integrity_fail("replayed output record differs from receipt metadata")
        verified = VerifiedTransformation(
            receipt_sha256=receipt.sha256,
            output_values=output_values,
            output_record=output_record,
            input_records=tuple(item.row.record for item in input_rows),
        )
    except Exception as error:
        _audit_receipt_outcome(
            firewall=firewall,
            action="verify_transformation_receipt",
            outcome="rejected",
            plan=plan,
            error=error,
            receipt_sha256=(
                receipt.sha256 if isinstance(receipt, TransformationReceipt) else None
            ),
        )
        raise
    _audit_receipt_outcome(
        firewall=firewall,
        action="verify_transformation_receipt",
        outcome="allowed",
        plan=receipt.plan,
        receipt_sha256=receipt.sha256,
        input_records_sha256=temporal_records_sha256(verified.input_records),
        output_payload_sha256=verified.output_record.content_hash,
    )
    return verified


__all__ = [
    "TRANSFORMATION_RECEIPT_SCHEMA_VERSION",
    "TransformationInputRow",
    "TransformationOperator",
    "TransformationPlan",
    "TransformationReceipt",
    "TransformationReceiptError",
    "TransformationReceiptIntegrityError",
    "TransformationReceiptSchemaError",
    "VerifiedTransformation",
    "decode_temporal_table",
    "issue_transformation_receipt",
    "transformation_implementation_sha256",
    "verify_transformation_receipt",
]
