"""Deterministic text deidentification without raw-text retention.

Only the returned sanitized text is retained.  Its receipt contains aggregate
redaction counts and content commitments, never the input text, matched terms,
real entity identifiers, or wall-clock values.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import re
from dataclasses import dataclass
from enum import StrEnum
from typing import NoReturn

DEIDENTIFICATION_SCHEMA_VERSION = 1
ENTITY_MARKER = "<PT_ENTITY>"
POLICY_MARKER = "<PT_POLICY>"
YEAR_MARKER = "<PT_YEAR>"

_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_YEAR_PATTERN = re.compile(
    r"(?<![A-Za-z0-9_])(?:1[89][0-9]{2}|20[0-9]{2}|21[0-9]{2})"
    r"(?![A-Za-z0-9_])"
)
_SAFE_MARKER_PATTERN = re.compile(
    r"<PT_(?:ENTITY|POLICY|YEAR(?:_(?:CURRENT|PLUS_[0-9]{1,4}|"
    r"MINUS_[0-9]{1,4}))?)>"
)


class DeidentificationError(ValueError):
    """Base class for invalid policies, inputs, or sanitized outputs."""


class UnsafeTextError(DeidentificationError):
    """Raised when text still contains a configured sensitive token or year."""


class YearMode(StrEnum):
    """Permitted handling of absolute four-digit years."""

    REMOVE = "remove"
    RELATIVE = "relative"


class RedactionKind(StrEnum):
    """Closed categories used internally for aggregate counters."""

    ENTITY = "entity"
    POLICY = "policy"


def _fail(message: str) -> NoReturn:
    raise DeidentificationError(message)


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _sha256(value: object, field: str) -> str:
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        _fail(f"{field} must be a lowercase SHA-256 commitment")
    return value


def _non_negative_int(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        _fail(f"{field} must be a non-negative integer")
    return value


def _canonical_json_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise DeidentificationError(
            "deidentification policy is not canonical JSON"
        ) from error


def _validate_term(value: object) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or len(value) > 256
        or any(ord(character) < 32 for character in value)
        or _SAFE_MARKER_PATTERN.search(value) is not None
    ):
        _fail("configured redaction terms must be bounded printable strings")
    return value


@dataclass(frozen=True, slots=True, repr=False)
class DeidentificationPolicy:
    """Immutable sensitive-term and absolute-year handling policy.

    ``repr`` and ``safe_mapping`` intentionally expose counts and commitments
    only, not configured real entity or policy names.
    """

    entity_names: tuple[str, ...] = ()
    policy_names: tuple[str, ...] = ()
    year_mode: YearMode = YearMode.REMOVE
    reference_year: int | None = None
    case_sensitive: bool = False
    commitment_key: bytes = b""
    schema_version: int = DEIDENTIFICATION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != DEIDENTIFICATION_SCHEMA_VERSION:
            _fail("unsupported deidentification schema version")
        if not isinstance(self.entity_names, tuple) or not isinstance(
            self.policy_names, tuple
        ):
            _fail("entity_names and policy_names must be immutable tuples")
        for term in (*self.entity_names, *self.policy_names):
            _validate_term(term)
        if not isinstance(self.year_mode, YearMode):
            _fail("year_mode must be a YearMode")
        if not isinstance(self.case_sensitive, bool):
            _fail("case_sensitive must be a boolean")
        if not isinstance(self.commitment_key, bytes) or len(self.commitment_key) < 32:
            _fail("commitment_key must contain at least 32 secret bytes")
        if self.year_mode is YearMode.RELATIVE:
            if (
                isinstance(self.reference_year, bool)
                or not isinstance(self.reference_year, int)
                or not 1800 <= self.reference_year <= 2199
            ):
                _fail("relative year mode requires reference_year in [1800, 2199]")
        elif self.reference_year is not None:
            _fail("reference_year is permitted only in relative year mode")

        normalized: set[str] = set()
        for term in (*self.entity_names, *self.policy_names):
            key = term if self.case_sensitive else term.casefold()
            if key in normalized:
                _fail("configured redaction terms must be unique across categories")
            normalized.add(key)

    @property
    def sha256(self) -> str:
        """Return the aggregate commitment to the complete private policy."""
        entity_names = sorted(
            self.entity_names,
            key=lambda value: (value.casefold(), value),
        )
        policy_names = sorted(
            self.policy_names,
            key=lambda value: (value.casefold(), value),
        )
        material = {
            "case_sensitive": self.case_sensitive,
            "entity_names": entity_names,
            "policy_names": policy_names,
            "reference_year": self.reference_year,
            "schema_version": self.schema_version,
            "year_mode": self.year_mode.value,
        }
        return hmac.new(
            self.commitment_key,
            b"policytwin.deidentification.policy.v1\0"
            + _canonical_json_bytes(material),
            hashlib.sha256,
        ).hexdigest()

    def input_commitment(self, raw_text: str) -> str:
        """Return a secret, domain-separated HMAC without retaining raw text."""
        if not isinstance(raw_text, str) or "\x00" in raw_text:
            _fail("raw_text must be a string without NUL bytes")
        return hmac.new(
            self.commitment_key,
            b"policytwin.deidentification.input.v1\0" + raw_text.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()

    def safe_mapping(self) -> dict[str, object]:
        """Return metadata safe for provenance; raw configured terms are absent."""
        return {
            "entity_count": len(self.entity_names),
            "policy_count": len(self.policy_names),
            "policy_sha256": self.sha256,
            "schema_version": self.schema_version,
            "year_mode": self.year_mode.value,
        }

    def __repr__(self) -> str:
        return (
            "DeidentificationPolicy("
            f"policy_sha256={self.sha256!r}, "
            f"entity_count={len(self.entity_names)}, "
            f"policy_count={len(self.policy_names)}, "
            f"year_mode={self.year_mode.value!r})"
        )


@dataclass(frozen=True, slots=True)
class RedactionCounts:
    """Aggregate counts that reveal no matched term or source span."""

    entity_count: int
    policy_count: int
    year_count: int

    def __post_init__(self) -> None:
        _non_negative_int(self.entity_count, "entity_count")
        _non_negative_int(self.policy_count, "policy_count")
        _non_negative_int(self.year_count, "year_count")

    @property
    def total_count(self) -> int:
        """Return the total number of replacements."""
        return self.entity_count + self.policy_count + self.year_count

    def to_mapping(self) -> dict[str, int]:
        """Return the fixed aggregate-counter representation."""
        return {
            "entity_count": self.entity_count,
            "policy_count": self.policy_count,
            "year_count": self.year_count,
        }


@dataclass(frozen=True, slots=True)
class DeidentificationReceipt:
    """Hash-and-count-only evidence for one sanitization operation."""

    input_sha256: str
    output_sha256: str
    policy_sha256: str
    counts: RedactionCounts

    def __post_init__(self) -> None:
        _sha256(self.input_sha256, "input_sha256")
        _sha256(self.output_sha256, "output_sha256")
        _sha256(self.policy_sha256, "policy_sha256")
        if not isinstance(self.counts, RedactionCounts):
            _fail("counts must be RedactionCounts")

    def to_mapping(self) -> dict[str, object]:
        """Return a log-safe representation with no raw or matched text."""
        return {
            "counts": self.counts.to_mapping(),
            "input_sha256": self.input_sha256,
            "output_sha256": self.output_sha256,
            "policy_sha256": self.policy_sha256,
        }


@dataclass(frozen=True, slots=True)
class SanitizedText:
    """Sanitized output plus its receipt; the original text is never retained."""

    text: str
    receipt: DeidentificationReceipt

    def __post_init__(self) -> None:
        if not isinstance(self.text, str) or "\x00" in self.text:
            _fail("sanitized text must be a string without NUL bytes")
        if not isinstance(self.receipt, DeidentificationReceipt):
            _fail("receipt must be a DeidentificationReceipt")
        if _sha256_text(self.text) != self.receipt.output_sha256:
            _fail("sanitized text does not match its output commitment")

    def to_mapping(self) -> dict[str, object]:
        """Return the sanitized payload and hash-and-count-only receipt."""
        return {"receipt": self.receipt.to_mapping(), "text": self.text}


class Anonymizer:
    """Compiled deterministic anonymizer for one immutable policy."""

    __slots__ = ("_kinds", "_pattern", "policy")

    def __init__(self, policy: DeidentificationPolicy) -> None:
        if not isinstance(policy, DeidentificationPolicy):
            _fail("policy must be a DeidentificationPolicy")
        self.policy = policy
        entries = [
            *((term, RedactionKind.ENTITY) for term in policy.entity_names),
            *((term, RedactionKind.POLICY) for term in policy.policy_names),
        ]
        entries.sort(
            key=lambda item: (
                -len(item[0]),
                item[0].casefold(),
                item[0],
                item[1].value,
            )
        )
        self._kinds = tuple(kind for _, kind in entries)
        if entries:
            alternatives = [
                f"(?P<t{index}>(?<![A-Za-z0-9_]){re.escape(term)}(?![A-Za-z0-9_]))"
                for index, (term, _) in enumerate(entries)
            ]
            flags = 0 if policy.case_sensitive else re.IGNORECASE
            self._pattern: re.Pattern[str] | None = re.compile(
                "|".join(alternatives), flags
            )
        else:
            self._pattern = None

    def sanitize(self, raw_text: str) -> SanitizedText:
        """Return sanitized text while retaining only its input commitment."""
        if not isinstance(raw_text, str) or "\x00" in raw_text:
            _fail("raw_text must be a string without NUL bytes")
        if _SAFE_MARKER_PATTERN.search(raw_text) is not None:
            _fail("raw_text contains a reserved deidentification marker")

        entity_count = 0
        policy_count = 0

        def replace_term(match: re.Match[str]) -> str:
            nonlocal entity_count, policy_count
            group = match.lastgroup
            if group is None or not group.startswith("t"):
                raise DeidentificationError("redaction matcher lost its category")
            kind = self._kinds[int(group[1:])]
            if kind is RedactionKind.ENTITY:
                entity_count += 1
                return ENTITY_MARKER
            policy_count += 1
            return POLICY_MARKER

        output = (
            raw_text
            if self._pattern is None
            else self._pattern.sub(replace_term, raw_text)
        )
        year_count = 0

        def replace_year(match: re.Match[str]) -> str:
            nonlocal year_count
            year_count += 1
            if self.policy.year_mode is YearMode.REMOVE:
                return YEAR_MARKER
            assert self.policy.reference_year is not None
            delta = int(match.group(0)) - self.policy.reference_year
            if delta == 0:
                return "<PT_YEAR_CURRENT>"
            direction = "PLUS" if delta > 0 else "MINUS"
            return f"<PT_YEAR_{direction}_{abs(delta)}>"

        output = _YEAR_PATTERN.sub(replace_year, output)
        counts = RedactionCounts(
            entity_count=entity_count,
            policy_count=policy_count,
            year_count=year_count,
        )
        receipt = DeidentificationReceipt(
            input_sha256=self.policy.input_commitment(raw_text),
            output_sha256=_sha256_text(output),
            policy_sha256=self.policy.sha256,
            counts=counts,
        )
        sanitized = SanitizedText(text=output, receipt=receipt)
        self.assert_safe(sanitized)
        return sanitized

    def assert_safe(self, value: str | SanitizedText) -> None:
        """Fail closed if configured terms or absolute years remain."""
        if isinstance(value, SanitizedText):
            if value.receipt.policy_sha256 != self.policy.sha256:
                raise UnsafeTextError(
                    "sanitized text was produced under a different policy"
                )
            text = value.text
        elif isinstance(value, str):
            text = value
        else:
            raise UnsafeTextError("value must be text or SanitizedText")
        if "\x00" in text:
            raise UnsafeTextError("text contains a NUL byte")

        marker_free = _SAFE_MARKER_PATTERN.sub("", text)
        if self._pattern is not None and self._pattern.search(marker_free) is not None:
            raise UnsafeTextError("text contains a configured sensitive term")
        if _YEAR_PATTERN.search(marker_free) is not None:
            raise UnsafeTextError("text contains an absolute year")


def assert_safe(value: str | SanitizedText, policy: DeidentificationPolicy) -> None:
    """Validate sanitized output against an immutable deidentification policy."""
    Anonymizer(policy).assert_safe(value)


__all__ = [
    "DEIDENTIFICATION_SCHEMA_VERSION",
    "ENTITY_MARKER",
    "POLICY_MARKER",
    "YEAR_MARKER",
    "Anonymizer",
    "DeidentificationError",
    "DeidentificationPolicy",
    "DeidentificationReceipt",
    "RedactionCounts",
    "RedactionKind",
    "SanitizedText",
    "UnsafeTextError",
    "YearMode",
    "assert_safe",
]
