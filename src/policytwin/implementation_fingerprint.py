"""Deterministic fingerprints for live PolicyTwin Python implementations.

Disk source hashes do not detect an accidental in-process replacement of a
function or method.  This module supplements source commitments with a stable
description of the Python bytecode, defaults, closures, and class methods that
are currently bound at a verifier boundary.  It is a drift detector, not a
cryptographic attestation against a malicious process owner.
"""

from __future__ import annotations

import os
import re
import stat
import sys
from collections.abc import Mapping
from dataclasses import MISSING, fields, is_dataclass
from datetime import date, datetime, time, tzinfo
from enum import Enum
from pathlib import Path
from types import (
    BuiltinFunctionType,
    BuiltinMethodType,
    CodeType,
    FunctionType,
    ModuleType,
)
from typing import NoReturn

from policytwin.provenance import canonical_json_bytes, sha256_bytes


class ImplementationFingerprintError(RuntimeError):
    """Raised when a live implementation cannot be committed fail closed."""


def _fail(message: str) -> NoReturn:
    raise ImplementationFingerprintError(message)


def _stable_file_bytes(path: Path, *, label: str) -> bytes:
    try:
        before = path.stat()
        content = path.read_bytes()
        after = path.stat()
    except OSError as error:
        raise ImplementationFingerprintError(f"cannot read {label}") from error
    before_identity = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    )
    after_identity = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    )
    if (
        not stat.S_ISREG(before.st_mode)
        or before.st_size != len(content)
        or before_identity != after_identity
    ):
        _fail(f"{label} changed while it was fingerprinted")
    return content


def _code_constant_material(value: object) -> object:
    """Canonicalize Python compiler constants without adaptive VM state."""
    if value is None or value is Ellipsis or isinstance(value, bool | int | str):
        return {
            "constant_type": type(value).__name__,
            "value": "Ellipsis" if value is Ellipsis else value,
        }
    if isinstance(value, float):
        return {"constant_type": "float", "value_hex": value.hex()}
    if isinstance(value, complex):
        return {
            "constant_type": "complex",
            "imag_hex": value.imag.hex(),
            "real_hex": value.real.hex(),
        }
    if isinstance(value, bytes):
        return {
            "constant_type": "bytes",
            "sha256": sha256_bytes(value),
            "size_bytes": len(value),
        }
    if isinstance(value, CodeType):
        return {"constant_type": "code", "material": _code_material(value)}
    if isinstance(value, tuple):
        return {
            "constant_type": "tuple",
            "values": [_code_constant_material(item) for item in value],
        }
    if isinstance(value, frozenset):
        members = [_code_constant_material(item) for item in value]
        return {
            "constant_type": "frozenset",
            "values": sorted(members, key=canonical_json_bytes),
        }
    _fail(
        "unsupported compiler constant in implementation fingerprint: "
        f"{type(value).__module__}.{type(value).__qualname__}"
    )


def _code_material(code: CodeType) -> dict[str, object]:
    """Describe a code object using stable public fields only.

    ``marshal.dumps(code)`` is unsuitable here: CPython 3.11+ can serialize
    interpreter quickening/specialization state after a function executes,
    even while the public ``co_code`` bytes remain unchanged.  That made an
    implementation commitment depend on test order.  The explicit fields
    below bind executable bytecode, nested code, names, flags, and exception
    tables while intentionally excluding filenames and source line offsets.
    """
    return {
        "argcount": code.co_argcount,
        "bytecode_sha256": sha256_bytes(code.co_code),
        "cellvars": list(code.co_cellvars),
        "constants": [_code_constant_material(item) for item in code.co_consts],
        "exceptiontable_sha256": sha256_bytes(code.co_exceptiontable),
        "flags": code.co_flags,
        "freevars": list(code.co_freevars),
        "kwonlyargcount": code.co_kwonlyargcount,
        "name": code.co_name,
        "names": list(code.co_names),
        "nlocals": code.co_nlocals,
        "posonlyargcount": code.co_posonlyargcount,
        "qualname": code.co_qualname,
        "stacksize": code.co_stacksize,
        "varnames": list(code.co_varnames),
    }


def _code_sha256(code: CodeType) -> str:
    return sha256_bytes(canonical_json_bytes(_code_material(code)))


def _stable_runtime_value(
    value: object,
    *,
    active: frozenset[int] = frozenset(),
) -> object:
    """Return canonical material without process-specific object addresses."""
    if value is None or isinstance(value, bool | int | str):
        return value
    if isinstance(value, float):
        return {"float_hex": value.hex()}
    if isinstance(value, bytes):
        return {"bytes_sha256": sha256_bytes(value), "size_bytes": len(value)}
    if isinstance(value, Enum):
        return {
            "enum_type": f"{type(value).__module__}.{type(value).__qualname__}",
            "value": _stable_runtime_value(value.value, active=active),
        }
    if isinstance(value, CodeType):
        return {"code_sha256": _code_sha256(value)}
    if isinstance(value, FunctionType):
        identity = id(value)
        if identity in active:
            return {"recursive_function": f"{value.__module__}.{value.__qualname__}"}
        return _function_material(value, active=active | {identity})
    if isinstance(value, BuiltinFunctionType | BuiltinMethodType):
        return {
            "builtin_callable": (
                f"{getattr(value, '__module__', None)}."
                f"{getattr(value, '__qualname__', getattr(value, '__name__', None))}"
            )
        }
    if isinstance(value, type):
        return {"type": f"{value.__module__}.{value.__qualname__}"}
    if isinstance(value, tuple | list):
        identity = id(value)
        if identity in active:
            _fail("recursive sequence cannot be fingerprinted")
        return {
            "sequence_type": type(value).__name__,
            "values": [
                _stable_runtime_value(item, active=active | {identity})
                for item in value
            ],
        }
    if isinstance(value, set | frozenset):
        identity = id(value)
        if identity in active:
            _fail("recursive set cannot be fingerprinted")
        members = [
            _stable_runtime_value(item, active=active | {identity}) for item in value
        ]
        return {
            "set_type": type(value).__name__,
            "values": sorted(members, key=canonical_json_bytes),
        }
    if isinstance(value, Mapping):
        identity = id(value)
        if identity in active:
            _fail("recursive mapping cannot be fingerprinted")
        entries = [
            {
                "key": _stable_runtime_value(key, active=active | {identity}),
                "value": _stable_runtime_value(item, active=active | {identity}),
            }
            for key, item in value.items()
        ]
        return {
            "mapping_type": type(value).__name__,
            "entries": sorted(entries, key=canonical_json_bytes),
        }
    if isinstance(value, os.PathLike):
        return {"path_like": os.fspath(value)}
    if isinstance(value, re.Pattern):
        return {"regular_expression": value.pattern, "flags": value.flags}
    if isinstance(value, datetime | date | time):
        return {
            "temporal_type": f"{type(value).__module__}.{type(value).__qualname__}",
            "iso8601": value.isoformat(),
        }
    if isinstance(value, tzinfo):
        offset = value.utcoffset(None)
        return {
            "timezone_type": f"{type(value).__module__}.{type(value).__qualname__}",
            "name": value.tzname(None),
            "offset_seconds": None if offset is None else offset.total_seconds(),
        }
    if is_dataclass(value) and not isinstance(value, type):
        identity = id(value)
        if identity in active:
            _fail("recursive dataclass value cannot be fingerprinted")
        return {
            "dataclass_type": f"{type(value).__module__}.{type(value).__qualname__}",
            "fields": {
                field.name: _stable_runtime_value(
                    getattr(value, field.name), active=active | {identity}
                )
                for field in fields(value)
            },
        }
    representation = repr(value)
    if type(value).__module__ == "dataclasses":
        return {
            "dataclasses_sentinel": (
                f"{type(value).__module__}.{type(value).__qualname__}"
            ),
            "stable_representation": (
                representation
                if "0x" not in representation and len(representation) <= 128
                else None
            ),
        }
    _fail(
        "unsupported live value in implementation fingerprint: "
        f"{type(value).__module__}.{type(value).__qualname__}"
    )


def _function_material(
    function: FunctionType,
    *,
    active: frozenset[int] = frozenset(),
) -> dict[str, object]:
    closure = () if function.__closure__ is None else function.__closure__
    return {
        "code_sha256": _code_sha256(function.__code__),
        "defaults": _stable_runtime_value(function.__defaults__, active=active),
        "kwdefaults": _stable_runtime_value(function.__kwdefaults__, active=active),
        "module": function.__module__,
        "qualname": function.__qualname__,
        "closure": [
            _stable_runtime_value(cell.cell_contents, active=active) for cell in closure
        ],
    }


def _class_material(value: type[object]) -> dict[str, object]:
    methods: dict[str, object] = {}
    for name, raw_method in sorted(vars(value).items()):
        functions_to_record: list[tuple[str, FunctionType]] = []
        if isinstance(raw_method, staticmethod | classmethod):
            descriptor_function = raw_method.__func__
            if isinstance(descriptor_function, FunctionType):
                functions_to_record.append((name, descriptor_function))
        elif isinstance(raw_method, FunctionType):
            functions_to_record.append((name, raw_method))
        elif isinstance(raw_method, property):
            for suffix, accessor in (
                ("fget", raw_method.fget),
                ("fset", raw_method.fset),
                ("fdel", raw_method.fdel),
            ):
                if isinstance(accessor, FunctionType):
                    functions_to_record.append((f"{name}.{suffix}", accessor))
        for method_name, function in functions_to_record:
            methods[method_name] = _function_material(function)
    dataclass_fields: list[dict[str, object]] = []
    if is_dataclass(value):
        for field in fields(value):
            default = (
                {"missing": True}
                if field.default is MISSING
                else _stable_runtime_value(field.default)
            )
            default_factory = (
                {"missing": True}
                if field.default_factory is MISSING
                else _stable_runtime_value(field.default_factory)
            )
            dataclass_fields.append(
                {
                    "compare": field.compare,
                    "default": default,
                    "default_factory": default_factory,
                    "hash": field.hash,
                    "init": field.init,
                    "kw_only": field.kw_only,
                    "name": field.name,
                    "repr": field.repr,
                }
            )
    enum_members: dict[str, object] = {}
    if issubclass(value, Enum):
        enum_members = {
            name: _stable_runtime_value(member.value)
            for name, member in value.__members__.items()
        }
    return {
        "bases": [f"{base.__module__}.{base.__qualname__}" for base in value.__bases__],
        "dataclass_fields": dataclass_fields,
        "enum_members": enum_members,
        "methods": methods,
        "module": value.__module__,
        "qualname": value.__qualname__,
    }


def _is_static_namespace_constant(value: object) -> bool:
    return (
        value is None
        or isinstance(
            value,
            bool
            | int
            | float
            | str
            | bytes
            | Enum
            | tuple
            | frozenset
            | os.PathLike
            | re.Pattern
            | datetime
            | date
            | time
            | tzinfo,
        )
        or (is_dataclass(value) and not isinstance(value, type))
    )


def live_policytwin_implementation_sha256(
    *,
    module_path: Path,
    namespace: Mapping[str, object],
) -> str:
    """Hash one module and every live Python function/class bound in its namespace."""
    target_bytes = _stable_file_bytes(
        module_path.resolve(), label="target implementation module"
    )
    helper_bytes = _stable_file_bytes(
        Path(__file__).resolve(), label="fingerprint implementation module"
    )
    functions = {
        name: (
            _function_material(value)
            if isinstance(value, FunctionType)
            else _stable_runtime_value(value)
        )
        for name, value in sorted(namespace.items())
        if isinstance(value, FunctionType | BuiltinFunctionType | BuiltinMethodType)
        and str(getattr(value, "__module__", "")).startswith("policytwin.")
    }
    classes_by_identity = {
        f"{value.__module__}.{value.__qualname__}": value
        for value in namespace.values()
        if isinstance(value, type)
        and str(getattr(value, "__module__", "")).startswith("policytwin.")
    }
    classes = {
        name: _class_material(value)
        for name, value in sorted(classes_by_identity.items())
    }
    constants = {
        name: _stable_runtime_value(value)
        for name, value in sorted(namespace.items())
        if not isinstance(
            value,
            FunctionType | BuiltinFunctionType | BuiltinMethodType | type | ModuleType,
        )
        and name.lstrip("_").isupper()
        and _is_static_namespace_constant(value)
    }
    external_callables = sorted(
        {
            (
                f"{getattr(value, '__module__', None)}."
                f"{getattr(value, '__qualname__', getattr(value, '__name__', None))}"
            )
            for value in namespace.values()
            if isinstance(
                value,
                FunctionType | BuiltinFunctionType | BuiltinMethodType | type,
            )
            and not str(getattr(value, "__module__", "")).startswith("policytwin.")
        }
    )
    return sha256_bytes(
        canonical_json_bytes(
            {
                "classes": classes,
                "constants": constants,
                "external_callables": external_callables,
                "fingerprinter_module_sha256": sha256_bytes(helper_bytes),
                "functions": functions,
                "schema_version": 2,
                "target_module_sha256": sha256_bytes(target_bytes),
            }
        )
    )


def live_policytwin_dependency_closure_sha256(
    *,
    root_modules: tuple[ModuleType, ...],
) -> str:
    """Commit the recursively discovered live PolicyTwin module dependency graph.

    Discovery follows currently bound modules, functions, classes, and instance
    types.  Every discovered module must already be loaded and backed by a local
    regular Python source file.  The result detects ordinary runtime drift in a
    transitive execution dependency; it remains a process-local drift detector,
    not an external execution attestation.
    """
    if not isinstance(root_modules, tuple) or not root_modules:
        _fail("implementation dependency closure requires root modules")
    pending = sorted(
        {
            module.__name__
            for module in root_modules
            if isinstance(module, ModuleType)
            and module.__name__.startswith("policytwin.")
        }
    )
    if len(pending) != len(root_modules):
        _fail("implementation dependency roots must be unique PolicyTwin modules")
    discovered: dict[str, str] = {}
    dependency_edges: dict[str, list[str]] = {}
    while pending:
        module_name = pending.pop(0)
        if module_name in discovered:
            continue
        module = sys.modules.get(module_name)
        if not isinstance(module, ModuleType):
            _fail(f"PolicyTwin dependency module is not loaded: {module_name}")
        module_file = getattr(module, "__file__", None)
        if not isinstance(module_file, str) or not module_file.endswith(".py"):
            _fail(f"PolicyTwin dependency lacks Python source: {module_name}")
        namespace = vars(module)
        discovered[module_name] = live_policytwin_implementation_sha256(
            module_path=Path(module_file),
            namespace=namespace,
        )
        dependencies: set[str] = set()
        for value in namespace.values():
            if isinstance(value, ModuleType):
                dependency_name = value.__name__
            elif isinstance(
                value,
                FunctionType | BuiltinFunctionType | BuiltinMethodType | type,
            ):
                dependency_name = getattr(value, "__module__", "")
            else:
                dependency_name = type(value).__module__
            if (
                isinstance(dependency_name, str)
                and dependency_name.startswith("policytwin.")
                and dependency_name != module_name
            ):
                dependencies.add(dependency_name)
        ordered_dependencies = sorted(dependencies)
        dependency_edges[module_name] = ordered_dependencies
        pending.extend(
            name
            for name in ordered_dependencies
            if name not in discovered and name not in pending
        )
        pending.sort()
    return sha256_bytes(
        canonical_json_bytes(
            {
                "dependency_edges": dependency_edges,
                "module_implementation_sha256": discovered,
                "root_modules": sorted(module.__name__ for module in root_modules),
                "schema_version": 1,
            }
        )
    )


__all__ = [
    "ImplementationFingerprintError",
    "live_policytwin_dependency_closure_sha256",
    "live_policytwin_implementation_sha256",
]
