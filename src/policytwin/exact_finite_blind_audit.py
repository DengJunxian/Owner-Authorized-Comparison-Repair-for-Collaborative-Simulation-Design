"""Source-only process entry for the bounded exact-finite scenario audit.

The entrypoint installs a narrow importer before any :mod:`policytwin` module
is imported.  That importer compiles the exact ``.py`` bytes it records and
never consults ``__pycache__``.  The evaluator later compares this loaded-byte
manifest with its fd-checked source-tree commitment.  This is a local workflow
guard, not an external execution attestation or protection from a malicious
workspace owner.
"""

from __future__ import annotations

import hashlib
import importlib.abc
import importlib.util
import os
import stat
import sys
from collections.abc import Sequence
from pathlib import Path
from types import ModuleType
from typing import Any

_MAX_SOURCE_BYTES = 2_097_152


def _read_stable_source(path: Path) -> bytes:
    """Read one regular source file through a no-follow descriptor."""
    nofollow = getattr(os, "O_NOFOLLOW", None)
    if not isinstance(nofollow, int):
        raise RuntimeError("source-only blind audit requires O_NOFOLLOW")
    try:
        descriptor = os.open(path, os.O_RDONLY | nofollow)
    except OSError as error:
        raise RuntimeError("cannot open source-only audit module") from error
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_size > _MAX_SOURCE_BYTES:
            raise RuntimeError("source-only audit module is invalid or oversized")
        chunks: list[bytes] = []
        remaining = before.st_size
        while remaining:
            chunk = os.read(descriptor, min(remaining, 131_072))
            if not chunk:
                raise RuntimeError("source-only audit module was truncated")
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1):
            raise RuntimeError("source-only audit module grew while read")
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    identity_before = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    )
    identity_after = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    )
    if identity_before != identity_after:
        raise RuntimeError("source-only audit module changed while read")
    return b"".join(chunks)


class _SourceOnlyPolicyTwinLoader(importlib.abc.Loader):
    """Compile recorded source bytes without reading bytecode caches."""

    def __init__(
        self,
        *,
        fullname: str,
        path: Path,
        owner: _SourceOnlyPolicyTwinFinder,
    ) -> None:
        self._fullname = fullname
        self._path = path
        self._owner = owner

    def create_module(self, spec: Any) -> ModuleType | None:
        del spec
        return None

    def exec_module(self, module: ModuleType) -> None:
        source = _read_stable_source(self._path)
        self._owner._record_loaded_source(self._fullname, self._path, source)
        module.__dict__["__cached__"] = None
        code = compile(
            source,
            str(self._path),
            "exec",
            dont_inherit=True,
            optimize=sys.flags.optimize,
        )
        exec(code, module.__dict__)


class _SourceOnlyPolicyTwinFinder(importlib.abc.MetaPathFinder):
    """Resolve every PolicyTwin package member to an explicit source file."""

    policytwin_source_only_blind_loader = True

    def __init__(self, source_root: Path) -> None:
        self.source_root = source_root.resolve(strict=True)
        self._loaded: dict[str, dict[str, object]] = {}

    def find_spec(
        self,
        fullname: str,
        path: Sequence[str] | None,
        target: ModuleType | None = None,
    ) -> Any:
        del path
        if target is not None:
            raise ImportError("source-only blind audit forbids module reload")
        if fullname == "policytwin":
            parts: tuple[str, ...] = ()
        elif fullname.startswith("policytwin."):
            parts = tuple(fullname.split(".")[1:])
        else:
            return None
        package_path = self.source_root.joinpath("policytwin", *parts, "__init__.py")
        module_path = self.source_root.joinpath("policytwin", *parts).with_suffix(".py")
        if package_path.is_file():
            selected = package_path
            locations: list[str] | None = [str(package_path.parent)]
        elif module_path.is_file():
            selected = module_path
            locations = None
        else:
            raise ImportError(
                f"source-only blind audit found no .py source for {fullname}"
            )
        resolved = selected.resolve(strict=True)
        try:
            resolved.relative_to(self.source_root)
        except ValueError as error:
            raise ImportError(
                "PolicyTwin source resolved outside source root"
            ) from error
        loader = _SourceOnlyPolicyTwinLoader(
            fullname=fullname,
            path=resolved,
            owner=self,
        )
        return importlib.util.spec_from_file_location(
            fullname,
            resolved,
            loader=loader,
            submodule_search_locations=locations,
        )

    def _record_loaded_source(
        self,
        fullname: str,
        path: Path,
        source: bytes,
    ) -> None:
        relative = path.relative_to(self.source_root.parent).as_posix()
        record: dict[str, object] = {
            "module": fullname,
            "path": relative,
            "sha256": hashlib.sha256(source).hexdigest(),
            "size_bytes": len(source),
        }
        previous = self._loaded.get(fullname)
        if previous is not None and previous != record:
            raise RuntimeError("PolicyTwin module source changed across imports")
        self._loaded[fullname] = record

    def loaded_source_manifest(self) -> tuple[dict[str, object], ...]:
        return tuple(dict(self._loaded[name]) for name in sorted(self._loaded))


def _canonical_direct_invocation() -> bool:
    script = Path(__file__).resolve()
    main = sys.modules.get("__main__")
    main_file = getattr(main, "__file__", None)
    if not isinstance(main_file, str) or Path(main_file).resolve() != script:
        return False
    if len(sys.argv) != 1 or Path(sys.argv[0]).resolve() != script:
        return False
    original = tuple(sys.orig_argv)
    return bool(
        len(original) == 4
        and original[1:3] == ("-I", "-S")
        and Path(original[3]).resolve() == script
    )


def _install_source_only_policytwin_importer() -> None:
    if any(
        name == "policytwin" or name.startswith("policytwin.") for name in sys.modules
    ):
        raise RuntimeError("PolicyTwin was imported before source-only bootstrap")
    source_root = Path(__file__).resolve().parents[1]
    finder = _SourceOnlyPolicyTwinFinder(source_root)
    sys.dont_write_bytecode = True
    sys.meta_path.insert(0, finder)


def _prepare_isolated_direct_script_import() -> None:
    if __package__ not in {None, ""}:
        return
    source_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(source_root))


def _isolated_flags_are_valid() -> bool:
    flags = sys.flags
    return bool(
        flags.isolated == 1
        and flags.no_site == 1
        and flags.ignore_environment == 1
        and flags.safe_path is True
        and not ({"site", "sitecustomize", "usercustomize"} & set(sys.modules))
    )


def main(argv: tuple[str, ...] | None = None) -> int:
    """Run the fixed s--x audit with no runtime candidate arguments."""
    arguments = tuple(sys.argv[1:]) if argv is None else argv
    if (
        arguments
        or not _isolated_flags_are_valid()
        or not _canonical_direct_invocation()
    ):
        sys.stderr.write(
            "usage: .venv/bin/python -I -S src/policytwin/exact_finite_blind_audit.py\n"
        )
        return 2
    _prepare_isolated_direct_script_import()
    _install_source_only_policytwin_importer()
    from policytwin.evaluation.policy_functional_exact_finite import (
        execute_exact_finite_scenario_blind_audit_and_finalize,
    )

    execute_exact_finite_scenario_blind_audit_and_finalize()
    return 0


if __name__ == "__main__":  # pragma: no cover - covered by subprocess test
    raise SystemExit(main())


__all__ = ["main"]
