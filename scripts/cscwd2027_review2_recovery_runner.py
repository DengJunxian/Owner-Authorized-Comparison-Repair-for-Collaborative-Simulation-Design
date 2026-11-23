"""Read-only reconstruction adapter for an already populated output directory."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from unittest.mock import patch

from scripts import cscwd2027_review2_recovery as recovery

PARENT = recovery.parent
SEAL = PARENT.ROOT / "configs/cscwd2027/review2_reconstruction_adapter.json"


def allow_existing_root(original: Any, path: Path, *args: Any, **kwargs: Any) -> Any:
    """Bypass only creation of the existing, already sealed artifact root."""
    if path == PARENT.OUT / "artifacts":
        assert path.is_dir()
        return None
    return original(path, *args, **kwargs)


def main(action: str) -> None:
    """Seal and execute an adapter without touching either frozen program."""
    if action == "freeze":
        paths = [
            Path(__file__).resolve(),
            recovery.SEAL,
            PARENT.ROOT / "tests/test_cscwd_reconstruction_adapter.py",
        ]
        PARENT.save(
            SEAL,
            {
                "created_at_utc": datetime.now(UTC).isoformat(),
                "reason": "Recovery must verify an existing directory, not recreate it",
                "scope": "existing root mkdir only; "
                "all other outputs are compared read-only",
                "members": {
                    str(p.relative_to(PARENT.ROOT)): PARENT.sha(p) for p in paths
                },
            },
        )
        print("RECONSTRUCTION_ADAPTER_SEALED", PARENT.sha(SEAL))
        return
    seal = PARENT.read(SEAL)
    for path, expected in seal["members"].items():
        assert PARENT.sha(PARENT.ROOT / path) == expected
    original = Path.mkdir

    def mkdir(path: Path, *args: Any, **kwargs: Any) -> Any:
        return allow_existing_root(original, path, *args, **kwargs)

    with patch.object(Path, "mkdir", mkdir):
        recovery.run()
    PARENT.save(
        PARENT.OUT / "artifacts/reconstruction_adapter_receipt.json",
        {
            "adapter_sha256": PARENT.sha(SEAL),
            "frozen_sources_modified": False,
            "failures_retained": [
                "summary serialization with integer histogram keys",
                "recovery directory already existed",
            ],
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("freeze", "run"))
    main(parser.parse_args().action)
