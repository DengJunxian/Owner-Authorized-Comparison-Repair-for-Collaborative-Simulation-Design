"""Fixed real-process attempt execution, with permanent failure records."""

from __future__ import annotations

import argparse
import json

from scripts.cscwd2027_design_contract.freeze import DEST, check
from scripts.cscwd2027_design_contract.protection import digest, verify
from scripts.cscwd2027_design_contract.store import RESULTS, execute


def main() -> None:
    """Run a disjoint parent shard; never retry a failed or interrupted attempt."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard", type=int, choices=(0, 1), required=True)
    args = parser.parse_args()
    check()
    verify()
    rows = json.loads((DEST / "attempts.json").read_text())
    blocks = json.loads((DEST / "sessions.json").read_text())
    own = {s["id"] for s in blocks if s["index"] % 2 == args.shard}
    for row in rows:
        if row["session"] not in own:
            continue
        folder = RESULTS / "objects" / row["id"]
        if (folder / "result.json").is_file():
            # Existing failed records are retained and NOT re-executed.
            continue
        if folder.exists():
            raise RuntimeError(
                f"interrupted attempt retained, needs recovery decision: {row['id']}"
            )
        check()
        result = execute(row, digest(DEST / "freeze.json"))
        print(
            json.dumps(
                {
                    "id": row["id"],
                    "status": result["status"],
                    "local_pass": result["local_pass"],
                    "seconds": result["elapsed_seconds"],
                }
            ),
            flush=True,
        )
    check()
    verify()
    print(json.dumps({"shard": args.shard, "status": "COMPLETED"}), flush=True)


if __name__ == "__main__":
    main()
