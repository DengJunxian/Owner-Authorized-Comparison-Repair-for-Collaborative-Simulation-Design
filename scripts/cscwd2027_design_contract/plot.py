"""Predeclared grayscale count figure from complete structured study artifacts.

Use the configured bundled plotting Python; the simulation environment and its
lock are unchanged. Plotting has no simulation calls or external access.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib  # type: ignore[import-not-found]

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # type: ignore[import-not-found]  # noqa: E402
import numpy as np  # type: ignore[import-not-found]  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]


def digest(path: Path) -> str:
    """Hash inputs and vector outputs exactly."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main() -> None:
    """Display all declared collectors and all unsafe/relabel challenge cases."""
    freeze_path = ROOT / "configs/cscwd2027/design_contract_v1/freeze.json"
    freeze = json.loads(freeze_path.read_text())
    if freeze["members"][str(Path(__file__).relative_to(ROOT))] != digest(
        Path(__file__)
    ):
        raise RuntimeError("frozen figure specification changed")
    source = ROOT / "results/cscwd2027/design_contract_v1/derived/summary.json"
    data: dict[str, Any] = json.loads(source.read_text())
    if data["missing_attempts"] or data["integrity_findings"]:
        raise RuntimeError("incomplete or unbound figure data")
    modes = [
        "local_full_only",
        "spec_grouped",
        "strict_plan_lock",
        "comparison_contract",
        "conventional_relational",
        "minus_scientific_binding",
        "minus_slot_coverage",
    ]
    short = [
        "Local",
        "Grouped",
        "Plan lock",
        "Contract",
        "Relational",
        "-Science",
        "-Coverage",
    ]
    cases = [
        "label_cycle",
        "foreign_session",
        "missing_slot",
        "duplicate_slot",
        "stale_revision",
    ]
    names = [
        "Label cycle",
        "Foreign session",
        "Missing slot",
        "Duplicate slot",
        "Stale revision",
    ]
    raw = np.array(
        [
            [
                data["case_statistics"][case][mode]["unauthorized_release"]
                for mode in modes
            ]
            for case in cases
        ]
    )
    den = np.array(
        [
            [data["case_statistics"][case][mode]["sets"] for mode in modes]
            for case in cases
        ]
    )
    with plt.rc_context(
        {
            "font.size": 8,
            "font.family": "DejaVu Sans",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "axes.linewidth": 0.6,
        }
    ):
        fig, axes = plt.subplots(
            1,
            2,
            figsize=(7.05, 2.5),
            gridspec_kw={"width_ratios": [2.45, 1]},
            layout="constrained",
        )
        matrix, bars = axes
        matrix.imshow(
            raw / den,
            vmin=0,
            vmax=1,
            cmap="Greys",
            aspect="auto",
            interpolation="nearest",
        )
        matrix.set_xticks(range(len(short)), short, rotation=30, ha="right")
        matrix.set_yticks(range(len(names)), names)
        matrix.tick_params(length=0, pad=3)
        for i in range(raw.shape[0]):
            for j in range(raw.shape[1]):
                matrix.text(
                    j,
                    i,
                    f"{raw[i, j]}/{den[i, j]}",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="white" if raw[i, j] / den[i, j] > 0.5 else "black",
                )
        for border in matrix.spines.values():
            border.set_visible(False)
        matrix.set_title("(a) Unauthorized numerical panels", loc="left", fontsize=9)
        values = [
            data["mode_statistics"][mode]["valid_control_release"] for mode in modes
        ]
        denominators = [
            data["mode_statistics"][mode]["evaluable_controls"] for mode in modes
        ]
        bars.barh(np.arange(len(modes)), values, color=".35", height=0.62)
        bars.set_yticks(np.arange(len(modes)), short)
        bars.invert_yaxis()
        upper = max(denominators, default=1)
        bars.set_xlim(0, max(1, upper) * 1.26)
        for i, (n, total) in enumerate(zip(values, denominators, strict=True)):
            bars.text(
                n + max(1, upper) * 0.025, i, f"{n}/{total}", va="center", fontsize=7
            )
        bars.set_xlabel("Valid controls accepted")
        bars.set_title("(b) Retained comparisons", loc="left", fontsize=9)
        bars.spines[["top", "right"]].set_visible(False)
        for axis in axes:
            axis.set_facecolor("white")
        destination = ROOT / "paper/cscwd2027_design_contract/generated"
        destination.mkdir(parents=True, exist_ok=True)
        outputs = []
        for extension in ("pdf", "svg"):
            path = destination / f"comparison_handoff.{extension}"
            fig.savefig(path, facecolor="white")
            outputs.append(path)
        plt.close(fig)
    (destination / "figure_manifest_design.json").write_text(
        json.dumps(
            {
                "input": str(source.relative_to(ROOT)),
                "input_sha256": digest(source),
                "plot_source_sha256": digest(Path(__file__)),
                "outputs": {str(p.relative_to(ROOT)): digest(p) for p in outputs},
                "dimensions_inches": [7.05, 2.5],
                "encoding": (
                    "raw counts; grayscale mapped to fraction of declared cases"
                ),
                "uncertainty": "finite designed case counts, no population fault CI",
            },
            indent=2,
        )
        + "\n"
    )
    print(json.dumps({"outputs": len(outputs), "source_sha256": digest(source)}))


if __name__ == "__main__":
    main()
