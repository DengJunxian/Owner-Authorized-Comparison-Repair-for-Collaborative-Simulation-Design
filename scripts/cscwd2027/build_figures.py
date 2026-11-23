"""Render artifact-derived IEEE-width figures; no simulator dependency required."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def main() -> None:
    """Export vector figures, table source, and their input/renderer manifest."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    root = Path(__file__).resolve().parents[2]
    source = root / "results/cscwd2027/derived_v1/summary.json"
    summary = json.loads(source.read_text())
    if not summary["data_complete"]:
        raise RuntimeError("refuse to render a complete-study figure from partial data")
    destination = root / "paper/cscwd2027/generated"
    destination.mkdir(parents=True, exist_ok=True)
    names = [
        "task_duplication",
        "task_omission",
        "wrong_owner",
        "duplicate_effect",
        "missing_message",
        "logical_order",
        "payload_mutation",
        "stale_plan",
    ]
    labels = [
        "Task duplicate",
        "Task missing",
        "Wrong owner",
        "Effect duplicate",
        "Message missing",
        "Commit order",
        "Payload",
        "Stale plan",
    ]
    modes = [
        "keys_only",
        "receipts",
        "coordination",
        "full",
        "conventional",
        "minus_replay",
        "minus_gate",
    ]
    mode_labels = ["Keys", "+Events", "+Coord.", "Full", "Conv.", "-Replay", "-Gate"]
    matrix = [
        [
            summary["faults"][name]["modes"][mode]["invalid_release"]["count"]
            for mode in modes
        ]
        for name in names
    ]
    outputs = []
    style = {
        "font.size": 8,
        "font.family": "DejaVu Sans",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
    with plt.rc_context(style):
        fig, ax = plt.subplots(figsize=(7.05, 2.4), layout="constrained")
        ax.imshow(
            matrix,
            cmap="Greys",
            vmin=0,
            vmax=32,
            aspect="auto",
            interpolation="nearest",
        )
        ax.set_xticks(range(len(modes)), mode_labels)
        ax.set_yticks(range(len(names)), labels)
        for i, name in enumerate(names):
            for j, mode in enumerate(modes):
                cell = summary["faults"][name]["modes"][mode]["invalid_release"]
                ax.text(
                    j,
                    i,
                    f"{cell['count']}/{cell['denominator']}",
                    ha="center",
                    va="center",
                    color="white" if cell["count"] > 16 else "black",
                    fontsize=8,
                )
        ax.set_xlabel(
            "Released invalid numerical artifacts / triggered violations "
            "(lower is better)"
        )
        for suffix in ("pdf", "svg", "png"):
            target = destination / f"invalid_release.{suffix}"
            fig.savefig(target, dpi=300, facecolor="white", transparent=False)
            outputs.append(target)
        plt.close(fig)
        fig, axes = plt.subplots(1, 2, figsize=(7.05, 2.45), layout="constrained")
        plotted = [
            ("no_evidence", "No evidence", "#000000", "o", "-"),
            ("keys_only", "Keys", "#0072B2", "s", "--"),
            ("receipts", "Event receipts", "#009E73", "^", ":"),
            ("full", "Full", "#D55E00", "D", "-"),
            ("conventional", "Conventional", "#7A4E9B", "x", "-."),
        ]
        for ax, agents in zip(axes, (6, 24), strict=True):
            for mode, label, color, marker, line in plotted:
                cells = [
                    next(
                        c
                        for c in summary["cost"]
                        if c["agents"] == agents
                        and c["workers"] == workers
                        and c["mode"] == mode
                    )
                    for workers in (1, 2, 4)
                ]
                values = [c["end_to_end_seconds"]["median"] for c in cells]
                low = [
                    v - c["end_to_end_seconds"]["q25"]
                    for c, v in zip(cells, values, strict=True)
                ]
                high = [
                    c["end_to_end_seconds"]["q75"] - v
                    for c, v in zip(cells, values, strict=True)
                ]
                ax.errorbar(
                    [1, 2, 4],
                    values,
                    yerr=[low, high],
                    color=color,
                    marker=marker,
                    linestyle=line,
                    linewidth=1.1,
                    markersize=4,
                    capsize=2,
                    label=label,
                )
            ax.set_title(f"{agents} agents")
            ax.set(
                xlabel="Worker processes",
                ylabel="Cold pair latency (s)",
                xticks=[1, 2, 4],
                ylim=(0, None),
            )
            ax.grid(axis="y", color="0.9", linewidth=0.5)
        handles, legend_labels = axes[0].get_legend_handles_labels()
        fig.legend(
            handles,
            legend_labels,
            loc="outside lower center",
            ncol=5,
            frameon=False,
            fontsize=7,
        )
        for suffix in ("pdf", "svg", "png"):
            target = destination / f"cost.{suffix}"
            fig.savefig(target, dpi=300, facecolor="white", transparent=False)
            outputs.append(target)
        plt.close(fig)
    table = [
        "% Generated counts; no manually entered results.",
        "\\begin{tabular}{lrrrr}\\toprule",
        "Fault & Changed & Same & Incomplete & Full release\\\\\\midrule",
    ]
    for name, label in zip(names, labels, strict=True):
        cell = summary["faults"][name]
        released = cell["modes"]["full"]["invalid_release"]
        table.append(
            f"{label} & {cell['changed']} & {cell['unchanged']} & "
            f"{cell['incomplete']} & {released['count']}/{released['denominator']}\\\\"
        )
    table.append("\\bottomrule\\end{tabular}")
    target = destination / "fault_outcomes.tex"
    target.write_text("\n".join(table) + "\n")
    outputs.append(target)
    manifest = {
        "input": str(source.relative_to(root)),
        "input_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "renderer": {"matplotlib": matplotlib.__version__},
        "cost_display": (
            "median and interquartile range; 20 measured paired blocks per cell"
        ),
        "release_display": "raw counts; fixed 0..32 grayscale and printed denominator",
        "outputs": {
            str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in outputs
        },
    }
    (destination / "figure_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
