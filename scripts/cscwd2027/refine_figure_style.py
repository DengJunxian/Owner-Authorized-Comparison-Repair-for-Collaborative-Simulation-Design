"""Post-freeze vector-only typesetting of the unchanged release-count matrix."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def main() -> None:
    """Use vector rectangles instead of an embedded bitmap; retain every count."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    root = Path(__file__).resolve().parents[2]
    data = root / "results/cscwd2027/derived_v1/summary.json"
    summary = json.loads(data.read_text())
    if not summary["data_complete"]:
        raise RuntimeError("publication figures require the complete frozen analysis")
    faults = (
        ("task_duplication", "Task duplicate"),
        ("task_omission", "Task missing"),
        ("wrong_owner", "Wrong owner"),
        ("duplicate_effect", "Effect duplicate"),
        ("missing_message", "Message missing"),
        ("logical_order", "Commit order"),
        ("payload_mutation", "Payload"),
        ("stale_plan", "Stale plan"),
    )
    modes = (
        ("keys_only", "Keys"),
        ("receipts", "+Events"),
        ("coordination", "+Coord."),
        ("full", "Full"),
        ("conventional", "Conv."),
        ("minus_replay", "-Replay"),
        ("minus_gate", "-Gate"),
    )
    destination = root / "paper/cscwd2027/generated"
    outputs = []
    with plt.rc_context(
        {
            "font.size": 8,
            "font.family": "DejaVu Sans",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    ):
        fig, ax = plt.subplots(figsize=(7.05, 2.4), layout="constrained")
        for i, (fault, _) in enumerate(faults):
            for j, (mode, _) in enumerate(modes):
                cell = summary["faults"][fault]["modes"][mode]["invalid_release"]
                fraction = cell["count"] / cell["denominator"]
                shade = 1 - fraction
                ax.add_patch(
                    Rectangle(
                        (j - 0.5, i - 0.5),
                        1,
                        1,
                        facecolor=str(shade),
                        edgecolor="0.85",
                        linewidth=0.35,
                    )
                )
                ax.text(
                    j,
                    i,
                    f"{cell['count']}/{cell['denominator']}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="white" if fraction > 0.5 else "black",
                )
        ax.set_xlim(-0.5, len(modes) - 0.5)
        ax.set_ylim(len(faults) - 0.5, -0.5)
        ax.set_xticks(range(len(modes)), [label for _, label in modes])
        ax.set_yticks(range(len(faults)), [label for _, label in faults])
        ax.tick_params(length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_xlabel("Released invalid numerical artifacts / triggered violations")
        for suffix in ("pdf", "svg", "png"):
            path = destination / f"invalid_release_publication.{suffix}"
            fig.savefig(path, dpi=300, facecolor="white")
            outputs.append(path)
        plt.close(fig)
    manifest = {
        "purpose": (
            "presentation-only vector refinement; unchanged rows, modes and counts"
        ),
        "input": str(data.relative_to(root)),
        "input_sha256": hashlib.sha256(data.read_bytes()).hexdigest(),
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "matplotlib": matplotlib.__version__,
        "mapping": "white=zero releases; black=all triggered violations released",
        "protocol_diagram_source": "paper/cscwd2027/sections/protocol.tex",
        "protocol_diagram_source_sha256": hashlib.sha256(
            (root / "paper/cscwd2027/sections/protocol.tex").read_bytes()
        ).hexdigest(),
        "outputs": {
            str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in outputs
        },
    }
    (destination / "publication_figure_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
