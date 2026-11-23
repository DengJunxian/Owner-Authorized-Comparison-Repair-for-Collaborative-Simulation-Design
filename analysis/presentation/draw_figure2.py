"""Redraw the sealed cost comparison as grouped bars and median/range marks."""

import json
import sys
from pathlib import Path

PAPER = Path(__file__).resolve().parent
ROOT = PAPER.parents[1]
sys.path.insert(0, str(ROOT / "ccfa-workfiles/figures/revision-workload"))
import plot as original  # noqa: E402
from matplotlib.legend_handler import HandlerTuple  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

# Edge, bar fill, hatch, timing marker, timing marker fill.
STYLES = (
    ("#2a3036", "#424850", "", "o", "#2a3036"),
    ("#56616e", "#e4e8ec", "//", "s", "white"),
    ("#2d5e88", "#b6cde4", "xx", "D", "#2d5e88"),
    ("#8b5115", "#f3dba7", "..", "^", "#8b5115"),
    ("#526477", None, "", "x", "white"),
)


def main():
    rows = json.loads(original.SOURCE.read_text(encoding="utf-8"))
    data = original.aggregate(rows)
    original_data = original.DEST / "revision_workload_plot_data.json"
    assert data == json.loads(original_data.read_text(encoding="utf-8"))
    theme, _ = original.load_theme()
    original.font_manager.fontManager.addfont(
        "/System/Library/Fonts/Supplemental/Times New Roman.ttf"
    )
    original.plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "font.size": 8,
            "axes.labelsize": 8.5,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
            "svg.hashsalt": "cscwd-visual-cost",
            "hatch.linewidth": 0.55,
        }
    )
    fig, (calls_ax, time_ax) = original.plt.subplots(1, 2, figsize=(7, 2.35))
    fig.subplots_adjust(left=0.078, right=0.989, bottom=0.205, top=0.759, wspace=0.29)
    points = {(p["method"], p["domain_size"]): p for p in data["points"]}
    sizes = sorted({size for _, size in points})
    assert len(points) == len(sizes) * len(original.METHODS)
    x = range(len(sizes))
    handles = []
    for i, (method, style) in enumerate(zip(original.METHODS, STYLES, strict=True)):
        edge, fill, hatch, marker, marker_fill = style
        series = [points[method, size] for size in sizes]
        calls = [p["search_calls_per_context"] for p in series]
        if fill is not None:
            bars = calls_ax.bar(
                [j + (i - 1.5) * 0.17 for j in x],
                calls,
                0.151,
                color=fill,
                edgecolor=edge,
                hatch=hatch,
                linewidth=0.65,
            )
            assert [bar.get_height() for bar in bars] == calls
        else:
            assert method == "complete_sql" and all(c is None for c in calls)
        medians = [p["median_wall_ms"] for p in series]
        markerline, _, _ = time_ax.errorbar(
            [j + (i - 2) * 0.155 for j in x],
            medians,
            yerr=[
                [m - p["min_wall_ms"] for m, p in zip(medians, series, strict=True)],
                [p["max_wall_ms"] - m for m, p in zip(medians, series, strict=True)],
            ],
            color=edge,
            marker=marker,
            markerfacecolor=marker_fill,
            markeredgewidth=0.85,
            markersize=4.6,
            linestyle="none",
            elinewidth=0.8,
            capsize=2,
            capthick=0.75,
        )
        handles.append(
            (
                Patch(facecolor=fill, edgecolor=edge, hatch=hatch, linewidth=0.6),
                markerline,
            )
            if fill is not None
            else markerline
        )

    for ax, title, ylabel in (
        (calls_ax, "(a) Policy evaluations", "Policy calls / revision"),
        (time_ax, "(b) Complete planning cost", "Wall time / revision (ms)"),
    ):
        original.axis_style(ax, theme)
        ax.set_title(title, fontsize=9.2, pad=6)
        ax.set_ylabel(ylabel)
        ax.set_xticks(x, [f"{size:,}" for size in sizes])
        ax.set_xlim(-0.51, len(sizes) - 0.49)
    calls_ax.text(
        0.031,
        0.96,
        "SQL calls unavailable",
        transform=calls_ax.transAxes,
        fontsize=7.5,
        va="top",
        color="#526477",
    )
    domain_label = fig.supxlabel("Targets per domain", y=0.043, fontsize=8.5)
    legend = fig.legend(
        handles,
        original.LABELS,
        loc="upper center",
        ncols=5,
        bbox_to_anchor=(0.5, 0.979),
        frameon=False,
        fontsize=7.8,
        handlelength=2.8,
        columnspacing=1.26,
        handletextpad=0.55,
        handler_map={tuple: HandlerTuple(ndivide=None, pad=0.25)},
    )
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    bbox = legend.get_window_extent(renderer)
    pad = 6 * fig.dpi / 72
    assert bbox.x0 >= pad and bbox.x1 <= fig.bbox.x1 - pad
    assert bbox.y1 <= fig.bbox.y1 - pad
    assert all(bbox.y0 > ax.title.get_window_extent(renderer).y1 for ax in fig.axes)
    for axis in fig.axes:
        bounds = axis.get_tightbbox(renderer)
        assert bounds.x0 > 0 and bounds.x1 < fig.bbox.x1
        assert bounds.y0 > domain_label.get_window_extent(renderer).y1
    assert [label.get_text() for label in legend.get_texts()] == list(original.LABELS)
    assert all(axis.get_ylim()[0] == 0 for axis in fig.axes)
    assert len(calls_ax.patches) == 4 * len(sizes)
    assert len(time_ax.containers) == 5
    stem = PAPER / "generated/revision_workload_cost_visual"
    fig.savefig(stem.with_suffix(".pdf"), metadata={"CreationDate": None})
    fig.savefig(stem.with_suffix(".svg"), metadata={"Date": None})
    fig.savefig(stem.with_suffix(".png"), dpi=250)
    print(f"Legend top clearance: {(fig.bbox.y1 - bbox.y1) * 72 / fig.dpi:.2f} pt")


if __name__ == "__main__":
    main()
