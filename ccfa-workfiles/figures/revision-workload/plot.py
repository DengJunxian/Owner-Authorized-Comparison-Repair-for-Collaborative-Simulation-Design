"""Artifact-only figures: complete planning cost is not predicate-call count.

The CCFA small-multiple and grouped-bar grammars are implemented in Matplotlib
for final-size IEEE typography. No simulation or selective resampling occurs.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
WORK = Path(__file__).resolve().parent
DEST = ROOT / "paper/cscwd2027_design_contract/generated"
SOURCE = ROOT / "results/cscwd2027/revision_workload_v1/timing_rows.json"
METHODS = ("exhaustive", "context_local", "persistent", "generic_memo", "complete_sql")
LABELS = (
    "Full search",
    "Context-local reuse",
    "Cross-revision reuse",
    "Annotated memo",
    "Complete SQL",
)
SHORT = ("Full", "Context", "Persist", "Memo", "SQL")
COLORS = ("#676d75", "#969ca4", "#172c51", "#30343b", "#6f7983")
MARKERS = ("o", "s", "D", "^", "x")
LINES = ("-", "--", "-", ":", "-.")
PHASE_GROUPS = {
    "Immutable dependencies": ("immutable_dependencies",),
    "Current work / rank": ("current_typed_work", "current_hard_filter_and_rank"),
    "Policy / search / approval": (
        "validation_and_policy_compile",
        "owner_search",
        "contract_and_exact_approval",
    ),
    "Serialization": ("result_serialization",),
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_json(path: Path, value) -> None:
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def load_theme():
    library = (
        Path.home()
        / ".codex/skills/ccfa-skills/.upstream/ccf-visual-composer"
        / "resources/python/ccfa_plot_recipes.py"
    )
    spec = importlib.util.spec_from_file_location("revision_ccfa_recipe", library)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.Theme(
        bg="#ffffff",
        panel="#ffffff",
        ink="#17212b",
        accent="#172c51",
        font="Times New Roman",
    ), library


def aggregate(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["domain_size"], row["method"], row["repeat"]].append(row)
    sizes = sorted({r["domain_size"] for r in rows})
    repeats = sorted({r["repeat"] for r in rows})
    assert sizes == [32, 256, 1024] and repeats == list(range(5))
    assert set(r["method"] for r in rows) == set(METHODS)
    expected_contexts = None
    points, phases = [], []
    for size in sizes:
        for method in METHODS:
            technical = []
            for repeat in repeats:
                group = grouped[size, method, repeat]
                contexts = {(r["session"], r["episode"], r["stage"]) for r in group}
                assert len(contexts) == len(group), "duplicate route measurement"
                if expected_contexts is None:
                    expected_contexts = contexts
                assert contexts == expected_contexts, "missing or selective context"
                count = len(group)
                calls = [r["counts"]["search_predicate_calls"] for r in group]
                assert (
                    all(v is None for v in calls)
                    if method == "complete_sql"
                    else all(type(v) is int for v in calls)
                )
                totals = {
                    "repeat": repeat,
                    "contexts": count,
                    "search_calls": None if method == "complete_sql" else sum(calls),
                    "wall_ms_per_context": sum(r["timing"]["wall_ns"] for r in group)
                    / count
                    / 1e6,
                    "cpu_ms_per_context": sum(r["timing"]["cpu_ns"] for r in group)
                    / count
                    / 1e6,
                    "cold_trajectory_compilations": sum(
                        r["counts"]["cold_trajectory_compilations"] for r in group
                    ),
                    "cold_evaluation_compilations": sum(
                        r["counts"]["cold_evaluation_compilations"] for r in group
                    ),
                }
                components = {
                    label: sum(
                        sum(r["timing"]["phases"][p]["wall_ns"] for p in names)
                        for r in group
                    )
                    / count
                    / 1e6
                    for label, names in PHASE_GROUPS.items()
                }
                components["Timer / control"] = totals["wall_ms_per_context"] - sum(
                    components.values()
                )
                assert components["Timer / control"] >= -1e-8
                totals["phase_ms_per_context"] = components
                technical.append(totals)
            assert len({r["search_calls"] for r in technical}) == 1, (
                "non-deterministic call count"
            )
            for key in ("cold_trajectory_compilations", "cold_evaluation_compilations"):
                assert len({r[key] for r in technical}) == 1
            wall = [r["wall_ms_per_context"] for r in technical]
            calls = technical[0]["search_calls"]
            points.append(
                {
                    "domain_size": size,
                    "method": method,
                    "contexts": technical[0]["contexts"],
                    "search_calls_per_context": None
                    if calls is None
                    else calls / technical[0]["contexts"],
                    "median_wall_ms": statistics.median(wall),
                    "min_wall_ms": min(wall),
                    "max_wall_ms": max(wall),
                    "median_cpu_ms": statistics.median(
                        r["cpu_ms_per_context"] for r in technical
                    ),
                    "technical_repeats": technical,
                }
            )
            middle = sorted(
                technical, key=lambda r: (r["wall_ms_per_context"], r["repeat"])
            )[len(repeats) // 2]
            phases.append(
                {
                    "domain_size": size,
                    "method": method,
                    "selected_repeat": middle["repeat"],
                    "wall_ms_per_context": middle["wall_ms_per_context"],
                    "components": middle["phase_ms_per_context"],
                    "total_ms": middle["wall_ms_per_context"],
                }
            )
    return {
        "points": points,
        "phases": phases,
        "contexts_per_domain": len(expected_contexts),
        "technical_repeats": len(repeats),
        "raw_route_measurements": len(rows),
        "inference": (
            "descriptive technical timing; no confidence interval "
            "or independent-replicate interpretation"
        ),
    }


def axis_style(ax, theme):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(theme.muted)
    ax.spines[["left", "bottom"]].set_linewidth(0.5)
    ax.tick_params(width=0.5, length=2.5)
    ax.grid(axis="y", color=theme.grid, linewidth=0.4)
    ax.set_axisbelow(True)
    ax.set_ylim(bottom=0)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))


def save_figure(fig, stem):
    exported = []
    for suffix in ("pdf", "svg", "png"):
        path = DEST / f"{stem}.{suffix}"
        kwargs = (
            {
                "metadata": {
                    "Creator": "Artifact-derived CSCWD figures",
                    "CreationDate": None,
                }
            }
            if suffix == "pdf"
            else {"metadata": {"Date": None}}
            if suffix == "svg"
            else {}
        )
        fig.savefig(path, dpi=300, facecolor="white", **kwargs)
        exported.append(path)
    plt.close(fig)
    return exported


def comparison(data, theme, grayscale=False):
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.35))
    fig.subplots_adjust(left=0.075, right=0.99, bottom=0.22, top=0.77, wspace=0.35)
    sizes = sorted({r["domain_size"] for r in data["points"]})
    lookup = {(r["method"], r["domain_size"]): r for r in data["points"]}
    handles = []
    for index, method in enumerate(METHODS):
        rgb = [int(COLORS[index][j : j + 2], 16) / 255 for j in (1, 3, 5)]
        luminance = sum(a * b for a, b in zip(rgb, (0.299, 0.587, 0.114), strict=True))
        color = str(luminance) if grayscale else COLORS[index]
        style = dict(
            color=color,
            marker=MARKERS[index],
            linestyle=LINES[index],
            markersize=4,
            linewidth=1.4 if method == "persistent" else 0.9,
            markerfacecolor="white",
            markeredgewidth=0.8,
        )
        series = [lookup[method, size] for size in sizes]
        if method != "complete_sql":
            axes[0].plot(
                range(len(sizes)),
                [r["search_calls_per_context"] for r in series],
                **style,
            )
        centers = [r["median_wall_ms"] for r in series]
        lower = [r["median_wall_ms"] - r["min_wall_ms"] for r in series]
        upper = [r["max_wall_ms"] - r["median_wall_ms"] for r in series]
        # Deterministic grouping within categorical domains, not data jitter.
        grouped_x = [j + (index - 2) * 0.07 for j in range(len(sizes))]
        line = axes[1].errorbar(
            grouped_x, centers, yerr=[lower, upper], capsize=2, **style
        )
        handles.append(line.lines[0])
    for ax, title, ylabel in zip(
        axes,
        ("(a) Search evaluations", "(b) Complete planning cost"),
        ("Policy calls / revision", "Wall time / revision (ms)"),
        strict=True,
    ):
        axis_style(ax, theme)
        ax.set_title(title, fontsize=8.5, loc="left", pad=6)
        ax.set_xticks(range(len(sizes)), [f"{v:,}" for v in sizes])
        ax.set_xlabel("Sequential nested target domains\n(immutable cache retained)")
        ax.set_ylabel(ylabel)
        ax.set_xlim(-0.25, len(sizes) - 0.75)
    axes[0].text(
        0.02,
        0.97,
        "SQL calls unavailable",
        transform=axes[0].transAxes,
        va="top",
        ha="left",
        fontsize=7,
        color=theme.muted,
    )
    fig.legend(
        handles,
        LABELS,
        loc="upper center",
        ncols=5,
        bbox_to_anchor=(0.53, 1.01),
        fontsize=7.2,
        handlelength=2.3,
        columnspacing=1.2,
        frameon=False,
    )
    return fig


def decomposition(data, theme):
    fig, ax = plt.subplots(figsize=(7.0, 2.75))
    fig.subplots_adjust(left=0.08, right=0.99, bottom=0.24, top=0.79)
    rows = data["phases"]
    labels = list(PHASE_GROUPS) + ["Timer / control"]
    colors = ("#172c51", "#9aa6b6", "#c2c8cf", "#e1e4e8", "#ffffff")
    hatches = ("", "//", "..", "xx", "++")
    bottom = [0.0] * len(rows)
    for label, color, hatch in zip(labels, colors, hatches, strict=True):
        values = [r["components"][label] for r in rows]
        ax.bar(
            range(len(rows)),
            values,
            bottom=bottom,
            width=0.68,
            color=color,
            hatch=hatch,
            edgecolor="#4b5563",
            linewidth=0.35,
            label=label,
        )
        bottom = [a + b for a, b in zip(bottom, values, strict=True)]
    axis_style(ax, theme)
    ax.set_ylabel("Wall time / revision (ms)")
    ax.set_xticks(range(len(rows)), SHORT * 3, rotation=45, ha="right")
    for index, size in enumerate(sorted({r["domain_size"] for r in rows})):
        ax.text(
            index * 5 + 2,
            1.02,
            f"{size:,} targets",
            transform=ax.get_xaxis_transform(),
            ha="center",
            fontsize=8,
        )
    for x in (4.5, 9.5):
        ax.axvline(x, color=theme.grid, linewidth=0.5)
    fig.legend(
        loc="upper center",
        ncols=3,
        bbox_to_anchor=(0.55, 1.01),
        fontsize=7.2,
        frameon=False,
    )
    return fig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=SOURCE)
    args = parser.parse_args()
    source = args.input.resolve()
    rows = json.loads(source.read_text(encoding="utf-8"))
    assert isinstance(rows, list)
    data = aggregate(rows)
    theme, library = load_theme()
    font_manager.fontManager.addfont(
        "/System/Library/Fonts/Supplemental/Times New Roman.ttf"
    )
    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "font.size": 8,
            "axes.labelsize": 8,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "svg.hashsalt": "cscwd-revision-workload-v1",
        }
    )
    DEST.mkdir(parents=True, exist_ok=True)
    data_path = DEST / "revision_workload_plot_data.json"
    save_json(data_path, data)
    csv_path = DEST / "revision_workload_plot_data.csv"
    columns = [k for k in data["points"][0] if k != "technical_repeats"]
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows({k: r[k] for k in columns} for r in data["points"])
    exports = save_figure(comparison(data, theme), "revision_workload_cost")
    exports += save_figure(decomposition(data, theme), "revision_workload_phases")
    gray = WORK / "comparison_gray.png"
    fig = comparison(data, theme, grayscale=True)
    fig.savefig(gray, dpi=250, facecolor="white")
    plt.close(fig)
    outputs = [data_path, csv_path, gray, *exports]
    manifest = {
        "study": "revision_workload_v1",
        "status": "GENERATED_REQUIRES_VISUAL_INSPECTION",
        "inputs": {str(source.relative_to(ROOT)): sha(source)},
        "source": {str(Path(__file__).relative_to(ROOT)): sha(Path(__file__))},
        "ccfa_recipe_library_sha256": sha(library),
        "recipe": (
            "small-multiple lines and grouped additive bars, translated to Matplotlib"
        ),
        "outputs": {str(p.relative_to(ROOT)): sha(p) for p in outputs},
        "quantitative_axes_start_at_zero": True,
        "timing_scope": (
            "complete planner on prevalidated tokens, including immutable "
            "dependency construction; not acquisition/replay latency"
        ),
        "technical_range_not_confidence_interval": True,
    }
    save_json(DEST / "figure_manifest_revision_workload.json", manifest)
    print(
        json.dumps(
            {
                "outputs": len(outputs),
                "contexts_per_domain": data["contexts_per_domain"],
                "route_measurements": data["raw_route_measurements"],
            }
        )
    )


if __name__ == "__main__":
    main()
