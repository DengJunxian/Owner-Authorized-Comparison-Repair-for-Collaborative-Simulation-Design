"""Build Table I from the two sealed studies; never write to their directories."""

from __future__ import annotations

import json
from pathlib import Path

PAPER = Path(__file__).resolve().parent
RESULTS = PAPER.parents[1] / "results/cscwd2027"


def load(study: str, name: str):
    return json.loads((RESULTS / study / name).read_text(encoding="utf-8"))


def block(title, cases, methods, metrics, physical):
    yield rf"\multicolumn{{5}}{{@{{}}l}}{{\emph{{{title} ({cases:,})}}}} \\"
    for key, name in methods:
        n, exact, calls, missed = metrics(key)
        assert n == cases and 0 <= exact <= n and 0 <= missed <= n - exact
        reports = [p for p in physical if p["method"] == key]
        comparisons = [p["exact"] for p in reports if p["exact"] is not None]
        assert all(p["status"] == "released" for p in reports if p["exact"] is not None)
        assert not reports or comparisons
        values = (
            name,
            f"{exact}/{n}",
            "--" if calls is None else f"{calls:,}",
            str(missed),
            f"{sum(comparisons)}/{len(comparisons)}" if comparisons else "n/a",
        )
        yield " & ".join(values) + r" \\"


def main():
    staged = "revision_workload_v1"
    generated = "consent_projection_v1"
    revision = load(staged, "summary.json")
    projection = load(generated, "summary.json")

    def staged_metrics(key):
        row = revision["methods"][key]
        return (
            row["contexts"], row["exact_selection_and_work"],
            row["search_predicate_calls"], row["false_abstention"],
        )

    def generated_metrics(key):
        row = projection["methods"][key]
        return (
            row["requests"], row["requests"] - row["selection_disagreements"],
            row["search_predicate_calls"], row["false_abstentions"],
        )

    lines = [
        r"\begin{tabular}{@{}lrrrr@{}}",
        r"\toprule",
        r"Method & Plan match & Calls & Missed & Exact/ref. \\",
        r"\midrule",
        *block(
            "Staged revisions", revision["planning_contexts"],
            (
                ("exhaustive", "Full search"),
                ("context_local", "Context-local reuse"),
                ("persistent", "Cross-revision reuse"),
                ("generic_memo", "Annotated memo"),
                ("complete_sql", "Complete SQL"),
            ),
            staged_metrics, load(staged, "physical_rows.json"),
        ),
        r"\midrule",
        *block(
            "Generated policies", projection["planning_requests"],
            (
                ("exhaustive", "Full search"),
                ("certificates", "Refusal reuse"),
                ("policy_sql", "Complete SQL"),
                ("full_key_memo", "Full-proposal key"),
                ("omit_work", "Omitted work deps."),
                ("work_pruned", "Pre-pruning"),
            ),
            generated_metrics, load(generated, "analysis/physical_rows.json"),
        ),
        r"\bottomrule",
        r"\end{tabular}",
    ]
    (PAPER / "generated/editorial_method_summary.tex").write_text(
        "\n".join(lines) + "\n", encoding="utf-8",
    )


if __name__ == "__main__":
    main()
