"""Hand-verifiable fixtures for the second-review evidence package."""

from fractions import Fraction

from policytwin.design_contract.contract import (
    ComparisonContract,
    Evidence,
    ExpectedSlot,
    Member,
    revision_plan,
)
from scripts.cscwd2027_review2 import conditional, cuts, explain


def token(name, key, value=3):
    return Evidence(name, key, "p", name, True, value, "a", 0)


def test_slot_and_reason_diagnostics_do_not_use_outcomes():
    contract = ComparisonContract("fixture", 1, (ExpectedSlot("a", 0, "ka"),), ("p",))
    panel = [Member("a", 0, token("x", "wrong"))]
    assert explain(contract, panel)["issues"] == [["a", 0, "key_mismatch"]]
    panel = [
        Member("a", 0, token("z", "ka", 99)),
        Member("a", 0, token("b", "ka", -99)),
    ]
    assert explain(contract, panel)["issues"] == [["a", 0, "duplicate_slot"]]
    assert revision_plan(contract, panel)["reuse"][0]["object_id"] == "b"
    assert explain(contract, [])["issues"] == [["a", 0, "missing_slot"]]


def test_conditional_denominator_retains_original_total_and_undefined_empty():
    def row(valid, regret, flip):
        return {
            "mode": "m",
            "released": True,
            "valid": valid,
            "reference_gap": [5, 1],
            "effects": {
                "assessable": True,
                "regret": [regret, 1],
                "optimal_set_error": regret > 0,
                "top1_disagreement": flip,
            },
        }

    values = conditional(
        [row(True, 0, False), row(False, 2, True), row(False, 0, False)],
        [Fraction(1), Fraction(4), Fraction(8)],
    )["m"]
    assert values["all_assessable_reports"] == 3
    assert values["unauthorized_reports"] == 2 and values["conditional_flips"] == 1
    assert values["positive_regret_median"] == [2, 1]
    empty = conditional([row(True, 0, False)], [Fraction(1), Fraction(4), Fraction(8)])[
        "m"
    ]
    assert (
        empty["unauthorized_reports"] == 0 and empty["positive_regret_median"] is None
    )


def test_exact_gap_strata_and_coupling_counterexample():
    assert cuts([Fraction(v) for v in (0, 4, 8, 12, 16)]) == [4, 8, 12]
    # Same Bernoulli marginals, distinct paired-contrast laws.
    shared = [u - u for u in (0, 1)]
    opposite = [(1 - u) - u for u in (0, 1)]
    assert sum(shared) == sum(opposite) == 0
    assert shared == [0, 0] and opposite == [1, -1]


def test_new_output_root_uses_full_gate_and_old_replay(tmp_path, monkeypatch):
    """One engineering-only critical-path check; not publication evidence."""
    from dataclasses import replace

    from scripts import cscwd2027_review2 as study
    from scripts.cscwd2027_design_contract.design import recipes

    original_root = study.store.RESULTS
    session = study.read(study.DEST / "sessions.json")[0]
    recipe = recipes(session, 2)["cancellation", 0]
    recipe = replace(recipe, spec=replace(recipe.spec, seed="review2-engineering-only"))
    seal = tmp_path / "engineering.json"
    study.save(seal, {"purpose": "engineering only; excluded from formal results"})
    monkeypatch.setattr(study, "SEAL", seal)
    value, cap = study.execute_new(
        tmp_path / "new-store", {"id": "engineering"}, recipe, "smoke"
    )
    assert value.evidence.local_pass and cap["worlds"]["baseline"]["completed"]
    old, _, decision = study.revalidate_old(
        session, "cancellation", 0, tmp_path / "replay"
    )
    assert old.evidence.local_pass and decision["release_authorized"]
    assert original_root == study.store.RESULTS
    assert not (original_root / "objects/engineering-smoke-cancellation-r0").exists()
