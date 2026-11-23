"""Read-only closed-form checks over serialized task and accounting evidence.

This module intentionally imports no policy, RNG, CLOB, replay, contract compiler
or conformance auditor. It checks a small declared specification, not a second
implementation of the simulation or a complete numerical oracle.
"""

from __future__ import annotations

from typing import Any


def specification_witnesses(capsule: dict[str, Any]) -> dict[str, Any]:
    """Check action congruences and terminal trade/account identities exactly."""
    findings: list[dict[str, Any]] = []
    decisions_checked = 0
    accounts_checked = 0
    spec = capsule["spec"]
    population = [f"agent-{i:03d}" for i in range(spec["agent_count"])]
    target = population[len(population) // 2]
    names = {*population, "book-bid", "book-ask"}

    def flag(code: str, world: str, epoch: int, agent: str) -> None:
        findings.append({"code": code, "world": world, "epoch": epoch, "agent": agent})

    for world in ("baseline", "policy"):
        value = capsule["worlds"][world]
        for row in value["epochs"]:
            requests = {r["task_id"]: r for r in row["requests"]}
            for result in row["results"]:
                if result["status"] != "active":
                    continue
                decisions_checked += 1
                agent, epoch = result["agent"], row["epoch"]
                view = requests[result["task_id"]]["view"]
                replacement = (
                    world == "policy"
                    and spec["branch"] == "cancellation"
                    and agent == target
                    and epoch == spec["branch_epoch"]
                )
                prefix = "replacement_" if replacement else ""
                draws = {
                    d["address"]["draw_role"]: d["value"]
                    for d in result["draws"]
                    if d["state"] == "consumed"
                }
                market = [
                    c for c in result["commands"] if c["command_type"] == "market_order"
                ]
                if len(market) != 1:
                    flag("one_market_action", world, epoch, agent)
                    continue
                command = market[0]
                # Congruences state the rule algebraically; they do not invoke F.
                signal = (
                    3 * view["bid_quantity"]
                    + 5 * view["ask_quantity"]
                    + view["best_bid"]
                    + view["best_ask"]
                    + 7 * view["asset"]
                    + 7 * (view["cash"] // 100)
                )
                q = command["quantity"]
                if not (
                    1 <= q <= 9
                    and (q - 1 - draws[prefix + "quantity"] - signal) % 9 == 0
                ):
                    flag("quantity_congruence", world, epoch, agent)
                parity = (draws[prefix + "side"] + signal) % 2
                if (command["side"] == "buy") != (parity == 0):
                    flag("side_congruence", world, epoch, agent)
        accounts = {a["trader_id"]: a for a in value["final_state"]["accounts"]}
        if set(accounts) != names:
            flag("account_set", world, -1, "")
            continue
        trades = value["trades"]
        for name in sorted(names):
            accounts_checked += 1
            asset_delta = sum(
                t["quantity"] * ((t["buyer_id"] == name) - (t["seller_id"] == name))
                for t in trades
            )
            cash_delta = sum(
                t["price"]
                * t["quantity"]
                * ((t["seller_id"] == name) - (t["buyer_id"] == name))
                for t in trades
            )
            fees = sum(
                (t["maker_fee"] if t["maker_trader_id"] == name else 0)
                + (t["taker_fee"] if t["taker_trader_id"] == name else 0)
                for t in trades
            )
            if accounts[name]["asset"] != 1_000_000 + asset_delta:
                flag("asset_transfer_identity", world, -1, name)
            if accounts[name]["cash"] != 100_000_000 + cash_delta - fees:
                flag("cash_transfer_identity", world, -1, name)
        if value["traded_notional"] != sum(t["price"] * t["quantity"] for t in trades):
            flag("notional_identity", world, -1, "")
    return {
        "pass": not findings,
        "findings": findings,
        "decisions_checked": decisions_checked,
        "accounts_checked": accounts_checked,
        "scope": (
            "closed-form action and trade/account identities, "
            "not complete simulation equivalence"
        ),
    }
