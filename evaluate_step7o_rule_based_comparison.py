"""
Step 7O-B: FC-HMARL vs passive grid-only vs active rule-based BESS EMS
on the same 30 held-out TEST episodes.

IMPORTANT
---------
The rule-based EMS is a transparent reconstructed benchmark, not a manuscript-
claimed controller and not an optimized MPC/EMS.

Rule-based policy:
1. No inter-MG energy sharing.
2. No reserve participation.
3. If PV surplus exists -> charge BESS.
4. If net deficit exists and SOC > 50% -> discharge BESS.
5. Otherwise BESS idles.
6. Grid automatically closes the remaining physical balance.

The BESS command magnitude is proportional to the local net imbalance and is
converted to the normalized local action using the manuscript microgrid BESS
rated powers.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import evaluate_real_fc_hmarl_test as t


PROJECT_ROOT = Path(r"D:\Molvi paper review\FC_HMARL")
OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "evaluation"
    / "step7o_rule_based_comparison"
)

LOCKED_CHECKPOINT = 1000
NUMBER_OF_MGS = 5
EPISODE_LENGTH = 24

# Manuscript Table-2 BESS rated powers, kW.
BESS_RATED_POWER_KW = np.array(
    [250.0, 300.0, 250.0, 300.0, 350.0],
    dtype=float,
)

SOC_DISCHARGE_THRESHOLD = 0.50


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=int, default=1000)
    p.add_argument("--test-episodes", type=int, default=30)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cpu")
    return p.parse_args()


def passive_actions():
    local_actions = []
    for _ in range(NUMBER_OF_MGS):
        a = np.zeros(5, dtype=np.float32)
        a[0] = 0.0
        a[1:] = -1.0
        local_actions.append(a)

    # Disable coordinator reserve and sharing.
    coordinator_action = np.array(
        [-1.0, -1.0, -1.0],
        dtype=np.float32,
    )

    return t.val.HierarchicalActionBundle(
        local_actions=local_actions,
        coordinator_action=coordinator_action,
    )


def rule_based_actions(observation):
    """
    Local state structure:
        [SOC, PV, load, EV, grid, S_pred(96)]

    Positive normalized BESS command -> discharge.
    Negative normalized BESS command -> charge.
    """
    local_actions = []

    for i, state in enumerate(observation.local_states):
        s = np.asarray(state, dtype=float)

        soc = float(s[0])
        pv_kw = float(s[1])
        load_kw = float(s[2])
        ev_kw = float(s[3])

        net_demand_kw = load_kw + ev_kw - pv_kw
        rated_kw = float(BESS_RATED_POWER_KW[i])

        if net_demand_kw < 0.0:
            # PV surplus: charge according to surplus magnitude.
            desired_charge_kw = min(
                abs(net_demand_kw),
                rated_kw,
            )
            bess_normalized = -desired_charge_kw / rated_kw

        elif net_demand_kw > 0.0 and soc > SOC_DISCHARGE_THRESHOLD:
            # Deficit and sufficient SOC: discharge according to deficit.
            desired_discharge_kw = min(
                net_demand_kw,
                rated_kw,
            )
            bess_normalized = desired_discharge_kw / rated_kw

        else:
            bess_normalized = 0.0

        a = np.zeros(5, dtype=np.float32)
        a[0] = float(
            np.clip(bess_normalized, -1.0, 1.0)
        )

        # No sharing in this benchmark.
        a[1:] = -1.0
        local_actions.append(a)

    # No upper-level market intervention/reserve/sharing in rule-based EMS.
    coordinator_action = np.array(
        [-1.0, -1.0, -1.0],
        dtype=np.float32,
    )

    return t.val.HierarchicalActionBundle(
        local_actions=local_actions,
        coordinator_action=coordinator_action,
    )


def evaluate(
    mode,
    bridge,
    local_agents,
    coordinator_agent,
    starts,
    episodes,
):
    rows = []

    for ep in range(1, episodes + 1):
        obs = bridge.reset(episode=ep)

        total_return = 0.0
        local_return = np.zeros(NUMBER_OF_MGS, dtype=float)
        coordinator_return = 0.0

        grid_import = 0.0
        grid_export = 0.0
        purchase_cost = 0.0
        sale_revenue = 0.0
        reserve_revenue = 0.0
        market_profit = 0.0
        bess_throughput = 0.0
        sharing = 0.0
        sharing_loss = 0.0

        grid_power = []
        balance_violations = 0
        transformer_violations = 0

        with torch.no_grad():
            for _ in range(EPISODE_LENGTH):

                if mode == "full_fc_hmarl":
                    actions = t.val.select_deterministic_actions(
                        local_agents=local_agents,
                        coordinator_agent=coordinator_agent,
                        observation=obs,
                    )
                elif mode == "passive_grid_only":
                    actions = passive_actions()
                elif mode == "rule_based_bess_ems":
                    actions = rule_based_actions(obs)
                else:
                    raise ValueError(mode)

                result = bridge.step(actions)

                lr = np.asarray(
                    result.local_rewards,
                    dtype=float,
                )
                cr = float(result.coordinator_reward)

                total_return += float(lr.sum()) + cr
                local_return += lr
                coordinator_return += cr

                physical = result.info["physical_result"]
                totals = physical["totals"]
                constraints = physical["constraints"]

                grid_import += float(totals["grid_import_kw"])
                grid_export += float(totals["grid_export_kw"])
                purchase_cost += float(
                    totals["grid_purchase_cost_usd"]
                )
                sale_revenue += float(
                    totals["grid_sale_revenue_usd"]
                )
                reserve_revenue += float(
                    totals["reserve_revenue_usd"]
                )
                market_profit += float(
                    totals["market_profit_usd"]
                )
                sharing += float(
                    totals["sharing_scheduled_kw"]
                )
                sharing_loss += float(
                    totals["sharing_loss_kw"]
                )

                for local in physical["local_results"]:
                    bess_throughput += float(
                        local["bess"]["energy_throughput_kwh"]
                    )

                grid_power.append(
                    float(totals["grid_power_kw"])
                )

                if not constraints["all_power_balanced"]:
                    balance_violations += 1

                if not constraints["all_transformers_feasible"]:
                    transformer_violations += 1

                obs = result.next_observation

                if result.done:
                    break

        net_market_cost = (
            purchase_cost
            - sale_revenue
            - reserve_revenue
        )

        g = np.asarray(grid_power, dtype=float)

        rows.append({
            "mode": mode,
            "test_episode": ep,
            "start_index": int(starts[ep - 1]),
            "total_return": total_return,
            "coordinator_return": coordinator_return,
            "mean_local_return_per_mg": float(local_return.mean()),
            "grid_import_kwh": grid_import,
            "grid_export_kwh": grid_export,
            "grid_purchase_cost_usd": purchase_cost,
            "grid_sale_revenue_usd": sale_revenue,
            "reserve_revenue_usd": reserve_revenue,
            "net_market_cost_usd": net_market_cost,
            "market_profit_usd": market_profit,
            "bess_throughput_kwh": bess_throughput,
            "sharing_scheduled_kwh": sharing,
            "sharing_loss_kwh": sharing_loss,
            "peak_grid_import_kw": float(
                max(0.0, np.max(g))
            ),
            "peak_grid_export_kw": float(
                max(0.0, -np.min(g))
            ),
            "power_balance_violation_steps":
                balance_violations,
            "transformer_violation_steps":
                transformer_violations,
        })

        r = rows[-1]
        print(
            f"{mode:22s} | ep {ep:02d} | "
            f"return {r['total_return']: .6f} | "
            f"import {r['grid_import_kwh']:9.2f} kWh | "
            f"net cost ${r['net_market_cost_usd']:9.2f}"
        )

    return pd.DataFrame(rows)


def mean_metric(df, mode, metric):
    return float(
        df.loc[df["mode"] == mode, metric].mean()
    )


def reduction_percent(reference, proposed):
    if abs(reference) < 1e-12:
        return np.nan
    return 100.0 * (reference - proposed) / abs(reference)


def main():
    args = parse_args()

    if args.checkpoint != LOCKED_CHECKPOINT:
        raise ValueError(
            "Final TEST benchmark is locked to checkpoint 1000."
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 80)
    print("STEP 7O-B - THREE-WAY BENCHMARK COMPARISON")
    print("=" * 80)
    print("1. Passive grid-only")
    print("2. Active rule-based BESS EMS")
    print("3. FC-HMARL checkpoint 1000")
    print("Same held-out TEST episodes; deterministic; no learning.")

    data = t.TestData()

    starts = np.asarray(
        t.build_fixed_test_episode_starts(
            data=data,
            number_of_episodes=args.test_episodes,
            seed=args.seed,
        ),
        dtype=int,
    )

    local_agents, coordinator_agent = t.val.load_policy(
        checkpoint_episode=args.checkpoint,
        device=args.device,
        seed=args.seed,
    )

    frames = []

    for mode in (
        "passive_grid_only",
        "rule_based_bess_ems",
        "full_fc_hmarl",
    ):
        bridge = t.build_test_bridge(
            data=data,
            episode_starts=starts,
        )

        frames.append(
            evaluate(
                mode=mode,
                bridge=bridge,
                local_agents=local_agents,
                coordinator_agent=coordinator_agent,
                starts=starts,
                episodes=args.test_episodes,
            )
        )

    df = pd.concat(
        frames,
        ignore_index=True,
    )

    metrics = [
        "total_return",
        "grid_import_kwh",
        "grid_export_kwh",
        "net_market_cost_usd",
        "bess_throughput_kwh",
        "sharing_scheduled_kwh",
        "peak_grid_import_kw",
    ]

    summary_rows = []

    for mode in (
        "passive_grid_only",
        "rule_based_bess_ems",
        "full_fc_hmarl",
    ):
        row = {"mode": mode}
        for metric in metrics:
            row[metric] = mean_metric(
                df,
                mode,
                metric,
            )

        row["power_balance_violation_steps"] = int(
            df.loc[
                df["mode"] == mode,
                "power_balance_violation_steps",
            ].sum()
        )
        row["transformer_violation_steps"] = int(
            df.loc[
                df["mode"] == mode,
                "transformer_violation_steps",
            ].sum()
        )

        summary_rows.append(row)

    summary = pd.DataFrame(summary_rows)

    passive = summary[
        summary["mode"] == "passive_grid_only"
    ].iloc[0]

    rule = summary[
        summary["mode"] == "rule_based_bess_ems"
    ].iloc[0]

    fc = summary[
        summary["mode"] == "full_fc_hmarl"
    ].iloc[0]

    comparison = {
        "fc_vs_passive_return_delta":
            float(fc["total_return"] - passive["total_return"]),
        "fc_vs_rule_return_delta":
            float(fc["total_return"] - rule["total_return"]),

        "fc_vs_passive_grid_import_reduction_percent":
            reduction_percent(
                passive["grid_import_kwh"],
                fc["grid_import_kwh"],
            ),

        "fc_vs_rule_grid_import_reduction_percent":
            reduction_percent(
                rule["grid_import_kwh"],
                fc["grid_import_kwh"],
            ),

        "fc_vs_passive_market_cost_reduction_percent":
            reduction_percent(
                passive["net_market_cost_usd"],
                fc["net_market_cost_usd"],
            ),

        "fc_vs_rule_market_cost_reduction_percent":
            reduction_percent(
                rule["net_market_cost_usd"],
                fc["net_market_cost_usd"],
            ),

        "fc_vs_passive_peak_import_reduction_percent":
            reduction_percent(
                passive["peak_grid_import_kw"],
                fc["peak_grid_import_kw"],
            ),

        "fc_vs_rule_peak_import_reduction_percent":
            reduction_percent(
                rule["peak_grid_import_kw"],
                fc["peak_grid_import_kw"],
            ),
    }

    # Paired return wins.
    fc_ep = df[
        df["mode"] == "full_fc_hmarl"
    ].sort_values("test_episode")

    passive_ep = df[
        df["mode"] == "passive_grid_only"
    ].sort_values("test_episode")

    rule_ep = df[
        df["mode"] == "rule_based_bess_ems"
    ].sort_values("test_episode")

    comparison["fc_better_than_passive_episodes"] = int(
        np.sum(
            fc_ep["total_return"].to_numpy()
            > passive_ep["total_return"].to_numpy()
        )
    )

    comparison["fc_better_than_rule_episodes"] = int(
        np.sum(
            fc_ep["total_return"].to_numpy()
            > rule_ep["total_return"].to_numpy()
        )
    )

    episode_file = (
        OUTPUT_DIR
        / "step7o_three_way_episode_results.csv"
    )
    summary_file = (
        OUTPUT_DIR
        / "step7o_three_way_summary.csv"
    )
    json_file = (
        OUTPUT_DIR
        / "step7o_three_way_comparison.json"
    )

    df.to_csv(
        episode_file,
        index=False,
    )

    summary.to_csv(
        summary_file,
        index=False,
    )

    json_file.write_text(
        json.dumps(comparison, indent=4),
        encoding="utf-8",
    )

    print()
    print("=" * 80)
    print("STEP 7O-B THREE-WAY SUMMARY")
    print("=" * 80)
    print(
        summary.to_string(index=False)
    )

    print()
    print("FC-HMARL vs passive:")
    print(
        f"  Return improvement : "
        f"{comparison['fc_vs_passive_return_delta']:.6f}"
    )
    print(
        f"  Grid import reduction : "
        f"{comparison['fc_vs_passive_grid_import_reduction_percent']:.4f}%"
    )
    print(
        f"  Market cost reduction : "
        f"{comparison['fc_vs_passive_market_cost_reduction_percent']:.4f}%"
    )
    print(
        f"  Peak import reduction : "
        f"{comparison['fc_vs_passive_peak_import_reduction_percent']:.4f}%"
    )

    print()
    print("FC-HMARL vs rule-based EMS:")
    print(
        f"  Return improvement : "
        f"{comparison['fc_vs_rule_return_delta']:.6f}"
    )
    print(
        f"  Grid import reduction : "
        f"{comparison['fc_vs_rule_grid_import_reduction_percent']:.4f}%"
    )
    print(
        f"  Market cost reduction : "
        f"{comparison['fc_vs_rule_market_cost_reduction_percent']:.4f}%"
    )
    print(
        f"  Peak import reduction : "
        f"{comparison['fc_vs_rule_peak_import_reduction_percent']:.4f}%"
    )
    print(
        f"  FC better episodes : "
        f"{comparison['fc_better_than_rule_episodes']}/"
        f"{args.test_episodes}"
    )

    print()
    print("[OK] Step 7O-B three-way benchmark complete.")
    print("Episode results :", episode_file)
    print("Summary         :", summary_file)
    print("Comparison JSON :", json_file)


if __name__ == "__main__":
    main()
