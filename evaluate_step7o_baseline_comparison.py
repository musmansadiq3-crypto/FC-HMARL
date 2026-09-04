"""
Step 7O-A: FC-HMARL versus passive grid-only baseline
on the same 30 held-out TEST episodes.

Baseline definition
-------------------
Passive grid-only:
- BESS request = 0 for every MG.
- Inter-MG energy sharing = 0.
- Coordinator reserve participation = 0.
- Coordinator sharing multiplier = 0.
- Grid auto-balances each microgrid through the validated physical environment.

This is a deterministic, non-learning reference baseline. It is NOT claimed to
be an optimized conventional EMS.
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
    / "step7o_baseline_comparison"
)

LOCKED_CHECKPOINT = 1000
NUMBER_OF_MGS = 5
EPISODE_LENGTH = 24
DEGRADATION_USD_PER_KWH = 0.02


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=int, default=1000)
    p.add_argument("--test-episodes", type=int, default=30)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cpu")
    return p.parse_args()


def make_passive_grid_only_actions():
    """
    Local reconstructed software action:
        [BESS, share_to_MG2, share_to_MG3, share_to_MG4, share_to_MG5]
    with destination order adjusted internally per MG.

    BESS=0 -> zero requested battery power.
    Sharing=-1 -> no sharing request.

    Coordinator reconstructed action:
        [market/grid-related, reserve, sharing multiplier]
    -1 values disable reserve/sharing in the current mapper.
    """
    local_actions = []

    for _ in range(NUMBER_OF_MGS):
        action = np.zeros(5, dtype=np.float32)
        action[0] = 0.0
        action[1:] = -1.0
        local_actions.append(action)

    coordinator_action = np.array(
        [-1.0, -1.0, -1.0],
        dtype=np.float32,
    )

    return t.val.HierarchicalActionBundle(
        local_actions=local_actions,
        coordinator_action=coordinator_action,
    )


def summarize_episode(
    mode,
    episode,
    start_index,
    total_return,
    local_returns,
    coordinator_return,
    totals_acc,
    grid_series,
    violation_steps,
):
    throughput = totals_acc["bess_throughput_kwh"]
    degradation = throughput * DEGRADATION_USD_PER_KWH

    net_market_cost = (
        totals_acc["grid_purchase_cost_usd"]
        - totals_acc["grid_sale_revenue_usd"]
        - totals_acc["reserve_revenue_usd"]
    )

    g = np.asarray(grid_series, dtype=float)

    return {
        "mode": mode,
        "test_episode": int(episode),
        "start_index": int(start_index),
        "total_return": float(total_return),
        "coordinator_return": float(coordinator_return),
        "mean_local_return_per_mg": float(np.mean(local_returns)),
        "load_kwh": float(totals_acc["load_kwh"]),
        "pv_kwh": float(totals_acc["pv_kwh"]),
        "ev_kwh": float(totals_acc["ev_kwh"]),
        "grid_import_kwh": float(totals_acc["grid_import_kwh"]),
        "grid_export_kwh": float(totals_acc["grid_export_kwh"]),
        "grid_purchase_cost_usd": float(
            totals_acc["grid_purchase_cost_usd"]
        ),
        "grid_sale_revenue_usd": float(
            totals_acc["grid_sale_revenue_usd"]
        ),
        "reserve_revenue_usd": float(
            totals_acc["reserve_revenue_usd"]
        ),
        "market_profit_usd": float(
            totals_acc["market_profit_usd"]
        ),
        "net_market_cost_usd": float(net_market_cost),
        "bess_throughput_kwh": float(throughput),
        "bess_degradation_cost_usd": float(degradation),
        "sharing_scheduled_kwh": float(
            totals_acc["sharing_scheduled_kwh"]
        ),
        "sharing_received_kwh": float(
            totals_acc["sharing_received_kwh"]
        ),
        "sharing_loss_kwh": float(
            totals_acc["sharing_loss_kwh"]
        ),
        "peak_grid_import_kw": float(
            max(0.0, np.max(g))
        ),
        "peak_grid_export_kw": float(
            max(0.0, -np.min(g))
        ),
        "mean_absolute_grid_exchange_kw": float(
            np.mean(np.abs(g))
        ),
        "power_balance_violation_steps": int(
            violation_steps["power_balance"]
        ),
        "transformer_violation_steps": int(
            violation_steps["transformer"]
        ),
    }


def evaluate_mode(
    *,
    mode,
    bridge,
    local_agents,
    coordinator_agent,
    starts,
    number_of_episodes,
):
    rows = []

    for episode in range(1, number_of_episodes + 1):
        obs = bridge.reset(episode=episode)

        totals_acc = {
            "load_kwh": 0.0,
            "pv_kwh": 0.0,
            "ev_kwh": 0.0,
            "grid_import_kwh": 0.0,
            "grid_export_kwh": 0.0,
            "grid_purchase_cost_usd": 0.0,
            "grid_sale_revenue_usd": 0.0,
            "reserve_revenue_usd": 0.0,
            "market_profit_usd": 0.0,
            "bess_throughput_kwh": 0.0,
            "sharing_scheduled_kwh": 0.0,
            "sharing_received_kwh": 0.0,
            "sharing_loss_kwh": 0.0,
        }

        total_return = 0.0
        coordinator_return = 0.0
        local_returns = np.zeros(NUMBER_OF_MGS, dtype=float)
        grid_series = []

        violation_steps = {
            "power_balance": 0,
            "transformer": 0,
        }

        with torch.no_grad():
            for _ in range(EPISODE_LENGTH):

                if mode == "full_fc_hmarl":
                    actions = t.val.select_deterministic_actions(
                        local_agents=local_agents,
                        coordinator_agent=coordinator_agent,
                        observation=obs,
                    )
                elif mode == "passive_grid_only":
                    actions = make_passive_grid_only_actions()
                else:
                    raise ValueError(mode)

                result = bridge.step(actions)

                local_reward = np.asarray(
                    result.local_rewards,
                    dtype=float,
                )
                coord_reward = float(
                    result.coordinator_reward
                )

                total_return += (
                    float(local_reward.sum())
                    + coord_reward
                )
                local_returns += local_reward
                coordinator_return += coord_reward

                physical = result.info["physical_result"]
                totals = physical["totals"]
                constraints = physical["constraints"]

                totals_acc["load_kwh"] += float(
                    totals["load_power_kw"]
                )
                totals_acc["pv_kwh"] += float(
                    totals["pv_power_kw"]
                )
                totals_acc["ev_kwh"] += float(
                    totals["ev_power_kw"]
                )
                totals_acc["grid_import_kwh"] += float(
                    totals["grid_import_kw"]
                )
                totals_acc["grid_export_kwh"] += float(
                    totals["grid_export_kw"]
                )
                totals_acc["grid_purchase_cost_usd"] += float(
                    totals["grid_purchase_cost_usd"]
                )
                totals_acc["grid_sale_revenue_usd"] += float(
                    totals["grid_sale_revenue_usd"]
                )
                totals_acc["reserve_revenue_usd"] += float(
                    totals["reserve_revenue_usd"]
                )
                totals_acc["market_profit_usd"] += float(
                    totals["market_profit_usd"]
                )
                totals_acc["sharing_scheduled_kwh"] += float(
                    totals["sharing_scheduled_kw"]
                )
                totals_acc["sharing_received_kwh"] += float(
                    totals["sharing_received_kw"]
                )
                totals_acc["sharing_loss_kwh"] += float(
                    totals["sharing_loss_kw"]
                )

                for local in physical["local_results"]:
                    totals_acc["bess_throughput_kwh"] += float(
                        local["bess"]["energy_throughput_kwh"]
                    )

                grid_series.append(
                    float(totals["grid_power_kw"])
                )

                if not bool(
                    constraints["all_power_balanced"]
                ):
                    violation_steps["power_balance"] += 1

                if not bool(
                    constraints["all_transformers_feasible"]
                ):
                    violation_steps["transformer"] += 1

                obs = result.next_observation

                if result.done:
                    break

        row = summarize_episode(
            mode=mode,
            episode=episode,
            start_index=starts[episode - 1],
            total_return=total_return,
            local_returns=local_returns,
            coordinator_return=coordinator_return,
            totals_acc=totals_acc,
            grid_series=grid_series,
            violation_steps=violation_steps,
        )

        rows.append(row)

        print(
            f"{mode:20s} | episode {episode:02d} | "
            f"return {row['total_return']: .6f} | "
            f"import {row['grid_import_kwh']:9.2f} kWh | "
            f"net market cost ${row['net_market_cost_usd']:9.2f}"
        )

    return pd.DataFrame(rows)


def mean_row(df, mode):
    d = df[df["mode"] == mode]
    numeric = d.select_dtypes(include=[np.number])
    means = numeric.mean().to_dict()
    means["mode"] = mode
    return means


def percent_reduction(baseline, proposed):
    if abs(baseline) < 1e-12:
        return np.nan
    return 100.0 * (baseline - proposed) / abs(baseline)


def main():
    args = parse_args()

    if args.checkpoint != LOCKED_CHECKPOINT:
        raise ValueError(
            "Final TEST comparison is locked to checkpoint 1000."
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("STEP 7O-A - FC-HMARL VS PASSIVE GRID-ONLY BASELINE")
    print("=" * 80)
    print(f"Checkpoint    : {args.checkpoint}")
    print(f"TEST episodes : {args.test_episodes}")
    print(f"Seed          : {args.seed}")
    print("Actions       : deterministic")
    print("Learning      : disabled")
    print("Baseline      : passive grid-only, non-learning")

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

    full_bridge = t.build_test_bridge(
        data=data,
        episode_starts=starts,
    )

    passive_bridge = t.build_test_bridge(
        data=data,
        episode_starts=starts,
    )

    full_df = evaluate_mode(
        mode="full_fc_hmarl",
        bridge=full_bridge,
        local_agents=local_agents,
        coordinator_agent=coordinator_agent,
        starts=starts,
        number_of_episodes=args.test_episodes,
    )

    passive_df = evaluate_mode(
        mode="passive_grid_only",
        bridge=passive_bridge,
        local_agents=local_agents,
        coordinator_agent=coordinator_agent,
        starts=starts,
        number_of_episodes=args.test_episodes,
    )

    all_df = pd.concat(
        [full_df, passive_df],
        ignore_index=True,
    )

    # Means
    full_mean = mean_row(all_df, "full_fc_hmarl")
    passive_mean = mean_row(all_df, "passive_grid_only")

    summary = pd.DataFrame(
        [full_mean, passive_mean]
    )

    # Paired episode differences: FC-HMARL minus passive.
    paired = full_df.sort_values(
        "test_episode"
    ).reset_index(drop=True).copy()

    passive_sorted = passive_df.sort_values(
        "test_episode"
    ).reset_index(drop=True)

    paired_out = pd.DataFrame({
        "test_episode": paired["test_episode"],
        "start_index": paired["start_index"],
        "return_delta_fc_minus_passive":
            paired["total_return"]
            - passive_sorted["total_return"],
        "grid_import_reduction_kwh":
            passive_sorted["grid_import_kwh"]
            - paired["grid_import_kwh"],
        "net_market_cost_reduction_usd":
            passive_sorted["net_market_cost_usd"]
            - paired["net_market_cost_usd"],
        "peak_import_reduction_kw":
            passive_sorted["peak_grid_import_kw"]
            - paired["peak_grid_import_kw"],
    })

    comparison = {
        "mean_return_fc_hmarl": float(
            full_df["total_return"].mean()
        ),
        "mean_return_passive": float(
            passive_df["total_return"].mean()
        ),
        "mean_return_delta_fc_minus_passive": float(
            (
                full_df["total_return"].to_numpy()
                - passive_df["total_return"].to_numpy()
            ).mean()
        ),
        "fc_better_return_episodes": int(
            np.sum(
                full_df["total_return"].to_numpy()
                > passive_df["total_return"].to_numpy()
            )
        ),
        "mean_grid_import_fc_hmarl_kwh": float(
            full_df["grid_import_kwh"].mean()
        ),
        "mean_grid_import_passive_kwh": float(
            passive_df["grid_import_kwh"].mean()
        ),
        "grid_import_reduction_percent": float(
            percent_reduction(
                passive_df["grid_import_kwh"].mean(),
                full_df["grid_import_kwh"].mean(),
            )
        ),
        "mean_net_market_cost_fc_hmarl_usd": float(
            full_df["net_market_cost_usd"].mean()
        ),
        "mean_net_market_cost_passive_usd": float(
            passive_df["net_market_cost_usd"].mean()
        ),
        "net_market_cost_reduction_percent": float(
            percent_reduction(
                passive_df["net_market_cost_usd"].mean(),
                full_df["net_market_cost_usd"].mean(),
            )
        ),
        "mean_peak_import_fc_hmarl_kw": float(
            full_df["peak_grid_import_kw"].mean()
        ),
        "mean_peak_import_passive_kw": float(
            passive_df["peak_grid_import_kw"].mean()
        ),
        "peak_import_reduction_percent": float(
            percent_reduction(
                passive_df["peak_grid_import_kw"].mean(),
                full_df["peak_grid_import_kw"].mean(),
            )
        ),
        "mean_bess_throughput_fc_hmarl_kwh": float(
            full_df["bess_throughput_kwh"].mean()
        ),
        "mean_bess_throughput_passive_kwh": float(
            passive_df["bess_throughput_kwh"].mean()
        ),
        "mean_sharing_fc_hmarl_kwh": float(
            full_df["sharing_scheduled_kwh"].mean()
        ),
        "mean_sharing_passive_kwh": float(
            passive_df["sharing_scheduled_kwh"].mean()
        ),
        "fc_power_balance_violation_steps": int(
            full_df["power_balance_violation_steps"].sum()
        ),
        "passive_power_balance_violation_steps": int(
            passive_df["power_balance_violation_steps"].sum()
        ),
        "fc_transformer_violation_steps": int(
            full_df["transformer_violation_steps"].sum()
        ),
        "passive_transformer_violation_steps": int(
            passive_df["transformer_violation_steps"].sum()
        ),
    }

    all_file = OUTPUT_DIR / "step7o_episode_results.csv"
    summary_file = OUTPUT_DIR / "step7o_mean_summary.csv"
    paired_file = OUTPUT_DIR / "step7o_paired_comparison.csv"
    comparison_file = OUTPUT_DIR / "step7o_comparison_summary.json"

    all_df.to_csv(all_file, index=False)
    summary.to_csv(summary_file, index=False)
    paired_out.to_csv(paired_file, index=False)
    comparison_file.write_text(
        json.dumps(comparison, indent=4),
        encoding="utf-8",
    )

    print()
    print("=" * 80)
    print("STEP 7O-A COMPARISON SUMMARY")
    print("=" * 80)

    print(
        f"Mean return, FC-HMARL : "
        f"{comparison['mean_return_fc_hmarl']:.6f}"
    )
    print(
        f"Mean return, passive  : "
        f"{comparison['mean_return_passive']:.6f}"
    )
    print(
        f"Return improvement    : "
        f"{comparison['mean_return_delta_fc_minus_passive']:.6f}"
    )
    print(
        f"FC better episodes    : "
        f"{comparison['fc_better_return_episodes']}/{args.test_episodes}"
    )

    print()
    print(
        f"Grid import FC        : "
        f"{comparison['mean_grid_import_fc_hmarl_kwh']:.6f} kWh"
    )
    print(
        f"Grid import passive   : "
        f"{comparison['mean_grid_import_passive_kwh']:.6f} kWh"
    )
    print(
        f"Grid import reduction : "
        f"{comparison['grid_import_reduction_percent']:.4f}%"
    )

    print()
    print(
        f"Net market cost FC    : "
        f"${comparison['mean_net_market_cost_fc_hmarl_usd']:.6f}"
    )
    print(
        f"Net market cost passive: "
        f"${comparison['mean_net_market_cost_passive_usd']:.6f}"
    )
    print(
        f"Market-cost reduction : "
        f"{comparison['net_market_cost_reduction_percent']:.4f}%"
    )

    print()
    print(
        f"Peak import FC        : "
        f"{comparison['mean_peak_import_fc_hmarl_kw']:.6f} kW"
    )
    print(
        f"Peak import passive   : "
        f"{comparison['mean_peak_import_passive_kw']:.6f} kW"
    )
    print(
        f"Peak-import reduction : "
        f"{comparison['peak_import_reduction_percent']:.4f}%"
    )

    print()
    print(
        f"BESS throughput FC    : "
        f"{comparison['mean_bess_throughput_fc_hmarl_kwh']:.6f} kWh"
    )
    print(
        f"BESS throughput passive: "
        f"{comparison['mean_bess_throughput_passive_kwh']:.6f} kWh"
    )
    print(
        f"Sharing FC            : "
        f"{comparison['mean_sharing_fc_hmarl_kwh']:.6f} kWh"
    )
    print(
        f"Sharing passive       : "
        f"{comparison['mean_sharing_passive_kwh']:.6f} kWh"
    )

    print()
    print(
        "FC physical violations      : "
        f"{comparison['fc_power_balance_violation_steps']} balance, "
        f"{comparison['fc_transformer_violation_steps']} transformer"
    )
    print(
        "Passive physical violations : "
        f"{comparison['passive_power_balance_violation_steps']} balance, "
        f"{comparison['passive_transformer_violation_steps']} transformer"
    )

    print()
    print("[OK] Step 7O-A baseline comparison complete.")
    print("Episode results :", all_file)
    print("Mean summary    :", summary_file)
    print("Paired results  :", paired_file)
    print("Comparison JSON :", comparison_file)


if __name__ == "__main__":
    main()
