"""
Step 7N-B: Operational, economic, and risk evaluation of locked FC-HMARL
checkpoint 1000 on the same 30 held-out TEST episodes.

No learning, no checkpoint selection, no TEST-driven tuning.
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
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "evaluation" / "step7n_operational_economic"
LOCKED_CHECKPOINT = 1000
DEGRADATION_USD_PER_KWH = 0.02
RISK_AVERSION = 1.0
CVAR_ALPHA = 0.95


def empirical_upper_cvar(values, alpha=0.95):
    x = np.asarray(values, dtype=float).reshape(-1)
    if x.size == 0:
        return float("nan"), float("nan")
    var = float(np.quantile(x, alpha))
    tail = x[x >= var]
    return var, float(tail.mean()) if tail.size else var


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=int, default=1000)
    p.add_argument("--test-episodes", type=int, default=30)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cpu")
    return p.parse_args()


def main():
    args = parse_args()
    if args.checkpoint != LOCKED_CHECKPOINT:
        raise ValueError("Final TEST evaluation is locked to checkpoint 1000.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("STEP 7N-B - FC-HMARL OPERATIONAL / ECONOMIC / RISK EVALUATION")
    print("=" * 80)
    print(f"Checkpoint    : {args.checkpoint}")
    print(f"TEST episodes : {args.test_episodes}")
    print(f"Seed          : {args.seed}")
    print("Actions       : deterministic")
    print("Learning      : disabled")
    print("TEST tuning   : disabled")

    data = t.TestData()
    starts = np.asarray(
        t.build_fixed_test_episode_starts(
            data=data,
            number_of_episodes=args.test_episodes,
            seed=args.seed,
        ),
        dtype=int,
    )

    bridge = t.build_test_bridge(data=data, episode_starts=starts)
    local_agents, coordinator_agent = t.val.load_policy(
        checkpoint_episode=args.checkpoint,
        device=args.device,
        seed=args.seed,
    )

    episode_rows = []
    step_rows = []

    for ep in range(1, args.test_episodes + 1):
        obs = bridge.reset(episode=ep)

        sums = {
            "load_kwh": 0.0, "pv_kwh": 0.0, "ev_kwh": 0.0,
            "bess_net_kwh": 0.0, "bess_throughput_kwh": 0.0,
            "grid_import_kwh": 0.0, "grid_export_kwh": 0.0,
            "grid_purchase_cost_usd": 0.0, "grid_sale_revenue_usd": 0.0,
            "reserve_revenue_usd": 0.0, "market_profit_usd": 0.0,
            "sharing_scheduled_kwh": 0.0, "sharing_received_kwh": 0.0,
            "sharing_loss_kwh": 0.0,
            "power_balance_violation_kw_sum": 0.0,
            "transformer_violation_kw_sum": 0.0,
            "total_return": 0.0,
            "coordinator_return": 0.0,
        }
        local_return = np.zeros(t.NUMBER_OF_MICROGRIDS, dtype=float)
        grid_series = []
        confidence_series = []

        with torch.no_grad():
            for hour in range(t.EPISODE_LENGTH):
                actions = t.val.select_deterministic_actions(
                    local_agents=local_agents,
                    coordinator_agent=coordinator_agent,
                    observation=obs,
                )
                result = bridge.step(actions)
                physical = result.info["physical_result"]
                totals = physical["totals"]
                constraints = physical["constraints"]

                local_rewards = np.asarray(result.local_rewards, dtype=float)
                coord_reward = float(result.coordinator_reward)
                local_return += local_rewards
                sums["total_return"] += float(local_rewards.sum()) + coord_reward
                sums["coordinator_return"] += coord_reward

                # Hourly simulation => kW * 1 h = kWh.
                sums["load_kwh"] += float(totals["load_power_kw"])
                sums["pv_kwh"] += float(totals["pv_power_kw"])
                sums["ev_kwh"] += float(totals["ev_power_kw"])
                sums["bess_net_kwh"] += float(totals["bess_power_kw"])
                sums["grid_import_kwh"] += float(totals["grid_import_kw"])
                sums["grid_export_kwh"] += float(totals["grid_export_kw"])
                sums["grid_purchase_cost_usd"] += float(totals["grid_purchase_cost_usd"])
                sums["grid_sale_revenue_usd"] += float(totals["grid_sale_revenue_usd"])
                sums["reserve_revenue_usd"] += float(totals["reserve_revenue_usd"])
                sums["market_profit_usd"] += float(totals["market_profit_usd"])
                sums["sharing_scheduled_kwh"] += float(totals["sharing_scheduled_kw"])
                sums["sharing_received_kwh"] += float(totals["sharing_received_kw"])
                sums["sharing_loss_kwh"] += float(totals["sharing_loss_kw"])
                sums["power_balance_violation_kw_sum"] += float(
                    constraints["total_power_balance_violation_kw"]
                )
                sums["transformer_violation_kw_sum"] += float(
                    constraints["total_transformer_violation_kw"]
                )

                throughput = 0.0
                for local in physical["local_results"]:
                    throughput += float(local["bess"]["energy_throughput_kwh"])
                sums["bess_throughput_kwh"] += throughput

                grid_series.append(float(totals["grid_power_kw"]))
                confidence = float(result.info.get("confidence", data.phi[0]))
                confidence_series.append(confidence)

                step_rows.append({
                    "test_episode": ep,
                    "start_index": int(starts[ep - 1]),
                    "hour": hour + 1,
                    "grid_power_kw": float(totals["grid_power_kw"]),
                    "grid_import_kw": float(totals["grid_import_kw"]),
                    "grid_export_kw": float(totals["grid_export_kw"]),
                    "load_kw": float(totals["load_power_kw"]),
                    "pv_kw": float(totals["pv_power_kw"]),
                    "ev_kw": float(totals["ev_power_kw"]),
                    "bess_kw": float(totals["bess_power_kw"]),
                    "sharing_scheduled_kw": float(totals["sharing_scheduled_kw"]),
                    "sharing_loss_kw": float(totals["sharing_loss_kw"]),
                    "market_profit_usd": float(totals["market_profit_usd"]),
                    "confidence": confidence,
                })

                obs = result.next_observation
                if result.done:
                    break

        degradation = sums["bess_throughput_kwh"] * DEGRADATION_USD_PER_KWH
        risk_cost = sum(RISK_AVERSION * (1.0 - c) for c in confidence_series)
        net_market_cost = (
            sums["grid_purchase_cost_usd"]
            - sums["grid_sale_revenue_usd"]
            - sums["reserve_revenue_usd"]
        )
        operating_cost = net_market_cost + degradation + risk_cost

        g = np.asarray(grid_series, dtype=float)
        peak_import = float(max(0.0, np.max(g)))
        peak_export = float(max(0.0, -np.min(g)))
        mean_abs_grid = float(np.mean(np.abs(g)))
        par = float(np.max(np.abs(g)) / mean_abs_grid) if mean_abs_grid > 0 else 0.0

        demand = sums["load_kwh"] + sums["ev_kwh"]
        renewable_fraction = sums["pv_kwh"] / demand if demand > 0 else 0.0

        episode_rows.append({
            "test_episode": ep,
            "start_index": int(starts[ep - 1]),
            **sums,
            "bess_degradation_cost_usd": degradation,
            "risk_cost_usd": risk_cost,
            "net_market_cost_usd": net_market_cost,
            "operating_cost_usd": operating_cost,
            "peak_grid_import_kw": peak_import,
            "peak_grid_export_kw": peak_export,
            "mean_absolute_grid_exchange_kw": mean_abs_grid,
            "grid_peak_to_average_ratio": par,
            "renewable_fraction_of_load_plus_ev": renewable_fraction,
            "mean_confidence": float(np.mean(confidence_series)),
            "mean_local_return_per_mg": float(local_return.mean()),
        })

        print(
            f"Episode {ep:02d} | return {sums['total_return']: .6f} | "
            f"import {sums['grid_import_kwh']:9.2f} kWh | "
            f"op.cost ${operating_cost:9.2f}"
        )

    ep_df = pd.DataFrame(episode_rows)
    step_df = pd.DataFrame(step_rows)

    # Risk across episode operating costs: upper tail = adverse high-cost episodes.
    var95, cvar95 = empirical_upper_cvar(ep_df["operating_cost_usd"], CVAR_ALPHA)

    numeric_cols = ep_df.select_dtypes(include=[np.number]).columns
    summary_rows = []
    for col in numeric_cols:
        if col in ("test_episode", "start_index"):
            continue
        x = ep_df[col].to_numpy(dtype=float)
        summary_rows.append({
            "metric": col,
            "mean": float(np.mean(x)),
            "std": float(np.std(x, ddof=0)),
            "median": float(np.median(x)),
            "minimum": float(np.min(x)),
            "maximum": float(np.max(x)),
        })
    summary_df = pd.DataFrame(summary_rows)

    ep_file = OUTPUT_DIR / "step7n_episode_metrics.csv"
    step_file = OUTPUT_DIR / "step7n_hourly_metrics.csv"
    summary_file = OUTPUT_DIR / "step7n_summary_metrics.csv"
    risk_file = OUTPUT_DIR / "step7n_risk_summary.json"

    ep_df.to_csv(ep_file, index=False)
    step_df.to_csv(step_file, index=False)
    summary_df.to_csv(summary_file, index=False)

    risk_summary = {
        "checkpoint": LOCKED_CHECKPOINT,
        "test_episodes": int(args.test_episodes),
        "cvar_alpha": CVAR_ALPHA,
        "operating_cost_var95_usd": var95,
        "operating_cost_cvar95_usd": cvar95,
        "mean_operating_cost_usd": float(ep_df["operating_cost_usd"].mean()),
        "mean_market_profit_usd": float(ep_df["market_profit_usd"].mean()),
        "mean_bess_degradation_cost_usd": float(
            ep_df["bess_degradation_cost_usd"].mean()
        ),
        "mean_grid_import_kwh": float(ep_df["grid_import_kwh"].mean()),
        "mean_grid_export_kwh": float(ep_df["grid_export_kwh"].mean()),
        "mean_sharing_scheduled_kwh": float(ep_df["sharing_scheduled_kwh"].mean()),
        "total_power_balance_violation_kw": float(
            ep_df["power_balance_violation_kw_sum"].sum()
        ),
        "total_transformer_violation_kw": float(
            ep_df["transformer_violation_kw_sum"].sum()
        ),
        "notes": [
            "Checkpoint 1000 was selected using validation only.",
            "Held-out TEST is used only for final evaluation.",
            "BESS degradation coefficient is 0.02 USD/kWh.",
            "CVaR is an empirical upper-tail operating-cost diagnostic.",
            "Risk cost uses reconstructed rho=1 with manuscript form rho*(1-Phi).",
        ],
    }
    risk_file.write_text(json.dumps(risk_summary, indent=4), encoding="utf-8")

    print()
    print("=" * 80)
    print("STEP 7N SUMMARY")
    print("=" * 80)
    keys = [
        "total_return", "operating_cost_usd", "market_profit_usd",
        "grid_import_kwh", "grid_export_kwh", "bess_throughput_kwh",
        "bess_degradation_cost_usd", "sharing_scheduled_kwh",
        "sharing_loss_kwh", "peak_grid_import_kw",
        "renewable_fraction_of_load_plus_ev",
    ]
    print(summary_df[summary_df["metric"].isin(keys)].to_string(index=False))
    print()
    print(f"Operating-cost VaR95  : ${var95:.6f}")
    print(f"Operating-cost CVaR95 : ${cvar95:.6f}")
    print(
        "Power-balance violation total : "
        f"{ep_df['power_balance_violation_kw_sum'].sum():.12f} kW"
    )
    print(
        "Transformer violation total   : "
        f"{ep_df['transformer_violation_kw_sum'].sum():.12f} kW"
    )
    print()
    print("[OK] Step 7N operational/economic/risk evaluation complete.")
    print("Episode file :", ep_file)
    print("Hourly file  :", step_file)
    print("Summary file :", summary_file)
    print("Risk file    :", risk_file)


if __name__ == "__main__":
    main()
