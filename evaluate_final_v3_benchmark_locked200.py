"""
STEP 7R-P1 — FINAL V3 BENCHMARK EVALUATION, LOCKED CHECKPOINT 200

Compares on the same 1,268 rolling 24-hour TEST windows:
1) Passive grid-only
2) Rule-based BESS EMS
3) Final FC-HMARL checkpoint 200

Protocol
--------
- TEST archive only
- checkpoint 200 only
- deterministic FC-HMARL actions
- no learning
- no replay writes
- no checkpoint ranking or re-selection
- same FINAL V3 physical environment and reward implementation
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from marl.training_loop import HierarchicalActionBundle


ROOT = Path(__file__).resolve().parent

V3_LAUNCHER = ROOT / "train_real_fc_hmarl_final_v3.py"
TEST_ARCHIVE = ROOT / "outputs" / "rl_data" / "real_rl_test_archive.npz"
CHECKPOINT_DIR = ROOT / "outputs" / "checkpoints" / "real_fc_hmarl_final_v3"
LOCKED_CHECKPOINT = 200

OUTPUT_DIR = (
    ROOT
    / "outputs"
    / "results"
    / "real_fc_hmarl_final_v3_benchmark"
)

NUMBER_OF_MGS = 5
EPISODE_LENGTH = 24

# Manuscript Table-2 BESS rated powers, kW.
BESS_RATED_POWER_KW = np.array(
    [250.0, 300.0, 250.0, 300.0, 350.0],
    dtype=float,
)

SOC_DISCHARGE_THRESHOLD = 0.50

MODES = (
    "passive_grid_only",
    "rule_based_bess_ems",
    "full_fc_hmarl",
)


def section(title):
    print()
    print("=" * 88)
    print(title)
    print("=" * 88)


def load_v3_module():
    spec = importlib.util.spec_from_file_location(
        "fc_hmarl_final_v3",
        V3_LAUNCHER,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not import FINAL V3 launcher.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestData:
    REQUIRED = (
        "current_actual_original",
        "current_actual_normalized",
        "forecast_original",
        "forecast_normalized",
        "predictive_state_matrix",
        "predictive_state_flat",
        "causal_confidence_24h",
        "test_indices",
        "source_split",
    )

    def __init__(self, path: Path):
        if not path.exists():
            raise FileNotFoundError(path)

        z = np.load(path, allow_pickle=True)

        missing = [k for k in self.REQUIRED if k not in z.files]
        if missing:
            raise RuntimeError(
                "TEST archive missing keys:\n  " + "\n  ".join(missing)
            )

        split = str(np.asarray(z["source_split"]).reshape(-1)[0]).lower()
        if split != "test":
            raise RuntimeError(
                f"Expected source_split='test', got {split!r}"
            )

        self.current_actual_original = np.asarray(
            z["current_actual_original"],
            dtype=np.float32,
        )
        self.current_actual_normalized = np.asarray(
            z["current_actual_normalized"],
            dtype=np.float32,
        )
        self.forecast_original = np.asarray(
            z["forecast_original"],
            dtype=np.float32,
        )
        self.forecast_normalized = np.asarray(
            z["forecast_normalized"],
            dtype=np.float32,
        )
        self.predictive_state_matrix = np.asarray(
            z["predictive_state_matrix"],
            dtype=np.float32,
        )
        self.predictive_state_flat = np.asarray(
            z["predictive_state_flat"],
            dtype=np.float32,
        )
        self.causal_confidence_24h = np.asarray(
            z["causal_confidence_24h"],
            dtype=np.float32,
        )
        self.test_indices = np.asarray(z["test_indices"])

        self.number_of_samples = len(self.current_actual_original)


class SequentialTestProvider:
    def __init__(self, v3, data: TestData, episodes: int):
        self.v3 = v3
        self.data = data
        self.starts = np.arange(
            data.number_of_samples - EPISODE_LENGTH + 1,
            dtype=int,
        )

        if episodes != len(self.starts):
            raise RuntimeError(
                f"Final benchmark requires all {len(self.starts)} TEST windows."
            )

        self.number_of_episodes = episodes

    def index(self, episode: int, step: int) -> int:
        return int(self.starts[episode - 1] + step)

    def __call__(self, episode: int, step: int):
        idx = self.index(episode, step)

        actual = self.data.current_actual_original[idx]

        pv_reference = float(actual[0])
        load_reference = float(actual[1])
        ev_reference = float(actual[2])
        price_reference = float(actual[3])

        irradiance = float(
            np.clip(pv_reference * 1000.0, 0.0, 1000.0)
        )
        irradiances = np.full(NUMBER_OF_MGS, irradiance, dtype=float)

        load_shape = self.v3.safe_fraction(
            load_reference,
            0.0,
            self.v3.TRAIN_LOAD_MAX,
        )
        loads = self.v3.PEAK_LOAD_KW * load_shape

        ev_shape = self.v3.safe_fraction(
            ev_reference,
            self.v3.TRAIN_EV_MIN,
            self.v3.TRAIN_EV_MAX,
        )

        max_ev_power = (
            self.v3.EV_COUNTS.astype(float)
            * self.v3.EV_CHARGER_POWER_KW
            * self.v3.EV_SIMULTANEOUS_FRACTION
        )

        aggregate_ev_requests = max_ev_power * ev_shape

        per_ev_requests = self.v3.build_per_ev_charging_requests(
            aggregate_ev_requests
        )

        price_shape = self.v3.safe_fraction(
            price_reference,
            self.v3.TRAIN_PRICE_MIN,
            self.v3.TRAIN_PRICE_MAX,
        )

        buy_price = (
            self.v3.MINIMUM_BUY_PRICE
            + price_shape
            * (
                self.v3.MAXIMUM_BUY_PRICE
                - self.v3.MINIMUM_BUY_PRICE
            )
        )

        sell_price = (
            self.v3.MINIMUM_SELL_PRICE
            + price_shape
            * (
                self.v3.MAXIMUM_SELL_PRICE
                - self.v3.MINIMUM_SELL_PRICE
            )
        )

        return self.v3.VPPExogenousInput(
            time=float(step),
            loads_kw=np.asarray(loads, dtype=float),
            irradiances_w_m2=irradiances,
            buy_prices_usd_per_kwh=np.full(NUMBER_OF_MGS, buy_price),
            sell_prices_usd_per_kwh=np.full(NUMBER_OF_MGS, sell_price),
            predictive_state=self.data.predictive_state_matrix[idx].copy(),
            confidence=float(self.data.causal_confidence_24h[0]),
            market_price=price_reference,
            ev_requested_charging_powers_kw=per_ev_requests,
        )


def verify_locked_checkpoint():
    expected = [
        CHECKPOINT_DIR / f"local_agent_{i}_episode_{LOCKED_CHECKPOINT}.pt"
        for i in range(1, 6)
    ]
    expected.append(
        CHECKPOINT_DIR / f"coordinator_episode_{LOCKED_CHECKPOINT}.pt"
    )

    missing = [p for p in expected if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Checkpoint 200 incomplete:\n"
            + "\n".join(str(p) for p in missing)
        )


def load_policy(v3, seed, device):
    v3.set_global_seed(seed)

    local_agents = v3.build_local_agents(
        seed=seed,
        smoke_test=False,
        device=device,
    )
    coordinator = v3.build_coordinator_agent(
        seed=seed,
        smoke_test=False,
        device=device,
    )

    for agent in local_agents:
        path = (
            CHECKPOINT_DIR
            / f"local_agent_{agent.microgrid_id}_episode_{LOCKED_CHECKPOINT}.pt"
        )
        agent.load(path, load_optimizers=False)

    coordinator.load(
        CHECKPOINT_DIR / f"coordinator_episode_{LOCKED_CHECKPOINT}.pt",
        load_optimizers=False,
    )

    return local_agents, coordinator


def full_actions(local_agents, coordinator, observation):
    local = [
        np.asarray(
            agent.select_action(state, deterministic=True),
            dtype=np.float32,
        )
        for agent, state in zip(
            local_agents,
            observation.local_states,
        )
    ]

    coord = np.asarray(
        coordinator.select_action(
            observation.coordinator_state,
            deterministic=True,
        ),
        dtype=np.float32,
    )

    return HierarchicalActionBundle(
        local_actions=local,
        coordinator_action=coord,
    )


def passive_actions():
    local = []

    for _ in range(NUMBER_OF_MGS):
        a = np.zeros(5, dtype=np.float32)
        a[0] = 0.0
        a[1:] = -1.0
        local.append(a)

    coord = np.array(
        [-1.0, -1.0, -1.0],
        dtype=np.float32,
    )

    return HierarchicalActionBundle(
        local_actions=local,
        coordinator_action=coord,
    )


def rule_based_actions(observation):
    local = []

    for i, state in enumerate(observation.local_states):
        s = np.asarray(state, dtype=float)

        soc = float(s[0])
        pv_kw = float(s[1])
        load_kw = float(s[2])
        ev_kw = float(s[3])

        net_demand_kw = load_kw + ev_kw - pv_kw
        rated_kw = float(BESS_RATED_POWER_KW[i])

        if net_demand_kw < 0.0:
            desired_charge_kw = min(
                abs(net_demand_kw),
                rated_kw,
            )
            command = -desired_charge_kw / rated_kw

        elif (
            net_demand_kw > 0.0
            and soc > SOC_DISCHARGE_THRESHOLD
        ):
            desired_discharge_kw = min(
                net_demand_kw,
                rated_kw,
            )
            command = desired_discharge_kw / rated_kw

        else:
            command = 0.0

        a = np.zeros(5, dtype=np.float32)
        a[0] = float(np.clip(command, -1.0, 1.0))
        a[1:] = -1.0
        local.append(a)

    coord = np.array(
        [-1.0, -1.0, -1.0],
        dtype=np.float32,
    )

    return HierarchicalActionBundle(
        local_actions=local,
        coordinator_action=coord,
    )


def evaluate_mode(
    *,
    mode,
    v3,
    data,
    provider,
    local_agents,
    coordinator,
    episodes,
):

    bridge = v3.build_real_training_bridge(
        data=data,
        maximum_episodes=episodes,
        seed=2026,
    )
    bridge.exogenous_provider = provider

    rows = []

    with torch.no_grad():

        for ep in range(1, episodes + 1):
            obs = bridge.reset(episode=ep)

            total_return = 0.0
            coordinator_return = 0.0
            local_return = np.zeros(NUMBER_OF_MGS, dtype=float)

            grid_import = 0.0
            grid_export = 0.0
            purchase_cost = 0.0
            sale_revenue = 0.0
            reserve_revenue = 0.0
            market_profit = 0.0

            reserve_requested = 0.0
            reserve_feasible = 0.0
            reserve_curtailed = 0.0

            bess_throughput = 0.0
            sharing = 0.0
            sharing_received = 0.0
            sharing_loss = 0.0

            grid_power = []

            balance_total = 0.0
            balance_max = 0.0
            balance_steps = 0

            transformer_steps = 0
            transformer_total = 0.0

            soc_low = 0
            soc_high = 0

            steps = 0

            for _ in range(EPISODE_LENGTH):

                if mode == "full_fc_hmarl":
                    actions = full_actions(
                        local_agents,
                        coordinator,
                        obs,
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

                reserve_requested += float(
                    totals.get("reserve_requested_kw", 0.0)
                )
                reserve_feasible += float(
                    totals.get("reserve_feasible_kw", 0.0)
                )
                reserve_curtailed += float(
                    totals.get("reserve_curtailed_kw", 0.0)
                )

                sharing += float(
                    totals["sharing_scheduled_kw"]
                )
                sharing_received += float(
                    totals.get("sharing_received_kw", 0.0)
                )
                sharing_loss += float(
                    totals["sharing_loss_kw"]
                )

                grid_power.append(
                    float(totals["grid_power_kw"])
                )

                step_balance = float(
                    constraints[
                        "total_power_balance_violation_kw"
                    ]
                )
                balance_total += step_balance
                balance_max = max(
                    balance_max,
                    step_balance,
                )

                if not constraints["all_power_balanced"]:
                    balance_steps += 1

                step_transformer = float(
                    constraints.get(
                        "total_transformer_violation_kw",
                        0.0,
                    )
                )
                transformer_total += step_transformer

                if not constraints[
                    "all_transformers_feasible"
                ]:
                    transformer_steps += 1

                for local in physical["local_results"]:
                    bess_throughput += float(
                        local["bess"][
                            "energy_throughput_kwh"
                        ]
                    )

                for mg in bridge.environment.microgrids:
                    soc = float(mg.bess.soc)
                    minimum_soc = float(mg.bess.minimum_soc)
                    maximum_soc = float(mg.bess.maximum_soc)

                    if soc < minimum_soc - 1e-9:
                        soc_low += 1
                    if soc > maximum_soc + 1e-9:
                        soc_high += 1

                obs = result.next_observation
                steps += 1

                if result.done:
                    break

            g = np.asarray(grid_power, dtype=float)

            rows.append({
                "mode": mode,
                "test_episode": ep,
                "start_index": int(
                    provider.starts[ep - 1]
                ),
                "steps": steps,

                "total_return": total_return,
                "coordinator_return": coordinator_return,
                "mean_local_return_per_mg": float(
                    local_return.mean()
                ),

                "grid_import_kwh": grid_import,
                "grid_export_kwh": grid_export,
                "grid_purchase_cost_usd": purchase_cost,
                "grid_sale_revenue_usd": sale_revenue,
                "reserve_revenue_usd": reserve_revenue,
                "net_market_cost_usd": (
                    purchase_cost
                    - sale_revenue
                    - reserve_revenue
                ),
                "market_profit_usd": market_profit,

                "reserve_requested_kwh": reserve_requested,
                "reserve_feasible_kwh": reserve_feasible,
                "reserve_curtailed_kwh": reserve_curtailed,

                "bess_throughput_kwh": bess_throughput,
                "sharing_scheduled_kwh": sharing,
                "sharing_received_kwh": sharing_received,
                "sharing_loss_kwh": sharing_loss,

                "peak_grid_import_kw": float(
                    max(0.0, np.max(g))
                ),
                "peak_grid_export_kw": float(
                    max(0.0, -np.min(g))
                ),

                "mean_balance_violation_kw_per_step": (
                    balance_total / max(steps, 1)
                ),
                "maximum_balance_violation_kw": balance_max,
                "power_balance_violation_steps": balance_steps,

                "total_transformer_violation_kw": transformer_total,
                "transformer_violation_steps": transformer_steps,

                "soc_below_min_count": soc_low,
                "soc_above_max_count": soc_high,
            })

            if (
                ep == 1
                or ep % 100 == 0
                or ep == episodes
            ):
                r = rows[-1]
                print(
                    f"{mode:22s} | "
                    f"{ep:4d}/{episodes} | "
                    f"return={r['total_return']:.6f} | "
                    f"net_cost=${r['net_market_cost_usd']:.2f} | "
                    f"balance={r['mean_balance_violation_kw_per_step']:.6f} | "
                    f"transformer={r['transformer_violation_steps']}"
                )

    return pd.DataFrame(rows)


def summary_for_mode(df, mode):
    m = df[df["mode"] == mode].copy()

    numeric_mean = [
        "total_return",
        "coordinator_return",
        "mean_local_return_per_mg",
        "grid_import_kwh",
        "grid_export_kwh",
        "grid_purchase_cost_usd",
        "grid_sale_revenue_usd",
        "reserve_revenue_usd",
        "net_market_cost_usd",
        "market_profit_usd",
        "reserve_requested_kwh",
        "reserve_feasible_kwh",
        "reserve_curtailed_kwh",
        "bess_throughput_kwh",
        "sharing_scheduled_kwh",
        "sharing_received_kwh",
        "sharing_loss_kwh",
        "peak_grid_import_kw",
        "peak_grid_export_kw",
        "mean_balance_violation_kw_per_step",
        "maximum_balance_violation_kw",
    ]

    row = {
        "mode": mode,
        "episodes": len(m),
        "mean_return": float(m["total_return"].mean()),
        "std_return": float(m["total_return"].std(ddof=0)),
        "median_return": float(m["total_return"].median()),
        "minimum_return": float(m["total_return"].min()),
        "maximum_return": float(m["total_return"].max()),
    }

    for col in numeric_mean:
        row[f"mean_{col}"] = float(m[col].mean())

    row["power_balance_violation_steps"] = int(
        m["power_balance_violation_steps"].sum()
    )
    row["transformer_violation_steps"] = int(
        m["transformer_violation_steps"].sum()
    )
    row["soc_below_min_count"] = int(
        m["soc_below_min_count"].sum()
    )
    row["soc_above_max_count"] = int(
        m["soc_above_max_count"].sum()
    )

    return row


def reduction_percent(reference, proposed):
    if abs(reference) < 1e-12:
        return np.nan
    return 100.0 * (
        reference - proposed
    ) / abs(reference)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "cuda", "auto"],
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2026,
    )
    args = parser.parse_args()

    section("STEP 7R-P1 — FINAL V3 BENCHMARK, LOCKED CHECKPOINT 200")

    verify_locked_checkpoint()

    v3 = load_v3_module()
    device = v3.resolve_device(args.device)

    data = TestData(TEST_ARCHIVE)

    episodes = data.number_of_samples - EPISODE_LENGTH + 1

    if episodes != 1268:
        raise RuntimeError(
            f"Expected 1268 TEST windows, found {episodes}."
        )

    provider = SequentialTestProvider(
        v3=v3,
        data=data,
        episodes=episodes,
    )

    local_agents, coordinator = load_policy(
        v3=v3,
        seed=args.seed,
        device=device,
    )

    print(f"Checkpoint               : {LOCKED_CHECKPOINT}")
    print(f"TEST windows             : {episodes}")
    print(f"Device                   : {device}")
    print("Deterministic FC-HMARL   : True")
    print("Learning                 : False")
    print("Replay writes            : False")
    print("Checkpoint ranking       : False")

    frames = []

    for mode in MODES:
        section(f"EVALUATING: {mode}")
        frames.append(
            evaluate_mode(
                mode=mode,
                v3=v3,
                data=data,
                provider=provider,
                local_agents=local_agents,
                coordinator=coordinator,
                episodes=episodes,
            )
        )

    results = pd.concat(
        frames,
        ignore_index=True,
    )

    summaries = pd.DataFrame(
        [
            summary_for_mode(
                results,
                mode,
            )
            for mode in MODES
        ]
    )

    passive = summaries[
        summaries["mode"] == "passive_grid_only"
    ].iloc[0]

    rule = summaries[
        summaries["mode"] == "rule_based_bess_ems"
    ].iloc[0]

    fc = summaries[
        summaries["mode"] == "full_fc_hmarl"
    ].iloc[0]

    comparison = {
        "locked_checkpoint": LOCKED_CHECKPOINT,
        "test_windows": episodes,
        "fc_vs_passive_return_delta": float(
            fc["mean_return"]
            - passive["mean_return"]
        ),
        "fc_vs_rule_return_delta": float(
            fc["mean_return"]
            - rule["mean_return"]
        ),
        "fc_vs_passive_grid_import_reduction_percent":
            reduction_percent(
                passive["mean_grid_import_kwh"],
                fc["mean_grid_import_kwh"],
            ),
        "fc_vs_rule_grid_import_reduction_percent":
            reduction_percent(
                rule["mean_grid_import_kwh"],
                fc["mean_grid_import_kwh"],
            ),
        "fc_vs_passive_market_cost_reduction_percent":
            reduction_percent(
                passive["mean_net_market_cost_usd"],
                fc["mean_net_market_cost_usd"],
            ),
        "fc_vs_rule_market_cost_reduction_percent":
            reduction_percent(
                rule["mean_net_market_cost_usd"],
                fc["mean_net_market_cost_usd"],
            ),
        "fc_vs_passive_peak_import_reduction_percent":
            reduction_percent(
                passive["mean_peak_grid_import_kw"],
                fc["mean_peak_grid_import_kw"],
            ),
        "fc_vs_rule_peak_import_reduction_percent":
            reduction_percent(
                rule["mean_peak_grid_import_kw"],
                fc["mean_peak_grid_import_kw"],
            ),
    }

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    episode_file = (
        OUTPUT_DIR
        / "final_v3_benchmark_episode_results.csv"
    )

    summary_file = (
        OUTPUT_DIR
        / "final_v3_benchmark_summary.csv"
    )

    comparison_file = (
        OUTPUT_DIR
        / "final_v3_benchmark_comparison.json"
    )

    protocol_file = (
        OUTPUT_DIR
        / "final_v3_benchmark_protocol.json"
    )

    results.to_csv(
        episode_file,
        index=False,
    )

    summaries.to_csv(
        summary_file,
        index=False,
    )

    comparison_file.write_text(
        json.dumps(
            comparison,
            indent=4,
        ),
        encoding="utf-8",
    )

    protocol = {
        "checkpoint": LOCKED_CHECKPOINT,
        "checkpoint_selection_source": "validation",
        "test_split_role": "evaluation only",
        "test_windows": episodes,
        "modes": list(MODES),
        "rule_based_benchmark_definition": (
            "No sharing, no reserve; charge on PV surplus; "
            "discharge on deficit when SOC > 0.50; grid closes balance."
        ),
        "passive_definition": (
            "BESS idle, no sharing, no reserve; grid closes balance."
        ),
        "deterministic_fc_hmarl": True,
        "learning_disabled": True,
        "replay_writes_disabled": True,
        "checkpoint_ranking_disabled": True,
        "checkpoint_reselection_disabled": True,
    }

    protocol_file.write_text(
        json.dumps(
            protocol,
            indent=4,
        ),
        encoding="utf-8",
    )

    section("FINAL BENCHMARK SUMMARY")

    display = [
        "mode",
        "mean_return",
        "std_return",
        "mean_grid_import_kwh",
        "mean_grid_export_kwh",
        "mean_net_market_cost_usd",
        "mean_peak_grid_import_kw",
        "mean_bess_throughput_kwh",
        "mean_sharing_scheduled_kwh",
        "mean_reserve_feasible_kwh",
        "power_balance_violation_steps",
        "transformer_violation_steps",
        "soc_below_min_count",
        "soc_above_max_count",
    ]

    print(
        summaries[display].to_string(
            index=False
        )
    )

    print()
    print("FC-HMARL vs passive:")
    print(
        "  Return delta             : "
        f"{comparison['fc_vs_passive_return_delta']:.6f}"
    )
    print(
        "  Grid import reduction    : "
        f"{comparison['fc_vs_passive_grid_import_reduction_percent']:.4f}%"
    )
    print(
        "  Market cost reduction    : "
        f"{comparison['fc_vs_passive_market_cost_reduction_percent']:.4f}%"
    )
    print(
        "  Peak import reduction    : "
        f"{comparison['fc_vs_passive_peak_import_reduction_percent']:.4f}%"
    )

    print()
    print("FC-HMARL vs rule-based EMS:")
    print(
        "  Return delta             : "
        f"{comparison['fc_vs_rule_return_delta']:.6f}"
    )
    print(
        "  Grid import reduction    : "
        f"{comparison['fc_vs_rule_grid_import_reduction_percent']:.4f}%"
    )
    print(
        "  Market cost reduction    : "
        f"{comparison['fc_vs_rule_market_cost_reduction_percent']:.4f}%"
    )
    print(
        "  Peak import reduction    : "
        f"{comparison['fc_vs_rule_peak_import_reduction_percent']:.4f}%"
    )

    section("STEP 7R-P1 COMPLETE")
    print("Episode results :", episode_file)
    print("Summary         :", summary_file)
    print("Comparison JSON :", comparison_file)
    print("Protocol        :", protocol_file)
    print()
    print("[OK] Same 1268 TEST windows used for all three controllers.")
    print("[OK] FINAL V3 physical environment used.")
    print("[OK] Checkpoint 200 only.")
    print("[OK] No learning or checkpoint re-selection performed.")


if __name__ == "__main__":
    main()
