from __future__ import annotations
import copy
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import evaluate_final_v3_benchmark_locked200 as bench
ROOT = Path(__file__).resolve().parent
LOCKED_SUMMARY = (
    ROOT
    / "outputs"
    / "results"
    / "real_fc_hmarl_final_v3_test"
    / "final_test_summary.csv"
)

OUTPUT_DIR = (
    ROOT
    / "outputs"
    / "results"
    / "real_fc_hmarl_final_v3_ablation"
)

MODES = (
    "full_fc_hmarl",
    "no_confidence_awareness",
    "no_energy_sharing",
    "no_upper_level_coordination",
)


def section(title):
    print()
    print("=" * 92)
    print(title)
    print("=" * 92)


def build_mode_data(base_data, mode):
    data = copy.copy(base_data)

    if mode == "no_confidence_awareness":
        data.causal_confidence_24h = np.ones_like(
            base_data.causal_confidence_24h,
            dtype=np.float32,
        )

        data.predictive_state_matrix = np.asarray(
            base_data.forecast_normalized,
            dtype=np.float32,
        ).copy()

        data.predictive_state_flat = (
            data.predictive_state_matrix
            .reshape(
                len(data.predictive_state_matrix),
                -1,
            )
            .astype(np.float32)
        )

        if data.predictive_state_flat.shape[1] != 96:
            raise RuntimeError(
                "Expected 96-dimensional predictive state."
            )

    return data


def modified_actions(
    *,
    mode,
    local_agents,
    coordinator,
    observation,
):
    original = bench.full_actions(
        local_agents,
        coordinator,
        observation,
    )

    local_actions = [
        np.asarray(a, dtype=np.float32).copy()
        for a in original.local_actions
    ]

    coordinator_action = np.asarray(
        original.coordinator_action,
        dtype=np.float32,
    ).copy()

    if mode in (
        "full_fc_hmarl",
        "no_confidence_awareness",
    ):
        pass

    elif mode == "no_energy_sharing":

        for action in local_actions:
            if action.size > 1:
                action[1:] = -1.0

        if coordinator_action.size >= 3:
            coordinator_action[2] = -1.0

    elif mode == "no_upper_level_coordination":

        coordinator_action[:] = 0.0
        if coordinator_action.size >= 1:
            coordinator_action[0] = -1.0

        if coordinator_action.size >= 2:
            coordinator_action[1] = -1.0

        if coordinator_action.size >= 3:
            coordinator_action[2] = 1.0

    else:
        raise ValueError(mode)

    return bench.HierarchicalActionBundle(
        local_actions=local_actions,
        coordinator_action=coordinator_action,
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

            obs = bridge.reset(
                episode=ep
            )

            total_return = 0.0
            coordinator_return = 0.0
            local_return = np.zeros(
                bench.NUMBER_OF_MGS,
                dtype=float,
            )

            grid_import = 0.0
            grid_export = 0.0
            purchase_cost = 0.0
            sale_revenue = 0.0
            reserve_revenue = 0.0
            market_profit = 0.0

            reserve_feasible = 0.0
            bess_throughput = 0.0
            sharing = 0.0
            sharing_loss = 0.0

            balance_total = 0.0
            balance_steps = 0
            transformer_steps = 0
            soc_low = 0
            soc_high = 0

            grid_power = []
            steps = 0

            for _ in range(
                bench.EPISODE_LENGTH
            ):

                actions = modified_actions(
                    mode=mode,
                    local_agents=local_agents,
                    coordinator=coordinator,
                    observation=obs,
                )

                result = bridge.step(
                    actions
                )

                lr = np.asarray(
                    result.local_rewards,
                    dtype=float,
                )

                cr = float(
                    result.coordinator_reward
                )

                total_return += (
                    float(lr.sum())
                    + cr
                )

                local_return += lr
                coordinator_return += cr

                physical = (
                    result.info[
                        "physical_result"
                    ]
                )

                totals = physical[
                    "totals"
                ]

                constraints = physical[
                    "constraints"
                ]

                grid_import += float(
                    totals[
                        "grid_import_kw"
                    ]
                )

                grid_export += float(
                    totals[
                        "grid_export_kw"
                    ]
                )

                purchase_cost += float(
                    totals[
                        "grid_purchase_cost_usd"
                    ]
                )

                sale_revenue += float(
                    totals[
                        "grid_sale_revenue_usd"
                    ]
                )

                reserve_revenue += float(
                    totals[
                        "reserve_revenue_usd"
                    ]
                )

                market_profit += float(
                    totals[
                        "market_profit_usd"
                    ]
                )

                reserve_feasible += float(
                    totals.get(
                        "reserve_feasible_kw",
                        0.0,
                    )
                )

                sharing += float(
                    totals[
                        "sharing_scheduled_kw"
                    ]
                )

                sharing_loss += float(
                    totals[
                        "sharing_loss_kw"
                    ]
                )

                grid_power.append(
                    float(
                        totals[
                            "grid_power_kw"
                        ]
                    )
                )

                balance = float(
                    constraints[
                        "total_power_balance_violation_kw"
                    ]
                )

                balance_total += balance

                if not constraints[
                    "all_power_balanced"
                ]:
                    balance_steps += 1

                if not constraints[
                    "all_transformers_feasible"
                ]:
                    transformer_steps += 1

                for local in physical[
                    "local_results"
                ]:
                    bess_throughput += float(
                        local["bess"][
                            "energy_throughput_kwh"
                        ]
                    )

                for mg in (
                    bridge
                    .environment
                    .microgrids
                ):
                    soc = float(
                        mg.bess.soc
                    )

                    if soc < (
                        float(
                            mg.bess.minimum_soc
                        )
                        - 1e-9
                    ):
                        soc_low += 1

                    if soc > (
                        float(
                            mg.bess.maximum_soc
                        )
                        + 1e-9
                    ):
                        soc_high += 1

                obs = (
                    result.next_observation
                )

                steps += 1

                if result.done:
                    break

            g = np.asarray(
                grid_power,
                dtype=float,
            )

            row = {
                "mode":
                    mode,

                "test_episode":
                    ep,

                "start_index":
                    int(
                        provider.starts[
                            ep - 1
                        ]
                    ),

                "steps":
                    steps,

                "total_return":
                    total_return,

                "coordinator_return":
                    coordinator_return,

                "mean_local_return_per_mg":
                    float(
                        local_return.mean()
                    ),

                "grid_import_kwh":
                    grid_import,

                "grid_export_kwh":
                    grid_export,

                "grid_purchase_cost_usd":
                    purchase_cost,

                "grid_sale_revenue_usd":
                    sale_revenue,

                "reserve_revenue_usd":
                    reserve_revenue,

                "net_market_cost_usd":
                    (
                        purchase_cost
                        - sale_revenue
                        - reserve_revenue
                    ),

                "market_profit_usd":
                    market_profit,

                "reserve_feasible_kwh":
                    reserve_feasible,

                "bess_throughput_kwh":
                    bess_throughput,

                "sharing_scheduled_kwh":
                    sharing,

                "sharing_loss_kwh":
                    sharing_loss,

                "peak_grid_import_kw":
                    float(
                        max(
                            0.0,
                            np.max(g),
                        )
                    ),

                "peak_grid_export_kw":
                    float(
                        max(
                            0.0,
                            -np.min(g),
                        )
                    ),

                "mean_balance_violation_kw_per_step":
                    float(
                        balance_total
                        / max(
                            steps,
                            1,
                        )
                    ),

                "power_balance_violation_steps":
                    balance_steps,

                "transformer_violation_steps":
                    transformer_steps,

                "soc_below_min_count":
                    soc_low,

                "soc_above_max_count":
                    soc_high,
            }

            rows.append(
                row
            )

            if (
                ep == 1
                or ep % 100 == 0
                or ep == episodes
            ):
                print(
                    f"{mode:30s} | "
                    f"{ep:4d}/{episodes} | "
                    f"return="
                    f"{total_return:.6f} | "
                    f"net_cost="
                    f"${row['net_market_cost_usd']:.2f}"
                )

    return pd.DataFrame(
        rows
    )


def summarize_mode(
    mode_df,
):

    returns = (
        mode_df[
            "total_return"
        ]
        .to_numpy(
            dtype=float
        )
    )

    return {
        "mode":
            str(
                mode_df[
                    "mode"
                ].iloc[0]
            ),

        "episodes":
            int(
                len(mode_df)
            ),

        "mean_return":
            float(
                np.mean(
                    returns
                )
            ),

        "std_return":
            float(
                np.std(
                    returns,
                    ddof=0,
                )
            ),

        "median_return":
            float(
                np.median(
                    returns
                )
            ),

        "minimum_return":
            float(
                np.min(
                    returns
                )
            ),

        "maximum_return":
            float(
                np.max(
                    returns
                )
            ),

        "mean_net_market_cost_usd":
            float(
                mode_df[
                    "net_market_cost_usd"
                ].mean()
            ),

        "mean_grid_import_kwh":
            float(
                mode_df[
                    "grid_import_kwh"
                ].mean()
            ),

        "mean_grid_export_kwh":
            float(
                mode_df[
                    "grid_export_kwh"
                ].mean()
            ),

        "mean_peak_grid_import_kw":
            float(
                mode_df[
                    "peak_grid_import_kw"
                ].mean()
            ),

        "mean_bess_throughput_kwh":
            float(
                mode_df[
                    "bess_throughput_kwh"
                ].mean()
            ),

        "mean_sharing_scheduled_kwh":
            float(
                mode_df[
                    "sharing_scheduled_kwh"
                ].mean()
            ),

        "mean_reserve_feasible_kwh":
            float(
                mode_df[
                    "reserve_feasible_kwh"
                ].mean()
            ),

        "power_balance_violation_steps":
            int(
                mode_df[
                    "power_balance_violation_steps"
                ].sum()
            ),

        "transformer_violation_steps":
            int(
                mode_df[
                    "transformer_violation_steps"
                ].sum()
            ),

        "soc_below_min_count":
            int(
                mode_df[
                    "soc_below_min_count"
                ].sum()
            ),

        "soc_above_max_count":
            int(
                mode_df[
                    "soc_above_max_count"
                ].sum()
            ),
    }


def paired_comparison(
    full_df,
    alternative_df,
):

    full = (
        full_df
        .sort_values(
            "test_episode"
        )[
            "total_return"
        ]
        .to_numpy(
            dtype=float
        )
    )

    alt = (
        alternative_df
        .sort_values(
            "test_episode"
        )[
            "total_return"
        ]
        .to_numpy(
            dtype=float
        )
    )

    delta = (
        alt
        - full
    )

    return {
        "comparison":
            (
                f"{alternative_df['mode'].iloc[0]}"
                "_minus_full"
            ),

        "mean_delta_return":
            float(
                np.mean(
                    delta
                )
            ),

        "median_delta_return":
            float(
                np.median(
                    delta
                )
            ),

        "std_delta_return":
            float(
                np.std(
                    delta,
                    ddof=0,
                )
            ),

        "alternative_better_windows":
            int(
                np.sum(
                    delta > 0.0
                )
            ),

        "full_better_windows":
            int(
                np.sum(
                    delta < 0.0
                )
            ),

        "equal_windows":
            int(
                np.sum(
                    np.isclose(
                        delta,
                        0.0,
                        atol=1e-12,
                    )
                )
            ),
    }


def main():

    section(
        "STEP 7R-P2 — FINAL V3 POST-TRAINING ABLATION"
    )

    bench.verify_locked_checkpoint()

    v3 = bench.load_v3_module()

    device = v3.resolve_device(
        "cpu"
    )

    base_data = bench.TestData(
        bench.TEST_ARCHIVE
    )

    episodes = (
        base_data.number_of_samples
        - bench.EPISODE_LENGTH
        + 1
    )

    if episodes != 1268:
        raise RuntimeError(
            f"Expected 1268 TEST windows, got {episodes}."
        )

    local_agents, coordinator = (
        bench.load_policy(
            v3=v3,
            seed=2026,
            device=device,
        )
    )

    frames = []

    for mode in MODES:

        section(
            f"ABLATION MODE: {mode}"
        )

        mode_data = build_mode_data(
            base_data,
            mode,
        )

        provider = (
            bench.SequentialTestProvider(
                v3=v3,
                data=mode_data,
                episodes=episodes,
            )
        )

        frames.append(
            evaluate_mode(
                mode=mode,
                v3=v3,
                data=mode_data,
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

    summary = pd.DataFrame(
        [
            summarize_mode(
                results[
                    results["mode"]
                    == mode
                ]
            )
            for mode in MODES
        ]
    )

    full_df = results[
        results["mode"]
        == "full_fc_hmarl"
    ]

    paired = pd.DataFrame(
        [
            paired_comparison(
                full_df,
                results[
                    results["mode"]
                    == mode
                ],
            )
            for mode in MODES[1:]
        ]
    )

    if not LOCKED_SUMMARY.exists():
        raise FileNotFoundError(
            LOCKED_SUMMARY
        )

    locked = pd.read_csv(
        LOCKED_SUMMARY
    ).iloc[0]

    locked_mean = float(
        locked[
            "mean_return"
        ]
    )

    reproduced_mean = float(
        summary.loc[
            summary["mode"]
            == "full_fc_hmarl",
            "mean_return",
        ].iloc[0]
    )

    reproduction_difference = (
        reproduced_mean
        - locked_mean
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    episode_file = (
        OUTPUT_DIR
        / "final_v3_ablation_episode_results.csv"
    )

    summary_file = (
        OUTPUT_DIR
        / "final_v3_ablation_summary.csv"
    )

    paired_file = (
        OUTPUT_DIR
        / "final_v3_ablation_paired_comparisons.csv"
    )

    metadata_file = (
        OUTPUT_DIR
        / "final_v3_ablation_metadata.json"
    )

    results.to_csv(
        episode_file,
        index=False,
    )

    summary.to_csv(
        summary_file,
        index=False,
    )

    paired.to_csv(
        paired_file,
        index=False,
    )

    metadata = {
        "checkpoint":
            200,

        "test_windows":
            1268,

        "ablation_type":
            "post_training_inference_diagnostic",

        "deterministic_actions":
            True,

        "learning_disabled":
            True,

        "test_used_for_checkpoint_selection":
            False,

        "full_locked_test_mean":
            locked_mean,

        "full_reproduced_mean":
            reproduced_mean,

        "full_reproduction_difference":
            reproduction_difference,

        "no_confidence_awareness_definition":
            (
                "S_pred = Z_hat and Phi = 1 "
                "instead of S_pred = Phi*Z_hat."
            ),

        "no_energy_sharing_definition":
            (
                "All local sharing commands and "
                "coordinator sharing multiplier disabled."
            ),

        "no_upper_level_coordination_definition":
            (
                "Trained local policies retained; "
                "coordinator action replaced by fixed "
                "non-adaptive action with reserve disabled "
                "and sharing multiplier fixed to pass local sharing."
            ),

        "reporting_warning":
            (
                "These are inference-time diagnostic ablations "
                "of one trained policy, not separately retrained "
                "ablation controllers."
            ),
    }

    metadata_file.write_text(
        json.dumps(
            metadata,
            indent=4,
        ),
        encoding="utf-8",
    )

    section(
        "ABLATION SUMMARY"
    )

    display = [
        "mode",
        "mean_return",
        "std_return",
        "mean_net_market_cost_usd",
        "mean_grid_import_kwh",
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
        summary[
            display
        ].to_string(
            index=False
        )
    )

    section(
        "PAIRED COMPARISONS VS FULL FC-HMARL"
    )

    print(
        paired.to_string(
            index=False
        )
    )

    section(
        "FULL-MODE REPRODUCTION CHECK"
    )

    print(
        f"Locked final TEST mean : "
        f"{locked_mean:.6f}"
    )

    print(
        f"Reproduced full mean   : "
        f"{reproduced_mean:.6f}"
    )

    print(
        f"Difference             : "
        f"{reproduction_difference:.12e}"
    )

    if abs(
        reproduction_difference
    ) > 1e-6:
        raise RuntimeError(
            "Full-mode ablation run does not reproduce "
            "the locked final TEST mean. "
            "Do not interpret ablation results."
        )

    print(
        "[OK] Full mode exactly reproduces the locked TEST result."
    )

    section(
        "STEP 7R-P2 COMPLETE"
    )

    print(
        "Episode results :",
        episode_file,
    )

    print(
        "Summary         :",
        summary_file,
    )

    print(
        "Paired results  :",
        paired_file,
    )

    print(
        "Metadata        :",
        metadata_file,
    )

    print()
    print(
        "[OK] Same 1268 TEST windows used for every ablation mode."
    )
    print(
        "[OK] Checkpoint 200 only."
    )
    print(
        "[OK] No training, replay write, or checkpoint selection occurred."
    )
    print(
        "[IMPORTANT] Report these as post-training inference ablations, "
        "not retrained ablation agents."
    )


if __name__ == "__main__":
    main()
