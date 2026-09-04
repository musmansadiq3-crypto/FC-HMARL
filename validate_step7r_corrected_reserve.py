"""
Step 7R-D: Re-run the representative FC-HMARL physical audit after
the corrected reserve-feasibility model has been installed.

IMPORTANT
---------
This is still a DIAGNOSTIC on the OLD checkpoint-1000 policy.
It is NOT a valid final economic evaluation because the policy was
trained before the reserve model was corrected.

Purpose
-------
Verify that the corrected environment now enforces:
1) reserve <= instantaneous discharge-power headroom,
2) reserve <= SOC/energy-based reserve capability,
3) zero reserve at minimum SOC,
4) no reserve-headroom excess,
5) power-balance and transformer feasibility remain intact.

Run from project root:
    python validate_step7r_corrected_reserve.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import evaluate_real_fc_hmarl_test as t


PROJECT_ROOT = Path(r"D:\Molvi paper review\FC_HMARL")

STEP7O_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "evaluation"
    / "step7o_rule_based_comparison"
    / "step7o_three_way_episode_results.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "evaluation"
    / "step7r_corrected_reserve_validation"
)

LOCKED_CHECKPOINT = 1000
TEST_EPISODES = 30
SEED = 42
DEVICE = "cpu"
EPISODE_LENGTH = 24
NMG = 5

BESS_RATED_POWER_KW = np.array(
    [250.0, 300.0, 250.0, 300.0, 350.0],
    dtype=float,
)


def choose_representative_episode():
    df = pd.read_csv(STEP7O_FILE)
    fc = df[df["mode"] == "full_fc_hmarl"].copy()

    median_return = float(fc["total_return"].median())
    fc["distance_to_median"] = np.abs(
        fc["total_return"] - median_return
    )

    selected = fc.sort_values(
        ["distance_to_median", "test_episode"]
    ).iloc[0]

    return {
        "test_episode": int(selected["test_episode"]),
        "start_index": int(selected["start_index"]),
        "old_episode_return": float(selected["total_return"]),
        "old_median_return": median_return,
    }


def first(mapping, keys, default=np.nan):
    for key in keys:
        if key in mapping:
            return mapping[key]
    return default


def scalar(value, default=np.nan):
    if value is None:
        return float(default)

    arr = np.asarray(value, dtype=float)

    if arr.size == 0:
        return float(default)

    return float(arr.reshape(-1)[0])


def main():
    if not STEP7O_FILE.exists():
        raise FileNotFoundError(STEP7O_FILE)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    selected = choose_representative_episode()

    print("=" * 88)
    print("STEP 7R-D - CORRECTED RESERVE MODEL VALIDATION")
    print("=" * 88)
    print("Checkpoint              :", LOCKED_CHECKPOINT)
    print("Policy status            : OLD policy, pre-correction")
    print("Purpose                  : physical validation only")
    print("Representative TEST ep   :", selected["test_episode"])
    print("TEST start index         :", selected["start_index"])
    print("Learning                 : disabled")
    print("Actions                  : deterministic")
    print()

    data = t.TestData()

    starts = np.asarray(
        t.build_fixed_test_episode_starts(
            data=data,
            number_of_episodes=TEST_EPISODES,
            seed=SEED,
        ),
        dtype=int,
    )

    start = int(
        starts[selected["test_episode"] - 1]
    )

    if start != selected["start_index"]:
        raise RuntimeError(
            "Representative TEST start index mismatch."
        )

    local_agents, coordinator_agent = t.val.load_policy(
        checkpoint_episode=LOCKED_CHECKPOINT,
        device=DEVICE,
        seed=SEED,
    )

    bridge = t.build_test_bridge(
        data=data,
        episode_starts=np.array(
            [start],
            dtype=int,
        ),
    )

    obs = bridge.reset(episode=1)

    rows = []
    mg_rows = []

    corrected_total_return = 0.0

    with torch.no_grad():

        for hour in range(
            1,
            EPISODE_LENGTH + 1,
        ):

            actions = t.val.select_deterministic_actions(
                local_agents=local_agents,
                coordinator_agent=coordinator_agent,
                observation=obs,
            )

            result = bridge.step(actions)

            corrected_total_return += (
                float(
                    np.sum(
                        result.local_rewards
                    )
                )
                + float(
                    result.coordinator_reward
                )
            )

            physical = result.info[
                "physical_result"
            ]

            totals = physical["totals"]
            local_results = physical[
                "local_results"
            ]
            constraints = physical[
                "constraints"
            ]

            requested_total = scalar(
                first(
                    totals,
                    [
                        "reserve_requested_kw",
                    ],
                    default=np.nan,
                )
            )

            feasible_total = scalar(
                first(
                    totals,
                    [
                        "reserve_feasible_kw",
                    ],
                    default=np.nan,
                )
            )

            curtailed_total = scalar(
                first(
                    totals,
                    [
                        "reserve_curtailed_kw",
                    ],
                    default=np.nan,
                )
            )

            reserve_revenue = scalar(
                first(
                    totals,
                    [
                        "reserve_revenue_usd",
                    ],
                    default=0.0,
                ),
                default=0.0,
            )

            if np.isnan(feasible_total):
                feasible_total = 0.0

                for local in local_results:
                    feasible_total += scalar(
                        first(
                            local.get(
                                "market",
                                {},
                            ),
                            [
                                "reserve_power_kw",
                            ],
                            default=0.0,
                        ),
                        default=0.0,
                    )

            hour_headroom_excess = 0
            hour_min_soc_reserve = 0

            for i, local in enumerate(
                local_results
            ):

                bess = local.get(
                    "bess",
                    {},
                )
                market = local.get(
                    "market",
                    {},
                )

                feasible_bess = scalar(
                    first(
                        bess,
                        [
                            "feasible_power_kw",
                            "net_bess_power_kw",
                        ],
                        default=0.0,
                    ),
                    default=0.0,
                )

                new_soc = scalar(
                    first(
                        bess,
                        [
                            "new_soc",
                        ],
                        default=np.nan,
                    )
                )

                reserve = scalar(
                    first(
                        market,
                        [
                            "reserve_power_kw",
                        ],
                        default=0.0,
                    ),
                    default=0.0,
                )

                rated = float(
                    BESS_RATED_POWER_KW[i]
                )

                instantaneous_headroom = max(
                    0.0,
                    rated
                    - max(
                        feasible_bess,
                        0.0,
                    ),
                )

                excess = max(
                    0.0,
                    reserve
                    - instantaneous_headroom,
                )

                if excess > 1e-6:
                    hour_headroom_excess += 1

                if (
                    not np.isnan(new_soc)
                    and new_soc <= 0.2000001
                    and reserve > 1e-6
                ):
                    hour_min_soc_reserve += 1

                mg_rows.append({
                    "hour": hour,
                    "test_sample_index":
                        start + hour - 1,
                    "mg": i + 1,
                    "feasible_bess_power_kw":
                        feasible_bess,
                    "new_soc":
                        new_soc,
                    "reserve_power_kw":
                        reserve,
                    "rated_power_kw":
                        rated,
                    "instantaneous_headroom_kw":
                        instantaneous_headroom,
                    "reserve_minus_headroom_kw":
                        reserve
                        - instantaneous_headroom,
                    "headroom_excess_kw":
                        excess,
                    "reserve_at_minimum_soc":
                        bool(
                            not np.isnan(
                                new_soc
                            )
                            and new_soc
                            <= 0.2000001
                            and reserve
                            > 1e-6
                        ),
                })

            rows.append({
                "hour": hour,
                "test_sample_index":
                    start + hour - 1,
                "reserve_requested_kw":
                    requested_total,
                "reserve_feasible_kw":
                    feasible_total,
                "reserve_curtailed_kw":
                    curtailed_total,
                "reserve_revenue_usd":
                    reserve_revenue,
                "headroom_excess_mg_count":
                    hour_headroom_excess,
                "minimum_soc_reserve_mg_count":
                    hour_min_soc_reserve,
                "all_power_balanced":
                    bool(
                        constraints[
                            "all_power_balanced"
                        ]
                    ),
                "all_transformers_feasible":
                    bool(
                        constraints[
                            "all_transformers_feasible"
                        ]
                    ),
            })

            obs = result.next_observation

            if result.done:
                break

    hourly = pd.DataFrame(rows)
    mg = pd.DataFrame(mg_rows)

    if len(hourly) != EPISODE_LENGTH:
        raise RuntimeError(
            f"Expected {EPISODE_LENGTH} hours, "
            f"got {len(hourly)}."
        )

    # --------------------------------------------------------
    # Validation conditions
    # --------------------------------------------------------
    total_headroom_excess = int(
        (
            mg["headroom_excess_kw"]
            > 1e-6
        ).sum()
    )

    min_soc_reserve_count = int(
        mg[
            "reserve_at_minimum_soc"
        ].sum()
    )

    all_balance_ok = bool(
        hourly[
            "all_power_balanced"
        ].all()
    )

    all_transformer_ok = bool(
        hourly[
            "all_transformers_feasible"
        ].all()
    )

    if "reserve_requested_kw" in hourly:
        requested_sum = float(
            np.nansum(
                hourly[
                    "reserve_requested_kw"
                ]
            )
        )
    else:
        requested_sum = np.nan

    feasible_sum = float(
        np.nansum(
            hourly[
                "reserve_feasible_kw"
            ]
        )
    )

    curtailed_sum = float(
        np.nansum(
            hourly[
                "reserve_curtailed_kw"
            ]
        )
    )

    reserve_revenue_sum = float(
        np.nansum(
            hourly[
                "reserve_revenue_usd"
            ]
        )
    )

    pass_headroom = (
        total_headroom_excess == 0
    )

    pass_min_soc = (
        min_soc_reserve_count == 0
    )

    pass_all = bool(
        pass_headroom
        and pass_min_soc
        and all_balance_ok
        and all_transformer_ok
    )

    hourly_file = (
        OUTPUT_DIR
        / "step7r_d_hourly_corrected_reserve.csv"
    )

    mg_file = (
        OUTPUT_DIR
        / "step7r_d_microgrid_corrected_reserve.csv"
    )

    summary_file = (
        OUTPUT_DIR
        / "step7r_d_validation_summary.json"
    )

    hourly.to_csv(
        hourly_file,
        index=False,
    )

    mg.to_csv(
        mg_file,
        index=False,
    )

    summary = {
        "checkpoint": LOCKED_CHECKPOINT,
        "policy_status":
            "old_pre_correction_policy",
        "representative_test_episode":
            selected["test_episode"],
        "test_start_index": start,
        "old_episode_return":
            selected["old_episode_return"],
        "corrected_environment_episode_return":
            corrected_total_return,
        "reserve_requested_sum_kw_samples":
            requested_sum,
        "reserve_feasible_sum_kw_samples":
            feasible_sum,
        "reserve_curtailed_sum_kw_samples":
            curtailed_sum,
        "reserve_revenue_usd":
            reserve_revenue_sum,
        "headroom_excess_mg_hours":
            total_headroom_excess,
        "reserve_at_minimum_soc_mg_hours":
            min_soc_reserve_count,
        "all_power_balanced":
            all_balance_ok,
        "all_transformers_feasible":
            all_transformer_ok,
        "reserve_headroom_test_passed":
            pass_headroom,
        "minimum_soc_reserve_test_passed":
            pass_min_soc,
        "overall_physical_validation_passed":
            pass_all,
        "warning":
            "Economic return is diagnostic only because "
            "checkpoint 1000 was trained before reserve correction.",
    }

    summary_file.write_text(
        json.dumps(
            summary,
            indent=4,
        ),
        encoding="utf-8",
    )

    print("=" * 88)
    print("STEP 7R-D VALIDATION SUMMARY")
    print("=" * 88)
    print(
        f"Old-policy return in corrected env    : "
        f"{corrected_total_return:.6f}"
    )
    print(
        f"Reserve requested sum                 : "
        f"{requested_sum:.3f} kW-samples"
    )
    print(
        f"Reserve feasible sum                  : "
        f"{feasible_sum:.3f} kW-samples"
    )
    print(
        f"Reserve curtailed sum                 : "
        f"{curtailed_sum:.3f} kW-samples"
    )
    print(
        f"Reserve revenue                       : "
        f"${reserve_revenue_sum:.6f}"
    )
    print(
        f"Reserve > power-headroom MG-hours     : "
        f"{total_headroom_excess}"
    )
    print(
        f"Reserve at minimum SOC MG-hours       : "
        f"{min_soc_reserve_count}"
    )
    print(
        f"All power balances feasible           : "
        f"{all_balance_ok}"
    )
    print(
        f"All transformer constraints feasible  : "
        f"{all_transformer_ok}"
    )
    print(
        f"Reserve-headroom test passed          : "
        f"{pass_headroom}"
    )
    print(
        f"Minimum-SOC reserve test passed       : "
        f"{pass_min_soc}"
    )
    print(
        f"OVERALL PHYSICAL VALIDATION           : "
        f"{'PASS' if pass_all else 'FAIL'}"
    )
    print()
    print(
        "NOTE: The return above is NOT a final "
        "publication result because the policy was "
        "trained using the old reserve model."
    )
    print()
    print("[OK] Step 7R-D validation complete.")
    print("Hourly file :", hourly_file)
    print("MG file     :", mg_file)
    print("Summary     :", summary_file)


if __name__ == "__main__":
    main()
