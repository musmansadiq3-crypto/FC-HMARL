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
    / "step7r_physical_audit"
)
LOCKED_CHECKPOINT = 1000
TEST_EPISODES = 30
SEED = 42
DEVICE = "cpu"
EPISODE_LENGTH = 24
NMG = 5

# Manuscript Table-2 BESS rated powers (kW)
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
        "episode_return": float(selected["total_return"]),
        "median_return": median_return,
    }


def get(mapping, *keys, default=np.nan):
    if mapping is None:
        return default
    for k in keys:
        if k in mapping:
            return mapping[k]
    return default


def fscalar(x, default=np.nan):
    if x is None:
        return float(default)
    arr = np.asarray(x, dtype=float)
    if arr.size == 0:
        return float(default)
    return float(arr.reshape(-1)[0])


def main():
    if not STEP7O_FILE.exists():
        raise FileNotFoundError(STEP7O_FILE)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    selected = choose_representative_episode()

    print("=" * 88)
    print("STEP 7R-A - FC-HMARL PHYSICAL DISPATCH AUDIT")
    print("=" * 88)
    print("Checkpoint              :", LOCKED_CHECKPOINT)
    print("Representative TEST ep  :", selected["test_episode"])
    print("TEST start index         :", selected["start_index"])
    print("Episode return           :", f"{selected['episode_return']:.6f}")
    print("Median TEST return       :", f"{selected['median_return']:.6f}")
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

    start = int(starts[selected["test_episode"] - 1])

    if start != selected["start_index"]:
        raise RuntimeError(
            f"Start mismatch: Step7O={selected['start_index']}, current={start}"
        )

    local_agents, coordinator_agent = t.val.load_policy(
        checkpoint_episode=LOCKED_CHECKPOINT,
        device=DEVICE,
        seed=SEED,
    )

    bridge = t.build_test_bridge(
        data=data,
        episode_starts=np.array([start], dtype=int),
    )

    obs = bridge.reset(episode=1)

    hourly_rows = []
    mg_rows = []

    with torch.no_grad():

        for hour in range(1, EPISODE_LENGTH + 1):

            actions = t.val.select_deterministic_actions(
                local_agents=local_agents,
                coordinator_agent=coordinator_agent,
                observation=obs,
            )

            local_action_arrays = [
                np.asarray(a, dtype=float).reshape(-1)
                for a in actions.local_actions
            ]

            coordinator_action = np.asarray(
                actions.coordinator_action,
                dtype=float,
            ).reshape(-1)

            result = bridge.step(actions)

            physical = result.info["physical_result"]
            totals = physical["totals"]
            constraints = physical["constraints"]
            local_results = physical["local_results"]

            total_requested = 0.0
            total_feasible = 0.0
            total_reserve = 0.0
            total_headroom = 0.0
            apparent_headroom_violations = 0

            for i, local in enumerate(local_results):

                bess = local.get("bess", {})
                market = local.get("market", {})

                requested = fscalar(
                    get(
                        bess,
                        "requested_power_kw",
                        "requested_bess_power_kw",
                        default=np.nan,
                    )
                )

                feasible = fscalar(
                    get(
                        bess,
                        "feasible_power_kw",
                        "net_power_kw",
                        "power_kw",
                        "bess_power_kw",
                        default=np.nan,
                    )
                )

                old_soc = fscalar(
                    get(
                        bess,
                        "old_soc",
                        "soc_before",
                        default=np.nan,
                    )
                )

                new_soc = fscalar(
                    get(
                        bess,
                        "new_soc",
                        "soc",
                        "state_of_charge",
                        default=np.nan,
                    )
                )

                reserve = fscalar(
                    get(
                        market,
                        "reserve_power_kw",
                        "reserve_kw",
                        default=0.0,
                    ),
                    default=0.0,
                )

                throughput = fscalar(
                    get(
                        bess,
                        "energy_throughput_kwh",
                        default=0.0,
                    ),
                    default=0.0,
                )

                normalized_bess_action = (
                    float(local_action_arrays[i][0])
                    if i < len(local_action_arrays)
                    and local_action_arrays[i].size > 0
                    else np.nan
                )

                # Simple power headroom diagnostic:
                # if positive feasible power is discharge, remaining upward
                # reserve capability cannot exceed rated - discharge.
                rated = float(BESS_RATED_POWER_KW[i])

                if np.isnan(feasible):
                    discharge_for_headroom = 0.0
                else:
                    discharge_for_headroom = max(feasible, 0.0)

                apparent_headroom = max(
                    rated - discharge_for_headroom,
                    0.0,
                )

                headroom_excess = max(
                    reserve - apparent_headroom,
                    0.0,
                )

                if headroom_excess > 1e-6:
                    apparent_headroom_violations += 1

                total_requested += (
                    0.0 if np.isnan(requested) else requested
                )
                total_feasible += (
                    0.0 if np.isnan(feasible) else feasible
                )
                total_reserve += reserve
                total_headroom += apparent_headroom

                mg_rows.append({
                    "episode_relative_hour": hour,
                    "test_sample_index": start + hour - 1,
                    "mg": i + 1,
                    "normalized_bess_action": normalized_bess_action,
                    "requested_bess_power_kw": requested,
                    "feasible_bess_power_kw": feasible,
                    "old_soc_percent":
                        old_soc * 100.0 if not np.isnan(old_soc) else np.nan,
                    "new_soc_percent":
                        new_soc * 100.0 if not np.isnan(new_soc) else np.nan,
                    "bess_throughput_kwh": throughput,
                    "reserve_power_kw": reserve,
                    "rated_bess_power_kw": rated,
                    "apparent_upward_headroom_kw": apparent_headroom,
                    "reserve_minus_headroom_kw": reserve - apparent_headroom,
                    "apparent_headroom_excess_kw": headroom_excess,
                })

            hourly_rows.append({
                "episode_relative_hour": hour,
                "test_sample_index": start + hour - 1,
                "coordinator_action_0": (
                    coordinator_action[0]
                    if coordinator_action.size > 0
                    else np.nan
                ),
                "coordinator_action_1": (
                    coordinator_action[1]
                    if coordinator_action.size > 1
                    else np.nan
                ),
                "coordinator_action_2": (
                    coordinator_action[2]
                    if coordinator_action.size > 2
                    else np.nan
                ),
                "total_requested_bess_power_kw": total_requested,
                "total_feasible_bess_power_kw": total_feasible,
                "total_reserve_power_kw": total_reserve,
                "total_apparent_upward_headroom_kw": total_headroom,
                "number_of_apparent_headroom_excess_mgs":
                    apparent_headroom_violations,
                "grid_import_kw": fscalar(
                    get(totals, "grid_import_kw", default=0.0),
                    default=0.0,
                ),
                "grid_export_kw": fscalar(
                    get(totals, "grid_export_kw", default=0.0),
                    default=0.0,
                ),
                "market_profit_usd": fscalar(
                    get(totals, "market_profit_usd", default=np.nan)
                ),
                "reserve_revenue_usd": fscalar(
                    get(totals, "reserve_revenue_usd", default=0.0),
                    default=0.0,
                ),
                "all_power_balanced": bool(
                    constraints.get("all_power_balanced", False)
                ),
                "all_transformers_feasible": bool(
                    constraints.get("all_transformers_feasible", False)
                ),
            })

            obs = result.next_observation

            if result.done:
                break

    hourly = pd.DataFrame(hourly_rows)
    mg = pd.DataFrame(mg_rows)

    hourly_file = OUTPUT_DIR / "step7r_hourly_audit.csv"
    mg_file = OUTPUT_DIR / "step7r_microgrid_bess_reserve_audit.csv"

    hourly.to_csv(hourly_file, index=False)
    mg.to_csv(mg_file, index=False)

    # ------------------------------------------------------------
    # Diagnostic summaries
    # ------------------------------------------------------------
    soc_min = float(mg["new_soc_percent"].min())
    soc_max = float(mg["new_soc_percent"].max())

    first_hour_bess = float(
        hourly.iloc[0]["total_feasible_bess_power_kw"]
    )
    second_hour_bess = float(
        hourly.iloc[1]["total_feasible_bess_power_kw"]
    )

    after_hour2_abs_bess = float(
        np.abs(
            hourly.loc[
                hourly["episode_relative_hour"] >= 3,
                "total_feasible_bess_power_kw",
            ]
        ).sum()
    )

    reserve_total = float(
        hourly["total_reserve_power_kw"].sum()
    )

    reserve_revenue_total = float(
        hourly["reserve_revenue_usd"].sum()
    )

    headroom_excess_count = int(
        (mg["apparent_headroom_excess_kw"] > 1e-6).sum()
    )

    headroom_excess_hours = int(
        (
            hourly[
                "number_of_apparent_headroom_excess_mgs"
            ] > 0
        ).sum()
    )

    simultaneous_dispatch_and_reserve = int(
        (
            (np.abs(mg["feasible_bess_power_kw"]) > 1e-6)
            & (mg["reserve_power_kw"] > 1e-6)
        ).sum()
    )

    power_balance_ok = bool(
        hourly["all_power_balanced"].all()
    )
    transformer_ok = bool(
        hourly["all_transformers_feasible"].all()
    )

    # Check whether SOC reaches lower bound by hour 2.
    hour2 = mg[
        mg["episode_relative_hour"] == 2
    ]

    all_at_20_after_hour2 = bool(
        np.allclose(
            hour2["new_soc_percent"].to_numpy(float),
            20.0,
            atol=1e-6,
        )
    )

    summary = {
        "checkpoint": LOCKED_CHECKPOINT,
        "representative_test_episode": selected["test_episode"],
        "test_start_index": start,
        "hours_audited": int(len(hourly)),
        "minimum_soc_percent": soc_min,
        "maximum_soc_percent": soc_max,
        "all_mgs_at_20_percent_after_hour_2":
            all_at_20_after_hour2,
        "hour_1_total_feasible_bess_power_kw":
            first_hour_bess,
        "hour_2_total_feasible_bess_power_kw":
            second_hour_bess,
        "absolute_bess_dispatch_after_hour_2_kwh_proxy":
            after_hour2_abs_bess,
        "total_reserve_power_sum_kw":
            reserve_total,
        "total_reserve_revenue_usd":
            reserve_revenue_total,
        "simultaneous_bess_dispatch_and_reserve_mg_hours":
            simultaneous_dispatch_and_reserve,
        "apparent_reserve_headroom_excess_mg_hours":
            headroom_excess_count,
        "hours_with_any_apparent_headroom_excess":
            headroom_excess_hours,
        "all_power_balanced": power_balance_ok,
        "all_transformers_feasible": transformer_ok,
        "time_axis_note":
            "episode_relative_hour 1..24; test_sample_index is also saved",
        "headroom_diagnostic_note":
            "Diagnostic assumes positive feasible BESS power is discharge and "
            "upward reserve should not exceed rated power minus simultaneous "
            "discharge. This checks power headroom only; energy-duration "
            "reserve feasibility is not evaluated here.",
    }

    summary_file = OUTPUT_DIR / "step7r_physical_audit_summary.json"
    summary_file.write_text(
        json.dumps(summary, indent=4),
        encoding="utf-8",
    )

    print("=" * 88)
    print("STEP 7R-A AUDIT SUMMARY")
    print("=" * 88)
    print(f"SOC range                              : {soc_min:.3f}% to {soc_max:.3f}%")
    print(f"All MGs at 20% after hour 2           : {all_at_20_after_hour2}")
    print(f"Hour-1 aggregate BESS power           : {first_hour_bess:.3f} kW")
    print(f"Hour-2 aggregate BESS power           : {second_hour_bess:.3f} kW")
    print(f"|BESS power| after hour 2             : {after_hour2_abs_bess:.6f} kWh-proxy")
    print(f"Aggregate reserve-power sum           : {reserve_total:.3f} kW-h samples")
    print(f"Reserve revenue over episode          : ${reserve_revenue_total:.6f}")
    print(f"BESS dispatch + reserve MG-hours      : {simultaneous_dispatch_and_reserve}")
    print(f"Reserve > apparent headroom MG-hours  : {headroom_excess_count}")
    print(f"Hours with any headroom excess        : {headroom_excess_hours}")
    print(f"All power balances feasible           : {power_balance_ok}")
    print(f"All transformer constraints feasible  : {transformer_ok}")
    print()
    print("IMPORTANT:")
    print("  Hour 1..24 are episode-relative hours, not automatically clock time.")
    print("  See test_sample_index in the CSV to trace the underlying test sequence.")
    print()
    print("[OK] Step 7R-A physical-dispatch audit complete.")
    print("Hourly audit :", hourly_file)
    print("MG audit     :", mg_file)
    print("Summary JSON :", summary_file)


if __name__ == "__main__":
    main()
