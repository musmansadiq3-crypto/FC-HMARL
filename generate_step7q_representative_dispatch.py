"""
Step 7Q-B: Representative 24-hour FC-HMARL operational profiles.

Purpose
-------
Create publication-ready 24-hour figures from the locked checkpoint-1000
FC-HMARL policy using the same held-out TEST episode set already used in
Steps 7O/7P.

Representative episode
----------------------
The episode whose FC-HMARL total return is closest to the median FC-HMARL
return across the 30 held-out TEST episodes is selected automatically.

No retraining.
No checkpoint selection.
No TEST-driven tuning.

Outputs
-------
outputs/figures/step7q/
    Fig_7Q_B1_Load_PV_EV.png/.pdf
    Fig_7Q_B2_Grid_BESS.png/.pdf
    Fig_7Q_B3_BESS_SOC.png/.pdf
    Fig_7Q_B4_Sharing_Price.png/.pdf
    Table_7Q_B_Representative_24h.csv
    Step_7Q_B_Metadata.json
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
    / "figures"
    / "step7q"
)

LOCKED_CHECKPOINT = 1000
TEST_EPISODES = 30
SEED = 42
DEVICE = "cpu"
EPISODE_LENGTH = 24
NUMBER_OF_MGS = 5


def get_first(mapping, candidates, default=np.nan):
    """
    Safely retrieve the first available key from a dictionary.
    This keeps the script compatible with small naming differences
    across the validated environment result dictionaries.
    """
    if mapping is None:
        return default

    for key in candidates:
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


def choose_representative_episode():
    if not STEP7O_FILE.exists():
        raise FileNotFoundError(
            f"Step 7O-B episode file not found:\n{STEP7O_FILE}"
        )

    df = pd.read_csv(STEP7O_FILE)

    fc = df[
        df["mode"] == "full_fc_hmarl"
    ].copy()

    if len(fc) != TEST_EPISODES:
        raise ValueError(
            f"Expected {TEST_EPISODES} FC-HMARL TEST episodes, "
            f"found {len(fc)}."
        )

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
        "distance_to_median": float(
            selected["distance_to_median"]
        ),
    }


def save_figure(fig, stem):
    png = OUTPUT_DIR / f"{stem}.png"
    pdf = OUTPUT_DIR / f"{stem}.pdf"

    fig.savefig(
        png,
        dpi=600,
        bbox_inches="tight",
    )
    fig.savefig(
        pdf,
        bbox_inches="tight",
    )

    print("Saved:", png)
    print("Saved:", pdf)


def main():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    selected = choose_representative_episode()

    print("=" * 80)
    print("STEP 7Q-B - REPRESENTATIVE 24-HOUR FC-HMARL OPERATION")
    print("=" * 80)
    print("Checkpoint              :", LOCKED_CHECKPOINT)
    print("TEST episodes considered:", TEST_EPISODES)
    print("Selection criterion      : closest FC-HMARL return to TEST median")
    print("Representative episode   :", selected["test_episode"])
    print("Representative start idx :", selected["start_index"])
    print("Representative return    :", f"{selected['episode_return']:.6f}")
    print("Median FC-HMARL return   :", f"{selected['median_return']:.6f}")
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

    selected_start = int(
        starts[selected["test_episode"] - 1]
    )

    if selected_start != selected["start_index"]:
        raise RuntimeError(
            "Representative episode start index does not match "
            "the validated Step 7O-B results."
        )

    # The bridge indexes episodes sequentially, so construct a start list
    # whose first episode is exactly the selected representative start.
    representative_starts = np.array(
        [selected_start],
        dtype=int,
    )

    local_agents, coordinator_agent = t.val.load_policy(
        checkpoint_episode=LOCKED_CHECKPOINT,
        device=DEVICE,
        seed=SEED,
    )

    bridge = t.build_test_bridge(
        data=data,
        episode_starts=representative_starts,
    )

    obs = bridge.reset(episode=1)

    records = []

    with torch.no_grad():
        for hour in range(1, EPISODE_LENGTH + 1):

            actions = t.val.select_deterministic_actions(
                local_agents=local_agents,
                coordinator_agent=coordinator_agent,
                observation=obs,
            )

            result = bridge.step(actions)

            physical = result.info["physical_result"]
            totals = physical["totals"]
            local_results = physical["local_results"]

            # Prefer aggregate totals if available.
            pv_kw = scalar(
                get_first(
                    totals,
                    [
                        "pv_power_kw",
                        "total_pv_power_kw",
                        "total_pv_kw",
                        "pv_kw",
                    ],
                    default=np.nan,
                )
            )

            load_kw = scalar(
                get_first(
                    totals,
                    [
                        "load_power_kw",
                        "total_load_power_kw",
                        "total_load_kw",
                        "load_kw",
                    ],
                    default=np.nan,
                )
            )

            ev_kw = scalar(
                get_first(
                    totals,
                    [
                        "ev_power_kw",
                        "total_ev_power_kw",
                        "total_ev_kw",
                        "ev_kw",
                    ],
                    default=np.nan,
                )
            )

            bess_kw = scalar(
                get_first(
                    totals,
                    [
                        "bess_power_kw",
                        "total_bess_power_kw",
                        "total_bess_kw",
                        "bess_kw",
                    ],
                    default=np.nan,
                )
            )

            grid_kw = scalar(
                get_first(
                    totals,
                    [
                        "grid_power_kw",
                        "total_grid_power_kw",
                        "total_grid_kw",
                        "grid_kw",
                    ],
                    default=np.nan,
                )
            )

            sharing_kw = scalar(
                get_first(
                    totals,
                    [
                        "sharing_scheduled_kw",
                        "total_sharing_scheduled_kw",
                    ],
                    default=0.0,
                ),
                default=0.0,
            )

            sharing_loss_kw = scalar(
                get_first(
                    totals,
                    [
                        "sharing_loss_kw",
                        "total_sharing_loss_kw",
                    ],
                    default=0.0,
                ),
                default=0.0,
            )

            # If aggregate keys are absent, reconstruct from local results.
            local_pv = []
            local_load = []
            local_ev = []
            local_grid = []
            local_bess = []
            local_soc = []
            local_buy_price = []

            for local in local_results:

                local_pv.append(
                    scalar(
                        get_first(
                            local,
                            [
                                "pv_power_kw",
                                "pv_kw",
                                "pv_power",
                            ],
                            default=0.0,
                        ),
                        default=0.0,
                    )
                )

                local_load.append(
                    scalar(
                        get_first(
                            local,
                            [
                                "load_power_kw",
                                "load_kw",
                                "load_power",
                            ],
                            default=0.0,
                        ),
                        default=0.0,
                    )
                )

                local_ev.append(
                    scalar(
                        get_first(
                            local,
                            [
                                "ev_power_kw",
                                "ev_kw",
                                "ev_power",
                            ],
                            default=0.0,
                        ),
                        default=0.0,
                    )
                )

                local_grid.append(
                    scalar(
                        get_first(
                            local,
                            [
                                "grid_power_kw",
                                "grid_kw",
                                "grid_power",
                            ],
                            default=0.0,
                        ),
                        default=0.0,
                    )
                )

                bess = local.get("bess", {})
                local_bess.append(
                    scalar(
                        get_first(
                            bess,
                            [
                                "feasible_power_kw",
                                "net_power_kw",
                                "power_kw",
                                "bess_power_kw",
                            ],
                            default=0.0,
                        ),
                        default=0.0,
                    )
                )

                local_soc.append(
                    scalar(
                        get_first(
                            bess,
                            [
                                "new_soc",
                                "soc",
                                "state_of_charge",
                            ],
                            default=np.nan,
                        )
                    )
                )

                market = local.get("market", {})
                local_buy_price.append(
                    scalar(
                        get_first(
                            market,
                            [
                                "buy_price_usd_per_kwh",
                                "buy_price",
                                "purchase_price_usd_per_kwh",
                            ],
                            default=np.nan,
                        )
                    )
                )

            if np.isnan(pv_kw):
                pv_kw = float(np.sum(local_pv))
            if np.isnan(load_kw):
                load_kw = float(np.sum(local_load))
            if np.isnan(ev_kw):
                ev_kw = float(np.sum(local_ev))
            if np.isnan(grid_kw):
                grid_kw = float(np.sum(local_grid))
            if np.isnan(bess_kw):
                bess_kw = float(np.sum(local_bess))

            buy_price = float(
                np.nanmean(local_buy_price)
            )

            record = {
                "hour": hour,
                "pv_kw": pv_kw,
                "load_kw": load_kw,
                "ev_kw": ev_kw,
                "total_demand_kw": load_kw + ev_kw,
                "bess_power_kw": bess_kw,
                "grid_power_kw": grid_kw,
                "sharing_scheduled_kw": sharing_kw,
                "sharing_loss_kw": sharing_loss_kw,
                "buy_price_usd_per_kwh": buy_price,
                "total_step_reward":
                    float(np.sum(result.local_rewards))
                    + float(result.coordinator_reward),
            }

            for i in range(NUMBER_OF_MGS):
                record[f"mg{i+1}_soc"] = (
                    float(local_soc[i])
                    if i < len(local_soc)
                    else np.nan
                )

            records.append(record)

            obs = result.next_observation

            if result.done:
                break

    df = pd.DataFrame(records)

    if len(df) != EPISODE_LENGTH:
        raise RuntimeError(
            f"Expected {EPISODE_LENGTH} hours, got {len(df)}."
        )

    table_file = (
        OUTPUT_DIR
        / "Table_7Q_B_Representative_24h.csv"
    )

    df.to_csv(
        table_file,
        index=False,
    )

    # ------------------------------------------------------------
    # Figure B1: load / EV / PV
    # ------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8.4, 5.0))

    ax.plot(
        df["hour"],
        df["load_kw"],
        marker="o",
        label="Base load",
    )
    ax.plot(
        df["hour"],
        df["ev_kw"],
        marker="s",
        label="EV charging demand",
    )
    ax.plot(
        df["hour"],
        df["pv_kw"],
        marker="^",
        label="PV generation",
    )
    ax.plot(
        df["hour"],
        df["total_demand_kw"],
        linestyle="--",
        label="Load + EV",
    )

    ax.set_xlabel("Hour")
    ax.set_ylabel("Power (kW)")
    ax.set_xticks(np.arange(1, 25, 2))
    ax.set_title(
        "Representative 24-h Demand and PV Profile",
        fontweight="bold",
    )
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()

    save_figure(
        fig,
        "Fig_7Q_B1_Load_PV_EV",
    )
    plt.close(fig)

    # ------------------------------------------------------------
    # Figure B2: grid exchange and BESS dispatch
    # ------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8.4, 5.0))

    ax.plot(
        df["hour"],
        df["grid_power_kw"],
        marker="o",
        label="Grid exchange",
    )
    ax.plot(
        df["hour"],
        df["bess_power_kw"],
        marker="s",
        label="BESS power",
    )

    ax.axhline(
        0.0,
        linewidth=1.0,
    )
    ax.set_xlabel("Hour")
    ax.set_ylabel("Power (kW)")
    ax.set_xticks(np.arange(1, 25, 2))
    ax.set_title(
        "Representative 24-h Grid and BESS Dispatch",
        fontweight="bold",
    )
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()

    save_figure(
        fig,
        "Fig_7Q_B2_Grid_BESS",
    )
    plt.close(fig)

    # ------------------------------------------------------------
    # Figure B3: five microgrid SOC profiles
    # ------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8.4, 5.0))

    for i in range(NUMBER_OF_MGS):
        ax.plot(
            df["hour"],
            df[f"mg{i+1}_soc"] * 100.0,
            marker="o",
            label=f"MG{i+1}",
        )

    ax.set_xlabel("Hour")
    ax.set_ylabel("BESS SOC (%)")
    ax.set_xticks(np.arange(1, 25, 2))
    ax.set_ylim(15, 100)
    ax.set_title(
        "Representative 24-h BESS State of Charge",
        fontweight="bold",
    )
    ax.grid(alpha=0.25)
    ax.legend(
        ncol=3,
        frameon=False,
    )
    fig.tight_layout()

    save_figure(
        fig,
        "Fig_7Q_B3_BESS_SOC",
    )
    plt.close(fig)

    # ------------------------------------------------------------
    # Figure B4: sharing and tariff
    # ------------------------------------------------------------
    fig, ax1 = plt.subplots(figsize=(8.4, 5.0))

    ax1.plot(
        df["hour"],
        df["sharing_scheduled_kw"],
        marker="o",
        label="Scheduled sharing",
    )
    ax1.set_xlabel("Hour")
    ax1.set_ylabel("Scheduled sharing (kW)")
    ax1.set_xticks(np.arange(1, 25, 2))
    ax1.grid(alpha=0.25)

    ax2 = ax1.twinx()
    ax2.plot(
        df["hour"],
        df["buy_price_usd_per_kwh"],
        marker="s",
        linestyle="--",
        label="Buy tariff",
    )
    ax2.set_ylabel("Buy tariff (USD/kWh)")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()

    ax1.legend(
        lines1 + lines2,
        labels1 + labels2,
        frameon=False,
        loc="best",
    )

    ax1.set_title(
        "Representative 24-h Energy Sharing and Buy Tariff",
        fontweight="bold",
    )
    fig.tight_layout()

    save_figure(
        fig,
        "Fig_7Q_B4_Sharing_Price",
    )
    plt.close(fig)

    metadata = {
        "checkpoint": LOCKED_CHECKPOINT,
        "test_episodes_considered": TEST_EPISODES,
        "seed": SEED,
        "device": DEVICE,
        "selection_rule":
            "FC-HMARL episode with return closest to TEST median",
        **selected,
        "selected_start_verified": selected_start,
        "hours_recorded": int(len(df)),
        "learning": False,
        "deterministic_actions": True,
    }

    metadata_file = (
        OUTPUT_DIR
        / "Step_7Q_B_Metadata.json"
    )

    metadata_file.write_text(
        json.dumps(metadata, indent=4),
        encoding="utf-8",
    )

    print()
    print("=" * 80)
    print("STEP 7Q-B SUMMARY")
    print("=" * 80)
    print("Representative TEST episode :", selected["test_episode"])
    print("Start index                 :", selected_start)
    print("Episode return              :", f"{selected['episode_return']:.6f}")
    print("Median TEST return          :", f"{selected['median_return']:.6f}")
    print("24-h PV energy proxy        :", f"{df['pv_kw'].sum():.3f} kWh")
    print("24-h load energy            :", f"{df['load_kw'].sum():.3f} kWh")
    print("24-h EV energy              :", f"{df['ev_kw'].sum():.3f} kWh")
    print("24-h grid net-power sum     :", f"{df['grid_power_kw'].sum():.3f} kWh")
    print("24-h BESS signed-power sum  :", f"{df['bess_power_kw'].sum():.3f} kWh")
    print("24-h sharing scheduled      :", f"{df['sharing_scheduled_kw'].sum():.3f} kWh")
    print("Minimum MG SOC              :", f"{100.0 * df[[f'mg{i+1}_soc' for i in range(NUMBER_OF_MGS)]].min().min():.3f}%")
    print("Maximum MG SOC              :", f"{100.0 * df[[f'mg{i+1}_soc' for i in range(NUMBER_OF_MGS)]].max().max():.3f}%")
    print()
    print("[OK] Step 7Q-B representative operational figures complete.")
    print("Data file     :", table_file)
    print("Metadata file :", metadata_file)


if __name__ == "__main__":
    main()
