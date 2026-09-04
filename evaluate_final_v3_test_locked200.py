"""
FC-HMARL FINAL V3 — LOCKED CHECKPOINT 200 TEST EVALUATION

Protocol
--------
TRAIN       -> policy learning only
VALIDATION  -> forecasting-method selection, Phi calibration, checkpoint selection
TEST        -> one-time final evaluation of the already frozen checkpoint 200

This script:
- loads ONLY checkpoint 200;
- evaluates deterministically;
- performs no gradient updates;
- writes no replay transitions;
- does not rank or compare checkpoints;
- evaluates every valid rolling 24-hour TEST window by default.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch

from marl.training_loop import HierarchicalActionBundle


PROJECT_ROOT = Path(__file__).resolve().parent

V3_LAUNCHER = (
    PROJECT_ROOT
    / "train_real_fc_hmarl_final_v3.py"
)

TEST_ARCHIVE = (
    PROJECT_ROOT
    / "outputs"
    / "rl_data"
    / "real_rl_test_archive.npz"
)

TEST_METADATA = (
    PROJECT_ROOT
    / "outputs"
    / "rl_data"
    / "real_rl_test_metadata.json"
)

CHECKPOINT_DIRECTORY = (
    PROJECT_ROOT
    / "outputs"
    / "checkpoints"
    / "real_fc_hmarl_final_v3"
)

OUTPUT_DIRECTORY = (
    PROJECT_ROOT
    / "outputs"
    / "results"
    / "real_fc_hmarl_final_v3_test"
)

LOCKED_CHECKPOINT = 200


def section(title: str) -> None:
    print()
    print("=" * 80)
    print(title)
    print("=" * 80)


def load_v3_module():
    if not V3_LAUNCHER.exists():
        raise FileNotFoundError(
            f"FINAL V3 launcher not found:\n{V3_LAUNCHER}"
        )

    spec = importlib.util.spec_from_file_location(
        "fc_hmarl_final_v3_training",
        V3_LAUNCHER,
    )

    if spec is None or spec.loader is None:
        raise RuntimeError(
            "Could not import FINAL V3 launcher."
        )

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestData:
    """Read-only TEST archive with the V3 data interface."""

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

    def __init__(self, archive_path: Path):

        if not archive_path.exists():
            raise FileNotFoundError(
                "TEST archive not found:\n"
                f"{archive_path}"
            )

        archive = np.load(
            archive_path,
            allow_pickle=True,
        )

        missing = [
            key
            for key in self.REQUIRED
            if key not in archive.files
        ]

        if missing:
            raise RuntimeError(
                "TEST archive is missing keys:\n  "
                + "\n  ".join(missing)
            )

        source_split = str(
            np.asarray(
                archive["source_split"]
            ).reshape(-1)[0]
        ).lower()

        if source_split != "test":
            raise RuntimeError(
                "Refusing evaluation: archive source_split "
                f"is {source_split!r}, expected 'test'."
            )

        self.current_actual_original = np.asarray(
            archive["current_actual_original"],
            dtype=np.float32,
        )

        self.current_actual_normalized = np.asarray(
            archive["current_actual_normalized"],
            dtype=np.float32,
        )

        self.forecast_original = np.asarray(
            archive["forecast_original"],
            dtype=np.float32,
        )

        self.forecast_normalized = np.asarray(
            archive["forecast_normalized"],
            dtype=np.float32,
        )

        self.predictive_state_matrix = np.asarray(
            archive["predictive_state_matrix"],
            dtype=np.float32,
        )

        self.predictive_state_flat = np.asarray(
            archive["predictive_state_flat"],
            dtype=np.float32,
        )

        self.causal_confidence_24h = np.asarray(
            archive["causal_confidence_24h"],
            dtype=np.float32,
        )

        self.test_indices = np.asarray(
            archive["test_indices"]
        )

        self.number_of_samples = int(
            len(self.current_actual_original)
        )

        if self.current_actual_original.shape != (
            self.number_of_samples,
            4,
        ):
            raise RuntimeError(
                "Unexpected TEST physical-data shape."
            )

        if self.predictive_state_matrix.shape != (
            self.number_of_samples,
            24,
            4,
        ):
            raise RuntimeError(
                "Unexpected TEST predictive-state shape."
            )

        if self.predictive_state_flat.shape != (
            self.number_of_samples,
            96,
        ):
            raise RuntimeError(
                "Unexpected TEST flattened predictive-state shape."
            )

        if self.causal_confidence_24h.shape != (24,):
            raise RuntimeError(
                "Expected 24 causal-confidence values."
            )

        if self.number_of_samples < 24:
            raise RuntimeError(
                "TEST archive is too short for a 24-hour episode."
            )


class SequentialTestProvider:
    """
    Deterministic TEST provider.

    Episode 1 starts at TEST index 0, episode 2 at index 1, etc.
    Therefore requesting all valid starts evaluates every rolling
    24-hour TEST window exactly once.
    """

    def __init__(
        self,
        v3,
        data: TestData,
        number_of_episodes: int,
    ):
        self.v3 = v3
        self.data = data
        self.number_of_episodes = int(
            number_of_episodes
        )

        self.valid_starts = np.arange(
            data.number_of_samples
            - v3.EPISODE_LENGTH
            + 1,
            dtype=int,
        )

        if not (
            1
            <= self.number_of_episodes
            <= len(self.valid_starts)
        ):
            raise ValueError(
                "Requested TEST episodes exceed available "
                "24-hour windows."
            )

        self.episode_starts = (
            self.valid_starts[
                : self.number_of_episodes
            ]
        )

    def get_sample_index(
        self,
        episode: int,
        step: int,
    ) -> int:

        episode_index = int(episode) - 1

        if not (
            0
            <= episode_index
            < self.number_of_episodes
        ):
            raise IndexError(
                "Invalid TEST episode."
            )

        if not (
            0
            <= int(step)
            < self.v3.EPISODE_LENGTH
        ):
            raise IndexError(
                "Invalid TEST step."
            )

        return int(
            self.episode_starts[
                episode_index
            ]
            + int(step)
        )

    def __call__(
        self,
        episode: int,
        step: int,
    ):
        index = self.get_sample_index(
            episode,
            step,
        )

        actual = (
            self.data
            .current_actual_original[
                index
            ]
        )

        pv_reference = float(
            actual[0]
        )

        load_reference = float(
            actual[1]
        )

        ev_reference = float(
            actual[2]
        )

        price_reference = float(
            actual[3]
        )

        irradiance = float(
            np.clip(
                pv_reference * 1000.0,
                0.0,
                1000.0,
            )
        )

        irradiances = np.full(
            self.v3.NUMBER_OF_MICROGRIDS,
            irradiance,
            dtype=float,
        )

        load_shape = (
            self.v3.safe_fraction(
                load_reference,
                0.0,
                self.v3.TRAIN_LOAD_MAX,
            )
        )

        loads = (
            self.v3.PEAK_LOAD_KW
            * load_shape
        )

        ev_shape = (
            self.v3.safe_fraction(
                ev_reference,
                self.v3.TRAIN_EV_MIN,
                self.v3.TRAIN_EV_MAX,
            )
        )

        maximum_ev_power = (
            self.v3.EV_COUNTS.astype(
                float
            )
            * self.v3.EV_CHARGER_POWER_KW
            * self.v3.EV_SIMULTANEOUS_FRACTION
        )

        aggregate_ev_requests = (
            maximum_ev_power
            * ev_shape
        )

        per_ev_requests = (
            self.v3
            .build_per_ev_charging_requests(
                aggregate_ev_requests
            )
        )

        price_shape = (
            self.v3.safe_fraction(
                price_reference,
                self.v3.TRAIN_PRICE_MIN,
                self.v3.TRAIN_PRICE_MAX,
            )
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

        buy_prices = np.full(
            self.v3.NUMBER_OF_MICROGRIDS,
            buy_price,
            dtype=float,
        )

        sell_prices = np.full(
            self.v3.NUMBER_OF_MICROGRIDS,
            sell_price,
            dtype=float,
        )

        predictive_state = (
            self.data
            .predictive_state_matrix[
                index
            ]
            .copy()
        )

        # Same immediate-control confidence convention as V3.
        scalar_confidence = float(
            self.data
            .causal_confidence_24h[0]
        )

        return self.v3.VPPExogenousInput(
            time=float(step),
            loads_kw=loads,
            irradiances_w_m2=irradiances,
            buy_prices_usd_per_kwh=(
                buy_prices
            ),
            sell_prices_usd_per_kwh=(
                sell_prices
            ),
            predictive_state=(
                predictive_state
            ),
            confidence=(
                scalar_confidence
            ),
            market_price=(
                price_reference
            ),
            ev_requested_charging_powers_kw=(
                per_ev_requests
            ),
        )


def verify_locked_checkpoint() -> None:

    expected = [
        CHECKPOINT_DIRECTORY
        / (
            f"local_agent_{i}_"
            f"episode_{LOCKED_CHECKPOINT}.pt"
        )
        for i in range(1, 6)
    ]

    expected.append(
        CHECKPOINT_DIRECTORY
        / (
            "coordinator_episode_"
            f"{LOCKED_CHECKPOINT}.pt"
        )
    )

    missing = [
        path
        for path in expected
        if not path.exists()
    ]

    if missing:
        raise FileNotFoundError(
            "Locked checkpoint 200 is incomplete:\n"
            + "\n".join(
                str(path)
                for path in missing
            )
        )


def verify_test_metadata() -> None:

    if not TEST_METADATA.exists():
        print(
            "[WARN] TEST metadata JSON not found; "
            "archive source_split check still passed."
        )
        return

    with open(
        TEST_METADATA,
        "r",
        encoding="utf-8",
    ) as file:
        metadata = json.load(file)

    source_split = str(
        metadata.get(
            "source_split",
            ""
        )
    ).lower()

    if source_split != "test":
        raise RuntimeError(
            "TEST metadata source_split is not 'test'."
        )

    locked = metadata.get(
        "locked_checkpoint_episode",
        LOCKED_CHECKPOINT,
    )

    if int(locked) != LOCKED_CHECKPOINT:
        raise RuntimeError(
            "TEST metadata does not record checkpoint 200 "
            "as the frozen checkpoint."
        )


def load_locked_policy(
    v3,
    device: str,
    seed: int,
):

    v3.set_global_seed(
        seed
    )

    local_agents = (
        v3.build_local_agents(
            seed=seed,
            smoke_test=False,
            device=device,
        )
    )

    coordinator_agent = (
        v3.build_coordinator_agent(
            seed=seed,
            smoke_test=False,
            device=device,
        )
    )

    for agent in local_agents:

        path = (
            CHECKPOINT_DIRECTORY
            / (
                f"local_agent_"
                f"{agent.microgrid_id}_"
                f"episode_"
                f"{LOCKED_CHECKPOINT}.pt"
            )
        )

        agent.load(
            path,
            load_optimizers=False,
        )

        if hasattr(
            agent,
            "set_training_mode",
        ):
            agent.set_training_mode(
                False
            )

    coordinator_path = (
        CHECKPOINT_DIRECTORY
        / (
            "coordinator_episode_"
            f"{LOCKED_CHECKPOINT}.pt"
        )
    )

    coordinator_agent.load(
        coordinator_path,
        load_optimizers=False,
    )

    if hasattr(
        coordinator_agent,
        "set_training_mode",
    ):
        coordinator_agent.set_training_mode(
            False
        )

    return (
        local_agents,
        coordinator_agent,
    )


def deterministic_actions(
    local_agents,
    coordinator_agent,
    observation,
):

    local_actions = [
        np.asarray(
            agent.select_action(
                state,
                deterministic=True,
            ),
            dtype=np.float32,
        )
        for agent, state
        in zip(
            local_agents,
            observation.local_states,
        )
    ]

    coordinator_action = np.asarray(
        coordinator_agent.select_action(
            observation.coordinator_state,
            deterministic=True,
        ),
        dtype=np.float32,
    )

    return HierarchicalActionBundle(
        local_actions=local_actions,
        coordinator_action=(
            coordinator_action
        ),
    )


def write_csv(
    rows,
    path: Path,
) -> None:

    if not rows:
        return

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        path,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=list(
                rows[0].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(
            rows
        )


def evaluate_test(
    v3,
    data: TestData,
    number_of_test_episodes: int,
    seed: int,
    device: str,
):

    bridge = (
        v3.build_real_training_bridge(
            data=data,
            maximum_episodes=(
                number_of_test_episodes
            ),
            seed=seed,
        )
    )

    provider = SequentialTestProvider(
        v3=v3,
        data=data,
        number_of_episodes=(
            number_of_test_episodes
        ),
    )

    bridge.exogenous_provider = (
        provider
    )

    (
        local_agents,
        coordinator_agent,
    ) = load_locked_policy(
        v3=v3,
        device=device,
        seed=seed,
    )

    rows = []

    with torch.no_grad():

        for episode in range(
            1,
            number_of_test_episodes + 1,
        ):

            observation = bridge.reset(
                episode=episode
            )

            local_return = np.zeros(
                v3.NUMBER_OF_MICROGRIDS,
                dtype=float,
            )

            coordinator_return = 0.0

            grid_import_kwh = 0.0
            grid_export_kwh = 0.0

            purchase_cost_usd = 0.0
            sale_revenue_usd = 0.0
            reserve_revenue_usd = 0.0
            market_profit_usd = 0.0

            reserve_requested_kwh = 0.0
            reserve_feasible_kwh = 0.0
            reserve_curtailed_kwh = 0.0

            bess_throughput_kwh = 0.0

            sharing_scheduled_kwh = 0.0
            sharing_received_kwh = 0.0
            sharing_loss_kwh = 0.0

            total_balance_violation_kw = 0.0
            max_balance_violation_kw = 0.0
            balance_violation_steps = 0

            transformer_violation_steps = 0
            total_transformer_violation_kw = 0.0

            soc_below_min_count = 0
            soc_above_max_count = 0

            grid_power_series = []

            steps = 0

            for _ in range(
                v3.EPISODE_LENGTH
            ):

                actions = (
                    deterministic_actions(
                        local_agents,
                        coordinator_agent,
                        observation,
                    )
                )

                result = bridge.step(
                    actions
                )

                local_rewards = np.asarray(
                    result.local_rewards,
                    dtype=float,
                )

                coordinator_reward = float(
                    result.coordinator_reward
                )

                local_return += (
                    local_rewards
                )

                coordinator_return += (
                    coordinator_reward
                )

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

                grid_import_kwh += float(
                    totals[
                        "grid_import_kw"
                    ]
                )

                grid_export_kwh += float(
                    totals[
                        "grid_export_kw"
                    ]
                )

                purchase_cost_usd += float(
                    totals[
                        "grid_purchase_cost_usd"
                    ]
                )

                sale_revenue_usd += float(
                    totals[
                        "grid_sale_revenue_usd"
                    ]
                )

                reserve_revenue_usd += float(
                    totals[
                        "reserve_revenue_usd"
                    ]
                )

                market_profit_usd += float(
                    totals[
                        "market_profit_usd"
                    ]
                )

                reserve_requested_kwh += float(
                    totals.get(
                        "reserve_requested_kw",
                        0.0,
                    )
                )

                reserve_feasible_kwh += float(
                    totals.get(
                        "reserve_feasible_kw",
                        0.0,
                    )
                )

                reserve_curtailed_kwh += float(
                    totals.get(
                        "reserve_curtailed_kw",
                        0.0,
                    )
                )

                sharing_scheduled_kwh += float(
                    totals[
                        "sharing_scheduled_kw"
                    ]
                )

                sharing_received_kwh += float(
                    totals.get(
                        "sharing_received_kw",
                        0.0,
                    )
                )

                sharing_loss_kwh += float(
                    totals[
                        "sharing_loss_kw"
                    ]
                )

                grid_power_series.append(
                    float(
                        totals[
                            "grid_power_kw"
                        ]
                    )
                )

                step_balance = float(
                    constraints[
                        "total_power_balance_violation_kw"
                    ]
                )

                total_balance_violation_kw += (
                    step_balance
                )

                max_balance_violation_kw = max(
                    max_balance_violation_kw,
                    step_balance,
                )

                if not bool(
                    constraints[
                        "all_power_balanced"
                    ]
                ):
                    balance_violation_steps += 1

                step_transformer_violation = float(
                    constraints.get(
                        "total_transformer_violation_kw",
                        0.0,
                    )
                )

                total_transformer_violation_kw += (
                    step_transformer_violation
                )

                if not bool(
                    constraints[
                        "all_transformers_feasible"
                    ]
                ):
                    transformer_violation_steps += 1

                for mg in physical[
                    "local_results"
                ]:

                    bess = mg.get(
                        "bess",
                        {}
                    )

                    bess_throughput_kwh += float(
                        bess.get(
                            "energy_throughput_kwh",
                            0.0,
                        )
                    )

                # Explicit SOC feasibility audit after the physical step.
                for mg in bridge.environment.microgrids:

                    soc = float(
                        mg.bess.soc
                    )

                    minimum_soc = float(
                        mg.bess.minimum_soc
                    )

                    maximum_soc = float(
                        mg.bess.maximum_soc
                    )

                    tolerance = 1e-9

                    if soc < (
                        minimum_soc
                        - tolerance
                    ):
                        soc_below_min_count += 1

                    if soc > (
                        maximum_soc
                        + tolerance
                    ):
                        soc_above_max_count += 1

                observation = (
                    result.next_observation
                )

                steps += 1

                if result.done:
                    break

            grid_power_array = np.asarray(
                grid_power_series,
                dtype=float,
            )

            total_return = float(
                local_return.sum()
                + coordinator_return
            )

            row = {
                "test_episode":
                    int(episode),

                "start_index":
                    int(
                        provider.episode_starts[
                            episode - 1
                        ]
                    ),

                "steps":
                    int(steps),

                "total_return":
                    total_return,

                "coordinator_return":
                    float(
                        coordinator_return
                    ),

                "sum_local_return":
                    float(
                        local_return.sum()
                    ),

                "mean_local_return_per_mg":
                    float(
                        local_return.mean()
                    ),

                "mg1_return":
                    float(
                        local_return[0]
                    ),

                "mg2_return":
                    float(
                        local_return[1]
                    ),

                "mg3_return":
                    float(
                        local_return[2]
                    ),

                "mg4_return":
                    float(
                        local_return[3]
                    ),

                "mg5_return":
                    float(
                        local_return[4]
                    ),

                "grid_import_kwh":
                    float(
                        grid_import_kwh
                    ),

                "grid_export_kwh":
                    float(
                        grid_export_kwh
                    ),

                "grid_purchase_cost_usd":
                    float(
                        purchase_cost_usd
                    ),

                "grid_sale_revenue_usd":
                    float(
                        sale_revenue_usd
                    ),

                "reserve_revenue_usd":
                    float(
                        reserve_revenue_usd
                    ),

                "net_market_cost_usd":
                    float(
                        purchase_cost_usd
                        - sale_revenue_usd
                        - reserve_revenue_usd
                    ),

                "market_profit_usd":
                    float(
                        market_profit_usd
                    ),

                "reserve_requested_kwh":
                    float(
                        reserve_requested_kwh
                    ),

                "reserve_feasible_kwh":
                    float(
                        reserve_feasible_kwh
                    ),

                "reserve_curtailed_kwh":
                    float(
                        reserve_curtailed_kwh
                    ),

                "bess_throughput_kwh":
                    float(
                        bess_throughput_kwh
                    ),

                "sharing_scheduled_kwh":
                    float(
                        sharing_scheduled_kwh
                    ),

                "sharing_received_kwh":
                    float(
                        sharing_received_kwh
                    ),

                "sharing_loss_kwh":
                    float(
                        sharing_loss_kwh
                    ),

                "peak_grid_import_kw":
                    float(
                        max(
                            0.0,
                            np.max(
                                grid_power_array
                            ),
                        )
                    ),

                "peak_grid_export_kw":
                    float(
                        max(
                            0.0,
                            -np.min(
                                grid_power_array
                            ),
                        )
                    ),

                "mean_balance_violation_kw_per_step":
                    float(
                        total_balance_violation_kw
                        / max(
                            steps,
                            1,
                        )
                    ),

                "maximum_balance_violation_kw":
                    float(
                        max_balance_violation_kw
                    ),

                "power_balance_violation_steps":
                    int(
                        balance_violation_steps
                    ),

                "total_transformer_violation_kw":
                    float(
                        total_transformer_violation_kw
                    ),

                "transformer_violation_steps":
                    int(
                        transformer_violation_steps
                    ),

                "soc_below_min_count":
                    int(
                        soc_below_min_count
                    ),

                "soc_above_max_count":
                    int(
                        soc_above_max_count
                    ),
            }

            rows.append(
                row
            )

            if (
                episode == 1
                or episode
                % 100 == 0
                or episode
                == number_of_test_episodes
            ):
                print(
                    f"Episode "
                    f"{episode:4d}/"
                    f"{number_of_test_episodes} | "
                    f"return="
                    f"{total_return:.6f} | "
                    f"balance="
                    f"{row['mean_balance_violation_kw_per_step']:.6f} | "
                    f"transformer="
                    f"{row['transformer_violation_steps']}"
                )

    return rows


def summarize(
    rows,
    checkpoint_episode: int,
):

    def values(key):
        return np.asarray(
            [
                float(row[key])
                for row in rows
            ],
            dtype=float,
        )

    returns = values(
        "total_return"
    )

    summary = {
        "evaluation_split":
            "test",

        "locked_checkpoint_episode":
            int(
                checkpoint_episode
            ),

        "checkpoint_selection_source":
            "validation",

        "deterministic_actions":
            True,

        "gradient_updates":
            False,

        "replay_writes":
            False,

        "checkpoint_comparison_on_test":
            False,

        "test_episodes":
            int(
                len(rows)
            ),

        "mean_return":
            float(
                returns.mean()
            ),

        "std_return":
            float(
                returns.std()
            ),

        "median_return":
            float(
                np.median(
                    returns
                )
            ),

        "minimum_return":
            float(
                returns.min()
            ),

        "maximum_return":
            float(
                returns.max()
            ),
    }

    mean_keys = [
        "coordinator_return",
        "sum_local_return",
        "mean_local_return_per_mg",
        "mg1_return",
        "mg2_return",
        "mg3_return",
        "mg4_return",
        "mg5_return",
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

    for key in mean_keys:
        summary[
            f"mean_{key}"
        ] = float(
            values(key).mean()
        )

    summary[
        "total_power_balance_violation_steps"
    ] = int(
        sum(
            int(
                row[
                    "power_balance_violation_steps"
                ]
            )
            for row in rows
        )
    )

    summary[
        "total_transformer_violation_steps"
    ] = int(
        sum(
            int(
                row[
                    "transformer_violation_steps"
                ]
            )
            for row in rows
        )
    )

    summary[
        "total_soc_below_min_count"
    ] = int(
        sum(
            int(
                row[
                    "soc_below_min_count"
                ]
            )
            for row in rows
        )
    )

    summary[
        "total_soc_above_max_count"
    ] = int(
        sum(
            int(
                row[
                    "soc_above_max_count"
                ]
            )
            for row in rows
        )
    )

    summary[
        "all_transformers_feasible"
    ] = bool(
        summary[
            "total_transformer_violation_steps"
        ]
        == 0
    )

    summary[
        "all_soc_feasible"
    ] = bool(
        summary[
            "total_soc_below_min_count"
        ]
        == 0
        and summary[
            "total_soc_above_max_count"
        ]
        == 0
    )

    return summary


def parse_arguments():

    parser = argparse.ArgumentParser(
        description=(
            "One-time final TEST evaluation of "
            "the validation-locked FC-HMARL "
            "FINAL V3 checkpoint 200."
        )
    )

    parser.add_argument(
        "--test-episodes",
        type=int,
        default=None,
        help=(
            "Number of sequential rolling TEST windows. "
            "Default: all available 24-hour windows."
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=2026,
        help=(
            "Seed used only for deterministic agent construction."
        ),
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=[
            "cpu",
            "cuda",
            "auto",
        ],
    )

    return parser.parse_args()


def main():

    args = parse_arguments()

    section(
        "FC-HMARL FINAL V3 — ONE-TIME FINAL TEST"
    )

    print(
        f"Locked checkpoint        : "
        f"{LOCKED_CHECKPOINT}"
    )

    print(
        f"TEST archive             : "
        f"{TEST_ARCHIVE}"
    )

    print(
        f"Checkpoint directory     : "
        f"{CHECKPOINT_DIRECTORY}"
    )

    print(
        "Deterministic actions    : True"
    )

    print(
        "Gradient updates         : False"
    )

    print(
        "Replay writes            : False"
    )

    print(
        "Checkpoint ranking       : False"
    )

    print(
        "Checkpoint re-selection  : FORBIDDEN"
    )

    verify_locked_checkpoint()
    verify_test_metadata()

    v3 = load_v3_module()

    device = v3.resolve_device(
        args.device
    )

    data = TestData(
        TEST_ARCHIVE
    )

    all_test_windows = (
        data.number_of_samples
        - v3.EPISODE_LENGTH
        + 1
    )

    if args.test_episodes is None:
        number_of_test_episodes = (
            all_test_windows
        )
    else:
        number_of_test_episodes = int(
            args.test_episodes
        )

    if not (
        1
        <= number_of_test_episodes
        <= all_test_windows
    ):
        raise ValueError(
            "--test-episodes must be between "
            f"1 and {all_test_windows}."
        )

    section(
        "FINAL TEST PROTOCOL"
    )

    print(
        f"TEST samples             : "
        f"{data.number_of_samples}"
    )

    print(
        f"Valid 24-h TEST windows  : "
        f"{all_test_windows}"
    )

    print(
        f"Windows evaluated        : "
        f"{number_of_test_episodes}"
    )

    print(
        f"Device                   : "
        f"{device}"
    )

    print(
        "[OK] TEST source_split verified."
    )

    print(
        "[OK] Checkpoint 200 existence verified."
    )

    print(
        "[OK] No other checkpoint will be loaded."
    )

    section(
        "EVALUATING LOCKED CHECKPOINT 200 ON TEST"
    )

    rows = evaluate_test(
        v3=v3,
        data=data,
        number_of_test_episodes=(
            number_of_test_episodes
        ),
        seed=int(
            args.seed
        ),
        device=device,
    )

    summary = summarize(
        rows=rows,
        checkpoint_episode=(
            LOCKED_CHECKPOINT
        ),
    )

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    episode_csv = (
        OUTPUT_DIRECTORY
        / "final_test_episode_results.csv"
    )

    summary_json = (
        OUTPUT_DIRECTORY
        / "final_test_summary.json"
    )

    summary_csv = (
        OUTPUT_DIRECTORY
        / "final_test_summary.csv"
    )

    protocol_json = (
        OUTPUT_DIRECTORY
        / "final_test_protocol.json"
    )

    write_csv(
        rows,
        episode_csv,
    )

    with open(
        summary_json,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary,
            file,
            indent=4,
        )

    write_csv(
        [summary],
        summary_csv,
    )

    protocol = {
        "train_split_role":
            "RL policy learning",

        "validation_split_role":
            (
                "forecasting-method selection, "
                "Phi calibration, checkpoint selection"
            ),

        "test_split_role":
            "one-time final evaluation only",

        "locked_checkpoint_episode":
            LOCKED_CHECKPOINT,

        "checkpoint_selection_source":
            "validation",

        "deterministic_actions":
            True,

        "gradient_updates":
            False,

        "replay_writes":
            False,

        "checkpoint_ranking_on_test":
            False,

        "checkpoint_reselection_after_test":
            False,

        "test_windows_evaluated":
            number_of_test_episodes,

        "all_available_test_windows":
            all_test_windows,
    }

    with open(
        protocol_json,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            protocol,
            file,
            indent=4,
        )

    section(
        "FINAL TEST COMPLETE"
    )

    print(
        f"Checkpoint               : "
        f"{LOCKED_CHECKPOINT}"
    )

    print(
        f"Mean TEST return         : "
        f"{summary['mean_return']:.6f}"
    )

    print(
        f"TEST return std          : "
        f"{summary['std_return']:.6f}"
    )

    print(
        f"Median TEST return       : "
        f"{summary['median_return']:.6f}"
    )

    print(
        f"Mean net market cost     : "
        f"${summary['mean_net_market_cost_usd']:.6f}"
    )

    print(
        f"Mean grid import         : "
        f"{summary['mean_grid_import_kwh']:.6f} kWh/day"
    )

    print(
        f"Mean grid export         : "
        f"{summary['mean_grid_export_kwh']:.6f} kWh/day"
    )

    print(
        f"Mean BESS throughput     : "
        f"{summary['mean_bess_throughput_kwh']:.6f} kWh/day"
    )

    print(
        f"Mean sharing scheduled   : "
        f"{summary['mean_sharing_scheduled_kwh']:.6f} kWh/day"
    )

    print(
        f"Mean balance violation   : "
        f"{summary['mean_mean_balance_violation_kw_per_step']:.9f} kW/step"
    )

    print(
        f"Transformer violations   : "
        f"{summary['total_transformer_violation_steps']}"
    )

    print(
        f"SOC bound violations     : "
        f"{summary['total_soc_below_min_count'] + summary['total_soc_above_max_count']}"
    )

    print()
    print("Results:")
    print(episode_csv)
    print(summary_csv)
    print(summary_json)
    print(protocol_json)

    print()
    print(
        "[OK] TEST evaluated checkpoint 200 only."
    )

    print(
        "[OK] No checkpoint comparison or re-selection was performed."
    )

    print(
        "[OK] No RL training/update operation was executed."
    )

    print(
        "[OK] Final TEST result is now locked for reporting."
    )


if __name__ == "__main__":
    main()
