# ============================================================
# FC-HMARL FINAL V3
# VALIDATION CHECKPOINT SELECTION
# ============================================================
#
# PURPOSE
# -------
# Evaluate FINAL V3 checkpoints on VALIDATION only.
# No training, no optimizer updates, no replay-buffer writes.
# Deterministic SAC actions only.
#
# IMPORTANT
# ---------
# This program expects a validation RL archive with the same core
# schema used by the training archive:
#   current_actual_original
#   current_actual_normalized
#   forecast_original
#   forecast_normalized
#   predictive_state_matrix
#   predictive_state_flat
#   causal_confidence_24h
#
# Default validation archive:
#   outputs/rl_data/real_rl_validation_archive.npz
#
# It imports the FINAL V3 training launcher so the physical VPP,
# action mapper, state builder, reward builder, and scaling constants
# are exactly the same as those used in training.
# ============================================================

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch

from marl.training_loop import HierarchicalActionBundle


PROJECT_ROOT = Path(r"D:\Molvi paper review\FC_HMARL")

V3_LAUNCHER = PROJECT_ROOT / "train_real_fc_hmarl_final_v3.py"

VALIDATION_ARCHIVE = (
    PROJECT_ROOT
    / "outputs"
    / "rl_data"
    / "real_rl_validation_archive.npz"
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
    / "real_fc_hmarl_final_v3_validation"
)


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
        raise RuntimeError("Could not import FINAL V3 launcher.")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ValidationData:
    """Read-only validation archive with the V3 data interface."""

    REQUIRED = (
        "current_actual_original",
        "current_actual_normalized",
        "forecast_original",
        "forecast_normalized",
        "predictive_state_matrix",
        "predictive_state_flat",
        "causal_confidence_24h",
    )

    def __init__(self, archive_path: Path):
        if not archive_path.exists():
            raise FileNotFoundError(
                "Validation archive not found:\n"
                f"{archive_path}\n\n"
                "Do NOT substitute the TRAIN or TEST archive. "
                "Create the validation archive first."
            )

        archive = np.load(archive_path, allow_pickle=True)

        missing = [
            key for key in self.REQUIRED
            if key not in archive.files
        ]

        if missing:
            raise RuntimeError(
                "Validation archive is missing keys:\n  "
                + "\n  ".join(missing)
            )

        self.current_actual_original = np.asarray(
            archive["current_actual_original"], dtype=np.float32
        )
        self.current_actual_normalized = np.asarray(
            archive["current_actual_normalized"], dtype=np.float32
        )
        self.forecast_original = np.asarray(
            archive["forecast_original"], dtype=np.float32
        )
        self.forecast_normalized = np.asarray(
            archive["forecast_normalized"], dtype=np.float32
        )
        self.predictive_state_matrix = np.asarray(
            archive["predictive_state_matrix"], dtype=np.float32
        )
        self.predictive_state_flat = np.asarray(
            archive["predictive_state_flat"], dtype=np.float32
        )
        self.causal_confidence_24h = np.asarray(
            archive["causal_confidence_24h"], dtype=np.float32
        )

        self.number_of_samples = len(
            self.current_actual_original
        )

        if self.current_actual_original.shape != (
            self.number_of_samples, 4
        ):
            raise RuntimeError(
                "Unexpected validation physical-data shape."
            )

        if self.predictive_state_matrix.shape != (
            self.number_of_samples, 24, 4
        ):
            raise RuntimeError(
                "Unexpected validation predictive-state shape."
            )

        if self.predictive_state_flat.shape != (
            self.number_of_samples, 96
        ):
            raise RuntimeError(
                "Unexpected validation flattened predictive-state shape."
            )

        if self.causal_confidence_24h.shape != (24,):
            raise RuntimeError(
                "Expected 24 validation causal-confidence values."
            )

        if self.number_of_samples < 24:
            raise RuntimeError(
                "Validation archive is too short for one 24-hour episode."
            )


class FixedValidationProvider:
    """
    Deterministic validation provider.

    Every checkpoint receives the exact same episode starts.
    Starts are selected once from VALIDATION using a fixed seed.
    """

    def __init__(
        self,
        v3,
        data: ValidationData,
        number_of_episodes: int,
        seed: int,
    ):
        self.v3 = v3
        self.data = data
        self.number_of_episodes = int(number_of_episodes)
        self.seed = int(seed)

        maximum_start = (
            data.number_of_samples
            - v3.EPISODE_LENGTH
        )

        valid_starts = np.arange(
            maximum_start + 1,
            dtype=int,
        )

        if self.number_of_episodes > len(valid_starts):
            raise ValueError(
                "Requested more validation episodes than available "
                "24-hour starting positions."
            )

        rng = np.random.default_rng(self.seed)

        self.episode_starts = rng.choice(
            valid_starts,
            size=self.number_of_episodes,
            replace=False,
        )

    def get_sample_index(self, episode: int, step: int) -> int:
        episode_index = int(episode) - 1

        if not 0 <= episode_index < self.number_of_episodes:
            raise IndexError("Invalid validation episode.")

        if not 0 <= int(step) < self.v3.EPISODE_LENGTH:
            raise IndexError("Invalid validation step.")

        return int(
            self.episode_starts[episode_index]
            + int(step)
        )

    def __call__(self, episode: int, step: int):
        # Reuse the exact V3 physical conversion logic by constructing
        # a lightweight V3 provider object and selecting our fixed index.
        index = self.get_sample_index(episode, step)

        actual = self.data.current_actual_original[index]

        pv_reference = float(actual[0])
        load_reference = float(actual[1])
        ev_reference = float(actual[2])
        price_reference = float(actual[3])

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

        load_shape = self.v3.safe_fraction(
            load_reference,
            0.0,
            self.v3.TRAIN_LOAD_MAX,
        )

        loads = (
            self.v3.PEAK_LOAD_KW
            * load_shape
        )

        ev_shape = self.v3.safe_fraction(
            ev_reference,
            self.v3.TRAIN_EV_MIN,
            self.v3.TRAIN_EV_MAX,
        )

        maximum_ev_power = (
            self.v3.EV_COUNTS.astype(float)
            * self.v3.EV_CHARGER_POWER_KW
            * self.v3.EV_SIMULTANEOUS_FRACTION
        )

        aggregate_ev_requests = (
            maximum_ev_power * ev_shape
        )

        per_ev_requests = (
            self.v3.build_per_ev_charging_requests(
                aggregate_ev_requests
            )
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
            self.data.predictive_state_matrix[index].copy()
        )

        scalar_confidence = float(
            self.data.causal_confidence_24h[0]
        )

        return self.v3.VPPExogenousInput(
            time=float(step),
            loads_kw=loads,
            irradiances_w_m2=irradiances,
            buy_prices_usd_per_kwh=buy_prices,
            sell_prices_usd_per_kwh=sell_prices,
            predictive_state=predictive_state,
            confidence=scalar_confidence,
            market_price=price_reference,
            ev_requested_charging_powers_kw=per_ev_requests,
        )


def discover_checkpoint_episodes():
    episodes = []

    for episode in range(100, 5001, 100):
        coordinator = (
            CHECKPOINT_DIRECTORY
            / f"coordinator_episode_{episode}.pt"
        )

        local_files = [
            CHECKPOINT_DIRECTORY
            / f"local_agent_{i}_episode_{episode}.pt"
            for i in range(1, 6)
        ]

        if coordinator.exists() and all(
            path.exists() for path in local_files
        ):
            episodes.append(episode)

    if not episodes:
        raise FileNotFoundError(
            "No complete FINAL V3 evaluation checkpoints were found in:\n"
            f"{CHECKPOINT_DIRECTORY}"
        )

    return episodes


def load_policy_checkpoint(
    v3,
    local_agents,
    coordinator_agent,
    episode: int,
):
    """
    Evaluation load only.

    Optimizers and replay buffers are intentionally not loaded.
    """

    for agent in local_agents:
        path = (
            CHECKPOINT_DIRECTORY
            / (
                f"local_agent_{agent.microgrid_id}_"
                f"episode_{episode}.pt"
            )
        )

        if not path.exists():
            raise FileNotFoundError(path)

        agent.load(
            path,
            load_optimizers=False,
        )

    coordinator_path = (
        CHECKPOINT_DIRECTORY
        / f"coordinator_episode_{episode}.pt"
    )

    if not coordinator_path.exists():
        raise FileNotFoundError(coordinator_path)

    coordinator_agent.load(
        coordinator_path,
        load_optimizers=False,
    )


def scalar(value, default=0.0):
    if value is None:
        return float(default)

    arr = np.asarray(value, dtype=float)

    if arr.size == 0:
        return float(default)

    return float(np.sum(arr))


def evaluate_checkpoint(
    v3,
    checkpoint_episode: int,
    data: ValidationData,
    number_of_validation_episodes: int,
    validation_seed: int,
    device: str,
):
    # Fixed seed makes agent construction reproducible; checkpoint
    # weights replace fresh network weights before evaluation.
    v3.set_global_seed(validation_seed)

    bridge = v3.build_real_training_bridge(
        data=data,
        maximum_episodes=number_of_validation_episodes,
        seed=validation_seed,
    )

    # Replace training schedule with the fixed validation schedule.
    bridge.exogenous_provider = FixedValidationProvider(
        v3=v3,
        data=data,
        number_of_episodes=number_of_validation_episodes,
        seed=validation_seed,
    )

    local_agents = v3.build_local_agents(
        seed=validation_seed,
        smoke_test=False,
        device=device,
    )

    coordinator_agent = v3.build_coordinator_agent(
        seed=validation_seed,
        smoke_test=False,
        device=device,
    )

    load_policy_checkpoint(
        v3=v3,
        local_agents=local_agents,
        coordinator_agent=coordinator_agent,
        episode=checkpoint_episode,
    )

    total_returns = []
    local_return_rows = []
    coordinator_returns = []

    total_balance_violation = 0.0
    maximum_balance_violation = 0.0
    transformer_violation_steps = 0
    transformer_checks = 0

    market_profit_total = 0.0
    purchase_cost_total = 0.0
    sale_revenue_total = 0.0
    reserve_revenue_total = 0.0

    with torch.no_grad():

        for episode in range(
            1,
            number_of_validation_episodes + 1,
        ):
            observation = bridge.reset(
                episode=episode
            )

            episode_local_returns = np.zeros(
                v3.NUMBER_OF_MICROGRIDS,
                dtype=float,
            )

            episode_coordinator_return = 0.0

            for step in range(v3.EPISODE_LENGTH):

                local_actions = []

                for agent, state in zip(
                    local_agents,
                    observation.local_states,
                ):
                    action = agent.select_action(
                        state,
                        deterministic=True,
                    )

                    local_actions.append(
                        np.asarray(action, dtype=float)
                    )

                coordinator_action = (
                    coordinator_agent.select_action(
                        observation.coordinator_state,
                        deterministic=True,
                    )
                )

                actions = HierarchicalActionBundle(
                    local_actions=[
                        np.asarray(
                            action,
                            dtype=np.float32,
                        )
                        for action in local_actions
                    ],
                    coordinator_action=np.asarray(
                        coordinator_action,
                        dtype=np.float32,
                    ),
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

                episode_local_returns += local_rewards
                episode_coordinator_return += (
                    coordinator_reward
                )

                info = getattr(
                    result,
                    "info",
                    {},
                )

                if not isinstance(info, dict):
                    info = {}

                env_info = info.get(
                    "physical_result",
                    getattr(
                        bridge,
                        "last_physical_result",
                        {},
                    ),
                )

                if not isinstance(env_info, dict):
                    env_info = {}

                constraints = env_info.get(
                    "constraints", {}
                )

                balance_violation = scalar(
                    constraints.get(
                        "total_power_balance_violation_kw",
                        0.0,
                    )
                )

                total_balance_violation += (
                    balance_violation
                )

                maximum_balance_violation = max(
                    maximum_balance_violation,
                    balance_violation,
                )

                local_results = env_info.get(
                    "local_results", []
                )

                for local_result in local_results:
                    transformer_checks += 1

                    if not bool(
                        local_result.get(
                            "transformer_feasible",
                            True,
                        )
                    ):
                        transformer_violation_steps += 1

                totals = env_info.get(
                    "totals", {}
                )

                market_profit_total += scalar(
                    totals.get(
                        "market_profit_usd",
                        0.0,
                    )
                )

                purchase_cost_total += scalar(
                    totals.get(
                        "grid_purchase_cost_usd",
                        totals.get(
                            "purchase_cost_usd",
                            0.0,
                        ),
                    )
                )

                sale_revenue_total += scalar(
                    totals.get(
                        "grid_sale_revenue_usd",
                        totals.get(
                            "sale_revenue_usd",
                            0.0,
                        ),
                    )
                )

                reserve_revenue_total += scalar(
                    totals.get(
                        "reserve_revenue_usd",
                        0.0,
                    )
                )

                observation = result.next_observation

            local_return_rows.append(
                episode_local_returns.copy()
            )

            coordinator_returns.append(
                episode_coordinator_return
            )

            total_returns.append(
                float(
                    episode_local_returns.sum()
                    + episode_coordinator_return
                )
            )

    returns = np.asarray(
        total_returns,
        dtype=float,
    )

    local_matrix = np.asarray(
        local_return_rows,
        dtype=float,
    )

    coordinator_array = np.asarray(
        coordinator_returns,
        dtype=float,
    )

    total_steps = (
        number_of_validation_episodes
        * v3.EPISODE_LENGTH
    )

    row = {
        "checkpoint_episode":
            int(checkpoint_episode),

        "validation_episodes":
            int(number_of_validation_episodes),

        "mean_return":
            float(returns.mean()),

        "std_return":
            float(returns.std()),

        "minimum_return":
            float(returns.min()),

        "maximum_return":
            float(returns.max()),

        "median_return":
            float(np.median(returns)),

        "coordinator_mean_return":
            float(coordinator_array.mean()),

        "total_balance_violation_kw":
            float(total_balance_violation),

        "mean_balance_violation_kw_per_step":
            float(
                total_balance_violation
                / max(total_steps, 1)
            ),

        "maximum_balance_violation_kw":
            float(maximum_balance_violation),

        "transformer_violation_count":
            int(transformer_violation_steps),

        "transformer_checks":
            int(transformer_checks),

        "all_transformers_feasible":
            bool(
                transformer_violation_steps == 0
            ),

        "market_profit_total_usd":
            float(market_profit_total),

        "grid_purchase_cost_total_usd":
            float(purchase_cost_total),

        "grid_sale_revenue_total_usd":
            float(sale_revenue_total),

        "reserve_revenue_total_usd":
            float(reserve_revenue_total),
    }

    for mg_index in range(
        v3.NUMBER_OF_MICROGRIDS
    ):
        row[
            f"mg{mg_index + 1}_mean_return"
        ] = float(
            local_matrix[:, mg_index].mean()
        )

    return row


def rank_results(rows):
    """
    Feasibility-first checkpoint ranking.

    1) Transformer feasibility
    2) Lower mean balance violation
    3) Higher mean validation return
    4) Lower return standard deviation
    """

    return sorted(
        rows,
        key=lambda row: (
            0 if row["all_transformers_feasible"] else 1,
            row["mean_balance_violation_kw_per_step"],
            -row["mean_return"],
            row["std_return"],
            row["checkpoint_episode"],
        ),
    )


def write_csv(rows, path: Path):
    if not rows:
        return

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = list(rows[0].keys())

    with open(
        path,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(rows)


def parse_arguments():
    parser = argparse.ArgumentParser(
        description=(
            "Select the FINAL V3 FC-HMARL checkpoint using "
            "VALIDATION only."
        )
    )

    parser.add_argument(
        "--validation-episodes",
        type=int,
        default=50,
        help=(
            "Number of fixed 24-hour validation episodes "
            "used for every checkpoint."
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=2026,
        help=(
            "Fixed validation scenario-selection seed."
        ),
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda", "auto"],
    )

    parser.add_argument(
        "--start-checkpoint",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--end-checkpoint",
        type=int,
        default=5000,
    )

    parser.add_argument(
        "--checkpoint-step",
        type=int,
        default=100,
    )

    return parser.parse_args()


def main():
    args = parse_arguments()

    if args.validation_episodes <= 0:
        raise ValueError(
            "--validation-episodes must be positive."
        )

    section(
        "FC-HMARL FINAL V3 VALIDATION CHECKPOINT SELECTION"
    )

    v3 = load_v3_module()

    device = v3.resolve_device(
        args.device
    )

    print(f"Device                  : {device}")
    print(f"Validation archive      : {VALIDATION_ARCHIVE}")
    print(f"Checkpoint directory    : {CHECKPOINT_DIRECTORY}")
    print(f"Validation episodes     : {args.validation_episodes}")
    print(f"Validation seed         : {args.seed}")
    print("Deterministic actions    : True")
    print("Gradient updates         : False")
    print("Replay writes            : False")
    print("TEST data loaded         : False")

    section(
        "LOADING VALIDATION ARCHIVE"
    )

    data = ValidationData(
        VALIDATION_ARCHIVE
    )

    print(
        f"Validation samples       : "
        f"{data.number_of_samples}"
    )

    print(
        f"Predictive-state matrix  : "
        f"{data.predictive_state_matrix.shape}"
    )

    available = discover_checkpoint_episodes()

    requested = list(
        range(
            int(args.start_checkpoint),
            int(args.end_checkpoint) + 1,
            int(args.checkpoint_step),
        )
    )

    checkpoints = [
        episode
        for episode in requested
        if episode in available
    ]

    if not checkpoints:
        raise RuntimeError(
            "No requested complete checkpoints are available."
        )

    section(
        "EVALUATING CHECKPOINTS ON VALIDATION"
    )

    rows = []

    for index, checkpoint_episode in enumerate(
        checkpoints,
        start=1,
    ):
        print(
            f"[{index:02d}/{len(checkpoints):02d}] "
            f"Checkpoint {checkpoint_episode:4d} ... ",
            end="",
            flush=True,
        )

        row = evaluate_checkpoint(
            v3=v3,
            checkpoint_episode=checkpoint_episode,
            data=data,
            number_of_validation_episodes=(
                args.validation_episodes
            ),
            validation_seed=args.seed,
            device=device,
        )

        rows.append(row)

        print(
            f"mean={row['mean_return']:.6f} | "
            f"std={row['std_return']:.6f} | "
            f"balance="
            f"{row['mean_balance_violation_kw_per_step']:.6f} | "
            f"transformer="
            f"{'OK' if row['all_transformers_feasible'] else 'FAIL'}"
        )

    ranked = rank_results(rows)

    for rank, row in enumerate(
        ranked,
        start=1,
    ):
        row["validation_rank"] = rank

    # Re-sort a copy by checkpoint for convenient audit.
    chronological = sorted(
        ranked,
        key=lambda row: row["checkpoint_episode"],
    )

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    raw_csv = (
        OUTPUT_DIRECTORY
        / "validation_checkpoint_results.csv"
    )

    ranked_csv = (
        OUTPUT_DIRECTORY
        / "validation_checkpoint_ranking.csv"
    )

    write_csv(
        chronological,
        raw_csv,
    )

    write_csv(
        ranked,
        ranked_csv,
    )

    best = ranked[0]

    selection = {
        "selection_split": "validation",
        "test_data_used": False,
        "deterministic_actions": True,
        "gradient_updates": False,
        "replay_writes": False,
        "validation_seed": int(args.seed),
        "validation_episodes": int(
            args.validation_episodes
        ),
        "evaluated_checkpoints": [
            int(row["checkpoint_episode"])
            for row in chronological
        ],
        "ranking_rule": [
            "transformer_feasibility",
            "minimum_mean_balance_violation",
            "maximum_mean_validation_return",
            "minimum_return_standard_deviation",
        ],
        "selected_checkpoint_episode": int(
            best["checkpoint_episode"]
        ),
        "selected_checkpoint_metrics": best,
    }

    selection_file = (
        OUTPUT_DIRECTORY
        / "selected_validation_checkpoint.json"
    )

    with open(
        selection_file,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            selection,
            file,
            indent=4,
            default=str,
        )

    section(
        "VALIDATION CHECKPOINT SELECTION COMPLETE"
    )

    print(
        f"Selected checkpoint : "
        f"{best['checkpoint_episode']}"
    )

    print(
        f"Mean return         : "
        f"{best['mean_return']:.6f}"
    )

    print(
        f"Return std          : "
        f"{best['std_return']:.6f}"
    )

    print(
        f"Mean balance viol.  : "
        f"{best['mean_balance_violation_kw_per_step']:.9f} kW/step"
    )

    print(
        f"Transformer feasible: "
        f"{best['all_transformers_feasible']}"
    )

    print()
    print("Results:")
    print(raw_csv)
    print(ranked_csv)
    print(selection_file)

    print()
    print(
        "[OK] Checkpoint selection used VALIDATION only."
    )

    print(
        "[OK] Policies were evaluated deterministically."
    )

    print(
        "[OK] No SAC training/update operation was executed."
    )

    print(
        "[OK] TEST remains untouched."
    )


if __name__ == "__main__":
    main()
