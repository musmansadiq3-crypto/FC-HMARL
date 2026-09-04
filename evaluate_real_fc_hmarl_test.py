from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import evaluate_real_fc_hmarl_validation as val

from marl.vpp_training_bridge import VPPExogenousInput
from train_real_fc_hmarl import (
    NUMBER_OF_MICROGRIDS,
    EPISODE_LENGTH,
    EV_COUNTS,
    EV_CHARGER_POWER_KW,
    EV_SIMULTANEOUS_FRACTION,
    PEAK_LOAD_KW,
    TRAIN_LOAD_MAX,
    TRAIN_EV_MIN,
    TRAIN_EV_MAX,
    TRAIN_PRICE_MIN,
    TRAIN_PRICE_MAX,
    MINIMUM_BUY_PRICE,
    MAXIMUM_BUY_PRICE,
    MINIMUM_SELL_PRICE,
    MAXIMUM_SELL_PRICE,
    build_per_ev_charging_requests,
    safe_fraction,
)

PROJECT_ROOT = Path(r"D:\Molvi paper review\FC_HMARL")

SEQUENCE_FILE = (
    PROJECT_ROOT / "data" / "processed" / "forecasting"
    / "forecasting_sequences.npz"
)

TEST_TRANSFORMER_FILE = (
    PROJECT_ROOT / "outputs" / "forecasting"
    / "real_forecasting_test_predictions.npz"
)

CAUSAL_CONFIDENCE_FILE = (
    PROJECT_ROOT / "outputs" / "forecasting" / "causal_confidence"
    / "causal_confidence_24h.csv"
)

SCALER_FILE = (
    PROJECT_ROOT / "data" / "processed" / "forecasting"
    / "forecasting_scaler.csv"
)

OUTPUT_DIRECTORY = (
    PROJECT_ROOT / "outputs" / "evaluation" / "real_fc_hmarl_test"
)

NUMBER_OF_TEST_EPISODES = 30
RANDOM_SEED = 42
USE_LEAD_ONE_CONFIDENCE = True


def section(title: str) -> None:
    print()
    print("=" * 80)
    print(title)
    print("=" * 80)


def inverse_transform(normalized, minimums, ranges):
    normalized = np.asarray(normalized, dtype=float)
    return normalized * ranges + minimums


class TestData:
    def __init__(self):
        section("LOADING HELD-OUT TEST DATA")

        sequences = np.load(SEQUENCE_FILE, allow_pickle=True)

        self.X_test = np.asarray(
            sequences["X_test"], dtype=np.float32
        )
        self.y_test = np.asarray(
            sequences["y_test"], dtype=np.float32
        )
        self.idx_test = np.asarray(sequences["idx_test"])
        self.feature_names = np.asarray(sequences["feature_names"])

        print(f"X_test      : {self.X_test.shape}")
        print(f"y_test      : {self.y_test.shape}")

        transformer = np.load(
            TEST_TRANSFORMER_FILE, allow_pickle=True
        )

        # Support the archive names already used by the forecasting pipeline.
        if "predictions_normalized" in transformer.files:
            transformer_normalized = transformer["predictions_normalized"]
        elif "predictions" in transformer.files:
            transformer_normalized = transformer["predictions"]
        elif "y_pred_normalized" in transformer.files:
            transformer_normalized = transformer["y_pred_normalized"]
        else:
            raise RuntimeError(
                "Could not find normalized Transformer predictions in "
                f"{TEST_TRANSFORMER_FILE}. Available keys: "
                f"{transformer.files}"
            )

        self.transformer_normalized = np.asarray(
            transformer_normalized, dtype=np.float32
        )

        if self.transformer_normalized.shape != self.y_test.shape:
            raise RuntimeError(
                "TEST Transformer prediction shape does not match y_test: "
                f"{self.transformer_normalized.shape} vs "
                f"{self.y_test.shape}"
            )

        scaler_df = pd.read_csv(SCALER_FILE)
        self.minimums = scaler_df["train_min"].to_numpy(dtype=float)
        self.ranges = scaler_df["train_range"].to_numpy(dtype=float)

        # Validation-selected mapping:
        # PV=daily seasonal, Load=Transformer,
        # EV=Transformer, Price=daily seasonal.
        daily = self.X_test[:, -24:, :].copy()

        hybrid = np.empty_like(daily, dtype=np.float32)
        hybrid[:, :, 0] = daily[:, :, 0]
        hybrid[:, :, 1] = self.transformer_normalized[:, :, 1]
        hybrid[:, :, 2] = self.transformer_normalized[:, :, 2]
        hybrid[:, :, 3] = daily[:, :, 3]

        self.forecast_normalized = np.clip(
            hybrid, 0.0, None
        ).astype(np.float32)

        self.forecast_original = inverse_transform(
            self.forecast_normalized,
            self.minimums,
            self.ranges,
        ).astype(np.float32)

        # Only the current lead-1 realization enters the physical system.
        self.current_actual_normalized = (
            self.y_test[:, 0, :].astype(np.float32)
        )
        self.current_actual_original = inverse_transform(
            self.current_actual_normalized,
            self.minimums,
            self.ranges,
        ).astype(np.float32)

        confidence_df = pd.read_csv(CAUSAL_CONFIDENCE_FILE)
        if "Phi_causal" not in confidence_df.columns:
            raise RuntimeError(
                "Phi_causal column was not found."
            )

        self.phi = confidence_df[
            "Phi_causal"
        ].to_numpy(dtype=np.float32)

        if self.phi.shape != (24,):
            raise RuntimeError(
                "Causal Phi must contain 24 values."
            )

        # Manuscript predictive state: S_pred = Phi * Z_hat.
        self.predictive_state = (
            self.forecast_normalized
            * self.phi[None, :, None]
        ).astype(np.float32)

        self.predictive_state_flat = (
            self.predictive_state.reshape(
                len(self.predictive_state), -1
            ).astype(np.float32)
        )

        if self.predictive_state_flat.shape[1] != 96:
            raise RuntimeError(
                "Expected 96-dimensional S_pred."
            )

        self.number_of_samples = len(self.X_test)

        print(
            "Transformer : "
            f"{self.transformer_normalized.shape}"
        )
        print(
            "Hybrid forecast : "
            f"{self.forecast_normalized.shape}"
        )
        print(
            "Predictive state: "
            f"{self.predictive_state.shape}"
        )
        print(
            "Flat S_pred     : "
            f"{self.predictive_state_flat.shape}"
        )
        print(f"Lead-1 Phi      : {self.phi[0]:.8f}")
        print(f"Mean Phi        : {self.phi.mean():.8f}")


def build_fixed_test_episode_starts(
    data: TestData,
    number_of_episodes: int,
    seed: int,
):
    maximum_start = data.number_of_samples - EPISODE_LENGTH

    if maximum_start < 0:
        raise RuntimeError(
            "Not enough TEST samples."
        )

    candidate_starts = np.arange(
        maximum_start + 1, dtype=int
    )

    if number_of_episodes > len(candidate_starts):
        raise ValueError(
            "Requested more TEST episodes than available."
        )

    rng = np.random.default_rng(seed)
    starts = rng.choice(
        candidate_starts,
        size=number_of_episodes,
        replace=False,
    )

    return np.sort(starts).astype(int)


class TestExogenousProvider:
    def __init__(self, data: TestData, episode_starts):
        self.data = data
        self.episode_starts = np.asarray(
            episode_starts, dtype=int
        )

    def get_index(self, episode: int, step: int) -> int:
        episode_index = int(episode) - 1

        if not 0 <= episode_index < len(self.episode_starts):
            raise IndexError("Invalid TEST episode.")

        if not 0 <= int(step) < EPISODE_LENGTH:
            raise IndexError("Invalid TEST step.")

        return (
            int(self.episode_starts[episode_index])
            + int(step)
        )

    def __call__(
        self, episode: int, step: int
    ) -> VPPExogenousInput:
        index = self.get_index(episode, step)

        actual = self.data.current_actual_original[index]

        pv_reference = float(actual[0])
        load_reference = float(actual[1])
        ev_reference = float(actual[2])
        price_reference = float(actual[3])

        irradiance = float(
            np.clip(pv_reference * 1000.0, 0.0, 1000.0)
        )
        irradiances = np.full(
            NUMBER_OF_MICROGRIDS,
            irradiance,
            dtype=float,
        )

        # Fixed TRAIN scales only: no TEST-derived rescaling.
        load_shape = safe_fraction(
            load_reference, 0.0, TRAIN_LOAD_MAX
        )
        loads = PEAK_LOAD_KW * load_shape

        ev_shape = safe_fraction(
            ev_reference, TRAIN_EV_MIN, TRAIN_EV_MAX
        )
        maximum_ev_power = (
            EV_COUNTS.astype(float)
            * EV_CHARGER_POWER_KW
            * EV_SIMULTANEOUS_FRACTION
        )
        aggregate_ev_power = maximum_ev_power * ev_shape
        per_ev_requests = build_per_ev_charging_requests(
            aggregate_ev_power
        )

        price_shape = safe_fraction(
            price_reference,
            TRAIN_PRICE_MIN,
            TRAIN_PRICE_MAX,
        )

        buy_price = (
            MINIMUM_BUY_PRICE
            + price_shape
            * (MAXIMUM_BUY_PRICE - MINIMUM_BUY_PRICE)
        )
        sell_price = (
            MINIMUM_SELL_PRICE
            + price_shape
            * (MAXIMUM_SELL_PRICE - MINIMUM_SELL_PRICE)
        )

        buy_prices = np.full(
            NUMBER_OF_MICROGRIDS,
            buy_price,
            dtype=float,
        )
        sell_prices = np.full(
            NUMBER_OF_MICROGRIDS,
            sell_price,
            dtype=float,
        )

        predictive_state = (
            self.data.predictive_state[index].copy()
        )

        scalar_confidence = float(
            self.data.phi[0]
            if USE_LEAD_ONE_CONFIDENCE
            else self.data.phi.mean()
        )

        return VPPExogenousInput(
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


def build_test_bridge(data, episode_starts):
    # Reuse the already validated bridge construction, replacing
    # only its exogenous provider with the TEST provider.
    bridge = val.build_validation_bridge(
        data=data,
        episode_starts=episode_starts,
    )
    bridge.exogenous_provider = TestExogenousProvider(
        data=data,
        episode_starts=episode_starts,
    )
    return bridge


def evaluate_one_test_episode(
    bridge,
    local_agents,
    coordinator_agent,
    episode_number,
):
    observation = bridge.reset(episode=episode_number)

    total_return = 0.0
    coordinator_return = 0.0
    local_returns = np.zeros(
        NUMBER_OF_MICROGRIDS, dtype=float
    )
    physical_balance_violations = 0
    transformer_violations = 0
    steps = 0

    with torch.no_grad():
        for _ in range(EPISODE_LENGTH):
            actions = val.select_deterministic_actions(
                local_agents=local_agents,
                coordinator_agent=coordinator_agent,
                observation=observation,
            )

            result = bridge.step(actions)

            local_reward_vector = np.asarray(
                result.local_rewards, dtype=float
            )
            coordinator_reward = float(
                result.coordinator_reward
            )

            total_return += (
                float(local_reward_vector.sum())
                + coordinator_reward
            )
            coordinator_return += coordinator_reward
            local_returns += local_reward_vector

            info = result.info
            if isinstance(info, dict):
                constraints = info.get("constraints", {})
                if isinstance(constraints, dict):
                    if not constraints.get(
                        "all_power_balanced", True
                    ):
                        physical_balance_violations += 1
                    if not constraints.get(
                        "all_transformers_feasible", True
                    ):
                        transformer_violations += 1

            observation = result.next_observation
            steps += 1

            if result.done:
                break

    return {
        "test_episode": int(episode_number),
        "steps": int(steps),
        "total_return": float(total_return),
        "coordinator_return": float(coordinator_return),
        "sum_local_return": float(local_returns.sum()),
        "mean_local_return": float(local_returns.mean()),
        "mg1_return": float(local_returns[0]),
        "mg2_return": float(local_returns[1]),
        "mg3_return": float(local_returns[2]),
        "mg4_return": float(local_returns[3]),
        "mg5_return": float(local_returns[4]),
        "power_balance_violation_steps": int(
            physical_balance_violations
        ),
        "transformer_violation_steps": int(
            transformer_violations
        ),
    }


def summarize(dataframe, checkpoint):
    returns = dataframe[
        "total_return"
    ].to_numpy(dtype=float)

    return {
        "checkpoint_episode": int(checkpoint),
        "test_episodes": int(len(dataframe)),
        "mean_return": float(np.mean(returns)),
        "std_return": float(np.std(returns, ddof=0)),
        "median_return": float(np.median(returns)),
        "minimum_return": float(np.min(returns)),
        "maximum_return": float(np.max(returns)),
        "mean_coordinator_return": float(
            dataframe["coordinator_return"].mean()
        ),
        "mean_local_return": float(
            dataframe["mean_local_return"].mean()
        ),
        "power_balance_violation_steps": int(
            dataframe[
                "power_balance_violation_steps"
            ].sum()
        ),
        "transformer_violation_steps": int(
            dataframe[
                "transformer_violation_steps"
            ].sum()
        ),
    }


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoint",
        type=int,
        default=1000,
    )
    parser.add_argument(
        "--test-episodes",
        type=int,
        default=NUMBER_OF_TEST_EPISODES,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=RANDOM_SEED,
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda", "auto"],
    )

    return parser.parse_args()


def main():
    args = parse_arguments()

    section(
        "FC-HMARL STEP 7L-D - FINAL HELD-OUT TEST"
    )

    # Lock final testing to the model selected on VALIDATION.
    if args.checkpoint != 1000:
        raise ValueError(
            "Final TEST is locked to validation-selected "
            "checkpoint 1000."
        )

    if args.test_episodes < 1:
        raise ValueError(
            "At least one TEST episode is required."
        )

    device = val.resolve_device(args.device)
    val.set_global_seed(args.seed)

    print(f"Checkpoint          : {args.checkpoint}")
    print(f"Test episodes       : {args.test_episodes}")
    print("Evaluation actions  : DETERMINISTIC")
    print(f"Seed                : {args.seed}")
    print(f"Device              : {device}")
    print("Checkpoint selection: VALIDATION ONLY")
    print("Learning            : DISABLED")
    print("TEST data used      : YES - FINAL EVALUATION ONLY")

    data = TestData()

    starts = build_fixed_test_episode_starts(
        data=data,
        number_of_episodes=args.test_episodes,
        seed=args.seed,
    )

    section("FIXED TEST EPISODE STARTS")
    print(starts)

    OUTPUT_DIRECTORY.mkdir(
        parents=True, exist_ok=True
    )

    pd.DataFrame({
        "test_episode": np.arange(
            1, len(starts) + 1
        ),
        "start_index": starts,
    }).to_csv(
        OUTPUT_DIRECTORY / "fixed_test_episode_starts.csv",
        index=False,
    )

    local_agents, coordinator_agent = val.load_policy(
        checkpoint_episode=args.checkpoint,
        device=device,
        seed=args.seed,
    )

    bridge = build_test_bridge(
        data=data,
        episode_starts=starts,
    )

    records = []

    for episode_number in range(
        1, len(starts) + 1
    ):
        result = evaluate_one_test_episode(
            bridge=bridge,
            local_agents=local_agents,
            coordinator_agent=coordinator_agent,
            episode_number=episode_number,
        )

        result["checkpoint_episode"] = args.checkpoint
        result["test_start_index"] = int(
            starts[episode_number - 1]
        )
        records.append(result)

        print(
            f"Test episode {episode_number:2d}/"
            f"{len(starts)} | Return: "
            f"{result['total_return']:.6f}"
        )

    dataframe = pd.DataFrame(records)

    dataframe.to_csv(
        OUTPUT_DIRECTORY
        / f"checkpoint_{args.checkpoint}_test_results.csv",
        index=False,
    )

    summary = summarize(
        dataframe, args.checkpoint
    )

    summary_df = pd.DataFrame([summary])
    summary_df.to_csv(
        OUTPUT_DIRECTORY / "final_test_summary.csv",
        index=False,
    )

    with open(
        OUTPUT_DIRECTORY / "final_test_summary.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(summary, file, indent=4)

    section("FINAL HELD-OUT TEST RESULTS")
    print(summary_df.to_string(index=False))

    print()
    print("[OK] Validation-selected checkpoint 1000 used.")
    print("[OK] Deterministic policy evaluation used.")
    print("[OK] No learning or parameter updates occurred.")
    print("[OK] TEST used only after checkpoint selection.")
    print("[OK] Validation-selected forecast mapping retained.")
    print("[OK] Validation-calibrated causal Phi retained.")

    if (
        summary["power_balance_violation_steps"] == 0
        and summary["transformer_violation_steps"] == 0
    ):
        print(
            "[OK] No recorded power-balance or "
            "transformer violations."
        )
    else:
        print(
            "[WARNING] Physical constraint violations "
            "were recorded."
        )

    print()
    print(
        f"Evaluation directory:\n{OUTPUT_DIRECTORY}"
    )


if __name__ == "__main__":
    main()
