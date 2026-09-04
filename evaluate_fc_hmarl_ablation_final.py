"""
Final FC-HMARL post-training ablation evaluator.

This script reuses the already validated held-out TEST pipeline and the locked
validation-selected checkpoint 1000.

IMPORTANT:
These are inference-time component ablations of one trained policy. They are
NOT separately retrained ablation agents. Report them as post-training
diagnostic ablations.

Modes
-----
1. full_fc_hmarl
   Original locked policy and original confidence-aware TEST pipeline.

2. no_confidence_awareness
   Removes confidence from BOTH places where Phi enters the reconstructed
   manuscript pipeline:
       S_pred = Phi * Z_hat  ->  Z_hat
       C_risk = rho(1-Phi)   ->  0 by setting Phi = 1
   Forecast information itself is retained.

3. no_energy_sharing
   Disables all local sharing requests and sets the coordinator sharing
   multiplier to zero.

4. no_upper_level_coordination
   Keeps local policies active, disables coordinator-controlled reserve/market
   intervention, and allows local sharing proposals to pass without an
   adaptive coordinator multiplier. This is an inference-time diagnostic
   reconstruction because the manuscript does not uniquely specify a
   software-level "coordinator off" action.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import evaluate_real_fc_hmarl_test as test_eval


PROJECT_ROOT = Path(r"D:\Molvi paper review\FC_HMARL")

OUTPUT_DIRECTORY = (
    PROJECT_ROOT
    / "outputs"
    / "evaluation"
    / "real_fc_hmarl_ablation"
)

LOCKED_CHECKPOINT = 1000

MODES = (
    "full_fc_hmarl",
    "no_confidence_awareness",
    "no_energy_sharing",
    "no_upper_level_coordination",
)


def section(title: str) -> None:
    print()
    print("=" * 80)
    print(title)
    print("=" * 80)


def parse_arguments():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate post-training FC-HMARL component ablations "
            "on fixed held-out TEST episodes."
        )
    )

    parser.add_argument(
        "--checkpoint",
        type=int,
        default=LOCKED_CHECKPOINT,
    )

    parser.add_argument(
        "--test-episodes",
        type=int,
        default=30,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
    )

    return parser.parse_args()


def build_mode_data(base_data, mode: str):
    """
    Create the data object used by one ablation mode.

    The physical TEST realizations are never changed.
    """
    data = copy.copy(base_data)

    if mode == "no_confidence_awareness":
        # Retain the selected hybrid forecast but remove Phi weighting:
        # S_pred = Z_hat.
        data.phi = np.ones_like(
            base_data.phi,
            dtype=np.float32,
        )

        data.predictive_state = np.asarray(
            base_data.forecast_normalized,
            dtype=np.float32,
        ).copy()

        data.predictive_state_flat = (
            data.predictive_state.reshape(
                len(data.predictive_state),
                -1,
            ).astype(np.float32)
        )

        if data.predictive_state_flat.shape[1] != 96:
            raise RuntimeError(
                "Expected 96-dimensional predictive state."
            )

    return data


def build_modified_actions(
    mode: str,
    local_agents,
    coordinator_agent,
    observation,
):
    """
    Select deterministic trained-policy actions and apply only the requested
    inference-time ablation.
    """
    original = test_eval.val.select_deterministic_actions(
        local_agents=local_agents,
        coordinator_agent=coordinator_agent,
        observation=observation,
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
        # Local reconstructed action:
        # [P_BESS, share_to_other_1, ..., share_to_other_4].
        #
        # The action mapper interprets nonpositive normalized sharing
        # commands as zero transfer. Use -1.0 to force no sharing.
        for action in local_actions:
            if action.size > 1:
                action[1:] = -1.0

        # Reconstructed coordinator software action dimension:
        # [market/grid-related, reserve, sharing multiplier].
        # action[2] = -1 maps to zero sharing multiplier.
        if coordinator_action.size >= 3:
            coordinator_action[2] = -1.0

    elif mode == "no_upper_level_coordination":
        # Fixed non-adaptive coordinator action.
        #
        # Market/grid coordinator control is disabled in the current physical
        # mapper, reserve is forced to zero, and sharing multiplier is forced
        # to one so local sharing proposals are not adaptively scaled by the
        # coordinator.
        coordinator_action[:] = 0.0

        if coordinator_action.size >= 1:
            coordinator_action[0] = -1.0

        if coordinator_action.size >= 2:
            coordinator_action[1] = -1.0

        if coordinator_action.size >= 3:
            coordinator_action[2] = 1.0

    else:
        raise ValueError(
            f"Unknown ablation mode: {mode}"
        )

    return test_eval.val.HierarchicalActionBundle(
        local_actions=local_actions,
        coordinator_action=coordinator_action,
    )


def evaluate_one_ablation_episode(
    *,
    mode,
    bridge,
    local_agents,
    coordinator_agent,
    episode_number,
):
    observation = bridge.reset(
        episode=episode_number
    )

    total_return = 0.0
    coordinator_return = 0.0

    local_returns = np.zeros(
        test_eval.NUMBER_OF_MICROGRIDS,
        dtype=float,
    )

    power_balance_violations = 0
    transformer_violations = 0
    steps = 0

    with torch.no_grad():

        for _ in range(test_eval.EPISODE_LENGTH):

            actions = build_modified_actions(
                mode=mode,
                local_agents=local_agents,
                coordinator_agent=coordinator_agent,
                observation=observation,
            )

            result = bridge.step(actions)

            local_reward_vector = np.asarray(
                result.local_rewards,
                dtype=float,
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

                constraints = info.get(
                    "constraints",
                    {},
                )

                if isinstance(constraints, dict):

                    if not constraints.get(
                        "all_power_balanced",
                        True,
                    ):
                        power_balance_violations += 1

                    if not constraints.get(
                        "all_transformers_feasible",
                        True,
                    ):
                        transformer_violations += 1

            observation = result.next_observation
            steps += 1

            if result.done:
                break

    return {
        "mode": mode,
        "test_episode": int(episode_number),
        "steps": int(steps),
        "total_return": float(total_return),
        "coordinator_return": float(
            coordinator_return
        ),
        "sum_local_return": float(
            local_returns.sum()
        ),
        "mean_local_return": float(
            local_returns.mean()
        ),
        "mg1_return": float(local_returns[0]),
        "mg2_return": float(local_returns[1]),
        "mg3_return": float(local_returns[2]),
        "mg4_return": float(local_returns[3]),
        "mg5_return": float(local_returns[4]),
        "power_balance_violation_steps": int(
            power_balance_violations
        ),
        "transformer_violation_steps": int(
            transformer_violations
        ),
    }


def summarize_mode(mode_df: pd.DataFrame):
    returns = mode_df[
        "total_return"
    ].to_numpy(dtype=float)

    return {
        "mode": str(
            mode_df["mode"].iloc[0]
        ),
        "episodes": int(len(mode_df)),
        "mean_return": float(
            np.mean(returns)
        ),
        "std_return": float(
            np.std(returns, ddof=0)
        ),
        "median_return": float(
            np.median(returns)
        ),
        "minimum_return": float(
            np.min(returns)
        ),
        "maximum_return": float(
            np.max(returns)
        ),
        "mean_coordinator_return": float(
            mode_df[
                "coordinator_return"
            ].mean()
        ),
        "mean_local_return_per_mg": float(
            mode_df[
                "mean_local_return"
            ].mean()
        ),
        "power_balance_violation_steps": int(
            mode_df[
                "power_balance_violation_steps"
            ].sum()
        ),
        "transformer_violation_steps": int(
            mode_df[
                "transformer_violation_steps"
            ].sum()
        ),
    }


def paired_comparison(
    full_df: pd.DataFrame,
    alternative_df: pd.DataFrame,
):
    full = full_df.sort_values(
        "test_episode"
    )["total_return"].to_numpy(dtype=float)

    alternative = alternative_df.sort_values(
        "test_episode"
    )["total_return"].to_numpy(dtype=float)

    if full.shape != alternative.shape:
        raise RuntimeError(
            "Paired comparison episode counts differ."
        )

    delta = alternative - full

    return {
        "comparison": (
            f"{alternative_df['mode'].iloc[0]}"
            "_minus_full"
        ),
        "mean_delta_return": float(
            np.mean(delta)
        ),
        "median_delta_return": float(
            np.median(delta)
        ),
        "std_delta_return": float(
            np.std(delta, ddof=0)
        ),
        "alternative_better_episodes": int(
            np.sum(delta > 0.0)
        ),
        "full_better_episodes": int(
            np.sum(delta < 0.0)
        ),
        "equal_episodes": int(
            np.sum(np.isclose(
                delta,
                0.0,
                atol=1e-12,
            ))
        ),
        "minimum_delta": float(
            np.min(delta)
        ),
        "maximum_delta": float(
            np.max(delta)
        ),
    }


def main():
    args = parse_arguments()

    if args.checkpoint != LOCKED_CHECKPOINT:
        raise ValueError(
            "This final TEST diagnostic is locked to "
            f"checkpoint {LOCKED_CHECKPOINT}. "
            "Do not use TEST ablations for checkpoint selection."
        )

    if args.test_episodes <= 0:
        raise ValueError(
            "--test-episodes must be > 0."
        )

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    section(
        "STEP 7M - FC-HMARL POST-TRAINING "
        "ABLATION EVALUATION"
    )

    print(
        f"Checkpoint        : {args.checkpoint}"
    )
    print(
        f"Test episodes     : {args.test_episodes}"
    )
    print(
        f"Seed              : {args.seed}"
    )
    print(
        f"Device            : {args.device}"
    )
    print(
        "Actions           : DETERMINISTIC"
    )
    print(
        "Learning          : DISABLED"
    )
    print(
        "Checkpoint select : VALIDATION ONLY"
    )
    print(
        "TEST use          : FINAL DIAGNOSTIC ONLY"
    )
    print(
        "Ablation type     : POST-TRAINING INFERENCE"
    )

    # ============================================================
    # LOAD THE SAME HELD-OUT TEST DATA USED BY FINAL EVALUATION
    # ============================================================

    base_data = test_eval.TestData()

    episode_starts = (
        test_eval.build_fixed_test_episode_starts(
            data=base_data,
            number_of_episodes=args.test_episodes,
            seed=args.seed,
        )
    )

    episode_starts = np.asarray(
        episode_starts,
        dtype=int,
    ).reshape(-1)

    if len(episode_starts) != args.test_episodes:
        raise RuntimeError(
            "Unexpected number of TEST episode starts."
        )

    print()
    print(
        "Fixed TEST starts : "
        f"{episode_starts.tolist()}"
    )

    all_rows = []

    # ============================================================
    # RUN EACH MODE ON EXACTLY THE SAME TEST WINDOWS
    # ============================================================

    for mode in MODES:

        print()
        print("-" * 80)
        print(f"MODE: {mode}")
        print("-" * 80)

        mode_data = build_mode_data(
            base_data,
            mode,
        )

        bridge = test_eval.build_test_bridge(
            data=mode_data,
            episode_starts=episode_starts,
        )

        local_agents, coordinator_agent = (
            test_eval.val.load_policy(
                checkpoint_episode=args.checkpoint,
                device=args.device,
                seed=args.seed,
            )
        )

        for episode_number in range(
            1,
            args.test_episodes + 1,
        ):

            result = evaluate_one_ablation_episode(
                mode=mode,
                bridge=bridge,
                local_agents=local_agents,
                coordinator_agent=coordinator_agent,
                episode_number=episode_number,
            )

            result["start_index"] = int(
                episode_starts[
                    episode_number - 1
                ]
            )

            all_rows.append(result)

            print(
                f"Episode {episode_number:02d} | "
                f"start "
                f"{result['start_index']:4d} | "
                f"return "
                f"{result['total_return']: .6f}"
            )

    results_df = pd.DataFrame(
        all_rows
    )

    # ============================================================
    # SAVE EPISODE-LEVEL RESULTS
    # ============================================================

    episode_file = (
        OUTPUT_DIRECTORY
        / "ablation_episode_results.csv"
    )

    results_df.to_csv(
        episode_file,
        index=False,
    )

    # ============================================================
    # SUMMARIES
    # ============================================================

    summaries = []

    for mode in MODES:
        mode_df = results_df[
            results_df["mode"] == mode
        ].copy()

        summaries.append(
            summarize_mode(mode_df)
        )

    summary_df = pd.DataFrame(
        summaries
    )

    full_mean = float(
        summary_df.loc[
            summary_df["mode"]
            == "full_fc_hmarl",
            "mean_return",
        ].iloc[0]
    )

    summary_df[
        "delta_mean_return_vs_full"
    ] = (
        summary_df["mean_return"]
        - full_mean
    )

    summary_file = (
        OUTPUT_DIRECTORY
        / "ablation_summary.csv"
    )

    summary_df.to_csv(
        summary_file,
        index=False,
    )

    # ============================================================
    # PAIRED COMPARISONS
    # ============================================================

    full_df = results_df[
        results_df["mode"]
        == "full_fc_hmarl"
    ].copy()

    paired_rows = []

    for mode in MODES[1:]:

        alternative_df = results_df[
            results_df["mode"] == mode
        ].copy()

        paired_rows.append(
            paired_comparison(
                full_df,
                alternative_df,
            )
        )

    paired_df = pd.DataFrame(
        paired_rows
    )

    paired_file = (
        OUTPUT_DIRECTORY
        / "ablation_paired_comparisons.csv"
    )

    paired_df.to_csv(
        paired_file,
        index=False,
    )

    # ============================================================
    # SANITY CHECK AGAINST LOCKED FINAL TEST RESULT
    # ============================================================

    locked_reference_mean = -16817.256505

    reproduced_full_mean = float(
        summary_df.loc[
            summary_df["mode"]
            == "full_fc_hmarl",
            "mean_return",
        ].iloc[0]
    )

    reproduction_difference = (
        reproduced_full_mean
        - locked_reference_mean
    )

    # ============================================================
    # METADATA
    # ============================================================

    metadata = {
        "checkpoint": int(
            args.checkpoint
        ),
        "test_episodes": int(
            args.test_episodes
        ),
        "seed": int(args.seed),
        "device": str(args.device),
        "deterministic_actions": True,
        "learning_disabled": True,
        "checkpoint_selected_using_validation_only": True,
        "test_used_for_model_selection": False,
        "test_used_for_retraining": False,
        "ablation_type":
            "post_training_inference_ablation",
        "modes": list(MODES),
        "fixed_test_episode_starts":
            episode_starts.tolist(),
        "full_fc_hmarl_locked_reference_mean":
            locked_reference_mean,
        "full_fc_hmarl_reproduced_mean":
            reproduced_full_mean,
        "full_fc_hmarl_reproduction_difference":
            float(reproduction_difference),
        "no_confidence_awareness_definition": (
            "S_pred = Z_hat instead of Phi*Z_hat, "
            "and Phi is set to 1 for the confidence-risk term."
        ),
        "no_energy_sharing_definition": (
            "Local sharing commands forced off and coordinator "
            "sharing multiplier forced to zero."
        ),
        "no_upper_level_coordination_definition": (
            "Coordinator action fixed non-adaptively: market/grid "
            "and reserve commands disabled, sharing multiplier fixed "
            "at one so local sharing is not adaptively scaled."
        ),
        "reporting_warning": (
            "These results diagnose the already-trained policy. "
            "They must not be described as separately retrained "
            "ablation agents."
        ),
    }

    metadata_file = (
        OUTPUT_DIRECTORY
        / "ablation_metadata.json"
    )

    metadata_file.write_text(
        json.dumps(
            metadata,
            indent=4,
        ),
        encoding="utf-8",
    )

    # ============================================================
    # PRINT RESULTS
    # ============================================================

    section(
        "ABLATION SUMMARY"
    )

    print(
        summary_df.to_string(
            index=False
        )
    )

    section(
        "PAIRED COMPARISONS VS FULL FC-HMARL"
    )

    print(
        paired_df.to_string(
            index=False
        )
    )

    section(
        "FULL-MODE REPRODUCTION CHECK"
    )

    print(
        "Locked final TEST mean : "
        f"{locked_reference_mean:.6f}"
    )

    print(
        "Reproduced full mean   : "
        f"{reproduced_full_mean:.6f}"
    )

    print(
        "Difference             : "
        f"{reproduction_difference:.12f}"
    )

    if abs(reproduction_difference) <= 1e-5:
        print(
            "[OK] Full FC-HMARL reproduces the locked "
            "final TEST result."
        )
    else:
        print(
            "[WARNING] Full-mode result does not exactly "
            "reproduce the locked final TEST mean."
        )
        print(
            "Do not interpret ablation differences until "
            "this discrepancy is resolved."
        )

    section(
        "OUTPUT FILES"
    )

    print(
        f"Episode results : {episode_file}"
    )
    print(
        f"Summary         : {summary_file}"
    )
    print(
        f"Paired results  : {paired_file}"
    )
    print(
        f"Metadata        : {metadata_file}"
    )


if __name__ == "__main__":
    main()
