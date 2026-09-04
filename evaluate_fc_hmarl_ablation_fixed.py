"""
FC-HMARL ablation evaluator.

Evaluates the locked FC-HMARL checkpoint under controlled action/state
ablations on the same held-out TEST episodes.

Important methodological note:
These are post-training inference ablations. They diagnose sensitivity of the
trained policy to hierarchy/confidence/sharing components. They are not
separately retrained ablation models and should be reported as such.

Ablations:
- full_fc_hmarl
- no_confidence_weighting
- no_energy_sharing
- no_upper_level_coordination

The script reuses the validated held-out TEST evaluation pipeline so that
physical data, checkpoint loading, deterministic inference, and episode starts
remain consistent.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

import evaluate_real_fc_hmarl_test as base_test


PROJECT_ROOT = Path(r"D:\Molvi paper review\FC_HMARL")
OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "evaluation"
    / "real_fc_hmarl_ablation"
)

LOCKED_CHECKPOINT = 1000


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate FC-HMARL post-training ablations."
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


def zero_predictive_tail(local_states, coordinator_state):
    """
    Remove the 96-D confidence-aware predictive state from all hierarchical
    observations while keeping instantaneous physical variables unchanged.
    """
    new_local = []

    for state in local_states:
        state = np.asarray(state, dtype=np.float32).copy()
        if state.size < 96:
            raise ValueError(
                f"Local state too short for predictive tail: {state.size}"
            )
        state[-96:] = 0.0
        new_local.append(state)

    coord = np.asarray(
        coordinator_state,
        dtype=np.float32,
    ).copy()

    if coord.size < 96:
        raise ValueError(
            f"Coordinator state too short for predictive tail: {coord.size}"
        )

    coord[-96:] = 0.0

    return new_local, coord


def deterministic_actions(
    local_agents,
    coordinator_agent,
    local_states,
    coordinator_state,
):
    local_actions = [
        agent.select_action(
            np.asarray(state, dtype=np.float32),
            deterministic=True,
        )
        for agent, state in zip(local_agents, local_states)
    ]

    coordinator_action = coordinator_agent.select_action(
        np.asarray(coordinator_state, dtype=np.float32),
        deterministic=True,
    )

    return local_actions, coordinator_action


def apply_action_ablation(
    mode: str,
    local_actions: List[np.ndarray],
    coordinator_action: np.ndarray,
):
    local_actions = [
        np.asarray(a, dtype=np.float32).copy()
        for a in local_actions
    ]
    coordinator_action = np.asarray(
        coordinator_action,
        dtype=np.float32,
    ).copy()

    if mode == "full_fc_hmarl":
        return local_actions, coordinator_action

    if mode == "no_energy_sharing":
        # Local software action convention in this reconstruction:
        # [BESS, share_to_MG_1, ..., share_to_MG_4]
        for action in local_actions:
            if action.size > 1:
                action[1:] = -1.0

        # Coordinator third software action is the sharing multiplier.
        if coordinator_action.size >= 3:
            coordinator_action[2] = -1.0

        return local_actions, coordinator_action

    if mode == "no_upper_level_coordination":
        # Keep local trained policies active but neutralize coordinator action.
        coordinator_action[:] = 0.0
        return local_actions, coordinator_action

    if mode == "no_confidence_weighting":
        # Action is unchanged here; state modification happens before inference.
        return local_actions, coordinator_action

    raise ValueError(f"Unknown ablation mode: {mode}")


def extract_return_fields(step_result):
    """
    Extract reward components robustly from the validated bridge step result.
    """
    local_rewards = getattr(step_result, "local_rewards", None)
    coordinator_reward = getattr(
        step_result,
        "coordinator_reward",
        None,
    )

    if local_rewards is None or coordinator_reward is None:
        rewards = getattr(step_result, "rewards", None)

        if rewards is not None:
            local_rewards = getattr(
                rewards,
                "local_rewards",
                local_rewards,
            )
            coordinator_reward = getattr(
                rewards,
                "coordinator_reward",
                coordinator_reward,
            )

    if local_rewards is None:
        raise AttributeError(
            "Could not find local_rewards in bridge step result."
        )

    if coordinator_reward is None:
        raise AttributeError(
            "Could not find coordinator_reward in bridge step result."
        )

    local_rewards = np.asarray(
        local_rewards,
        dtype=float,
    ).reshape(-1)

    return local_rewards, float(coordinator_reward)


def extract_violation_flags(step_result):
    """
    Read physical violation information without assuming one exact nested API.
    """
    power_balance_violation = 0
    transformer_violation = 0

    candidates = [
        step_result,
        getattr(step_result, "physical_result", None),
        getattr(step_result, "environment_result", None),
        getattr(step_result, "vpp_result", None),
    ]

    for obj in candidates:
        if obj is None:
            continue

        constraints = getattr(obj, "constraints", None)

        if isinstance(constraints, dict):
            if not constraints.get("all_power_balanced", True):
                power_balance_violation = 1
            if not constraints.get(
                "all_transformers_feasible",
                True,
            ):
                transformer_violation = 1

        if hasattr(obj, "all_power_balanced"):
            if not bool(obj.all_power_balanced):
                power_balance_violation = 1

        if hasattr(obj, "all_transformers_feasible"):
            if not bool(obj.all_transformers_feasible):
                transformer_violation = 1

    return power_balance_violation, transformer_violation


def get_observation_parts(observation):
    local_states = getattr(
        observation,
        "local_states",
        None,
    )
    coordinator_state = getattr(
        observation,
        "coordinator_state",
        None,
    )

    if local_states is None or coordinator_state is None:
        raise AttributeError(
            "Observation must expose local_states and coordinator_state."
        )

    return local_states, coordinator_state


def build_action_bundle(local_actions, coordinator_action):
    """
    Reuse the action-bundle class exposed by the validated evaluation module
    or its imported bridge module.
    """
    if hasattr(base_test, "HierarchicalActionBundle"):
        cls = base_test.HierarchicalActionBundle
    elif hasattr(base_test, "val") and hasattr(
        base_test.val,
        "HierarchicalActionBundle",
    ):
        cls = base_test.val.HierarchicalActionBundle
    else:
        from marl.vpp_training_bridge import HierarchicalActionBundle
        cls = HierarchicalActionBundle

    try:
        return cls(
            local_actions=local_actions,
            coordinator_action=coordinator_action,
        )
    except TypeError:
        return cls(
            local_actions,
            coordinator_action,
        )


def load_common_test_context(args):
    """
    Reuse the validated TEST evaluator objects. This intentionally depends on
    the already-successful held-out TEST pipeline rather than duplicating it.
    """
    # The user's validated TEST script exposes these helpers through its own
    # module or through the validation evaluator imported as `val`.
    module_candidates = [base_test]
    if hasattr(base_test, "val"):
        module_candidates.append(base_test.val)

    load_data = None
    choose_starts = None
    build_bridge = None
    load_policy = None

    for module in module_candidates:
        for name in (
            "TestData",
            "load_test_data",
            "build_test_data",
        ):
            if hasattr(module, name):
                load_data = getattr(module, name)
                break
        if load_data is not None:
            break

    for module in module_candidates:
        for name in (
            "select_fixed_episode_starts",
            "choose_fixed_episode_starts",
            "build_fixed_episode_starts",
        ):
            if hasattr(module, name):
                choose_starts = getattr(module, name)
                break
        if choose_starts is not None:
            break

    for module in module_candidates:
        for name in (
            "build_test_bridge",
            "build_validation_bridge",
        ):
            if hasattr(module, name):
                build_bridge = getattr(module, name)
                break
        if build_bridge is not None:
            break

    for module in module_candidates:
        if hasattr(module, "load_policy"):
            load_policy = getattr(module, "load_policy")
            break

    if load_policy is None or build_bridge is None:
        raise RuntimeError(
            "Could not reuse load_policy/build bridge helpers from the "
            "validated TEST/validation evaluator."
        )

    return load_data, choose_starts, build_bridge, load_policy


def print_api_hint_and_exit():
    """
    Print a concise diagnostic if the user's existing evaluator exposes API
    names different from the common reconstruction names.
    """
    names = sorted(
        n for n in dir(base_test)
        if not n.startswith("_")
    )

    print()
    print("[NEEDS API ADAPTER]")
    print(
        "The ablation script found the validated TEST evaluator but its "
        "public helper names differ from the expected names."
    )
    print("Available public names:")
    for name in names:
        print(" ", name)

    if hasattr(base_test, "val"):
        print()
        print("Public names in base_test.val:")
        for name in sorted(
            n for n in dir(base_test.val)
            if not n.startswith("_")
        ):
            print(" ", name)

    raise SystemExit(2)


def main():
    args = parse_args()

    if args.checkpoint != LOCKED_CHECKPOINT:
        raise ValueError(
            f"Final checkpoint is locked to {LOCKED_CHECKPOINT}; "
            "do not use TEST ablations to reselect checkpoints."
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("STEP 7M - FC-HMARL POST-TRAINING ABLATION EVALUATION")
    print("=" * 80)
    print(f"Checkpoint       : {args.checkpoint}")
    print(f"Test episodes    : {args.test_episodes}")
    print(f"Seed             : {args.seed}")
    print(f"Device           : {args.device}")
    print("Learning         : DISABLED")
    print("Checkpoint select: VALIDATION ONLY")
    print("TEST use         : FINAL DIAGNOSTIC ABLATIONS")
    print()

    load_data, choose_starts, build_bridge, load_policy = (
        load_common_test_context(args)
    )

    # Because the already-validated evaluator may implement its TestData as a
    # class rather than a function, attempt the common calling conventions.
    if load_data is None or choose_starts is None:
        print_api_hint_and_exit()

    try:
        test_data = load_data()
    except TypeError:
        try:
            test_data = load_data(PROJECT_ROOT)
        except TypeError:
            print_api_hint_and_exit()

    # Determine sample count.
    sample_count = None
    for attr in (
        "number_of_samples",
        "num_samples",
        "sample_count",
    ):
        if hasattr(test_data, attr):
            sample_count = int(getattr(test_data, attr))
            break

    if sample_count is None:
        for attr in ("y_test", "current_actual", "predictive_state"):
            if hasattr(test_data, attr):
                sample_count = len(getattr(test_data, attr))
                break

    if sample_count is None:
        print_api_hint_and_exit()

    # The validated evaluator's build_fixed_episode_starts expects
    # the data object itself (not an integer sample count). Try that
    # calling convention first, then fall back to other compatible APIs.
    try:
        starts = choose_starts(
            test_data,
            args.test_episodes,
            args.seed,
        )
    except (TypeError, AttributeError):
        try:
            starts = choose_starts(
                data=test_data,
                number_of_episodes=args.test_episodes,
                seed=args.seed,
            )
        except (TypeError, AttributeError):
            try:
                starts = choose_starts(
                    sample_count,
                    args.test_episodes,
                    args.seed,
                )
            except (TypeError, AttributeError):
                try:
                    starts = choose_starts(
                        number_of_samples=sample_count,
                        number_of_episodes=args.test_episodes,
                        seed=args.seed,
                    )
                except (TypeError, AttributeError):
                    print_api_hint_and_exit()

    starts = np.asarray(starts, dtype=int).reshape(-1)

    # Load fixed trained policy once per mode to keep each evaluation isolated.
    modes = [
        "full_fc_hmarl",
        "no_confidence_weighting",
        "no_energy_sharing",
        "no_upper_level_coordination",
    ]

    all_rows: List[Dict] = []

    for mode in modes:
        print()
        print("-" * 80)
        print(f"MODE: {mode}")
        print("-" * 80)

        try:
            local_agents, coordinator_agent = load_policy(
                args.checkpoint,
                args.device,
            )
        except TypeError:
            try:
                local_agents, coordinator_agent = load_policy(
                    checkpoint=args.checkpoint,
                    device=args.device,
                )
            except TypeError:
                print_api_hint_and_exit()

        for episode_number, start in enumerate(starts, start=1):

            # Build a fresh bridge/provider for this episode.
            try:
                bridge = build_bridge(
                    test_data,
                    int(start),
                )
            except TypeError:
                try:
                    bridge = build_bridge(
                        data=test_data,
                        episode_start=int(start),
                    )
                except TypeError:
                    try:
                        bridge = build_bridge(
                            test_data,
                            [int(start)],
                        )
                    except TypeError:
                        print_api_hint_and_exit()

            try:
                observation = bridge.reset(1)
            except TypeError:
                observation = bridge.reset()

            total_local = np.zeros(
                len(local_agents),
                dtype=float,
            )
            total_coord = 0.0
            pb_steps = 0
            tx_steps = 0
            step_count = 0

            while True:
                local_states, coordinator_state = (
                    get_observation_parts(observation)
                )

                if mode == "no_confidence_weighting":
                    local_states, coordinator_state = (
                        zero_predictive_tail(
                            local_states,
                            coordinator_state,
                        )
                    )

                local_actions, coordinator_action = (
                    deterministic_actions(
                        local_agents,
                        coordinator_agent,
                        local_states,
                        coordinator_state,
                    )
                )

                local_actions, coordinator_action = (
                    apply_action_ablation(
                        mode,
                        local_actions,
                        coordinator_action,
                    )
                )

                bundle = build_action_bundle(
                    local_actions,
                    coordinator_action,
                )

                step_result = bridge.step(bundle)

                local_rewards, coordinator_reward = (
                    extract_return_fields(step_result)
                )

                total_local += local_rewards
                total_coord += coordinator_reward

                pb, tx = extract_violation_flags(step_result)
                pb_steps += pb
                tx_steps += tx
                step_count += 1

                done = bool(
                    getattr(step_result, "done", False)
                )

                if done:
                    break

                observation = getattr(
                    step_result,
                    "next_observation",
                    None,
                )

                if observation is None:
                    observation = getattr(
                        step_result,
                        "observation",
                        None,
                    )

                if observation is None:
                    raise AttributeError(
                        "Could not find next observation in bridge step result."
                    )

            total_return = float(
                np.sum(total_local) + total_coord
            )

            row = {
                "mode": mode,
                "checkpoint": args.checkpoint,
                "test_episode": episode_number,
                "start_index": int(start),
                "steps": step_count,
                "total_return": total_return,
                "coordinator_return": float(total_coord),
                "mean_local_return": float(
                    np.mean(total_local)
                ),
                "power_balance_violation_steps": int(pb_steps),
                "transformer_violation_steps": int(tx_steps),
            }

            for i, value in enumerate(total_local, start=1):
                row[f"local_mg_{i}_return"] = float(value)

            all_rows.append(row)

            print(
                f"Episode {episode_number:02d} | "
                f"start {int(start):4d} | "
                f"return {total_return: .6f}"
            )

    df = pd.DataFrame(all_rows)

    episode_file = OUTPUT_DIR / "ablation_episode_results.csv"
    df.to_csv(episode_file, index=False)

    summary_rows = []

    for mode, group in df.groupby("mode", sort=False):
        returns = group["total_return"].to_numpy(dtype=float)

        summary_rows.append(
            {
                "mode": mode,
                "episodes": int(len(group)),
                "mean_return": float(np.mean(returns)),
                "std_return": float(np.std(returns, ddof=0)),
                "median_return": float(np.median(returns)),
                "minimum_return": float(np.min(returns)),
                "maximum_return": float(np.max(returns)),
                "mean_coordinator_return": float(
                    group["coordinator_return"].mean()
                ),
                "mean_local_return_per_mg": float(
                    group["mean_local_return"].mean()
                ),
                "power_balance_violation_steps": int(
                    group["power_balance_violation_steps"].sum()
                ),
                "transformer_violation_steps": int(
                    group["transformer_violation_steps"].sum()
                ),
            }
        )

    summary = pd.DataFrame(summary_rows)

    full_mean = float(
        summary.loc[
            summary["mode"] == "full_fc_hmarl",
            "mean_return",
        ].iloc[0]
    )

    summary["mean_return_change_vs_full"] = (
        summary["mean_return"] - full_mean
    )

    summary_file = OUTPUT_DIR / "ablation_summary.csv"
    summary.to_csv(summary_file, index=False)

    paired_rows = []
    full = df[df["mode"] == "full_fc_hmarl"].sort_values(
        "test_episode"
    )

    for mode in modes[1:]:
        alt = df[df["mode"] == mode].sort_values(
            "test_episode"
        )

        delta = (
            alt["total_return"].to_numpy(dtype=float)
            - full["total_return"].to_numpy(dtype=float)
        )

        paired_rows.append(
            {
                "comparison": f"{mode}_minus_full",
                "mean_delta": float(np.mean(delta)),
                "median_delta": float(np.median(delta)),
                "better_than_full_episodes": int(np.sum(delta > 0)),
                "worse_than_full_episodes": int(np.sum(delta < 0)),
                "equal_episodes": int(np.sum(delta == 0)),
                "minimum_delta": float(np.min(delta)),
                "maximum_delta": float(np.max(delta)),
            }
        )

    paired = pd.DataFrame(paired_rows)
    paired_file = OUTPUT_DIR / "ablation_paired_comparisons.csv"
    paired.to_csv(paired_file, index=False)

    metadata = {
        "checkpoint": args.checkpoint,
        "test_episodes": args.test_episodes,
        "seed": args.seed,
        "device": args.device,
        "deterministic_actions": True,
        "learning_disabled": True,
        "checkpoint_selection_used_test_data": False,
        "ablation_type": "post_training_inference_ablation",
        "important_reporting_note": (
            "These are inference-time component ablations of the locked "
            "trained policy, not separately retrained ablation agents."
        ),
        "modes": modes,
        "fixed_test_starts": starts.tolist(),
    }

    metadata_file = OUTPUT_DIR / "ablation_metadata.json"
    metadata_file.write_text(
        json.dumps(metadata, indent=4),
        encoding="utf-8",
    )

    print()
    print("=" * 80)
    print("ABLATION SUMMARY")
    print("=" * 80)
    print(summary.to_string(index=False))

    print()
    print("=" * 80)
    print("PAIRED COMPARISONS VS FULL FC-HMARL")
    print("=" * 80)
    print(paired.to_string(index=False))

    print()
    print(f"Episode results : {episode_file}")
    print(f"Summary         : {summary_file}")
    print(f"Paired results  : {paired_file}")
    print(f"Metadata        : {metadata_file}")


if __name__ == "__main__":
    main()
