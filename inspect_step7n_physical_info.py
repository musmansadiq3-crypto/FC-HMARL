"""
Step 7N-A diagnostic:
Inspect the exact physical/economic information exposed by one held-out TEST
step before building the final operational/economic evaluator.
"""

import numpy as np
import evaluate_real_fc_hmarl_test as t

CHECKPOINT = 1000
SEED = 42
DEVICE = "cpu"


def show(name, value, indent=0):
    pad = " " * indent
    if isinstance(value, dict):
        print(f"{pad}{name}: dict keys={list(value.keys())}")
        for k, v in value.items():
            show(str(k), v, indent + 2)
    elif isinstance(value, np.ndarray):
        print(
            f"{pad}{name}: ndarray shape={value.shape} "
            f"dtype={value.dtype}"
        )
        if value.size <= 20:
            print(f"{pad}  value={value}")
    elif isinstance(value, (list, tuple)):
        print(f"{pad}{name}: {type(value).__name__} len={len(value)}")
        if len(value) <= 10:
            for i, v in enumerate(value):
                show(f"[{i}]", v, indent + 2)
    else:
        print(f"{pad}{name}: {type(value).__name__} = {value}")


def main():
    print("=" * 80)
    print("STEP 7N-A - INSPECT PHYSICAL/ECONOMIC TEST STEP")
    print("=" * 80)

    data = t.TestData()

    starts = t.build_fixed_test_episode_starts(
        data=data,
        number_of_episodes=30,
        seed=SEED,
    )

    bridge = t.build_test_bridge(
        data=data,
        episode_starts=starts,
    )

    local_agents, coordinator_agent = t.val.load_policy(
        checkpoint_episode=CHECKPOINT,
        device=DEVICE,
        seed=SEED,
    )

    observation = bridge.reset(episode=1)

    actions = t.val.select_deterministic_actions(
        local_agents=local_agents,
        coordinator_agent=coordinator_agent,
        observation=observation,
    )

    result = bridge.step(actions)

    print("\nRESULT OBJECT")
    print("-" * 80)
    print("type:", type(result).__name__)
    print("attributes:")
    for name in sorted(
        n for n in dir(result)
        if not n.startswith("_")
    ):
        try:
            value = getattr(result, name)
        except Exception:
            continue
        if callable(value):
            continue
        if name in ("next_observation",):
            print(f"  {name}: {type(value).__name__}")
        else:
            print(f"  {name}: {type(value).__name__}")

    print("\nRESULT.INFO")
    print("-" * 80)
    show("info", result.info)

    print("\nBRIDGE")
    print("-" * 80)
    for name in sorted(
        n for n in dir(bridge)
        if not n.startswith("_")
    ):
        if name in {
            "environment",
            "last_physical_result",
            "current_exogenous",
            "last_exogenous",
        }:
            try:
                show(name, getattr(bridge, name))
            except Exception as exc:
                print(f"{name}: ERROR reading value: {exc}")

    print("\n[OK] Step 7N-A inspection complete.")


if __name__ == "__main__":
    main()
