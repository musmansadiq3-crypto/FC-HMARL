# ============================================================
# FC-HMARL
# STEP 7R-H
# REPLAY EPISODE 87 / STEP 1 AND DECOMPOSE MG3 OUTLIER
# ============================================================

from __future__ import annotations

from pathlib import Path
import json
import numpy as np

import train_real_fc_hmarl_reserve_corrected_v2 as tr

from marl.vpp_training_bridge import HierarchicalActionBundle


PROJECT_ROOT = Path(r"D:\Molvi paper review\FC_HMARL")

CHECKPOINT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "checkpoints"
    / "real_fc_hmarl_reserve_corrected"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "evaluation"
    / "step7r_h_mg3_step1_diagnostic"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_EPISODE = 87
CHECKPOINT_EPISODE = 100
TARGET_STEP = 1
STEPS_PER_EPISODE = 24


def section(title: str) -> None:
    print()
    print("=" * 88)
    print(title)
    print("=" * 88)


def load_action_from_replay(path: Path, transition_index: int) -> np.ndarray:
    archive = np.load(path, allow_pickle=True)

    action_key = None

    for candidate in ["actions", "action", "action_buffer"]:
        for key in archive.files:
            if str(key).lower() == candidate:
                action_key = key
                break
        if action_key is not None:
            break

    if action_key is None:
        for key in archive.files:
            if "action" in str(key).lower():
                action_key = key
                break

    if action_key is None:
        raise RuntimeError(
            f"No action array found in replay:\n{path}\nKeys={archive.files}"
        )

    action = np.asarray(archive[action_key])[transition_index]
    return np.asarray(action, dtype=float).reshape(-1)


def get_field(obj, names, default=None):
    if obj is None:
        return default

    if isinstance(obj, dict):
        for name in names:
            if name in obj:
                return obj[name]

    for name in names:
        if hasattr(obj, name):
            return getattr(obj, name)

    return default


def as_float(value, default=np.nan):
    try:
        if value is None:
            return float(default)
        arr = np.asarray(value)
        if arr.size == 1:
            return float(arr.reshape(-1)[0])
    except Exception:
        pass
    return float(default)


section("STEP 7R-H - REPLAYING EPISODE 87 / STEP 1")

data = tr.RealRLTrainingData(tr.RL_ARCHIVE_FILE)

bridge = tr.build_real_training_bridge(
    data=data,
    maximum_episodes=100,
    seed=42,
)

observation = bridge.reset(episode=TARGET_EPISODE)

transition_index = (
    (TARGET_EPISODE - 1) * STEPS_PER_EPISODE
    + (TARGET_STEP - 1)
)

local_actions = []

for mg_index in range(1, 6):
    replay_path = (
        CHECKPOINT_DIR
        / f"local_agent_{mg_index}_episode_{CHECKPOINT_EPISODE}.replay.npz"
    )

    if not replay_path.exists():
        # Some projects save microgrid IDs as zero-based.
        alt = (
            CHECKPOINT_DIR
            / f"local_agent_{mg_index - 1}_episode_{CHECKPOINT_EPISODE}.replay.npz"
        )
        if alt.exists():
            replay_path = alt

    if not replay_path.exists():
        raise FileNotFoundError(
            f"Local replay not found for MG{mg_index}."
        )

    action = load_action_from_replay(
        replay_path,
        transition_index,
    )

    local_actions.append(action)

    print(
        f"MG{mg_index} stored action : "
        f"{np.array2string(action, precision=6)}"
    )

coordinator_path = (
    CHECKPOINT_DIR
    / f"coordinator_episode_{CHECKPOINT_EPISODE}.replay.npz"
)

if not coordinator_path.exists():
    raise FileNotFoundError(
        f"Coordinator replay not found:\n{coordinator_path}"
    )

coordinator_action = load_action_from_replay(
    coordinator_path,
    transition_index,
)

print(
    "Coordinator stored action: "
    f"{np.array2string(coordinator_action, precision=6)}"
)

bundle = HierarchicalActionBundle(
    local_actions=local_actions,
    coordinator_action=coordinator_action,
)

result = bridge.step(bundle)

section("REPLAY RESULT")

print(f"Episode              : {TARGET_EPISODE}")
print(f"Step                 : {TARGET_STEP}")
print(f"Done                 : {result.done}")
print(f"Local rewards        : {np.asarray(result.local_rewards, dtype=float)}")
print(f"Coordinator reward   : {float(result.coordinator_reward):.6f}")
print(f"Total reward         : {float(np.sum(result.local_rewards) + result.coordinator_reward):.6f}")

# ------------------------------------------------------------
# Inspect physical result object
# ------------------------------------------------------------

physical = get_field(
    result,
    [
        "physical_result",
        "environment_result",
        "vpp_result",
        "step_result",
    ],
    default=None,
)

# Some bridge implementations expose the physical result indirectly.
if physical is None:
    physical = get_field(
        bridge,
        [
            "last_physical_result",
            "last_environment_result",
            "last_result",
        ],
        default=None,
    )

if physical is None:
    print()
    print("[WARN] Physical result object was not directly exposed by the bridge.")
    print("      Reward replay still succeeded; detailed fields below may be unavailable.")

local_results = get_field(
    physical,
    ["local_results", "microgrid_results"],
    default=None,
)

totals = get_field(
    physical,
    ["totals", "aggregate_totals"],
    default=None,
)

constraints = get_field(
    physical,
    ["constraints", "constraint_summary"],
    default=None,
)

summary = {
    "episode": TARGET_EPISODE,
    "step": TARGET_STEP,
    "transition_index_zero_based": transition_index,
    "local_actions": [a.tolist() for a in local_actions],
    "coordinator_action": coordinator_action.tolist(),
    "local_rewards": np.asarray(result.local_rewards, dtype=float).tolist(),
    "coordinator_reward": float(result.coordinator_reward),
    "total_reward": float(
        np.sum(result.local_rewards)
        + result.coordinator_reward
    ),
}

# ------------------------------------------------------------
# MG3 detailed physical inspection
# ------------------------------------------------------------

section("MG3 STEP-1 DETAIL")

mg3 = None

if local_results is not None:
    try:
        mg3 = local_results[2]
    except Exception:
        mg3 = None

candidate_fields = [
    ("load_kw", ["load_kw", "load_power_kw"]),
    ("pv_kw", ["pv_power_kw", "pv_kw"]),
    ("ev_kw", ["ev_power_kw", "ev_charging_power_kw"]),
    ("bess_kw", ["bess_power_kw", "battery_power_kw"]),
    ("grid_kw", ["grid_power_kw", "grid_exchange_kw"]),
    ("import_kw", ["grid_import_kw", "import_power_kw"]),
    ("export_kw", ["grid_export_kw", "export_power_kw"]),
    ("soc_before", ["soc_before", "initial_soc", "previous_soc"]),
    ("soc_after", ["soc_after", "soc", "new_soc"]),
    ("purchase_cost_usd", ["purchase_cost_usd", "grid_purchase_cost_usd", "market_purchase_cost_usd"]),
    ("sale_revenue_usd", ["sale_revenue_usd", "grid_sale_revenue_usd", "market_sale_revenue_usd"]),
    ("reserve_kw", ["reserve_power_kw", "reserve_kw", "reserve_feasible_kw"]),
    ("reserve_revenue_usd", ["reserve_revenue_usd"]),
    ("balance_violation_kw", ["balance_violation_kw", "power_balance_violation_kw"]),
    ("transformer_violation_kw", ["transformer_violation_kw", "grid_violation_kw"]),
]

mg3_values = {}

for label, names in candidate_fields:
    value = get_field(mg3, names, default=None)
    if value is None:
        continue

    numeric = as_float(value)
    mg3_values[label] = numeric
    print(f"{label:26s}: {numeric:.9f}")

summary["mg3_physical_fields"] = mg3_values

# ------------------------------------------------------------
# Reconstruct reward pieces with project reward builder API if
# public component methods/fields are available.
# ------------------------------------------------------------

reward_builder = bridge.reward_builder

print()
print("Reward builder class        :", type(reward_builder).__name__)

# Publicly inspect config.
reward_config = get_field(
    reward_builder,
    ["config"],
    default=None,
)

if reward_config is not None:
    config_dict = {}

    for key in [
        "beta_soc",
        "beta_grid",
        "risk_aversion",
    ]:
        value = get_field(reward_config, [key], default=None)
        if value is not None:
            config_dict[key] = as_float(value)
            print(f"{key:26s}: {as_float(value):.9f}")

    summary["reward_config"] = config_dict

# ------------------------------------------------------------
# Read likely reward/physical attributes from EnvironmentStepResult
# ------------------------------------------------------------

result_fields = {}

for name in [
    "local_grid_costs",
    "local_degradation_costs",
    "local_violation_costs",
    "coordinator_profit",
    "coordinator_imbalance_cost",
    "coordinator_risk_cost",
    "confidence",
]:
    value = get_field(result, [name], default=None)

    if value is not None:
        arr = np.asarray(value)
        result_fields[name] = arr.tolist() if arr.ndim else float(arr)
        print(f"{name:26s}: {value}")

summary["bridge_result_fields"] = result_fields

# ------------------------------------------------------------
# Directly compute the arithmetic decomposition if the bridge
# exposes the three local cost arrays.
# ------------------------------------------------------------

grid_costs = get_field(result, ["local_grid_costs"], default=None)
deg_costs = get_field(result, ["local_degradation_costs"], default=None)
viol_costs = get_field(result, ["local_violation_costs"], default=None)

if (
    grid_costs is not None
    and deg_costs is not None
    and viol_costs is not None
):
    grid3 = float(np.asarray(grid_costs).reshape(-1)[2])
    deg3 = float(np.asarray(deg_costs).reshape(-1)[2])
    viol3 = float(np.asarray(viol_costs).reshape(-1)[2])

    print()
    print("MG3 reward decomposition")
    print(f"  grid cost       : {grid3:.9f}")
    print(f"  degradation cost: {deg3:.9f}")
    print(f"  violation cost  : {viol3:.9f}")
    print(f"  reconstructed R : {- (grid3 + deg3 + viol3):.9f}")

    summary["mg3_reward_decomposition"] = {
        "grid_cost": grid3,
        "degradation_cost": deg3,
        "violation_cost": viol3,
        "reconstructed_reward": -(grid3 + deg3 + viol3),
    }

# ------------------------------------------------------------
# Generic physical totals / constraints
# ------------------------------------------------------------

if totals is not None:
    try:
        summary["physical_totals"] = dict(totals)
    except Exception:
        summary["physical_totals"] = str(totals)

if constraints is not None:
    try:
        summary["constraints"] = dict(constraints)
    except Exception:
        summary["constraints"] = str(constraints)

summary_file = OUTPUT_DIR / "step7r_h_mg3_step1_summary.json"

with open(summary_file, "w", encoding="utf-8") as file:
    json.dump(summary, file, indent=4, default=str)

section("STEP 7R-H COMPLETE")

print("[OK] Stored Episode-87 actions were replayed in the corrected environment.")
print("[OK] MG3 Step-1 reward was isolated.")
print()
print(f"Summary file: {summary_file}")
print()
print("Please send the COMPLETE console output from this script.")
