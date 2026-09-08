from __future__ import annotations
from pathlib import Path
import json
import re
import numpy as np
import pandas as pd
# ============================================================
# 1. PATHS / SETTINGS
# ============================================================
PROJECT_ROOT = Path(r"D:\Molvi paper review\FC_HMARL")

RESULTS_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "results"
    / "real_fc_hmarl_reserve_corrected"
)

CHECKPOINT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "checkpoints"
    / "real_fc_hmarl_reserve_corrected"
)

HISTORY_FILE = RESULTS_DIR / "training_history.csv"

DIAGNOSTIC_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "evaluation"
    / "step7r_g_episode87_diagnostic"
)

TARGET_EPISODE = 87
CHECKPOINT_EPISODE = 100
STEPS_PER_EPISODE = 24

DIAGNOSTIC_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. HELPERS
# ============================================================

def section(title: str) -> None:
    print()
    print("=" * 88)
    print(title)
    print("=" * 88)


def pick_episode_column(columns):
    candidates = [
        "episode",
        "episodes",
        "episode_number",
        "episode_index",
    ]
    lower_map = {str(c).lower(): c for c in columns}

    for candidate in candidates:
        if candidate in lower_map:
            return lower_map[candidate]

    for c in columns:
        if "episode" in str(c).lower():
            return c

    return None


def find_reward_key(files):
    exact = [
        "rewards",
        "reward",
        "reward_buffer",
    ]

    lower_map = {str(k).lower(): k for k in files}

    for name in exact:
        if name in lower_map:
            return lower_map[name]

    for key in files:
        if "reward" in str(key).lower():
            return key

    return None


def find_done_key(files):
    for wanted in ["dones", "done", "terminals", "terminal"]:
        for key in files:
            if str(key).lower() == wanted:
                return key

    for key in files:
        low = str(key).lower()
        if "done" in low or "terminal" in low:
            return key

    return None


def find_action_key(files):
    for wanted in ["actions", "action", "action_buffer"]:
        for key in files:
            if str(key).lower() == wanted:
                return key

    for key in files:
        if "action" in str(key).lower():
            return key

    return None


def infer_valid_length(archive, reward_array):
    # Preferred explicit metadata.
    metadata_candidates = [
        "size",
        "current_size",
        "buffer_size",
        "length",
        "num_samples",
        "number_of_samples",
    ]

    for key in archive.files:
        if str(key).lower() in metadata_candidates:
            try:
                value = int(np.asarray(archive[key]).reshape(-1)[0])
                if 0 < value <= len(reward_array):
                    return value
            except Exception:
                pass

    # For this run only 100*24 = 2400 transitions were created,
    # well below the manuscript replay capacity (1e6). Therefore
    # the first 2400 chronological slots should be valid if the
    # replay save uses preallocated arrays.
    expected = CHECKPOINT_EPISODE * STEPS_PER_EPISODE

    if len(reward_array) >= expected:
        return expected

    return len(reward_array)


def normalize_reward_array(arr):
    arr = np.asarray(arr)
    if arr.ndim == 0:
        return arr.reshape(1)
    if arr.ndim == 1:
        return arr.astype(float)
    # Typical replay storage is (N, 1).
    return arr.reshape(arr.shape[0], -1)[:, 0].astype(float)


def agent_label(path: Path) -> str:
    name = path.name

    m = re.search(r"local_agent_(.+?)_episode_", name)
    if m:
        return f"local_{m.group(1)}"

    if name.startswith("coordinator_"):
        return "coordinator"

    return path.stem


# ============================================================
# 3. TRAINING-HISTORY DIAGNOSTIC
# ============================================================

section("STEP 7R-G - EPISODE 87 TRAINING-HISTORY DIAGNOSTIC")

if not HISTORY_FILE.exists():
    raise FileNotFoundError(
        f"Training history not found:\n{HISTORY_FILE}"
    )

history = pd.read_csv(HISTORY_FILE)

print(f"History file      : {HISTORY_FILE}")
print(f"Rows              : {len(history)}")
print(f"Columns           : {list(history.columns)}")

episode_col = pick_episode_column(history.columns)

if episode_col is None:
    # Fall back to 1-based row number if the export has no episode column.
    history = history.copy()
    history["_episode_number"] = np.arange(1, len(history) + 1)
    episode_col = "_episode_number"

target_rows = history.loc[
    pd.to_numeric(history[episode_col], errors="coerce") == TARGET_EPISODE
]

if target_rows.empty:
    raise RuntimeError(
        f"Episode {TARGET_EPISODE} was not found in training history."
    )

target_row = target_rows.iloc[0]

print()
print(f"Target episode    : {TARGET_EPISODE}")
print("Target history row:")
for col in history.columns:
    print(f"  {col}: {target_row[col]}")

window = history.loc[
    pd.to_numeric(history[episode_col], errors="coerce").between(
        TARGET_EPISODE - 3,
        TARGET_EPISODE + 3,
    )
].copy()

window_file = DIAGNOSTIC_DIR / "episode87_history_window.csv"
window.to_csv(window_file, index=False)

print()
print(f"Saved history window: {window_file}")


# ============================================================
# 4. REPLAY-BUFFER DIAGNOSTIC
# ============================================================

section("READING EPISODE 87 TRANSITIONS FROM CHECKPOINT-100 REPLAY BUFFERS")

if not CHECKPOINT_DIR.exists():
    raise FileNotFoundError(
        f"Checkpoint directory not found:\n{CHECKPOINT_DIR}"
    )

replay_files = sorted(
    CHECKPOINT_DIR.glob(f"*episode_{CHECKPOINT_EPISODE}.replay.npz")
)

if not replay_files:
    raise FileNotFoundError(
        "No replay-buffer sidecars were found for checkpoint "
        f"{CHECKPOINT_EPISODE} in:\n{CHECKPOINT_DIR}"
    )

print(f"Replay files found: {len(replay_files)}")

# Episode 87 occupies transitions 2064..2087 in chronological 0-based storage.
start = (TARGET_EPISODE - 1) * STEPS_PER_EPISODE
stop = TARGET_EPISODE * STEPS_PER_EPISODE

rows = []
step_rows = []

for replay_path in replay_files:
    label = agent_label(replay_path)
    archive = np.load(replay_path, allow_pickle=True)

    reward_key = find_reward_key(archive.files)

    if reward_key is None:
        print()
        print(f"[WARN] {label}: no reward-like array found.")
        print(f"       Keys: {list(archive.files)}")
        continue

    rewards_all = normalize_reward_array(archive[reward_key])
    valid_length = infer_valid_length(archive, rewards_all)

    if stop > valid_length:
        raise RuntimeError(
            f"{label}: replay has only {valid_length} valid transitions; "
            f"cannot extract episode {TARGET_EPISODE} slice [{start}:{stop}]."
        )

    rewards = rewards_all[start:stop]

    action_key = find_action_key(archive.files)
    done_key = find_done_key(archive.files)

    actions = None
    dones = None

    if action_key is not None:
        actions = np.asarray(archive[action_key])[start:stop]

    if done_key is not None:
        dones = np.asarray(archive[done_key])[start:stop]

    total_reward = float(np.sum(rewards))
    mean_reward = float(np.mean(rewards))
    minimum_reward = float(np.min(rewards))
    maximum_reward = float(np.max(rewards))
    minimum_step = int(np.argmin(rewards) + 1)

    rows.append(
        {
            "agent": label,
            "replay_file": str(replay_path),
            "reward_key": reward_key,
            "valid_transitions": int(valid_length),
            "episode_reward_sum": total_reward,
            "episode_reward_mean": mean_reward,
            "episode_reward_min": minimum_reward,
            "episode_reward_max": maximum_reward,
            "worst_step_1_based": minimum_step,
            "action_key": action_key,
            "done_key": done_key,
        }
    )

    for step_idx, reward in enumerate(rewards, start=1):
        record = {
            "agent": label,
            "episode": TARGET_EPISODE,
            "step": step_idx,
            "reward": float(reward),
        }

        if actions is not None:
            action_flat = np.asarray(actions[step_idx - 1]).reshape(-1)
            for j, value in enumerate(action_flat):
                record[f"action_{j}"] = float(value)

        if dones is not None:
            done_value = np.asarray(dones[step_idx - 1]).reshape(-1)[0]
            record["done"] = bool(done_value)

        step_rows.append(record)

    print()
    print(f"{label}")
    print(f"  reward key       : {reward_key}")
    print(f"  valid transitions: {valid_length}")
    print(f"  episode sum      : {total_reward:.6f}")
    print(f"  episode mean     : {mean_reward:.6f}")
    print(f"  minimum reward   : {minimum_reward:.6f}")
    print(f"  worst step       : {minimum_step}")
    print(f"  maximum reward   : {maximum_reward:.6f}")


# ============================================================
# 5. AGGREGATE DIAGNOSTIC
# ============================================================

if not rows:
    raise RuntimeError(
        "No replay reward arrays could be diagnosed."
    )

agent_summary = pd.DataFrame(rows).sort_values(
    "episode_reward_sum"
)

step_detail = pd.DataFrame(step_rows)

agent_summary_file = (
    DIAGNOSTIC_DIR
    / "episode87_agent_reward_summary.csv"
)

step_detail_file = (
    DIAGNOSTIC_DIR
    / "episode87_step_rewards_actions.csv"
)

agent_summary.to_csv(
    agent_summary_file,
    index=False,
)

step_detail.to_csv(
    step_detail_file,
    index=False,
)

# Build per-step total across all agents.
per_step_total = (
    step_detail
    .groupby("step", as_index=False)["reward"]
    .sum()
    .rename(columns={"reward": "sum_agent_rewards"})
)

per_step_total_file = (
    DIAGNOSTIC_DIR
    / "episode87_total_reward_by_step.csv"
)

per_step_total.to_csv(
    per_step_total_file,
    index=False,
)

worst_total_step_row = per_step_total.loc[
    per_step_total["sum_agent_rewards"].idxmin()
]

sum_replay_rewards = float(
    agent_summary["episode_reward_sum"].sum()
)

# Try to identify total-return column in history.
return_col = None
for candidate in [
    "total_return",
    "episode_return",
    "return",
]:
    for col in history.columns:
        if str(col).lower() == candidate:
            return_col = col
            break
    if return_col is not None:
        break

history_total_return = None

if return_col is not None:
    history_total_return = float(target_row[return_col])


section("STEP 7R-G SUMMARY")

print(agent_summary[
    [
        "agent",
        "episode_reward_sum",
        "episode_reward_min",
        "worst_step_1_based",
    ]
].to_string(index=False))

print()
print(f"Replay sum across agents       : {sum_replay_rewards:.6f}")

if history_total_return is not None:
    print(f"History total return           : {history_total_return:.6f}")
    print(
        "Replay/history absolute diff  : "
        f"{abs(sum_replay_rewards - history_total_return):.6f}"
    )

print(
    "Worst aggregate reward step   : "
    f"{int(worst_total_step_row['step'])}"
)

print(
    "Worst aggregate step reward   : "
    f"{float(worst_total_step_row['sum_agent_rewards']):.6f}"
)

dominant = agent_summary.iloc[0]

print()
print(
    "Most negative agent component : "
    f"{dominant['agent']}"
)

print(
    "Its episode reward sum        : "
    f"{dominant['episode_reward_sum']:.6f}"
)

print(
    "Its worst individual step     : "
    f"{int(dominant['worst_step_1_based'])}"
)

# A simple diagnostic classification.
abs_components = np.abs(
    agent_summary["episode_reward_sum"].to_numpy(dtype=float)
)

dominance_ratio = float(
    abs(float(dominant["episode_reward_sum"]))
    / max(float(np.sum(abs_components)), 1e-12)
)

print(
    "Absolute-component dominance  : "
    f"{100.0 * dominance_ratio:.2f}%"
)

if dominance_ratio >= 0.70:
    classification = (
        "CONCENTRATED: one agent/reward stream dominates the episode outlier."
    )
else:
    classification = (
        "DISTRIBUTED: the outlier is spread across multiple reward streams."
    )

print(f"Diagnostic classification     : {classification}")

summary = {
    "target_episode": TARGET_EPISODE,
    "checkpoint_episode_used_for_replay": CHECKPOINT_EPISODE,
    "transition_slice_zero_based": [start, stop - 1],
    "sum_replay_rewards": sum_replay_rewards,
    "history_total_return": history_total_return,
    "worst_aggregate_step": int(worst_total_step_row["step"]),
    "worst_aggregate_step_reward": float(
        worst_total_step_row["sum_agent_rewards"]
    ),
    "most_negative_agent": str(dominant["agent"]),
    "most_negative_agent_episode_sum": float(
        dominant["episode_reward_sum"]
    ),
    "most_negative_agent_worst_step": int(
        dominant["worst_step_1_based"]
    ),
    "absolute_component_dominance_ratio": dominance_ratio,
    "classification": classification,
}

summary_file = (
    DIAGNOSTIC_DIR
    / "episode87_diagnostic_summary.json"
)

with open(
    summary_file,
    "w",
    encoding="utf-8",
) as file:
    json.dump(
        summary,
        file,
        indent=4,
    )


print()
print("[OK] Step 7R-G diagnostic complete.")
print(f"Agent summary : {agent_summary_file}")
print(f"Step detail   : {step_detail_file}")
print(f"Step totals   : {per_step_total_file}")
print(f"Summary JSON  : {summary_file}")
