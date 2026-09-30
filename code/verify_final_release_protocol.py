import pandas as pd
from pathlib import Path

p = Path(
    "outputs/comment2_final_release_analysis/"
    "all_35_final_test_summaries.csv"
)

df = pd.read_csv(p)

print("=" * 70)
print("FINAL RELEASE PROTOCOL VERIFICATION")
print("=" * 70)

print("\nTotal runs:")
print(len(df))

print("\nConfigurations:")
print(df["configuration"].value_counts().sort_index())

print("\nSeeds:")
print(sorted(df["seed"].astype(int).unique()))

print("\nLocked checkpoints:")
print(df["locked_checkpoint_episode"].value_counts(dropna=False))

print("\nCheckpoint selection source:")
print(df["checkpoint_selection_source"].value_counts(dropna=False))

print("\nTEST episodes:")
print(df["test_episodes"].value_counts(dropna=False))

print("\nCheckpoint comparison on TEST:")
print(df["checkpoint_comparison_on_test"].value_counts(dropna=False))

print("\nDeterministic actions:")
print(df["deterministic_actions"].value_counts(dropna=False))

print("\nGradient updates:")
print(df["gradient_updates"].value_counts(dropna=False))

print("\nReplay writes:")
print(df["replay_writes"].value_counts(dropna=False))

assert len(df) == 35
assert set(df["locked_checkpoint_episode"].astype(int)) == {5000}
assert set(df["test_episodes"].astype(int)) == {1268}

assert (
    df["checkpoint_selection_source"]
    == "fixed_terminal_training_checkpoint"
).all()

assert (
    df["checkpoint_comparison_on_test"]
    .astype(str)
    .str.lower()
    == "false"
).all()

assert (
    df["deterministic_actions"]
    .astype(str)
    .str.lower()
    == "true"
).all()

assert (
    df["gradient_updates"]
    .astype(str)
    .str.lower()
    == "false"
).all()

assert (
    df["replay_writes"]
    .astype(str)
    .str.lower()
    == "false"
).all()

print("\n" + "=" * 70)
print("PASS: FINAL 35-RUN PROTOCOL IS INTERNALLY CONSISTENT")
print("Checkpoint = 5000")
print("TEST windows per run = 1268")
print("Checkpoint comparison on TEST = False")
print("Gradient updates on TEST = False")
print("=" * 70)
