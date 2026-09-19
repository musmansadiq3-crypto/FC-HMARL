from pathlib import Path
import json

import numpy as np
import pandas as pd
from scipy.stats import t

ROOT = Path("outputs/final_benchmark_5000")
OUT = ROOT / "final_statistical_evidence"
OUT.mkdir(parents=True, exist_ok=True)

SEEDS = [11, 22, 33, 44, 55]

BASELINES = {
    "passive_grid_only":
        ROOT / "passive_grid_only" / "seed_11" / "final_test_summary.csv",

    "rule_based_bess_ems":
        ROOT / "rule_based_bess_ems" / "seed_11" / "final_test_summary.csv",
}

FULL_FILES = {
    seed:
        ROOT / "full_fc_hmarl" / f"seed_{seed}" / "final_test_summary.csv"
    for seed in SEEDS
}


def read_one(path):
    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_csv(path)

    if len(df) != 1:
        raise RuntimeError(
            f"Expected exactly one summary row in {path}"
        )

    return df.iloc[0]


def mean_sd_ci(values):
    x = np.asarray(values, dtype=float)

    n = len(x)
    mean = float(np.mean(x))

    if n < 2:
        return mean, np.nan, np.nan, np.nan

    sd = float(np.std(x, ddof=1))
    se = sd / np.sqrt(n)

    critical = float(
        t.ppf(0.975, df=n - 1)
    )

    lower = mean - critical * se
    upper = mean + critical * se

    return mean, sd, lower, upper


# ============================================================
# Load fixed deterministic baselines
# ============================================================

baseline_rows = {}

for name, path in BASELINES.items():
    row = read_one(path)

    if int(row["test_episodes"]) != 1268:
        raise RuntimeError(
            f"{name}: expected 1268 test windows."
        )

    baseline_rows[name] = row


# ============================================================
# Load five independently trained Full FC-HMARL policies
# ============================================================

full_rows = {}

for seed, path in FULL_FILES.items():
    row = read_one(path)

    if int(row["test_episodes"]) != 1268:
        raise RuntimeError(
            f"Full seed {seed}: expected 1268 test windows."
        )

    full_rows[seed] = row


# ============================================================
# Metrics used in final benchmark evidence
# ============================================================

METRICS = {
    "mean_return":
        "return",

    "mean_net_market_cost_usd":
        "net_market_cost_usd",

    "mean_grid_import_kwh":
        "grid_import_kwh",

    "mean_grid_export_kwh":
        "grid_export_kwh",

    "mean_bess_throughput_kwh":
        "bess_throughput_kwh",

    "mean_sharing_scheduled_kwh":
        "sharing_scheduled_kwh",

    "mean_mean_balance_violation_kw_per_step":
        "balance_violation_kw_per_step",
}


# ============================================================
# Raw evidence table
# ============================================================

raw_rows = []

for name, row in baseline_rows.items():
    item = {
        "controller": name,
        "training_seed": "",
        "controller_type": "fixed_deterministic_baseline",
        "test_windows": int(row["test_episodes"]),
    }

    for source, label in METRICS.items():
        item[label] = float(row[source])

    item["transformer_violation_steps"] = int(
        row["total_transformer_violation_steps"]
    )

    item["soc_below_min_count"] = int(
        row["total_soc_below_min_count"]
    )

    item["soc_above_max_count"] = int(
        row["total_soc_above_max_count"]
    )

    raw_rows.append(item)


for seed, row in full_rows.items():
    item = {
        "controller": "full_fc_hmarl",
        "training_seed": seed,
        "controller_type": "independently_trained_policy",
        "test_windows": int(row["test_episodes"]),
    }

    for source, label in METRICS.items():
        item[label] = float(row[source])

    item["transformer_violation_steps"] = int(
        row["total_transformer_violation_steps"]
    )

    item["soc_below_min_count"] = int(
        row["total_soc_below_min_count"]
    )

    item["soc_above_max_count"] = int(
        row["total_soc_above_max_count"]
    )

    raw_rows.append(item)


raw_df = pd.DataFrame(raw_rows)

raw_df.to_csv(
    OUT / "benchmark_raw_summary.csv",
    index=False,
)


# ============================================================
# Full FC-HMARL across-training-seed statistics
# ============================================================

full_stats = []

for source, label in METRICS.items():

    values = [
        float(full_rows[seed][source])
        for seed in SEEDS
    ]

    mean, sd, ci_low, ci_high = mean_sd_ci(values)

    full_stats.append({
        "metric": label,
        "n_independent_training_seeds": len(SEEDS),
        "mean": mean,
        "sample_sd": sd,
        "ci95_lower": ci_low,
        "ci95_upper": ci_high,
    })


pd.DataFrame(full_stats).to_csv(
    OUT / "full_fc_hmarl_seed_statistics.csv",
    index=False,
)


# ============================================================
# Fixed baseline values
#
# IMPORTANT:
# No artificial SD or CI is assigned to deterministic baselines.
# ============================================================

baseline_table = []

for name, row in baseline_rows.items():

    item = {
        "controller": name,
        "controller_type": "fixed_deterministic_baseline",
        "test_windows": int(row["test_episodes"]),
    }

    for source, label in METRICS.items():
        item[label] = float(row[source])

    baseline_table.append(item)


pd.DataFrame(baseline_table).to_csv(
    OUT / "fixed_baseline_results.csv",
    index=False,
)


# ============================================================
# Full-minus-fixed-baseline seed-level differences
#
# These are NOT paired t-tests between two trained methods.
# Each deterministic baseline is fixed.
#
# CI is calculated across the five independently trained
# Full FC-HMARL policies:
#
# Delta = Full(seed) - fixed baseline
# ============================================================

delta_rows = []

for baseline_name, baseline in baseline_rows.items():

    for source, label in METRICS.items():

        baseline_value = float(
            baseline[source]
        )

        deltas = np.asarray(
            [
                float(full_rows[seed][source])
                - baseline_value
                for seed in SEEDS
            ],
            dtype=float,
        )

        mean, sd, ci_low, ci_high = mean_sd_ci(
            deltas
        )

        delta_rows.append({
            "comparison":
                f"full_fc_hmarl_minus_{baseline_name}",

            "metric":
                label,

            "delta_definition":
                "Full FC-HMARL - fixed baseline",

            "n_independent_full_training_seeds":
                len(SEEDS),

            "fixed_baseline_value":
                baseline_value,

            "mean_delta":
                mean,

            "sample_sd_of_delta":
                sd,

            "ci95_lower":
                ci_low,

            "ci95_upper":
                ci_high,
        })


delta_df = pd.DataFrame(delta_rows)

delta_df.to_csv(
    OUT / "full_minus_fixed_baseline_differences.csv",
    index=False,
)


# ============================================================
# Cost-reduction percentages
#
# Positive value = Full has lower cost than baseline.
# Calculated separately for every independently trained Full seed.
# ============================================================

cost_rows = []

for baseline_name, baseline in baseline_rows.items():

    baseline_cost = float(
        baseline["mean_net_market_cost_usd"]
    )

    reductions = []

    for seed in SEEDS:

        full_cost = float(
            full_rows[seed][
                "mean_net_market_cost_usd"
            ]
        )

        reduction_pct = (
            (baseline_cost - full_cost)
            / baseline_cost
            * 100.0
        )

        reductions.append(
            reduction_pct
        )

        cost_rows.append({
            "baseline": baseline_name,
            "full_training_seed": seed,
            "baseline_cost_usd": baseline_cost,
            "full_cost_usd": full_cost,
            "cost_reduction_percent":
                reduction_pct,
        })

    mean, sd, ci_low, ci_high = mean_sd_ci(
        reductions
    )

    cost_rows.append({
        "baseline":
            baseline_name,

        "full_training_seed":
            "aggregate",

        "baseline_cost_usd":
            baseline_cost,

        "full_cost_usd":
            np.mean([
                float(
                    full_rows[seed][
                        "mean_net_market_cost_usd"
                    ]
                )
                for seed in SEEDS
            ]),

        "cost_reduction_percent":
            mean,

        "cost_reduction_sample_sd":
            sd,

        "cost_reduction_ci95_lower":
            ci_low,

        "cost_reduction_ci95_upper":
            ci_high,
    })


pd.DataFrame(cost_rows).to_csv(
    OUT / "net_market_cost_reduction.csv",
    index=False,
)


# ============================================================
# Feasibility
# ============================================================

feasibility = []

for _, row in raw_df.iterrows():

    feasibility.append({
        "controller":
            row["controller"],

        "training_seed":
            row["training_seed"],

        "test_windows":
            int(row["test_windows"]),

        "transformer_violation_steps":
            int(row["transformer_violation_steps"]),

        "soc_below_min_count":
            int(row["soc_below_min_count"]),

        "soc_above_max_count":
            int(row["soc_above_max_count"]),

        "all_transformers_feasible":
            int(row["transformer_violation_steps"]) == 0,

        "all_soc_feasible":
            (
                int(row["soc_below_min_count"]) == 0
                and int(row["soc_above_max_count"]) == 0
            ),
    })


pd.DataFrame(feasibility).to_csv(
    OUT / "benchmark_feasibility.csv",
    index=False,
)


# ============================================================
# Protocol metadata
# ============================================================

metadata = {
    "evaluation_split":
        "locked_test",

    "test_windows":
        1268,

    "episode_length_hours":
        24,

    "locked_checkpoint_episode":
        5000,

    "full_training_seeds":
        SEEDS,

    "full_policy_count":
        len(SEEDS),

    "baseline_training_seed_variability":
        False,

    "baseline_evaluation":
        (
            "Each deterministic baseline was evaluated once "
            "on the same locked 1268 TEST windows."
        ),

    "statistical_unit_for_full":
        "independently trained FC-HMARL policy seed",

    "baseline_uncertainty_treatment":
        (
            "No artificial seed SD or CI assigned to fixed "
            "deterministic baselines."
        ),

    "difference_interval_definition":
        (
            "Student-t 95% CI across five Full FC-HMARL "
            "training-seed differences relative to each fixed baseline."
        ),

    "window_level_independence_assumed":
        False,

    "naive_window_level_p_values_reported":
        False,

    "deterministic_actions":
        True,

    "gradient_updates":
        False,

    "replay_writes":
        False,

    "test_checkpoint_selection":
        False,
}

with open(
    OUT / "benchmark_statistical_protocol.json",
    "w",
    encoding="utf-8",
) as f:
    json.dump(
        metadata,
        f,
        indent=4,
    )


print()
print("=" * 72)
print("FINAL BENCHMARK STATISTICAL EVIDENCE COMPLETE")
print("=" * 72)

print(
    f"Full FC-HMARL independent training seeds: "
    f"{len(SEEDS)}"
)

print(
    "Fixed deterministic baselines: "
    "Passive grid-only and Rule-based BESS EMS"
)

print()
print("Files written to:")
print(OUT)

print()
print("[OK] No artificial baseline seed variance was created.")
print("[OK] No rolling-window independence assumption was used.")
print("[OK] 95% CIs use independent Full training seeds.")
