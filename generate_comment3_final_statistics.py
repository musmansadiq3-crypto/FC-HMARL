from pathlib import Path
import pandas as pd
import numpy as np
from scipy import stats

ROOT = Path("outputs/comment3_ablation/final_35_test_summaries")
OUT = Path("outputs/comment3_ablation/final_statistical_evidence")
OUT.mkdir(parents=True, exist_ok=True)

CONFIG_ORDER = [
    "full",
    "no_confidence",
    "state_only",
    "reward_only",
    "concatenated",
    "multiplicative",
    "no_coordinator",
]

SEEDS = [11, 22, 33, 44, 55]

# ------------------------------------------------------------
# 1. Read and validate all 35 locked final-test summaries
# ------------------------------------------------------------
records = []

for config in CONFIG_ORDER:
    for seed in SEEDS:
        path = (
            ROOT
            / config
            / f"seed_{seed}"
            / "evaluation_test_5000"
            / "final_test_summary.csv"
        )

        if not path.exists():
            raise FileNotFoundError(f"Missing required result: {path}")

        df = pd.read_csv(path)

        if len(df) != 1:
            raise ValueError(f"Expected one summary row in {path}")

        row = df.iloc[0].to_dict()
        row["configuration"] = config
        row["seed"] = seed
        row["source_file"] = str(path)
        records.append(row)

raw = pd.DataFrame(records)

if len(raw) != 35:
    raise RuntimeError(f"Expected 35 final summaries, found {len(raw)}")

# Locked evaluation-protocol checks
assert (raw["evaluation_split"] == "test").all()
assert (raw["locked_checkpoint_episode"] == 5000).all()
assert (
    raw["checkpoint_selection_source"]
    == "fixed_terminal_training_checkpoint"
).all()
assert raw["deterministic_actions"].astype(bool).all()
assert (~raw["gradient_updates"].astype(bool)).all()
assert (~raw["replay_writes"].astype(bool)).all()
assert (~raw["checkpoint_comparison_on_test"].astype(bool)).all()
assert (raw["test_episodes"] == 1268).all()

raw.to_csv(OUT / "all_35_final_test_summaries.csv", index=False)

# ------------------------------------------------------------
# 2. Across-seed descriptive statistics and 95% t confidence
#    intervals. Statistical unit = independently trained seed.
# ------------------------------------------------------------
METRICS = [
    "mean_return",
    "mean_net_market_cost_usd",
    "mean_sharing_scheduled_kwh",
    "mean_grid_import_kwh",
    "mean_grid_export_kwh",
    "mean_bess_throughput_kwh",
    "mean_mean_balance_violation_kw_per_step",
    "total_transformer_violation_steps",
    "total_soc_below_min_count",
    "total_soc_above_max_count",
]

summary_rows = []

for config in CONFIG_ORDER:
    g = raw[raw["configuration"] == config]

    for metric in METRICS:
        x = pd.to_numeric(g[metric], errors="raise").to_numpy(dtype=float)

        n = len(x)
        mean = np.mean(x)
        sd = np.std(x, ddof=1)
        sem = stats.sem(x)

        if n > 1 and sd > 0:
            low, high = stats.t.interval(
                0.95,
                df=n - 1,
                loc=mean,
                scale=sem,
            )
        else:
            low = mean
            high = mean

        summary_rows.append(
            {
                "configuration": config,
                "metric": metric,
                "n_seeds": n,
                "mean": mean,
                "sample_sd": sd,
                "ci95_lower": low,
                "ci95_upper": high,
            }
        )

summary = pd.DataFrame(summary_rows)
summary.to_csv(
    OUT / "all_configurations_statistics.csv",
    index=False,
)

# ------------------------------------------------------------
# 3. Seed-matched paired comparisons:
#    Delta R = R_Full - R_Ablation
# ------------------------------------------------------------
full = (
    raw[raw["configuration"] == "full"]
    .set_index("seed")["mean_return"]
    .astype(float)
)

paired_rows = []

for config in CONFIG_ORDER[1:]:
    other = (
        raw[raw["configuration"] == config]
        .set_index("seed")["mean_return"]
        .astype(float)
    )

    common = sorted(set(full.index).intersection(other.index))

    full_values = full.loc[common].to_numpy(dtype=float)
    other_values = other.loc[common].to_numpy(dtype=float)

    delta = full_values - other_values

    n = len(delta)
    mean_delta = np.mean(delta)
    sd_delta = np.std(delta, ddof=1)
    sem_delta = stats.sem(delta)

    if n > 1 and sd_delta > 0:
        ci_low, ci_high = stats.t.interval(
            0.95,
            df=n - 1,
            loc=mean_delta,
            scale=sem_delta,
        )
        t_stat, p_value = stats.ttest_rel(
            full_values,
            other_values,
        )
    else:
        ci_low = mean_delta
        ci_high = mean_delta
        t_stat = np.nan
        p_value = np.nan

    paired_rows.append(
        {
            "comparison": f"full_vs_{config}",
            "definition": "Delta_R = R_Full - R_Ablation",
            "n_paired_seeds": n,
            "mean_delta_return": mean_delta,
            "sample_sd_delta": sd_delta,
            "ci95_lower": ci_low,
            "ci95_upper": ci_high,
            "paired_t_statistic": t_stat,
            "paired_p_value_two_sided": p_value,
            "significant_at_0.05": bool(p_value < 0.05),
        }
    )

paired = pd.DataFrame(paired_rows)
paired.to_csv(
    OUT / "paired_statistical_comparisons.csv",
    index=False,
)

# ------------------------------------------------------------
# 4. Operational summary for manuscript/reviewer inspection
# ------------------------------------------------------------
operational_metrics = [
    "mean_return",
    "mean_net_market_cost_usd",
    "mean_sharing_scheduled_kwh",
    "mean_reserve_requested_kwh",
    "mean_reserve_feasible_kwh",
    "mean_reserve_revenue_usd",
    "mean_grid_import_kwh",
    "mean_grid_export_kwh",
    "mean_bess_throughput_kwh",
    "mean_mean_balance_violation_kw_per_step",
]

operational_rows = []

for config in CONFIG_ORDER:
    g = raw[raw["configuration"] == config]

    result = {"configuration": config, "n_seeds": len(g)}

    for metric in operational_metrics:
        x = pd.to_numeric(g[metric], errors="raise")
        result[f"{metric}_mean"] = x.mean()
        result[f"{metric}_sample_sd"] = x.std(ddof=1)

    operational_rows.append(result)

pd.DataFrame(operational_rows).to_csv(
    OUT / "operational_metrics_summary.csv",
    index=False,
)

# ------------------------------------------------------------
# 5. Constraint-feasibility summary
# ------------------------------------------------------------
feasibility_rows = []

for config in CONFIG_ORDER:
    g = raw[raw["configuration"] == config]

    feasibility_rows.append(
        {
            "configuration": config,
            "n_seeds": len(g),
            "transformer_violation_steps_sum":
                pd.to_numeric(
                    g["total_transformer_violation_steps"]
                ).sum(),
            "soc_below_min_count_sum":
                pd.to_numeric(
                    g["total_soc_below_min_count"]
                ).sum(),
            "soc_above_max_count_sum":
                pd.to_numeric(
                    g["total_soc_above_max_count"]
                ).sum(),
            "all_transformers_feasible_all_seeds":
                g["all_transformers_feasible"].astype(bool).all(),
            "all_soc_feasible_all_seeds":
                g["all_soc_feasible"].astype(bool).all(),
        }
    )

pd.DataFrame(feasibility_rows).to_csv(
    OUT / "constraint_feasibility_summary.csv",
    index=False,
)

# ------------------------------------------------------------
# 6. Evaluation protocol record
# ------------------------------------------------------------
protocol = pd.DataFrame(
    [
        {
            "number_of_configurations": 7,
            "seeds_per_configuration": 5,
            "total_final_evaluations": 35,
            "seeds": "11,22,33,44,55",
            "evaluation_split": "test",
            "test_windows": 1268,
            "checkpoint_episode": 5000,
            "checkpoint_selection":
                "fixed_terminal_training_checkpoint",
            "deterministic_actions": True,
            "gradient_updates_during_test": False,
            "replay_writes_during_test": False,
            "checkpoint_selection_on_test": False,
            "statistical_unit":
                "independently trained random seed",
            "confidence_interval":
                "two-sided 95% Student-t interval across seeds",
            "paired_test":
                "two-sided paired t-test using matched seeds",
        }
    ]
)

protocol.to_csv(
    OUT / "evaluation_protocol.csv",
    index=False,
)

print("=" * 72)
print("COMMENT 3 FINAL STATISTICAL EVIDENCE")
print("=" * 72)
print(f"Validated final summaries : {len(raw)}")
print(f"Output directory          : {OUT}")
print()

print("MEAN RETURN ACROSS FIVE SEEDS")
print(
    summary[summary["metric"] == "mean_return"][
        [
            "configuration",
            "mean",
            "sample_sd",
            "ci95_lower",
            "ci95_upper",
        ]
    ].to_string(index=False)
)

print()
print("PAIRED FULL-vs-ABLATION TESTS")
print(
    paired[
        [
            "comparison",
            "mean_delta_return",
            "ci95_lower",
            "ci95_upper",
            "paired_p_value_two_sided",
            "significant_at_0.05",
        ]
    ].to_string(index=False)
)

print()
print("SUCCESS: final statistical evidence generated.")
