import pandas as pd
import numpy as np
from pathlib import Path

INPUT = Path(
    "outputs/comment2_final_release_analysis/"
    "all_35_final_test_summaries.csv"
)

OUTPUT_DIR = Path(
    "outputs/comment2_final_release_analysis"
)

df = pd.read_csv(INPUT)

configs = [
    "full",
    "no_confidence",
    "state_only",
    "reward_only",
    "concatenated",
    "multiplicative",
    "no_coordinator",
]

# Metrics selected directly from the verified final TEST summaries.
metrics = [
    "mean_return",
    "mean_coordinator_return",
    "mean_sum_local_return",
    "mean_net_market_cost_usd",
    "mean_market_profit_usd",
    "mean_grid_import_kwh",
    "mean_grid_export_kwh",
    "mean_reserve_revenue_usd",
    "mean_reserve_requested_kwh",
    "mean_reserve_feasible_kwh",
    "mean_reserve_curtailed_kwh",
    "mean_bess_throughput_kwh",
    "mean_sharing_scheduled_kwh",
    "mean_sharing_received_kwh",
    "mean_sharing_loss_kwh",
    "mean_peak_grid_import_kw",
    "mean_peak_grid_export_kw",
    "mean_mean_balance_violation_kw_per_step",
    "mean_maximum_balance_violation_kw",
    "total_power_balance_violation_steps",
    "total_transformer_violation_steps",
    "total_soc_below_min_count",
    "total_soc_above_max_count",
]

# Convert explicitly to numeric.
for metric in metrics:
    df[metric] = pd.to_numeric(
        df[metric],
        errors="coerce"
    )

rows = []

for config in configs:

    sub = df[df["configuration"] == config].copy()

    if len(sub) != 5:
        raise RuntimeError(
            f"{config}: expected 5 seeds, found {len(sub)}"
        )

    seeds = sorted(
        pd.to_numeric(sub["seed"]).astype(int).tolist()
    )

    if seeds != [11, 22, 33, 44, 55]:
        raise RuntimeError(
            f"{config}: unexpected seeds {seeds}"
        )

    row = {
        "configuration": config,
        "n_seeds": len(sub),
    }

    for metric in metrics:

        values = sub[metric].to_numpy(dtype=float)

        row[f"{metric}_mean"] = np.mean(values)

        # Sample SD across the five independent seeds.
        row[f"{metric}_sd"] = np.std(
            values,
            ddof=1
        )

    rows.append(row)

summary = pd.DataFrame(rows)

summary_file = (
    OUTPUT_DIR /
    "ablation_5seed_mean_sd.csv"
)

summary.to_csv(
    summary_file,
    index=False
)

print("=" * 80)
print("5-SEED ABLATION AGGREGATION")
print("=" * 80)

print(
    summary[
        [
            "configuration",
            "n_seeds",
            "mean_return_mean",
            "mean_return_sd",
            "mean_net_market_cost_usd_mean",
            "mean_net_market_cost_usd_sd",
            "mean_market_profit_usd_mean",
            "mean_market_profit_usd_sd",
            "mean_mean_balance_violation_kw_per_step_mean",
            "mean_mean_balance_violation_kw_per_step_sd",
        ]
    ].to_string(index=False)
)

print()
print("Output:")
print(summary_file)
print()
print("PASS: 7 configurations x 5 seeds = 35 runs.")
