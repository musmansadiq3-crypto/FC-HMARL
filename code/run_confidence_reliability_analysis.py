from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr


# ============================================================
# Paths
# ============================================================

ROOT = Path(__file__).resolve().parent

FORECAST_FILE = (
    ROOT
    / "outputs"
    / "forecasting"
    / "causal_confidence"
    / "final_leakage_free_test_forecast.npz"
)

PI_FILE = (
    ROOT
    / "outputs"
    / "forecasting"
    / "prediction_interval_evidence"
    / "locked_test_prediction_intervals.npz"
)

OUT_DIR = (
    ROOT
    / "outputs"
    / "forecasting"
    / "confidence_reliability"
)

OUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Load frozen forecasting evidence
# ============================================================

forecast_data = np.load(FORECAST_FILE, allow_pickle=True)
pi_data = np.load(PI_FILE, allow_pickle=True)

actual = forecast_data["actual_original"].astype(float)
forecast = forecast_data["forecast_original"].astype(float)

confidence = forecast_data["causal_confidence_24h"].astype(float)

validation_error = (
    forecast_data["validation_mean_error_by_horizon"].astype(float)
)

feature_names = forecast_data["feature_names"].astype(str)

lower = pi_data["lower_bound_original"].astype(float)
upper = pi_data["upper_bound_original"].astype(float)

nominal_coverage = float(pi_data["nominal_coverage"])


# ============================================================
# Sanity checks
# ============================================================

assert actual.shape == forecast.shape
assert actual.shape == lower.shape == upper.shape

n_windows, n_horizons, n_targets = actual.shape

assert n_horizons == len(confidence)
assert n_targets == len(feature_names)

print("=" * 80)
print("FORECAST-CONFIDENCE RELIABILITY ANALYSIS")
print("=" * 80)

print(f"Locked TEST windows : {n_windows}")
print(f"Forecast horizons   : {n_horizons}")
print(f"Forecast variables  : {n_targets}")
print(f"Nominal PI coverage : {nominal_coverage:.4f}")

print("\nFeatures:")
for i, name in enumerate(feature_names):
    print(f"  {i}: {name}")

print(
    f"\nConfidence range: "
    f"{confidence.min():.6f} to {confidence.max():.6f}"
)

print(
    f"Confidence mean ± SD: "
    f"{confidence.mean():.6f} ± {confidence.std(ddof=1):.6f}"
)


# ============================================================
# Realized TEST errors
# ============================================================

absolute_error = np.abs(actual - forecast)

# Mean realized absolute error for each horizon and target
test_mae_horizon_target = absolute_error.mean(axis=0)

# Aggregate normalized error across targets.
# Each target is normalized by its own overall TEST mean absolute error
# so that variables with large physical units do not dominate.
target_scale = absolute_error.mean(axis=(0, 1))

eps = 1e-12

normalized_absolute_error = (
    absolute_error
    / (target_scale.reshape(1, 1, -1) + eps)
)

aggregate_normalized_error_by_horizon = (
    normalized_absolute_error.mean(axis=(0, 2))
)


# ============================================================
# Prediction interval coverage by horizon and target
# ============================================================

covered = (actual >= lower) & (actual <= upper)

picp_horizon_target = covered.mean(axis=0)

picp_by_horizon = covered.mean(axis=(0, 2))


# ============================================================
# Horizon-level evidence table
# ============================================================

horizon_rows = []

for h in range(n_horizons):

    row = {
        "horizon": h + 1,
        "confidence": confidence[h],
        "aggregate_normalized_test_error":
            aggregate_normalized_error_by_horizon[h],
        "overall_picp": picp_by_horizon[h],
    }

    for k, feature in enumerate(feature_names):
        row[f"{feature}_test_mae"] = (
            test_mae_horizon_target[h, k]
        )
        row[f"{feature}_picp"] = (
            picp_horizon_target[h, k]
        )
        row[f"{feature}_validation_error"] = (
            validation_error[h, k]
        )

    horizon_rows.append(row)

horizon_df = pd.DataFrame(horizon_rows)

horizon_file = OUT_DIR / "confidence_reliability_by_horizon.csv"
horizon_df.to_csv(horizon_file, index=False)


# ============================================================
# Correlation analysis
# ============================================================

correlation_rows = []

# ------------------------------------------------------------
# Aggregate normalized TEST error
# ------------------------------------------------------------

rho, p_s = spearmanr(
    confidence,
    aggregate_normalized_error_by_horizon
)

r, p_p = pearsonr(
    confidence,
    aggregate_normalized_error_by_horizon
)

correlation_rows.append(
    {
        "variable": "aggregate_normalized_error",
        "spearman_rho": rho,
        "spearman_p": p_s,
        "pearson_r": r,
        "pearson_p": p_p,
        "n_horizons": n_horizons,
    }
)


# ------------------------------------------------------------
# Individual forecast variables
# ------------------------------------------------------------

for k, feature in enumerate(feature_names):

    errors = test_mae_horizon_target[:, k]

    rho, p_s = spearmanr(confidence, errors)
    r, p_p = pearsonr(confidence, errors)

    correlation_rows.append(
        {
            "variable": feature,
            "spearman_rho": rho,
            "spearman_p": p_s,
            "pearson_r": r,
            "pearson_p": p_p,
            "n_horizons": n_horizons,
        }
    )


# ------------------------------------------------------------
# Prediction interval coverage relationship
# ------------------------------------------------------------

rho, p_s = spearmanr(confidence, picp_by_horizon)
r, p_p = pearsonr(confidence, picp_by_horizon)

correlation_rows.append(
    {
        "variable": "overall_prediction_interval_coverage",
        "spearman_rho": rho,
        "spearman_p": p_s,
        "pearson_r": r,
        "pearson_p": p_p,
        "n_horizons": n_horizons,
    }
)

correlation_df = pd.DataFrame(correlation_rows)

correlation_file = (
    OUT_DIR
    / "confidence_error_correlation.csv"
)

correlation_df.to_csv(correlation_file, index=False)


# ============================================================
# Confidence quintile analysis
# ============================================================

bin_df = horizon_df[
    [
        "horizon",
        "confidence",
        "aggregate_normalized_test_error",
        "overall_picp",
    ]
].copy()

# Rank first so qcut remains robust if confidence values tie.
bin_df["confidence_group"] = pd.qcut(
    bin_df["confidence"].rank(method="first"),
    q=5,
    labels=[
        "Q1_lowest",
        "Q2",
        "Q3",
        "Q4",
        "Q5_highest",
    ],
)

bin_summary = (
    bin_df
    .groupby("confidence_group", observed=True)
    .agg(
        n_horizons=("horizon", "count"),
        mean_confidence=("confidence", "mean"),
        min_confidence=("confidence", "min"),
        max_confidence=("confidence", "max"),
        mean_normalized_test_error=(
            "aggregate_normalized_test_error",
            "mean",
        ),
        median_normalized_test_error=(
            "aggregate_normalized_test_error",
            "median",
        ),
        mean_picp=("overall_picp", "mean"),
    )
    .reset_index()
)

bin_file = (
    OUT_DIR
    / "confidence_reliability_quintiles.csv"
)

bin_summary.to_csv(bin_file, index=False)


# ============================================================
# Validation-error consistency check
# ============================================================

validation_mean_by_horizon = validation_error.mean(axis=1)

rho_val, p_val = spearmanr(
    confidence,
    validation_mean_by_horizon
)

validation_df = pd.DataFrame(
    {
        "horizon": np.arange(1, n_horizons + 1),
        "confidence": confidence,
        "mean_validation_error":
            validation_mean_by_horizon,
        "aggregate_normalized_test_error":
            aggregate_normalized_error_by_horizon,
    }
)

validation_file = (
    OUT_DIR
    / "confidence_validation_test_consistency.csv"
)

validation_df.to_csv(validation_file, index=False)


# ============================================================
# Text summary
# ============================================================

summary_file = (
    OUT_DIR
    / "confidence_reliability_summary.txt"
)

with open(summary_file, "w", encoding="utf-8") as f:

    f.write(
        "FORECAST-CONFIDENCE RELIABILITY ANALYSIS\n"
    )
    f.write("=" * 70 + "\n\n")

    f.write(
        f"Locked TEST windows: {n_windows}\n"
    )
    f.write(
        f"Forecast horizons: {n_horizons}\n"
    )
    f.write(
        f"Forecast variables: {n_targets}\n"
    )
    f.write(
        f"Nominal PI coverage: "
        f"{nominal_coverage:.4f}\n\n"
    )

    f.write(
        "Confidence statistics\n"
    )
    f.write(
        f"Minimum: {confidence.min():.6f}\n"
    )
    f.write(
        f"Maximum: {confidence.max():.6f}\n"
    )
    f.write(
        f"Mean: {confidence.mean():.6f}\n"
    )
    f.write(
        f"SD: {confidence.std(ddof=1):.6f}\n\n"
    )

    f.write(
        "Confidence vs realized TEST error\n"
    )
    f.write("-" * 70 + "\n")
    f.write(
        correlation_df.to_string(index=False)
    )
    f.write("\n\n")

    f.write(
        "Confidence quintile analysis\n"
    )
    f.write("-" * 70 + "\n")
    f.write(
        bin_summary.to_string(index=False)
    )
    f.write("\n\n")

    f.write(
        "Confidence vs validation mean error\n"
    )
    f.write("-" * 70 + "\n")
    f.write(
        f"Spearman rho = {rho_val:.6f}\n"
    )
    f.write(
        f"p-value = {p_val:.6g}\n"
    )


# ============================================================
# Console output
# ============================================================

print("\n" + "=" * 80)
print("CONFIDENCE VS REALIZED TEST ERROR")
print("=" * 80)

print(
    correlation_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}",
    )
)

print("\n" + "=" * 80)
print("CONFIDENCE QUINTILES")
print("=" * 80)

print(
    bin_summary.to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}",
    )
)

print("\n" + "=" * 80)
print("CONFIDENCE VS VALIDATION ERROR")
print("=" * 80)

print(
    f"Spearman rho = {rho_val:.6f}"
)
print(
    f"p-value      = {p_val:.6g}"
)

print("\nFiles written:")

for path in [
    horizon_file,
    correlation_file,
    bin_file,
    validation_file,
    summary_file,
]:
    print(f"  {path}")

print("\nAnalysis complete.")