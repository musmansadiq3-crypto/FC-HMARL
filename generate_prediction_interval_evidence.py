from pathlib import Path
import json

import numpy as np
import pandas as pd

from forecasting.uncertainty import (
    ForecastUncertaintyConfig,
    ForecastUncertaintyEstimator,
    calculate_interval_coverage,
)


PROJECT_ROOT = Path(__file__).resolve().parent

SEQUENCE_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "forecasting"
    / "forecasting_sequences.npz"
)

SCALER_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "forecasting"
    / "forecasting_scaler.csv"
)

VALIDATION_TRANSFORMER_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "forecasting"
    / "causal_confidence"
    / "validation_transformer_predictions.npz"
)

SELECTION_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "forecasting"
    / "causal_confidence"
    / "validation_selected_forecasters.csv"
)

FINAL_TEST_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "forecasting"
    / "causal_confidence"
    / "final_leakage_free_test_forecast.npz"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "forecasting"
    / "prediction_interval_evidence"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


FEATURE_NAMES = [
    "pv_power_kw",
    "load_kw",
    "ev_power_kw",
    "price_usd_per_kwh",
]

NOMINAL_COVERAGE = 0.95
TOL = 1e-10


def inverse_scale_array(values, scaler):
    values = np.asarray(values, dtype=np.float64)

    output = np.zeros_like(
        values,
        dtype=np.float64,
    )

    for j, feature_name in enumerate(FEATURE_NAMES):
        minimum = float(
            scaler.loc[
                feature_name,
                "train_min",
            ]
        )

        maximum = float(
            scaler.loc[
                feature_name,
                "train_max",
            ]
        )

        output[:, :, j] = (
            values[:, :, j]
            * (maximum - minimum)
            + minimum
        )

    return output


def apply_physical_clipping(forecast):
    output = np.asarray(
        forecast,
        dtype=np.float64,
    ).copy()

    # PV, load and EV power are non-negative in
    # the final forecasting pipeline.
    output[:, :, 0] = np.clip(
        output[:, :, 0],
        0.0,
        None,
    )

    output[:, :, 1] = np.clip(
        output[:, :, 1],
        0.0,
        None,
    )

    output[:, :, 2] = np.clip(
        output[:, :, 2],
        0.0,
        None,
    )

    return output


print("=" * 78)
print("FC-HMARL FINAL PREDICTION-INTERVAL EVIDENCE")
print("=" * 78)


required_files = [
    SEQUENCE_FILE,
    SCALER_FILE,
    VALIDATION_TRANSFORMER_FILE,
    SELECTION_FILE,
    FINAL_TEST_FILE,
]

for file_path in required_files:
    if not file_path.exists():
        raise FileNotFoundError(
            f"Required file not found: {file_path}"
        )


# ------------------------------------------------------------
# Load validation/test sequence data
# ------------------------------------------------------------

sequence_data = np.load(
    SEQUENCE_FILE,
    allow_pickle=True,
)

X_val = np.asarray(
    sequence_data["X_val"],
    dtype=np.float64,
)

y_val_norm = np.asarray(
    sequence_data["y_val"],
    dtype=np.float64,
)

y_test_norm = np.asarray(
    sequence_data["y_test"],
    dtype=np.float64,
)


if y_val_norm.ndim != 3:
    raise RuntimeError(
        "y_val must have shape (samples, horizon, targets)."
    )

if y_test_norm.ndim != 3:
    raise RuntimeError(
        "y_test must have shape (samples, horizon, targets)."
    )

if y_val_norm.shape[1:] != (24, 4):
    raise RuntimeError(
        f"Unexpected validation shape: {y_val_norm.shape}"
    )

if y_test_norm.shape[1:] != (24, 4):
    raise RuntimeError(
        f"Unexpected test shape: {y_test_norm.shape}"
    )


# ------------------------------------------------------------
# Load training-derived scaler
# ------------------------------------------------------------

scaler_df = pd.read_csv(
    SCALER_FILE
)

required_scaler_columns = {
    "target",
    "train_min",
    "train_max",
}

if not required_scaler_columns.issubset(
    scaler_df.columns
):
    raise RuntimeError(
        "Scaler does not contain the required columns."
    )

scaler = scaler_df.set_index(
    "target"
)

for feature_name in FEATURE_NAMES:
    if feature_name not in scaler.index:
        raise RuntimeError(
            f"Missing scaler entry for {feature_name}"
        )


actual_val_original = inverse_scale_array(
    y_val_norm,
    scaler,
)


# ------------------------------------------------------------
# Reconstruct validation candidate forecasts
# ------------------------------------------------------------

# Daily seasonal forecast is the most recent 24 h
# contained in the 168-h input window.
daily_val_norm = np.asarray(
    X_val[:, -24:, :],
    dtype=np.float64,
)

daily_val_original = inverse_scale_array(
    daily_val_norm,
    scaler,
)

daily_val_original = apply_physical_clipping(
    daily_val_original
)


transformer_data = np.load(
    VALIDATION_TRANSFORMER_FILE,
    allow_pickle=False,
)

transformer_val_original = np.asarray(
    transformer_data["predictions_original"],
    dtype=np.float64,
)

transformer_val_targets = np.asarray(
    transformer_data["targets_original"],
    dtype=np.float64,
)

transformer_feature_names = [
    str(x)
    for x in transformer_data["feature_names"]
]


if transformer_feature_names != FEATURE_NAMES:
    raise RuntimeError(
        "Validation Transformer feature order mismatch."
    )

if transformer_val_original.shape != actual_val_original.shape:
    raise RuntimeError(
        "Validation Transformer forecast shape mismatch."
    )

if not np.allclose(
    transformer_val_targets,
    actual_val_original,
    atol=1e-5,
    rtol=0.0,
):
    raise RuntimeError(
        "Validation targets do not match sequence archive."
    )


# ------------------------------------------------------------
# Load validation-selected forecasters
# ------------------------------------------------------------

selection_df = pd.read_csv(
    SELECTION_FILE
)

selected_methods = dict(
    zip(
        selection_df["feature"],
        selection_df["selected_method"],
    )
)

if set(selected_methods.keys()) != set(FEATURE_NAMES):
    raise RuntimeError(
        "Selected-forecaster feature set mismatch."
    )


allowed_methods = {
    "daily_seasonal",
    "transformer",
}

for feature_name in FEATURE_NAMES:
    method = selected_methods[feature_name]

    if method not in allowed_methods:
        raise RuntimeError(
            f"Unsupported selected method for interval "
            f"reconstruction: {feature_name} -> {method}"
        )


# ------------------------------------------------------------
# Construct validation forecast corresponding exactly
# to the selected final forecaster for each target
# ------------------------------------------------------------

hybrid_val_original = np.zeros_like(
    actual_val_original,
    dtype=np.float64,
)

for j, feature_name in enumerate(FEATURE_NAMES):
    method = selected_methods[feature_name]

    if method == "daily_seasonal":
        hybrid_val_original[:, :, j] = (
            daily_val_original[:, :, j]
        )

    elif method == "transformer":
        hybrid_val_original[:, :, j] = (
            transformer_val_original[:, :, j]
        )

    print(
        f"{feature_name:22s} -> {method}"
    )


# ------------------------------------------------------------
# Load locked final TEST forecast
# ------------------------------------------------------------

test_data = np.load(
    FINAL_TEST_FILE,
    allow_pickle=False,
)

test_forecast_original = np.asarray(
    test_data["forecast_original"],
    dtype=np.float64,
)

test_actual_original = np.asarray(
    test_data["actual_original"],
    dtype=np.float64,
)

test_actual_normalized = np.asarray(
    test_data["actual_normalized"],
    dtype=np.float64,
)

test_feature_names = [
    str(x)
    for x in test_data["feature_names"]
]

test_selected_methods = [
    str(x)
    for x in test_data["selected_methods"]
]


if test_feature_names != FEATURE_NAMES:
    raise RuntimeError(
        "Final TEST feature order mismatch."
    )

expected_selected_methods = [
    selected_methods[name]
    for name in FEATURE_NAMES
]

if test_selected_methods != expected_selected_methods:
    raise RuntimeError(
        "Final TEST selected methods do not match "
        "validation selection."
    )

if not np.array_equal(
    y_test_norm,
    test_actual_normalized,
):
    raise RuntimeError(
        "Locked TEST targets do not exactly match y_test."
    )

if test_forecast_original.shape != test_actual_original.shape:
    raise RuntimeError(
        "Locked TEST forecast/actual shape mismatch."
    )


# ------------------------------------------------------------
# Fit interval estimator ONLY on validation residuals
# ------------------------------------------------------------

config = ForecastUncertaintyConfig(
    confidence_level=NOMINAL_COVERAGE,
    minimum_residual_samples=10,
    use_horizon_specific_quantiles=True,
    use_target_specific_quantiles=True,
)

estimator = ForecastUncertaintyEstimator(
    config
)

estimator.fit(
    actual=actual_val_original,
    forecast=hybrid_val_original,
    target_names=FEATURE_NAMES,
)


# ------------------------------------------------------------
# Freeze validation-derived interval parameters and
# apply them to locked TEST point forecasts
# ------------------------------------------------------------

test_interval_result = estimator.transform(
    test_forecast_original
)

lower = np.asarray(
    test_interval_result.lower_bound,
    dtype=np.float64,
)

upper = np.asarray(
    test_interval_result.upper_bound,
    dtype=np.float64,
)

width = np.asarray(
    test_interval_result.interval_width,
    dtype=np.float64,
)


# ------------------------------------------------------------
# Overall PICP / MPIW
# ------------------------------------------------------------

overall_picp = calculate_interval_coverage(
    test_actual_original,
    lower,
    upper,
)

overall_mpiw = float(
    np.mean(width)
)

overall_calibration_error = float(
    overall_picp - NOMINAL_COVERAGE
)


# ------------------------------------------------------------
# Per-target evidence
# ------------------------------------------------------------

target_rows = []

for j, feature_name in enumerate(FEATURE_NAMES):

    target_picp = calculate_interval_coverage(
        test_actual_original[:, :, j],
        lower[:, :, j],
        upper[:, :, j],
    )

    target_mpiw = float(
        np.mean(
            width[:, :, j]
        )
    )

    target_rows.append(
        {
            "feature": feature_name,
            "selected_method": selected_methods[feature_name],
            "nominal_coverage": NOMINAL_COVERAGE,
            "PICP": target_picp,
            "coverage_percentage": 100.0 * target_picp,
            "coverage_error": target_picp - NOMINAL_COVERAGE,
            "MPIW_original_units": target_mpiw,
        }
    )


target_df = pd.DataFrame(
    target_rows
)


# ------------------------------------------------------------
# Per-horizon evidence
# ------------------------------------------------------------

horizon_rows = []

for h in range(24):

    horizon_picp = calculate_interval_coverage(
        test_actual_original[:, h, :],
        lower[:, h, :],
        upper[:, h, :],
    )

    horizon_mpiw = float(
        np.mean(
            width[:, h, :]
        )
    )

    horizon_rows.append(
        {
            "forecast_hour": h + 1,
            "nominal_coverage": NOMINAL_COVERAGE,
            "PICP": horizon_picp,
            "coverage_percentage": 100.0 * horizon_picp,
            "coverage_error": horizon_picp - NOMINAL_COVERAGE,
            "MPIW_mixed_original_units": horizon_mpiw,
        }
    )


horizon_df = pd.DataFrame(
    horizon_rows
)


# ------------------------------------------------------------
# Per-target/per-horizon evidence
# ------------------------------------------------------------

detailed_rows = []

for h in range(24):
    for j, feature_name in enumerate(FEATURE_NAMES):

        actual_slice = test_actual_original[:, h, j]
        lower_slice = lower[:, h, j]
        upper_slice = upper[:, h, j]

        coverage = calculate_interval_coverage(
            actual_slice,
            lower_slice,
            upper_slice,
        )

        mean_width = float(
            np.mean(
                upper_slice - lower_slice
            )
        )

        detailed_rows.append(
            {
                "forecast_hour": h + 1,
                "feature": feature_name,
                "selected_method": selected_methods[feature_name],
                "nominal_coverage": NOMINAL_COVERAGE,
                "PICP": coverage,
                "coverage_percentage": 100.0 * coverage,
                "coverage_error": coverage - NOMINAL_COVERAGE,
                "MPIW_original_units": mean_width,
            }
        )


detailed_df = pd.DataFrame(
    detailed_rows
)


# ------------------------------------------------------------
# Save validation-derived interval parameters
# ------------------------------------------------------------

model = estimator.model

if model is None:
    raise RuntimeError(
        "Interval estimator unexpectedly has no fitted model."
    )


quantile_rows = []

for h in range(24):
    for j, feature_name in enumerate(FEATURE_NAMES):

        quantile_rows.append(
            {
                "forecast_hour": h + 1,
                "feature": feature_name,
                "selected_method": selected_methods[feature_name],
                "lower_residual_quantile": float(
                    model.lower_quantile[h, j]
                ),
                "upper_residual_quantile": float(
                    model.upper_quantile[h, j]
                ),
                "residual_mean": float(
                    model.residual_mean[h, j]
                ),
                "residual_std": float(
                    model.residual_std[h, j]
                ),
            }
        )


quantile_df = pd.DataFrame(
    quantile_rows
)


# ------------------------------------------------------------
# Save numerical evidence
# ------------------------------------------------------------

summary_df = pd.DataFrame(
    [
        {
            "evaluation_split": "locked_test",
            "interval_calibration_split": "validation_only",
            "nominal_coverage": NOMINAL_COVERAGE,
            "PICP": overall_picp,
            "coverage_percentage": 100.0 * overall_picp,
            "coverage_error": overall_calibration_error,
            "absolute_coverage_error": abs(
                overall_calibration_error
            ),
            "MPIW_mixed_original_units": overall_mpiw,
            "validation_samples": int(
                y_val_norm.shape[0]
            ),
            "test_samples": int(
                y_test_norm.shape[0]
            ),
            "forecast_horizon": 24,
            "number_of_targets": 4,
            "test_used_for_interval_calibration": False,
            "test_used_for_forecaster_selection": False,
        }
    ]
)


summary_df.to_csv(
    OUTPUT_DIR
    / "prediction_interval_summary.csv",
    index=False,
)

target_df.to_csv(
    OUTPUT_DIR
    / "prediction_interval_by_target.csv",
    index=False,
)

horizon_df.to_csv(
    OUTPUT_DIR
    / "prediction_interval_by_horizon.csv",
    index=False,
)

detailed_df.to_csv(
    OUTPUT_DIR
    / "prediction_interval_by_target_and_horizon.csv",
    index=False,
)

quantile_df.to_csv(
    OUTPUT_DIR
    / "validation_residual_quantiles.csv",
    index=False,
)


np.savez_compressed(
    OUTPUT_DIR
    / "locked_test_prediction_intervals.npz",
    actual_original=test_actual_original,
    point_forecast_original=test_forecast_original,
    lower_bound_original=lower,
    upper_bound_original=upper,
    interval_width_original=width,
    feature_names=np.array(FEATURE_NAMES),
    selected_methods=np.array(
        expected_selected_methods
    ),
    nominal_coverage=np.array(
        NOMINAL_COVERAGE
    ),
)


metadata = {
    "purpose": (
        "Post-hoc prediction-interval calibration and "
        "locked-test coverage evaluation for Reviewer 5."
    ),
    "interval_method": (
        "Empirical validation residual quantiles."
    ),
    "residual_definition": "actual - forecast",
    "nominal_coverage": NOMINAL_COVERAGE,
    "lower_residual_probability": 0.025,
    "upper_residual_probability": 0.975,
    "horizon_specific_quantiles": True,
    "target_specific_quantiles": True,
    "forecaster_selection_basis": "validation MAE only",
    "interval_calibration_split": "validation only",
    "test_used_for_forecaster_selection": False,
    "test_used_for_interval_calibration": False,
    "selected_methods": selected_methods,
    "validation_shape": list(
        y_val_norm.shape
    ),
    "test_shape": list(
        y_test_norm.shape
    ),
    "important_interpretation": (
        "Prediction-interval coverage is a post-hoc "
        "forecast evaluation metric. It is distinct from "
        "the validation-derived causal confidence Phi used "
        "by FC-HMARL."
    ),
}

with open(
    OUTPUT_DIR
    / "prediction_interval_metadata.json",
    "w",
    encoding="utf-8",
) as f:
    json.dump(
        metadata,
        f,
        indent=4,
    )


# ------------------------------------------------------------
# Console report
# ------------------------------------------------------------

print()
print("=" * 78)
print("LOCKED TEST PREDICTION-INTERVAL RESULTS")
print("=" * 78)

print(
    f"Nominal coverage : {NOMINAL_COVERAGE:.4f}"
)

print(
    f"Overall PICP     : {overall_picp:.6f} "
    f"({100.0 * overall_picp:.3f}%)"
)

print(
    f"Coverage error   : "
    f"{overall_calibration_error:+.6f}"
)

print(
    f"Overall MPIW     : {overall_mpiw:.8f}"
)

print()
print("Per-target results:")
print(
    target_df.to_string(
        index=False
    )
)

print()
print(
    f"Evidence saved to:\n{OUTPUT_DIR}"
)

print()
print(
    "IMPORTANT: PICP/MPIW are post-hoc forecasting "
    "evaluation evidence and are not RL training metrics."
)
