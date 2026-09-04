# ============================================================
# FC-HMARL
# STEP 7G - FINAL HYBRID FORECAST + CONFIDENCE EVALUATION
# ============================================================
#
# Final forecasting strategy selected from Steps 7C-7F:
#
#   PV    -> Daily seasonal baseline
#   Load  -> Daily seasonal baseline
#   EV    -> Original Transformer forecast, physically clipped
#   Price -> Daily seasonal baseline
#
# Confidence equations:
#
#   epsilon_RL = sqrt(e_PV^2 + e_Load^2)
#
#   omega_RL = exp(-epsilon_RL)
#
#   epsilon_EM = sqrt(e_EV^2 + e_Price^2)
#
#   Phi = omega_RL * exp(-epsilon_EM)
#
#       = exp(-(epsilon_RL + epsilon_EM))
#
# IMPORTANT:
# Confidence in this script is REALIZED TEST-SET confidence.
# Actual future values are used only for offline evaluation.
# This confidence must NOT yet be used as causal RL input.
#
# ============================================================

from pathlib import Path

import numpy as np
import pandas as pd

from forecasting.confidence import (
    calculate_renewable_load_uncertainty,
    calculate_renewable_load_confidence,
    calculate_ev_market_uncertainty,
    calculate_global_confidence,
    calculate_confidence_from_errors,
)


# ============================================================
# 1. PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(
    r"D:\Molvi paper review\FC_HMARL"
)


SEQUENCE_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "forecasting"
    / "forecasting_sequences.npz"
)


RESIDUAL_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "forecasting"
    / "forecasting_residual_sequences.npz"
)


SCALER_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "forecasting"
    / "forecasting_scaler.csv"
)


TRANSFORMER_PREDICTION_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "forecasting"
    / "real_forecasting_test_predictions.npz"
)


OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "forecasting"
    / "hybrid"
)


HYBRID_ARCHIVE_FILE = (
    OUTPUT_DIR
    / "final_hybrid_test_predictions.npz"
)


METRICS_FILE = (
    OUTPUT_DIR
    / "final_hybrid_test_metrics.csv"
)


CONFIDENCE_FILE = (
    OUTPUT_DIR
    / "final_hybrid_confidence.csv"
)


CONFIDENCE_SUMMARY_FILE = (
    OUTPUT_DIR
    / "final_hybrid_confidence_summary.csv"
)


HORIZON_FILE = (
    OUTPUT_DIR
    / "final_hybrid_horizon_metrics.csv"
)


# ============================================================
# 2. FEATURE DEFINITIONS
# ============================================================

FEATURE_NAMES = [
    "pv_power_kw",
    "load_kw",
    "ev_power_kw",
    "price_usd_per_kwh",
]


PV_INDEX = 0
LOAD_INDEX = 1
EV_INDEX = 2
PRICE_INDEX = 3

EPS = 1e-12


# ============================================================
# 3. HELPERS
# ============================================================

def section(title):

    print()

    print("=" * 80)

    print(title)

    print("=" * 80)


def mae(actual, forecast):

    return float(
        np.mean(
            np.abs(
                actual - forecast
            )
        )
    )


def rmse(actual, forecast):

    return float(
        np.sqrt(
            np.mean(
                (
                    actual - forecast
                ) ** 2
            )
        )
    )


def r2(actual, forecast):

    actual = np.asarray(
        actual,
        dtype=np.float64,
    )

    forecast = np.asarray(
        forecast,
        dtype=np.float64,
    )


    ss_res = np.sum(
        (
            actual - forecast
        ) ** 2
    )

    ss_tot = np.sum(
        (
            actual - np.mean(actual)
        ) ** 2
    )


    if ss_tot <= EPS:

        return np.nan


    return float(
        1.0
        - ss_res
        / ss_tot
    )


def nmae(actual, forecast):

    denominator = np.mean(
        np.abs(actual)
    )


    if denominator <= EPS:

        return np.nan


    return float(
        100.0
        * mae(
            actual,
            forecast
        )
        / denominator
    )


def nrmse(actual, forecast):

    denominator = np.mean(
        np.abs(actual)
    )


    if denominator <= EPS:

        return np.nan


    return float(
        100.0
        * rmse(
            actual,
            forecast
        )
        / denominator
    )


def print_npz_keys(
    name,
    archive,
):

    print(
        f"{name} keys:"
    )

    for key in archive.files:

        print(
            f"  {key}"
        )


# ============================================================
# 4. CHECK INPUT FILES
# ============================================================

section(
    "FC-HMARL - STEP 7G FINAL HYBRID FORECAST"
)


required_files = [

    SEQUENCE_FILE,

    RESIDUAL_FILE,

    SCALER_FILE,

    TRANSFORMER_PREDICTION_FILE,
]


for file_path in required_files:

    if not file_path.exists():

        raise FileNotFoundError(
            f"Required file not found:\n"
            f"{file_path}"
        )


OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# 5. LOAD FORECAST DATA
# ============================================================

sequences = np.load(
    SEQUENCE_FILE,
    allow_pickle=True,
)


residual_data = np.load(
    RESIDUAL_FILE,
    allow_pickle=True,
)


transformer_data = np.load(
    TRANSFORMER_PREDICTION_FILE,
    allow_pickle=True,
)


print_npz_keys(
    "Sequence archive",
    sequences,
)

print()

print_npz_keys(
    "Residual archive",
    residual_data,
)

print()

print_npz_keys(
    "Original Transformer prediction archive",
    transformer_data,
)


# ============================================================
# 6. LOAD NORMALIZED TEST TARGETS
# ============================================================

y_test_norm = np.asarray(
    sequences["y_test"],
    dtype=np.float64,
)


X_test_norm = np.asarray(
    sequences["X_test"],
    dtype=np.float64,
)


print()

print(
    f"X_test normalized : "
    f"{X_test_norm.shape}"
)

print(
    f"y_test normalized : "
    f"{y_test_norm.shape}"
)


if y_test_norm.shape[1:] != (
    24,
    4,
):

    raise RuntimeError(
        f"Unexpected test target shape: "
        f"{y_test_norm.shape}"
    )


# ============================================================
# 7. SEASONAL BASELINE
# ============================================================
#
# Step 7D selected:
#
# PV    -> daily
# Load  -> daily
# EV    -> weekly
# Price -> daily
#
# For the final hybrid forecast we need:
#
# PV    daily
# Load  daily
# Price daily
#
# EV seasonal baseline is retained only for comparison.
#
# ============================================================

seasonal_test_norm = np.asarray(
    residual_data[
        "baseline_test"
    ],
    dtype=np.float64,
)


if seasonal_test_norm.shape != y_test_norm.shape:

    raise RuntimeError(
        "Seasonal baseline shape mismatch."
    )


# ============================================================
# 8. LOAD ORIGINAL TRANSFORMER FORECAST
# ============================================================
#
# We attempt several possible archive names so the script
# remains compatible with the Step 7A output.
#
# Preference:
#
#   predictions_original
#   predictions_original_clipped
#   predictions
#
# ============================================================

possible_prediction_keys = [

    "predictions_original",

    "predictions_original_clipped",

    "predictions",

    "test_predictions",

    "forecast_original",
]


transformer_original = None

selected_prediction_key = None


for key in possible_prediction_keys:

    if key in transformer_data.files:

        candidate = np.asarray(
            transformer_data[key],
            dtype=np.float64,
        )


        if candidate.shape == y_test_norm.shape:

            transformer_original = (
                candidate
            )

            selected_prediction_key = (
                key
            )

            break


if transformer_original is None:

    raise RuntimeError(

        "\nCould not automatically identify the "
        "original Transformer prediction array.\n\n"

        "Look at the archive keys printed above.\n"

        "The prediction array must have shape "
        "(1291, 24, 4).\n"
    )


print()

print(
    f"Transformer prediction key selected: "
    f"{selected_prediction_key}"
)

print(
    f"Transformer prediction shape: "
    f"{transformer_original.shape}"
)


# ============================================================
# 9. LOAD SCALER
# ============================================================

scaler = pd.read_csv(
    SCALER_FILE
)


required_columns = [

    "target",

    "train_min",

    "train_max",
]


for column in required_columns:

    if column not in scaler.columns:

        raise RuntimeError(
            f"Scaler missing required "
            f"column: {column}"
        )


scaler = scaler.set_index(
    "target"
)


def inverse_scale_feature(
    values,
    feature_name,
):

    minimum = float(
        scaler.loc[
            feature_name,
            "train_min"
        ]
    )

    maximum = float(
        scaler.loc[
            feature_name,
            "train_max"
        ]
    )


    return (
        values
        * (
            maximum - minimum
        )
        + minimum
    )


def normalize_feature(
    values,
    feature_name,
):

    minimum = float(
        scaler.loc[
            feature_name,
            "train_min"
        ]
    )

    maximum = float(
        scaler.loc[
            feature_name,
            "train_max"
        ]
    )


    scale = (
        maximum - minimum
    )


    if abs(scale) <= EPS:

        raise RuntimeError(
            f"Invalid scaler range "
            f"for {feature_name}"
        )


    return (
        values - minimum
    ) / scale


# ============================================================
# 10. ORIGINAL-SCALE ACTUAL TARGET
# ============================================================

actual_original = np.zeros_like(
    y_test_norm,
    dtype=np.float64,
)


seasonal_original = np.zeros_like(
    seasonal_test_norm,
    dtype=np.float64,
)


for feature_index, feature_name in enumerate(
    FEATURE_NAMES
):

    actual_original[
        :,
        :,
        feature_index
    ] = inverse_scale_feature(

        y_test_norm[
            :,
            :,
            feature_index
        ],

        feature_name,
    )


    seasonal_original[
        :,
        :,
        feature_index
    ] = inverse_scale_feature(

        seasonal_test_norm[
            :,
            :,
            feature_index
        ],

        feature_name,
    )


# ============================================================
# 11. PHYSICAL CLIPPING OF TRANSFORMER
# ============================================================
#
# The original Transformer performed best for EV.
#
# EV charging demand cannot be negative in the present
# forecasting representation.
#
# ============================================================

transformer_original_clipped = (
    transformer_original.copy()
)


transformer_original_clipped[
    :,
    :,
    PV_INDEX
] = np.clip(

    transformer_original_clipped[
        :,
        :,
        PV_INDEX
    ],

    0.0,

    None,
)


transformer_original_clipped[
    :,
    :,
    LOAD_INDEX
] = np.clip(

    transformer_original_clipped[
        :,
        :,
        LOAD_INDEX
    ],

    0.0,

    None,
)


transformer_original_clipped[
    :,
    :,
    EV_INDEX
] = np.clip(

    transformer_original_clipped[
        :,
        :,
        EV_INDEX
    ],

    0.0,

    None,
)


# ============================================================
# 12. BUILD FINAL HYBRID IN PHYSICAL UNITS
# ============================================================

section(
    "BUILDING FINAL HYBRID FORECAST"
)


hybrid_original = np.zeros_like(
    actual_original,
    dtype=np.float64,
)


# ------------------------------------------------------------
# PV -> daily seasonal
# ------------------------------------------------------------

hybrid_original[
    :,
    :,
    PV_INDEX
] = seasonal_original[
    :,
    :,
    PV_INDEX
]


# ------------------------------------------------------------
# Load -> daily seasonal
# ------------------------------------------------------------

hybrid_original[
    :,
    :,
    LOAD_INDEX
] = seasonal_original[
    :,
    :,
    LOAD_INDEX
]


# ------------------------------------------------------------
# EV -> clipped original Transformer
# ------------------------------------------------------------

hybrid_original[
    :,
    :,
    EV_INDEX
] = transformer_original_clipped[
    :,
    :,
    EV_INDEX
]


# ------------------------------------------------------------
# Price -> daily seasonal
# ------------------------------------------------------------

hybrid_original[
    :,
    :,
    PRICE_INDEX
] = seasonal_original[
    :,
    :,
    PRICE_INDEX
]


print(
    "PV    -> Daily seasonal"
)

print(
    "Load  -> Daily seasonal"
)

print(
    "EV    -> Original clipped Transformer"
)

print(
    "Price -> Daily seasonal"
)


# ============================================================
# 13. CONVERT HYBRID FORECAST TO NORMALIZED SPACE
# ============================================================
#
# Confidence equations combine different targets.
#
# Therefore the forecast errors are calculated in normalized
# dimensionless space rather than mixing:
#
#   kW + USD/kWh
#
# directly.
#
# ============================================================

hybrid_norm = np.zeros_like(
    hybrid_original,
    dtype=np.float64,
)


for feature_index, feature_name in enumerate(
    FEATURE_NAMES
):

    hybrid_norm[
        :,
        :,
        feature_index
    ] = normalize_feature(

        hybrid_original[
            :,
            :,
            feature_index
        ],

        feature_name,
    )


# ============================================================
# 14. ERROR ARRAYS
# ============================================================

section(
    "CALCULATING NORMALIZED FORECAST ERRORS"
)


absolute_error_norm = np.abs(
    y_test_norm
    - hybrid_norm
)


pv_error = absolute_error_norm[
    :,
    :,
    PV_INDEX
]


load_error = absolute_error_norm[
    :,
    :,
    LOAD_INDEX
]


ev_error = absolute_error_norm[
    :,
    :,
    EV_INDEX
]


price_error = absolute_error_norm[
    :,
    :,
    PRICE_INDEX
]


print(
    f"PV error shape    : "
    f"{pv_error.shape}"
)

print(
    f"Load error shape  : "
    f"{load_error.shape}"
)

print(
    f"EV error shape    : "
    f"{ev_error.shape}"
)

print(
    f"Price error shape : "
    f"{price_error.shape}"
)


# ============================================================
# 15. MANUSCRIPT CONFIDENCE COMPONENTS
# ============================================================

section(
    "CALCULATING FC-HMARL CONFIDENCE"
)


epsilon_RL = (
    calculate_renewable_load_uncertainty(
        pv_error,
        load_error,
    )
)


omega_RL = (
    calculate_renewable_load_confidence(
        epsilon_RL,
        error_scale=1.0,
    )
)


epsilon_EM = (
    calculate_ev_market_uncertainty(
        ev_error,
        price_error,
    )
)


Phi = (
    calculate_global_confidence(

        omega_RL,

        epsilon_EM,

        error_scale=1.0,

        minimum_confidence=0.0,

        maximum_confidence=1.0,
    )
)


# ============================================================
# 16. DIRECT FORMULA VERIFICATION
# ============================================================

epsilon_RL_direct = np.sqrt(
    pv_error ** 2
    + load_error ** 2
)


omega_RL_direct = np.exp(
    -epsilon_RL_direct
)


epsilon_EM_direct = np.sqrt(
    ev_error ** 2
    + price_error ** 2
)


Phi_direct = np.exp(
    -(
        epsilon_RL_direct
        + epsilon_EM_direct
    )
)


print(
    f"epsilon_RL module/direct max difference : "
    f"{np.max(np.abs(epsilon_RL - epsilon_RL_direct)):.12e}"
)

print(
    f"omega_RL module/direct max difference   : "
    f"{np.max(np.abs(omega_RL - omega_RL_direct)):.12e}"
)

print(
    f"epsilon_EM module/direct max difference : "
    f"{np.max(np.abs(epsilon_EM - epsilon_EM_direct)):.12e}"
)

print(
    f"Phi module/direct max difference        : "
    f"{np.max(np.abs(Phi - Phi_direct)):.12e}"
)


if not np.allclose(
    epsilon_RL,
    epsilon_RL_direct,
    atol=1e-10,
):

    raise RuntimeError(
        "Renewable-load uncertainty formula mismatch."
    )


if not np.allclose(
    omega_RL,
    omega_RL_direct,
    atol=1e-10,
):

    raise RuntimeError(
        "Renewable-load confidence formula mismatch."
    )


if not np.allclose(
    epsilon_EM,
    epsilon_EM_direct,
    atol=1e-10,
):

    raise RuntimeError(
        "EV-market uncertainty formula mismatch."
    )


if not np.allclose(
    Phi,
    Phi_direct,
    atol=1e-10,
):

    raise RuntimeError(
        "Global confidence formula mismatch."
    )


print()

print(
    "[OK] Existing forecasting.confidence module "
    "matches the required equations."
)


# ============================================================
# 17. SECOND API CROSS-CHECK
# ============================================================
#
# calculate_confidence_from_errors should produce the
# same global Phi.
#
# ============================================================

Phi_from_combined_function = (
    calculate_confidence_from_errors(

        pv_error,

        load_error,

        ev_error,

        price_error,

        error_scale=1.0,
    )
)


combined_difference = np.max(
    np.abs(
        Phi
        - Phi_from_combined_function
    )
)


print()

print(
    f"Combined-function Phi max difference: "
    f"{combined_difference:.12e}"
)


if not np.allclose(
    Phi,
    Phi_from_combined_function,
    atol=1e-10,
):

    raise RuntimeError(
        "calculate_confidence_from_errors "
        "does not match component calculation."
    )


# ============================================================
# 18. CONFIDENCE STATISTICS
# ============================================================

section(
    "REALIZED CONFIDENCE STATISTICS"
)


confidence_statistics = {


    "epsilon_RL": {

        "mean":
            float(
                np.mean(epsilon_RL)
            ),

        "std":
            float(
                np.std(epsilon_RL)
            ),

        "min":
            float(
                np.min(epsilon_RL)
            ),

        "max":
            float(
                np.max(epsilon_RL)
            ),
    },


    "omega_RL": {

        "mean":
            float(
                np.mean(omega_RL)
            ),

        "std":
            float(
                np.std(omega_RL)
            ),

        "min":
            float(
                np.min(omega_RL)
            ),

        "max":
            float(
                np.max(omega_RL)
            ),
    },


    "epsilon_EM": {

        "mean":
            float(
                np.mean(epsilon_EM)
            ),

        "std":
            float(
                np.std(epsilon_EM)
            ),

        "min":
            float(
                np.min(epsilon_EM)
            ),

        "max":
            float(
                np.max(epsilon_EM)
            ),
    },


    "Phi": {

        "mean":
            float(
                np.mean(Phi)
            ),

        "std":
            float(
                np.std(Phi)
            ),

        "min":
            float(
                np.min(Phi)
            ),

        "max":
            float(
                np.max(Phi)
            ),
    },
}


for name, statistics in (
    confidence_statistics.items()
):

    print()

    print(
        name
    )

    print(
        f"  Mean : "
        f"{statistics['mean']:.8f}"
    )

    print(
        f"  Std  : "
        f"{statistics['std']:.8f}"
    )

    print(
        f"  Min  : "
        f"{statistics['min']:.8f}"
    )

    print(
        f"  Max  : "
        f"{statistics['max']:.8f}"
    )


# ============================================================
# 19. FINAL HYBRID FORECAST METRICS
# ============================================================

section(
    "FINAL HYBRID FORECAST TEST METRICS"
)


metric_rows = []


for feature_index, feature_name in enumerate(
    FEATURE_NAMES
):

    actual = actual_original[
        :,
        :,
        feature_index
    ].reshape(-1)


    forecast = hybrid_original[
        :,
        :,
        feature_index
    ].reshape(-1)


    feature_mae = mae(
        actual,
        forecast
    )

    feature_rmse = rmse(
        actual,
        forecast
    )

    feature_r2 = r2(
        actual,
        forecast
    )

    feature_nmae = nmae(
        actual,
        forecast
    )

    feature_nrmse = nrmse(
        actual,
        forecast
    )


    metric_rows.append(
        {

            "feature":
                feature_name,

            "MAE":
                feature_mae,

            "RMSE":
                feature_rmse,

            "R2":
                feature_r2,

            "NMAE_percent":
                feature_nmae,

            "NRMSE_percent":
                feature_nrmse,
        }
    )


    print()

    print(
        feature_name
    )

    print(
        f"  MAE   : "
        f"{feature_mae:.8f}"
    )

    print(
        f"  RMSE  : "
        f"{feature_rmse:.8f}"
    )

    print(
        f"  R2    : "
        f"{feature_r2:.6f}"
    )

    print(
        f"  NMAE  : "
        f"{feature_nmae:.4f}%"
    )

    print(
        f"  NRMSE : "
        f"{feature_nrmse:.4f}%"
    )


metrics_df = pd.DataFrame(
    metric_rows
)


# ============================================================
# 20. HORIZON-WISE METRICS
# ============================================================

section(
    "24-HOUR HORIZON-WISE HYBRID METRICS"
)


horizon_rows = []


for horizon_index in range(
    24
):

    forecast_hour = (
        horizon_index + 1
    )


    for feature_index, feature_name in enumerate(
        FEATURE_NAMES
    ):

        actual = actual_original[
            :,
            horizon_index,
            feature_index
        ]


        forecast = hybrid_original[
            :,
            horizon_index,
            feature_index
        ]


        horizon_rows.append(
            {

                "forecast_hour":
                    forecast_hour,

                "feature":
                    feature_name,

                "MAE":
                    mae(
                        actual,
                        forecast
                    ),

                "RMSE":
                    rmse(
                        actual,
                        forecast
                    ),

                "R2":
                    r2(
                        actual,
                        forecast
                    ),
            }
        )


horizon_df = pd.DataFrame(
    horizon_rows
)


# ============================================================
# 21. SAVE FULL CONFIDENCE TABLE
# ============================================================

number_of_samples = (
    y_test_norm.shape[0]
)


confidence_rows = []


for sample_index in range(
    number_of_samples
):

    for horizon_index in range(
        24
    ):

        confidence_rows.append(
            {

                "sample_index":
                    sample_index,

                "forecast_hour":
                    horizon_index + 1,

                "pv_error_normalized":
                    float(
                        pv_error[
                            sample_index,
                            horizon_index
                        ]
                    ),

                "load_error_normalized":
                    float(
                        load_error[
                            sample_index,
                            horizon_index
                        ]
                    ),

                "ev_error_normalized":
                    float(
                        ev_error[
                            sample_index,
                            horizon_index
                        ]
                    ),

                "price_error_normalized":
                    float(
                        price_error[
                            sample_index,
                            horizon_index
                        ]
                    ),

                "epsilon_RL":
                    float(
                        epsilon_RL[
                            sample_index,
                            horizon_index
                        ]
                    ),

                "omega_RL":
                    float(
                        omega_RL[
                            sample_index,
                            horizon_index
                        ]
                    ),

                "epsilon_EM":
                    float(
                        epsilon_EM[
                            sample_index,
                            horizon_index
                        ]
                    ),

                "Phi":
                    float(
                        Phi[
                            sample_index,
                            horizon_index
                        ]
                    ),
            }
        )


confidence_df = pd.DataFrame(
    confidence_rows
)


# ============================================================
# 22. CONFIDENCE SUMMARY TABLE
# ============================================================

confidence_summary_rows = []


for variable_name, values in [

    (
        "epsilon_RL",
        epsilon_RL,
    ),

    (
        "omega_RL",
        omega_RL,
    ),

    (
        "epsilon_EM",
        epsilon_EM,
    ),

    (
        "Phi",
        Phi,
    ),

]:

    flat_values = (
        np.asarray(values)
        .reshape(-1)
    )


    confidence_summary_rows.append(
        {

            "variable":
                variable_name,

            "mean":
                float(
                    np.mean(
                        flat_values
                    )
                ),

            "std":
                float(
                    np.std(
                        flat_values
                    )
                ),

            "minimum":
                float(
                    np.min(
                        flat_values
                    )
                ),

            "q05":
                float(
                    np.quantile(
                        flat_values,
                        0.05,
                    )
                ),

            "q25":
                float(
                    np.quantile(
                        flat_values,
                        0.25,
                    )
                ),

            "median":
                float(
                    np.quantile(
                        flat_values,
                        0.50,
                    )
                ),

            "q75":
                float(
                    np.quantile(
                        flat_values,
                        0.75,
                    )
                ),

            "q95":
                float(
                    np.quantile(
                        flat_values,
                        0.95,
                    )
                ),

            "maximum":
                float(
                    np.max(
                        flat_values
                    )
                ),
        }
    )


confidence_summary_df = pd.DataFrame(
    confidence_summary_rows
)


# ============================================================
# 23. SAVE RESULTS
# ============================================================

section(
    "SAVING FINAL HYBRID OUTPUTS"
)


metrics_df.to_csv(
    METRICS_FILE,
    index=False,
)


confidence_df.to_csv(
    CONFIDENCE_FILE,
    index=False,
)


confidence_summary_df.to_csv(
    CONFIDENCE_SUMMARY_FILE,
    index=False,
)


horizon_df.to_csv(
    HORIZON_FILE,
    index=False,
)


np.savez_compressed(

    HYBRID_ARCHIVE_FILE,


    # --------------------------------
    # Final forecast in physical units
    # --------------------------------

    hybrid_forecast_original=
        hybrid_original,


    # --------------------------------
    # Ground truth
    # --------------------------------

    actual_original=
        actual_original,


    # --------------------------------
    # Normalized final forecast
    # --------------------------------

    hybrid_forecast_normalized=
        hybrid_norm,


    # --------------------------------
    # Normalized actual
    # --------------------------------

    actual_normalized=
        y_test_norm,


    # --------------------------------
    # Forecast error
    # --------------------------------

    absolute_error_normalized=
        absolute_error_norm,


    # --------------------------------
    # Confidence components
    # --------------------------------

    epsilon_RL=
        epsilon_RL,

    omega_RL=
        omega_RL,

    epsilon_EM=
        epsilon_EM,

    Phi=
        Phi,


    # --------------------------------
    # Component forecasts
    # --------------------------------

    seasonal_baseline_original=
        seasonal_original,

    transformer_forecast_original=
        transformer_original_clipped,


    # --------------------------------
    # Metadata
    # --------------------------------

    feature_names=
        np.array(
            FEATURE_NAMES
        ),

    forecast_source=
        np.array(
            [
                "daily_seasonal",
                "daily_seasonal",
                "original_transformer_clipped",
                "daily_seasonal",
            ]
        ),
)


print(
    f"Hybrid archive:\n"
    f"{HYBRID_ARCHIVE_FILE}"
)

print()

print(
    f"Hybrid metrics:\n"
    f"{METRICS_FILE}"
)

print()

print(
    f"Confidence values:\n"
    f"{CONFIDENCE_FILE}"
)

print()

print(
    f"Confidence summary:\n"
    f"{CONFIDENCE_SUMMARY_FILE}"
)

print()

print(
    f"Horizon metrics:\n"
    f"{HORIZON_FILE}"
)


# ============================================================
# 24. FINAL CHECKS
# ============================================================

section(
    "FINAL VALIDATION"
)


print(
    f"Hybrid forecast shape : "
    f"{hybrid_original.shape}"
)

print(
    f"Phi shape             : "
    f"{Phi.shape}"
)


print()


print(
    "Phi range:"
)

print(
    f"  minimum = "
    f"{np.min(Phi):.8f}"
)

print(
    f"  mean    = "
    f"{np.mean(Phi):.8f}"
)

print(
    f"  maximum = "
    f"{np.max(Phi):.8f}"
)


if np.any(
    Phi < 0.0
):

    raise RuntimeError(
        "Phi contains values below zero."
    )


if np.any(
    Phi > 1.0
):

    raise RuntimeError(
        "Phi contains values above one."
    )


print()

print(
    "[OK] Global confidence is bounded "
    "between 0 and 1."
)


# ============================================================
# 25. COMPLETE
# ============================================================

section(
    "STEP 7G COMPLETE"
)


print(
    "Final hybrid forecast generated successfully."
)

print()

print(
    "Forecast source:"
)

print(
    "  PV    -> daily seasonal"
)

print(
    "  Load  -> daily seasonal"
)

print(
    "  EV    -> original clipped Transformer"
)

print(
    "  Price -> daily seasonal"
)

print()

print(
    "Manuscript confidence equations were "
    "verified against forecasting.confidence."
)

print()

print(
    "IMPORTANT:"
)

print(
    "The present Phi is REALIZED/OFFLINE confidence."
)

print(
    "It uses known test actual values and must not "
    "yet be used as future information in RL."
)

print()

print(
    "STEP 7G COMPLETE."
)