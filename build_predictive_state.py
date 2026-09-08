from pathlib import Path
import numpy as np
import pandas as pd
from forecasting.confidence import (
    build_confidence_aware_predictive_state,
)
# ============================================================
# 1. PATHS
# ============================================================
PROJECT_ROOT = Path(
    r"D:\Molvi paper review\FC_HMARL"
)


INPUT_FILE = (
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
    / "predictive_state"
)


OUTPUT_ARCHIVE = (
    OUTPUT_DIR
    / "confidence_aware_predictive_state.npz"
)


SUMMARY_FILE = (
    OUTPUT_DIR
    / "predictive_state_summary.csv"
)


SAMPLE_FILE = (
    OUTPUT_DIR
    / "predictive_state_sample_001.csv"
)


# ============================================================
# 2. CONSTANTS
# ============================================================

FEATURE_NAMES = [
    "pv_power_kw",
    "load_kw",
    "ev_power_kw",
    "price_usd_per_kwh",
]


FORECAST_HORIZON = 24
NUMBER_OF_FEATURES = 4
FLATTENED_DIMENSION = 96


# ============================================================
# 3. HELPER
# ============================================================

def section(title):

    print()
    print("=" * 80)
    print(title)
    print("=" * 80)


# ============================================================
# 4. LOAD FINAL LEAKAGE-FREE FORECAST
# ============================================================

section(
    "FC-HMARL STEP 7I - PREDICTIVE STATE"
)


if not INPUT_FILE.exists():

    raise FileNotFoundError(
        f"Input file not found:\n{INPUT_FILE}"
    )


OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


data = np.load(
    INPUT_FILE,
    allow_pickle=True,
)


print(
    "Archive keys:"
)

for key in data.files:

    print(
        f"  {key}"
    )


forecast_original = np.asarray(
    data["forecast_original"],
    dtype=np.float64,
)


forecast_normalized = np.asarray(
    data["forecast_normalized"],
    dtype=np.float64,
)


Phi_causal = np.asarray(
    data["causal_confidence_24h"],
    dtype=np.float64,
)


selected_methods = np.asarray(
    data["selected_methods"]
)


print()

print(
    f"Forecast original shape   : "
    f"{forecast_original.shape}"
)

print(
    f"Forecast normalized shape : "
    f"{forecast_normalized.shape}"
)

print(
    f"Causal confidence shape   : "
    f"{Phi_causal.shape}"
)
# ============================================================
# 5. INPUT VALIDATION
# ============================================================
if forecast_normalized.ndim != 3:

    raise RuntimeError(
        "Forecast must have shape "
        "[samples, horizon, features]."
    )

number_of_samples = (
    forecast_normalized.shape[0]
)

if forecast_normalized.shape[1:] != (
    FORECAST_HORIZON,
    NUMBER_OF_FEATURES,
):

    raise RuntimeError(
        f"Unexpected forecast dimensions: "
        f"{forecast_normalized.shape}"
    )


if Phi_causal.shape != (
    FORECAST_HORIZON,
):

    raise RuntimeError(
        "Causal confidence must contain "
        "exactly 24 values."
    )

if np.any(
    Phi_causal < 0.0
) or np.any(
    Phi_causal > 1.0
):

    raise RuntimeError(
        "Confidence must remain inside [0,1]."
    )

section(
    "BUILDING S_pred = Phi * Z_hat"
)

confidence_matrix = np.broadcast_to(

    Phi_causal[
        None,
        :
    ],

    (
        number_of_samples,
        FORECAST_HORIZON,
    )

).copy()


print(
    f"Forecast batch shape    : "
    f"{forecast_normalized.shape}"
)

print(
    f"Confidence batch shape  : "
    f"{confidence_matrix.shape}"
)

predictive_state_matrix = (
    build_confidence_aware_predictive_state(

        forecast_normalized,

        confidence_matrix,
    )
)

predictive_state_matrix = np.asarray(
    predictive_state_matrix,
    dtype=np.float64,
)

print(
    f"Predictive-state matrix shape: "
    f"{predictive_state_matrix.shape}"
)

expected_shape = (

    number_of_samples,

    FORECAST_HORIZON,

    NUMBER_OF_FEATURES,
)


if predictive_state_matrix.shape != (
    expected_shape
):

    raise RuntimeError(

        "build_confidence_aware_predictive_state "
        "returned unexpected shape.\n"

        f"Expected: {expected_shape}\n"

        f"Received: "
        f"{predictive_state_matrix.shape}"
    )


print()

print(
    "[OK] Batch confidence-aware predictive "
    "state constructed successfully."
)

section(
    "VERIFYING MANUSCRIPT EQUATION"
)


direct_predictive_state = (

    forecast_normalized
    * Phi_causal[
        None,
        :,
        None
    ]
)


maximum_difference = float(

    np.max(

        np.abs(

            predictive_state_matrix
            - direct_predictive_state
        )
    )
)


print(
    f"Module/direct maximum difference: "
    f"{maximum_difference:.12e}"
)


if not np.allclose(
    predictive_state_matrix,
    direct_predictive_state,
    atol=1e-10,
):

    raise RuntimeError(
        "Predictive-state module does not match "
        "S_pred = Phi * Z_hat."
    )


print()

print(
    "[OK] forecasting.confidence implementation "
    "matches S_pred = Phi * Z_hat."
)



section(
    "FLATTENING 24 x 4 -> 96"
)


predictive_state_flat = (
    predictive_state_matrix.reshape(
        number_of_samples,
        -1,
    )
)


print(
    f"Flattened shape: "
    f"{predictive_state_flat.shape}"
)


if predictive_state_flat.shape[1] != (
    FLATTENED_DIMENSION
):

    raise RuntimeError(
        "Predictive-state dimension must be 96."
    )


print()

print(
    "[OK] Each forecasting sample produces "
    "a 96-dimensional predictive state."
)


# ============================================================
# 9. MANUAL FIRST-SAMPLE CHECK
# ============================================================

section(
    "FIRST-SAMPLE CHECK"
)


for horizon_index in range(
    24
):

    print(

        f"h={horizon_index + 1:02d} | "

        f"Phi={Phi_causal[horizon_index]:.6f} | "

        f"PV={predictive_state_matrix[0,horizon_index,0]:.6f} | "

        f"Load={predictive_state_matrix[0,horizon_index,1]:.6f} | "

        f"EV={predictive_state_matrix[0,horizon_index,2]:.6f} | "

        f"Price={predictive_state_matrix[0,horizon_index,3]:.6f}"
    )


# ============================================================
# 10. SUMMARY STATISTICS
# ============================================================

summary_rows = []


for feature_index, feature_name in enumerate(
    FEATURE_NAMES
):

    forecast_values = forecast_normalized[
        :,
        :,
        feature_index
    ]


    predictive_values = predictive_state_matrix[
        :,
        :,
        feature_index
    ]


    summary_rows.append(
        {

            "feature":
                feature_name,

            "forecast_mean_normalized":
                float(
                    np.mean(
                        forecast_values
                    )
                ),

            "forecast_std_normalized":
                float(
                    np.std(
                        forecast_values
                    )
                ),

            "predictive_state_mean":
                float(
                    np.mean(
                        predictive_values
                    )
                ),

            "predictive_state_std":
                float(
                    np.std(
                        predictive_values
                    )
                ),

            "predictive_state_min":
                float(
                    np.min(
                        predictive_values
                    )
                ),

            "predictive_state_max":
                float(
                    np.max(
                        predictive_values
                    )
                ),
        }
    )


summary_df = pd.DataFrame(
    summary_rows
)


summary_df.to_csv(
    SUMMARY_FILE,
    index=False,
)


# ============================================================
# 11. SAVE FIRST SAMPLE FOR INSPECTION
# ============================================================

sample_df = pd.DataFrame(
    {

        "forecast_hour":
            np.arange(
                1,
                25
            ),

        "Phi_causal":
            Phi_causal,

        "forecast_pv_normalized":
            forecast_normalized[
                0,
                :,
                0
            ],

        "forecast_load_normalized":
            forecast_normalized[
                0,
                :,
                1
            ],

        "forecast_ev_normalized":
            forecast_normalized[
                0,
                :,
                2
            ],

        "forecast_price_normalized":
            forecast_normalized[
                0,
                :,
                3
            ],

        "S_pred_pv":
            predictive_state_matrix[
                0,
                :,
                0
            ],

        "S_pred_load":
            predictive_state_matrix[
                0,
                :,
                1
            ],

        "S_pred_ev":
            predictive_state_matrix[
                0,
                :,
                2
            ],

        "S_pred_price":
            predictive_state_matrix[
                0,
                :,
                3
            ],
    }
)


sample_df.to_csv(
    SAMPLE_FILE,
    index=False,
)


# ============================================================
# 12. SAVE ARCHIVE
# ============================================================

np.savez_compressed(

    OUTPUT_ARCHIVE,


    # ----------------------------------------
    # Confidence-aware predictive state
    # ----------------------------------------

    predictive_state_matrix=
        predictive_state_matrix,

    predictive_state_flat=
        predictive_state_flat,


    # ----------------------------------------
    # Forecast information
    # ----------------------------------------

    forecast_normalized=
        forecast_normalized,

    forecast_original=
        forecast_original,


    # ----------------------------------------
    # Confidence
    # ----------------------------------------

    causal_confidence_24h=
        Phi_causal,


    # ----------------------------------------
    # Metadata
    # ----------------------------------------

    feature_names=
        np.array(
            FEATURE_NAMES
        ),

    selected_methods=
        selected_methods,

    forecast_horizon=
        np.array(
            FORECAST_HORIZON
        ),

    number_of_features=
        np.array(
            NUMBER_OF_FEATURES
        ),

    flattened_dimension=
        np.array(
            FLATTENED_DIMENSION
        ),
)


# ============================================================
# 13. FINAL VALIDATION
# ============================================================

section(
    "STEP 7I FINAL VALIDATION"
)


print(
    f"Number of samples       : "
    f"{number_of_samples}"
)

print(
    f"Forecast horizon        : "
    f"{FORECAST_HORIZON}"
)

print(
    f"Forecast variables      : "
    f"{NUMBER_OF_FEATURES}"
)

print(
    f"Predictive matrix       : "
    f"{predictive_state_matrix.shape}"
)

print(
    f"Flattened S_pred        : "
    f"{predictive_state_flat.shape}"
)

print(
    f"Expected RL dimension   : "
    f"{FLATTENED_DIMENSION}"
)


print()

print(
    "Selected forecast methods:"
)


for feature_index, feature_name in enumerate(
    FEATURE_NAMES
):

    print(

        f"  {feature_name:22s} -> "
        f"{selected_methods[feature_index]}"
    )


print()

print(
    f"Mean causal confidence  : "
    f"{np.mean(Phi_causal):.8f}"
)


print(
    f"Min causal confidence   : "
    f"{np.min(Phi_causal):.8f}"
)


print(
    f"Max causal confidence   : "
    f"{np.max(Phi_causal):.8f}"
)


# ============================================================
# 14. OUTPUT FILES
# ============================================================

section(
    "STEP 7I COMPLETE"
)


print(
    f"Predictive-state archive:\n"
    f"{OUTPUT_ARCHIVE}"
)

print()

print(
    f"Summary:\n"
    f"{SUMMARY_FILE}"
)

print()

print(
    f"First-sample inspection:\n"
    f"{SAMPLE_FILE}"
)

print()

print(
    "[OK] Causal confidence is incorporated "
    "into the forecast."
)

print(
    "[OK] S_pred has 24 x 4 structure."
)

print(
    "[OK] Flattened S_pred dimension is 96."
)

print(
    "[OK] Predictive state is ready for "
    "FC-HMARL state construction."
)

print()

print(
    "STEP 7I COMPLETE."
)
