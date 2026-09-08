from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch

from forecasting.model import (
    ForecastModelConfig,
    MultiHorizonTransformerForecaster,
)
from forecasting.confidence import (
    calculate_renewable_load_uncertainty,
    calculate_renewable_load_confidence,
    calculate_ev_market_uncertainty,
    calculate_global_confidence,
)
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
SCALER_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "forecasting"
    / "forecasting_scaler.csv"
)

BEST_MODEL_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "checkpoints"
    / "best_real_forecasting_model.pt"
)
TEST_TRANSFORMER_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "forecasting"
    / "real_forecasting_test_predictions.npz"
)
OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "forecasting"
    / "causal_confidence"
)
VALIDATION_TRANSFORMER_FILE = (
    OUTPUT_DIR
    / "validation_transformer_predictions.npz"
)


VALIDATION_COMPARISON_FILE = (
    OUTPUT_DIR
    / "validation_model_comparison.csv"
)


SELECTION_FILE = (
    OUTPUT_DIR
    / "validation_selected_forecasters.csv"
)


CAUSAL_CONFIDENCE_FILE = (
    OUTPUT_DIR
    / "causal_confidence_24h.csv"
)


FINAL_TEST_METRICS_FILE = (
    OUTPUT_DIR
    / "final_leakage_free_test_metrics.csv"
)


FINAL_TEST_ARCHIVE = (
    OUTPUT_DIR
    / "final_leakage_free_test_forecast.npz"
)


METADATA_FILE = (
    OUTPUT_DIR
    / "causal_confidence_metadata.json"
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
        1.0 - ss_res / ss_tot
    )


def inverse_scale_feature(
    values,
    feature_name,
    scaler,
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
        * (maximum - minimum)
        + minimum
    )


def normalize_feature(
    values,
    feature_name,
    scaler,
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

    difference = (
        maximum - minimum
    )

    if abs(difference) <= EPS:

        raise RuntimeError(
            f"Invalid scaler range for {feature_name}"
        )

    return (
        values - minimum
    ) / difference


def inverse_scale_array(
    normalized_values,
    scaler,
):

    output = np.zeros_like(
        normalized_values,
        dtype=np.float64,
    )

    for feature_index, feature_name in enumerate(
        FEATURE_NAMES
    ):

        output[
            :,
            :,
            feature_index
        ] = inverse_scale_feature(

            normalized_values[
                :,
                :,
                feature_index
            ],

            feature_name,

            scaler,
        )

    return output


def normalize_array(
    original_values,
    scaler,
):

    output = np.zeros_like(
        original_values,
        dtype=np.float64,
    )

    for feature_index, feature_name in enumerate(
        FEATURE_NAMES
    ):

        output[
            :,
            :,
            feature_index
        ] = normalize_feature(

            original_values[
                :,
                :,
                feature_index
            ],

            feature_name,

            scaler,
        )

    return output


def apply_physical_clipping(
    forecast,
):

    output = np.asarray(
        forecast,
        dtype=np.float64,
    ).copy()

    # PV >= 0

    output[
        :,
        :,
        PV_INDEX
    ] = np.clip(

        output[
            :,
            :,
            PV_INDEX
        ],

        0.0,
        None,
    )


    # Load >= 0

    output[
        :,
        :,
        LOAD_INDEX
    ] = np.clip(

        output[
            :,
            :,
            LOAD_INDEX
        ],

        0.0,
        None,
    )


    # EV charging demand >= 0

    output[
        :,
        :,
        EV_INDEX
    ] = np.clip(

        output[
            :,
            :,
            EV_INDEX
        ],

        0.0,
        None,
    )


    # Price is intentionally NOT clipped.

    return output


# ============================================================
# 4. CHECK FILES
# ============================================================

section(
    "FC-HMARL STEP 7H"
)


for file_path in [

    SEQUENCE_FILE,

    SCALER_FILE,

    BEST_MODEL_FILE,

    TEST_TRANSFORMER_FILE,

]:

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
# 5. LOAD PREPARED SEQUENCES
# ============================================================

data = np.load(
    SEQUENCE_FILE,
    allow_pickle=True,
)


X_val = np.asarray(
    data["X_val"],
    dtype=np.float32,
)


y_val = np.asarray(
    data["y_val"],
    dtype=np.float32,
)


X_test = np.asarray(
    data["X_test"],
    dtype=np.float32,
)


y_test = np.asarray(
    data["y_test"],
    dtype=np.float32,
)


print(
    f"Validation X : {X_val.shape}"
)

print(
    f"Validation y : {y_val.shape}"
)

print(
    f"Test X       : {X_test.shape}"
)

print(
    f"Test y       : {y_test.shape}"
)


# ============================================================
# 6. LOAD SCALER
# ============================================================

scaler_df = pd.read_csv(
    SCALER_FILE
)


required_scaler_columns = [
    "target",
    "train_min",
    "train_max",
]


for column in required_scaler_columns:

    if column not in scaler_df.columns:

        raise RuntimeError(
            f"Scaler is missing column: {column}"
        )


scaler = scaler_df.set_index(
    "target"
)


actual_val_original = inverse_scale_array(
    y_val,
    scaler,
)


actual_test_original = inverse_scale_array(
    y_test,
    scaler,
)


# ============================================================
# 7. DAILY AND WEEKLY SEASONAL BASELINES
# ============================================================
#
# X shape:
#
#   [sample, 168 historical hours, 4 variables]
#
# Daily baseline:
#
#   last 24 h of history
#
# Weekly baseline:
#
#   first 24 h of 168 h history
#
# because:
#
#   168 h = 7 days
#
# ============================================================

section(
    "BUILDING VALIDATION SEASONAL BASELINES"
)


daily_val_norm = np.asarray(
    X_val[
        :,
        -24:,
        :
    ],
    dtype=np.float64,
)


weekly_val_norm = np.asarray(
    X_val[
        :,
        0:24,
        :
    ],
    dtype=np.float64,
)


daily_test_norm = np.asarray(
    X_test[
        :,
        -24:,
        :
    ],
    dtype=np.float64,
)


weekly_test_norm = np.asarray(
    X_test[
        :,
        0:24,
        :
    ],
    dtype=np.float64,
)


daily_val_original = inverse_scale_array(
    daily_val_norm,
    scaler,
)


weekly_val_original = inverse_scale_array(
    weekly_val_norm,
    scaler,
)


daily_test_original = inverse_scale_array(
    daily_test_norm,
    scaler,
)


weekly_test_original = inverse_scale_array(
    weekly_test_norm,
    scaler,
)


# ============================================================
# 8. BUILD ORIGINAL TRANSFORMER
# ============================================================

section(
    "LOADING BEST ORIGINAL TRANSFORMER CHECKPOINT"
)


model_config = ForecastModelConfig(

    input_window=168,

    forecast_horizon=24,

    input_features=4,

    target_features=4,

    hidden_dimension=128,

    number_of_attention_heads=4,

    number_of_encoder_layers=2,

    feedforward_dimension=256,

    dropout=0.1,

    activation="gelu",

    use_learnable_positional_encoding=True,
)


model = MultiHorizonTransformerForecaster(
    model_config
)


checkpoint = torch.load(
    BEST_MODEL_FILE,
    map_location="cpu",
)


# ------------------------------------------------------------
# Robust checkpoint handling
# ------------------------------------------------------------

if isinstance(
    checkpoint,
    dict,
):

    if "model_state_dict" in checkpoint:

        state_dict = checkpoint[
            "model_state_dict"
        ]

    elif "state_dict" in checkpoint:

        state_dict = checkpoint[
            "state_dict"
        ]

    else:

        # May already be a plain state_dict.

        state_dict = checkpoint

else:

    raise RuntimeError(
        "Unsupported checkpoint format."
    )


model.load_state_dict(
    state_dict
)


model.eval()


print(
    f"Loaded checkpoint:\n{BEST_MODEL_FILE}"
)


# ============================================================
# 9. GENERATE VALIDATION TRANSFORMER FORECASTS
# ============================================================

section(
    "GENERATING VALIDATION TRANSFORMER FORECASTS"
)


validation_predictions_norm = []


batch_size = 256


with torch.no_grad():

    for start_index in range(
        0,
        X_val.shape[0],
        batch_size,
    ):

        end_index = min(
            start_index + batch_size,
            X_val.shape[0],
        )


        batch = torch.tensor(

            X_val[
                start_index:end_index
            ],

            dtype=torch.float32,
        )


        prediction = model(
            batch
        )


        validation_predictions_norm.append(

            prediction
            .cpu()
            .numpy()
        )


transformer_val_norm_raw = np.concatenate(
    validation_predictions_norm,
    axis=0,
)


print(
    f"Validation Transformer shape: "
    f"{transformer_val_norm_raw.shape}"
)


# ============================================================
# 10. INVERSE-SCALE + CLIP VALIDATION TRANSFORMER
# ============================================================

transformer_val_original_raw = inverse_scale_array(
    transformer_val_norm_raw,
    scaler,
)


transformer_val_original = apply_physical_clipping(
    transformer_val_original_raw
)


transformer_val_norm = normalize_array(
    transformer_val_original,
    scaler,
)


np.savez_compressed(

    VALIDATION_TRANSFORMER_FILE,

    predictions_normalized=
        transformer_val_norm,

    predictions_original=
        transformer_val_original,

    predictions_original_raw=
        transformer_val_original_raw,

    targets_normalized=
        y_val,

    targets_original=
        actual_val_original,

    feature_names=
        np.array(
            FEATURE_NAMES
        ),
)


# ============================================================
# 11. LOAD EXISTING TEST TRANSFORMER PREDICTIONS
# ============================================================

test_transformer_data = np.load(
    TEST_TRANSFORMER_FILE,
    allow_pickle=True,
)


transformer_test_original = np.asarray(
    test_transformer_data[
        "predictions_original"
    ],
    dtype=np.float64,
)


if transformer_test_original.shape != (
    y_test.shape
):

    raise RuntimeError(
        "Test Transformer shape mismatch."
    )


transformer_test_original = (
    apply_physical_clipping(
        transformer_test_original
    )
)


transformer_test_norm = normalize_array(
    transformer_test_original,
    scaler,
)


# ============================================================
# 12. VALIDATION MODEL COMPARISON
# ============================================================

section(
    "VALIDATION-ONLY FORECAST MODEL COMPARISON"
)


candidate_val = {

    "daily_seasonal":
        daily_val_original,

    "weekly_seasonal":
        weekly_val_original,

    "transformer":
        transformer_val_original,
}


comparison_rows = []


for feature_index, feature_name in enumerate(
    FEATURE_NAMES
):

    actual = actual_val_original[
        :,
        :,
        feature_index
    ].reshape(-1)


    for method_name, forecast_array in (
        candidate_val.items()
    ):

        forecast = forecast_array[
            :,
            :,
            feature_index
        ].reshape(-1)


        row = {

            "feature":
                feature_name,

            "method":
                method_name,

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


        comparison_rows.append(
            row
        )


        print(
            f"{feature_name:22s} | "
            f"{method_name:16s} | "
            f"MAE={row['MAE']:.8f} | "
            f"RMSE={row['RMSE']:.8f} | "
            f"R2={row['R2']:.6f}"
        )


comparison_df = pd.DataFrame(
    comparison_rows
)


comparison_df.to_csv(
    VALIDATION_COMPARISON_FILE,
    index=False,
)


# ============================================================
# 13. SELECT BEST FORECASTER ON VALIDATION
# ============================================================

section(
    "VALIDATION FORECASTER SELECTION"
)


selected_methods = {}


selection_rows = []


for feature_name in FEATURE_NAMES:

    feature_results = comparison_df[
        comparison_df["feature"]
        == feature_name
    ].copy()


    best_row = (
        feature_results
        .sort_values(
            by="MAE",
            ascending=True,
        )
        .iloc[0]
    )


    selected_method = str(
        best_row["method"]
    )


    selected_methods[
        feature_name
    ] = selected_method


    selection_rows.append(
        {

            "feature":
                feature_name,

            "selected_method":
                selected_method,

            "validation_MAE":
                float(
                    best_row["MAE"]
                ),

            "validation_RMSE":
                float(
                    best_row["RMSE"]
                ),

            "validation_R2":
                float(
                    best_row["R2"]
                ),
        }
    )


    print(
        f"{feature_name:22s} -> "
        f"{selected_method}"
    )


selection_df = pd.DataFrame(
    selection_rows
)


selection_df.to_csv(
    SELECTION_FILE,
    index=False,
)


# ============================================================
# 14. BUILD VALIDATION-SELECTED HYBRID
# ============================================================

candidate_val_norm = {

    "daily_seasonal":
        daily_val_norm,

    "weekly_seasonal":
        weekly_val_norm,

    "transformer":
        transformer_val_norm,
}


candidate_test_original = {

    "daily_seasonal":
        daily_test_original,

    "weekly_seasonal":
        weekly_test_original,

    "transformer":
        transformer_test_original,
}


candidate_test_norm = {

    "daily_seasonal":
        daily_test_norm,

    "weekly_seasonal":
        weekly_test_norm,

    "transformer":
        transformer_test_norm,
}


hybrid_val_norm = np.zeros_like(
    y_val,
    dtype=np.float64,
)


hybrid_test_original = np.zeros_like(
    actual_test_original,
    dtype=np.float64,
)


hybrid_test_norm = np.zeros_like(
    y_test,
    dtype=np.float64,
)


for feature_index, feature_name in enumerate(
    FEATURE_NAMES
):

    method = selected_methods[
        feature_name
    ]


    hybrid_val_norm[
        :,
        :,
        feature_index
    ] = candidate_val_norm[
        method
    ][
        :,
        :,
        feature_index
    ]


    hybrid_test_original[
        :,
        :,
        feature_index
    ] = candidate_test_original[
        method
    ][
        :,
        :,
        feature_index
    ]


    hybrid_test_norm[
        :,
        :,
        feature_index
    ] = candidate_test_norm[
        method
    ][
        :,
        :,
        feature_index
    ]


# ============================================================
# 15. VALIDATION FORECAST ERRORS
# ============================================================

section(
    "CALIBRATING CAUSAL CONFIDENCE FROM VALIDATION"
)


validation_absolute_error = np.abs(
    y_val
    - hybrid_val_norm
)
mean_error_by_horizon = np.mean(
    validation_absolute_error,
    axis=0,
)

mean_pv_error = mean_error_by_horizon[
    :,
    PV_INDEX
]


mean_load_error = mean_error_by_horizon[
    :,
    LOAD_INDEX
]


mean_ev_error = mean_error_by_horizon[
    :,
    EV_INDEX
]


mean_price_error = mean_error_by_horizon[
    :,
    PRICE_INDEX
]

epsilon_RL = (
    calculate_renewable_load_uncertainty(

        mean_pv_error,

        mean_load_error,
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

        mean_ev_error,

        mean_price_error,
    )
)


Phi_causal = (
    calculate_global_confidence(

        omega_RL,

        epsilon_EM,

        error_scale=1.0,

        minimum_confidence=0.0,

        maximum_confidence=1.0,
    )
)


epsilon_RL_direct = np.sqrt(
    mean_pv_error ** 2
    + mean_load_error ** 2
)


epsilon_EM_direct = np.sqrt(
    mean_ev_error ** 2
    + mean_price_error ** 2
)


Phi_direct = np.exp(
    -(
        epsilon_RL_direct
        + epsilon_EM_direct
    )
)


difference = np.max(
    np.abs(
        Phi_causal
        - Phi_direct
    )
)


print(
    f"Direct/module Phi difference: "
    f"{difference:.12e}"
)


if not np.allclose(
    Phi_causal,
    Phi_direct,
    atol=1e-10,
):

    raise RuntimeError(
        "Causal confidence formula mismatch."
    )

confidence_df = pd.DataFrame(
    {

        "forecast_hour":
            np.arange(
                1,
                25,
            ),

        "mean_pv_error_normalized":
            mean_pv_error,

        "mean_load_error_normalized":
            mean_load_error,

        "mean_ev_error_normalized":
            mean_ev_error,

        "mean_price_error_normalized":
            mean_price_error,

        "epsilon_RL":
            epsilon_RL,

        "omega_RL":
            omega_RL,

        "epsilon_EM":
            epsilon_EM,

        "Phi_causal":
            Phi_causal,
    }
)


confidence_df.to_csv(
    CAUSAL_CONFIDENCE_FILE,
    index=False,
)


print()

print(
    "24-hour causal confidence:"
)


for row in confidence_df.itertuples():

    print(
        f"h={row.forecast_hour:02d} | "
        f"eps_RL={row.epsilon_RL:.6f} | "
        f"eps_EM={row.epsilon_EM:.6f} | "
        f"Phi={row.Phi_causal:.6f}"
    )


print()

print(
    f"Mean causal Phi : "
    f"{np.mean(Phi_causal):.8f}"
)

print(
    f"Min causal Phi  : "
    f"{np.min(Phi_causal):.8f}"
)

print(
    f"Max causal Phi  : "
    f"{np.max(Phi_causal):.8f}"
)
section(
    "LEAKAGE-FREE FINAL TEST RESULTS"
)


test_metric_rows = []


for feature_index, feature_name in enumerate(
    FEATURE_NAMES
):

    actual = actual_test_original[
        :,
        :,
        feature_index
    ].reshape(-1)


    forecast = hybrid_test_original[
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


    test_metric_rows.append(
        {

            "feature":
                feature_name,

            "selected_method":
                selected_methods[
                    feature_name
                ],

            "MAE":
                feature_mae,

            "RMSE":
                feature_rmse,

            "R2":
                feature_r2,
        }
    )


    print()

    print(
        feature_name
    )

    print(
        f"  Method : "
        f"{selected_methods[feature_name]}"
    )

    print(
        f"  MAE    : "
        f"{feature_mae:.8f}"
    )

    print(
        f"  RMSE   : "
        f"{feature_rmse:.8f}"
    )

    print(
        f"  R2     : "
        f"{feature_r2:.6f}"
    )


test_metrics_df = pd.DataFrame(
    test_metric_rows
)


test_metrics_df.to_csv(
    FINAL_TEST_METRICS_FILE,
    index=False,
)


# ============================================================
# 20. SAVE FINAL LEAKAGE-FREE ARCHIVE
# ============================================================

np.savez_compressed(

    FINAL_TEST_ARCHIVE,

    forecast_original=
        hybrid_test_original,

    forecast_normalized=
        hybrid_test_norm,

    actual_original=
        actual_test_original,

    actual_normalized=
        y_test,

    causal_confidence_24h=
        Phi_causal,

    epsilon_RL_24h=
        epsilon_RL,

    omega_RL_24h=
        omega_RL,

    epsilon_EM_24h=
        epsilon_EM,

    validation_mean_error_by_horizon=
        mean_error_by_horizon,

    feature_names=
        np.array(
            FEATURE_NAMES
        ),

    selected_methods=
        np.array(
            [
                selected_methods[
                    feature_name
                ]
                for feature_name
                in FEATURE_NAMES
            ]
        ),
)


# ============================================================
# 21. SAVE METADATA
# ============================================================

metadata = {

    "selection_basis":
        "validation MAE only",

    "test_used_for_selection":
        False,

    "confidence_calibration":
        (
            "mean absolute normalized validation "
            "forecast error by forecast horizon"
        ),

    "confidence_is_causal":
        True,

    "confidence_horizon":
        24,

    "selected_methods":
        selected_methods,

    "equations": {

        "epsilon_RL":
            "sqrt(e_PV^2 + e_Load^2)",

        "omega_RL":
            "exp(-epsilon_RL)",

        "epsilon_EM":
            "sqrt(e_EV^2 + e_Price^2)",

        "Phi":
            "exp(-(epsilon_RL + epsilon_EM))",
    },

    "note":
        (
            "Test actual values are not used "
            "to calculate causal confidence."
        ),
}


with open(
    METADATA_FILE,
    "w",
    encoding="utf-8",
) as file:

    json.dump(
        metadata,
        file,
        indent=4,
    )


# ============================================================
# 22. FINAL VALIDATION
# ============================================================

section(
    "STEP 7H FINAL VALIDATION"
)


print(
    f"Selected forecasters:"
)


for feature_name in FEATURE_NAMES:

    print(
        f"  {feature_name:22s} -> "
        f"{selected_methods[feature_name]}"
    )


print()


print(
    f"Causal Phi shape : "
    f"{Phi_causal.shape}"
)


print(
    f"Causal Phi range : "
    f"{np.min(Phi_causal):.8f} "
    f"to "
    f"{np.max(Phi_causal):.8f}"
)


if Phi_causal.shape != (
    24,
):

    raise RuntimeError(
        "Causal confidence must contain "
        "exactly 24 horizon values."
    )


if np.any(
    Phi_causal < 0.0
) or np.any(
    Phi_causal > 1.0
):

    raise RuntimeError(
        "Causal confidence outside [0,1]."
    )


if hybrid_test_original.shape != (
    1291,
    24,
    4,
):

    raise RuntimeError(
        "Unexpected final test forecast shape."
    )


print()

print(
    "[OK] Forecast model selection used "
    "validation only."
)

print(
    "[OK] Test set remained untouched "
    "until final evaluation."
)

print(
    "[OK] Causal confidence uses "
    "validation errors only."
)

print(
    "[OK] No future test actual value "
    "is needed by the confidence signal."
)


# ============================================================
# 23. OUTPUT LOCATIONS
# ============================================================

section(
    "STEP 7H COMPLETE"
)


print(
    f"Validation Transformer predictions:\n"
    f"{VALIDATION_TRANSFORMER_FILE}"
)

print()

print(
    f"Validation candidate comparison:\n"
    f"{VALIDATION_COMPARISON_FILE}"
)

print()

print(
    f"Selected forecasters:\n"
    f"{SELECTION_FILE}"
)

print()

print(
    f"24-hour causal confidence:\n"
    f"{CAUSAL_CONFIDENCE_FILE}"
)

print()

print(
    f"Final leakage-free test metrics:\n"
    f"{FINAL_TEST_METRICS_FILE}"
)

print()

print(
    f"Final leakage-free forecast archive:\n"
    f"{FINAL_TEST_ARCHIVE}"
)

print()

print(
    "STEP 7H COMPLETE."
)
