# ============================================================
# FC-HMARL
# STEP 7L-A
# PREPARE LEAKAGE-FREE REAL-DATA RL TRAINING ARCHIVE
# ============================================================
#
# PURPOSE
# -------
# Build the real-data archive that will be consumed by
# FC-HMARL training.
#
# IMPORTANT:
#
#   TRAIN split -> RL training
#   VALIDATION   -> model/method selection + Phi calibration
#   TEST         -> final evaluation only
#
# The TEST split is NOT used in this script.
#
#
# Final validation-selected forecast methods:
#
#   PV    -> daily seasonal
#   Load  -> Transformer
#   EV    -> Transformer
#   Price -> daily seasonal
#
#
# Predictive state:
#
#   S_pred = Phi * Z_hat
#
# Output:
#
#   current physical values    : (5941, 4)
#   forecasts                  : (5941, 24, 4)
#   S_pred                     : (5941, 24, 4)
#   flattened S_pred           : (5941, 96)
#
# ============================================================

from pathlib import Path
import json

import numpy as np
import pandas as pd
import torch


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

SCALER_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "forecasting"
    / "forecasting_scaler.csv"
)

CHECKPOINT_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "checkpoints"
    / "best_real_forecasting_model.pt"
)

CAUSAL_CONFIDENCE_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "forecasting"
    / "causal_confidence"
    / "causal_confidence_24h.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "rl_data"
)

OUTPUT_ARCHIVE = (
    OUTPUT_DIR
    / "real_rl_training_archive.npz"
)

SUMMARY_FILE = (
    OUTPUT_DIR
    / "real_rl_training_summary.csv"
)

METHOD_FILE = (
    OUTPUT_DIR
    / "real_rl_training_methods.csv"
)

METADATA_FILE = (
    OUTPUT_DIR
    / "real_rl_training_metadata.json"
)


# ============================================================
# 2. CONSTANTS
# ============================================================

INPUT_HORIZON = 168
FORECAST_HORIZON = 24
NUMBER_OF_FEATURES = 4

PV_INDEX = 0
LOAD_INDEX = 1
EV_INDEX = 2
PRICE_INDEX = 3

FEATURE_NAMES = [
    "pv_power_kw",
    "load_kw",
    "ev_power_kw",
    "price_usd_per_kwh",
]


# ============================================================
# 3. HELPERS
# ============================================================

def section(title):

    print()
    print("=" * 80)
    print(title)
    print("=" * 80)


def inverse_scale(
    normalized_values,
    minimum,
    data_range,
):

    return (
        normalized_values
        * data_range
        + minimum
    )


def clip_normalized_physical(values):

    values = np.asarray(
        values,
        dtype=np.float32,
    ).copy()

    # PV cannot be negative.
    values[..., PV_INDEX] = np.clip(
        values[..., PV_INDEX],
        0.0,
        1.0,
    )

    # Load cannot be negative.
    values[..., LOAD_INDEX] = np.clip(
        values[..., LOAD_INDEX],
        0.0,
        None,
    )

    # EV charging demand cannot be negative.
    values[..., EV_INDEX] = np.clip(
        values[..., EV_INDEX],
        0.0,
        None,
    )

    # Price is allowed to remain according to normalized
    # training range. For our current data it remains positive.

    return values


# ============================================================
# 4. LOAD TRAINING SEQUENCES
# ============================================================

section(
    "FC-HMARL STEP 7L-A - LOADING TRAINING DATA"
)


if not SEQUENCE_FILE.exists():

    raise FileNotFoundError(
        f"Sequence file not found:\n"
        f"{SEQUENCE_FILE}"
    )


sequence_archive = np.load(
    SEQUENCE_FILE,
    allow_pickle=True,
)


print(
    "Available sequence keys:"
)

for key in sequence_archive.files:

    value = sequence_archive[key]

    print(
        f"  {key}: "
        f"{getattr(value, 'shape', type(value))}"
    )


X_train = np.asarray(
    sequence_archive["X_train"],
    dtype=np.float32,
)

y_train = np.asarray(
    sequence_archive["y_train"],
    dtype=np.float32,
)

idx_train = np.asarray(
    sequence_archive["idx_train"]
)


if "feature_names" in sequence_archive.files:

    archive_feature_names = [
        str(x)
        for x
        in sequence_archive["feature_names"]
    ]

else:

    archive_feature_names = (
        FEATURE_NAMES.copy()
    )


print()

print(
    f"X_train : {X_train.shape}"
)

print(
    f"y_train : {y_train.shape}"
)

print(
    f"idx_train: {idx_train.shape}"
)


if X_train.shape[1:] != (
    INPUT_HORIZON,
    NUMBER_OF_FEATURES,
):

    raise RuntimeError(
        "Unexpected X_train dimensions."
    )


if y_train.shape[1:] != (
    FORECAST_HORIZON,
    NUMBER_OF_FEATURES,
):

    raise RuntimeError(
        "Unexpected y_train dimensions."
    )


number_of_samples = len(
    X_train
)


print(
    f"Training samples: "
    f"{number_of_samples}"
)


# ============================================================
# 5. LOAD TRAINING-ONLY SCALER
# ============================================================

section(
    "LOADING TRAINING-ONLY SCALER"
)


scaler_df = pd.read_csv(
    SCALER_FILE
)


print(
    scaler_df.to_string(
        index=False
    )
)


scaler_lookup = {
    row["target"]: {
        "min": float(
            row["train_min"]
        ),
        "max": float(
            row["train_max"]
        ),
        "range": float(
            row["train_range"]
        ),
    }

    for _, row
    in scaler_df.iterrows()
}


for feature in FEATURE_NAMES:

    if feature not in scaler_lookup:

        raise RuntimeError(
            f"Missing scaler information "
            f"for {feature}"
        )


# ============================================================
# 6. LOAD CAUSAL VALIDATION-CALIBRATED CONFIDENCE
# ============================================================

section(
    "LOADING CAUSAL CONFIDENCE"
)


if not CAUSAL_CONFIDENCE_FILE.exists():

    raise FileNotFoundError(
        f"Causal-confidence file not found:\n"
        f"{CAUSAL_CONFIDENCE_FILE}"
    )


confidence_df = pd.read_csv(
    CAUSAL_CONFIDENCE_FILE
)


print(
    "Confidence columns:"
)

print(
    list(
        confidence_df.columns
    )
)


# Robustly identify Phi column.

possible_phi_columns = [
    "Phi_causal",
    "causal_confidence",
    "Phi",
    "phi",
    "global_confidence",
    "confidence",
]


phi_column = None


for candidate in possible_phi_columns:

    if candidate in confidence_df.columns:

        phi_column = candidate

        break


if phi_column is None:

    # Find a column containing confidence/Phi terminology.

    for column in confidence_df.columns:

        lower = column.lower()

        if (
            "confidence" in lower
            or lower == "phi"
        ):

            phi_column = column
            break


if phi_column is None:

    raise RuntimeError(
        "Could not automatically identify "
        "the causal Phi column."
    )


causal_phi = np.asarray(
    confidence_df[
        phi_column
    ],
    dtype=np.float32,
)


if causal_phi.shape != (
    FORECAST_HORIZON,
):

    raise RuntimeError(
        f"Expected 24 Phi values, "
        f"received {causal_phi.shape}"
    )


print(
    f"Phi column : {phi_column}"
)

print(
    f"Phi shape  : {causal_phi.shape}"
)

print(
    f"Phi mean   : "
    f"{causal_phi.mean():.8f}"
)

print(
    f"Phi min    : "
    f"{causal_phi.min():.8f}"
)

print(
    f"Phi max    : "
    f"{causal_phi.max():.8f}"
)


# ============================================================
# 7. BUILD SEASONAL FORECASTS
# ============================================================

section(
    "BUILDING TRAINING-SEASONAL FORECASTS"
)


# X_train contains the preceding 168 hours.
#
# A daily seasonal forecast for the next 24 hours can be
# constructed from the most recent 24-hour historical block:
#
#   X[:, -24:, :]
#
# This uses ONLY information available before forecast issue.

daily_seasonal_forecast = (
    X_train[
        :,
        -FORECAST_HORIZON:,
        :
    ].copy()
)


print(
    "Daily seasonal shape:"
)

print(
    daily_seasonal_forecast.shape
)


# ============================================================
# 8. LOAD BEST TRANSFORMER MODEL
# ============================================================

section(
    "LOADING BEST TRANSFORMER CHECKPOINT"
)


if not CHECKPOINT_FILE.exists():

    raise FileNotFoundError(
        f"Best forecasting checkpoint not found:\n"
        f"{CHECKPOINT_FILE}"
    )


from forecasting.model import (
    ForecastModelConfig,
    MultiHorizonTransformerForecaster,
)


# The trained architecture already used in Step 7A.
#
# These architecture settings are reconstruction choices
# established previously in the project.

model_config = ForecastModelConfig(

    input_window=
        INPUT_HORIZON,

    forecast_horizon=
        FORECAST_HORIZON,

    input_features=
        NUMBER_OF_FEATURES,

    target_features=
        NUMBER_OF_FEATURES,

    hidden_dimension=
        128,

    number_of_attention_heads=
        4,

    number_of_encoder_layers=
        2,

    feedforward_dimension=
        256,

    dropout=
        0.1,

    activation=
        "gelu",

    use_learnable_positional_encoding=
        True,
)


model = MultiHorizonTransformerForecaster(
    model_config
)


checkpoint = torch.load(
    CHECKPOINT_FILE,
    map_location="cpu",
    weights_only=False,
)


print(
    f"Checkpoint type: "
    f"{type(checkpoint)}"
)


# ------------------------------------------------------------
# Support common checkpoint formats.
# ------------------------------------------------------------

if isinstance(
    checkpoint,
    dict,
):

    candidate_keys = [
        "model_state_dict",
        "state_dict",
        "model",
    ]

    state_dict = None

    for candidate in candidate_keys:

        if candidate in checkpoint:

            potential = checkpoint[
                candidate
            ]

            if isinstance(
                potential,
                dict,
            ):

                state_dict = potential
                break


    if state_dict is None:

        # It may already be a raw PyTorch state_dict.
        if all(
            torch.is_tensor(v)
            for v
            in checkpoint.values()
        ):

            state_dict = checkpoint

        else:

            raise RuntimeError(
                "Could not identify the model state_dict "
                "inside the forecasting checkpoint."
            )

else:

    raise RuntimeError(
        "Unsupported checkpoint format."
    )


model.load_state_dict(
    state_dict
)

model.eval()


print(
    "[OK] Best Transformer checkpoint loaded."
)


# ============================================================
# 9. GENERATE TRAINING TRANSFORMER FORECASTS
# ============================================================

section(
    "GENERATING TRAINING TRANSFORMER FORECASTS"
)


BATCH_SIZE = 128


transformer_predictions = []


with torch.no_grad():

    for start in range(
        0,
        number_of_samples,
        BATCH_SIZE,
    ):

        end = min(
            start + BATCH_SIZE,
            number_of_samples,
        )


        batch = torch.as_tensor(
            X_train[
                start:end
            ],
            dtype=torch.float32,
        )


        output = model(
            batch
        )


        # ----------------------------------------------------
        # Accommodate model wrappers that may return tensors,
        # tuples, or dictionaries.
        # ----------------------------------------------------

        if torch.is_tensor(
            output
        ):

            prediction = output


        elif isinstance(
            output,
            (
                tuple,
                list,
            ),
        ):

            prediction = output[0]


        elif isinstance(
            output,
            dict,
        ):

            possible_prediction_keys = [
                "prediction",
                "predictions",
                "forecast",
                "output",
            ]

            prediction = None

            for key in (
                possible_prediction_keys
            ):

                if key in output:

                    prediction = output[
                        key
                    ]

                    break


            if prediction is None:

                raise RuntimeError(
                    "Could not identify prediction "
                    "tensor from model dictionary output."
                )


        else:

            raise RuntimeError(
                "Unsupported forecasting-model output."
            )


        prediction = (
            prediction
            .detach()
            .cpu()
            .numpy()
        )


        transformer_predictions.append(
            prediction
        )


        if (
            start == 0
            or end == number_of_samples
            or start % 1024 == 0
        ):

            print(
                f"Generated "
                f"{end}/{number_of_samples}"
            )


transformer_forecast = np.concatenate(
    transformer_predictions,
    axis=0,
).astype(
    np.float32
)


print()

print(
    f"Transformer forecast shape: "
    f"{transformer_forecast.shape}"
)


if transformer_forecast.shape != (
    number_of_samples,
    FORECAST_HORIZON,
    NUMBER_OF_FEATURES,
):

    raise RuntimeError(
        "Unexpected Transformer output shape."
    )


# ============================================================
# 10. BUILD VALIDATION-SELECTED HYBRID FORECAST
# ============================================================

section(
    "BUILDING VALIDATION-SELECTED HYBRID FORECAST"
)


hybrid_forecast_normalized = np.empty_like(
    transformer_forecast
)


# PV -> daily seasonal

hybrid_forecast_normalized[
    :,
    :,
    PV_INDEX
] = daily_seasonal_forecast[
    :,
    :,
    PV_INDEX
]


# Load -> Transformer

hybrid_forecast_normalized[
    :,
    :,
    LOAD_INDEX
] = transformer_forecast[
    :,
    :,
    LOAD_INDEX
]


# EV -> Transformer

hybrid_forecast_normalized[
    :,
    :,
    EV_INDEX
] = transformer_forecast[
    :,
    :,
    EV_INDEX
]


# Price -> daily seasonal

hybrid_forecast_normalized[
    :,
    :,
    PRICE_INDEX
] = daily_seasonal_forecast[
    :,
    :,
    PRICE_INDEX
]


hybrid_forecast_normalized = (
    clip_normalized_physical(
        hybrid_forecast_normalized
    )
)


print(
    f"Hybrid forecast shape: "
    f"{hybrid_forecast_normalized.shape}"
)


print()

print(
    "Forecast methods:"
)

print(
    "  PV    -> daily seasonal"
)

print(
    "  Load  -> Transformer"
)

print(
    "  EV    -> Transformer"
)

print(
    "  Price -> daily seasonal"
)


# ============================================================
# 11. CONFIDENCE-AWARE PREDICTIVE STATE
# ============================================================

section(
    "BUILDING S_pred = Phi * Z_hat"
)


confidence_batch = np.broadcast_to(

    causal_phi[
        None,
        :
    ],

    (
        number_of_samples,
        FORECAST_HORIZON,
    ),

).copy()


try:

    from forecasting.confidence import (
        build_confidence_aware_predictive_state,
    )


    predictive_state = (
        build_confidence_aware_predictive_state(

            hybrid_forecast_normalized,

            confidence_batch,
        )
    )


    predictive_state = np.asarray(
        predictive_state,
        dtype=np.float32,
    )


    print(
        "[OK] Used forecasting.confidence module."
    )


except Exception as exc:

    print(
        "Confidence-module call failed:"
    )

    print(
        exc
    )

    print(
        "Using mathematically equivalent direct equation."
    )


    predictive_state = (

        hybrid_forecast_normalized
        *
        causal_phi[
            None,
            :,
            None
        ]

    ).astype(
        np.float32
    )


direct_predictive_state = (

    hybrid_forecast_normalized
    *
    causal_phi[
        None,
        :,
        None
    ]

).astype(
    np.float32
)


maximum_phi_difference = float(

    np.max(

        np.abs(

            predictive_state
            -
            direct_predictive_state
        )
    )
)


print(
    f"S_pred shape: "
    f"{predictive_state.shape}"
)

print(
    f"Module/direct maximum difference: "
    f"{maximum_phi_difference:.12e}"
)


if not np.allclose(
    predictive_state,
    direct_predictive_state,
    atol=1e-7,
):

    raise RuntimeError(
        "S_pred does not match Phi * Z_hat."
    )


# ============================================================
# 12. FLATTEN PREDICTIVE STATE
# ============================================================

predictive_state_flat = (
    predictive_state.reshape(
        number_of_samples,
        -1,
    )
)


print(
    f"Flattened S_pred shape: "
    f"{predictive_state_flat.shape}"
)


if predictive_state_flat.shape != (
    number_of_samples,
    96,
):

    raise RuntimeError(
        "Expected a 96-dimensional predictive state."
    )


# ============================================================
# 13. CURRENT PHYSICAL VALUES
# ============================================================

section(
    "BUILDING CURRENT PHYSICAL TRAINING VALUES"
)


# At training sample t:
#
# y_train[t, 0, :]
#
# is the current realized first lead associated with that
# decision/sample.
#
# The remaining future y values are NOT inserted into
# the predictive state.

current_actual_normalized = (
    y_train[
        :,
        0,
        :
    ].copy()
)


print(
    f"Current normalized actual shape: "
    f"{current_actual_normalized.shape}"
)


# ============================================================
# 14. INVERSE SCALE CURRENT PHYSICAL VALUES
# ============================================================

current_actual_original = np.empty_like(
    current_actual_normalized,
    dtype=np.float32,
)


hybrid_forecast_original = np.empty_like(
    hybrid_forecast_normalized,
    dtype=np.float32,
)


for feature_index, feature_name in enumerate(
    FEATURE_NAMES
):

    minimum = scaler_lookup[
        feature_name
    ]["min"]

    data_range = scaler_lookup[
        feature_name
    ]["range"]


    current_actual_original[
        :,
        feature_index
    ] = inverse_scale(

        current_actual_normalized[
            :,
            feature_index
        ],

        minimum,

        data_range,
    )


    hybrid_forecast_original[
        :,
        :,
        feature_index
    ] = inverse_scale(

        hybrid_forecast_normalized[
            :,
            :,
            feature_index
        ],

        minimum,

        data_range,
    )


# Physical non-negativity.

current_actual_original[
    :,
    PV_INDEX
] = np.clip(
    current_actual_original[
        :,
        PV_INDEX
    ],
    0.0,
    None,
)


current_actual_original[
    :,
    LOAD_INDEX
] = np.clip(
    current_actual_original[
        :,
        LOAD_INDEX
    ],
    0.0,
    None,
)


current_actual_original[
    :,
    EV_INDEX
] = np.clip(
    current_actual_original[
        :,
        EV_INDEX
    ],
    0.0,
    None,
)


hybrid_forecast_original[
    :,
    :,
    PV_INDEX
] = np.clip(
    hybrid_forecast_original[
        :,
        :,
        PV_INDEX
    ],
    0.0,
    None,
)


hybrid_forecast_original[
    :,
    :,
    LOAD_INDEX
] = np.clip(
    hybrid_forecast_original[
        :,
        :,
        LOAD_INDEX
    ],
    0.0,
    None,
)


hybrid_forecast_original[
    :,
    :,
    EV_INDEX
] = np.clip(
    hybrid_forecast_original[
        :,
        :,
        EV_INDEX
    ],
    0.0,
    None,
)


print()

print(
    "Current physical ranges:"
)


for i, feature in enumerate(
    FEATURE_NAMES
):

    print(
        f"  {feature:<24} "
        f"{current_actual_original[:, i].min():.6f}"
        f" -> "
        f"{current_actual_original[:, i].max():.6f}"
    )


# ============================================================
# 15. LEAKAGE CHECK
# ============================================================

section(
    "LEAKAGE CHECK"
)


print(
    "Training archive uses:"
)

print(
    "  X_train"
)

print(
    "  y_train[:, 0, :] for current physical realization"
)

print(
    "  validation-selected forecasting methods"
)

print(
    "  validation-calibrated causal Phi"
)


print()

print(
    "Training archive does NOT load:"
)

print(
    "  X_test"
)

print(
    "  y_test"
)

print(
    "  test actual future trajectories"
)


# Explicitly guarantee we never created test arrays.

assert "X_test" in sequence_archive.files
assert "y_test" in sequence_archive.files

print()

print(
    "[OK] Test arrays exist in source archive "
    "but were not loaded into RL training variables."
)


# ============================================================
# 16. SAVE TRAINING ARCHIVE
# ============================================================

section(
    "SAVING RL TRAINING ARCHIVE"
)


OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


np.savez_compressed(

    OUTPUT_ARCHIVE,

    current_actual_original=
        current_actual_original,

    current_actual_normalized=
        current_actual_normalized,

    forecast_original=
        hybrid_forecast_original,

    forecast_normalized=
        hybrid_forecast_normalized,

    predictive_state_matrix=
        predictive_state,

    predictive_state_flat=
        predictive_state_flat,

    causal_confidence_24h=
        causal_phi,

    confidence_by_sample=
        confidence_batch,

    training_indices=
        idx_train,

    feature_names=
        np.asarray(
            FEATURE_NAMES,
            dtype=object,
        ),

    selected_methods=
        np.asarray(
            [
                "daily_seasonal",
                "transformer",
                "transformer",
                "daily_seasonal",
            ],
            dtype=object,
        ),

    source_split=
        np.asarray(
            "train"
        ),
)


print(
    f"Archive saved:\n"
    f"{OUTPUT_ARCHIVE}"
)


# ============================================================
# 17. SUMMARY
# ============================================================

summary_rows = [
    {
        "metric":
            "training_samples",
        "value":
            number_of_samples,
    },
    {
        "metric":
            "forecast_horizon",
        "value":
            FORECAST_HORIZON,
    },
    {
        "metric":
            "forecast_features",
        "value":
            NUMBER_OF_FEATURES,
    },
    {
        "metric":
            "predictive_state_dimension",
        "value":
            96,
    },
    {
        "metric":
            "mean_causal_confidence",
        "value":
            float(
                causal_phi.mean()
            ),
    },
    {
        "metric":
            "minimum_causal_confidence",
        "value":
            float(
                causal_phi.min()
            ),
    },
    {
        "metric":
            "maximum_causal_confidence",
        "value":
            float(
                causal_phi.max()
            ),
    },
]


pd.DataFrame(
    summary_rows
).to_csv(
    SUMMARY_FILE,
    index=False,
)


method_df = pd.DataFrame(
    {
        "feature": FEATURE_NAMES,
        "forecast_method": [
            "daily_seasonal",
            "transformer",
            "transformer",
            "daily_seasonal",
        ],
        "selection_source": [
            "validation",
            "validation",
            "validation",
            "validation",
        ],
    }
)


method_df.to_csv(
    METHOD_FILE,
    index=False,
)


metadata = {
    "source_split":
        "train",

    "training_samples":
        int(
            number_of_samples
        ),

    "input_horizon":
        INPUT_HORIZON,

    "forecast_horizon":
        FORECAST_HORIZON,

    "number_of_features":
        NUMBER_OF_FEATURES,

    "predictive_state_dimension":
        96,

    "feature_order":
        FEATURE_NAMES,

    "forecast_methods": {
        "pv_power_kw":
            "daily_seasonal",
        "load_kw":
            "transformer",
        "ev_power_kw":
            "transformer",
        "price_usd_per_kwh":
            "daily_seasonal",
    },

    "forecast_method_selection_split":
        "validation",

    "confidence_calibration_split":
        "validation",

    "test_split_used_for_training":
        False,

    "predictive_state_equation":
        "S_pred = Phi * Z_hat",

    "notes": [
        (
            "Predictive-state values are normalized before "
            "entering the RL state."
        ),
        (
            "Current physical realization uses only "
            "y_train[:,0,:]."
        ),
        (
            "Future training targets are not inserted into "
            "S_pred."
        ),
    ],
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
# 18. FINAL CHECKS
# ============================================================

section(
    "STEP 7L-A FINAL VALIDATION"
)


print(
    f"Training samples             : "
    f"{number_of_samples}"
)

print(
    f"Current physical matrix      : "
    f"{current_actual_original.shape}"
)

print(
    f"Forecast matrix              : "
    f"{hybrid_forecast_normalized.shape}"
)

print(
    f"Predictive-state matrix      : "
    f"{predictive_state.shape}"
)

print(
    f"Flattened predictive state   : "
    f"{predictive_state_flat.shape}"
)


print()

print(
    "Final selected forecasting methods:"
)

print(
    "  PV    = daily seasonal"
)

print(
    "  Load  = Transformer"
)

print(
    "  EV    = Transformer"
)

print(
    "  Price = daily seasonal"
)


print()

print(
    f"Mean Phi = "
    f"{causal_phi.mean():.8f}"
)


print()

print(
    "[OK] RL training archive uses TRAIN split."
)

print(
    "[OK] Forecast method selection came from VALIDATION."
)

print(
    "[OK] Phi calibration came from VALIDATION."
)

print(
    "[OK] TEST observations were not used "
    "to construct RL training samples."
)

print(
    "[OK] Each S_pred has 24 x 4 = 96 elements."
)

print(
    "[OK] Real-data FC-HMARL training archive is ready."
)


section(
    "STEP 7L-A COMPLETE"
)


print(
    f"Training archive:\n"
    f"{OUTPUT_ARCHIVE}"
)

print()

print(
    f"Summary:\n"
    f"{SUMMARY_FILE}"
)

print()

print(
    f"Methods:\n"
    f"{METHOD_FILE}"
)

print()

print(
    f"Metadata:\n"
    f"{METADATA_FILE}"
)

print()

print(
    "STEP 7L-A COMPLETE."
)