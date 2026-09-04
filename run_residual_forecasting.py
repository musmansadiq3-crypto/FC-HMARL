# ============================================================
# FC-HMARL
# STEP 7F - SEASONAL-RESIDUAL TRANSFORMER TRAINING
# ============================================================
#
# Residual forecasting formulation:
#
#   residual = actual_future - seasonal_baseline
#
# Transformer learns:
#
#   historical 168 h  ->  future 24 h residual
#
# Final forecast:
#
#   forecast = seasonal_baseline + predicted_residual
#
# Seasonal baseline selected in Step 7D:
#
#   PV    -> daily  (t-24)
#   Load  -> daily  (t-24)
#   EV    -> weekly (t-168)
#   Price -> daily  (t-24)
#
# ============================================================

from pathlib import Path
import json
import time
import random

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from forecasting.model import (
    ForecastModelConfig,
    MultiHorizonTransformerForecaster,
)

from forecasting.trainer import (
    ForecastTrainerConfig,
    train_forecasting_model,
)


# ============================================================
# 1. PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(
    r"D:\Molvi paper review\FC_HMARL"
)

RESIDUAL_DATA_FILE = (
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

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "forecasting"
    / "residual"
)

CHECKPOINT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "checkpoints"
)

FINAL_MODEL_FILE = (
    OUTPUT_DIR
    / "final_residual_forecasting_model.pt"
)

TRAINING_HISTORY_FILE = (
    OUTPUT_DIR
    / "residual_forecasting_training_history.csv"
)

TEST_METRICS_FILE = (
    OUTPUT_DIR
    / "residual_forecasting_test_metrics.csv"
)

PREDICTIONS_FILE = (
    OUTPUT_DIR
    / "residual_forecasting_test_predictions.npz"
)

COMPARISON_FILE = (
    OUTPUT_DIR
    / "residual_vs_baseline_comparison.csv"
)

CONFIGURATION_FILE = (
    OUTPUT_DIR
    / "residual_forecasting_configuration.json"
)

BEST_CHECKPOINT_FILE = (
    CHECKPOINT_DIR
    / "best_residual_forecasting_model.pt"
)


# ============================================================
# 2. CONFIGURATION
# ============================================================

RANDOM_SEED = 42

INPUT_WINDOW = 168
FORECAST_HORIZON = 24
INPUT_FEATURES = 4
TARGET_FEATURES = 4

HIDDEN_DIMENSION = 128
NUMBER_OF_HEADS = 4
NUMBER_OF_ENCODER_LAYERS = 2
FEEDFORWARD_DIMENSION = 256
DROPOUT = 0.10

LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
BATCH_SIZE = 64
MAXIMUM_EPOCHS = 100

EARLY_STOPPING_PATIENCE = 15
MINIMUM_IMPROVEMENT = 1e-6

GRADIENT_CLIP_NORM = 1.0

SCHEDULER_FACTOR = 0.5
SCHEDULER_PATIENCE = 5
MINIMUM_LEARNING_RATE = 1e-6


FEATURE_NAMES = [
    "pv_power_kw",
    "load_kw",
    "ev_power_kw",
    "price_usd_per_kwh",
]

EPSILON = 1e-8


# ============================================================
# 3. REPRODUCIBILITY
# ============================================================

random.seed(
    RANDOM_SEED
)

np.random.seed(
    RANDOM_SEED
)

torch.manual_seed(
    RANDOM_SEED
)

if torch.cuda.is_available():

    torch.cuda.manual_seed_all(
        RANDOM_SEED
    )


# ============================================================
# 4. PRINT HELPERS
# ============================================================

def section(title):

    print()

    print(
        "=" * 80
    )

    print(
        title
    )

    print(
        "=" * 80
    )


# ============================================================
# 5. DATASET WRAPPER
# ============================================================

class ResidualForecastDataset(Dataset):

    def __init__(
        self,
        X,
        residual_targets,
    ):

        self.X = torch.tensor(
            X,
            dtype=torch.float32,
        )

        self.y = torch.tensor(
            residual_targets,
            dtype=torch.float32,
        )


    def __len__(self):

        return self.X.shape[0]


    def __getitem__(
        self,
        index,
    ):

        return (
            self.X[index],
            self.y[index],
        )


# ============================================================
# 6. METRIC FUNCTIONS
# ============================================================

def mae(actual, predicted):

    return float(
        np.mean(
            np.abs(
                predicted - actual
            )
        )
    )


def mse(actual, predicted):

    return float(
        np.mean(
            (
                predicted - actual
            ) ** 2
        )
    )


def rmse(actual, predicted):

    return float(
        np.sqrt(
            mse(
                actual,
                predicted
            )
        )
    )


def r2_score(
    actual,
    predicted,
):

    actual = np.asarray(
        actual,
        dtype=np.float64
    )

    predicted = np.asarray(
        predicted,
        dtype=np.float64
    )

    ss_res = np.sum(
        (
            actual
            - predicted
        ) ** 2
    )

    ss_tot = np.sum(
        (
            actual
            - np.mean(actual)
        ) ** 2
    )

    if ss_tot <= EPSILON:

        return np.nan

    return float(
        1.0
        - ss_res
        / ss_tot
    )


def nmae(
    actual,
    predicted,
):

    denominator = np.mean(
        np.abs(actual)
    )

    if denominator <= EPSILON:

        return np.nan

    return float(
        100.0
        * mae(
            actual,
            predicted
        )
        / denominator
    )


def nrmse(
    actual,
    predicted,
):

    denominator = np.mean(
        np.abs(actual)
    )

    if denominator <= EPSILON:

        return np.nan

    return float(
        100.0
        * rmse(
            actual,
            predicted
        )
        / denominator
    )


def mape(
    actual,
    predicted,
):

    actual = np.asarray(
        actual,
        dtype=np.float64
    )

    predicted = np.asarray(
        predicted,
        dtype=np.float64
    )

    valid = (
        np.abs(actual)
        > EPSILON
    )

    if not np.any(valid):

        return np.nan

    return float(
        np.mean(
            np.abs(
                (
                    predicted[valid]
                    - actual[valid]
                )
                /
                actual[valid]
            )
        )
        * 100.0
    )


def smape(
    actual,
    predicted,
):

    actual = np.asarray(
        actual,
        dtype=np.float64
    )

    predicted = np.asarray(
        predicted,
        dtype=np.float64
    )

    denominator = (
        np.abs(actual)
        + np.abs(predicted)
    )

    valid = (
        denominator
        > EPSILON
    )

    if not np.any(valid):

        return np.nan

    return float(
        np.mean(
            2.0
            * np.abs(
                predicted[valid]
                - actual[valid]
            )
            / denominator[valid]
        )
        * 100.0
    )


# ============================================================
# 7. LOAD DATA
# ============================================================

section(
    "FC-HMARL - STEP 7F RESIDUAL TRANSFORMER TRAINING"
)


if not RESIDUAL_DATA_FILE.exists():

    raise FileNotFoundError(
        f"Residual dataset not found:\n"
        f"{RESIDUAL_DATA_FILE}"
    )


if not SCALER_FILE.exists():

    raise FileNotFoundError(
        f"Scaler file not found:\n"
        f"{SCALER_FILE}"
    )


OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

CHECKPOINT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


data = np.load(
    RESIDUAL_DATA_FILE,
    allow_pickle=True,
)


X_train = np.asarray(
    data["X_train"],
    dtype=np.float32,
)

X_val = np.asarray(
    data["X_val"],
    dtype=np.float32,
)

X_test = np.asarray(
    data["X_test"],
    dtype=np.float32,
)


y_train = np.asarray(
    data["y_train"],
    dtype=np.float32,
)

y_val = np.asarray(
    data["y_val"],
    dtype=np.float32,
)

y_test = np.asarray(
    data["y_test"],
    dtype=np.float32,
)


baseline_train = np.asarray(
    data["baseline_train"],
    dtype=np.float32,
)

baseline_val = np.asarray(
    data["baseline_val"],
    dtype=np.float32,
)

baseline_test = np.asarray(
    data["baseline_test"],
    dtype=np.float32,
)


residual_train = np.asarray(
    data["residual_train"],
    dtype=np.float32,
)

residual_val = np.asarray(
    data["residual_val"],
    dtype=np.float32,
)

residual_test = np.asarray(
    data["residual_test"],
    dtype=np.float32,
)


print(
    f"Training samples   : "
    f"{X_train.shape[0]:,}"
)

print(
    f"Validation samples : "
    f"{X_val.shape[0]:,}"
)

print(
    f"Testing samples    : "
    f"{X_test.shape[0]:,}"
)

print()

print(
    f"Input shape        : "
    f"{X_train.shape}"
)

print(
    f"Residual target    : "
    f"{residual_train.shape}"
)


# ============================================================
# 8. DATASET OBJECTS
# ============================================================

train_dataset = ResidualForecastDataset(
    X_train,
    residual_train,
)

validation_dataset = ResidualForecastDataset(
    X_val,
    residual_val,
)

test_dataset = ResidualForecastDataset(
    X_test,
    residual_test,
)


# ============================================================
# 9. MODEL
# ============================================================

section(
    "BUILDING RESIDUAL TRANSFORMER"
)


model_config = ForecastModelConfig(

    input_window=
        INPUT_WINDOW,

    forecast_horizon=
        FORECAST_HORIZON,

    input_features=
        INPUT_FEATURES,

    target_features=
        TARGET_FEATURES,

    hidden_dimension=
        HIDDEN_DIMENSION,

    number_of_attention_heads=
        NUMBER_OF_HEADS,

    number_of_encoder_layers=
        NUMBER_OF_ENCODER_LAYERS,

    feedforward_dimension=
        FEEDFORWARD_DIMENSION,

    dropout=
        DROPOUT,

    activation=
        "gelu",

    use_learnable_positional_encoding=
        True,
)


model = MultiHorizonTransformerForecaster(
    model_config
)


number_of_parameters = sum(
    parameter.numel()
    for parameter in model.parameters()
)


print(
    f"Trainable parameters : "
    f"{number_of_parameters:,}"
)

print(
    f"Architecture          : "
    f"{HIDDEN_DIMENSION}-dim Transformer"
)

print(
    f"Attention heads       : "
    f"{NUMBER_OF_HEADS}"
)

print(
    f"Encoder layers        : "
    f"{NUMBER_OF_ENCODER_LAYERS}"
)

print(
    f"Forecast target       : "
    f"24-hour seasonal residual"
)


# ============================================================
# 10. FORWARD-PASS CHECK
# ============================================================

section(
    "MODEL FORWARD-PASS CHECK"
)


with torch.no_grad():

    sample_X = torch.tensor(
        X_train[:2],
        dtype=torch.float32,
    )

    sample_output = model(
        sample_X
    )


print(
    f"Input  : "
    f"{tuple(sample_X.shape)}"
)

print(
    f"Output : "
    f"{tuple(sample_output.shape)}"
)


if tuple(
    sample_output.shape
) != (
    2,
    24,
    4,
):

    raise RuntimeError(
        "Unexpected model output shape."
    )


print()

print(
    "[OK] Model forward pass validated."
)


# ============================================================
# 11. TRAINER CONFIGURATION
# ============================================================

trainer_config = ForecastTrainerConfig(

    optimizer=
        "adam",

    learning_rate=
        LEARNING_RATE,

    weight_decay=
        WEIGHT_DECAY,

    batch_size=
        BATCH_SIZE,

    maximum_epochs=
        MAXIMUM_EPOCHS,

    early_stopping_patience=
        EARLY_STOPPING_PATIENCE,

    minimum_improvement=
        MINIMUM_IMPROVEMENT,

    gradient_clip_norm=
        GRADIENT_CLIP_NORM,

    scheduler=
        "reduce_on_plateau",

    scheduler_factor=
        SCHEDULER_FACTOR,

    scheduler_patience=
        SCHEDULER_PATIENCE,

    minimum_learning_rate=
        MINIMUM_LEARNING_RATE,

    loss_function=
        "mse",

    random_seed=
        RANDOM_SEED,

    device=
        "auto",

    checkpoint_directory=
        str(
            CHECKPOINT_DIR
        ),

    checkpoint_filename=
        "best_residual_forecasting_model.pt",
)


# ============================================================
# 12. SAVE CONFIGURATION
# ============================================================

configuration = {

    "method":
        "seasonal_residual_transformer",

    "random_seed":
        RANDOM_SEED,

    "input_window":
        INPUT_WINDOW,

    "forecast_horizon":
        FORECAST_HORIZON,

    "features":
        FEATURE_NAMES,

    "seasonal_baseline": {

        "pv_power_kw":
            "daily_t_minus_24",

        "load_kw":
            "daily_t_minus_24",

        "ev_power_kw":
            "weekly_t_minus_168",

        "price_usd_per_kwh":
            "daily_t_minus_24",
    },

    "model": {

        "hidden_dimension":
            HIDDEN_DIMENSION,

        "attention_heads":
            NUMBER_OF_HEADS,

        "encoder_layers":
            NUMBER_OF_ENCODER_LAYERS,

        "feedforward_dimension":
            FEEDFORWARD_DIMENSION,

        "dropout":
            DROPOUT,

        "activation":
            "gelu",
    },

    "training": {

        "learning_rate":
            LEARNING_RATE,

        "weight_decay":
            WEIGHT_DECAY,

        "batch_size":
            BATCH_SIZE,

        "maximum_epochs":
            MAXIMUM_EPOCHS,

        "early_stopping_patience":
            EARLY_STOPPING_PATIENCE,
    },
}


with open(
    CONFIGURATION_FILE,
    "w",
    encoding="utf-8",
) as file:

    json.dump(
        configuration,
        file,
        indent=4,
    )


# ============================================================
# 13. TRAIN MODEL
# ============================================================

section(
    "TRAINING SEASONAL-RESIDUAL TRANSFORMER"
)


print(
    "IMPORTANT:"
)

print(
    "The network is learning residual corrections,"
)

print(
    "not absolute PV/load/EV/price trajectories."
)

print()


start_time = time.time()


training_result = train_forecasting_model(

    model=
        model,

    train_dataset=
        train_dataset,

    validation_dataset=
        validation_dataset,

    config=
        trainer_config,

    verbose=
        True,
)


training_time = (
    time.time()
    - start_time
)


# ============================================================
# 14. HANDLE TRAINER RETURN VALUE
# ============================================================

#
# Existing trainer implementations may return:
#
#   model
#
# or:
#
#   (model, history)
#
# This handles both cases.
#

history = None


if isinstance(
    training_result,
    tuple,
):

    trained_model = (
        training_result[0]
    )

    if len(
        training_result
    ) > 1:

        history = (
            training_result[1]
        )

else:

    trained_model = (
        training_result
    )


print()

print(
    f"Training time: "
    f"{training_time:.2f} seconds"
)


# ============================================================
# 15. SAVE FINAL MODEL
# ============================================================

torch.save(

    trained_model.state_dict(),

    FINAL_MODEL_FILE,
)


print()

print(
    f"Final residual model saved:\n"
    f"{FINAL_MODEL_FILE}"
)


# ============================================================
# 16. SAVE TRAINING HISTORY
# ============================================================

section(
    "SAVING TRAINING HISTORY"
)


history_saved = False


if history is not None:

    # --------------------------------------------------------
    # First try an existing dataframe conversion.
    # --------------------------------------------------------

    if hasattr(
        history,
        "to_dataframe",
    ):

        try:

            history_df = (
                history.to_dataframe()
            )

            history_df.to_csv(
                TRAINING_HISTORY_FILE,
                index=False,
            )

            history_saved = True

        except Exception:

            pass


    # --------------------------------------------------------
    # Common history attributes.
    # --------------------------------------------------------

    if not history_saved:

        train_losses = getattr(
            history,
            "training_losses",
            None,
        )

        if train_losses is None:

            train_losses = getattr(
                history,
                "train_losses",
                None,
            )


        validation_losses = getattr(
            history,
            "validation_losses",
            None,
        )

        if validation_losses is None:

            validation_losses = getattr(
                history,
                "val_losses",
                None,
            )


        learning_rates = getattr(
            history,
            "learning_rates",
            None,
        )


        if (
            train_losses is not None
            and validation_losses is not None
        ):

            number_of_epochs = min(
                len(train_losses),
                len(validation_losses),
            )

            history_dictionary = {

                "epoch":
                    np.arange(
                        1,
                        number_of_epochs + 1,
                    ),

                "train_loss":
                    np.asarray(
                        train_losses
                    )[:number_of_epochs],

                "validation_loss":
                    np.asarray(
                        validation_losses
                    )[:number_of_epochs],
            }


            if learning_rates is not None:

                history_dictionary[
                    "learning_rate"
                ] = np.asarray(
                    learning_rates
                )[:number_of_epochs]


            history_df = pd.DataFrame(
                history_dictionary
            )

            history_df.to_csv(
                TRAINING_HISTORY_FILE,
                index=False,
            )

            history_saved = True


if history_saved:

    print(
        f"Training history saved:\n"
        f"{TRAINING_HISTORY_FILE}"
    )

else:

    print(
        "Trainer history object could not be converted "
        "automatically."
    )

    print(
        "This does not affect model training or evaluation."
    )


# ============================================================
# 17. DEVICE
# ============================================================

device = next(
    trained_model.parameters()
).device


trained_model.eval()


print()

print(
    f"Evaluation device: {device}"
)


# ============================================================
# 18. TEST RESIDUAL PREDICTION
# ============================================================

section(
    "RUNNING TEST-SET RESIDUAL FORECASTING"
)


test_predictions_residual = []


inference_batch_size = 256


with torch.no_grad():

    for start in range(
        0,
        X_test.shape[0],
        inference_batch_size,
    ):

        end = min(
            start
            + inference_batch_size,
            X_test.shape[0],
        )


        batch_X = torch.tensor(
            X_test[
                start:end
            ],
            dtype=torch.float32,
            device=device,
        )


        batch_prediction = trained_model(
            batch_X
        )


        test_predictions_residual.append(

            batch_prediction
            .detach()
            .cpu()
            .numpy()
        )


predicted_residual = np.concatenate(
    test_predictions_residual,
    axis=0,
)


print(
    f"Predicted residual shape : "
    f"{predicted_residual.shape}"
)

print(
    f"True residual shape      : "
    f"{residual_test.shape}"
)


# ============================================================
# 19. RECONSTRUCT NORMALIZED FORECAST
# ============================================================

section(
    "RECONSTRUCTING FINAL FORECAST"
)


prediction_normalized_raw = (

    baseline_test
    + predicted_residual
)


target_normalized = (
    y_test.copy()
)


baseline_normalized = (
    baseline_test.copy()
)


print(
    f"Seasonal baseline : "
    f"{baseline_normalized.shape}"
)

print(
    f"Residual forecast : "
    f"{predicted_residual.shape}"
)

print(
    f"Final forecast    : "
    f"{prediction_normalized_raw.shape}"
)


# ============================================================
# 20. LOAD TRAINING SCALER
# ============================================================

scaler = pd.read_csv(
    SCALER_FILE
)


required_scaler_columns = [
    "target",
    "train_min",
    "train_max",
]


for column in required_scaler_columns:

    if column not in scaler.columns:

        raise RuntimeError(
            f"Scaler missing column: {column}"
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


# ============================================================
# 21. INVERSE SCALE
# ============================================================

prediction_original_raw = np.zeros_like(
    prediction_normalized_raw,
    dtype=np.float64,
)

target_original = np.zeros_like(
    target_normalized,
    dtype=np.float64,
)

baseline_original = np.zeros_like(
    baseline_normalized,
    dtype=np.float64,
)


for feature_index, feature_name in enumerate(
    FEATURE_NAMES
):

    prediction_original_raw[
        :,
        :,
        feature_index
    ] = inverse_scale_feature(

        prediction_normalized_raw[
            :,
            :,
            feature_index
        ],

        feature_name,
    )


    target_original[
        :,
        :,
        feature_index
    ] = inverse_scale_feature(

        target_normalized[
            :,
            :,
            feature_index
        ],

        feature_name,
    )


    baseline_original[
        :,
        :,
        feature_index
    ] = inverse_scale_feature(

        baseline_normalized[
            :,
            :,
            feature_index
        ],

        feature_name,
    )


# ============================================================
# 22. PHYSICAL POST-PROCESSING
# ============================================================
#
# Current forecasting targets represent:
#
#   PV generation >= 0
#   Load          >= 0
#   EV charging    >= 0
#   Price           retained without clipping
#
# ============================================================

prediction_original_clipped = (
    prediction_original_raw.copy()
)


# PV >= 0

prediction_original_clipped[
    :,
    :,
    0
] = np.clip(

    prediction_original_clipped[
        :,
        :,
        0
    ],

    0.0,

    None,
)


# Load >= 0

prediction_original_clipped[
    :,
    :,
    1
] = np.clip(

    prediction_original_clipped[
        :,
        :,
        1
    ],

    0.0,

    None,
)


# EV charging >= 0

prediction_original_clipped[
    :,
    :,
    2
] = np.clip(

    prediction_original_clipped[
        :,
        :,
        2
    ],

    0.0,

    None,
)


# ============================================================
# 23. NEGATIVE PREDICTION DIAGNOSTIC
# ============================================================

section(
    "PHYSICAL PREDICTION DIAGNOSTIC"
)


for feature_index, feature_name in enumerate(
    FEATURE_NAMES
):

    negative_raw = int(
        np.sum(
            prediction_original_raw[
                :,
                :,
                feature_index
            ]
            < 0.0
        )
    )


    negative_final = int(
        np.sum(
            prediction_original_clipped[
                :,
                :,
                feature_index
            ]
            < 0.0
        )
    )


    print(
        f"{feature_name:22s} | "
        f"raw negative={negative_raw:6d} | "
        f"final negative={negative_final:6d}"
    )


# ============================================================
# 24. FINAL TEST METRICS
# ============================================================

section(
    "RESIDUAL TRANSFORMER TEST METRICS"
)


metric_rows = []


for feature_index, feature_name in enumerate(
    FEATURE_NAMES
):

    actual = target_original[
        :,
        :,
        feature_index
    ].reshape(-1)


    predicted = prediction_original_clipped[
        :,
        :,
        feature_index
    ].reshape(-1)


    metric_values = {

        "feature":
            feature_name,

        "MAE":
            mae(
                actual,
                predicted
            ),

        "RMSE":
            rmse(
                actual,
                predicted
            ),

        "MSE":
            mse(
                actual,
                predicted
            ),

        "R2":
            r2_score(
                actual,
                predicted
            ),

        "NMAE_percent":
            nmae(
                actual,
                predicted
            ),

        "NRMSE_percent":
            nrmse(
                actual,
                predicted
            ),

        "MAPE_percent":
            mape(
                actual,
                predicted
            ),

        "sMAPE_percent":
            smape(
                actual,
                predicted
            ),
    }


    metric_rows.append(
        metric_values
    )


    print()

    print(
        feature_name
    )

    print(
        f"  MAE   : "
        f"{metric_values['MAE']:.8f}"
    )

    print(
        f"  RMSE  : "
        f"{metric_values['RMSE']:.8f}"
    )

    print(
        f"  R2    : "
        f"{metric_values['R2']:.6f}"
    )

    print(
        f"  NMAE  : "
        f"{metric_values['NMAE_percent']:.4f}%"
    )

    print(
        f"  NRMSE : "
        f"{metric_values['NRMSE_percent']:.4f}%"
    )

    print(
        f"  MAPE  : "
        f"{metric_values['MAPE_percent']:.4f}%"
    )

    print(
        f"  sMAPE : "
        f"{metric_values['sMAPE_percent']:.4f}%"
    )


metrics_df = pd.DataFrame(
    metric_rows
)


metrics_df.to_csv(
    TEST_METRICS_FILE,
    index=False,
)


# ============================================================
# 25. COMPARE AGAINST SELECTED SEASONAL BASELINE
# ============================================================

section(
    "RESIDUAL TRANSFORMER VS SELECTED SEASONAL BASELINE"
)


comparison_rows = []


for feature_index, feature_name in enumerate(
    FEATURE_NAMES
):

    actual = target_original[
        :,
        :,
        feature_index
    ].reshape(-1)


    seasonal = baseline_original[
        :,
        :,
        feature_index
    ].reshape(-1)


    residual_model = prediction_original_clipped[
        :,
        :,
        feature_index
    ].reshape(-1)


    seasonal_mae = mae(
        actual,
        seasonal
    )

    residual_mae = mae(
        actual,
        residual_model
    )


    seasonal_rmse = rmse(
        actual,
        seasonal
    )

    residual_rmse = rmse(
        actual,
        residual_model
    )


    seasonal_r2 = r2_score(
        actual,
        seasonal
    )

    residual_r2 = r2_score(
        actual,
        residual_model
    )


    if seasonal_mae > EPSILON:

        mae_improvement = (
            (
                seasonal_mae
                - residual_mae
            )
            /
            seasonal_mae
            * 100.0
        )

    else:

        mae_improvement = np.nan


    if seasonal_rmse > EPSILON:

        rmse_improvement = (
            (
                seasonal_rmse
                - residual_rmse
            )
            /
            seasonal_rmse
            * 100.0
        )

    else:

        rmse_improvement = np.nan


    comparison_rows.append(
        {

            "feature":
                feature_name,

            "seasonal_MAE":
                seasonal_mae,

            "residual_transformer_MAE":
                residual_mae,

            "MAE_improvement_percent":
                mae_improvement,

            "seasonal_RMSE":
                seasonal_rmse,

            "residual_transformer_RMSE":
                residual_rmse,

            "RMSE_improvement_percent":
                rmse_improvement,

            "seasonal_R2":
                seasonal_r2,

            "residual_transformer_R2":
                residual_r2,
        }
    )


    print()

    print(
        feature_name
    )

    print(
        f"  Seasonal MAE      : "
        f"{seasonal_mae:.8f}"
    )

    print(
        f"  Residual-model MAE: "
        f"{residual_mae:.8f}"
    )

    print(
        f"  MAE improvement   : "
        f"{mae_improvement:.3f}%"
    )

    print(
        f"  Seasonal RMSE     : "
        f"{seasonal_rmse:.8f}"
    )

    print(
        f"  Residual RMSE     : "
        f"{residual_rmse:.8f}"
    )

    print(
        f"  RMSE improvement  : "
        f"{rmse_improvement:.3f}%"
    )

    print(
        f"  Seasonal R2       : "
        f"{seasonal_r2:.6f}"
    )

    print(
        f"  Residual-model R2 : "
        f"{residual_r2:.6f}"
    )


comparison_df = pd.DataFrame(
    comparison_rows
)


comparison_df.to_csv(
    COMPARISON_FILE,
    index=False,
)


# ============================================================
# 26. SAVE PREDICTIONS
# ============================================================

np.savez_compressed(

    PREDICTIONS_FILE,

    # Final physical forecast

    predictions_original=
        prediction_original_clipped,

    # Before clipping

    predictions_original_raw=
        prediction_original_raw,

    # Ground truth

    targets_original=
        target_original,

    # Seasonal reference

    seasonal_baseline_original=
        baseline_original,

    # Normalized residuals

    predicted_residual_normalized=
        predicted_residual,

    true_residual_normalized=
        residual_test,

    # Normalized reconstructed predictions

    predictions_normalized_raw=
        prediction_normalized_raw,

    targets_normalized=
        target_normalized,

    seasonal_baseline_normalized=
        baseline_normalized,

    feature_names=
        np.array(
            FEATURE_NAMES
        ),
)


print()

print(
    f"Predictions saved:\n"
    f"{PREDICTIONS_FILE}"
)


# ============================================================
# 27. ACCEPTANCE CHECK
# ============================================================

section(
    "STEP 7F ACCEPTANCE CHECK"
)


number_better_mae = int(
    np.sum(
        comparison_df[
            "MAE_improvement_percent"
        ]
        > 0.0
    )
)


number_better_rmse = int(
    np.sum(
        comparison_df[
            "RMSE_improvement_percent"
        ]
        > 0.0
    )
)


print(
    f"Variables beating seasonal baseline "
    f"by MAE  : {number_better_mae}/4"
)

print(
    f"Variables beating seasonal baseline "
    f"by RMSE : {number_better_rmse}/4"
)


print()


if (
    number_better_mae == 4
    and number_better_rmse == 4
):

    print(
        "[PASS] Residual Transformer improves all "
        "four variables."
    )

elif (
    number_better_mae >= 3
    and number_better_rmse >= 3
):

    print(
        "[PARTIAL PASS] Residual Transformer improves "
        "most variables."
    )

else:

    print(
        "[REVIEW REQUIRED] Residual Transformer does "
        "not consistently beat the seasonal baseline."
    )


# ============================================================
# 28. OUTPUT SUMMARY
# ============================================================

section(
    "STEP 7F COMPLETE"
)


print(
    f"Best checkpoint:\n"
    f"{BEST_CHECKPOINT_FILE}"
)

print()

print(
    f"Final model:\n"
    f"{FINAL_MODEL_FILE}"
)

print()

print(
    f"Test metrics:\n"
    f"{TEST_METRICS_FILE}"
)

print()

print(
    f"Baseline comparison:\n"
    f"{COMPARISON_FILE}"
)

print()

print(
    f"Prediction archive:\n"
    f"{PREDICTIONS_FILE}"
)

print()

print(
    "Residual Transformer training completed."
)

print()

print(
    "STEP 7F COMPLETE."
)