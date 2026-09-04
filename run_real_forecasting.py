# ============================================================
# FC-HMARL
# STEP 7A - TRAIN FORECASTING MODEL ON PREPARED REAL DATA
# ============================================================
#
# Input:
#   data/processed/forecasting/forecasting_sequences.npz
#   data/processed/forecasting/forecasting_scaler.csv
#
# Existing project modules used:
#   forecasting.model
#   forecasting.trainer
#
# Model:
#   Multi-Horizon Transformer
#
# Input:
#   168 hours x 4 variables
#
# Output:
#   24 hours x 4 variables
#
# Variables:
#   0 = PV power
#   1 = Load
#   2 = EV charging power
#   3 = Electricity price
#
# ============================================================

from pathlib import Path
import json
import time

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

DATA_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "forecasting"
)

SEQUENCE_FILE = (
    DATA_DIR
    / "forecasting_sequences.npz"
)

SCALER_FILE = (
    DATA_DIR
    / "forecasting_scaler.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "forecasting"
)

CHECKPOINT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "checkpoints"
)

BEST_MODEL_FILE = (
    CHECKPOINT_DIR
    / "best_real_forecasting_model.pt"
)

FINAL_MODEL_FILE = (
    OUTPUT_DIR
    / "final_real_forecasting_model.pt"
)

HISTORY_FILE = (
    OUTPUT_DIR
    / "real_forecasting_training_history.csv"
)

METRICS_FILE = (
    OUTPUT_DIR
    / "real_forecasting_test_metrics.csv"
)

PREDICTIONS_FILE = (
    OUTPUT_DIR
    / "real_forecasting_test_predictions.npz"
)

CONFIG_FILE = (
    OUTPUT_DIR
    / "real_forecasting_configuration.json"
)


# ============================================================
# 2. REPRODUCIBILITY
# ============================================================

RANDOM_SEED = 42

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
# 3. MODEL SETTINGS
# ============================================================
#
# These are the defaults already implemented in the tested
# forecasting package.
#
# Exact hidden widths/layer counts of the lost original code
# were not recovered, so these remain reconstruction choices.
# ============================================================

INPUT_WINDOW = 168
FORECAST_HORIZON = 24

INPUT_FEATURES = 4
TARGET_FEATURES = 4

HIDDEN_DIMENSION = 128

NUMBER_OF_ATTENTION_HEADS = 4

NUMBER_OF_ENCODER_LAYERS = 2

FEEDFORWARD_DIMENSION = 256

DROPOUT = 0.1

ACTIVATION = "gelu"

USE_LEARNABLE_POSITIONAL_ENCODING = True


# ============================================================
# 4. TRAINING SETTINGS
# ============================================================

LEARNING_RATE = 1e-4

WEIGHT_DECAY = 1e-5

BATCH_SIZE = 64

MAXIMUM_EPOCHS = 100

EARLY_STOPPING_PATIENCE = 15

MINIMUM_IMPROVEMENT = 1e-6

GRADIENT_CLIP_NORM = 1.0

SCHEDULER = "reduce_on_plateau"

SCHEDULER_FACTOR = 0.5

SCHEDULER_PATIENCE = 5

MINIMUM_LEARNING_RATE = 1e-6

LOSS_FUNCTION = "mse"


# ============================================================
# 5. FEATURE NAMES
# ============================================================

FEATURE_NAMES = [

    "pv_power_kw",

    "load_kw",

    "ev_power_kw",

    "price_usd_per_kwh",
]


# ============================================================
# 6. PRINT HELPERS
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


def subsection(title):

    print()

    print(
        title
    )

    print(
        "-" * 80
    )


# ============================================================
# 7. DATASET WRAPPER
# ============================================================
#
# forecasting_sequences.npz already contains the correct
# chronological windows.
#
# Therefore we should NOT create windows again.
# ============================================================

class PreparedForecastDataset(Dataset):

    def __init__(
        self,
        X,
        y,
    ):

        self.X = torch.tensor(
            X,
            dtype=torch.float32,
        )

        self.y = torch.tensor(
            y,
            dtype=torch.float32,
        )


        if len(self.X) != len(self.y):

            raise ValueError(
                "X and y contain different numbers of samples."
            )


    def __len__(self):

        return len(
            self.X
        )


    def __getitem__(
        self,
        index,
    ):

        return (
            self.X[index],
            self.y[index],
        )


# ============================================================
# 8. DEVICE
# ============================================================

section(
    "FC-HMARL - REAL-DATA FORECASTING TRAINING"
)


device_name = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


print(
    f"PyTorch version : {torch.__version__}"
)

print(
    f"CUDA available  : {torch.cuda.is_available()}"
)

print(
    f"Training device : {device_name}"
)


# ============================================================
# 9. CHECK FILES
# ============================================================

if not SEQUENCE_FILE.exists():

    raise FileNotFoundError(
        f"Sequence file not found:\n"
        f"{SEQUENCE_FILE}"
    )


if not SCALER_FILE.exists():

    raise FileNotFoundError(
        f"Scaler file not found:\n"
        f"{SCALER_FILE}"
    )


OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

CHECKPOINT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# 10. LOAD PREPARED SEQUENCES
# ============================================================

section(
    "LOADING PREPARED FORECASTING SEQUENCES"
)


data = np.load(
    SEQUENCE_FILE,
    allow_pickle=True,
)


X_train = data[
    "X_train"
]

y_train = data[
    "y_train"
]

X_val = data[
    "X_val"
]

y_val = data[
    "y_val"
]

X_test = data[
    "X_test"
]

y_test = data[
    "y_test"
]


print(
    f"X_train : {X_train.shape}"
)

print(
    f"y_train : {y_train.shape}"
)

print()

print(
    f"X_val   : {X_val.shape}"
)

print(
    f"y_val   : {y_val.shape}"
)

print()

print(
    f"X_test  : {X_test.shape}"
)

print(
    f"y_test  : {y_test.shape}"
)


# ============================================================
# 11. HARD DATA VALIDATION
# ============================================================

expected_x = (
    INPUT_WINDOW,
    INPUT_FEATURES,
)

expected_y = (
    FORECAST_HORIZON,
    TARGET_FEATURES,
)


for name, array in [

    (
        "X_train",
        X_train,
    ),

    (
        "X_val",
        X_val,
    ),

    (
        "X_test",
        X_test,
    ),

]:

    if array.ndim != 3:

        raise RuntimeError(
            f"{name} must be 3D."
        )


    if array.shape[1:] != expected_x:

        raise RuntimeError(

            f"{name} expected (*,{expected_x[0]},"
            f"{expected_x[1]}), got {array.shape}."
        )


for name, array in [

    (
        "y_train",
        y_train,
    ),

    (
        "y_val",
        y_val,
    ),

    (
        "y_test",
        y_test,
    ),

]:

    if array.ndim != 3:

        raise RuntimeError(
            f"{name} must be 3D."
        )


    if array.shape[1:] != expected_y:

        raise RuntimeError(

            f"{name} expected (*,{expected_y[0]},"
            f"{expected_y[1]}), got {array.shape}."
        )


for name, array in [

    ("X_train", X_train),

    ("y_train", y_train),

    ("X_val", X_val),

    ("y_val", y_val),

    ("X_test", X_test),

    ("y_test", y_test),

]:

    if not np.isfinite(
        array
    ).all():

        raise RuntimeError(
            f"{name} contains NaN or infinity."
        )


print()

print(
    "[OK] Prepared sequences passed validation."
)


# ============================================================
# 12. BUILD TORCH DATASETS
# ============================================================

section(
    "BUILDING TORCH DATASETS"
)


train_dataset = PreparedForecastDataset(
    X_train,
    y_train,
)

validation_dataset = PreparedForecastDataset(
    X_val,
    y_val,
)

test_dataset = PreparedForecastDataset(
    X_test,
    y_test,
)


print(
    f"Training samples   : {len(train_dataset):,}"
)

print(
    f"Validation samples : {len(validation_dataset):,}"
)

print(
    f"Testing samples    : {len(test_dataset):,}"
)


# ============================================================
# 13. MODEL CONFIGURATION
# ============================================================

section(
    "BUILDING MULTI-HORIZON TRANSFORMER"
)


model_config = ForecastModelConfig(

    input_window=INPUT_WINDOW,

    forecast_horizon=FORECAST_HORIZON,

    input_features=INPUT_FEATURES,

    target_features=TARGET_FEATURES,

    hidden_dimension=HIDDEN_DIMENSION,

    number_of_attention_heads=NUMBER_OF_ATTENTION_HEADS,

    number_of_encoder_layers=NUMBER_OF_ENCODER_LAYERS,

    feedforward_dimension=FEEDFORWARD_DIMENSION,

    dropout=DROPOUT,

    activation=ACTIVATION,

    use_learnable_positional_encoding=(
        USE_LEARNABLE_POSITIONAL_ENCODING
    ),
)


model = MultiHorizonTransformerForecaster(
    config=model_config
)


number_of_parameters = sum(

    parameter.numel()

    for parameter
    in model.parameters()
)


number_of_trainable_parameters = sum(

    parameter.numel()

    for parameter
    in model.parameters()

    if parameter.requires_grad
)


print(
    f"Input window              : {INPUT_WINDOW}"
)

print(
    f"Forecast horizon          : {FORECAST_HORIZON}"
)

print(
    f"Input features            : {INPUT_FEATURES}"
)

print(
    f"Target features           : {TARGET_FEATURES}"
)

print(
    f"Hidden dimension          : {HIDDEN_DIMENSION}"
)

print(
    f"Attention heads           : {NUMBER_OF_ATTENTION_HEADS}"
)

print(
    f"Encoder layers            : {NUMBER_OF_ENCODER_LAYERS}"
)

print(
    f"Feedforward dimension     : {FEEDFORWARD_DIMENSION}"
)

print(
    f"Total parameters          : {number_of_parameters:,}"
)

print(
    f"Trainable parameters      : "
    f"{number_of_trainable_parameters:,}"
)


# ============================================================
# 14. QUICK FORWARD-PASS CHECK
# ============================================================

subsection(
    "FORWARD-PASS VALIDATION"
)


model.eval()


with torch.no_grad():

    sample_x = torch.tensor(

        X_train[:2],

        dtype=torch.float32,
    )


    sample_output = model(
        sample_x
    )


print(
    f"Input shape  : {tuple(sample_x.shape)}"
)

print(
    f"Output shape : {tuple(sample_output.shape)}"
)


expected_output_shape = (

    2,

    FORECAST_HORIZON,

    TARGET_FEATURES,
)


if tuple(
    sample_output.shape
) != expected_output_shape:

    raise RuntimeError(

        "Unexpected model output shape. "

        f"Expected {expected_output_shape}, "

        f"received {tuple(sample_output.shape)}."
    )


print(
    "[OK] Transformer forward pass is correct."
)


# ============================================================
# 15. TRAINER CONFIGURATION
# ============================================================

section(
    "CONFIGURING FORECAST TRAINER"
)


trainer_config = ForecastTrainerConfig(

    optimizer="adam",

    learning_rate=LEARNING_RATE,

    weight_decay=WEIGHT_DECAY,

    batch_size=BATCH_SIZE,

    maximum_epochs=MAXIMUM_EPOCHS,

    early_stopping_patience=(
        EARLY_STOPPING_PATIENCE
    ),

    minimum_improvement=(
        MINIMUM_IMPROVEMENT
    ),

    gradient_clip_norm=(
        GRADIENT_CLIP_NORM
    ),

    scheduler=SCHEDULER,

    scheduler_factor=(
        SCHEDULER_FACTOR
    ),

    scheduler_patience=(
        SCHEDULER_PATIENCE
    ),

    minimum_learning_rate=(
        MINIMUM_LEARNING_RATE
    ),

    loss_function=(
        LOSS_FUNCTION
    ),

    random_seed=(
        RANDOM_SEED
    ),

    device=(
        device_name
    ),

    checkpoint_directory=str(
        CHECKPOINT_DIR
    ),

    checkpoint_filename=(
        BEST_MODEL_FILE.name
    ),
)


print(
    f"Optimizer             : adam"
)

print(
    f"Learning rate         : {LEARNING_RATE}"
)

print(
    f"Weight decay          : {WEIGHT_DECAY}"
)

print(
    f"Batch size            : {BATCH_SIZE}"
)

print(
    f"Maximum epochs        : {MAXIMUM_EPOCHS}"
)

print(
    f"Early-stop patience   : "
    f"{EARLY_STOPPING_PATIENCE}"
)

print(
    f"Loss function         : {LOSS_FUNCTION}"
)

print(
    f"Checkpoint            : {BEST_MODEL_FILE}"
)


# ============================================================
# 16. SAVE RUN CONFIGURATION
# ============================================================

run_configuration = {

    "random_seed": RANDOM_SEED,

    "device": device_name,

    "input_window": INPUT_WINDOW,

    "forecast_horizon": FORECAST_HORIZON,

    "features": FEATURE_NAMES,

    "model": {

        "hidden_dimension":
            HIDDEN_DIMENSION,

        "number_of_attention_heads":
            NUMBER_OF_ATTENTION_HEADS,

        "number_of_encoder_layers":
            NUMBER_OF_ENCODER_LAYERS,

        "feedforward_dimension":
            FEEDFORWARD_DIMENSION,

        "dropout":
            DROPOUT,

        "activation":
            ACTIVATION,

        "learnable_positional_encoding":
            USE_LEARNABLE_POSITIONAL_ENCODING,
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

        "loss_function":
            LOSS_FUNCTION,
    },

    "data": {

        "training_samples":
            len(train_dataset),

        "validation_samples":
            len(validation_dataset),

        "test_samples":
            len(test_dataset),
    },

    "reconstruction_note": (
        "Transformer hidden dimensions and related "
        "architectural hyperparameters are reconstruction "
        "choices unless independently supported by the "
        "manuscript."
    ),
}


with open(
    CONFIG_FILE,
    "w",
    encoding="utf-8",
) as file:

    json.dump(
        run_configuration,
        file,
        indent=4,
    )


# ============================================================
# 17. TRAIN
# ============================================================

section(
    "TRAINING FORECASTING MODEL"
)


start_time = time.time()


trained_model, history = train_forecasting_model(

    model=model,

    train_dataset=train_dataset,

    validation_dataset=validation_dataset,

    config=trainer_config,

    verbose=True,
)


elapsed_seconds = (
    time.time()
    - start_time
)


print()

print(
    f"Training time: "
    f"{elapsed_seconds:.2f} seconds"
)


# ============================================================
# 18. SAVE FINAL MODEL STATE
# ============================================================

torch.save(

    {
        "model_state_dict":
            trained_model.state_dict(),

        "model_config":
            model_config.__dict__,

        "feature_names":
            FEATURE_NAMES,

        "input_window":
            INPUT_WINDOW,

        "forecast_horizon":
            FORECAST_HORIZON,
    },

    FINAL_MODEL_FILE,
)


print(
    f"Final model saved:\n"
    f"{FINAL_MODEL_FILE}"
)


# ============================================================
# 19. SAVE TRAINING HISTORY
# ============================================================

section(
    "SAVING TRAINING HISTORY"
)


history_dict = {}


if hasattr(
    history,
    "__dict__",
):

    history_dict = history.__dict__.copy()


history_lengths = {

    key: len(value)

    for key, value
    in history_dict.items()

    if isinstance(
        value,
        (
            list,
            tuple,
            np.ndarray,
        ),
    )
}


if history_lengths:

    maximum_history_length = max(
        history_lengths.values()
    )


    history_table = {}


    for key, value in history_dict.items():

        if isinstance(
            value,
            (
                list,
                tuple,
                np.ndarray,
            ),
        ):

            values = list(
                value
            )


            values = (
                values
                + [np.nan]
                * (
                    maximum_history_length
                    - len(values)
                )
            )


            history_table[
                key
            ] = values


    history_df = pd.DataFrame(
        history_table
    )


    history_df.to_csv(
        HISTORY_FILE,
        index=False
    )


    print(
        f"Training history saved:\n"
        f"{HISTORY_FILE}"
    )

else:

    print(
        "History object does not expose "
        "list-like epoch records."
    )


# ============================================================
# 20. TEST INFERENCE
# ============================================================

section(
    "RUNNING TEST-SET FORECASTING"
)


trained_model = trained_model.to(
    device_name
)

trained_model.eval()


test_predictions = []


TEST_BATCH_SIZE = 128


with torch.no_grad():

    for start in range(
        0,
        len(X_test),
        TEST_BATCH_SIZE,
    ):

        end = min(
            start + TEST_BATCH_SIZE,
            len(X_test),
        )


        batch_x = torch.tensor(

            X_test[
                start:end
            ],

            dtype=torch.float32,

            device=device_name,
        )


        batch_prediction = trained_model(
            batch_x
        )


        test_predictions.append(

            batch_prediction
            .detach()
            .cpu()
            .numpy()
        )


test_predictions = np.concatenate(

    test_predictions,

    axis=0,
)


print(
    f"Prediction shape : "
    f"{test_predictions.shape}"
)

print(
    f"Target shape     : "
    f"{y_test.shape}"
)


if test_predictions.shape != y_test.shape:

    raise RuntimeError(
        "Prediction and test-target shapes differ."
    )


# ============================================================
# 21. LOAD TRAINING SCALER
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
            f"Scaler is missing column: {column}"
        )


scaler = scaler.set_index(
    "target"
)


# ============================================================
# 22. INVERSE NORMALIZATION
# ============================================================

def inverse_scale(
    normalized,
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

        normalized
        * (
            maximum
            - minimum
        )
        + minimum
    )


predictions_original = np.empty_like(

    test_predictions,

    dtype=np.float64,
)


targets_original = np.empty_like(

    y_test,

    dtype=np.float64,
)


for feature_index, feature_name in enumerate(
    FEATURE_NAMES
):

    predictions_original[
        :,
        :,
        feature_index
    ] = inverse_scale(

        test_predictions[
            :,
            :,
            feature_index
        ],

        feature_name,
    )


    targets_original[
        :,
        :,
        feature_index
    ] = inverse_scale(

        y_test[
            :,
            :,
            feature_index
        ],

        feature_name,
    )


# ============================================================
# 23. METRICS
# ============================================================

section(
    "TEST FORECASTING METRICS"
)


metric_rows = []


for feature_index, feature_name in enumerate(
    FEATURE_NAMES
):

    actual = targets_original[
        :,
        :,
        feature_index
    ]


    predicted = predictions_original[
        :,
        :,
        feature_index
    ]


    error = (
        predicted
        - actual
    )


    mae = float(
        np.mean(
            np.abs(
                error
            )
        )
    )


    mse = float(
        np.mean(
            error ** 2
        )
    )


    rmse = float(
        np.sqrt(
            mse
        )
    )


    denominator = np.abs(
        actual
    )


    valid_mape = (
        denominator
        > 1e-8
    )


    if np.any(
        valid_mape
    ):

        mape = float(

            np.mean(

                np.abs(
                    error[
                        valid_mape
                    ]
                )

                /

                denominator[
                    valid_mape
                ]

            )

            * 100.0
        )

    else:

        mape = np.nan


    metric_rows.append(

        {
            "feature":
                feature_name,

            "MAE":
                mae,

            "RMSE":
                rmse,

            "MSE":
                mse,

            "MAPE_percent":
                mape,
        }
    )


    print()

    print(
        feature_name
    )

    print(
        f"  MAE  : {mae:.8f}"
    )

    print(
        f"  RMSE : {rmse:.8f}"
    )

    print(
        f"  MSE  : {mse:.8f}"
    )

    print(
        f"  MAPE : {mape:.4f}%"
    )


metrics_df = pd.DataFrame(
    metric_rows
)


metrics_df.to_csv(
    METRICS_FILE,
    index=False
)


# ============================================================
# 24. SAVE TEST PREDICTIONS
# ============================================================

np.savez_compressed(

    PREDICTIONS_FILE,

    predictions_normalized=(
        test_predictions
    ),

    targets_normalized=(
        y_test
    ),

    predictions_original=(
        predictions_original
    ),

    targets_original=(
        targets_original
    ),

    feature_names=np.asarray(
        FEATURE_NAMES
    ),
)


print()

print(
    f"Metrics saved:\n"
    f"{METRICS_FILE}"
)

print()

print(
    f"Predictions saved:\n"
    f"{PREDICTIONS_FILE}"
)


# ============================================================
# 25. FINAL SUMMARY
# ============================================================

section(
    "STEP 7A COMPLETE"
)


print(
    "Real-data forecasting training completed."
)

print()

print(
    f"Training samples   : "
    f"{len(train_dataset):,}"
)

print(
    f"Validation samples : "
    f"{len(validation_dataset):,}"
)

print(
    f"Testing samples    : "
    f"{len(test_dataset):,}"
)

print()

print(
    f"Model input        : "
    f"168 hours x 4 variables"
)

print(
    f"Model output       : "
    f"24 hours x 4 variables"
)

print()

print(
    "Variables:"
)

for index, feature in enumerate(
    FEATURE_NAMES
):

    print(
        f"  {index} -> {feature}"
    )

print()

print(
    f"Best checkpoint:\n"
    f"{BEST_MODEL_FILE}"
)

print()

print(
    f"Final model:\n"
    f"{FINAL_MODEL_FILE}"
)

print()

print(
    f"Test metrics:\n"
    f"{METRICS_FILE}"
)

print()

print(
    "STEP 7A COMPLETE."
)