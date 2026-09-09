from pathlib import Path
import numpy as np
import pandas as pd
# ============================================================
# 1. PATHS
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

PREDICTION_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "forecasting"
    / "real_forecasting_test_predictions.npz"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "forecasting"
    / "evaluation"
)

OUTPUT_FILE = (
    OUTPUT_DIR
    / "forecasting_baseline_comparison.csv"
)


FEATURE_NAMES = [
    "pv_power_kw",
    "load_kw",
    "ev_power_kw",
    "price_usd_per_kwh",
]

EPSILON = 1e-8

# ============================================================
# 2. HELPERS
# ============================================================

def section(title):

    print()

    print("=" * 80)

    print(title)

    print("=" * 80)


def mae(actual, predicted):

    return float(
        np.mean(
            np.abs(
                predicted - actual
            )
        )
    )


def rmse(actual, predicted):

    return float(
        np.sqrt(
            np.mean(
                (
                    predicted - actual
                ) ** 2
            )
        )
    )


def r2(actual, predicted):

    actual = np.asarray(
        actual,
        dtype=float
    )

    predicted = np.asarray(
        predicted,
        dtype=float
    )

    ss_res = np.sum(
        (
            actual - predicted
        ) ** 2
    )

    ss_tot = np.sum(
        (
            actual - np.mean(actual)
        ) ** 2
    )

    if ss_tot <= EPSILON:

        return np.nan

    return float(
        1.0
        - ss_res / ss_tot
    )


def nmae(actual, predicted):

    denominator = np.mean(
        np.abs(actual)
    )

    if denominator <= EPSILON:

        return np.nan

    return float(
        100.0
        * mae(actual, predicted)
        / denominator
    )


def nrmse(actual, predicted):

    denominator = np.mean(
        np.abs(actual)
    )

    if denominator <= EPSILON:

        return np.nan

    return float(
        100.0
        * rmse(actual, predicted)
        / denominator
    )


# ============================================================
# 3. START
# ============================================================

section(
    "FC-HMARL - STEP 7C BASELINE FORECAST COMPARISON"
)


for file in [
    SEQUENCE_FILE,
    SCALER_FILE,
    PREDICTION_FILE,
]:

    if not file.exists():

        raise FileNotFoundError(
            file
        )


OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 4. LOAD TEST WINDOWS
# ============================================================

sequences = np.load(
    SEQUENCE_FILE,
    allow_pickle=True
)

X_test_norm = np.asarray(
    sequences["X_test"],
    dtype=np.float64
)

y_test_norm = np.asarray(
    sequences["y_test"],
    dtype=np.float64
)


prediction_archive = np.load(
    PREDICTION_FILE,
    allow_pickle=True
)


transformer_prediction = np.asarray(
    prediction_archive[
        "predictions_original"
    ],
    dtype=np.float64
)


actual = np.asarray(
    prediction_archive[
        "targets_original"
    ],
    dtype=np.float64
)


print(
    f"X_test               : {X_test_norm.shape}"
)

print(
    f"Targets              : {actual.shape}"
)

print(
    f"Transformer forecasts: {transformer_prediction.shape}"
)


# ============================================================
# 5. LOAD SCALER
# ============================================================

scaler = pd.read_csv(
    SCALER_FILE
)

scaler = scaler.set_index(
    "target"
)


def inverse_feature(
    values,
    feature,
):

    minimum = float(
        scaler.loc[
            feature,
            "train_min"
        ]
    )

    maximum = float(
        scaler.loc[
            feature,
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
# 6. INVERSE-SCALE INPUT HISTORY
# ============================================================

X_test = np.empty_like(
    X_test_norm,
    dtype=np.float64
)


for feature_index, feature_name in enumerate(
    FEATURE_NAMES
):

    X_test[
        :,
        :,
        feature_index
    ] = inverse_feature(

        X_test_norm[
            :,
            :,
            feature_index
        ],

        feature_name,
    )


# ============================================================
# 7. PERSISTENCE BASELINE
# ============================================================
#
# Last observed value is repeated for all 24 forecast hours.
# ============================================================

last_observation = X_test[
    :,
    -1,
    :
]


persistence_prediction = np.repeat(

    last_observation[
        :,
        None,
        :
    ],

    repeats=24,

    axis=1,
)
# ============================================================
# 8. DAILY SEASONAL BASELINE
# ============================================================
seasonal_prediction = X_test[
    :,
    -24:,
    :
].copy()
if seasonal_prediction.shape != actual.shape:

    raise RuntimeError(
        "Seasonal prediction shape mismatch."
    )
# ============================================================
# 9. PHYSICALLY CLIPPED TRANSFORMER
# ============================================================

transformer_clipped = (
    transformer_prediction.copy()
)
# PV cannot be negative

transformer_clipped[
    :,
    :,
    0
] = np.clip(

    transformer_clipped[
        :,
        :,
        0
    ],

    0.0,

    None,
)


# Load cannot be negative

transformer_clipped[
    :,
    :,
    1
] = np.clip(

    transformer_clipped[
        :,
        :,
        1
    ],

    0.0,

    None,
)


# EV charging demand cannot be negative
# in the current forecasting target definition.

transformer_clipped[
    :,
    :,
    2
] = np.clip(

    transformer_clipped[
        :,
        :,
        2
    ],

    0.0,

    None,
)


# ============================================================
# 10. MODELS TO COMPARE
# ============================================================

models = {

    "Transformer_raw":
        transformer_prediction,

    "Transformer_clipped":
        transformer_clipped,

    "Persistence":
        persistence_prediction,

    "Daily_seasonal":
        seasonal_prediction,
}
# ============================================================
# 11. EVALUATE
# ============================================================

section(
    "BASELINE COMPARISON"
)
rows = []
for model_name, prediction in models.items():

    print()

    print(
        model_name
    )

    print(
        "-" * 80
    )

    for feature_index, feature_name in enumerate(
        FEATURE_NAMES
    ):

        target_values = actual[
            :,
            :,
            feature_index
        ].reshape(-1)


        prediction_values = prediction[
            :,
            :,
            feature_index
        ].reshape(-1)


        metric_mae = mae(
            target_values,
            prediction_values
        )

        metric_rmse = rmse(
            target_values,
            prediction_values
        )

        metric_r2 = r2(
            target_values,
            prediction_values
        )

        metric_nmae = nmae(
            target_values,
            prediction_values
        )

        metric_nrmse = nrmse(
            target_values,
            prediction_values
        )


        rows.append(
            {
                "model":
                    model_name,

                "feature":
                    feature_name,

                "MAE":
                    metric_mae,

                "RMSE":
                    metric_rmse,

                "R2":
                    metric_r2,

                "NMAE_percent":
                    metric_nmae,

                "NRMSE_percent":
                    metric_nrmse,
            }
        )


        print(
            f"{feature_name:22s} | "
            f"MAE={metric_mae:.8f} | "
            f"RMSE={metric_rmse:.8f} | "
            f"R2={metric_r2:.6f}"
        )


# ============================================================
# 12. SAVE
# ============================================================

comparison = pd.DataFrame(
    rows
)


comparison.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# 13. TRANSFORMER IMPROVEMENT VS DAILY SEASONAL
# ============================================================

section(
    "TRANSFORMER IMPROVEMENT VS DAILY-SEASONAL BASELINE"
)


for feature_name in FEATURE_NAMES:

    transformer_row = comparison[
        (
            comparison["model"]
            == "Transformer_clipped"
        )
        &
        (
            comparison["feature"]
            == feature_name
        )
    ].iloc[0]


    baseline_row = comparison[
        (
            comparison["model"]
            == "Daily_seasonal"
        )
        &
        (
            comparison["feature"]
            == feature_name
        )
    ].iloc[0]


    baseline_mae = float(
        baseline_row[
            "MAE"
        ]
    )


    transformer_mae = float(
        transformer_row[
            "MAE"
        ]
    )


    improvement = (

        (
            baseline_mae
            - transformer_mae
        )

        /

        baseline_mae

        * 100.0

        if baseline_mae > EPSILON

        else np.nan
    )


    print()

    print(
        feature_name
    )

    print(
        f"  Daily seasonal MAE : "
        f"{baseline_mae:.8f}"
    )

    print(
        f"  Transformer MAE    : "
        f"{transformer_mae:.8f}"
    )

    print(
        f"  MAE improvement    : "
        f"{improvement:.3f}%"
    )


# ============================================================
# 14. NEGATIVE OUTPUT AFTER CLIPPING
# ============================================================

section(
    "PHYSICAL OUTPUT CHECK AFTER CLIPPING"
)


for feature_index, feature_name in enumerate(
    FEATURE_NAMES
):

    negative = int(

        np.sum(

            transformer_clipped[
                :,
                :,
                feature_index
            ]

            < 0.0
        )
    )


    print(
        f"{feature_name:22s} : "
        f"{negative:,} negative values"
    )


# ============================================================
# 15. COMPLETE
# ============================================================

section(
    "STEP 7C COMPLETE"
)


print(
    f"Comparison saved:\n"
    f"{OUTPUT_FILE}"
)

print()

print(
    "Compared:"
)

print(
    "  [OK] Raw Transformer"
)

print(
    "  [OK] Physically clipped Transformer"
)

print(
    "  [OK] Persistence baseline"
)

print(
    "  [OK] Daily seasonal baseline"
)

print()

print(
    "No neural-network retraining was performed."
)

print()

print(
    "STEP 7C COMPLETE."
)
