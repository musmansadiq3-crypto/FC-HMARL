# ============================================================
# FC-HMARL
# STEP 7D - DAILY VS WEEKLY SEASONAL DIAGNOSTIC
# ============================================================
#
# Purpose:
# Compare two simple seasonal forecasting baselines:
#
#   Daily seasonal:
#       future hour h = same hour from previous day
#
#   Weekly seasonal:
#       future hour h = same hour from previous week
#
# Input window = 168 hours
# Forecast horizon = 24 hours
#
# No model retraining is performed.
#
# ============================================================

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
    / "daily_weekly_seasonal_comparison.csv"
)


FEATURE_NAMES = [
    "pv_power_kw",
    "load_kw",
    "ev_power_kw",
    "price_usd_per_kwh",
]

EPS = 1e-8


# ============================================================
# 2. HELPER FUNCTIONS
# ============================================================

def section(title):
    print()
    print("=" * 80)
    print(title)
    print("=" * 80)


def mae(actual, predicted):
    return float(
        np.mean(
            np.abs(predicted - actual)
        )
    )


def rmse(actual, predicted):
    return float(
        np.sqrt(
            np.mean(
                (predicted - actual) ** 2
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
        (actual - predicted) ** 2
    )

    ss_tot = np.sum(
        (actual - np.mean(actual)) ** 2
    )

    if ss_tot <= EPS:
        return np.nan

    return float(
        1.0 - ss_res / ss_tot
    )


def nmae(actual, predicted):

    denominator = np.mean(
        np.abs(actual)
    )

    if denominator <= EPS:
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

    if denominator <= EPS:
        return np.nan

    return float(
        100.0
        * rmse(actual, predicted)
        / denominator
    )


# ============================================================
# 3. LOAD DATA
# ============================================================

section(
    "FC-HMARL - STEP 7D DAILY VS WEEKLY SEASONAL DIAGNOSTIC"
)


for file_path in [
    SEQUENCE_FILE,
    SCALER_FILE,
    PREDICTION_FILE,
]:
    if not file_path.exists():
        raise FileNotFoundError(
            file_path
        )


OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


sequences = np.load(
    SEQUENCE_FILE,
    allow_pickle=True
)

X_test_norm = np.asarray(
    sequences["X_test"],
    dtype=np.float64
)


prediction_archive = np.load(
    PREDICTION_FILE,
    allow_pickle=True
)

targets = np.asarray(
    prediction_archive[
        "targets_original"
    ],
    dtype=np.float64
)


print(
    f"X_test shape  : {X_test_norm.shape}"
)

print(
    f"Targets shape : {targets.shape}"
)


if X_test_norm.shape[1:] != (168, 4):
    raise RuntimeError(
        f"Expected X_test (*,168,4), "
        f"received {X_test_norm.shape}"
    )


if targets.shape[1:] != (24, 4):
    raise RuntimeError(
        f"Expected targets (*,24,4), "
        f"received {targets.shape}"
    )


# ============================================================
# 4. LOAD SCALER
# ============================================================

scaler = pd.read_csv(
    SCALER_FILE
)

scaler = scaler.set_index(
    "target"
)


def inverse_feature(
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
        * (maximum - minimum)
        + minimum
    )


# ============================================================
# 5. INVERSE-SCALE HISTORY
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
# 6. DAILY SEASONAL BASELINE
# ============================================================
#
# Final 24 hours in the history:
#
#   t-24 ... t-1
#
# are used as the 24-hour forecast.
#
# ============================================================

daily_prediction = X_test[
    :,
    -24:,
    :
].copy()


# ============================================================
# 7. WEEKLY SEASONAL BASELINE
# ============================================================
#
# History contains exactly 168 hours.
#
# For a 24-hour forecast, the corresponding hours from
# one week earlier are the FIRST 24 hours of the input:
#
#   t-168 ... t-145
#
# ============================================================

weekly_prediction = X_test[
    :,
    0:24,
    :
].copy()


print()

print(
    f"Daily prediction shape  : "
    f"{daily_prediction.shape}"
)

print(
    f"Weekly prediction shape : "
    f"{weekly_prediction.shape}"
)


if daily_prediction.shape != targets.shape:
    raise RuntimeError(
        "Daily baseline shape mismatch."
    )


if weekly_prediction.shape != targets.shape:
    raise RuntimeError(
        "Weekly baseline shape mismatch."
    )


# ============================================================
# 8. EVALUATE BOTH BASELINES
# ============================================================

section(
    "DAILY VS WEEKLY RESULTS"
)


methods = {
    "Daily_seasonal":
        daily_prediction,

    "Weekly_seasonal":
        weekly_prediction,
}


rows = []


for method_name, prediction in methods.items():

    print()
    print(method_name)
    print("-" * 80)


    for feature_index, feature_name in enumerate(
        FEATURE_NAMES
    ):

        actual_values = targets[
            :,
            :,
            feature_index
        ].reshape(-1)

        predicted_values = prediction[
            :,
            :,
            feature_index
        ].reshape(-1)


        metric_mae = mae(
            actual_values,
            predicted_values
        )

        metric_rmse = rmse(
            actual_values,
            predicted_values
        )

        metric_r2 = r2(
            actual_values,
            predicted_values
        )

        metric_nmae = nmae(
            actual_values,
            predicted_values
        )

        metric_nrmse = nrmse(
            actual_values,
            predicted_values
        )


        rows.append(
            {
                "method":
                    method_name,

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


results = pd.DataFrame(
    rows
)


# ============================================================
# 9. DETERMINE BEST SEASONAL BASELINE
# ============================================================

section(
    "BEST SEASONAL BASELINE BY VARIABLE"
)


summary_rows = []


for feature_name in FEATURE_NAMES:

    feature_results = results[
        results["feature"]
        == feature_name
    ].copy()


    best_index = (
        feature_results["MAE"]
        .idxmin()
    )


    best_row = feature_results.loc[
        best_index
    ]


    daily_row = feature_results[
        feature_results["method"]
        == "Daily_seasonal"
    ].iloc[0]


    weekly_row = feature_results[
        feature_results["method"]
        == "Weekly_seasonal"
    ].iloc[0]


    daily_mae = float(
        daily_row["MAE"]
    )

    weekly_mae = float(
        weekly_row["MAE"]
    )


    if daily_mae > EPS:

        weekly_vs_daily = (
            (daily_mae - weekly_mae)
            / daily_mae
            * 100.0
        )

    else:

        weekly_vs_daily = np.nan


    summary_rows.append(
        {
            "feature":
                feature_name,

            "daily_MAE":
                daily_mae,

            "weekly_MAE":
                weekly_mae,

            "weekly_improvement_vs_daily_percent":
                weekly_vs_daily,

            "best_method":
                best_row["method"],

            "best_MAE":
                float(
                    best_row["MAE"]
                ),

            "best_RMSE":
                float(
                    best_row["RMSE"]
                ),

            "best_R2":
                float(
                    best_row["R2"]
                ),
        }
    )


    print()

    print(
        feature_name
    )

    print(
        f"  Daily MAE  : "
        f"{daily_mae:.8f}"
    )

    print(
        f"  Weekly MAE : "
        f"{weekly_mae:.8f}"
    )

    print(
        f"  Weekly improvement vs daily : "
        f"{weekly_vs_daily:.3f}%"
    )

    print(
        f"  Best seasonal baseline      : "
        f"{best_row['method']}"
    )


summary = pd.DataFrame(
    summary_rows
)


# ============================================================
# 10. HORIZON-WISE COMPARISON
# ============================================================

section(
    "HORIZON-WISE DAILY VS WEEKLY COMPARISON"
)


horizon_rows = []


for horizon_index in range(24):

    forecast_hour = (
        horizon_index + 1
    )


    for feature_index, feature_name in enumerate(
        FEATURE_NAMES
    ):

        actual_values = targets[
            :,
            horizon_index,
            feature_index
        ]


        daily_values = daily_prediction[
            :,
            horizon_index,
            feature_index
        ]


        weekly_values = weekly_prediction[
            :,
            horizon_index,
            feature_index
        ]


        daily_mae = mae(
            actual_values,
            daily_values
        )

        weekly_mae = mae(
            actual_values,
            weekly_values
        )


        best_method = (
            "Daily_seasonal"
            if daily_mae <= weekly_mae
            else "Weekly_seasonal"
        )


        horizon_rows.append(
            {
                "forecast_hour":
                    forecast_hour,

                "feature":
                    feature_name,

                "daily_MAE":
                    daily_mae,

                "weekly_MAE":
                    weekly_mae,

                "best_method":
                    best_method,
            }
        )


horizon_df = pd.DataFrame(
    horizon_rows
)


# ============================================================
# 11. SAVE RESULTS
# ============================================================

results.to_csv(
    OUTPUT_FILE,
    index=False
)


SUMMARY_FILE = (
    OUTPUT_DIR
    / "daily_weekly_seasonal_summary.csv"
)


summary.to_csv(
    SUMMARY_FILE,
    index=False
)


HORIZON_FILE = (
    OUTPUT_DIR
    / "daily_weekly_horizon_comparison.csv"
)


horizon_df.to_csv(
    HORIZON_FILE,
    index=False
)


# ============================================================
# 12. FINAL DECISION TABLE
# ============================================================

section(
    "SEASONAL MODEL DECISION"
)


for _, row in summary.iterrows():

    print(
        f"{row['feature']:22s} -> "
        f"{row['best_method']:18s} | "
        f"MAE={row['best_MAE']:.8f} | "
        f"R2={row['best_R2']:.6f}"
    )


# ============================================================
# 13. OUTPUT FILES
# ============================================================

section(
    "SAVING STEP 7D RESULTS"
)


print(
    f"Full comparison:\n"
    f"{OUTPUT_FILE}"
)

print()

print(
    f"Summary:\n"
    f"{SUMMARY_FILE}"
)

print()

print(
    f"Horizon comparison:\n"
    f"{HORIZON_FILE}"
)


# ============================================================
# 14. COMPLETE
# ============================================================

section(
    "STEP 7D COMPLETE"
)


print(
    "Daily and weekly seasonal baselines "
    "were compared successfully."
)

print()

print(
    "No neural-network training was performed."
)

print()

print(
    "STEP 7D COMPLETE."
)