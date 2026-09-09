from pathlib import Path
import numpy as np
import pandas as pd
# ============================================================
# 1. PROJECT PATHS
# ============================================================
PROJECT_ROOT = Path(
    r"D:\Molvi paper review\FC_HMARL"
)
INPUT_FILE = (
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

OVERALL_METRICS_FILE = (
    OUTPUT_DIR
    / "forecasting_extended_metrics.csv"
)

HORIZON_METRICS_FILE = (
    OUTPUT_DIR
    / "forecasting_horizon_metrics.csv"
)

SPECIAL_METRICS_FILE = (
    OUTPUT_DIR
    / "forecasting_special_metrics.csv"
)

SUMMARY_FILE = (
    OUTPUT_DIR
    / "forecasting_evaluation_summary.txt"
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
# ============================================================
# 3. SPECIAL THRESHOLDS
# ============================================================
PV_DAYLIGHT_THRESHOLD_KW = 0.05
EV_ACTIVE_THRESHOLD_KW = 1.0
EPSILON = 1e-8
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


def subsection(title):

    print()

    print(
        title
    )

    print(
        "-" * 80
    )


# ============================================================
# 5. METRIC FUNCTIONS
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


def r2_score(actual, predicted):

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
        1.0 - ss_res / ss_tot
    )


def mape(actual, predicted):

    actual = np.asarray(
        actual,
        dtype=float
    )

    predicted = np.asarray(
        predicted,
        dtype=float
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


def smape(actual, predicted):

    actual = np.asarray(
        actual,
        dtype=float
    )

    predicted = np.asarray(
        predicted,
        dtype=float
    )

    denominator = (
        np.abs(actual)
        +
        np.abs(predicted)
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
            *
            np.abs(
                predicted[valid]
                - actual[valid]
            )
            /
            denominator[valid]

        )
        * 100.0
    )


def nmae_mean(actual, predicted):

    actual = np.asarray(
        actual,
        dtype=float
    )

    mean_actual = float(
        np.mean(
            np.abs(actual)
        )
    )

    if mean_actual <= EPSILON:

        return np.nan

    return float(
        mae(actual, predicted)
        /
        mean_actual
        *
        100.0
    )


def nrmse_mean(actual, predicted):

    actual = np.asarray(
        actual,
        dtype=float
    )

    mean_actual = float(
        np.mean(
            np.abs(actual)
        )
    )

    if mean_actual <= EPSILON:

        return np.nan

    return float(
        rmse(actual, predicted)
        /
        mean_actual
        *
        100.0
    )


def cvrmse(actual, predicted):

    actual = np.asarray(
        actual,
        dtype=float
    )

    mean_actual = float(
        np.mean(actual)
    )

    if abs(mean_actual) <= EPSILON:

        return np.nan

    return float(
        rmse(actual, predicted)
        /
        abs(mean_actual)
        *
        100.0
    )


def calculate_metrics(
    actual,
    predicted,
):

    actual = np.asarray(
        actual,
        dtype=float
    ).reshape(-1)

    predicted = np.asarray(
        predicted,
        dtype=float
    ).reshape(-1)

    return {

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
            nmae_mean(
                actual,
                predicted
            ),

        "NRMSE_percent":
            nrmse_mean(
                actual,
                predicted
            ),

        "CVRMSE_percent":
            cvrmse(
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

        "actual_mean":
            float(
                np.mean(actual)
            ),

        "actual_std":
            float(
                np.std(actual)
            ),

        "actual_min":
            float(
                np.min(actual)
            ),

        "actual_max":
            float(
                np.max(actual)
            ),

        "prediction_mean":
            float(
                np.mean(predicted)
            ),

        "number_of_values":
            int(
                actual.size
            ),
    }
# ============================================================
# 6. START
# ============================================================

section(
    "FC-HMARL - STEP 7B FORECASTING EVALUATION"
)


if not INPUT_FILE.exists():

    raise FileNotFoundError(
        f"Prediction archive not found:\n"
        f"{INPUT_FILE}"
    )


OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


print(
    f"Prediction archive:\n"
    f"{INPUT_FILE}"
)

print()

print(
    f"Evaluation output:\n"
    f"{OUTPUT_DIR}"
)


# ============================================================
# 7. LOAD PREDICTIONS
# ============================================================

section(
    "LOADING TEST PREDICTIONS"
)


archive = np.load(
    INPUT_FILE,
    allow_pickle=True,
)


required_arrays = [

    "predictions_original",

    "targets_original",
]


for name in required_arrays:

    if name not in archive.files:

        raise RuntimeError(
            f"Archive is missing: {name}"
        )


predictions = np.asarray(
    archive[
        "predictions_original"
    ],
    dtype=np.float64
)

targets = np.asarray(
    archive[
        "targets_original"
    ],
    dtype=np.float64
)


print(
    f"Predictions shape : "
    f"{predictions.shape}"
)

print(
    f"Targets shape     : "
    f"{targets.shape}"
)


expected_shape_tail = (
    24,
    4,
)


if predictions.shape != targets.shape:

    raise RuntimeError(
        "Prediction and target shapes differ."
    )


if predictions.shape[1:] != expected_shape_tail:

    raise RuntimeError(
        "Expected shape (*, 24, 4), "
        f"received {predictions.shape}."
    )


if not np.isfinite(
    predictions
).all():

    raise RuntimeError(
        "Predictions contain NaN or infinity."
    )


if not np.isfinite(
    targets
).all():

    raise RuntimeError(
        "Targets contain NaN or infinity."
    )


print()

print(
    "[OK] Prediction arrays passed validation."
)


# ============================================================
# 8. OVERALL METRICS
# ============================================================

section(
    "OVERALL TEST METRICS"
)


overall_rows = []


for feature_index, feature_name in enumerate(
    FEATURE_NAMES
):

    actual = targets[
        :,
        :,
        feature_index
    ]

    predicted = predictions[
        :,
        :,
        feature_index
    ]


    metrics = calculate_metrics(
        actual,
        predicted
    )


    metrics[
        "feature"
    ] = feature_name


    overall_rows.append(
        metrics
    )


    print()

    print(
        feature_name
    )

    print(
        f"  MAE       : "
        f"{metrics['MAE']:.8f}"
    )

    print(
        f"  RMSE      : "
        f"{metrics['RMSE']:.8f}"
    )

    print(
        f"  R2        : "
        f"{metrics['R2']:.6f}"
    )

    print(
        f"  NMAE      : "
        f"{metrics['NMAE_percent']:.4f}%"
    )

    print(
        f"  NRMSE     : "
        f"{metrics['NRMSE_percent']:.4f}%"
    )

    print(
        f"  MAPE      : "
        f"{metrics['MAPE_percent']:.4f}%"
    )

    print(
        f"  sMAPE     : "
        f"{metrics['sMAPE_percent']:.4f}%"
    )


overall_df = pd.DataFrame(
    overall_rows
)


overall_columns = [

    "feature",

    "MAE",
    "RMSE",
    "MSE",
    "R2",

    "NMAE_percent",
    "NRMSE_percent",
    "CVRMSE_percent",

    "MAPE_percent",
    "sMAPE_percent",

    "actual_mean",
    "actual_std",
    "actual_min",
    "actual_max",

    "prediction_mean",

    "number_of_values",
]


overall_df = overall_df[
    overall_columns
]


overall_df.to_csv(
    OVERALL_METRICS_FILE,
    index=False
)


# ============================================================
# 9. SPECIAL PV EVALUATION
# ============================================================

section(
    "DAYLIGHT / ACTIVE-POWER EVALUATION"
)


special_rows = []


# ------------------------------------------------------------
# PV DAYLIGHT
# ------------------------------------------------------------

pv_actual = targets[
    :,
    :,
    0
]

pv_predicted = predictions[
    :,
    :,
    0
]


pv_mask = (
    pv_actual
    >= PV_DAYLIGHT_THRESHOLD_KW
)


pv_actual_active = (
    pv_actual[
        pv_mask
    ]
)

pv_pred_active = (
    pv_predicted[
        pv_mask
    ]
)


pv_metrics = calculate_metrics(
    pv_actual_active,
    pv_pred_active
)


pv_metrics[
    "evaluation"
] = "PV daylight"

pv_metrics[
    "threshold"
] = (
    PV_DAYLIGHT_THRESHOLD_KW
)


special_rows.append(
    pv_metrics
)


print(
    f"PV daylight threshold : "
    f"{PV_DAYLIGHT_THRESHOLD_KW:.3f} kW"
)

print(
    f"PV daylight values    : "
    f"{pv_actual_active.size:,}"
)

print(
    f"PV daylight MAE       : "
    f"{pv_metrics['MAE']:.8f}"
)

print(
    f"PV daylight RMSE      : "
    f"{pv_metrics['RMSE']:.8f}"
)

print(
    f"PV daylight R2        : "
    f"{pv_metrics['R2']:.6f}"
)

print(
    f"PV daylight MAPE      : "
    f"{pv_metrics['MAPE_percent']:.4f}%"
)

print(
    f"PV daylight sMAPE     : "
    f"{pv_metrics['sMAPE_percent']:.4f}%"
)


# ------------------------------------------------------------
# EV ACTIVE-CHARGING
# ------------------------------------------------------------

ev_actual = targets[
    :,
    :,
    2
]

ev_predicted = predictions[
    :,
    :,
    2
]


ev_mask = (
    ev_actual
    >= EV_ACTIVE_THRESHOLD_KW
)


ev_actual_active = (
    ev_actual[
        ev_mask
    ]
)

ev_pred_active = (
    ev_predicted[
        ev_mask
    ]
)


ev_metrics = calculate_metrics(
    ev_actual_active,
    ev_pred_active
)


ev_metrics[
    "evaluation"
] = "EV active charging"

ev_metrics[
    "threshold"
] = (
    EV_ACTIVE_THRESHOLD_KW
)


special_rows.append(
    ev_metrics
)


print()

print(
    f"EV active threshold  : "
    f"{EV_ACTIVE_THRESHOLD_KW:.3f} kW"
)

print(
    f"EV active values     : "
    f"{ev_actual_active.size:,}"
)

print(
    f"EV active MAE        : "
    f"{ev_metrics['MAE']:.8f}"
)

print(
    f"EV active RMSE       : "
    f"{ev_metrics['RMSE']:.8f}"
)

print(
    f"EV active R2         : "
    f"{ev_metrics['R2']:.6f}"
)

print(
    f"EV active MAPE       : "
    f"{ev_metrics['MAPE_percent']:.4f}%"
)

print(
    f"EV active sMAPE      : "
    f"{ev_metrics['sMAPE_percent']:.4f}%"
)


special_df = pd.DataFrame(
    special_rows
)


special_df.to_csv(
    SPECIAL_METRICS_FILE,
    index=False
)


# ============================================================
# 10. HORIZON-WISE METRICS
# ============================================================

section(
    "24-HOUR HORIZON-WISE EVALUATION"
)


horizon_rows = []


for horizon_index in range(
    24
):

    horizon_hour = (
        horizon_index + 1
    )


    for feature_index, feature_name in enumerate(
        FEATURE_NAMES
    ):

        actual = targets[
            :,
            horizon_index,
            feature_index
        ]

        predicted = predictions[
            :,
            horizon_index,
            feature_index
        ]


        metrics = calculate_metrics(
            actual,
            predicted
        )


        horizon_rows.append(
            {
                "forecast_hour":
                    horizon_hour,

                "feature":
                    feature_name,

                "MAE":
                    metrics[
                        "MAE"
                    ],

                "RMSE":
                    metrics[
                        "RMSE"
                    ],

                "R2":
                    metrics[
                        "R2"
                    ],

                "NMAE_percent":
                    metrics[
                        "NMAE_percent"
                    ],

                "NRMSE_percent":
                    metrics[
                        "NRMSE_percent"
                    ],

                "MAPE_percent":
                    metrics[
                        "MAPE_percent"
                    ],

                "sMAPE_percent":
                    metrics[
                        "sMAPE_percent"
                    ],
            }
        )


horizon_df = pd.DataFrame(
    horizon_rows
)


horizon_df.to_csv(
    HORIZON_METRICS_FILE,
    index=False
)


# ============================================================
# 11. HORIZON SUMMARY
# ============================================================

subsection(
    "FIRST-HOUR VS 24TH-HOUR FORECAST"
)


for feature_name in FEATURE_NAMES:

    feature_horizon = horizon_df[
        horizon_df[
            "feature"
        ]
        == feature_name
    ]


    h1 = feature_horizon[
        feature_horizon[
            "forecast_hour"
        ]
        == 1
    ].iloc[0]


    h24 = feature_horizon[
        feature_horizon[
            "forecast_hour"
        ]
        == 24
    ].iloc[0]


    print()

    print(
        feature_name
    )

    print(
        f"  h=1  MAE  : "
        f"{h1['MAE']:.8f}"
    )

    print(
        f"  h=24 MAE  : "
        f"{h24['MAE']:.8f}"
    )

    print(
        f"  h=1  RMSE : "
        f"{h1['RMSE']:.8f}"
    )

    print(
        f"  h=24 RMSE : "
        f"{h24['RMSE']:.8f}"
    )


# ============================================================
# 12. CHECK NEGATIVE PREDICTIONS
# ============================================================

section(
    "PHYSICAL PREDICTION CHECK"
)


for feature_index, feature_name in enumerate(
    FEATURE_NAMES
):

    predicted = predictions[
        :,
        :,
        feature_index
    ]


    negative_count = int(
        np.sum(
            predicted < 0
        )
    )


    print(
        f"{feature_name:22s} "
        f"negative predictions: "
        f"{negative_count:,}"
    )


# ============================================================
# 13. SAVE TEXT SUMMARY
# ============================================================

summary_lines = []


summary_lines.append(
    "FC-HMARL STEP 7B FORECASTING EVALUATION"
)

summary_lines.append(
    "=" * 60
)

summary_lines.append("")

summary_lines.append(
    f"Prediction tensor: {predictions.shape}"
)

summary_lines.append(
    f"Target tensor: {targets.shape}"
)

summary_lines.append("")

summary_lines.append(
    "OVERALL METRICS"
)

summary_lines.append(
    "-" * 60
)


for _, row in overall_df.iterrows():

    summary_lines.append(
        (
            f"{row['feature']}: "
            f"MAE={row['MAE']:.8f}, "
            f"RMSE={row['RMSE']:.8f}, "
            f"R2={row['R2']:.6f}, "
            f"NMAE={row['NMAE_percent']:.4f}%, "
            f"NRMSE={row['NRMSE_percent']:.4f}%, "
            f"MAPE={row['MAPE_percent']:.4f}%, "
            f"sMAPE={row['sMAPE_percent']:.4f}%"
        )
    )


summary_lines.append("")

summary_lines.append(
    "SPECIAL EVALUATION"
)

summary_lines.append(
    "-" * 60
)

summary_lines.append(
    (
        f"PV daylight threshold "
        f"= {PV_DAYLIGHT_THRESHOLD_KW} kW"
    )
)

summary_lines.append(
    (
        f"PV daylight: "
        f"MAE={pv_metrics['MAE']:.8f}, "
        f"RMSE={pv_metrics['RMSE']:.8f}, "
        f"R2={pv_metrics['R2']:.6f}, "
        f"MAPE={pv_metrics['MAPE_percent']:.4f}%"
    )
)

summary_lines.append(
    (
        f"EV active threshold "
        f"= {EV_ACTIVE_THRESHOLD_KW} kW"
    )
)

summary_lines.append(
    (
        f"EV active: "
        f"MAE={ev_metrics['MAE']:.8f}, "
        f"RMSE={ev_metrics['RMSE']:.8f}, "
        f"R2={ev_metrics['R2']:.6f}, "
        f"MAPE={ev_metrics['MAPE_percent']:.4f}%"
    )
)

summary_lines.append("")

summary_lines.append(
    "IMPORTANT INTERPRETATION"
)

summary_lines.append(
    "-" * 60
)

summary_lines.append(
    "Raw MAPE is unreliable when actual PV or EV values "
    "are zero or close to zero."
)

summary_lines.append(
    "Therefore MAE, RMSE, R2, normalized errors, sMAPE, "
    "and active-period metrics must be considered together."
)

summary_lines.append(
    "No forecasting model was retrained in Step 7B."
)


SUMMARY_FILE.write_text(

    "\n".join(
        summary_lines
    ),

    encoding="utf-8"
)


# ============================================================
# 14. OUTPUT FILES
# ============================================================

section(
    "SAVING STEP 7B RESULTS"
)


print(
    f"Overall metrics:\n"
    f"{OVERALL_METRICS_FILE}"
)

print()

print(
    f"Special metrics:\n"
    f"{SPECIAL_METRICS_FILE}"
)

print()

print(
    f"Horizon metrics:\n"
    f"{HORIZON_METRICS_FILE}"
)

print()

print(
    f"Summary:\n"
    f"{SUMMARY_FILE}"
)


# ============================================================
# 15. COMPLETE
# ============================================================

section(
    "STEP 7B COMPLETE"
)


print(
    "Rigorous forecasting evaluation completed."
)

print()

print(
    "Calculated:"
)

print(
    "  [OK] Overall MAE / RMSE / MSE"
)

print(
    "  [OK] R2"
)

print(
    "  [OK] NMAE / NRMSE / CVRMSE"
)

print(
    "  [OK] MAPE / sMAPE"
)

print(
    "  [OK] PV daylight-only metrics"
)

print(
    "  [OK] EV active-charging metrics"
)

print(
    "  [OK] 1-to-24-hour horizon metrics"
)

print(
    "  [OK] Negative-prediction diagnostics"
)

print()

print(
    "No retraining was performed."
)

print()

print(
    "STEP 7B COMPLETE."
)
