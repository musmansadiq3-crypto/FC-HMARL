from pathlib import Path
import json
import numpy as np
import pandas as pd
# ============================================================
# 1. CONFIGURATION
# ============================================================
PROJECT_ROOT = Path(
    r"D:\Molvi paper review\FC_HMARL"
)

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "FC_HMARL_Unified_Forecasting_8760.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "forecasting"
)

CLEAN_FILE = (
    OUTPUT_DIR
    / "FC_HMARL_Forecasting_Clean.csv"
)

TRAIN_FILE = (
    OUTPUT_DIR
    / "FC_HMARL_Forecasting_Train.csv"
)

VAL_FILE = (
    OUTPUT_DIR
    / "FC_HMARL_Forecasting_Validation.csv"
)

TEST_FILE = (
    OUTPUT_DIR
    / "FC_HMARL_Forecasting_Test.csv"
)

SCALER_FILE = (
    OUTPUT_DIR
    / "forecasting_scaler.csv"
)

SEQUENCE_FILE = (
    OUTPUT_DIR
    / "forecasting_sequences.npz"
)

METADATA_FILE = (
    OUTPUT_DIR
    / "forecasting_metadata.txt"
)


# ============================================================
# 2. FORECASTING PARAMETERS
# ============================================================

INPUT_WINDOW = 168
FORECAST_HORIZON = 24

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

THREE_SIGMA = 3.0

PV_REFERENCE_CAPACITY_KW = 1.0
PV_STC_IRRADIANCE_W_M2 = 1000.0


TARGET_COLUMNS = [
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


def subsection(title):

    print()
    print(title)
    print("-" * 80)


def ensure_exists(path):

    if not path.exists():

        raise FileNotFoundError(
            f"Required input file not found:\n{path}"
        )


# ============================================================
# 4. START
# ============================================================

section(
    "FC-HMARL - FINAL FORECASTING DATA PREPARATION"
)

ensure_exists(
    INPUT_FILE
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

print(
    f"Input:\n{INPUT_FILE}"
)

print(
    f"\nOutput directory:\n{OUTPUT_DIR}"
)


# ============================================================
# 5. LOAD UNIFIED DATA
# ============================================================

subsection(
    "LOADING UNIFIED DATASET"
)

df = pd.read_csv(
    INPUT_FILE
)

print(
    f"Rows loaded    : {len(df):,}"
)

print(
    f"Columns loaded : {len(df.columns):,}"
)


if len(df) != 8760:

    raise RuntimeError(
        f"Expected 8760 rows, found {len(df):,}."
    )


# ============================================================
# 6. PARSE TIMESTAMP
# ============================================================

df["timestamp"] = pd.to_datetime(
    df["timestamp"],
    errors="coerce"
)

if df["timestamp"].isna().any():

    raise RuntimeError(
        "Invalid timestamp values were found."
    )


if df["timestamp"].duplicated().any():

    raise RuntimeError(
        "Duplicate timestamps were found."
    )


# ============================================================
# 7. CONVERT ORIGINAL TARGETS TO NUMERIC
# ============================================================

for column in [
    "pv",
    "load",
    "ev",
    "price",
]:

    df[column] = pd.to_numeric(
        df[column],
        errors="coerce"
    )


# ============================================================
# 8. PV IRRADIANCE -> REFERENCE PV POWER
# ============================================================

section(
    "PV POWER CONVERSION"
)

print(
    "Source PV-related variable : GHI"
)

print(
    f"Reference PV capacity       : "
    f"{PV_REFERENCE_CAPACITY_KW:.3f} kW"
)

print(
    f"STC irradiance              : "
    f"{PV_STC_IRRADIANCE_W_M2:.1f} W/m2"
)

print(
    "Conversion:"
)

print(
    "P_pv = P_rated * clip(GHI / 1000, 0, 1)"
)


ghi = df["pv"].to_numpy(
    dtype=float
)

irradiance_ratio = np.clip(
    ghi / PV_STC_IRRADIANCE_W_M2,
    0.0,
    1.0
)

df["pv_power_kw_raw"] = (
    PV_REFERENCE_CAPACITY_KW
    * irradiance_ratio
)


print()

print(
    f"PV raw minimum : "
    f"{df['pv_power_kw_raw'].min():.6f} kW"
)

print(
    f"PV raw mean    : "
    f"{df['pv_power_kw_raw'].mean():.6f} kW"
)

print(
    f"PV raw maximum : "
    f"{df['pv_power_kw_raw'].max():.6f} kW"
)


# ============================================================
# 9. BUILD CONSISTENT FOUR-TARGET TABLE
# ============================================================

df["load_kw_raw"] = (
    df["load"]
)

df["ev_power_kw_raw"] = (
    df["ev"]
)

df["price_usd_per_kwh_raw"] = (
    df["price"]
)


RAW_TARGETS = {
    "pv_power_kw": "pv_power_kw_raw",
    "load_kw": "load_kw_raw",
    "ev_power_kw": "ev_power_kw_raw",
    "price_usd_per_kwh": "price_usd_per_kwh_raw",
}


# ============================================================
# 10. THREE-SIGMA FILTERING
# ============================================================

section(
    "THREE-SIGMA FILTERING"
)

sigma_records = []


for final_name, raw_name in RAW_TARGETS.items():

    values = df[raw_name]

    mean_value = values.mean()
    std_value = values.std()

    lower_bound = (
        mean_value
        - THREE_SIGMA * std_value
    )

    upper_bound = (
        mean_value
        + THREE_SIGMA * std_value
    )


    # --------------------------------------------
    # Physical lower bounds
    # --------------------------------------------

    if final_name in [
        "pv_power_kw",
        "load_kw",
        "ev_power_kw",
    ]:

        effective_lower = max(
            0.0,
            lower_bound
        )

    else:

        effective_lower = lower_bound


    flag = (
        (values < effective_lower)
        |
        (values > upper_bound)
    )


    flag_column = (
        f"{final_name}_three_sigma_flag"
    )


    df[flag_column] = (
        flag.astype(int)
    )


    filtered_column = (
        f"{final_name}_filtered"
    )


    df[filtered_column] = (
        values.mask(flag)
    )


    flagged_count = int(
        flag.sum()
    )


    sigma_records.append(
        {
            "target": final_name,
            "mean": mean_value,
            "std": std_value,
            "lower": effective_lower,
            "upper": upper_bound,
            "flagged": flagged_count,
        }
    )


    print()

    print(
        f"{final_name}"
    )

    print(
        f"  Mean       : {mean_value:.8f}"
    )

    print(
        f"  Std        : {std_value:.8f}"
    )

    print(
        f"  Lower      : {effective_lower:.8f}"
    )

    print(
        f"  Upper      : {upper_bound:.8f}"
    )

    print(
        f"  Flagged    : {flagged_count:,}"
    )


# ============================================================
# 11. INTERPOLATION
# ============================================================

section(
    "LINEAR INTERPOLATION"
)


for final_name in TARGET_COLUMNS:

    filtered_column = (
        f"{final_name}_filtered"
    )

    missing_before = int(
        df[filtered_column]
        .isna()
        .sum()
    )


    # --------------------------------------------------------
    # Chronological linear interpolation
    # --------------------------------------------------------

    interpolated = (
        df[filtered_column]
        .interpolate(
            method="linear",
            limit_direction="both"
        )
    )


    # Extra edge protection
    interpolated = (
        interpolated
        .ffill()
        .bfill()
    )


    df[final_name] = interpolated


    missing_after = int(
        df[final_name]
        .isna()
        .sum()
    )


    print(
        f"{final_name:20s} | "
        f"before={missing_before:4d} | "
        f"after={missing_after:4d}"
    )


# ============================================================
# 12. FINAL CLEAN TARGET CHECK
# ============================================================

section(
    "CLEAN TARGET VALIDATION"
)


for column in TARGET_COLUMNS:

    print()

    print(
        column
    )

    print(
        df[column]
        .describe()
        .to_string()
    )


    if df[column].isna().any():

        raise RuntimeError(
            f"{column} still contains missing data."
        )


# ============================================================
# 13. CHRONOLOGICAL SPLIT
# ============================================================

section(
    "CHRONOLOGICAL 70/15/15 SPLIT"
)


n_total = len(df)

train_end = int(
    np.floor(
        TRAIN_RATIO * n_total
    )
)

val_length = int(
    np.floor(
        VAL_RATIO * n_total
    )
)

val_end = (
    train_end
    + val_length
)

test_length = (
    n_total
    - val_end
)


print(
    f"Total rows      : {n_total:,}"
)

print(
    f"Training rows   : {train_end:,}"
)

print(
    f"Validation rows : {val_length:,}"
)

print(
    f"Testing rows    : {test_length:,}"
)

print()

print(
    f"Training       : "
    f"{df.iloc[0]['timestamp']} "
    f"to "
    f"{df.iloc[train_end - 1]['timestamp']}"
)

print(
    f"Validation     : "
    f"{df.iloc[train_end]['timestamp']} "
    f"to "
    f"{df.iloc[val_end - 1]['timestamp']}"
)

print(
    f"Testing        : "
    f"{df.iloc[val_end]['timestamp']} "
    f"to "
    f"{df.iloc[-1]['timestamp']}"
)


# ============================================================
# 14. FIT MIN-MAX SCALER ON TRAINING DATA ONLY
# ============================================================

section(
    "TRAINING-ONLY MIN-MAX SCALING"
)


scaler_rows = []


for column in TARGET_COLUMNS:

    train_values = (
        df.loc[
            0:train_end - 1,
            column
        ]
    )

    minimum = float(
        train_values.min()
    )

    maximum = float(
        train_values.max()
    )

    value_range = (
        maximum - minimum
    )


    if value_range == 0:

        raise RuntimeError(
            f"Zero training range for {column}."
        )


    scaler_rows.append(
        {
            "target": column,
            "train_min": minimum,
            "train_max": maximum,
            "train_range": value_range,
        }
    )


    normalized_column = (
        f"{column}_norm"
    )


    df[normalized_column] = (
        (
            df[column]
            - minimum
        )
        /
        value_range
    )


    print(
        f"{column:20s} | "
        f"min={minimum:.8f} | "
        f"max={maximum:.8f}"
    )


scaler_df = pd.DataFrame(
    scaler_rows
)


# ============================================================
# 15. NORMALIZATION DIAGNOSTIC
# ============================================================

subsection(
    "NORMALIZED DATA RANGE CHECK"
)


NORMALIZED_TARGETS = [
    f"{column}_norm"
    for column in TARGET_COLUMNS
]


for column in NORMALIZED_TARGETS:

    print(
        f"{column:25s} | "
        f"all_min={df[column].min():.6f} | "
        f"all_max={df[column].max():.6f}"
    )


print()

print(
    "Values outside [0,1] in validation/test are permitted "
    "when they exceed the training-set range."
)

print(
    "This avoids future-information leakage."
)


# ============================================================
# 16. SPLIT LABELS
# ============================================================

df["split"] = "test"

df.loc[
    0:train_end - 1,
    "split"
] = "train"

df.loc[
    train_end:val_end - 1,
    "split"
] = "validation"


# ============================================================
# 17. SAVE CLEAN FULL DATASET
# ============================================================

section(
    "SAVING CLEAN AND NORMALIZED DATASETS"
)


df.to_csv(
    CLEAN_FILE,
    index=False
)


train_df = (
    df.iloc[
        :train_end
    ]
    .copy()
)

val_df = (
    df.iloc[
        train_end:val_end
    ]
    .copy()
)

test_df = (
    df.iloc[
        val_end:
    ]
    .copy()
)


train_df.to_csv(
    TRAIN_FILE,
    index=False
)

val_df.to_csv(
    VAL_FILE,
    index=False
)

test_df.to_csv(
    TEST_FILE,
    index=False
)

scaler_df.to_csv(
    SCALER_FILE,
    index=False
)


print(
    f"Clean dataset:\n{CLEAN_FILE}"
)

print()

print(
    f"Training dataset:\n{TRAIN_FILE}"
)

print()

print(
    f"Validation dataset:\n{VAL_FILE}"
)

print()

print(
    f"Test dataset:\n{TEST_FILE}"
)

print()

print(
    f"Scaler:\n{SCALER_FILE}"
)


# ============================================================
# 18. CREATE MULTIVARIATE ARRAY
# ============================================================

normalized_matrix = (
    df[
        NORMALIZED_TARGETS
    ]
    .to_numpy(
        dtype=np.float32
    )
)


# ============================================================
# 19. WINDOW CREATION FUNCTION
# ============================================================

def make_windows(
    data,
    target_start,
    target_end,
    input_window,
    horizon,
):
    """
    Build windows such that all forecast TARGET hours belong
    to the requested split.

    Historical input is allowed to come from the immediately
    preceding split. This is correct for chronological
    forecasting because prior observations are available
    when forecasting the future split.
    """

    X = []
    Y = []
    forecast_start_indices = []


    first_forecast_start = max(
        target_start,
        input_window
    )


    last_forecast_start = (
        target_end
        - horizon
    )


    for forecast_start in range(
        first_forecast_start,
        last_forecast_start + 1
    ):

        history_start = (
            forecast_start
            - input_window
        )

        history_end = (
            forecast_start
        )

        forecast_end = (
            forecast_start
            + horizon
        )


        x = data[
            history_start:history_end
        ]

        y = data[
            forecast_start:forecast_end
        ]


        if x.shape != (
            input_window,
            data.shape[1]
        ):

            continue


        if y.shape != (
            horizon,
            data.shape[1]
        ):

            continue


        X.append(
            x
        )

        Y.append(
            y
        )

        forecast_start_indices.append(
            forecast_start
        )


    return (
        np.asarray(
            X,
            dtype=np.float32
        ),
        np.asarray(
            Y,
            dtype=np.float32
        ),
        np.asarray(
            forecast_start_indices,
            dtype=np.int64
        ),
    )


# ============================================================
# 20. BUILD TRAINING SEQUENCES
# ============================================================

section(
    "CREATING 168-HOUR INPUT / 24-HOUR FORECAST SEQUENCES"
)


X_train, y_train, idx_train = make_windows(
    data=normalized_matrix,
    target_start=0,
    target_end=train_end,
    input_window=INPUT_WINDOW,
    horizon=FORECAST_HORIZON,
)


# ============================================================
# 21. BUILD VALIDATION SEQUENCES
# ============================================================

X_val, y_val, idx_val = make_windows(
    data=normalized_matrix,
    target_start=train_end,
    target_end=val_end,
    input_window=INPUT_WINDOW,
    horizon=FORECAST_HORIZON,
)


# ============================================================
# 22. BUILD TEST SEQUENCES
# ============================================================

X_test, y_test, idx_test = make_windows(
    data=normalized_matrix,
    target_start=val_end,
    target_end=n_total,
    input_window=INPUT_WINDOW,
    horizon=FORECAST_HORIZON,
)


# ============================================================
# 23. DISPLAY SEQUENCE SHAPES
# ============================================================

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
# 24. HARD SHAPE CHECK
# ============================================================

expected_x_shape = (
    INPUT_WINDOW,
    len(TARGET_COLUMNS)
)

expected_y_shape = (
    FORECAST_HORIZON,
    len(TARGET_COLUMNS)
)


for name, array in [
    ("X_train", X_train),
    ("X_val", X_val),
    ("X_test", X_test),
]:

    if array.ndim != 3:

        raise RuntimeError(
            f"{name} is not 3-dimensional."
        )


    if array.shape[1:] != expected_x_shape:

        raise RuntimeError(
            f"Incorrect {name} shape: "
            f"{array.shape}"
        )


for name, array in [
    ("y_train", y_train),
    ("y_val", y_val),
    ("y_test", y_test),
]:

    if array.ndim != 3:

        raise RuntimeError(
            f"{name} is not 3-dimensional."
        )


    if array.shape[1:] != expected_y_shape:

        raise RuntimeError(
            f"Incorrect {name} shape: "
            f"{array.shape}"
        )


# ============================================================
# 25. SAVE NUMPY SEQUENCES
# ============================================================

np.savez_compressed(
    SEQUENCE_FILE,

    X_train=X_train,
    y_train=y_train,

    X_val=X_val,
    y_val=y_val,

    X_test=X_test,
    y_test=y_test,

    idx_train=idx_train,
    idx_val=idx_val,
    idx_test=idx_test,

    feature_names=np.asarray(
        TARGET_COLUMNS
    ),
)


print()

print(
    f"Sequence archive:\n{SEQUENCE_FILE}"
)


# ============================================================
# 26. FORECAST START TIMESTAMP CHECK
# ============================================================

subsection(
    "SEQUENCE TIMESTAMP VALIDATION"
)


def timestamp_for_index(index):

    return df.iloc[
        int(index)
    ][
        "timestamp"
    ]


print(
    "First training forecast : "
    f"{timestamp_for_index(idx_train[0])}"
)

print(
    "Last training forecast  : "
    f"{timestamp_for_index(idx_train[-1])}"
)

print()

print(
    "First validation forecast: "
    f"{timestamp_for_index(idx_val[0])}"
)

print(
    "Last validation forecast : "
    f"{timestamp_for_index(idx_val[-1])}"
)

print()

print(
    "First test forecast       : "
    f"{timestamp_for_index(idx_test[0])}"
)

print(
    "Last test forecast        : "
    f"{timestamp_for_index(idx_test[-1])}"
)


# ============================================================
# 27. METADATA
# ============================================================

metadata = f"""
FC-HMARL FORECASTING DATASET METADATA
=====================================

Input source:
{INPUT_FILE}

Total hours:
{n_total}

Targets:
1. pv_power_kw
2. load_kw
3. ev_power_kw
4. price_usd_per_kwh

PV conversion:
Reference capacity = {PV_REFERENCE_CAPACITY_KW} kW
STC irradiance = {PV_STC_IRRADIANCE_W_M2} W/m2
P_pv = P_rated * clip(GHI / STC_irradiance, 0, 1)

IMPORTANT:
The GHI-to-PV conversion is a reconstruction choice and not
claimed to be the exact lost implementation.

Three-sigma threshold:
{THREE_SIGMA}

Filtering:
Values outside mean +/- 3 sigma were flagged and replaced
temporarily with NaN.

Missing/outlier reconstruction:
Chronological linear interpolation, followed by edge
forward/backward filling if required.

Chronological split:
Training   = {train_end} hours
Validation = {val_length} hours
Testing    = {test_length} hours

Ratios:
Training   = {TRAIN_RATIO}
Validation = {VAL_RATIO}
Testing    = {TEST_RATIO}

Scaling:
Min-max normalization fitted on TRAINING DATA ONLY.

Input window:
{INPUT_WINDOW} hours

Forecast horizon:
{FORECAST_HORIZON} hours

Number of features:
{len(TARGET_COLUMNS)}

Feature ordering:
{TARGET_COLUMNS}

Training sequences:
{len(X_train)}

Validation sequences:
{len(X_val)}

Test sequences:
{len(X_test)}

Source provenance:
PV:
NSRDB 2022.

Load:
Pecan Street source observations 2014-2018, representative
profile mapped to common simulation calendar.

EV:
ACN-Data Caltech 2018-2020, representative annual profile.

Price:
PJM-RTO Day-Ahead 2019 annual chronological profile.

The common calendar is an alignment/simulation calendar.
It does NOT mean that every source was physically observed
during 2022.
""".strip()


METADATA_FILE.write_text(
    metadata,
    encoding="utf-8"
)


print()

print(
    f"Metadata:\n{METADATA_FILE}"
)


# ============================================================
# 28. FINAL VALIDATION SUMMARY
# ============================================================

section(
    "FINAL FORECASTING DATA VALIDATION"
)


print(
    f"Total hourly rows          : {len(df):,}"
)

print(
    f"Training hourly rows       : {len(train_df):,}"
)

print(
    f"Validation hourly rows     : {len(val_df):,}"
)

print(
    f"Testing hourly rows        : {len(test_df):,}"
)

print()

print(
    f"Input sequence length      : {INPUT_WINDOW}"
)

print(
    f"Forecast horizon           : {FORECAST_HORIZON}"
)

print(
    f"Forecast variables         : {len(TARGET_COLUMNS)}"
)

print()

print(
    f"Training sequences         : {len(X_train):,}"
)

print(
    f"Validation sequences       : {len(X_val):,}"
)

print(
    f"Testing sequences          : {len(X_test):,}"
)

print()

print(
    f"X_train shape              : {X_train.shape}"
)

print(
    f"y_train shape              : {y_train.shape}"
)

print()

print(
    f"X_validation shape         : {X_val.shape}"
)

print(
    f"y_validation shape         : {y_val.shape}"
)

print()

print(
    f"X_test shape               : {X_test.shape}"
)

print(
    f"y_test shape               : {y_test.shape}"
)


# ============================================================
# 29. COMPLETE
# ============================================================

section(
    "STEP 6B COMPLETE"
)


print(
    "Forecasting preprocessing is complete."
)

print()

print(
    "Applied:"
)

print(
    "  [OK] GHI converted to reference PV electrical power"
)

print(
    "  [OK] Three-sigma filtering"
)

print(
    "  [OK] Linear interpolation"
)

print(
    "  [OK] Chronological 70/15/15 split"
)

print(
    "  [OK] Training-only min-max fitting"
)

print(
    "  [OK] 168-hour historical input"
)

print(
    "  [OK] 24-hour forecasting horizon"
)

print(
    "  [OK] Four forecasting variables"
)

print()

print(
    "No future validation/test values were used "
    "to fit the min-max scaler."
)

print()

print(
    "STEP 6B COMPLETE."
)
