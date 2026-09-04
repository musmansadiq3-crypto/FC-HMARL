# ============================================================
# FC-HMARL
# STEP 7E - PREPARE SEASONAL-RESIDUAL FORECASTING DATASET
# ============================================================
#
# Selected seasonal structure from Step 7D:
#
#   PV    -> Daily seasonal baseline (t - 24 h)
#   Load  -> Daily seasonal baseline (t - 24 h)
#   EV    -> Weekly seasonal baseline (t - 168 h)
#   Price -> Daily seasonal baseline (t - 24 h)
#
# Transformer target:
#
#   residual = actual_future - seasonal_baseline
#
# Final future forecast later:
#
#   forecast = seasonal_baseline + predicted_residual
#
# No neural-network training is performed in this step.
#
# ============================================================

from pathlib import Path
import numpy as np


# ============================================================
# 1. PATHS
# ============================================================

PROJECT_ROOT = Path(
    r"D:\Molvi paper review\FC_HMARL"
)

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "forecasting"
    / "forecasting_sequences.npz"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "forecasting"
    / "forecasting_residual_sequences.npz"
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


# ============================================================
# 3. HELPERS
# ============================================================

def section(title):

    print()
    print("=" * 80)
    print(title)
    print("=" * 80)


# ============================================================
# 4. LOAD EXISTING FORECAST WINDOWS
# ============================================================

section(
    "FC-HMARL - STEP 7E RESIDUAL DATASET PREPARATION"
)


if not INPUT_FILE.exists():

    raise FileNotFoundError(
        f"Input sequence file not found:\n{INPUT_FILE}"
    )


data = np.load(
    INPUT_FILE,
    allow_pickle=True
)


required_arrays = [
    "X_train",
    "y_train",
    "X_val",
    "y_val",
    "X_test",
    "y_test",
]


for name in required_arrays:

    if name not in data.files:

        raise RuntimeError(
            f"Missing required array: {name}"
        )


X_train = np.asarray(
    data["X_train"],
    dtype=np.float32
)

y_train = np.asarray(
    data["y_train"],
    dtype=np.float32
)

X_val = np.asarray(
    data["X_val"],
    dtype=np.float32
)

y_val = np.asarray(
    data["y_val"],
    dtype=np.float32
)

X_test = np.asarray(
    data["X_test"],
    dtype=np.float32
)

y_test = np.asarray(
    data["y_test"],
    dtype=np.float32
)


print(
    f"X_train : {X_train.shape}"
)

print(
    f"y_train : {y_train.shape}"
)

print(
    f"X_val   : {X_val.shape}"
)

print(
    f"y_val   : {y_val.shape}"
)

print(
    f"X_test  : {X_test.shape}"
)

print(
    f"y_test  : {y_test.shape}"
)


# ============================================================
# 5. VALIDATE SHAPES
# ============================================================

for split_name, X, y in [

    ("train", X_train, y_train),
    ("validation", X_val, y_val),
    ("test", X_test, y_test),

]:

    if X.shape[1:] != (168, 4):

        raise RuntimeError(
            f"{split_name}: expected X (*,168,4), "
            f"received {X.shape}"
        )


    if y.shape[1:] != (24, 4):

        raise RuntimeError(
            f"{split_name}: expected y (*,24,4), "
            f"received {y.shape}"
        )


    if X.shape[0] != y.shape[0]:

        raise RuntimeError(
            f"{split_name}: sample count mismatch."
        )


print()

print(
    "[OK] All sequence shapes validated."
)


# ============================================================
# 6. BUILD HYBRID SEASONAL BASELINE
# ============================================================

def build_seasonal_baseline(X):

    """
    Construct 24-hour seasonal baseline.

    PV:
        previous 24-hour profile

    Load:
        previous 24-hour profile

    EV:
        corresponding 24 hours one week earlier

    Price:
        previous 24-hour profile
    """

    number_of_samples = X.shape[0]


    baseline = np.zeros(
        (
            number_of_samples,
            24,
            4,
        ),
        dtype=np.float32,
    )


    # --------------------------------------------------------
    # DAILY BASELINE
    # Last 24 hours from the historical 168-hour window.
    # --------------------------------------------------------

    previous_day = X[
        :,
        -24:,
        :
    ]


    # --------------------------------------------------------
    # WEEKLY BASELINE
    # First 24 hours of the historical 168-hour window.
    #
    # These correspond to the same forecast hours
    # one week earlier.
    # --------------------------------------------------------

    previous_week = X[
        :,
        0:24,
        :
    ]


    # PV -> daily

    baseline[
        :,
        :,
        PV_INDEX
    ] = previous_day[
        :,
        :,
        PV_INDEX
    ]


    # Load -> daily

    baseline[
        :,
        :,
        LOAD_INDEX
    ] = previous_day[
        :,
        :,
        LOAD_INDEX
    ]


    # EV -> weekly

    baseline[
        :,
        :,
        EV_INDEX
    ] = previous_week[
        :,
        :,
        EV_INDEX
    ]


    # Price -> daily

    baseline[
        :,
        :,
        PRICE_INDEX
    ] = previous_day[
        :,
        :,
        PRICE_INDEX
    ]


    return baseline


# ============================================================
# 7. CREATE BASELINES
# ============================================================

section(
    "BUILDING HYBRID SEASONAL BASELINES"
)


baseline_train = build_seasonal_baseline(
    X_train
)

baseline_val = build_seasonal_baseline(
    X_val
)

baseline_test = build_seasonal_baseline(
    X_test
)


print(
    f"Train baseline : {baseline_train.shape}"
)

print(
    f"Val baseline   : {baseline_val.shape}"
)

print(
    f"Test baseline  : {baseline_test.shape}"
)


# ============================================================
# 8. CALCULATE RESIDUAL TARGETS
# ============================================================

section(
    "CALCULATING RESIDUAL TARGETS"
)


residual_train = (
    y_train
    - baseline_train
)

residual_val = (
    y_val
    - baseline_val
)

residual_test = (
    y_test
    - baseline_test
)


print(
    f"Train residual : {residual_train.shape}"
)

print(
    f"Val residual   : {residual_val.shape}"
)

print(
    f"Test residual  : {residual_test.shape}"
)


# ============================================================
# 9. RESIDUAL STATISTICS
# ============================================================

section(
    "RESIDUAL STATISTICS"
)


for feature_index, feature_name in enumerate(
    FEATURE_NAMES
):

    values = residual_train[
        :,
        :,
        feature_index
    ].reshape(-1)


    print()

    print(
        feature_name
    )

    print(
        f"  Mean : {np.mean(values):.8f}"
    )

    print(
        f"  Std  : {np.std(values):.8f}"
    )

    print(
        f"  Min  : {np.min(values):.8f}"
    )

    print(
        f"  Max  : {np.max(values):.8f}"
    )

    print(
        f"  MAE from zero residual : "
        f"{np.mean(np.abs(values)):.8f}"
    )


# ============================================================
# 10. VERIFY EXACT RECONSTRUCTION
# ============================================================

section(
    "VERIFYING RESIDUAL RECONSTRUCTION"
)


for split_name, y, baseline, residual in [

    (
        "train",
        y_train,
        baseline_train,
        residual_train,
    ),

    (
        "validation",
        y_val,
        baseline_val,
        residual_val,
    ),

    (
        "test",
        y_test,
        baseline_test,
        residual_test,
    ),

]:

    reconstructed = (
        baseline
        + residual
    )


    maximum_error = float(
        np.max(
            np.abs(
                reconstructed - y
            )
        )
    )


    print(
        f"{split_name:12s} "
        f"maximum reconstruction error: "
        f"{maximum_error:.12e}"
    )


    if maximum_error > 1e-6:

        raise RuntimeError(
            f"Residual reconstruction failed "
            f"for {split_name}."
        )


print()

print(
    "[OK] Residual decomposition is exact."
)


# ============================================================
# 11. COMPARE VARIANCE
# ============================================================

section(
    "TARGET VS RESIDUAL VARIABILITY"
)


for feature_index, feature_name in enumerate(
    FEATURE_NAMES
):

    original_values = y_train[
        :,
        :,
        feature_index
    ].reshape(-1)


    residual_values = residual_train[
        :,
        :,
        feature_index
    ].reshape(-1)


    original_std = float(
        np.std(
            original_values
        )
    )

    residual_std = float(
        np.std(
            residual_values
        )
    )


    if original_std > 1e-12:

        ratio = (
            residual_std
            / original_std
        )

    else:

        ratio = np.nan


    print()

    print(
        feature_name
    )

    print(
        f"  Original std : "
        f"{original_std:.8f}"
    )

    print(
        f"  Residual std : "
        f"{residual_std:.8f}"
    )

    print(
        f"  Residual/original ratio : "
        f"{ratio:.6f}"
    )


# ============================================================
# 12. SAVE DATASET
# ============================================================

section(
    "SAVING RESIDUAL DATASET"
)


OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)


np.savez_compressed(

    OUTPUT_FILE,

    # Inputs

    X_train=X_train,

    X_val=X_val,

    X_test=X_test,


    # Original targets

    y_train=y_train,

    y_val=y_val,

    y_test=y_test,


    # Seasonal baselines

    baseline_train=baseline_train,

    baseline_val=baseline_val,

    baseline_test=baseline_test,


    # Residual targets

    residual_train=residual_train,

    residual_val=residual_val,

    residual_test=residual_test,


    # Documentation

    feature_names=np.array(
        FEATURE_NAMES
    ),

    baseline_method=np.array(
        [
            "daily",
            "daily",
            "weekly",
            "daily",
        ]
    ),
)


print(
    f"Residual dataset saved:\n"
    f"{OUTPUT_FILE}"
)


# ============================================================
# 13. FINAL VALIDATION
# ============================================================

section(
    "STEP 7E COMPLETE"
)


print(
    "Selected seasonal baseline:"
)

print(
    "  PV    -> daily  (t-24)"
)

print(
    "  Load  -> daily  (t-24)"
)

print(
    "  EV    -> weekly (t-168)"
)

print(
    "  Price -> daily  (t-24)"
)

print()

print(
    "Residual-learning targets were generated successfully."
)

print()

print(
    "No Transformer training was performed."
)

print()

print(
    "STEP 7E COMPLETE."
)