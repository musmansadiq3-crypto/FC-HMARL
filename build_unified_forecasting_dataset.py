from pathlib import Path
import pandas as pd
import numpy as np
PROJECT_ROOT = Path(
    r"D:\Molvi paper review\FC_HMARL"
)

PV_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "pv"
    / "NSRDB_PV_2022_Clean.csv"
)

LOAD_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "load"
    / "PecanStreet_Load_2022_Clean.csv"
)

EV_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ev"
    / "ACN_EV_Representative_Annual_8760.csv"
)

PRICE_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "price"
    / "PJM_DA_RTO_2019_Clean.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "FC_HMARL_Unified_Forecasting_8760.csv"
)
COMMON_YEAR = 2022
# ============================================================
# 3. HELPER FUNCTIONS
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


def require_file(path, label):

    if not path.exists():

        raise FileNotFoundError(
            f"{label} file not found:\n{path}"
        )


def find_first_existing_column(
    df,
    candidates,
    label,
):

    for column in candidates:

        if column in df.columns:

            return column

    raise RuntimeError(
        f"Could not identify {label} column.\n"
        f"Tried:\n"
        + "\n".join(
            f"  - {c}"
            for c in candidates
        )
        + "\n\nAvailable columns:\n"
        + "\n".join(
            f"  - {c}"
            for c in df.columns
        )
    )


def check_8760(df, label):

    if len(df) != 8760:

        raise RuntimeError(
            f"{label} does not contain exactly 8760 rows. "
            f"Rows found: {len(df):,}"
        )


# ============================================================
# 4. START
# ============================================================

section(
    "FC-HMARL - BUILDING UNIFIED FORECASTING DATASET"
)


# ============================================================
# 5. VERIFY ALL INPUT FILES
# ============================================================

require_file(
    PV_FILE,
    "PV",
)

require_file(
    LOAD_FILE,
    "Load",
)

require_file(
    EV_FILE,
    "EV",
)

require_file(
    PRICE_FILE,
    "Price",
)


print(
    "Input files:"
)

print(
    f"PV    : {PV_FILE}"
)

print(
    f"Load  : {LOAD_FILE}"
)

print(
    f"EV    : {EV_FILE}"
)

print(
    f"Price : {PRICE_FILE}"
)


# ============================================================
# 6. LOAD ALL DATASETS
# ============================================================

subsection(
    "LOADING CLEANED DATASETS"
)


pv = pd.read_csv(
    PV_FILE
)

load = pd.read_csv(
    LOAD_FILE
)

ev = pd.read_csv(
    EV_FILE
)

price = pd.read_csv(
    PRICE_FILE
)


print(
    f"PV rows    : {len(pv):,}"
)

print(
    f"Load rows  : {len(load):,}"
)

print(
    f"EV rows    : {len(ev):,}"
)

print(
    f"Price rows : {len(price):,}"
)


# ============================================================
# 7. VERIFY EXPECTED LENGTHS
# ============================================================

check_8760(
    pv,
    "PV dataset",
)

check_8760(
    load,
    "Load dataset",
)

check_8760(
    ev,
    "EV dataset",
)

check_8760(
    price,
    "Price dataset",
)
subsection(
    "IDENTIFYING TARGET COLUMNS"
)


pv_target_column = find_first_existing_column(

    pv,

    [
        "pv_power_kw",
        "pv_kw",
        "pv_power",
        "ghi_w_m2",
        "ghi",
    ],

    "PV target",
)


print(
    f"PV target column selected    : "
    f"{pv_target_column}"
)


# ============================================================
# 9. IDENTIFY LOAD COLUMN
# ============================================================

load_target_column = find_first_existing_column(

    load,

    [
        "residential_load_kw",
        "load_kw",
        "grid_kw",
        "load_power_kw",
        "power_kw",
        "grid",
    ],

    "load target",
)


print(
    f"Load target column selected  : "
    f"{load_target_column}"
)


# ============================================================
# 10. IDENTIFY EV COLUMN
# ============================================================

ev_target_column = find_first_existing_column(

    ev,

    [
        "ev_power_kw",
    ],

    "EV target",
)


print(
    f"EV target column selected    : "
    f"{ev_target_column}"
)


# ============================================================
# 11. IDENTIFY PRICE COLUMN
# ============================================================

price_target_column = find_first_existing_column(

    price,

    [
        "total_lmp_da_usd_per_kwh",
        "total_lmp_da",
    ],

    "price target",
)


print(
    f"Price target column selected : "
    f"{price_target_column}"
)


# ============================================================
# 12. BUILD COMMON 2022 CALENDAR
# ============================================================

subsection(
    "BUILDING COMMON 8760-HOUR CALENDAR"
)


calendar = pd.DataFrame(

    {
        "timestamp":
            pd.date_range(
                start=f"{COMMON_YEAR}-01-01 00:00:00",
                end=f"{COMMON_YEAR}-12-31 23:00:00",
                freq="h",
            )
    }

)


if len(calendar) != 8760:

    raise RuntimeError(
        "Common calendar does not contain 8760 hours."
    )


calendar[
    "month"
] = (
    calendar[
        "timestamp"
    ]
    .dt
    .month
)


calendar[
    "day"
] = (
    calendar[
        "timestamp"
    ]
    .dt
    .day
)


calendar[
    "hour"
] = (
    calendar[
        "timestamp"
    ]
    .dt
    .hour
)


calendar[
    "day_of_week"
] = (
    calendar[
        "timestamp"
    ]
    .dt
    .dayofweek
)


calendar[
    "weekday"
] = (
    calendar[
        "timestamp"
    ]
    .dt
    .day_name()
)


calendar[
    "is_weekend"
] = (
    calendar[
        "day_of_week"
    ]
    >= 5
).astype(
    int
)


print(
    f"Common calendar rows: "
    f"{len(calendar):,}"
)

subsection(
    "PREPARING PV SERIES"
)


pv_values = pd.to_numeric(

    pv[
        pv_target_column
    ],

    errors="coerce",
)


print(
    f"PV missing values: "
    f"{pv_values.isna().sum():,}"
)
subsection(
    "PREPARING LOAD SERIES"
)
load_values = pd.to_numeric(

    load[
        load_target_column
    ],

    errors="coerce",
)


print(
    f"Load missing values: "
    f"{load_values.isna().sum():,}"
)
subsection(
    "PREPARING EV SERIES"
)


ev_values = pd.to_numeric(

    ev[
        ev_target_column
    ],

    errors="coerce",
)


print(
    f"EV missing values: "
    f"{ev_values.isna().sum():,}"
)

subsection(
    "PREPARING PRICE SERIES"
)


price_values = pd.to_numeric(

    price[
        price_target_column
    ],

    errors="coerce",
)


print(
    f"Price missing values: "
    f"{price_values.isna().sum():,}"
)
subsection(
    "BUILDING FOUR-TARGET FORECASTING TABLE"
)
unified = calendar.copy()


unified[
    "pv"
] = (
    pv_values
    .to_numpy()
)


unified[
    "load"
] = (
    load_values
    .to_numpy()
)


unified[
    "ev"
] = (
    ev_values
    .to_numpy()
)


unified[
    "price"
] = (
    price_values
    .to_numpy()
)
if pv_target_column.lower() in [
    "ghi_w_m2",
    "ghi",
]:

    pv_unit = "W/m2"

    pv_target_type = (
        "solar_irradiance_proxy"
    )

else:

    pv_unit = "kW"

    pv_target_type = (
        "pv_power"
    )


if price_target_column == (
    "total_lmp_da_usd_per_kwh"
):

    price_unit = "USD/kWh"

else:

    price_unit = "USD/MWh"


unified[
    "pv_unit"
] = pv_unit

unified[
    "load_unit"
] = "kW"

unified[
    "ev_unit"
] = "kW"

unified[
    "price_unit"
] = price_unit

unified[
    "pv_target_type"
] = pv_target_type


# ============================================================
# 19. PROVENANCE
# ============================================================

unified[
    "pv_source"
] = (
    "NSRDB 2022"
)

unified[
    "load_source"
] = (
    "Pecan Street 2014-2018 "
    "representative profile mapped to common calendar"
)

unified[
    "ev_source"
] = (
    "ACN-Data Caltech 2018-2020 "
    "representative annual profile"
)

unified[
    "price_source"
] = (
    "PJM-RTO Day-Ahead 2019 "
    "annual chronological profile"
)


unified[
    "common_calendar_year"
] = COMMON_YEAR


unified[
    "all_sources_measured_in_common_year"
] = 0


# ============================================================
# 20. BASIC QUALITY FLAGS
# ============================================================

target_columns = [

    "pv",
    "load",
    "ev",
    "price",

]


for column in target_columns:

    unified[
        f"{column}_missing"
    ] = (

        unified[
            column
        ]

        .isna()

    ).astype(
        int
    )


# ============================================================
# 21. VALIDATION - MISSING VALUES
# ============================================================

subsection(
    "MISSING-VALUE VALIDATION"
)


for column in target_columns:

    missing_count = int(

        unified[
            column
        ]

        .isna()

        .sum()
    )

    print(
        f"{column.upper():5s} missing values: "
        f"{missing_count:,}"
    )


# ============================================================
# 22. DUPLICATE TIMESTAMP VALIDATION
# ============================================================

duplicate_timestamps = int(

    unified[
        "timestamp"
    ]

    .duplicated()

    .sum()
)


print()

print(
    f"Duplicate timestamps: "
    f"{duplicate_timestamps:,}"
)


# ============================================================
# 23. TARGET STATISTICS
# ============================================================

subsection(
    "TARGET STATISTICS"
)


for column in target_columns:

    values = unified[
        column
    ]

    print()

    print(
        column.upper()
    )

    print(
        values
        .describe()
        .to_string()
    )


# ============================================================
# 24. CHECK PHYSICAL SIGNS
# ============================================================

subsection(
    "SIGN CHECKS"
)


print(
    f"Negative PV values    : "
    f"{(unified['pv'] < 0).sum():,}"
)

print(
    f"Negative load values  : "
    f"{(unified['load'] < 0).sum():,}"
)

print(
    f"Negative EV values    : "
    f"{(unified['ev'] < 0).sum():,}"
)

print(
    f"Negative price values : "
    f"{(unified['price'] < 0).sum():,}"
)


# ============================================================
# 25. THREE-SIGMA DIAGNOSTIC FLAGS
# ============================================================
#
# Identify only.
# Do NOT replace observations in Step 6A.
# ============================================================

subsection(
    "THREE-SIGMA DIAGNOSTIC"
)


for column in target_columns:

    mean_value = (
        unified[
            column
        ]
        .mean()
    )

    std_value = (
        unified[
            column
        ]
        .std()
    )

    lower = (
        mean_value
        - 3.0 * std_value
    )

    upper = (
        mean_value
        + 3.0 * std_value
    )

    flag_column = (
        f"{column}_three_sigma_flag"
    )


    unified[
        flag_column
    ] = (

        (
            unified[
                column
            ]
            < lower
        )

        |

        (
            unified[
                column
            ]
            > upper
        )

    ).astype(
        int
    )


    print(
        f"{column.upper():5s} | "
        f"mean={mean_value:.6f} | "
        f"std={std_value:.6f} | "
        f"lower={lower:.6f} | "
        f"upper={upper:.6f} | "
        f"flagged={unified[flag_column].sum():,}"
    )

output_columns = [

    # --------------------------------------------------------
    # Common time index
    # --------------------------------------------------------

    "timestamp",
    "month",
    "day",
    "hour",
    "day_of_week",
    "weekday",
    "is_weekend",

    # --------------------------------------------------------
    # Forecasting targets
    # --------------------------------------------------------

    "pv",
    "load",
    "ev",
    "price",

    # --------------------------------------------------------
    # Units / semantic meaning
    # --------------------------------------------------------

    "pv_unit",
    "load_unit",
    "ev_unit",
    "price_unit",
    "pv_target_type",

    # --------------------------------------------------------
    # Quality flags
    # --------------------------------------------------------

    "pv_missing",
    "load_missing",
    "ev_missing",
    "price_missing",

    "pv_three_sigma_flag",
    "load_three_sigma_flag",
    "ev_three_sigma_flag",
    "price_three_sigma_flag",

    # --------------------------------------------------------
    # Provenance
    # --------------------------------------------------------

    "pv_source",
    "load_source",
    "ev_source",
    "price_source",

    "common_calendar_year",
    "all_sources_measured_in_common_year",

]


unified = unified[
    output_columns
].copy()


# ============================================================
# 28. FINAL HARD VALIDATION
# ============================================================

section(
    "UNIFIED FORECASTING DATASET VALIDATION"
)


print(
    f"Rows                    : "
    f"{len(unified):,}"
)

print(
    f"Unique timestamps       : "
    f"{unified['timestamp'].nunique():,}"
)

print(
    f"Duplicate timestamps    : "
    f"{unified['timestamp'].duplicated().sum():,}"
)


for column in target_columns:

    print(
        f"Missing {column.upper():5s} values    : "
        f"{unified[column].isna().sum():,}"
    )


# ============================================================
# 29. FAIL IF CORE DATASET IS NOT COMPLETE
# ============================================================

if len(unified) != 8760:

    raise RuntimeError(
        "Unified dataset does not contain exactly 8760 rows."
    )


if unified[
    "timestamp"
].duplicated().any():

    raise RuntimeError(
        "Unified dataset contains duplicate timestamps."
    )


if unified[
    target_columns
].isna().any().any():

    raise RuntimeError(
        "Unified dataset contains missing forecasting targets."
    )


# ============================================================
# 30. SAVE
# ============================================================

subsection(
    "SAVING UNIFIED FORECASTING DATASET"
)


unified.to_csv(
    OUTPUT_FILE,
    index=False,
)


print(
    f"Saved:\n"
    f"{OUTPUT_FILE}"
)


# ============================================================
# 31. FINAL MESSAGE
# ============================================================

section(
    "UNIFIED FORECASTING DATASET COMPLETED"
)


print(
    f"Final dataset:\n"
    f"{OUTPUT_FILE}"
)

print()

print(
    "Four forecasting targets are aligned to one "
    "8760-hour common simulation calendar."
)

print(
    "Source provenance and source periods are preserved."
)

print(
    "The common calendar does NOT imply that all sources "
    "were measured in 2022."
)

print(
    "No normalization was performed."
)

print(
    "No chronological train/validation/test split was performed."
)

print(
    "Three-sigma observations were flagged but not modified."
)

print(
    "STEP 6A COMPLETE."
)
