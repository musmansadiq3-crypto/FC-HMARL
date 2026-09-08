from pathlib import Path
import zipfile
import pandas as pd
import numpy as np
# ============================================================
# 1. PROJECT PATHS
# ============================================================
PROJECT_ROOT = Path(
    r"D:\Molvi paper review\FC_HMARL"
)
RAW_PRICE_ZIP = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "price"
    / "price_pjm.zip"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "price"
)

OUTPUT_FILE = (
    OUTPUT_DIR
    / "PJM_DA_RTO_2019_Clean.csv"
)
# ============================================================
# 2. SOURCE FILE INSIDE ZIP
# ============================================================

ZIP_MEMBER = (
    "price_pjm/zone/2019_da_hrl_lmps.csv"
)


# ============================================================
# 3. PJM-RTO SELECTION
# ============================================================

TARGET_PNODE_ID = 1

TARGET_PNODE_NAME = "PJM-RTO"

TARGET_TYPE = "ZONE"


# ============================================================
# 4. HELPER FUNCTIONS
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


# ============================================================
# 5. START
# ============================================================

section(
    "FC-HMARL - PJM DAY-AHEAD PRICE CLEANING"
)


# ============================================================
# 6. VERIFY INPUT ZIP
# ============================================================

if not RAW_PRICE_ZIP.exists():

    raise FileNotFoundError(
        "PJM price ZIP was not found:\n"
        f"{RAW_PRICE_ZIP}"
    )


print(
    f"Input archive:\n"
    f"{RAW_PRICE_ZIP}"
)


# ============================================================
# 7. CREATE OUTPUT DIRECTORY
# ============================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# 8. OPEN ZIP AND LOAD ZONE FILE
# ============================================================

subsection(
    "LOADING PJM 2019 DAY-AHEAD ZONE DATA"
)


with zipfile.ZipFile(
    RAW_PRICE_ZIP,
    mode="r",
) as archive:

    if ZIP_MEMBER not in archive.namelist():

        raise FileNotFoundError(
            "Expected PJM zone file was not found inside ZIP:\n"
            f"{ZIP_MEMBER}"
        )

    with archive.open(
        ZIP_MEMBER
    ) as file:

        df = pd.read_csv(
            file
        )


print(
    f"Raw rows loaded: "
    f"{len(df):,}"
)

print(
    f"Raw columns: "
    f"{len(df.columns):,}"
)


# ============================================================
# 9. VERIFY REQUIRED COLUMNS
# ============================================================

required_columns = [

    "datetime_beginning_utc",
    "datetime_beginning_ept",
    "pnode_id",
    "pnode_name",
    "type",
    "system_energy_price_da",
    "total_lmp_da",
    "congestion_price_da",
    "marginal_loss_price_da",
    "row_is_current",
    "version_nbr",

]


missing_columns = [

    column

    for column
    in required_columns

    if column not in df.columns
]


if missing_columns:

    raise RuntimeError(
        "Required columns are missing:\n"
        + "\n".join(
            missing_columns
        )
    )


# ============================================================
# 10. FILTER PJM-RTO ZONE
# ============================================================

subsection(
    "FILTERING PJM-RTO ZONE"
)


pjm = df[

    (
        pd.to_numeric(
            df["pnode_id"],
            errors="coerce",
        )
        == TARGET_PNODE_ID
    )

    &

    (
        df["pnode_name"]
        .astype(str)
        .str.strip()
        .str.upper()
        == TARGET_PNODE_NAME
    )

    &

    (
        df["type"]
        .astype(str)
        .str.strip()
        .str.upper()
        == TARGET_TYPE
    )

].copy()


print(
    f"PJM-RTO rows selected: "
    f"{len(pjm):,}"
)


if len(pjm) == 0:

    raise RuntimeError(
        "No PJM-RTO records were found."
    )


# ============================================================
# 11. PARSE TIMESTAMPS
# ============================================================

subsection(
    "PARSING TIMESTAMPS"
)


pjm["timestamp_utc"] = pd.to_datetime(

    pjm[
        "datetime_beginning_utc"
    ],

    errors="coerce",
    utc=True,
)


pjm["timestamp_ept"] = pd.to_datetime(

    pjm[
        "datetime_beginning_ept"
    ],

    errors="coerce",
)


print(
    f"Invalid UTC timestamps: "
    f"{pjm['timestamp_utc'].isna().sum():,}"
)

print(
    f"Invalid EPT timestamps: "
    f"{pjm['timestamp_ept'].isna().sum():,}"
)


# ============================================================
# 12. CONVERT NUMERIC PRICE COLUMNS
# ============================================================

price_columns = [

    "system_energy_price_da",
    "total_lmp_da",
    "congestion_price_da",
    "marginal_loss_price_da",

]


for column in price_columns:

    pjm[column] = pd.to_numeric(

        pjm[
            column
        ],

        errors="coerce",
    )


pjm["version_nbr"] = pd.to_numeric(

    pjm[
        "version_nbr"
    ],

    errors="coerce",
)


# ============================================================
# 13. NORMALIZE CURRENT-ROW FLAG
# ============================================================

pjm["row_is_current_bool"] = (

    pjm[
        "row_is_current"
    ]

    .astype(str)

    .str.strip()

    .str.lower()

    .map(
        {
            "true": True,
            "false": False,
            "1": True,
            "0": False,
        }
    )
)


# ============================================================
# 14. RAW VERSION SUMMARY
# ============================================================

subsection(
    "VERSION / CURRENT-ROW SUMMARY"
)


print(
    "Version counts:"
)


print(

    pjm[
        "version_nbr"
    ]

    .value_counts(
        dropna=False
    )

    .sort_index()

    .to_string()
)

print()

print(
    "row_is_current counts:"
)

print(

    pjm[
        "row_is_current_bool"
    ]

    .value_counts(
        dropna=False
    )

    .to_string()
)

# ============================================================
# 15. CHECK DUPLICATE TIMESTAMPS BEFORE VERSION FILTERING
# ============================================================

subsection(
    "DUPLICATE TIMESTAMP CHECK BEFORE VERSION SELECTION"
)

duplicate_time_rows = int(

    pjm[
        "timestamp_utc"
    ]

    .duplicated(
        keep=False
    )

    .sum()
)
print(
    f"Rows participating in duplicate UTC timestamps: "
    f"{duplicate_time_rows:,}"
)
current_rows = pjm[

    pjm[
        "row_is_current_bool"
    ]
    == True

].copy()


if len(current_rows) > 0:

    selected = current_rows

else:

    selected = pjm.copy()


selected = (

    selected

    .sort_values(
        by=[
            "timestamp_utc",
            "version_nbr",
        ]
    )

    .drop_duplicates(
        subset=[
            "timestamp_utc"
        ],
        keep="last",
    )

    .reset_index(
        drop=True
    )
)


print()

print(
    f"Rows after current/version selection: "
    f"{len(selected):,}"
)


# ============================================================
# 17. REMOVE ONLY INVALID TIMESTAMP ROWS
# ============================================================

invalid_timestamp_rows = int(

    selected[
        "timestamp_utc"
    ]

    .isna()

    .sum()
)


if invalid_timestamp_rows > 0:

    selected = selected[

        selected[
            "timestamp_utc"
        ]

        .notna()

    ].copy()


print(
    f"Invalid timestamp rows removed: "
    f"{invalid_timestamp_rows:,}"
)


# ============================================================
# 18. SORT CHRONOLOGICALLY
# ============================================================

selected = (

    selected

    .sort_values(
        "timestamp_utc"
    )

    .reset_index(
        drop=True
    )
)


# ============================================================
# 19. CHECK TIME COVERAGE
# ============================================================

subsection(
    "TIME COVERAGE"
)


print(
    f"Earliest UTC timestamp: "
    f"{selected['timestamp_utc'].min()}"
)

print(
    f"Latest UTC timestamp  : "
    f"{selected['timestamp_utc'].max()}"
)

print(
    f"Unique hourly records : "
    f"{selected['timestamp_utc'].nunique():,}"
)


# ============================================================
# 20. EXPECTED HOURLY INDEX
# ============================================================

expected_index = pd.date_range(

    start=selected[
        "timestamp_utc"
    ].min(),

    end=selected[
        "timestamp_utc"
    ].max(),

    freq="h",
)


expected_hours = len(
    expected_index
)


actual_hours = selected[
    "timestamp_utc"
].nunique()


missing_hours = expected_index.difference(

    pd.DatetimeIndex(
        selected[
            "timestamp_utc"
        ]
    )
)


print(
    f"Expected hourly timestamps: "
    f"{expected_hours:,}"
)

print(
    f"Actual hourly timestamps  : "
    f"{actual_hours:,}"
)

print(
    f"Missing hourly timestamps : "
    f"{len(missing_hours):,}"
)


# ============================================================
# 21. SHOW MISSING HOURS IF ANY
# ============================================================

if len(
    missing_hours
) > 0:

    print()

    print(
        "First missing timestamps:"
    )

    for timestamp in (
        missing_hours[:20]
    ):

        print(
            timestamp
        )


# ============================================================
# 22. PRICE QUALITY CHECK
# ============================================================

subsection(
    "DAY-AHEAD LMP QUALITY CHECK"
)


price = selected[
    "total_lmp_da"
]


print(

    price

    .describe()

    .to_string()
)


print()

print(
    f"Missing/non-numeric prices: "
    f"{price.isna().sum():,}"
)

print(
    f"Negative prices           : "
    f"{(price < 0).sum():,}"
)

print(
    f"Zero prices               : "
    f"{(price == 0).sum():,}"
)

print(
    f"Positive prices           : "
    f"{(price > 0).sum():,}"
)


# ============================================================
# 23. DO NOT REMOVE NEGATIVE PRICES
# ============================================================
#
# Negative wholesale LMP values can be legitimate.
# They are preserved.
# ============================================================

selected[
    "is_negative_lmp"
] = (

    selected[
        "total_lmp_da"
    ]

    < 0

).astype(
    int
)


# ============================================================
# 24. MISSING PRICE FLAG
# ============================================================

selected[
    "missing_lmp"
] = (

    selected[
        "total_lmp_da"
    ]

    .isna()

).astype(
    int
)


# ============================================================
# 25. HOURLY GAP FLAG
# ============================================================
#
# The current rows are not interpolated here.
# This flag concerns the actual selected observations.
# ============================================================

selected[
    "valid_hourly_record"
] = (

    selected[
        "timestamp_utc"
    ]
    .notna()

    &

    selected[
        "total_lmp_da"
    ]
    .notna()

).astype(
    int
)


# ============================================================
# 26. DERIVE CALENDAR VARIABLES
# ============================================================

selected[
    "year"
] = (

    selected[
        "timestamp_utc"
    ]
    .dt
    .year
)


selected[
    "month"
] = (

    selected[
        "timestamp_utc"
    ]
    .dt
    .month
)


selected[
    "day"
] = (

    selected[
        "timestamp_utc"
    ]
    .dt
    .day
)


selected[
    "hour_utc"
] = (

    selected[
        "timestamp_utc"
    ]
    .dt
    .hour
)


selected[
    "weekday_utc"
] = (

    selected[
        "timestamp_utc"
    ]
    .dt
    .day_name()
)


# ============================================================
# 27. CONVERT $/MWh TO $/kWh
# ============================================================
#
# PJM LMP:
#
#   USD/MWh
#
# FC-HMARL market environment typically operates with:
#
#   USD/kWh
#
# Conversion:
#
#   1 MWh = 1000 kWh
#
# ============================================================

selected[
    "total_lmp_da_usd_per_kwh"
] = (

    selected[
        "total_lmp_da"
    ]

    / 1000.0
)


selected[
    "system_energy_price_da_usd_per_kwh"
] = (

    selected[
        "system_energy_price_da"
    ]

    / 1000.0
)


selected[
    "congestion_price_da_usd_per_kwh"
] = (

    selected[
        "congestion_price_da"
    ]

    / 1000.0
)


selected[
    "marginal_loss_price_da_usd_per_kwh"
] = (

    selected[
        "marginal_loss_price_da"
    ]

    / 1000.0
)


# ============================================================
# 28. PROVENANCE FIELDS
# ============================================================

selected[
    "source_dataset"
] = "PJM"

selected[
    "source_market"
] = "Day-Ahead"

selected[
    "source_location"
] = "PJM-RTO"

selected[
    "source_pnode_id"
] = TARGET_PNODE_ID

selected[
    "source_period"
] = "2019"

selected[
    "source_file"
] = ZIP_MEMBER

selected[
    "source_price_unit_original"
] = "USD/MWh"

selected[
    "processed_price_unit"
] = "USD/kWh"


# ============================================================
# 29. PREPARE OUTPUT COLUMNS
# ============================================================

output_columns = [

    # --------------------------------------------------------
    # Provenance
    # --------------------------------------------------------

    "source_dataset",
    "source_market",
    "source_location",
    "source_pnode_id",
    "source_period",
    "source_file",
    "source_price_unit_original",
    "processed_price_unit",

    # --------------------------------------------------------
    # Time
    # --------------------------------------------------------

    "timestamp_utc",
    "timestamp_ept",

    "year",
    "month",
    "day",
    "hour_utc",
    "weekday_utc",

    # --------------------------------------------------------
    # PJM identifiers
    # --------------------------------------------------------

    "pnode_id",
    "pnode_name",
    "type",

    # --------------------------------------------------------
    # Original PJM price components USD/MWh
    # --------------------------------------------------------

    "system_energy_price_da",
    "total_lmp_da",
    "congestion_price_da",
    "marginal_loss_price_da",

    # --------------------------------------------------------
    # Converted USD/kWh
    # --------------------------------------------------------

    "system_energy_price_da_usd_per_kwh",
    "total_lmp_da_usd_per_kwh",
    "congestion_price_da_usd_per_kwh",
    "marginal_loss_price_da_usd_per_kwh",

    # --------------------------------------------------------
    # Version information
    # --------------------------------------------------------

    "row_is_current_bool",
    "version_nbr",

    # --------------------------------------------------------
    # Quality flags
    # --------------------------------------------------------

    "is_negative_lmp",
    "missing_lmp",
    "valid_hourly_record",

]


clean = selected[
    output_columns
].copy()


# ============================================================
# 30. SAVE CLEAN DATASET
# ============================================================

subsection(
    "SAVING CLEAN PJM PRICE DATA"
)


clean.to_csv(
    OUTPUT_FILE,
    index=False,
)


print(
    f"Saved:\n"
    f"{OUTPUT_FILE}"
)


# ============================================================
# 31. FINAL VALIDATION
# ============================================================

section(
    "CLEAN PJM PRICE DATA VALIDATION"
)


print(
    f"Output rows                 : "
    f"{len(clean):,}"
)

print(
    f"Unique UTC timestamps       : "
    f"{clean['timestamp_utc'].nunique():,}"
)

print(
    f"Duplicate UTC timestamps    : "
    f"{clean['timestamp_utc'].duplicated().sum():,}"
)

print(
    f"Missing LMP values          : "
    f"{clean['missing_lmp'].sum():,}"
)

print(
    f"Negative LMP observations   : "
    f"{clean['is_negative_lmp'].sum():,}"
)

print(
    f"Missing hourly timestamps   : "
    f"{len(missing_hours):,}"
)


# ============================================================
# 32. FINAL PRICE SUMMARY USD/MWh
# ============================================================

subsection(
    "PJM-RTO TOTAL LMP SUMMARY - USD/MWh"
)


print(

    clean[
        "total_lmp_da"
    ]

    .describe()

    .to_string()
)


# ============================================================
# 33. FINAL PRICE SUMMARY USD/kWh
# ============================================================

subsection(
    "PJM-RTO TOTAL LMP SUMMARY - USD/kWh"
)


print(

    clean[
        "total_lmp_da_usd_per_kwh"
    ]

    .describe()

    .to_string()
)


# ============================================================
# 34. HOURLY PRICE PROFILE SUMMARY
# ============================================================

subsection(
    "MEAN PRICE BY UTC HOUR - USD/MWh"
)


hourly_mean = (

    clean

    .groupby(
        "hour_utc"
    )[
        "total_lmp_da"
    ]

    .mean()
)


print(
    hourly_mean
    .to_string()
)


# ============================================================
# 35. FINAL
# ============================================================

section(
    "PJM PRICE CLEANING COMPLETED"
)


print(
    f"Final clean dataset:\n"
    f"{OUTPUT_FILE}"
)

print()

print(
    "The original PJM ZIP archive was not modified."
)

print(
    "The selected series is PJM-RTO Day-Ahead total LMP."
)

print(
    "The selected source year remains explicitly 2019."
)

print(
    "Negative wholesale prices were preserved."
)

print(
    "No 2019 observations were relabeled as 2022."
)

print(
    "No synthetic prices were generated."
)

print(
    "No normalization was performed."
)

print(
    "No hourly interpolation was performed in this cleaning stage."
)

print()

print(
    "STEP 4B COMPLETE."
)
