from pathlib import Path
import pandas as pd
import numpy as np

# ============================================================
# FC-HMARL - STEP 2C
# PECAN STREET LOAD CLEANING + HOURLY PROFILE CONSTRUCTION
# ============================================================

PROJECT_ROOT = Path(r"D:\Molvi paper review\FC_HMARL")

RAW_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "load"
    / "15minute_data_california"
    / "15minute_data_california.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "load"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = (
    OUTPUT_DIR
    / "PecanStreet_Load_2022_Clean.csv"
)

print("=" * 80)
print("FC-HMARL - PECAN STREET LOAD CLEANING")
print("=" * 80)

# ============================================================
# 1. CHECK RAW FILE
# ============================================================

if not RAW_FILE.exists():
    raise FileNotFoundError(
        f"Raw Pecan Street file not found:\n{RAW_FILE}"
    )

print(f"\nRaw file:\n{RAW_FILE}")

# ============================================================
# 2. READ ONLY REQUIRED COLUMNS
# ============================================================

required_columns = [
    "dataid",
    "local_15min",
    "grid",
]

print("\nReading required columns only...")

df = pd.read_csv(
    RAW_FILE,
    usecols=required_columns,
    low_memory=False,
)

print(f"Raw rows: {len(df):,}")

# ============================================================
# 3. PARSE TIMESTAMPS
# ============================================================

df["timestamp"] = pd.to_datetime(
    df["local_15min"],
    errors="coerce",
    utc=True,
)

invalid_timestamp_count = int(
    df["timestamp"].isna().sum()
)

print(
    f"Invalid timestamps: "
    f"{invalid_timestamp_count:,}"
)

df = df.dropna(
    subset=["timestamp"]
).copy()

# ============================================================
# 4. CONVERT GRID TO NUMERIC
# ============================================================

df["grid_kw"] = pd.to_numeric(
    df["grid"],
    errors="coerce",
)

invalid_grid_count = int(
    df["grid_kw"].isna().sum()
)

print(
    f"Invalid/non-numeric grid values: "
    f"{invalid_grid_count:,}"
)

df = df.dropna(
    subset=["grid_kw"]
).copy()

# ============================================================
# 5. REMOVE DUPLICATE HOUSEHOLD-TIMESTAMP RECORDS
# ============================================================

duplicate_count = int(
    df.duplicated(
        subset=[
            "dataid",
            "timestamp",
        ]
    ).sum()
)

print(
    f"Duplicate household/timestamp rows: "
    f"{duplicate_count:,}"
)

if duplicate_count > 0:
    df = df.drop_duplicates(
        subset=[
            "dataid",
            "timestamp",
        ],
        keep="first",
    )

# ============================================================
# 6. PRESERVE RAW GRID VALUES
# ============================================================

# Negative grid values are retained.
# They may represent net export and are not automatically errors.

print("\nGRID VALUE CHECK")
print("-" * 80)

print(
    f"Minimum grid value: "
    f"{df['grid_kw'].min():.6f} kW"
)

print(
    f"Maximum grid value: "
    f"{df['grid_kw'].max():.6f} kW"
)

print(
    f"Negative observations: "
    f"{(df['grid_kw'] < 0).sum():,}"
)

# ============================================================
# 7. CONVERT TO LOCAL CLOCK COMPONENTS
# ============================================================

# Convert UTC-parsed timestamps to timezone-naive values
# only for calendar/hour grouping.
#
# We are constructing a representative load profile rather
# than claiming these historical measurements occurred in 2022.

local_clock = (
    df["timestamp"]
    .dt.tz_convert(None)
)

df["month"] = local_clock.dt.month
df["day"] = local_clock.dt.day
df["hour"] = local_clock.dt.hour
df["year"] = local_clock.dt.year

# ============================================================
# 8. THREE-SIGMA QUALITY CHECK BY HOUSEHOLD
# ============================================================

# This follows the manuscript-style 3-sigma screening concept.
# Extreme measurements are marked as missing rather than
# blindly clipped.

print("\nTHREE-SIGMA SCREENING")
print("-" * 80)

df["grid_clean_kw"] = df["grid_kw"].copy()

outlier_total = 0

for household_id, group in df.groupby("dataid"):

    values = group["grid_kw"]

    mean_value = values.mean()
    std_value = values.std()

    if pd.isna(std_value) or std_value == 0:
        continue

    lower_bound = mean_value - 3.0 * std_value
    upper_bound = mean_value + 3.0 * std_value

    mask = (
        (values < lower_bound)
        | (values > upper_bound)
    )

    indices = group.index[mask]

    outlier_total += len(indices)

    df.loc[
        indices,
        "grid_clean_kw"
    ] = np.nan

print(
    f"Three-sigma observations flagged: "
    f"{outlier_total:,}"
)

# ============================================================
# 9. INTERPOLATE SHORT GAPS WITHIN EACH HOUSEHOLD
# ============================================================

print("\nINTERPOLATING FLAGGED/MISSING VALUES")
print("-" * 80)

df = df.sort_values(
    [
        "dataid",
        "timestamp",
    ]
)

df["grid_clean_kw"] = (
    df
    .groupby("dataid")["grid_clean_kw"]
    .transform(
        lambda series:
        series.interpolate(
            method="linear",
            limit_direction="both",
        )
    )
)

remaining_missing = int(
    df["grid_clean_kw"].isna().sum()
)

print(
    f"Remaining missing grid values: "
    f"{remaining_missing:,}"
)

# ============================================================
# 10. CONVERT 15-MIN DATA TO HOURLY HOUSEHOLD DATA
# ============================================================

print("\nCONVERTING 15-MINUTE DATA TO HOURLY")
print("-" * 80)

df["hour_timestamp"] = (
    df["timestamp"]
    .dt.floor("h")
)

hourly_household = (
    df
    .groupby(
        [
            "dataid",
            "hour_timestamp",
        ],
        as_index=False,
    )
    .agg(
        load_kw=(
            "grid_clean_kw",
            "mean",
        ),
        interval_count=(
            "grid_clean_kw",
            "count",
        ),
    )
)

# Keep only complete or nearly complete hourly groups.
# Normally 4 x 15-minute samples form one hour.

print(
    "Hourly household observations: "
    f"{len(hourly_household):,}"
)

print(
    "Interval-count distribution:"
)

print(
    hourly_household[
        "interval_count"
    ]
    .value_counts()
    .sort_index()
    .to_string()
)

# ============================================================
# 11. BUILD REPRESENTATIVE CALENDAR PROFILE
# ============================================================

# Aggregate across households and historical years according
# to month/day/hour.
#
# This preserves real Pecan Street load behavior while creating
# a representative profile suitable for the 2022 simulation
# calendar.

hourly_household["month"] = (
    hourly_household[
        "hour_timestamp"
    ].dt.month
)

hourly_household["day"] = (
    hourly_household[
        "hour_timestamp"
    ].dt.day
)

hourly_household["hour"] = (
    hourly_household[
        "hour_timestamp"
    ].dt.hour
)

representative_profile = (
    hourly_household
    .groupby(
        [
            "month",
            "day",
            "hour",
        ],
        as_index=False,
    )
    .agg(
        residential_load_kw=(
            "load_kw",
            "mean",
        ),
        source_observation_count=(
            "load_kw",
            "count",
        ),
    )
)

print(
    "\nRepresentative calendar points: "
    f"{len(representative_profile):,}"
)

# ============================================================
# 12. CREATE COMPLETE 2022 HOURLY CALENDAR
# ============================================================

calendar_2022 = pd.DataFrame(
    {
        "timestamp":
        pd.date_range(
            start="2022-01-01 00:00:00",
            end="2022-12-31 23:00:00",
            freq="h",
        )
    }
)

calendar_2022["month"] = (
    calendar_2022["timestamp"].dt.month
)

calendar_2022["day"] = (
    calendar_2022["timestamp"].dt.day
)

calendar_2022["hour"] = (
    calendar_2022["timestamp"].dt.hour
)

clean = calendar_2022.merge(
    representative_profile,
    on=[
        "month",
        "day",
        "hour",
    ],
    how="left",
)

# ============================================================
# 13. HANDLE CALENDAR PROFILE GAPS
# ============================================================

missing_before = int(
    clean[
        "residential_load_kw"
    ].isna().sum()
)

print(
    f"\nMissing 2022 calendar load values "
    f"before interpolation: {missing_before}"
)

if missing_before > 0:

    clean[
        "residential_load_kw"
    ] = (
        clean[
            "residential_load_kw"
        ]
        .interpolate(
            method="linear",
            limit_direction="both",
        )
    )

missing_after = int(
    clean[
        "residential_load_kw"
    ].isna().sum()
)

print(
    f"Missing 2022 calendar load values "
    f"after interpolation: {missing_after}"
)

# ============================================================
# 14. ADD PROVENANCE FIELDS
# ============================================================

clean["source_dataset"] = (
    "Pecan Street California 15-minute"
)

clean["source_period"] = (
    "2014-2018"
)

clean["calendar_mapping"] = (
    "Historical month-day-hour mean mapped to 2022"
)

# ============================================================
# 15. SELECT FINAL COLUMNS
# ============================================================

clean = clean[
    [
        "timestamp",
        "residential_load_kw",
        "source_observation_count",
        "source_dataset",
        "source_period",
        "calendar_mapping",
    ]
].copy()

# ============================================================
# 16. FINAL VALIDATION
# ============================================================

print("\nFINAL VALIDATION")
print("-" * 80)

print(
    f"Rows: "
    f"{len(clean):,}"
)

print(
    f"Start: "
    f"{clean['timestamp'].min()}"
)

print(
    f"End: "
    f"{clean['timestamp'].max()}"
)

print(
    f"Missing load values: "
    f"{clean['residential_load_kw'].isna().sum():,}"
)

print(
    f"Duplicate timestamps: "
    f"{clean['timestamp'].duplicated().sum():,}"
)

print(
    f"Minimum load: "
    f"{clean['residential_load_kw'].min():.6f} kW"
)

print(
    f"Maximum load: "
    f"{clean['residential_load_kw'].max():.6f} kW"
)

print(
    f"Mean load: "
    f"{clean['residential_load_kw'].mean():.6f} kW"
)

if len(clean) != 8760:
    raise RuntimeError(
        "Final load dataset does not contain 8,760 rows."
    )

if clean[
    "residential_load_kw"
].isna().any():
    raise RuntimeError(
        "Missing load values remain."
    )

if clean[
    "timestamp"
].duplicated().any():
    raise RuntimeError(
        "Duplicate timestamps remain."
    )

# ============================================================
# 17. SAVE CLEAN DATASET
# ============================================================

clean.to_csv(
    OUTPUT_FILE,
    index=False,
)

print("\n" + "=" * 80)

print(
    "CLEAN PECAN STREET LOAD DATASET CREATED"
)

print("=" * 80)

print(
    f"\nSaved to:\n{OUTPUT_FILE}"
)

print("\nFirst 5 rows:")
print(
    clean.head().to_string(
        index=False
    )
)

print("\nLast 5 rows:")
print(
    clean.tail().to_string(
        index=False
    )
)

print(
    "\nIMPORTANT:\n"
    "- Raw Pecan Street data were NOT modified.\n"
    "- Historical observations were NOT relabeled as raw 2022 data.\n"
    "- A representative calendar profile was constructed from real data.\n"
    "- 15-minute observations were aggregated to hourly resolution.\n"
    "- No min-max normalization was applied.\n"
)