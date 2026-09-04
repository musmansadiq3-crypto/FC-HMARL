from pathlib import Path
import zipfile
import pandas as pd
import numpy as np

# ============================================================
# FC-HMARL
# NSRDB PV DATA CLEANING
# ============================================================

PROJECT_ROOT = Path(r"D:\Molvi paper review\FC_HMARL")

RAW_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "pv"
    / "NLR.zip"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "pv"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "NSRDB_PV_2022_Clean.csv"

print("=" * 70)
print("FC-HMARL - NSRDB PV DATA CLEANING")
print("=" * 70)

# ------------------------------------------------------------
# 1. Read raw NSRDB data directly from ZIP
# ------------------------------------------------------------

with zipfile.ZipFile(RAW_FILE, "r") as z:

    csv_files = [
        f for f in z.namelist()
        if f.lower().endswith(".csv")
    ]

    if len(csv_files) != 1:
        raise RuntimeError(
            f"Expected one CSV inside NLR.zip, found {len(csv_files)}"
        )

    raw_csv = csv_files[0]

    with z.open(raw_csv) as f:
        df = pd.read_csv(f, skiprows=2)

print(f"\nRaw observations: {len(df):,}")

# ------------------------------------------------------------
# 2. Construct timestamp
# ------------------------------------------------------------

df["timestamp"] = pd.to_datetime(
    {
        "year": df["Year"],
        "month": df["Month"],
        "day": df["Day"],
        "hour": df["Hour"],
        "minute": df["Minute"],
    },
    errors="raise",
)

# ------------------------------------------------------------
# 3. Select variables
# ------------------------------------------------------------

required_columns = [
    "timestamp",
    "GHI",
    "DNI",
    "DHI",
    "Temperature",
    "Wind Speed",
]

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Required columns missing: {missing_columns}"
    )

clean = df[required_columns].copy()

# ------------------------------------------------------------
# 4. Standardize column names
# ------------------------------------------------------------

clean = clean.rename(
    columns={
        "GHI": "ghi_w_m2",
        "DNI": "dni_w_m2",
        "DHI": "dhi_w_m2",
        "Temperature": "temperature_c",
        "Wind Speed": "wind_speed_m_s",
    }
)

# ------------------------------------------------------------
# 5. Sort chronologically
# ------------------------------------------------------------

clean = (
    clean
    .sort_values("timestamp")
    .reset_index(drop=True)
)

# ------------------------------------------------------------
# 6. Remove exact duplicate timestamps if any
# ------------------------------------------------------------

duplicate_timestamps = clean["timestamp"].duplicated().sum()

if duplicate_timestamps > 0:
    print(
        f"WARNING: removing {duplicate_timestamps} "
        "duplicate timestamps."
    )

    clean = clean.drop_duplicates(
        subset="timestamp",
        keep="first",
    ).reset_index(drop=True)

# ------------------------------------------------------------
# 7. Physical validity checks
# ------------------------------------------------------------

irradiance_columns = [
    "ghi_w_m2",
    "dni_w_m2",
    "dhi_w_m2",
]

negative_irradiance = (
    clean[irradiance_columns] < 0
).sum()

print("\nNegative irradiance observations:")
print(negative_irradiance.to_string())

# Negative solar irradiance is physically invalid.
# We do NOT silently modify positive high irradiance values.

for col in irradiance_columns:
    clean.loc[clean[col] < 0, col] = np.nan

# ------------------------------------------------------------
# 8. Missing-value check
# ------------------------------------------------------------

print("\nMissing values before interpolation:")
print(clean.isna().sum().to_string())

# Interpolate only if missing observations actually exist.
numeric_columns = [
    "ghi_w_m2",
    "dni_w_m2",
    "dhi_w_m2",
    "temperature_c",
    "wind_speed_m_s",
]

if clean[numeric_columns].isna().any().any():

    print("\nMissing observations detected.")
    print("Applying time-based linear interpolation...")

    temp = clean.set_index("timestamp")

    temp[numeric_columns] = (
        temp[numeric_columns]
        .interpolate(
            method="time",
            limit_direction="both",
        )
    )

    clean = temp.reset_index()

else:
    print(
        "\nNo missing observations detected. "
        "Interpolation was NOT applied."
    )

# ------------------------------------------------------------
# 9. Verify hourly continuity
# ------------------------------------------------------------

expected_index = pd.date_range(
    start=clean["timestamp"].min(),
    end=clean["timestamp"].max(),
    freq="h",
)

actual_index = pd.DatetimeIndex(clean["timestamp"])

missing_timestamps = expected_index.difference(actual_index)

print("\nTIME-SERIES VALIDATION")
print("-" * 70)

print(f"Start:              {clean['timestamp'].min()}")
print(f"End:                {clean['timestamp'].max()}")
print(f"Rows:               {len(clean):,}")
print(f"Missing timestamps: {len(missing_timestamps)}")

# ------------------------------------------------------------
# 10. Final quality checks
# ------------------------------------------------------------

remaining_missing = int(clean.isna().sum().sum())
remaining_duplicates = int(
    clean["timestamp"].duplicated().sum()
)

print(f"Missing values:     {remaining_missing}")
print(f"Duplicate times:    {remaining_duplicates}")

if len(clean) != 8760:
    print(
        "\nWARNING: Expected 8,760 observations "
        "for non-leap-year 2022."
    )

if len(missing_timestamps) != 0:
    raise RuntimeError(
        "Hourly timestamp continuity check failed."
    )

if remaining_missing != 0:
    raise RuntimeError(
        "Missing values remain after cleaning."
    )

if remaining_duplicates != 0:
    raise RuntimeError(
        "Duplicate timestamps remain."
    )

# ------------------------------------------------------------
# 11. Save clean dataset
# ------------------------------------------------------------

clean.to_csv(
    OUTPUT_FILE,
    index=False,
)

print("\n" + "=" * 70)
print("CLEAN PV DATASET CREATED SUCCESSFULLY")
print("=" * 70)

print(f"\nSaved to:\n{OUTPUT_FILE}")

print("\nFinal columns:")

for col in clean.columns:
    print(f"  - {col}")

print("\nFirst five observations:")
print(clean.head().to_string(index=False))

print("\nFinal five observations:")
print(clean.tail().to_string(index=False))

print(
    "\nIMPORTANT:\n"
    "Raw NLR.zip was NOT modified.\n"
    "No normalization was applied.\n"
    "No synthetic PV power was generated.\n"
)