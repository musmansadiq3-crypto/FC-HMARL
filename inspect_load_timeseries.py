from pathlib import Path
import pandas as pd

# ============================================================
# FC-HMARL - STEP 2B
# PECAN STREET 15-MINUTE LOAD TIME-SERIES INSPECTION
# ============================================================

PROJECT_ROOT = Path(r"D:\Molvi paper review\FC_HMARL")

LOAD_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "load"
    / "15minute_data_california"
    / "15minute_data_california.csv"
)

print("=" * 80)
print("FC-HMARL PECAN STREET 15-MINUTE TIME-SERIES INSPECTION")
print("=" * 80)

if not LOAD_FILE.exists():
    raise FileNotFoundError(
        f"Time-series file not found:\n{LOAD_FILE}"
    )

print(f"\nTime-series file found:\n{LOAD_FILE}")
print(
    f"File size: "
    f"{LOAD_FILE.stat().st_size / (1024 * 1024):.2f} MB"
)

# ============================================================
# 1. READ ONLY A SMALL SAMPLE FIRST
# ============================================================

print("\nReading first 20 rows for structure inspection...")

sample = pd.read_csv(
    LOAD_FILE,
    nrows=20,
    low_memory=False
)

print("\nCOLUMN NAMES")
print("-" * 80)

for i, col in enumerate(sample.columns, start=1):
    print(f"{i:03d}. {col}")

print("\nFIRST 10 ROWS")
print("-" * 80)

print(
    sample.head(10).to_string(
        index=False
    )
)

print("\nDATA TYPES FROM SAMPLE")
print("-" * 80)

print(
    sample.dtypes.to_string()
)

# ============================================================
# 2. IDENTIFY IMPORTANT COLUMNS
# ============================================================

possible_id_columns = [
    col
    for col in sample.columns
    if any(
        key in col.lower()
        for key in [
            "dataid",
            "house",
            "home",
            "site"
        ]
    )
]

possible_time_columns = [
    col
    for col in sample.columns
    if any(
        key in col.lower()
        for key in [
            "time",
            "date",
            "local",
            "utc"
        ]
    )
]

possible_load_columns = [
    col
    for col in sample.columns
    if any(
        key in col.lower()
        for key in [
            "grid",
            "use",
            "load",
            "house",
            "electric"
        ]
    )
]

print("\nPOSSIBLE HOUSEHOLD ID COLUMNS")
print("-" * 80)

if possible_id_columns:
    for col in possible_id_columns:
        print(col)
else:
    print("No obvious household ID column found.")

print("\nPOSSIBLE TIME COLUMNS")
print("-" * 80)

if possible_time_columns:
    for col in possible_time_columns:
        print(col)
else:
    print("No obvious timestamp column found.")

print("\nPOSSIBLE LOAD / POWER COLUMNS")
print("-" * 80)

if possible_load_columns:
    for col in possible_load_columns:
        print(col)
else:
    print("No obvious load column found.")

# ============================================================
# 3. FULL FILE - ONLY BASIC STRUCTURAL COUNTS
# ============================================================

print("\n" + "=" * 80)
print("READING FULL FILE FOR BASIC COUNTS")
print("=" * 80)

df = pd.read_csv(
    LOAD_FILE,
    low_memory=False
)

print(f"\nTotal rows:    {len(df):,}")
print(f"Total columns: {len(df.columns):,}")

# ============================================================
# 4. DUPLICATES
# ============================================================

duplicate_rows = int(
    df.duplicated().sum()
)

print("\nDUPLICATE ROWS")
print("-" * 80)

print(
    f"Duplicate rows: {duplicate_rows:,}"
)

# ============================================================
# 5. MISSING VALUES
# ============================================================

print("\nMISSING VALUES")
print("-" * 80)

missing = (
    df
    .isna()
    .sum()
    .sort_values(
        ascending=False
    )
)

missing_nonzero = missing[
    missing > 0
]

if missing_nonzero.empty:
    print("No missing values detected.")
else:
    print(
        missing_nonzero.to_string()
    )

# ============================================================
# 6. HOUSEHOLD COUNTS
# ============================================================

if "dataid" in df.columns:

    print("\nHOUSEHOLD SUMMARY")
    print("-" * 80)

    print(
        f"Unique dataid values: "
        f"{df['dataid'].nunique(dropna=True):,}"
    )

    print("\nTop households by number of records:")

    print(
        df["dataid"]
        .value_counts()
        .head(20)
        .to_string()
    )

# ============================================================
# 7. TIMESTAMP ANALYSIS
# ============================================================

timestamp_column = None

preferred_time_names = [
    "local_15min",
    "localminute",
    "local_time",
    "timestamp",
    "datetime",
    "utc_15min"
]

for name in preferred_time_names:
    if name in df.columns:
        timestamp_column = name
        break

if timestamp_column is None and possible_time_columns:
    timestamp_column = possible_time_columns[0]

print("\nTIMESTAMP ANALYSIS")
print("-" * 80)

if timestamp_column is not None:

    print(
        f"Selected timestamp column: "
        f"{timestamp_column}"
    )

    timestamps = pd.to_datetime(
        df[timestamp_column],
        errors="coerce"
    )

    print(
        f"Valid timestamps: "
        f"{timestamps.notna().sum():,}"
    )

    print(
        f"Invalid timestamps: "
        f"{timestamps.isna().sum():,}"
    )

    print(
        f"Start timestamp: "
        f"{timestamps.min()}"
    )

    print(
        f"End timestamp:   "
        f"{timestamps.max()}"
    )

else:

    print(
        "No timestamp column could be identified."
    )

# ============================================================
# 8. GRID / LOAD VARIABLE ANALYSIS
# ============================================================

print("\nGRID / LOAD VARIABLE ANALYSIS")
print("-" * 80)

if "grid" in df.columns:

    grid_numeric = pd.to_numeric(
        df["grid"],
        errors="coerce"
    )

    print(
        f"Grid valid numeric values: "
        f"{grid_numeric.notna().sum():,}"
    )

    print(
        f"Grid missing/non-numeric: "
        f"{grid_numeric.isna().sum():,}"
    )

    print(
        f"Grid minimum: "
        f"{grid_numeric.min()}"
    )

    print(
        f"Grid maximum: "
        f"{grid_numeric.max()}"
    )

    print(
        f"Grid mean: "
        f"{grid_numeric.mean()}"
    )

    print(
        f"Grid median: "
        f"{grid_numeric.median()}"
    )

    print(
        f"Negative grid values: "
        f"{(grid_numeric < 0).sum():,}"
    )

    print(
        f"Zero grid values: "
        f"{(grid_numeric == 0).sum():,}"
    )

else:

    print(
        "'grid' column was not found."
    )

# ============================================================
# 9. NUMERIC SUMMARY OF KEY COLUMNS
# ============================================================

key_columns = []

for col in [
    "grid",
    "solar",
    "use",
    "dataid"
]:
    if col in df.columns:
        key_columns.append(col)

print("\nKEY COLUMN SUMMARY")
print("-" * 80)

for col in key_columns:

    numeric_col = pd.to_numeric(
        df[col],
        errors="coerce"
    )

    print(f"\nColumn: {col}")

    print(
        numeric_col
        .describe()
        .to_string()
    )
# ============================================================
# 10. FINAL MESSAGE
# ============================================================

print("\n" + "=" * 80)
print("PECAN STREET TIME-SERIES INSPECTION COMPLETED")
print("=" * 80)

print(
    "\nNo raw Pecan Street data were modified.\n"
    "No 15-minute to hourly conversion was performed.\n"
    "No interpolation was performed.\n"
    "No normalization was performed."
)
