from pathlib import Path
import zipfile
import pandas as pd

# ============================================================
# FC-HMARL - STEP 1
# RAW NSRDB / PV DATA INSPECTION
# ============================================================

PROJECT_ROOT = Path(r"D:\Molvi paper review\FC_HMARL")
PV_ZIP = PROJECT_ROOT / "data" / "raw" / "pv" / "NLR.zip"

print("=" * 70)
print("FC-HMARL RAW PV DATA INSPECTION")
print("=" * 70)

if not PV_ZIP.exists():
    raise FileNotFoundError(f"File not found:\n{PV_ZIP}")

print(f"\nRaw ZIP found:\n{PV_ZIP}")
print(f"File size: {PV_ZIP.stat().st_size / 1024:.2f} KB")

# ------------------------------------------------------------
# 1. Inspect ZIP contents
# ------------------------------------------------------------
with zipfile.ZipFile(PV_ZIP, "r") as z:
    files = z.namelist()

    print("\nFILES INSIDE ZIP")
    print("-" * 70)

    for i, file_name in enumerate(files, start=1):
        info = z.getinfo(file_name)
        print(
            f"{i:02d}. {file_name} "
            f"({info.file_size / 1024:.2f} KB)"
        )

    csv_files = [
        f for f in files
        if f.lower().endswith(".csv")
    ]

    if not csv_files:
        raise RuntimeError("No CSV file was found inside NLR.zip")

    print("\nCSV FILE(S) FOUND")
    print("-" * 70)

    for csv_file in csv_files:
        print(csv_file)

    # --------------------------------------------------------
    # 2. Read the first CSV without modifying anything
    # --------------------------------------------------------
    target_csv = csv_files[0]

    print(f"\nInspecting:\n{target_csv}")

    with z.open(target_csv) as f:
        # NSRDB files usually contain metadata in the first two rows
        metadata = pd.read_csv(f, nrows=1)

    print("\nMETADATA ROW")
    print("-" * 70)
    print(metadata.to_string(index=False))

    # Re-open because the stream has already been consumed
    with z.open(target_csv) as f:
        df = pd.read_csv(f, skiprows=2)

# ------------------------------------------------------------
# 3. Basic dataset inspection
# ------------------------------------------------------------
print("\n" + "=" * 70)
print("DATASET SUMMARY")
print("=" * 70)

print(f"Rows:    {len(df):,}")
print(f"Columns: {len(df.columns)}")

print("\nCOLUMN NAMES")
print("-" * 70)

for i, col in enumerate(df.columns, start=1):
    print(f"{i:02d}. {col}")

print("\nFIRST 10 ROWS")
print("-" * 70)
print(df.head(10).to_string(index=False))

print("\nLAST 10 ROWS")
print("-" * 70)
print(df.tail(10).to_string(index=False))

# ------------------------------------------------------------
# 4. Missing values
# ------------------------------------------------------------
print("\nMISSING VALUES")
print("-" * 70)

missing = df.isna().sum()

for col, count in missing.items():
    print(f"{col:<25} : {count}")

# ------------------------------------------------------------
# 5. Duplicate rows
# ------------------------------------------------------------
duplicate_count = df.duplicated().sum()

print("\nDUPLICATE ROWS")
print("-" * 70)
print(f"Duplicate rows: {duplicate_count}")

# ------------------------------------------------------------
# 6. Numeric statistics
# ------------------------------------------------------------
print("\nNUMERIC STATISTICS")
print("-" * 70)

print(df.describe().transpose().to_string())

# ------------------------------------------------------------
# 7. Date/time coverage
# ------------------------------------------------------------
required_time_columns = [
    "Year",
    "Month",
    "Day",
    "Hour",
    "Minute",
]

if all(col in df.columns for col in required_time_columns):

    timestamps = pd.to_datetime(
        dict(
            year=df["Year"],
            month=df["Month"],
            day=df["Day"],
            hour=df["Hour"],
            minute=df["Minute"],
        ),
        errors="coerce",
    )

    print("\nTIME COVERAGE")
    print("-" * 70)

    print(f"Start timestamp : {timestamps.min()}")
    print(f"End timestamp   : {timestamps.max()}")
    print(f"Invalid times   : {timestamps.isna().sum()}")

    time_diff = timestamps.sort_values().diff().dropna()

    print("\nMOST COMMON TIME INTERVALS")
    print("-" * 70)

    print(time_diff.value_counts().head(10).to_string())

else:
    print("\nWARNING: Expected NSRDB date/time columns were not all found.")

# ------------------------------------------------------------
# 8. Important PV/weather variables
# ------------------------------------------------------------
important_variables = [
    "GHI",
    "DNI",
    "DHI",
    "Temperature",
    "Wind Speed",
]

print("\nIMPORTANT VARIABLE RANGES")
print("-" * 70)

for col in important_variables:
    if col in df.columns:
        print(
            f"{col:<15} "
            f"min={df[col].min():10.4f}  "
            f"max={df[col].max():10.4f}  "
            f"mean={df[col].mean():10.4f}"
        )

print("\n" + "=" * 70)
print("RAW PV INSPECTION COMPLETED")
print("=" * 70)

print(
    "\nNo raw data were modified.\n"
    "The next step will be PV cleaning only after these results are reviewed."
)