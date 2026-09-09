from pathlib import Path
import zipfile
import io
import pandas as pd
# ============================================================
# 1. PROJECT PATHS
# ============================================================
PROJECT_ROOT = Path(
    r"D:\Molvi paper review\FC_HMARL"
)

PRICE_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "price"
)

PRICE_ZIP = (
    PRICE_DIR
    / "price_pjm.zip"
)


# ============================================================
# 2. HELPER FUNCTIONS
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
# 3. START
# ============================================================

section(
    "FC-HMARL - PJM RAW ELECTRICITY PRICE INSPECTION"
)


# ============================================================
# 4. VERIFY PRICE DIRECTORY
# ============================================================

if not PRICE_DIR.exists():

    raise FileNotFoundError(
        "Raw price-data directory does not exist:\n"
        f"{PRICE_DIR}"
    )


# ============================================================
# 5. SHOW FILES IN PRICE DIRECTORY
# ============================================================

subsection(
    "FILES DETECTED IN RAW PRICE DIRECTORY"
)


files = sorted(
    [
        f
        for f in PRICE_DIR.iterdir()
        if f.is_file()
        and f.name != ".gitkeep"
    ],
    key=lambda x: x.name.lower(),
)


if not files:

    raise FileNotFoundError(
        "No files were found in:\n"
        f"{PRICE_DIR}"
    )


for file_path in files:

    size_mb = (
        file_path.stat().st_size
        / (1024 ** 2)
    )

    print(
        f"{file_path.name:45s} "
        f"| {size_mb:10.3f} MB"
    )


# ============================================================
# 6. VERIFY PJM ZIP
# ============================================================

subsection(
    "SELECTED PJM ARCHIVE"
)


if not PRICE_ZIP.exists():

    raise FileNotFoundError(
        "PJM ZIP archive was not found.\n\n"
        "Expected:\n"
        f"{PRICE_ZIP}"
    )


print(
    f"Selected archive:\n"
    f"{PRICE_ZIP}"
)


print(
    f"\nArchive size: "
    f"{PRICE_ZIP.stat().st_size / (1024 ** 2):.3f} MB"
)


# ============================================================
# 7. OPEN ZIP
# ============================================================

subsection(
    "OPENING PJM ZIP ARCHIVE"
)


try:

    archive = zipfile.ZipFile(
        PRICE_ZIP,
        mode="r",
    )

except zipfile.BadZipFile as exc:

    raise RuntimeError(
        "The PJM file is not a valid ZIP archive."
    ) from exc


print(
    "ZIP archive opened successfully."
)


# ============================================================
# 8. ARCHIVE INTEGRITY TEST
# ============================================================

subsection(
    "ZIP INTEGRITY CHECK"
)


bad_file = archive.testzip()


if bad_file is None:

    print(
        "[OK] No corrupted archive members detected."
    )

else:

    print(
        "[WARNING] Corrupted archive member:"
    )

    print(
        bad_file
    )


# ============================================================
# 9. LIST ARCHIVE CONTENTS
# ============================================================

members = [
    name
    for name in archive.namelist()
    if not name.endswith("/")
]


subsection(
    "ARCHIVE CONTENTS"
)


print(
    f"Number of files inside ZIP: "
    f"{len(members):,}"
)


print()


for number, name in enumerate(
    members,
    start=1,
):

    print(
        f"{number:03d}. {name}"
    )


# ============================================================
# 10. CSV FILES
# ============================================================

csv_members = [
    name
    for name in members
    if name.lower().endswith(".csv")
]


subsection(
    "CSV FILES"
)


print(
    f"CSV files detected: "
    f"{len(csv_members):,}"
)


# ============================================================
# 11. SEARCH FOR 2019 FILES
# ============================================================

files_2019 = [
    name
    for name in csv_members
    if "2019" in name.lower()
]


subsection(
    "FILES CONTAINING '2019'"
)


print(
    f"2019 candidate files: "
    f"{len(files_2019):,}"
)


for name in files_2019:

    print(
        name
    )

# ============================================================
# 12. SEARCH FOR DAY-AHEAD FILES
# ============================================================

day_ahead_keywords = [

    "da",
    "day_ahead",
    "day-ahead",
    "dayahead",

]


day_ahead_candidates = []


for name in files_2019:

    lower_name = name.lower()

    if any(
        keyword in lower_name
        for keyword in day_ahead_keywords
    ):

        day_ahead_candidates.append(
            name
        )


subsection(
    "2019 DAY-AHEAD CANDIDATES"
)


if day_ahead_candidates:

    for name in day_ahead_candidates:

        print(
            name
        )

else:

    print(
        "No filename was automatically identified "
        "as a 2019 Day-Ahead file."
    )


# ============================================================
# 13. INSPECT ALL 2019 CANDIDATES
# ============================================================

subsection(
    "2019 CANDIDATE FILE STRUCTURES"
)


candidate_information = []


for name in files_2019:

    try:

        with archive.open(
            name
        ) as file:

            sample_bytes = file.read(
                2_000_000
            )


        sample = pd.read_csv(
            io.BytesIO(
                sample_bytes
            ),
            nrows=5,
        )


        columns = list(
            sample.columns
        )


        candidate_information.append(
            {
                "name": name,
                "columns": columns,
            }
        )


        print()

        print(
            f"FILE: {name}"
        )

        print(
            f"Columns ({len(columns)}):"
        )


        for column in columns:

            print(
                f"  - {column}"
            )


        print()

        print(
            "First rows:"
        )


        print(
            sample
            .head()
            .to_string(
                index=False
            )
        )


    except Exception as exc:

        print()

        print(
            f"[WARNING] Could not inspect:"
        )

        print(
            name
        )

        print(
            f"Reason: {exc}"
        )


# ============================================================
# 14. AUTOMATICALLY SEARCH FOR TOTAL_LMP_DA
# ============================================================

subsection(
    "SEARCHING FOR DAY-AHEAD LMP COLUMN"
)


files_with_total_lmp_da = []


for info in candidate_information:

    lower_columns = {
        str(column).lower():
        column

        for column
        in info["columns"]
    }


    if "total_lmp_da" in lower_columns:

        files_with_total_lmp_da.append(
            info["name"]
        )


if files_with_total_lmp_da:

    print(
        "Files containing total_lmp_da:"
    )

    for name in files_with_total_lmp_da:

        print(
            f"  {name}"
        )

else:

    print(
        "No 2019 file containing "
        "'total_lmp_da' was detected."
    )


# ============================================================
# 15. SELECT 2019 DAY-AHEAD FILE
# ============================================================
#
# Selection priority:
#
#   1. 2019 file containing total_lmp_da
#   2. filename identified as Day-Ahead
#
# If more than one file matches, we inspect rather than
# silently assuming which one is correct.
# ============================================================

selected_file = None


if len(
    files_with_total_lmp_da
) == 1:

    selected_file = (
        files_with_total_lmp_da[0]
    )


elif len(
    files_with_total_lmp_da
) > 1:

    print()

    print(
        "[NOTICE] Multiple 2019 files contain "
        "total_lmp_da."
    )

    print(
        "The script will inspect the first one, "
        "but the alternatives are listed above."
    )

    selected_file = (
        files_with_total_lmp_da[0]
    )


elif len(
    day_ahead_candidates
) == 1:

    selected_file = (
        day_ahead_candidates[0]
    )


elif len(
    day_ahead_candidates
) > 1:

    selected_file = (
        day_ahead_candidates[0]
    )


# ============================================================
# 16. LOAD SELECTED FILE
# ============================================================

if selected_file is not None:

    subsection(
        "SELECTED 2019 DAY-AHEAD CANDIDATE"
    )


    print(
        selected_file
    )


    with archive.open(
        selected_file
    ) as file:

        selected_df = pd.read_csv(
            file
        )


    print()

    print(
        f"Rows:    "
        f"{len(selected_df):,}"
    )

    print(
        f"Columns: "
        f"{len(selected_df.columns):,}"
    )


    # ========================================================
    # 17. COLUMN NAMES
    # ========================================================

    subsection(
        "SELECTED FILE COLUMN NAMES"
    )


    for number, column in enumerate(
        selected_df.columns,
        start=1,
    ):

        print(
            f"{number:03d}. "
            f"{column}"
        )


    # ========================================================
    # 18. FIRST 10 ROWS
    # ========================================================

    subsection(
        "FIRST 10 ROWS"
    )


    print(
        selected_df
        .head(10)
        .to_string(
            index=False
        )
    )

    # ========================================================
    # 19. MISSING VALUES
    # ========================================================

    subsection(
        "MISSING VALUES"
    )


    missing = (
        selected_df
        .isna()
        .sum()
    )


    for column in selected_df.columns:

        print(
            f"{column:35s}: "
            f"{missing[column]:,}"
        )


    # ========================================================
    # 20. NUMERIC COLUMN SUMMARY
    # ========================================================

    subsection(
        "NUMERIC COLUMN SUMMARY"
    )


    numeric_df = (
        selected_df
        .select_dtypes(
            include="number"
        )
    )


    if not numeric_df.empty:

        print(
            numeric_df
            .describe()
            .transpose()
            .to_string()
        )

    else:

        print(
            "No numeric columns automatically detected."
        )


    # ========================================================
    # 21. TOTAL_LMP_DA ANALYSIS
    # ========================================================

    matching_lmp_columns = [

        column

        for column
        in selected_df.columns

        if str(column).lower()
        == "total_lmp_da"
    ]


    if matching_lmp_columns:

        lmp_column = (
            matching_lmp_columns[0]
        )


        lmp = pd.to_numeric(

            selected_df[
                lmp_column
            ],

            errors="coerce",
        )


        subsection(
            "TOTAL_LMP_DA SUMMARY"
        )


        print(
            lmp
            .describe()
            .to_string()
        )


        print()


        print(
            f"Missing/non-numeric prices: "
            f"{lmp.isna().sum():,}"
        )


        print(
            f"Negative prices           : "
            f"{(lmp < 0).sum():,}"
        )


        print(
            f"Zero prices               : "
            f"{(lmp == 0).sum():,}"
        )


        print(
            f"Positive prices           : "
            f"{(lmp > 0).sum():,}"
        )


# ============================================================
# 22. CLOSE ARCHIVE
# ============================================================

archive.close()


# ============================================================
# 23. FINAL
# ============================================================

section(
    "PJM RAW PRICE INSPECTION COMPLETED"
)


print(
    f"Raw archive:\n"
    f"{PRICE_ZIP}"
)


print()

print(
    "No PJM price observations were modified."
)

print(
    "No missing prices were interpolated."
)

print(
    "No outliers were removed."
)

print(
    "No price normalization was performed."
)

print(
    "No 2019 observations were relabeled as 2022."
)

print(
    "No synthetic market prices were generated."
)

print()

print(
    "STEP 4A COMPLETE."
)
