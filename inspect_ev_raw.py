from pathlib import Path
import json
from collections import Counter
import pandas as pd
# ============================================================
# 1. PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(
    r"D:\Molvi paper review\FC_HMARL"
)

EV_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "ev"
)

EV_FILE = (
    EV_DIR
    / "acndata_sessions.json"
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


def parse_datetime(series):

    return pd.to_datetime(
        series,
        errors="coerce",
        utc=True,
    )


def parse_numeric(series):

    return pd.to_numeric(
        series,
        errors="coerce",
    )


# ============================================================
# 3. START
# ============================================================

section(
    "FC-HMARL - ACN-DATA RAW EV INSPECTION"
)


# ============================================================
# 4. VERIFY RAW DATA DIRECTORY
# ============================================================

if not EV_DIR.exists():

    raise FileNotFoundError(
        "EV raw-data directory does not exist:\n"
        f"{EV_DIR}"
    )


# ============================================================
# 5. SHOW FILES IN EV DIRECTORY
# ============================================================

subsection(
    "FILES DETECTED IN RAW EV DIRECTORY"
)

files = sorted(
    [
        f
        for f in EV_DIR.iterdir()
        if f.is_file()
        and f.name != ".gitkeep"
    ],
    key=lambda x: x.name.lower(),
)

if not files:

    raise FileNotFoundError(
        "No files were found in:\n"
        f"{EV_DIR}"
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
subsection(
    "SELECTED ACN DATASET"
)

if not EV_FILE.exists():

    raise FileNotFoundError(
        "Completed merged ACN dataset was not found.\n\n"
        "Expected file:\n"
        f"{EV_FILE}"
    )


print(
    f"Selected file:\n{EV_FILE}"
)

print(
    f"\nFile size: "
    f"{EV_FILE.stat().st_size / (1024 ** 2):.3f} MB"
)


# ============================================================
# 7. LOAD JSON
# ============================================================

subsection(
    "LOADING JSON"
)

try:

    with open(
        EV_FILE,
        "r",
        encoding="utf-8",
    ) as file:

        raw = json.load(
            file
        )

except json.JSONDecodeError as exc:

    raise RuntimeError(
        "\nACN JSON parsing failed.\n"
        "The JSON file may be incomplete or corrupted.\n\n"
        f"Error:\n{exc}"
    ) from exc


print(
    "JSON loaded successfully."
)

print(
    f"Top-level Python type: "
    f"{type(raw).__name__}"
)


# ============================================================
# 8. TOP-LEVEL STRUCTURE
# ============================================================

subsection(
    "TOP-LEVEL STRUCTURE"
)


if isinstance(
    raw,
    dict,
):

    print(
        "Top-level keys:"
    )

    for key in raw.keys():

        print(
            f"  - {key}"
        )


elif isinstance(
    raw,
    list,
):

    print(
        f"Top-level list length: "
        f"{len(raw):,}"
    )


else:

    raise RuntimeError(
        "Unexpected JSON structure.\n"
        "Expected dictionary or list."
    )


# ============================================================
# 9. DISPLAY MERGED DATASET METADATA
# ============================================================

if (
    isinstance(raw, dict)
    and "_meta" in raw
):

    subsection(
        "DATASET METADATA"
    )

    meta = raw.get(
        "_meta",
        {}
    )

    if isinstance(
        meta,
        dict,
    ):

        for key, value in meta.items():

            print(
                f"{key:30s}: {value}"
            )


# ============================================================
# 10. LOCATE SESSION RECORDS
# ============================================================

sessions = None

session_container = None


if isinstance(
    raw,
    list,
):

    sessions = raw

    session_container = (
        "top-level list"
    )


elif isinstance(
    raw,
    dict,
):

    possible_keys = [
        "_items",
        "items",
        "sessions",
        "data",
        "results",
    ]

    for key in possible_keys:

        if (
            key in raw
            and isinstance(
                raw[key],
                list,
            )
        ):

            sessions = raw[key]

            session_container = key

            break


if sessions is None:

    raise RuntimeError(
        "No EV session list was found "
        "inside the JSON file."
    )


print()
print(
    f"Session data found under: "
    f"{session_container}"
)


# ============================================================
# 11. SESSION COUNT
# ============================================================

subsection(
    "SESSION COUNT"
)

print(
    f"Number of ACN session records: "
    f"{len(sessions):,}"
)


if len(
    sessions
) == 0:

    raise RuntimeError(
        "ACN session list is empty."
    )


# ============================================================
# 12. CHECK RECORD TYPES
# ============================================================

non_dictionary_records = sum(

    not isinstance(
        record,
        dict,
    )

    for record in sessions
)


print(
    f"Non-dictionary records: "
    f"{non_dictionary_records:,}"
)


if non_dictionary_records > 0:

    raise RuntimeError(
        "Unexpected records were found "
        "inside the session list."
    )


# ============================================================
# 13. FIRST SESSION KEYS
# ============================================================

first_session = sessions[0]


subsection(
    "FIRST SESSION KEYS"
)


for key in first_session.keys():

    print(
        key
    )


# ============================================================
# 14. CREATE DATAFRAME
# ============================================================

df = pd.json_normalize(
    sessions,
    sep=".",
)


subsection(
    "DATAFRAME STRUCTURE"
)


print(
    f"Rows:    "
    f"{len(df):,}"
)

print(
    f"Columns: "
    f"{len(df.columns):,}"
)


# ============================================================
# 15. ALL COLUMN NAMES
# ============================================================

subsection(
    "COLUMN NAMES"
)


for number, column in enumerate(
    df.columns,
    start=1,
):

    print(
        f"{number:03d}. "
        f"{column}"
    )


# ============================================================
# 16. FIRST 5 RECORDS
# ============================================================

subsection(
    "FIRST 5 SESSION RECORDS"
)


display_columns = [

    column

    for column in [

        "sessionID",
        "stationID",
        "siteID",
        "connectionTime",
        "disconnectTime",
        "doneChargingTime",
        "kWhDelivered",
        "timezone",

    ]

    if column
    in df.columns
]


if display_columns:

    print(

        df[
            display_columns
        ]

        .head()

        .to_string(
            index=False
        )
    )


else:

    print(

        df
        .head()
        .to_string(
            index=False
        )
    )


# ============================================================
# 17. REQUIRED FIELD CHECK
# ============================================================

required_fields = [

    "connectionTime",
    "disconnectTime",
    "kWhDelivered",
    "stationID",

]


subsection(
    "REQUIRED FIELD CHECK"
)


for field in required_fields:

    if field in df.columns:

        print(
            f"[OK]      "
            f"{field}"
        )

    else:

        print(
            f"[MISSING] "
            f"{field}"
        )


# ============================================================
# 18. OPTIONAL FIELD CHECK
# ============================================================

optional_fields = [

    "doneChargingTime",
    "sessionID",
    "siteID",
    "timezone",
    "userInputs",

]


subsection(
    "OPTIONAL FIELD CHECK"
)


for field in optional_fields:

    found = (

        field in df.columns

        or any(

            column.startswith(
                field + "."
            )

            for column
            in df.columns
        )
    )


    if found:

        print(
            f"[FOUND]     "
            f"{field}"
        )

    else:

        print(
            f"[NOT FOUND] "
            f"{field}"
        )


# ============================================================
# 19. MISSING VALUES
# ============================================================

subsection(
    "CORE VARIABLE MISSING VALUES"
)


core_fields = [

    "connectionTime",
    "disconnectTime",
    "doneChargingTime",
    "kWhDelivered",
    "stationID",
    "sessionID",
    "siteID",
    "timezone",

]


for field in core_fields:

    if field in df.columns:

        missing = int(

            df[
                field
            ]

            .isna()

            .sum()
        )


        print(
            f"{field:25s}: "
            f"{missing:,}"
        )


# ============================================================
# 20. TIMESTAMP ANALYSIS
# ============================================================

subsection(
    "TIMESTAMP ANALYSIS"
)


parsed_timestamps = {}


timestamp_fields = [

    "connectionTime",
    "disconnectTime",
    "doneChargingTime",

]


for field in timestamp_fields:

    if field not in df.columns:

        continue


    parsed = parse_datetime(

        df[
            field
        ]
    )


    parsed_timestamps[
        field
    ] = parsed


    print()

    print(
        field
    )


    print(
        f"  Valid timestamps  : "
        f"{parsed.notna().sum():,}"
    )


    print(
        f"  Invalid timestamps: "
        f"{parsed.isna().sum():,}"
    )


    if parsed.notna().any():

        print(
            f"  Earliest: "
            f"{parsed.min()}"
        )

        print(
            f"  Latest  : "
            f"{parsed.max()}"
        )


# ============================================================
# 21. YEAR DISTRIBUTION
# ============================================================

if (
    "connectionTime"
    in parsed_timestamps
):

    subsection(
        "CONNECTION YEAR DISTRIBUTION"
    )


    year_counts = (

        parsed_timestamps[
            "connectionTime"
        ]

        .dt.year

        .value_counts()

        .sort_index()
    )


    print(
        year_counts
        .to_string()
    )


# ============================================================
# 22. MONTH DISTRIBUTION
# ============================================================

if (
    "connectionTime"
    in parsed_timestamps
):

    subsection(
        "CONNECTION MONTH DISTRIBUTION"
    )


    month_counts = (

        parsed_timestamps[
            "connectionTime"
        ]

        .dt.month

        .value_counts()

        .sort_index()
    )


    print(
        month_counts
        .to_string()
    )


# ============================================================
# 23. ENERGY DELIVERY ANALYSIS
# ============================================================

subsection(
    "ENERGY DELIVERY ANALYSIS"
)


if (
    "kWhDelivered"
    in df.columns
):

    energy = parse_numeric(

        df[
            "kWhDelivered"
        ]
    )


    print(

        energy

        .describe()

        .to_string()
    )


    print()


    print(
        f"Missing/non-numeric : "
        f"{energy.isna().sum():,}"
    )


    print(
        f"Zero-energy sessions: "
        f"{(energy == 0).sum():,}"
    )


    print(
        f"Negative-energy     : "
        f"{(energy < 0).sum():,}"
    )


    print(
        f"Positive-energy     : "
        f"{(energy > 0).sum():,}"
    )


# ============================================================
# 24. STATION SUMMARY
# ============================================================

subsection(
    "STATION SUMMARY"
)


if (
    "stationID"
    in df.columns
):

    print(
        f"Unique stations: "
        f"{df['stationID'].nunique(dropna=True):,}"
    )


    print()

    print(
        "Top 20 stations:"
    )


    print(

        df[
            "stationID"
        ]

        .value_counts(
            dropna=False
        )

        .head(
            20
        )

        .to_string()
    )


# ============================================================
# 25. SITE SUMMARY
# ============================================================

if (
    "siteID"
    in df.columns
):

    subsection(
        "SITE SUMMARY"
    )


    print(

        df[
            "siteID"
        ]

        .value_counts(
            dropna=False
        )

        .to_string()
    )


# ============================================================
# 26. TIMEZONE SUMMARY
# ============================================================

if (
    "timezone"
    in df.columns
):

    subsection(
        "TIMEZONE SUMMARY"
    )


    print(

        df[
            "timezone"
        ]

        .value_counts(
            dropna=False
        )

        .to_string()
    )


# ============================================================
# 27. CONNECTION / PARKING DURATION
# ============================================================

if (

    "connectionTime"
    in parsed_timestamps

    and

    "disconnectTime"
    in parsed_timestamps

):

    connection = (

        parsed_timestamps[
            "connectionTime"
        ]
    )


    disconnection = (

        parsed_timestamps[
            "disconnectTime"
        ]
    )


    duration_hours = (

        disconnection
        - connection

    ).dt.total_seconds() / 3600.0


    subsection(
        "CONNECTION / PARKING DURATION"
    )


    print(

        duration_hours

        .describe()

        .to_string()
    )


    print()


    print(
        f"Missing durations     : "
        f"{duration_hours.isna().sum():,}"
    )


    print(
        f"Non-positive durations: "
        f"{(duration_hours <= 0).sum():,}"
    )


    print(
        f"Durations > 24 h      : "
        f"{(duration_hours > 24).sum():,}"
    )


    print(
        f"Durations > 48 h      : "
        f"{(duration_hours > 48).sum():,}"
    )


# ============================================================
# 28. ACTIVE CHARGING DURATION
# ============================================================

if (

    "connectionTime"
    in parsed_timestamps

    and

    "doneChargingTime"
    in parsed_timestamps

):

    connection = (

        parsed_timestamps[
            "connectionTime"
        ]
    )


    done_charging = (

        parsed_timestamps[
            "doneChargingTime"
        ]
    )


    charging_duration_hours = (

        done_charging
        - connection

    ).dt.total_seconds() / 3600.0


    subsection(
        "ACTIVE CHARGING DURATION"
    )


    print(

        charging_duration_hours

        .describe()

        .to_string()
    )


    print()


    print(
        f"Missing charging durations: "
        f"{charging_duration_hours.isna().sum():,}"
    )


    print(
        f"Non-positive durations     : "
        f"{(charging_duration_hours <= 0).sum():,}"
    )


# ============================================================
# 29. DUPLICATE SESSION CHECK
# ============================================================

subsection(
    "DUPLICATE CHECK"
)


if (
    "sessionID"
    in df.columns
):

    missing_session_ids = int(

        df[
            "sessionID"
        ]

        .isna()

        .sum()
    )


    valid_session_ids = (

        df.loc[
            df["sessionID"].notna(),
            "sessionID",
        ]
    )


    duplicate_session_ids = int(

        valid_session_ids

        .duplicated()

        .sum()
    )


    print(
        f"Missing session IDs  : "
        f"{missing_session_ids:,}"
    )


    print(
        f"Duplicate session IDs: "
        f"{duplicate_session_ids:,}"
    )


else:

    duplicate_rows = int(

        df

        .duplicated()

        .sum()
    )


    print(
        f"Duplicate rows: "
        f"{duplicate_rows:,}"
    )


# ============================================================
# 30. USER INPUT INSPECTION
# ============================================================

subsection(
    "USER INPUT INFORMATION"
)


if (
    "userInputs"
    in df.columns
):

    user_input_series = (

        df[
            "userInputs"
        ]
    )


    sessions_with_user_inputs = 0

    total_user_input_records = 0

    user_input_keys = Counter()

    examples = []


    for value in user_input_series:


        # ----------------------------------------------------
        # ACN normally stores userInputs as a list of objects
        # ----------------------------------------------------

        if isinstance(
            value,
            list,
        ):

            if len(value) > 0:

                sessions_with_user_inputs += 1


            total_user_input_records += len(
                value
            )


            for item in value:

                if isinstance(
                    item,
                    dict,
                ):

                    user_input_keys.update(
                        item.keys()
                    )


                    if len(
                        examples
                    ) < 5:

                        examples.append(
                            item
                        )


        # ----------------------------------------------------
        # Also support dictionary representation
        # ----------------------------------------------------

        elif isinstance(
            value,
            dict,
        ):

            if value:

                sessions_with_user_inputs += 1


            total_user_input_records += 1


            user_input_keys.update(
                value.keys()
            )


            if len(
                examples
            ) < 5:

                examples.append(
                    value
                )


    print(
        f"Sessions with userInputs : "
        f"{sessions_with_user_inputs:,}"
    )


    print(
        f"Total user-input records : "
        f"{total_user_input_records:,}"
    )


    print()

    print(
        "Observed userInputs keys:"
    )


    if user_input_keys:

        for (
            key,
            count
        ) in user_input_keys.most_common():

            print(
                f"  "
                f"{key:30s} "
                f"{count:,}"
            )


    else:

        print(
            "  No populated userInputs "
            "records detected."
        )


    if examples:

        print()

        print(
            "First 5 userInputs examples:"
        )


        for number, example in enumerate(
            examples,
            start=1,
        ):

            print(
                f"  Example "
                f"{number}: "
                f"{example}"
            )


else:

    flattened_columns = [

        column

        for column
        in df.columns

        if column.startswith(
            "userInputs."
        )
    ]


    if flattened_columns:

        print(
            "Flattened userInputs "
            "columns detected:"
        )


        for column in flattened_columns:

            print(
                f"  {column}"
            )


    else:

        print(
            "No userInputs information detected."
        )


# ============================================================
# 31. ARRIVAL / CONNECTION HOUR
# ============================================================
#
# IMPORTANT:
# pd.to_datetime(..., utc=True) converts the ACN timestamps
# to UTC for consistent validation.
#
# These distributions are inspection only.
# Later, behavioural modelling can explicitly convert to
# the appropriate local timezone.
# ============================================================

if (
    "connectionTime"
    in parsed_timestamps
):

    subsection(
        "ARRIVAL / CONNECTION HOUR DISTRIBUTION - UTC"
    )


    arrival_hour = (

        parsed_timestamps[
            "connectionTime"
        ]

        .dt.hour

        .value_counts()

        .sort_index()
    )


    print(
        arrival_hour
        .to_string()
    )


# ============================================================
# 32. DEPARTURE / DISCONNECTION HOUR
# ============================================================

if (
    "disconnectTime"
    in parsed_timestamps
):

    subsection(
        "DEPARTURE / DISCONNECTION HOUR DISTRIBUTION - UTC"
    )


    departure_hour = (

        parsed_timestamps[
            "disconnectTime"
        ]

        .dt.hour

        .value_counts()

        .sort_index()
    )


    print(
        departure_hour
        .to_string()
    )


# ============================================================
# 33. DAY-OF-WEEK DISTRIBUTION
# ============================================================

if (
    "connectionTime"
    in parsed_timestamps
):

    subsection(
        "CONNECTION DAY-OF-WEEK DISTRIBUTION"
    )


    day_counts = (

        parsed_timestamps[
            "connectionTime"
        ]

        .dt.day_name()

        .value_counts()
    )


    ordered_days = [

        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",

    ]


    day_counts = (

        day_counts

        .reindex(
            ordered_days,
            fill_value=0,
        )
    )


    print(
        day_counts
        .to_string()
    )


# ============================================================
# 34. BASIC DATA QUALITY FLAGS
# ============================================================

subsection(
    "BASIC DATA QUALITY FLAGS"
)


if (
    "connectionTime"
    in parsed_timestamps
    and
    "disconnectTime"
    in parsed_timestamps
):
    invalid_order = (

        parsed_timestamps[
            "disconnectTime"
        ]

        <=

        parsed_timestamps[
            "connectionTime"
        ]
    )


    print(
        f"Disconnect <= connection : "
        f"{invalid_order.sum():,}"
    )


if (
    "kWhDelivered"
    in df.columns
):

    energy = parse_numeric(

        df[
            "kWhDelivered"
        ]
    )


    print(
        f"Negative delivered energy: "
        f"{(energy < 0).sum():,}"
    )


    print(
        f"Zero delivered energy    : "
        f"{(energy == 0).sum():,}"
    )


# ============================================================
# 35. FINAL SUMMARY
# ============================================================

section(
    "ACN-DATA RAW EV INSPECTION COMPLETED"
)


print(
    f"Raw source file:\n"
    f"{EV_FILE}"
)


print()

print(
    f"Total session records: "
    f"{len(sessions):,}"
)


print()

print(
    "IMPORTANT:"
)


print(
    "No ACN sessions were modified."
)


print(
    "No records were deleted."
)


print(
    "No interpolation was performed."
)


print(
    "No synthetic EV variables were created."
)


print(
    "No EV battery capacity was assumed."
)


print(
    "No arrival SOC was calculated."
)


print(
    "No target SOC was calculated."
)


print(
    "No driving distance was created."
)


print(
    "No charging/V2G power trajectory was constructed."
)


print(
    "No 24-point EV trajectory was constructed."
)


print(
    "No 96-point EV trajectory was constructed."
)


print(
    "No normalization was performed."
)


print()

print(
    "This is an inspection-only stage."
)

print(
    "The raw ACN dataset remains unchanged."
)
