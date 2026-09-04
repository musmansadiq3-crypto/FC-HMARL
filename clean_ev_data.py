# ============================================================
# FC-HMARL
# STEP 3B - CLEAN ACN-DATA EV SESSIONS
# ============================================================
#
# Input:
#   data/raw/ev/acndata_sessions.json
#
# Output:
#   data/processed/ev/ACN_EV_2018_2020_Clean.csv
#
# Purpose:
#   Clean the observed ACN charging-session data and derive
#   transparent behavioural variables from observed timestamps.
#
# IMPORTANT:
#   This script DOES NOT create:
#       - battery capacity
#       - arrival SOC
#       - target SOC
#       - synthetic driving distance
#       - V2G trajectory
#       - 24-point charging profile
#       - 96-point charging profile
#
# ============================================================


from pathlib import Path
import json
import numpy as np
import pandas as pd


# ============================================================
# 1. PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(
    r"D:\Molvi paper review\FC_HMARL"
)

RAW_EV_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "ev"
    / "acndata_sessions.json"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ev"
)

OUTPUT_FILE = (
    OUTPUT_DIR
    / "ACN_EV_2018_2020_Clean.csv"
)


# ============================================================
# 2. CONSTANTS
# ============================================================

LOCAL_TIMEZONE = "America/Los_Angeles"

EXPECTED_SITE = "caltech"

MAX_REASONABLE_STAY_HOURS = 168.0

MAX_REASONABLE_CHARGING_HOURS = 168.0


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


def parse_datetime_utc(series):

    return pd.to_datetime(
        series,
        errors="coerce",
        utc=True,
    )


def to_local_time(series):

    return (
        series
        .dt
        .tz_convert(
            LOCAL_TIMEZONE
        )
    )


def numeric(series):

    return pd.to_numeric(
        series,
        errors="coerce",
    )


# ============================================================
# 4. START
# ============================================================

section(
    "FC-HMARL - ACN-DATA EV CLEANING"
)


# ============================================================
# 5. VERIFY INPUT
# ============================================================

if not RAW_EV_FILE.exists():

    raise FileNotFoundError(
        "Raw ACN file was not found:\n"
        f"{RAW_EV_FILE}"
    )


print(
    f"Input file:\n"
    f"{RAW_EV_FILE}"
)


# ============================================================
# 6. CREATE OUTPUT DIRECTORY
# ============================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# 7. LOAD RAW JSON
# ============================================================

subsection(
    "LOADING RAW ACN DATA"
)


with open(
    RAW_EV_FILE,
    "r",
    encoding="utf-8",
) as file:

    raw = json.load(
        file
    )


if isinstance(
    raw,
    dict,
):

    sessions = raw.get(
        "_items"
    )

elif isinstance(
    raw,
    list,
):

    sessions = raw

else:

    sessions = None


if sessions is None:

    raise RuntimeError(
        "Could not locate ACN session records."
    )


print(
    f"Raw sessions loaded: "
    f"{len(sessions):,}"
)


# ============================================================
# 8. NORMALIZE JSON
# ============================================================

df = pd.json_normalize(
    sessions,
    sep=".",
)


print(
    f"Raw dataframe shape: "
    f"{df.shape}"
)


# ============================================================
# 9. SAVE ORIGINAL ROW NUMBER
# ============================================================

df["raw_row_number"] = (
    np.arange(
        len(df)
    )
    + 1
)


# ============================================================
# 10. CORE IDENTIFIERS
# ============================================================

for column in [

    "sessionID",
    "stationID",
    "siteID",
    "clusterID",
    "spaceID",
    "userID",

]:

    if column not in df.columns:

        df[column] = pd.NA


# ============================================================
# 11. PARSE CORE TIMESTAMPS
# ============================================================

subsection(
    "PARSING TIMESTAMPS"
)


timestamp_columns = [

    "connectionTime",
    "disconnectTime",
    "doneChargingTime",

]


for column in timestamp_columns:

    if column not in df.columns:

        df[column] = pd.NA


df["connection_utc"] = parse_datetime_utc(
    df["connectionTime"]
)

df["disconnect_utc"] = parse_datetime_utc(
    df["disconnectTime"]
)

df["done_charging_utc"] = parse_datetime_utc(
    df["doneChargingTime"]
)


# ============================================================
# 12. CONVERT UTC TO LOS ANGELES LOCAL TIME
# ============================================================

df["arrival_local"] = to_local_time(
    df["connection_utc"]
)

df["departure_local"] = to_local_time(
    df["disconnect_utc"]
)

df["done_charging_local"] = to_local_time(
    df["done_charging_utc"]
)


# ============================================================
# 13. CORE ENERGY VARIABLE
# ============================================================

if "kWhDelivered" not in df.columns:

    df["kWhDelivered"] = np.nan


df["kwh_delivered"] = numeric(
    df["kWhDelivered"]
)


# ============================================================
# 14. DERIVE STAY DURATION
# ============================================================

df["stay_duration_h"] = (

    df["disconnect_utc"]
    - df["connection_utc"]

).dt.total_seconds() / 3600.0


# ============================================================
# 15. DERIVE ACTIVE CHARGING DURATION
# ============================================================

df["charging_duration_h"] = (

    df["done_charging_utc"]
    - df["connection_utc"]

).dt.total_seconds() / 3600.0


# ============================================================
# 16. ARRIVAL CHARACTERISTICS
# ============================================================

df["arrival_year"] = (
    df["arrival_local"]
    .dt
    .year
)

df["arrival_month"] = (
    df["arrival_local"]
    .dt
    .month
)

df["arrival_day"] = (
    df["arrival_local"]
    .dt
    .day
)

df["arrival_hour"] = (
    df["arrival_local"]
    .dt
    .hour
)

df["arrival_minute"] = (
    df["arrival_local"]
    .dt
    .minute
)

df["arrival_weekday"] = (
    df["arrival_local"]
    .dt
    .day_name()
)


# ============================================================
# 17. DEPARTURE CHARACTERISTICS
# ============================================================

df["departure_year"] = (
    df["departure_local"]
    .dt
    .year
)

df["departure_month"] = (
    df["departure_local"]
    .dt
    .month
)

df["departure_day"] = (
    df["departure_local"]
    .dt
    .day
)

df["departure_hour"] = (
    df["departure_local"]
    .dt
    .hour
)

df["departure_minute"] = (
    df["departure_local"]
    .dt
    .minute
)

df["departure_weekday"] = (
    df["departure_local"]
    .dt
    .day_name()
)


# ============================================================
# 18. WEEKEND FLAG
# ============================================================

df["is_weekend"] = (

    df["arrival_local"]
    .dt
    .weekday

    >= 5

).astype(
    "Int64"
)


# ============================================================
# 19. DERIVE ENERGY-TO-TIME RATIO
# ============================================================
#
# This is an observed-session average, not charger rated power.
# ============================================================

df["average_session_power_kw"] = (

    df["kwh_delivered"]
    / df["stay_duration_h"]

)


# ============================================================
# 20. EXTRACT USER INPUT VARIABLES
# ============================================================

subsection(
    "EXTRACTING USER INPUTS"
)


user_input_columns = {

    "miles_requested": [],
    "wh_per_mile": [],
    "minutes_available": [],
    "requested_departure_raw": [],
    "kwh_requested": [],
    "user_input_user_id": [],
    "user_input_available": [],

}


for _, row in df.iterrows():

    value = row.get(
        "userInputs",
        None,
    )


    selected = None


    if isinstance(
        value,
        list,
    ):

        if len(value) > 0:

            # ----------------------------------------------
            # If multiple userInputs exist for one session,
            # use the last available record.
            #
            # This represents the most recently recorded
            # user input for that charging session.
            # ----------------------------------------------

            for item in reversed(
                value
            ):

                if isinstance(
                    item,
                    dict,
                ):

                    selected = item
                    break


    elif isinstance(
        value,
        dict,
    ):

        selected = value


    if selected is None:

        user_input_columns[
            "miles_requested"
        ].append(
            np.nan
        )

        user_input_columns[
            "wh_per_mile"
        ].append(
            np.nan
        )

        user_input_columns[
            "minutes_available"
        ].append(
            np.nan
        )

        user_input_columns[
            "requested_departure_raw"
        ].append(
            None
        )

        user_input_columns[
            "kwh_requested"
        ].append(
            np.nan
        )

        user_input_columns[
            "user_input_user_id"
        ].append(
            np.nan
        )

        user_input_columns[
            "user_input_available"
        ].append(
            0
        )

        continue


    user_input_columns[
        "miles_requested"
    ].append(

        selected.get(
            "milesRequested",
            np.nan,
        )
    )


    user_input_columns[
        "wh_per_mile"
    ].append(

        selected.get(
            "WhPerMile",
            np.nan,
        )
    )


    user_input_columns[
        "minutes_available"
    ].append(

        selected.get(
            "minutesAvailable",
            np.nan,
        )
    )


    user_input_columns[
        "requested_departure_raw"
    ].append(

        selected.get(
            "requestedDeparture",
            None,
        )
    )


    user_input_columns[
        "kwh_requested"
    ].append(

        selected.get(
            "kWhRequested",
            np.nan,
        )
    )


    user_input_columns[
        "user_input_user_id"
    ].append(

        selected.get(
            "userID",
            np.nan,
        )
    )


    user_input_columns[
        "user_input_available"
    ].append(
        1
    )


# ============================================================
# 21. ADD USER INPUT COLUMNS
# ============================================================

for column, values in (
    user_input_columns.items()
):

    df[column] = values


# ============================================================
# 22. CONVERT USER INPUTS TO NUMERIC
# ============================================================

for column in [

    "miles_requested",
    "wh_per_mile",
    "minutes_available",
    "kwh_requested",
    "user_input_user_id",

]:

    df[column] = numeric(
        df[column]
    )


# ============================================================
# 23. REQUESTED DEPARTURE TIME
# ============================================================

df["requested_departure_utc"] = (

    pd.to_datetime(

        df[
            "requested_departure_raw"
        ],

        errors="coerce",
        utc=True,
    )
)


df["requested_departure_local"] = (

    df[
        "requested_departure_utc"
    ]

    .dt

    .tz_convert(
        LOCAL_TIMEZONE
    )
)


# ============================================================
# 24. USER-REQUESTED AVAILABLE HOURS
# ============================================================

df["requested_available_h"] = (

    df["minutes_available"]
    / 60.0
)


# ============================================================
# 25. REQUESTED DRIVING ENERGY
# ============================================================
#
# This uses only reported milesRequested and WhPerMile.
#
# E = miles * Wh/mile / 1000
#
# This is NOT battery capacity.
# ============================================================

df["requested_driving_energy_kwh"] = (

    df["miles_requested"]
    * df["wh_per_mile"]

    / 1000.0
)


# ============================================================
# 26. CHECK CONSISTENCY OF REPORTED USER REQUESTS
# ============================================================
#
# In many ACN user-input records:
#
# kWhRequested approximately equals:
#
# milesRequested * WhPerMile / 1000
#
# We retain both observed values separately.
# ============================================================

df["requested_energy_difference_kwh"] = (

    df["kwh_requested"]
    - df["requested_driving_energy_kwh"]
)


# ============================================================
# 27. DATA QUALITY FLAGS
# ============================================================

subsection(
    "CREATING DATA QUALITY FLAGS"
)


df["valid_connection_time"] = (

    df["connection_utc"]
    .notna()

).astype(
    int
)


df["valid_disconnect_time"] = (

    df["disconnect_utc"]
    .notna()

).astype(
    int
)


df["valid_energy"] = (

    df["kwh_delivered"]
    .notna()

    &

    (
        df["kwh_delivered"]
        > 0
    )

).astype(
    int
)


df["valid_stay_duration"] = (

    df["stay_duration_h"]
    .notna()

    &

    (
        df["stay_duration_h"]
        > 0
    )

    &

    (
        df["stay_duration_h"]
        <= MAX_REASONABLE_STAY_HOURS
    )

).astype(
    int
)


df["valid_charging_duration"] = (

    df["charging_duration_h"]
    .notna()

    &

    (
        df["charging_duration_h"]
        > 0
    )

    &

    (
        df["charging_duration_h"]
        <= MAX_REASONABLE_CHARGING_HOURS
    )

).astype(
    int
)


# ============================================================
# 28. CORE SESSION VALID FLAG
# ============================================================
#
# We do NOT require doneChargingTime because it is missing
# in some otherwise valid ACN sessions.
# ============================================================

df["valid_core_session"] = (

    (
        df["valid_connection_time"]
        == 1
    )

    &

    (
        df["valid_disconnect_time"]
        == 1
    )

    &

    (
        df["valid_energy"]
        == 1
    )

    &

    (
        df["valid_stay_duration"]
        == 1
    )

).astype(
    int
)


# ============================================================
# 29. DUPLICATE SESSION FLAG
# ============================================================

df["duplicate_session_id"] = (

    df["sessionID"]

    .duplicated(
        keep=False
    )

).astype(
    int
)


# ============================================================
# 30. SOURCE / PROVENANCE FIELDS
# ============================================================

df["source_dataset"] = (
    "ACN-Data"
)

df["source_site"] = (
    "Caltech"
)

df["source_timezone"] = (
    LOCAL_TIMEZONE
)

df["source_period"] = (
    "2018-2020"
)

df["source_api"] = (
    "ACN-Data REST API"
)


# ============================================================
# 31. NORMALIZE SITE ID REPRESENTATION
# ============================================================
#
# Raw data may contain:
#
#   "0002"
#   "2"
#
# These refer to the same Caltech site representation
# in this downloaded dataset.
#
# Preserve original site ID separately.
# ============================================================

df["site_id_raw"] = (
    df["siteID"]
    .astype(
        "string"
    )
)


df["site_id_numeric"] = (

    pd.to_numeric(
        df["siteID"],
        errors="coerce",
    )

    .astype(
        "Int64"
    )
)


# ============================================================
# 32. PREPARE CLEAN OUTPUT COLUMNS
# ============================================================

output_columns = [

    # --------------------------------------------------------
    # Provenance
    # --------------------------------------------------------

    "raw_row_number",
    "source_dataset",
    "source_site",
    "source_period",
    "source_timezone",
    "source_api",

    # --------------------------------------------------------
    # Identifiers
    # --------------------------------------------------------

    "sessionID",
    "stationID",
    "spaceID",
    "clusterID",
    "site_id_raw",
    "site_id_numeric",
    "userID",

    # --------------------------------------------------------
    # Observed UTC timestamps
    # --------------------------------------------------------

    "connection_utc",
    "disconnect_utc",
    "done_charging_utc",

    # --------------------------------------------------------
    # Local timestamps
    # --------------------------------------------------------

    "arrival_local",
    "departure_local",
    "done_charging_local",

    # --------------------------------------------------------
    # Behavioural variables
    # --------------------------------------------------------

    "arrival_year",
    "arrival_month",
    "arrival_day",
    "arrival_hour",
    "arrival_minute",
    "arrival_weekday",

    "departure_year",
    "departure_month",
    "departure_day",
    "departure_hour",
    "departure_minute",
    "departure_weekday",

    "is_weekend",

    # --------------------------------------------------------
    # Duration and energy
    # --------------------------------------------------------

    "stay_duration_h",
    "charging_duration_h",
    "kwh_delivered",
    "average_session_power_kw",

    # --------------------------------------------------------
    # User-input variables
    # --------------------------------------------------------

    "user_input_available",
    "miles_requested",
    "wh_per_mile",
    "minutes_available",
    "requested_available_h",
    "requested_departure_utc",
    "requested_departure_local",
    "kwh_requested",
    "requested_driving_energy_kwh",
    "requested_energy_difference_kwh",
    "user_input_user_id",

    # --------------------------------------------------------
    # Data-quality flags
    # --------------------------------------------------------

    "valid_connection_time",
    "valid_disconnect_time",
    "valid_energy",
    "valid_stay_duration",
    "valid_charging_duration",
    "valid_core_session",
    "duplicate_session_id",

]


# ============================================================
# 33. CREATE CLEAN DATAFRAME
# ============================================================

clean = df[
    output_columns
].copy()


# ============================================================
# 34. SORT CHRONOLOGICALLY
# ============================================================

clean = (

    clean

    .sort_values(
        by=[
            "connection_utc",
            "sessionID",
        ],
        kind="stable",
    )

    .reset_index(
        drop=True
    )
)


# ============================================================
# 35. SAVE CLEAN DATA
# ============================================================

subsection(
    "SAVING CLEAN EV DATA"
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
# 36. VALIDATION SUMMARY
# ============================================================

section(
    "CLEAN EV DATA VALIDATION"
)


print(
    f"Input sessions              : "
    f"{len(df):,}"
)

print(
    f"Output sessions             : "
    f"{len(clean):,}"
)

print(
    f"Core-valid sessions         : "
    f"{clean['valid_core_session'].sum():,}"
)

print(
    f"Core-invalid sessions       : "
    f"{(clean['valid_core_session'] == 0).sum():,}"
)

print(
    f"Valid charging durations    : "
    f"{clean['valid_charging_duration'].sum():,}"
)

print(
    f"Invalid/missing charge time : "
    f"{(clean['valid_charging_duration'] == 0).sum():,}"
)

print(
    f"Sessions with userInputs    : "
    f"{clean['user_input_available'].sum():,}"
)

print(
    f"Duplicate session IDs       : "
    f"{clean['duplicate_session_id'].sum():,}"
)


# ============================================================
# 37. ENERGY SUMMARY
# ============================================================

subsection(
    "DELIVERED ENERGY SUMMARY"
)


print(

    clean[
        "kwh_delivered"
    ]

    .describe()

    .to_string()
)


# ============================================================
# 38. STAY DURATION SUMMARY
# ============================================================

subsection(
    "STAY DURATION SUMMARY"
)


print(

    clean[
        "stay_duration_h"
    ]

    .describe()

    .to_string()
)


# ============================================================
# 39. USER INPUT COVERAGE
# ============================================================

subsection(
    "USER INPUT COVERAGE"
)


for column in [

    "miles_requested",
    "wh_per_mile",
    "minutes_available",
    "kwh_requested",
    "requested_departure_local",

]:

    available = int(

        clean[
            column
        ]

        .notna()

        .sum()
    )


    percentage = (

        available
        / len(clean)
        * 100.0
    )


    print(
        f"{column:30s}: "
        f"{available:6,d} "
        f"({percentage:6.2f}%)"
    )


# ============================================================
# 40. LOCAL ARRIVAL HOUR DISTRIBUTION
# ============================================================

subsection(
    "LOCAL ARRIVAL HOUR DISTRIBUTION"
)


print(

    clean[
        "arrival_hour"
    ]

    .value_counts()

    .sort_index()

    .to_string()
)


# ============================================================
# 41. LOCAL DEPARTURE HOUR DISTRIBUTION
# ============================================================

subsection(
    "LOCAL DEPARTURE HOUR DISTRIBUTION"
)


print(

    clean[
        "departure_hour"
    ]

    .value_counts()

    .sort_index()

    .to_string()
)


# ============================================================
# 42. YEAR DISTRIBUTION
# ============================================================

subsection(
    "SOURCE-YEAR DISTRIBUTION"
)


print(

    clean[
        "arrival_year"
    ]

    .value_counts()

    .sort_index()

    .to_string()
)


# ============================================================
# 43. FINAL
# ============================================================

section(
    "ACN EV CLEANING COMPLETED"
)


print(
    f"Final clean dataset:\n"
    f"{OUTPUT_FILE}"
)


print()

print(
    "The original raw ACN JSON file was not modified."
)

print(
    "The clean dataset preserves all original sessions."
)

print(
    "Invalid observations are flagged rather than silently deleted."
)

print(
    "Arrival and departure times are converted to "
    "America/Los_Angeles local time."
)

print(
    "User-input variables are retained only where actually observed."
)

print(
    "No missing userInputs were synthetically imputed."
)

print(
    "No battery capacities were generated."
)

print(
    "No SOC values were generated."
)

print(
    "No V2G trajectory was generated."
)

print(
    "No 96-point EV power profile was generated."
)

print()

print(
    "STEP 3B COMPLETE."
)