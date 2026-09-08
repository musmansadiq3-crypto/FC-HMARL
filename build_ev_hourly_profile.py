from pathlib import Path
import pandas as pd
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
    / "ev"
    / "ACN_EV_2018_2020_Clean.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ev"
)

OUTPUT_FILE = (
    OUTPUT_DIR
    / "ACN_EV_Hourly_Profile_2018_2020.csv"
)

# ============================================================
# 2. CONSTANTS
# ============================================================

LOCAL_TIMEZONE = "America/Los_Angeles"

EPSILON_HOURS = 1e-9

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


def parse_utc(series):

    return pd.to_datetime(
        series,
        errors="coerce",
        utc=True,
    )


# ============================================================
# 4. START
# ============================================================

section(
    "FC-HMARL - BUILDING HOURLY ACN EV PROFILE"
)


# ============================================================
# 5. VERIFY INPUT
# ============================================================

if not INPUT_FILE.exists():

    raise FileNotFoundError(
        "Clean ACN EV file not found:\n"
        f"{INPUT_FILE}"
    )


print(
    f"Input file:\n"
    f"{INPUT_FILE}"
)


OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# 6. LOAD CLEAN SESSION DATA
# ============================================================

subsection(
    "LOADING CLEAN ACN EV SESSIONS"
)


df = pd.read_csv(
    INPUT_FILE
)


print(
    f"Rows loaded: "
    f"{len(df):,}"
)

print(
    f"Columns loaded: "
    f"{len(df.columns):,}"
)


# ============================================================
# 7. VERIFY REQUIRED COLUMNS
# ============================================================

required_columns = [

    "sessionID",
    "connection_utc",
    "disconnect_utc",
    "done_charging_utc",
    "kwh_delivered",

]


missing_columns = [

    column

    for column in required_columns

    if column not in df.columns
]


if missing_columns:

    raise RuntimeError(
        "Missing required columns:\n"
        + "\n".join(
            missing_columns
        )
    )


# ============================================================
# 8. PARSE TIMESTAMPS
# ============================================================

subsection(
    "PARSING SESSION TIMESTAMPS"
)


df["connection_utc"] = parse_utc(
    df["connection_utc"]
)

df["disconnect_utc"] = parse_utc(
    df["disconnect_utc"]
)

df["done_charging_utc"] = parse_utc(
    df["done_charging_utc"]
)


df["kwh_delivered"] = pd.to_numeric(

    df[
        "kwh_delivered"
    ],

    errors="coerce",
)


print(
    f"Invalid connection timestamps: "
    f"{df['connection_utc'].isna().sum():,}"
)

print(
    f"Invalid disconnect timestamps: "
    f"{df['disconnect_utc'].isna().sum():,}"
)

print(
    f"Missing done-charging times   : "
    f"{df['done_charging_utc'].isna().sum():,}"
)

print(
    f"Missing delivered energy     : "
    f"{df['kwh_delivered'].isna().sum():,}"
)


# ============================================================
# 9. DETERMINE CHARGING END
# ============================================================
#
# Preferred:
#
#   connection -> doneCharging
#
# Fallback:
#
#   connection -> disconnect
#
# doneCharging is only used when:
#
#   connection < doneCharging <= disconnect
#
# Otherwise disconnect is used.
# ============================================================

subsection(
    "SELECTING CHARGING INTERVAL"
)


valid_done_charging = (

    df["done_charging_utc"].notna()

    &

    df["connection_utc"].notna()

    &

    df["disconnect_utc"].notna()

    &

    (
        df["done_charging_utc"]
        > df["connection_utc"]
    )

    &

    (
        df["done_charging_utc"]
        <= df["disconnect_utc"]
    )

)


df["charging_end_source"] = np.where(

    valid_done_charging,

    "doneChargingTime",

    "disconnectTime",
)


df["charging_end_utc"] = (

    df["done_charging_utc"]
    .where(
        valid_done_charging,
        df["disconnect_utc"],
    )
)


print(
    f"Sessions using doneChargingTime: "
    f"{valid_done_charging.sum():,}"
)

print(
    f"Sessions using disconnect fallback: "
    f"{(~valid_done_charging).sum():,}"
)


# ============================================================
# 10. CHARGING INTERVAL DURATION
# ============================================================

df["selected_charging_duration_h"] = (

    df["charging_end_utc"]
    - df["connection_utc"]

).dt.total_seconds() / 3600.0


# ============================================================
# 11. VALID PROFILE SESSION FLAG
# ============================================================

df["valid_profile_session"] = (

    df["connection_utc"].notna()

    &

    df["charging_end_utc"].notna()

    &

    df["kwh_delivered"].notna()

    &

    (
        df["kwh_delivered"]
        > 0
    )

    &

    (
        df["selected_charging_duration_h"]
        > 0
    )

)


valid_sessions = df[

    df[
        "valid_profile_session"
    ]

].copy()


print(
    f"Sessions valid for hourly profile: "
    f"{len(valid_sessions):,}"
)

print(
    f"Sessions excluded from profile    : "
    f"{len(df) - len(valid_sessions):,}"
)


# ============================================================
# 12. SESSION AVERAGE CHARGING POWER
# ============================================================
#
# P = E / duration
#
# This is an average session charging power.
# ============================================================

valid_sessions[
    "session_average_power_kw"
] = (

    valid_sessions[
        "kwh_delivered"
    ]

    /

    valid_sessions[
        "selected_charging_duration_h"
    ]
)


# ============================================================
# 13. TOTAL INPUT ENERGY
# ============================================================

total_input_energy_kwh = (

    valid_sessions[
        "kwh_delivered"
    ]

    .sum()
)


print(
    f"Total energy represented: "
    f"{total_input_energy_kwh:,.6f} kWh"
)


# ============================================================
# 14. BUILD HOURLY ENERGY CONTRIBUTIONS
# ============================================================

subsection(
    "ALLOCATING SESSION ENERGY INTO HOURLY BINS"
)


hourly_records = []


for row_number, row in valid_sessions.iterrows():

    start = row[
        "connection_utc"
    ]

    end = row[
        "charging_end_utc"
    ]

    power_kw = row[
        "session_average_power_kw"
    ]


    if pd.isna(start) or pd.isna(end):

        continue


    if end <= start:

        continue


    # --------------------------------------------------------
    # First hourly bin containing the session start
    # --------------------------------------------------------

    current_hour = start.floor(
        "h"
    )


    # --------------------------------------------------------
    # Continue until the charging interval ends
    # --------------------------------------------------------

    while current_hour < end:

        next_hour = (

            current_hour
            + pd.Timedelta(
                hours=1
            )
        )


        overlap_start = max(
            start,
            current_hour,
        )

        overlap_end = min(
            end,
            next_hour,
        )


        overlap_hours = (

            overlap_end
            - overlap_start

        ).total_seconds() / 3600.0


        if overlap_hours > EPSILON_HOURS:

            energy_kwh = (

                power_kw
                * overlap_hours
            )


            hourly_records.append(

                {
                    "timestamp_utc":
                        current_hour,

                    "session_id":
                        row["sessionID"],

                    "energy_kwh":
                        energy_kwh,

                    "session_power_kw":
                        power_kw,

                    "overlap_hours":
                        overlap_hours,
                }

            )


        current_hour = next_hour


print(
    f"Hourly session contributions created: "
    f"{len(hourly_records):,}"
)


# ============================================================
# 15. CONVERT CONTRIBUTIONS TO DATAFRAME
# ============================================================

if not hourly_records:

    raise RuntimeError(
        "No hourly EV records were generated."
    )


contributions = pd.DataFrame(
    hourly_records
)


# ============================================================
# 16. AGGREGATE BY HOUR
# ============================================================

subsection(
    "AGGREGATING EV CHARGING BY HOUR"
)


hourly = (

    contributions

    .groupby(
        "timestamp_utc",
        as_index=False,
    )

    .agg(

        ev_energy_kwh=(
            "energy_kwh",
            "sum",
        ),

        active_session_contributions=(
            "session_id",
            "count",
        ),

        unique_sessions=(
            "session_id",
            "nunique",
        ),
    )

)


# ============================================================
# 17. HOURLY AVERAGE POWER
# ============================================================
#
# For a one-hour bin:
#
#   kWh / 1 h = average kW
# ============================================================

hourly[
    "ev_power_kw"
] = (

    hourly[
        "ev_energy_kwh"
    ]
)


# ============================================================
# 18. BUILD COMPLETE HOURLY TIMELINE
# ============================================================

start_hour = (

    valid_sessions[
        "connection_utc"
    ]

    .min()

    .floor(
        "h"
    )
)


end_hour = (

    valid_sessions[
        "charging_end_utc"
    ]

    .max()

    .ceil(
        "h"
    )

    - pd.Timedelta(
        hours=1
    )
)


complete_index = pd.date_range(

    start=start_hour,

    end=end_hour,

    freq="h",

    tz="UTC",
)


complete = pd.DataFrame(

    {
        "timestamp_utc":
            complete_index
    }

)


hourly = complete.merge(

    hourly,

    on="timestamp_utc",

    how="left",
)

zero_fill_columns = [

    "ev_energy_kwh",
    "ev_power_kw",
    "active_session_contributions",
    "unique_sessions",

]

for column in zero_fill_columns:

    hourly[column] = (

        hourly[column]

        .fillna(
            0
        )
    )


hourly[
    "active_session_contributions"
] = (

    hourly[
        "active_session_contributions"
    ]

    .astype(
        int
    )
)


hourly[
    "unique_sessions"
] = (

    hourly[
        "unique_sessions"
    ]

    .astype(
        int
    )
)


# ============================================================
# 20. LOCAL LOS ANGELES TIME
# ============================================================

hourly[
    "timestamp_local"
] = (

    hourly[
        "timestamp_utc"
    ]

    .dt

    .tz_convert(
        LOCAL_TIMEZONE
    )
)


# ============================================================
# 21. LOCAL CALENDAR VARIABLES
# ============================================================

hourly[
    "source_year"
] = (

    hourly[
        "timestamp_local"
    ]

    .dt

    .year
)


hourly[
    "month"
] = (

    hourly[
        "timestamp_local"
    ]

    .dt

    .month
)


hourly[
    "day"
] = (

    hourly[
        "timestamp_local"
    ]

    .dt

    .day
)


hourly[
    "hour"
] = (

    hourly[
        "timestamp_local"
    ]

    .dt

    .hour
)


hourly[
    "weekday"
] = (

    hourly[
        "timestamp_local"
    ]

    .dt

    .day_name()
)


hourly[
    "is_weekend"
] = (

    hourly[
        "timestamp_local"
    ]

    .dt

    .weekday

    >= 5

).astype(
    int
)


# ============================================================
# 22. ENERGY CONSERVATION CHECK
# ============================================================

subsection(
    "ENERGY CONSERVATION CHECK"
)


total_hourly_energy_kwh = (

    hourly[
        "ev_energy_kwh"
    ]

    .sum()
)


energy_difference_kwh = (

    total_hourly_energy_kwh
    - total_input_energy_kwh
)


relative_error = (

    abs(
        energy_difference_kwh
    )

    /

    max(
        abs(
            total_input_energy_kwh
        ),
        EPSILON_HOURS,
    )
)


print(
    f"Session energy total : "
    f"{total_input_energy_kwh:,.9f} kWh"
)

print(
    f"Hourly energy total  : "
    f"{total_hourly_energy_kwh:,.9f} kWh"
)

print(
    f"Difference           : "
    f"{energy_difference_kwh:,.12f} kWh"
)

print(
    f"Relative error       : "
    f"{relative_error:.12e}"
)


if relative_error > 1e-9:

    print(
        "[WARNING] Energy conservation error "
        "is larger than expected."
    )

else:

    print(
        "[OK] Delivered energy is conserved."
    )


# ============================================================
# 23. QUALITY FLAGS
# ============================================================

hourly[
    "has_ev_activity"
] = (

    hourly[
        "ev_energy_kwh"
    ]

    > 0

).astype(
    int
)


hourly[
    "missing_hour"
] = 0


# ============================================================
# 24. PROVENANCE
# ============================================================

hourly[
    "source_dataset"
] = "ACN-Data"

hourly[
    "source_site"
] = "Caltech"

hourly[
    "source_period"
] = "2018-2020"

hourly[
    "source_timezone"
] = LOCAL_TIMEZONE

hourly[
    "construction_method"
] = (
    "Uniform observed delivered energy over "
    "connection-to-valid-doneCharging interval; "
    "disconnect fallback otherwise"
)


# ============================================================
# 25. SORT
# ============================================================

hourly = (

    hourly

    .sort_values(
        "timestamp_utc"
    )

    .reset_index(
        drop=True
    )
)


# ============================================================
# 26. OUTPUT COLUMN ORDER
# ============================================================

output_columns = [

    # Time
    "timestamp_utc",
    "timestamp_local",

    # Calendar
    "source_year",
    "month",
    "day",
    "hour",
    "weekday",
    "is_weekend",

    # EV quantities
    "ev_energy_kwh",
    "ev_power_kw",
    "active_session_contributions",
    "unique_sessions",

    # Flags
    "has_ev_activity",
    "missing_hour",

    # Provenance
    "source_dataset",
    "source_site",
    "source_period",
    "source_timezone",
    "construction_method",
]


hourly = hourly[
    output_columns
]


# ============================================================
# 27. SAVE OUTPUT
# ============================================================

subsection(
    "SAVING HOURLY EV PROFILE"
)


hourly.to_csv(
    OUTPUT_FILE,
    index=False,
)


print(
    f"Saved:\n"
    f"{OUTPUT_FILE}"
)


# ============================================================
# 28. FINAL VALIDATION
# ============================================================

section(
    "HOURLY EV PROFILE VALIDATION"
)


print(
    f"Hourly rows                 : "
    f"{len(hourly):,}"
)

print(
    f"Unique timestamps           : "
    f"{hourly['timestamp_utc'].nunique():,}"
)

print(
    f"Duplicate timestamps        : "
    f"{hourly['timestamp_utc'].duplicated().sum():,}"
)

print(
    f"Hours with EV activity      : "
    f"{hourly['has_ev_activity'].sum():,}"
)

print(
    f"Hours with zero EV activity : "
    f"{(hourly['has_ev_activity'] == 0).sum():,}"
)

print(
    f"Minimum EV power            : "
    f"{hourly['ev_power_kw'].min():.6f} kW"
)

print(
    f"Mean EV power               : "
    f"{hourly['ev_power_kw'].mean():.6f} kW"
)

print(
    f"Maximum EV power            : "
    f"{hourly['ev_power_kw'].max():.6f} kW"
)


# ============================================================
# 29. YEAR COVERAGE
# ============================================================

subsection(
    "SOURCE-YEAR COVERAGE"
)


print(

    hourly[
        "source_year"
    ]

    .value_counts()

    .sort_index()

    .to_string()
)


# ============================================================
# 30. MEAN EV POWER BY LOCAL HOUR
# ============================================================

subsection(
    "MEAN EV POWER BY LOCAL HOUR"
)


hourly_profile = (

    hourly

    .groupby(
        "hour"
    )[
        "ev_power_kw"
    ]

    .mean()
)


print(
    hourly_profile
    .to_string()
)


# ============================================================
# 31. FINAL
# ============================================================

section(
    "HOURLY ACN EV PROFILE COMPLETED"
)


print(
    f"Final hourly EV profile:\n"
    f"{OUTPUT_FILE}"
)

print()

print(
    "Observed ACN delivered energy was preserved."
)

print(
    "Valid doneChargingTime was used when available."
)

print(
    "disconnectTime was used as a transparent fallback."
)

print(
    "No battery capacity was generated."
)

print(
    "No arrival SOC was generated."
)

print(
    "No target SOC was generated."
)

print(
    "No V2G discharge was generated."
)

print(
    "No manuscript EV-count scaling was performed."
)

print(
    "No 96-point EV trajectory was generated."
)

print()

print(
    "STEP 5A COMPLETE."
)
