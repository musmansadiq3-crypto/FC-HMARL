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
    / "ACN_EV_Hourly_Profile_2018_2020.csv"
)
OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ev"
)

OUTPUT_FILE = (
    OUTPUT_DIR
    / "ACN_EV_Representative_Annual_8760.csv"
)


# ============================================================
# 2. REPRESENTATIVE CALENDAR YEAR
# ============================================================
REPRESENTATIVE_CALENDAR_YEAR = 2021
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


# ============================================================
# 4. START
# ============================================================

section(
    "FC-HMARL - PREPARING REPRESENTATIVE ANNUAL EV PROFILE"
)


# ============================================================
# 5. VERIFY INPUT
# ============================================================

if not INPUT_FILE.exists():

    raise FileNotFoundError(
        "Hourly ACN EV profile was not found:\n"
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
# 6. LOAD INPUT
# ============================================================

subsection(
    "LOADING HOURLY ACN EV PROFILE"
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

    "timestamp_local",
    "source_year",
    "month",
    "day",
    "hour",
    "ev_energy_kwh",
    "ev_power_kw",

]


missing_columns = [

    column

    for column
    in required_columns

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
# 8. NUMERIC CONVERSION
# ============================================================

for column in [

    "source_year",
    "month",
    "day",
    "hour",
    "ev_energy_kwh",
    "ev_power_kw",

]:

    df[column] = pd.to_numeric(

        df[
            column
        ],

        errors="coerce",
    )


# ============================================================
# 9. BASIC INPUT VALIDATION
# ============================================================

subsection(
    "INPUT VALIDATION"
)


print(
    f"Missing EV power values: "
    f"{df['ev_power_kw'].isna().sum():,}"
)

print(
    f"Negative EV power values: "
    f"{(df['ev_power_kw'] < 0).sum():,}"
)

print(
    f"Minimum EV power: "
    f"{df['ev_power_kw'].min():.6f} kW"
)

print(
    f"Maximum EV power: "
    f"{df['ev_power_kw'].max():.6f} kW"
)


# ============================================================
# 10. REMOVE LEAP DAY FROM REPRESENTATIVE CONSTRUCTION
# ============================================================
#
# The target annual profile must contain 365 * 24 = 8760 hours.
# Feb 29 is excluded only from the representative annual profile.
# The source file remains unchanged.
# ============================================================

subsection(
    "LEAP-DAY HANDLING"
)


is_feb_29 = (

    (df["month"] == 2)

    &

    (df["day"] == 29)

)


feb29_rows = int(
    is_feb_29.sum()
)


print(
    f"Feb 29 source rows detected: "
    f"{feb29_rows:,}"
)


working = df[
    ~is_feb_29
].copy()


print(
    f"Rows retained after excluding Feb 29: "
    f"{len(working):,}"
)


# ============================================================
# 11. COUNT OBSERVATIONS PER MONTH-DAY-HOUR
# ============================================================

subsection(
    "BUILDING MONTH-DAY-HOUR REPRESENTATIVE VALUES"
)


representative = (

    working

    .groupby(
        [
            "month",
            "day",
            "hour",
        ],
        as_index=False,
    )

    .agg(

        ev_power_kw=(
            "ev_power_kw",
            "mean",
        ),

        ev_energy_kwh=(
            "ev_energy_kwh",
            "mean",
        ),

        observation_count=(
            "ev_power_kw",
            "count",
        ),

        source_year_count=(
            "source_year",
            "nunique",
        ),
    )

)


print(
    f"Unique month-day-hour combinations: "
    f"{len(representative):,}"
)


# ============================================================
# 12. BUILD COMPLETE 8760-HOUR CALENDAR
# ============================================================

subsection(
    "CREATING COMPLETE NON-LEAP ANNUAL CALENDAR"
)


calendar = pd.DataFrame(

    {
        "timestamp":
            pd.date_range(
                start=(
                    f"{REPRESENTATIVE_CALENDAR_YEAR}-01-01 00:00:00"
                ),
                end=(
                    f"{REPRESENTATIVE_CALENDAR_YEAR}-12-31 23:00:00"
                ),
                freq="h",
            )
    }

)


calendar["month"] = (
    calendar["timestamp"]
    .dt
    .month
)

calendar["day"] = (
    calendar["timestamp"]
    .dt
    .day
)

calendar["hour"] = (
    calendar["timestamp"]
    .dt
    .hour
)

calendar["weekday"] = (
    calendar["timestamp"]
    .dt
    .day_name()
)

calendar["is_weekend"] = (

    calendar["timestamp"]
    .dt
    .weekday
    >= 5

).astype(
    int
)


print(
    f"Calendar rows: "
    f"{len(calendar):,}"
)


if len(calendar) != 8760:

    raise RuntimeError(
        "Representative calendar does not contain 8760 hours."
    )


# ============================================================
# 13. MERGE OBSERVED REPRESENTATIVE VALUES
# ============================================================

annual = calendar.merge(

    representative,

    on=[
        "month",
        "day",
        "hour",
    ],

    how="left",
)


# ============================================================
# 14. IDENTIFY MISSING MONTH-DAY-HOUR COMBINATIONS
# ============================================================

subsection(
    "MISSING REPRESENTATIVE HOUR CHECK"
)


missing_before_fill = int(

    annual[
        "ev_power_kw"
    ]

    .isna()

    .sum()
)


print(
    f"Missing representative hours before fallback: "
    f"{missing_before_fill:,}"
)


# ============================================================
# 15. MONTH-HOUR FALLBACK PROFILE
# ============================================================
#
# Used only if an exact month-day-hour value is unavailable.
# ============================================================

month_hour_profile = (

    working

    .groupby(
        [
            "month",
            "hour",
        ],
        as_index=False,
    )

    .agg(

        fallback_month_hour_power_kw=(
            "ev_power_kw",
            "mean",
        ),

        fallback_month_hour_energy_kwh=(
            "ev_energy_kwh",
            "mean",
        ),
    )

)


annual = annual.merge(

    month_hour_profile,

    on=[
        "month",
        "hour",
    ],

    how="left",
)


# ============================================================
# 16. HOUR-OF-DAY FALLBACK PROFILE
# ============================================================

hour_profile = (

    working

    .groupby(
        "hour",
        as_index=False,
    )

    .agg(

        fallback_hour_power_kw=(
            "ev_power_kw",
            "mean",
        ),

        fallback_hour_energy_kwh=(
            "ev_energy_kwh",
            "mean",
        ),
    )

)


annual = annual.merge(

    hour_profile,

    on="hour",

    how="left",
)


# ============================================================
# 17. FLAG ORIGINAL REPRESENTATIVE AVAILABILITY
# ============================================================

annual[
    "exact_month_day_hour_available"
] = (

    annual[
        "ev_power_kw"
    ]

    .notna()

).astype(
    int
)


# ============================================================
# 18. APPLY TRANSPARENT FALLBACKS
# ============================================================

annual[
    "ev_power_before_fallback"
] = (

    annual[
        "ev_power_kw"
    ]
)


annual[
    "ev_energy_before_fallback"
] = (

    annual[
        "ev_energy_kwh"
    ]
)


# First fallback:
# same month and hour

month_fallback_mask = (

    annual[
        "ev_power_kw"
    ]
    .isna()

    &

    annual[
        "fallback_month_hour_power_kw"
    ]
    .notna()

)


annual.loc[
    month_fallback_mask,
    "ev_power_kw"
] = (

    annual.loc[
        month_fallback_mask,
        "fallback_month_hour_power_kw"
    ]
)


annual.loc[
    month_fallback_mask,
    "ev_energy_kwh"
] = (

    annual.loc[
        month_fallback_mask,
        "fallback_month_hour_energy_kwh"
    ]
)


# Second fallback:
# same hour of day across all source observations

hour_fallback_mask = (

    annual[
        "ev_power_kw"
    ]
    .isna()

    &

    annual[
        "fallback_hour_power_kw"
    ]
    .notna()

)


annual.loc[
    hour_fallback_mask,
    "ev_power_kw"
] = (

    annual.loc[
        hour_fallback_mask,
        "fallback_hour_power_kw"
    ]
)


annual.loc[
    hour_fallback_mask,
    "ev_energy_kwh"
] = (

    annual.loc[
        hour_fallback_mask,
        "fallback_hour_energy_kwh"
    ]
)


# ============================================================
# 19. FINAL ZERO FALLBACK
# ============================================================
#
# This should normally not be required.
# It is kept only to guarantee a complete annual profile.
# ============================================================

zero_fallback_mask = (

    annual[
        "ev_power_kw"
    ]
    .isna()

)


annual.loc[
    zero_fallback_mask,
    "ev_power_kw"
] = 0.0


annual.loc[
    zero_fallback_mask,
    "ev_energy_kwh"
] = 0.0


# ============================================================
# 20. FALLBACK METHOD LABEL
# ============================================================

annual[
    "construction_source"
] = "month_day_hour_mean"


annual.loc[
    month_fallback_mask,
    "construction_source"
] = "month_hour_fallback"


annual.loc[
    hour_fallback_mask,
    "construction_source"
] = "hour_of_day_fallback"


annual.loc[
    zero_fallback_mask,
    "construction_source"
] = "zero_fallback"


# ============================================================
# 21. FINAL MISSING CHECK
# ============================================================

missing_after_fill = int(

    annual[
        "ev_power_kw"
    ]

    .isna()

    .sum()
)


print(
    f"Missing EV power values after fallback: "
    f"{missing_after_fill:,}"
)


if missing_after_fill != 0:

    raise RuntimeError(
        "Representative annual EV profile still contains missing values."
    )


# ============================================================
# 22. FALLBACK COUNTS
# ============================================================

subsection(
    "REPRESENTATIVE PROFILE CONSTRUCTION COUNTS"
)


print(
    f"Exact month-day-hour values : "
    f"{annual['exact_month_day_hour_available'].sum():,}"
)

print(
    f"Month-hour fallback values  : "
    f"{month_fallback_mask.sum():,}"
)

print(
    f"Hour-of-day fallback values : "
    f"{hour_fallback_mask.sum():,}"
)

print(
    f"Zero fallback values        : "
    f"{zero_fallback_mask.sum():,}"
)


# ============================================================
# 23. THREE-SIGMA DIAGNOSTIC
# ============================================================
#
# The manuscript mentions three-sigma filtering.
#
# At this stage we IDENTIFY statistical extremes but do not
# silently replace them. That keeps the preprocessing auditable.
#
# If we later decide to apply replacement, it can be done
# explicitly and reported.
# ============================================================

subsection(
    "THREE-SIGMA DIAGNOSTIC"
)


mean_power = (
    annual[
        "ev_power_kw"
    ]
    .mean()
)


std_power = (
    annual[
        "ev_power_kw"
    ]
    .std()
)


lower_3sigma = (
    mean_power
    - 3.0 * std_power
)


upper_3sigma = (
    mean_power
    + 3.0 * std_power
)


three_sigma_flag = (

    (
        annual[
            "ev_power_kw"
        ]
        < lower_3sigma
    )

    |

    (
        annual[
            "ev_power_kw"
        ]
        > upper_3sigma
    )

)


annual[
    "three_sigma_flag"
] = (
    three_sigma_flag
    .astype(int)
)


print(
    f"Mean EV power       : "
    f"{mean_power:.6f} kW"
)

print(
    f"Std EV power        : "
    f"{std_power:.6f} kW"
)

print(
    f"Lower 3-sigma bound : "
    f"{lower_3sigma:.6f} kW"
)

print(
    f"Upper 3-sigma bound : "
    f"{upper_3sigma:.6f} kW"
)

print(
    f"Hours outside bounds: "
    f"{three_sigma_flag.sum():,}"
)

# ============================================================
# 25. PROFILE QUALITY CHECK
# ============================================================

subsection(
    "FINAL PROFILE QUALITY"
)


print(
    f"Rows                 : "
    f"{len(annual):,}"
)

print(
    f"Duplicate timestamps : "
    f"{annual['timestamp'].duplicated().sum():,}"
)

print(
    f"Missing EV power     : "
    f"{annual['ev_power_kw'].isna().sum():,}"
)

print(
    f"Negative EV power    : "
    f"{(annual['ev_power_kw'] < 0).sum():,}"
)

print(
    f"Minimum EV power     : "
    f"{annual['ev_power_kw'].min():.6f} kW"
)

print(
    f"Mean EV power        : "
    f"{annual['ev_power_kw'].mean():.6f} kW"
)

print(
    f"Maximum EV power     : "
    f"{annual['ev_power_kw'].max():.6f} kW"
)


# ============================================================
# 26. MEAN PROFILE BY HOUR
# ============================================================

subsection(
    "MEAN REPRESENTATIVE EV POWER BY HOUR"
)


mean_by_hour = (

    annual

    .groupby(
        "hour"
    )[
        "ev_power_kw"
    ]

    .mean()
)


print(
    mean_by_hour
    .to_string()
)


# ============================================================
# 27. PROVENANCE
# ============================================================

annual[
    "source_dataset"
] = "ACN-Data"

annual[
    "source_site"
] = "Caltech"

annual[
    "source_observation_period"
] = "2018-2020"

annual[
    "profile_type"
] = "Representative annual profile"

annual[
    "representative_calendar_year"
] = REPRESENTATIVE_CALENDAR_YEAR

annual[
    "is_measured_2022_data"
] = 0

annual[
    "profile_method"
] = (
    "Mean by local month-day-hour across available "
    "ACN 2018-2020 hourly observations; Feb 29 excluded"
)


# ============================================================
# 28. OUTPUT COLUMN ORDER
# ============================================================

output_columns = [

    # Time
    "timestamp",
    "month",
    "day",
    "hour",
    "weekday",
    "is_weekend",

    # EV profile
    "ev_energy_kwh",
    "ev_power_kw",

    # Data support
    "observation_count",
    "source_year_count",
    "exact_month_day_hour_available",
    "construction_source",
    "three_sigma_flag",

    # Provenance
    "source_dataset",
    "source_site",
    "source_observation_period",
    "profile_type",
    "representative_calendar_year",
    "is_measured_2022_data",
    "profile_method",
]


annual = annual[
    output_columns
].copy()


# ============================================================
# 29. SAVE OUTPUT
# ============================================================

subsection(
    "SAVING REPRESENTATIVE ANNUAL EV PROFILE"
)


annual.to_csv(
    OUTPUT_FILE,
    index=False,
)


print(
    f"Saved:\n"
    f"{OUTPUT_FILE}"
)


# ============================================================
# 30. FINAL VALIDATION
# ============================================================

section(
    "REPRESENTATIVE EV PROFILE VALIDATION"
)


print(
    f"Output rows               : "
    f"{len(annual):,}"
)

print(
    f"Unique timestamps         : "
    f"{annual['timestamp'].nunique():,}"
)

print(
    f"Duplicate timestamps      : "
    f"{annual['timestamp'].duplicated().sum():,}"
)

print(
    f"Missing EV power values   : "
    f"{annual['ev_power_kw'].isna().sum():,}"
)

print(
    f"Negative EV power values  : "
    f"{(annual['ev_power_kw'] < 0).sum():,}"
)

print(
    f"Three-sigma flagged hours : "
    f"{annual['three_sigma_flag'].sum():,}"
)


# ============================================================
# 31. FINAL
# ============================================================

section(
    "REPRESENTATIVE ANNUAL EV PROFILE COMPLETED"
)


print(
    f"Final dataset:\n"
    f"{OUTPUT_FILE}"
)

print()

print(
    "The profile contains exactly 8760 hourly values."
)

print(
    "The profile is derived from ACN 2018-2020 observations."
)

print(
    "Feb 29 observations were excluded from the non-leap annual profile."
)

print(
    "Missing calendar combinations were filled transparently."
)

print(
    "Three-sigma values were flagged but NOT modified."
)

print(
    "The representative calendar year is only a calendar template."
)

print(
    "The data are NOT claimed to be measured 2022 EV observations."
)

print(
    "No SOC values were generated."
)

print(
    "No battery capacities were generated."
)

print(
    "No V2G discharge trajectory was generated."
)

print(
    "No manuscript EV-count scaling was performed."
)

print(
    "No 96-point EV trajectory was generated."
)

print()

print(
    "STEP 5B COMPLETE."
)
