# ============================================================
# FC-HMARL - ACN-DATA MONTHLY DOWNLOADER
# ============================================================
#
# Purpose:
#   Download Caltech ACN charging-session data month-by-month
#   instead of downloading the full dataset in one long API run.
#
# Advantages:
#   - Smaller pagination chains
#   - Easier restart after server errors
#   - Each completed month is saved permanently
#   - Final files are automatically merged
#   - Duplicate sessions are removed
#   - Final JSON is validated
#
# IMPORTANT:
#   Do NOT put your API token directly in this file.
#
# In PowerShell:
#
#   $env:ACN_API_TOKEN="YOUR_NEW_TOKEN"
#
# Then run:
#
#   python download_acn_monthly.py
#
# ============================================================

import os
import json
import time
import calendar
from pathlib import Path
from datetime import datetime, timezone
from urllib.parse import urljoin

import requests


# ============================================================
# 1. USER SETTINGS
# ============================================================

SITE_ID = "caltech"

START_YEAR = 2018
END_YEAR = 2020

BASE_API = "https://ev.caltech.edu/api/v1/"

SESSION_ENDPOINT = (
    f"{BASE_API}sessions/{SITE_ID}"
)

PROJECT_ROOT = Path(
    r"D:\Molvi paper review\FC_HMARL"
)

RAW_EV_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "ev"
)

MONTHLY_DIR = (
    RAW_EV_DIR
    / "acn_monthly"
)

FINAL_OUTPUT = (
    RAW_EV_DIR
    / "acndata_sessions.json"
)

FAILED_MONTHS_FILE = (
    RAW_EV_DIR
    / "acn_failed_months.json"
)


# ============================================================
# 2. CREATE DIRECTORIES
# ============================================================

RAW_EV_DIR.mkdir(
    parents=True,
    exist_ok=True
)

MONTHLY_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 3. READ API TOKEN
# ============================================================

TOKEN = os.environ.get(
    "ACN_API_TOKEN"
)

if not TOKEN:

    print()
    print("=" * 70)
    print("ERROR: ACN_API_TOKEN IS NOT SET")
    print("=" * 70)
    print()
    print("Run this first in PowerShell:")
    print()
    print(
        '$env:ACN_API_TOKEN="YOUR_NEW_TOKEN"'
    )
    print()
    print("Then run:")
    print()
    print(
        "python download_acn_monthly.py"
    )
    print()

    raise RuntimeError(
        "ACN_API_TOKEN environment variable is missing."
    )


# ============================================================
# 4. HTTP SESSION
# ============================================================

http = requests.Session()

# ACN authentication:
# username = token
# password = blank
http.auth = (
    TOKEN,
    ""
)

http.headers.update(
    {
        "User-Agent":
            "FC-HMARL-Research-Downloader/1.0",
        "Accept":
            "application/json",
        "Connection":
            "keep-alive",
    }
)


# ============================================================
# 5. MONTH UTILITIES
# ============================================================

def month_start_end(
    year,
    month
):
    """
    Return UTC start and end timestamps
    suitable for ACN connectionTime filtering.

    We use:
        start <= connectionTime < next_month_start

    This avoids overlap between months.
    """

    start = datetime(
        year,
        month,
        1,
        0,
        0,
        0,
        tzinfo=timezone.utc,
    )

    if month == 12:

        end = datetime(
            year + 1,
            1,
            1,
            0,
            0,
            0,
            tzinfo=timezone.utc,
        )

    else:

        end = datetime(
            year,
            month + 1,
            1,
            0,
            0,
            0,
            tzinfo=timezone.utc,
        )

    return start, end


def format_acn_datetime(dt):
    """
    Convert datetime into RFC-1123-style UTC
    string expected by ACN API.
    """

    return dt.strftime(
        "%a, %d %b %Y %H:%M:%S GMT"
    )


# ============================================================
# 6. REQUEST ONE API PAGE
# ============================================================

def request_page(
    url,
    params=None,
    max_attempts=12
):
    """
    Request a single ACN API page.

    Retries temporary network/server errors.
    """

    retry_delays = [
        2,
        4,
        8,
        15,
        30,
        45,
        60,
        60,
        60,
        60,
        60,
        60,
    ]

    for attempt in range(
        1,
        max_attempts + 1
    ):

        try:

            response = http.get(
                url,
                params=params,
                timeout=60,
            )

            # --------------------------------------------
            # Authentication errors
            # --------------------------------------------

            if response.status_code == 401:

                raise RuntimeError(
                    "\nACN authentication failed.\n"
                    "HTTP 401.\n"
                    "Check your API token."
                )

            if response.status_code == 403:

                raise RuntimeError(
                    "\nACN API access forbidden.\n"
                    "HTTP 403.\n"
                    "Check your API token."
                )

            # --------------------------------------------
            # Rate limiting
            # --------------------------------------------

            if response.status_code == 429:

                retry_after = (
                    response.headers.get(
                        "Retry-After"
                    )
                )

                try:

                    wait_seconds = int(
                        retry_after
                    )

                except (
                    TypeError,
                    ValueError,
                ):

                    wait_seconds = 60

                print(
                    f"    HTTP 429. "
                    f"Waiting "
                    f"{wait_seconds}s..."
                )

                time.sleep(
                    wait_seconds
                )

                continue

            # --------------------------------------------
            # Temporary server failures
            # --------------------------------------------

            if response.status_code in {
                500,
                502,
                503,
                504,
            }:

                raise requests.exceptions.HTTPError(
                    f"HTTP "
                    f"{response.status_code} "
                    f"for {response.url}",
                    response=response,
                )

            response.raise_for_status()

            # --------------------------------------------
            # Parse JSON
            # --------------------------------------------

            try:

                data = response.json()

            except ValueError as exc:

                raise requests.exceptions.RequestException(
                    "Server returned invalid JSON."
                ) from exc

            return data

        except RuntimeError:

            raise

        except (
            requests.exceptions.ConnectionError,
            requests.exceptions.Timeout,
            requests.exceptions.HTTPError,
            requests.exceptions.RequestException,
        ) as exc:

            print(
                f"    Attempt "
                f"{attempt}/"
                f"{max_attempts} failed:"
            )

            print(
                f"    {exc}"
            )

            if attempt >= max_attempts:

                return None

            delay = retry_delays[
                min(
                    attempt - 1,
                    len(
                        retry_delays
                    ) - 1,
                )
            ]

            print(
                f"    Retrying in "
                f"{delay}s..."
            )

            time.sleep(
                delay
            )

    return None


# ============================================================
# 7. SAVE JSON SAFELY
# ============================================================

def save_json(
    path,
    data
):
    """
    Save JSON using a temporary file first.
    """

    temp_path = Path(
        str(path) + ".tmp"
    )

    with open(
        temp_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            data,
            file,
            indent=2,
            ensure_ascii=False,
        )

    temp_path.replace(
        path
    )


# ============================================================
# 8. CHECK WHETHER MONTH IS ALREADY COMPLETE
# ============================================================

def month_file_complete(
    file_path
):

    if not file_path.exists():

        return False

    try:

        with open(
            file_path,
            "r",
            encoding="utf-8",
        ) as file:

            data = json.load(
                file
            )

        meta = data.get(
            "_meta",
            {}
        )

        return bool(
            meta.get(
                "download_complete",
                False,
            )
        )

    except Exception:

        return False


# ============================================================
# 9. DOWNLOAD ONE MONTH
# ============================================================

def download_month(
    year,
    month
):

    month_name = (
        f"{year}_{month:02d}"
    )

    output_file = (
        MONTHLY_DIR
        / f"caltech_{month_name}.json"
    )

    print()
    print(
        "=" * 70
    )

    print(
        f"CALTECH "
        f"{calendar.month_name[month]} "
        f"{year}"
    )

    print(
        "=" * 70
    )

    # --------------------------------------------
    # Skip completed months
    # --------------------------------------------

    if month_file_complete(
        output_file
    ):

        print(
            "Already complete. "
            "Skipping."
        )

        return {
            "success": True,
            "skipped": True,
            "year": year,
            "month": month,
        }

    # --------------------------------------------
    # Build month filter
    # --------------------------------------------

    start_dt, end_dt = (
        month_start_end(
            year,
            month,
        )
    )

    start_text = (
        format_acn_datetime(
            start_dt
        )
    )

    end_text = (
        format_acn_datetime(
            end_dt
        )
    )

    where_filter = (
        f'connectionTime>="'
        f'{start_text}'
        f'" and '
        f'connectionTime<"'
        f'{end_text}'
        f'"'
    )

    params = {
        "where":
            where_filter,
        "sort":
            "-connectionTime",
    }

    next_url = (
        SESSION_ENDPOINT
    )

    first_request = True

    all_sessions = []

    expected_total = None

    page_counter = 0

    # --------------------------------------------
    # Paginated download
    # --------------------------------------------

    while next_url:

        page_counter += 1

        if first_request:

            page = request_page(
                next_url,
                params=params,
            )

            first_request = False

        else:

            page = request_page(
                next_url
            )

        if page is None:

            print()
            print(
                "Month failed after "
                "multiple server retries."
            )

            print(
                "Moving to the next month."
            )

            return {
                "success": False,
                "skipped": False,
                "year": year,
                "month": month,
                "downloaded":
                    len(
                        all_sessions
                    ),
            }

        # --------------------------------------------
        # Read sessions
        # --------------------------------------------

        items = page.get(
            "_items",
            []
        )

        if expected_total is None:

            expected_total = (
                page.get(
                    "_meta",
                    {}
                ).get(
                    "total"
                )
            )

            print(
                f"Expected sessions: "
                f"{expected_total}"
            )

        all_sessions.extend(
            items
        )

        print(
            f"    Page "
            f"{page_counter:3d} "
            f"| Received "
            f"{len(items):3d} "
            f"| Saved "
            f"{len(all_sessions):5d}"
        )

        # --------------------------------------------
        # Find next link
        # --------------------------------------------

        links = page.get(
            "_links",
            {}
        )

        next_link = links.get(
            "next"
        )

        if not next_link:

            next_url = None

        else:

            if isinstance(
                next_link,
                dict
            ):

                next_href = (
                    next_link.get(
                        "href"
                    )
                )

            else:

                next_href = (
                    next_link
                )

            if next_href:

                next_url = urljoin(
                    BASE_API,
                    next_href,
                )

            else:

                next_url = None

        # Small pause to reduce server pressure.
        time.sleep(
            0.35
        )

    # --------------------------------------------
    # Deduplicate the month
    # --------------------------------------------

    unique_sessions = {}

    anonymous_sessions = []

    for session in all_sessions:

        key = (
            session.get(
                "sessionID"
            )
            or session.get(
                "_id"
            )
        )

        if key:

            unique_sessions[
                str(key)
            ] = session

        else:

            anonymous_sessions.append(
                session
            )

    cleaned_sessions = (
        list(
            unique_sessions.values()
        )
        +
        anonymous_sessions
    )

    # --------------------------------------------
    # Determine completeness
    # --------------------------------------------

    downloaded_count = len(
        cleaned_sessions
    )

    if expected_total is None:

        complete = True

    else:

        complete = (
            downloaded_count
            == expected_total
        )

    month_data = {

        "_meta": {

            "site":
                SITE_ID,

            "year":
                year,

            "month":
                month,

            "start_utc":
                start_dt.isoformat(),

            "end_utc_exclusive":
                end_dt.isoformat(),

            "api_endpoint":
                SESSION_ENDPOINT,

            "expected_total":
                expected_total,

            "downloaded_sessions":
                downloaded_count,

            "download_complete":
                complete,

            "downloaded_at_utc":
                datetime.now(
                    timezone.utc
                ).isoformat(),

        },

        "_items":
            cleaned_sessions,
    }

    save_json(
        output_file,
        month_data,
    )

    print()
    print(
        f"Saved month: "
        f"{output_file.name}"
    )

    print(
        f"Downloaded: "
        f"{downloaded_count}"
    )

    print(
        f"Complete: "
        f"{complete}"
    )

    return {
        "success":
            complete,
        "skipped":
            False,
        "year":
            year,
        "month":
            month,
        "downloaded":
            downloaded_count,
        "expected":
            expected_total,
    }


# ============================================================
# 10. MERGE ALL COMPLETE MONTHS
# ============================================================

def merge_months():

    print()
    print(
        "=" * 70
    )

    print(
        "MERGING COMPLETE MONTHS"
    )

    print(
        "=" * 70
    )

    all_sessions = []

    completed_files = []

    incomplete_files = []

    for file_path in sorted(
        MONTHLY_DIR.glob(
            "caltech_*.json"
        )
    ):

        try:

            with open(
                file_path,
                "r",
                encoding="utf-8",
            ) as file:

                data = json.load(
                    file
                )

        except Exception as exc:

            print(
                f"Could not read "
                f"{file_path.name}: "
                f"{exc}"
            )

            incomplete_files.append(
                file_path.name
            )

            continue

        meta = data.get(
            "_meta",
            {}
        )

        if not meta.get(
            "download_complete",
            False,
        ):

            incomplete_files.append(
                file_path.name
            )

            continue

        items = data.get(
            "_items",
            []
        )

        all_sessions.extend(
            items
        )

        completed_files.append(
            file_path.name
        )

    # --------------------------------------------
    # Global deduplication
    # --------------------------------------------

    unique_sessions = {}

    anonymous_sessions = []

    for session in all_sessions:

        key = (
            session.get(
                "sessionID"
            )
            or session.get(
                "_id"
            )
        )

        if key:

            unique_sessions[
                str(key)
            ] = session

        else:

            anonymous_sessions.append(
                session
            )

    merged_sessions = (
        list(
            unique_sessions.values()
        )
        +
        anonymous_sessions
    )

    # --------------------------------------------
    # Sort by connectionTime when possible
    # --------------------------------------------

    merged_sessions.sort(
        key=lambda x:
            x.get(
                "connectionTime",
                ""
            )
    )

    final_data = {

        "_meta": {

            "site":
                SITE_ID,

            "source":
                "ACN-Data REST API",

            "source_year_start":
                START_YEAR,

            "source_year_end":
                END_YEAR,

            "completed_month_files":
                len(
                    completed_files
                ),

            "incomplete_month_files":
                len(
                    incomplete_files
                ),

            "downloaded_sessions":
                len(
                    merged_sessions
                ),

            "download_complete":
                (
                    len(
                        incomplete_files
                    )
                    == 0
                ),

            "created_at_utc":
                datetime.now(
                    timezone.utc
                ).isoformat(),

        },

        "_items":
            merged_sessions,
    }

    save_json(
        FINAL_OUTPUT,
        final_data,
    )

    print(
        f"Completed months : "
        f"{len(completed_files)}"
    )

    print(
        f"Incomplete months: "
        f"{len(incomplete_files)}"
    )

    print(
        f"Unique sessions  : "
        f"{len(merged_sessions)}"
    )

    print()
    print(
        f"Final file:"
    )

    print(
        FINAL_OUTPUT
    )

    # --------------------------------------------
    # Save failed/incomplete month report
    # --------------------------------------------

    failed_report = {

        "incomplete_months":
            incomplete_files,

        "count":
            len(
                incomplete_files
            ),
    }

    save_json(
        FAILED_MONTHS_FILE,
        failed_report,
    )

    return (
        merged_sessions,
        incomplete_files
    )


# ============================================================
# 11. FINAL VALIDATION
# ============================================================

def validate_final(
    sessions
):

    print()
    print(
        "=" * 70
    )

    print(
        "FINAL VALIDATION"
    )

    print(
        "=" * 70
    )

    session_ids = []

    missing_connection = 0
    missing_disconnect = 0
    missing_energy = 0
    missing_station = 0

    for session in sessions:

        sid = session.get(
            "sessionID"
        )

        if sid is not None:

            session_ids.append(
                str(sid)
            )

        if not session.get(
            "connectionTime"
        ):

            missing_connection += 1

        if not session.get(
            "disconnectTime"
        ):

            missing_disconnect += 1

        if (
            session.get(
                "kWhDelivered"
            )
            is None
        ):

            missing_energy += 1

        if not session.get(
            "stationID"
        ):

            missing_station += 1

    duplicate_ids = (
        len(session_ids)
        -
        len(
            set(
                session_ids
            )
        )
    )

    print(
        f"Total sessions       : "
        f"{len(sessions)}"
    )

    print(
        f"Duplicate session IDs: "
        f"{duplicate_ids}"
    )

    print(
        f"Missing connection   : "
        f"{missing_connection}"
    )

    print(
        f"Missing disconnect   : "
        f"{missing_disconnect}"
    )

    print(
        f"Missing kWhDelivered : "
        f"{missing_energy}"
    )

    print(
        f"Missing stationID    : "
        f"{missing_station}"
    )


# ============================================================
# 12. MAIN PROGRAM
# ============================================================

def main():

    print()
    print(
        "=" * 70
    )

    print(
        "FC-HMARL ACN-DATA "
        "MONTHLY DOWNLOADER"
    )

    print(
        "=" * 70
    )

    print(
        f"Site        : "
        f"{SITE_ID}"
    )

    print(
        f"Years       : "
        f"{START_YEAR}-"
        f"{END_YEAR}"
    )

    print(
        f"Monthly dir : "
        f"{MONTHLY_DIR}"
    )

    print()

    failed_months = []

    successful_months = 0

    for year in range(
        START_YEAR,
        END_YEAR + 1
    ):

        for month in range(
            1,
            13
        ):

            result = download_month(
                year,
                month
            )

            if result[
                "success"
            ]:

                successful_months += 1

            else:

                failed_months.append(
                    {
                        "year":
                            year,
                        "month":
                            month,
                    }
                )

            # Small pause between months
            time.sleep(
                1
            )

    print()
    print(
        "=" * 70
    )

    print(
        "MONTHLY DOWNLOAD SUMMARY"
    )

    print(
        "=" * 70
    )

    print(
        f"Successful months: "
        f"{successful_months}"
    )

    print(
        f"Failed months    : "
        f"{len(failed_months)}"
    )

    if failed_months:

        print()
        print(
            "Failed:"
        )

        for item in failed_months:

            print(
                f"  "
                f"{item['year']}-"
                f"{item['month']:02d}"
            )

    # Merge all successfully completed months
    sessions, incomplete = (
        merge_months()
    )

    # Validate merged dataset
    validate_final(
        sessions
    )

    print()
    print(
        "=" * 70
    )

    if not incomplete:

        print(
            "ACN DOWNLOAD COMPLETE"
        )

    else:

        print(
            "PARTIAL DOWNLOAD COMPLETE"
        )

        print(
            "Run this script again later."
        )

        print(
            "Already completed months "
            "will automatically be skipped."
        )

    print(
        "=" * 70
    )


# ============================================================
# 13. RUN
# ============================================================

if __name__ == "__main__":

    main()