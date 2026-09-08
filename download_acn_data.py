import json
import os
import time
from pathlib import Path
from urllib.parse import urljoin
import requests
SITE_ID = "caltech"
BASE_API = "https://ev.caltech.edu/api/v1/"
START_URL = f"{BASE_API}sessions/{SITE_ID}"
PROJECT_ROOT = Path(
    r"D:\Molvi paper review\FC_HMARL"
)

EV_RAW_DIR = PROJECT_ROOT / "data" / "raw" / "ev"

OUTPUT_FILE = EV_RAW_DIR / "acndata_sessions.json"

CHECKPOINT_FILE = EV_RAW_DIR / "acndata_sessions_checkpoint.json"


# ============================================================
# 2. CREATE OUTPUT DIRECTORY
# ============================================================

EV_RAW_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 3. READ API TOKEN
# ============================================================
TOKEN = os.environ.get("ACN_API_TOKEN")


if not TOKEN:

    print()
    print("=" * 70)
    print("ERROR: ACN_API_TOKEN is not set.")
    print("=" * 70)
    print()
    print("Open PowerShell and run:")
    print()
    print('$env:ACN_API_TOKEN="YOUR_NEW_ACN_TOKEN"')
    print()
    print("Then run:")
    print()
    print("python download_acn_data.py")
    print()

    raise RuntimeError(
        "ACN_API_TOKEN environment variable is missing."
    )


# ============================================================
# 4. HTTP SESSION
# ============================================================

http = requests.Session()

# ACN uses HTTP Basic Authentication:
# username = API token
# password = blank
http.auth = (
    TOKEN,
    ""
)

http.headers.update(
    {
        "Accept": "application/json",
        "User-Agent": "FC-HMARL-ACN-Downloader/1.0",
    }
)


# ============================================================
# 5. REQUEST ONE PAGE
# ============================================================

def request_page(url, max_attempts=30):

    retry_delays = [
        2, 4, 8, 16, 32,
        60, 60, 60, 60, 60,
        60, 60, 60, 60, 60,
        60, 60, 60, 60, 60,
        60, 60, 60, 60, 60,
        60, 60, 60, 60
    ]

    for attempt in range(1, max_attempts + 1):

        try:
            response = http.get(
                url,
                timeout=60
            )

            # ------------------------------------------------
            # Authentication errors: do NOT keep retrying
            # ------------------------------------------------
            if response.status_code == 401:
                raise RuntimeError(
                    "\nACN authentication failed.\n"
                    "HTTP status: 401\n"
                    "Check your ACN API token."
                )

            if response.status_code == 403:
                raise RuntimeError(
                    "\nACN API access forbidden.\n"
                    "HTTP status: 403\n"
                    "Check whether the API token is active."
                )

            # ------------------------------------------------
            # Rate limiting
            # ------------------------------------------------
            if response.status_code == 429:

                retry_after = response.headers.get(
                    "Retry-After"
                )

                try:
                    wait_seconds = int(retry_after)
                except (TypeError, ValueError):
                    wait_seconds = 60

                print()
                print(
                    f"ACN rate limit reached (429). "
                    f"Waiting {wait_seconds} seconds..."
                )

                time.sleep(wait_seconds)
                continue

            # ------------------------------------------------
            # Temporary ACN/server errors
            # ------------------------------------------------
            if response.status_code in {
                500, 502, 503, 504
            }:
                raise requests.exceptions.HTTPError(
                    f"{response.status_code} Server Error "
                    f"for url: {url}",
                    response=response
                )

            # Other HTTP errors
            response.raise_for_status()

            # ------------------------------------------------
            # JSON decoding
            # ------------------------------------------------
            try:
                data = response.json()
            except ValueError as exc:
                raise RuntimeError(
                    f"ACN returned invalid JSON for:\n{url}"
                ) from exc

            return data

        except RuntimeError:
            # Authentication/access errors should stop.
            raise

        except (
            requests.exceptions.ConnectionError,
            requests.exceptions.Timeout,
            requests.exceptions.HTTPError
        ) as exc:

            print()
            print(
                f"Request attempt "
                f"{attempt}/{max_attempts} failed:"
            )
            print(exc)

            if attempt >= max_attempts:
                print()
                print("=" * 70)
                print("ACN SERVER STILL UNAVAILABLE")
                print("=" * 70)
                print(
                    "Maximum retry attempts reached."
                )
                print(
                    "Your previously downloaded sessions "
                    "remain saved in the checkpoint."
                )
                print(
                    "Run the downloader again later and "
                    "it will resume from that checkpoint."
                )
                print("=" * 70)

                raise

            delay_index = min(
                attempt - 1,
                len(retry_delays) - 1
            )

            wait_seconds = retry_delays[delay_index]

            print(
                f"Retrying in {wait_seconds} seconds..."
            )

            time.sleep(wait_seconds)

    raise RuntimeError(
        "Unexpected exit from ACN retry loop."
    )


# ============================================================
# 6. CHECKPOINT FUNCTIONS
# ============================================================

def save_checkpoint(
    sessions,
    next_url,
    expected_total
):

    data = {

        "_meta": {

            "site": SITE_ID,

            "download_status":
                "in_progress",

            "sessions_saved":
                len(sessions),

            "expected_total":
                expected_total,
        },

        "_next_url":
            next_url,

        "_items":
            sessions,
    }

    with CHECKPOINT_FILE.open(
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            data,
            file,
            indent=2,
            ensure_ascii=False
        )


def load_checkpoint():

    if not CHECKPOINT_FILE.exists():

        return (
            [],
            START_URL,
            None
        )

    print()
    print(
        "Existing checkpoint found."
    )

    print(
        "Recovering previous download progress..."
    )

    try:

        with CHECKPOINT_FILE.open(
            "r",
            encoding="utf-8"
        ) as file:

            data = json.load(file)

        sessions = data.get(
            "_items",
            []
        )

        next_url = data.get(
            "_next_url",
            START_URL
        )

        expected_total = (
            data
            .get("_meta", {})
            .get("expected_total")
        )

        print(
            f"Recovered sessions: "
            f"{len(sessions):,}"
        )

        return (
            sessions,
            next_url,
            expected_total
        )

    except Exception as error:

        print()
        print(
            "Checkpoint could not be loaded:"
        )

        print(error)

        print()
        print(
            "Starting download from the beginning."
        )

        return (
            [],
            START_URL,
            None
        )


# ============================================================
# 7. LOAD PREVIOUS PROGRESS
# ============================================================

(
    all_sessions,
    next_url,
    expected_total
) = load_checkpoint()


seen_ids = set()


for record in all_sessions:

    session_id = record.get(
        "sessionID"
    )

    if session_id is not None:

        seen_ids.add(
            session_id
        )


# ============================================================
# 8. START DOWNLOAD
# ============================================================

print()
print("=" * 70)
print("FC-HMARL ACN-DATA DOWNLOAD")
print("=" * 70)

print(
    f"Site                : "
    f"{SITE_ID}"
)

print(
    f"API endpoint        : "
    f"{START_URL}"
)

print(
    f"Output directory    : "
    f"{EV_RAW_DIR}"
)

print(
    f"Existing sessions   : "
    f"{len(all_sessions):,}"
)

print("=" * 70)
print()


page_number = 0


while next_url:

    page_number += 1

    page = request_page(
        next_url
    )


    # ========================================================
    # READ METADATA
    # ========================================================

    metadata = page.get(
        "_meta",
        {}
    )


    if expected_total is None:

        expected_total = (
            metadata.get("total")
        )

        print(
            f"ACN reported total sessions: "
            f"{expected_total}"
        )

        print()


    # ========================================================
    # READ SESSION RECORDS
    # ========================================================

    items = page.get(
        "_items",
        []
    )


    added_this_page = 0


    for item in items:

        session_id = item.get(
            "sessionID"
        )


        if session_id is None:

            unique_id = item.get(
                "_id"
            )

        else:

            unique_id = session_id


        if unique_id is None:

            continue


        if unique_id not in seen_ids:

            seen_ids.add(
                unique_id
            )

            all_sessions.append(
                item
            )

            added_this_page += 1


    # ========================================================
    # FIND NEXT PAGE
    # ========================================================

    links = page.get(
        "_links",
        {}
    )

    next_link = links.get(
        "next"
    )


    if isinstance(
        next_link,
        dict
    ):

        next_href = (
            next_link.get("href")
        )

    elif isinstance(
        next_link,
        str
    ):

        next_href = next_link

    else:

        next_href = None


    if next_href:

        next_url = urljoin(
            BASE_API,
            next_href
        )

    else:

        next_url = None


    # ========================================================
    # DISPLAY PROGRESS
    # ========================================================

    print(
        f"Page {page_number:5d} | "
        f"Received: {len(items):3d} | "
        f"New: {added_this_page:3d} | "
        f"Saved: {len(all_sessions):,}"
    )


    # ========================================================
    # SAVE CHECKPOINT
    # ========================================================

    save_checkpoint(
        sessions=all_sessions,
        next_url=next_url,
        expected_total=expected_total
    )


    # Small delay so we do not overload ACN
    time.sleep(
        0.20
    )


# ============================================================
# 9. CREATE FINAL JSON
# ============================================================

print()
print("=" * 70)
print(
    "All API pages have been downloaded."
)
print("=" * 70)
print()


download_complete = True


if expected_total is not None:

    download_complete = (
        len(all_sessions)
        == expected_total
    )


final_data = {

    "_meta": {

        "site":
            SITE_ID,

        "source":
            "ACN-Data REST API",

        "api_endpoint":
            START_URL,

        "expected_total":
            expected_total,

        "downloaded_sessions":
            len(all_sessions),

        "download_complete":
            download_complete,
    },

    "_items":
        all_sessions,
}


with OUTPUT_FILE.open(
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        final_data,
        file,
        indent=2,
        ensure_ascii=False
    )


# ============================================================
# 10. VALIDATE JSON
# ============================================================

print(
    "Validating final JSON..."
)


with OUTPUT_FILE.open(
    "r",
    encoding="utf-8"
) as file:

    validation = json.load(
        file
    )


validated_items = (
    validation.get(
        "_items",
        []
    )
)


validated_count = len(
    validated_items
)


# ============================================================
# 11. EXTRA DATA QUALITY CHECKS
# ============================================================

session_ids = []

missing_connection_time = 0
missing_disconnect_time = 0
missing_energy = 0


for item in validated_items:

    if item.get(
        "sessionID"
    ) is not None:

        session_ids.append(
            item["sessionID"]
        )


    if item.get(
        "connectionTime"
    ) is None:

        missing_connection_time += 1


    if item.get(
        "disconnectTime"
    ) is None:

        missing_disconnect_time += 1


    if item.get(
        "kWhDelivered"
    ) is None:

        missing_energy += 1


duplicate_session_ids = (
    len(session_ids)
    - len(set(session_ids))
)


# ============================================================
# 12. PRINT FINAL REPORT
# ============================================================

print()
print("=" * 70)
print("ACN-DATA DOWNLOAD REPORT")
print("=" * 70)

print(
    f"Site                     : "
    f"{SITE_ID}"
)

print(
    f"Expected ACN sessions    : "
    f"{expected_total}"
)

print(
    f"Downloaded sessions      : "
    f"{validated_count:,}"
)

print(
    f"Duplicate session IDs    : "
    f"{duplicate_session_ids}"
)

print(
    f"Missing connectionTime   : "
    f"{missing_connection_time}"
)

print(
    f"Missing disconnectTime   : "
    f"{missing_disconnect_time}"
)

print(
    f"Missing kWhDelivered     : "
    f"{missing_energy}"
)

print(
    f"JSON validation          : "
    f"PASSED"
)

print(
    f"Download complete        : "
    f"{download_complete}"
)

print()
print(
    f"Final file:"
)

print(
    OUTPUT_FILE
)

print("=" * 70)


# ============================================================
# 13. REMOVE CHECKPOINT ONLY AFTER SUCCESS
# ============================================================

if download_complete:

    if CHECKPOINT_FILE.exists():

        CHECKPOINT_FILE.unlink()

        print()
        print(
            "Checkpoint removed because "
            "the download completed successfully."
        )

else:

    print()
    print(
        "WARNING:"
    )

    print(
        "The downloaded count does not match "
        "the ACN expected total."
    )

    print(
        "The checkpoint has been preserved."
    )


print()
print(
    "ACN download procedure finished."
)
