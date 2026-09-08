from pathlib import Path
import re
PROJECT_ROOT = Path(r"D:\Molvi paper review\FC_HMARL")
FILES = [
    PROJECT_ROOT / "environment" / "bess.py",
    PROJECT_ROOT / "environment" / "market.py",
    PROJECT_ROOT / "environment" / "microgrid.py",
    PROJECT_ROOT / "environment" / "vpp_env.py",
    PROJECT_ROOT / "environment" / "constraints.py",
    PROJECT_ROOT / "marl" / "action_mapper.py",
]
KEYWORDS = [
    "reserve",
    "headroom",
    "bess",
    "battery",
    "feasible_power",
    "rated_power",
    "power_limit",
    "soc",
    "market_profit",
    "reserve_revenue",
    "reserve_power",
]
def print_context(path: Path, keyword: str, radius: int = 5):
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()

    matches = [
        i for i, line in enumerate(lines)
        if keyword.lower() in line.lower()
    ]
    if not matches:
        return False
    print()
    print("-" * 100)
    print(f"FILE: {path.relative_to(PROJECT_ROOT)}")
    print(f"KEYWORD: {keyword}")
    print("-" * 100)
    shown_ranges = []

    for idx in matches:
        start = max(0, idx - radius)
        end = min(len(lines), idx + radius + 1)

        # avoid printing heavily overlapping ranges repeatedly
        if any(start <= old_end and end >= old_start
               for old_start, old_end in shown_ranges):
            continue

        shown_ranges.append((start, end))

        print(f"\nLines {start + 1}-{end}:")
        for j in range(start, end):
            marker = ">>" if j == idx else "  "
            print(f"{marker} {j + 1:4d}: {lines[j]}")

    return True


def main():
    print("=" * 100)
    print("STEP 7R-B - SOURCE AUDIT: BESS / RESERVE COUPLING")
    print("=" * 100)

    missing = [p for p in FILES if not p.exists()]
    if missing:
        print("Missing expected files:")
        for p in missing:
            print("  ", p)
        raise FileNotFoundError(
            "One or more expected project files are missing."
        )

    found_any = False

    for path in FILES:
        for keyword in KEYWORDS:
            found = print_context(
                path,
                keyword,
                radius=4,
            )
            found_any = found_any or found

    print()
    print("=" * 100)
    print("AUTOMATIC PATTERN CHECK")
    print("=" * 100)

    combined = "\n".join(
        p.read_text(encoding="utf-8")
        for p in FILES
    ).lower()

    checks = {
        "reserve keyword exists":
            "reserve" in combined,

        "headroom keyword exists":
            "headroom" in combined,

        "reserve revenue exists":
            "reserve_revenue" in combined
            or "reserve revenue" in combined,

        "reserve power exists":
            "reserve_power" in combined
            or "reserve power" in combined,

        "SOC logic exists":
            "soc" in combined,

        "BESS rated/power-limit logic exists":
            (
                "rated_power" in combined
                or "power_limit" in combined
                or "maximum_power" in combined
            ),
    }

    for label, value in checks.items():
        print(f"{label:42s}: {value}")

    # Look for common explicit coupling expressions.
    coupling_patterns = [
        r"reserve.*rated.*bess",
        r"reserve.*feasible.*power",
        r"reserve.*headroom",
        r"reserve.*soc",
        r"rated.*power.*reserve",
        r"feasible.*power.*reserve",
        r"headroom.*reserve",
    ]

    coupling_hits = []

    for pat in coupling_patterns:
        if re.search(
            pat,
            combined,
            flags=re.IGNORECASE | re.DOTALL,
        ):
            coupling_hits.append(pat)

    print()
    print("Possible explicit reserve/BESS coupling patterns found:")
    if coupling_hits:
        for hit in coupling_hits:
            print("  ", hit)
    else:
        print("  NONE detected by the automatic text-pattern scan.")

    print()
    print("=" * 100)
    print("WHAT TO SEND BACK")
    print("=" * 100)
    print("Please copy the console sections for:")
    print("  1) environment/market.py reserve logic")
    print("  2) environment/bess.py feasible-power/SOC logic")
    print("  3) marl/action_mapper.py reserve mapping")
    print("  4) AUTOMATIC PATTERN CHECK")
    print()
    print("[OK] Step 7R-B source audit complete.")
    print("No source file was modified.")


if __name__ == "__main__":
    main()
