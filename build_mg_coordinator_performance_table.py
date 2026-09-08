from pathlib import Path
import pandas as pd
import numpy as np
ROOT = Path(__file__).resolve().parent
RESULT_DIR = ROOT / "outputs" / "results" / "real_fc_hmarl_final_v3_test"
EPISODE_CSV = RESULT_DIR / "final_test_episode_results.csv"
OUTPUT_DIR = RESULT_DIR / "publication_tables"

CSV_OUT = OUTPUT_DIR / "Table_MG_Coordinator_Performance.csv"
MD_OUT = OUTPUT_DIR / "Table_MG_Coordinator_Performance.md"
TXT_OUT = OUTPUT_DIR / "Table_MG_Coordinator_Performance.txt"

def section(title):
    print()
    print("=" * 80)
    print(title)
    print("=" * 80)

def stats(series):
    x = pd.to_numeric(series, errors="raise").to_numpy(dtype=float)
    return {
        "Mean return": x.mean(),
        "Std. deviation": x.std(ddof=0),
        "Median return": np.median(x),
        "Minimum return": x.min(),
        "Maximum return": x.max(),
    }

def main():
    section("STEP 7R-O2 — MG1–MG5 AND COORDINATOR PERFORMANCE")

    if not EPISODE_CSV.exists():
        raise FileNotFoundError(f"Required locked TEST results not found:\n{EPISODE_CSV}")

    df = pd.read_csv(EPISODE_CSV)

    required = [
        "mg1_return", "mg2_return", "mg3_return",
        "mg4_return", "mg5_return", "coordinator_return",
        "total_return",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError("Missing required columns: " + ", ".join(missing))

    if len(df) != 1268:
        raise RuntimeError(
            f"Expected 1268 locked TEST windows, found {len(df)}."
        )

    mapping = [
        ("MG1 local agent", "mg1_return"),
        ("MG2 local agent", "mg2_return"),
        ("MG3 local agent", "mg3_return"),
        ("MG4 local agent", "mg4_return"),
        ("MG5 local agent", "mg5_return"),
        ("Upper-level coordinator", "coordinator_return"),
        ("Complete FC-HMARL system", "total_return"),
    ]

    rows = []
    for label, column in mapping:
        s = stats(df[column])
        rows.append({
            "Controller level": label,
            "Mean return": s["Mean return"],
            "Std. deviation": s["Std. deviation"],
            "Median return": s["Median return"],
            "Minimum return": s["Minimum return"],
            "Maximum return": s["Maximum return"],
        })

    table = pd.DataFrame(rows)

    # Internal consistency check: local agents + coordinator = total return.
    reconstructed = (
        df["mg1_return"] + df["mg2_return"] + df["mg3_return"]
        + df["mg4_return"] + df["mg5_return"]
        + df["coordinator_return"]
    )
    max_diff = float(np.max(np.abs(reconstructed - df["total_return"])))

    if max_diff > 1e-6:
        raise RuntimeError(
            f"Reward decomposition inconsistency detected: max difference={max_diff}"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(CSV_OUT, index=False, float_format="%.6f")
    MD_OUT.write_text(table.to_markdown(index=False, floatfmt=".6f"), encoding="utf-8")
    TXT_OUT.write_text(
        "Table — Agent-level performance of the final FC-HMARL controller on the unseen TEST split\n\n"
        + table.to_string(index=False, float_format=lambda x: f"{x:,.6f}"),
        encoding="utf-8",
    )

    section("PUBLICATION TABLE")
    print(table.to_string(index=False, float_format=lambda x: f"{x:,.6f}"))

    section("INTERNAL CONSISTENCY")
    print(f"TEST windows evaluated       : {len(df)}")
    print(f"Max reward decomposition diff: {max_diff:.12e}")
    print("[OK] MG1+MG2+MG3+MG4+MG5+Coordinator = total FC-HMARL return.")

    section("STEP 7R-O2 COMPLETE")
    print("CSV:")
    print(CSV_OUT)
    print()
    print("Markdown:")
    print(MD_OUT)
    print()
    print("Text:")
    print(TXT_OUT)
    print()
    print("[OK] Read locked TEST results only.")
    print("[OK] No policy evaluation or checkpoint selection was performed.")
    print("[OK] Checkpoint 200 remains locked.")

if __name__ == "__main__":
    main()
