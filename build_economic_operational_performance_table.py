"""
STEP 7R-O3 — BUILD ECONOMIC AND OPERATIONAL PERFORMANCE TABLE

Reads only the already locked FINAL TEST episode results.
No model evaluation, training, checkpoint comparison, or re-selection occurs.
"""

from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
RESULT_DIR = ROOT / "outputs" / "results" / "real_fc_hmarl_final_v3_test"
EPISODE_CSV = RESULT_DIR / "final_test_episode_results.csv"
OUTPUT_DIR = RESULT_DIR / "publication_tables"

CSV_OUT = OUTPUT_DIR / "Table_Economic_Operational_Performance.csv"
MD_OUT = OUTPUT_DIR / "Table_Economic_Operational_Performance.md"
TXT_OUT = OUTPUT_DIR / "Table_Economic_Operational_Performance.txt"


def section(title):
    print()
    print("=" * 80)
    print(title)
    print("=" * 80)


def stat_row(category, metric, column, unit, df):
    x = pd.to_numeric(df[column], errors="raise").to_numpy(dtype=float)
    return {
        "Category": category,
        "Performance index": metric,
        "Mean": float(np.mean(x)),
        "Std. deviation": float(np.std(x, ddof=0)),
        "Median": float(np.median(x)),
        "Minimum": float(np.min(x)),
        "Maximum": float(np.max(x)),
        "Unit": unit,
    }


def main():
    section("STEP 7R-O3 — ECONOMIC AND OPERATIONAL PERFORMANCE")

    if not EPISODE_CSV.exists():
        raise FileNotFoundError(
            f"Locked TEST episode results not found:\n{EPISODE_CSV}"
        )

    df = pd.read_csv(EPISODE_CSV)

    if len(df) != 1268:
        raise RuntimeError(
            f"Expected 1268 locked TEST windows, found {len(df)}."
        )

    specs = [
        ("Economic", "Grid purchase cost",
         "grid_purchase_cost_usd", "USD/day"),
        ("Economic", "Grid sale revenue",
         "grid_sale_revenue_usd", "USD/day"),
        ("Economic", "Reserve revenue",
         "reserve_revenue_usd", "USD/day"),
        ("Economic", "Net market cost",
         "net_market_cost_usd", "USD/day"),
        ("Economic", "Market profit",
         "market_profit_usd", "USD/day"),

        ("Grid interaction", "Grid import",
         "grid_import_kwh", "kWh/day"),
        ("Grid interaction", "Grid export",
         "grid_export_kwh", "kWh/day"),
        ("Grid interaction", "Peak grid import",
         "peak_grid_import_kw", "kW"),
        ("Grid interaction", "Peak grid export",
         "peak_grid_export_kw", "kW"),

        ("Reserve", "Requested reserve",
         "reserve_requested_kwh", "kWh/day"),
        ("Reserve", "Feasible reserve",
         "reserve_feasible_kwh", "kWh/day"),
        ("Reserve", "Curtailed reserve",
         "reserve_curtailed_kwh", "kWh/day"),

        ("BESS", "BESS energy throughput",
         "bess_throughput_kwh", "kWh/day"),

        ("Energy sharing", "Scheduled sharing",
         "sharing_scheduled_kwh", "kWh/day"),
        ("Energy sharing", "Received sharing",
         "sharing_received_kwh", "kWh/day"),
        ("Energy sharing", "Sharing loss",
         "sharing_loss_kwh", "kWh/day"),

        ("Physical feasibility", "Mean balance violation",
         "mean_balance_violation_kw_per_step", "kW/step"),
        ("Physical feasibility", "Maximum balance violation",
         "maximum_balance_violation_kw", "kW"),
    ]

    missing = [
        column
        for _, _, column, _ in specs
        if column not in df.columns
    ]

    if missing:
        raise RuntimeError(
            "Missing required columns:\n  "
            + "\n  ".join(missing)
        )

    rows = [
        stat_row(category, metric, column, unit, df)
        for category, metric, column, unit in specs
    ]

    table = pd.DataFrame(rows)

    # Exact global feasibility counts.
    transformer_steps = int(
        pd.to_numeric(
            df["transformer_violation_steps"],
            errors="raise",
        ).sum()
    )

    balance_steps = int(
        pd.to_numeric(
            df["power_balance_violation_steps"],
            errors="raise",
        ).sum()
    )

    soc_low = int(
        pd.to_numeric(
            df["soc_below_min_count"],
            errors="raise",
        ).sum()
    )

    soc_high = int(
        pd.to_numeric(
            df["soc_above_max_count"],
            errors="raise",
        ).sum()
    )

    # Reserve consistency checks.
    reserve_requested = pd.to_numeric(
        df["reserve_requested_kwh"],
        errors="raise",
    ).to_numpy(dtype=float)

    reserve_feasible = pd.to_numeric(
        df["reserve_feasible_kwh"],
        errors="raise",
    ).to_numpy(dtype=float)

    reserve_curtailed = pd.to_numeric(
        df["reserve_curtailed_kwh"],
        errors="raise",
    ).to_numpy(dtype=float)

    reserve_diff = np.max(
        np.abs(
            reserve_requested
            - reserve_feasible
            - reserve_curtailed
        )
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    table.to_csv(
        CSV_OUT,
        index=False,
        float_format="%.6f",
    )

    MD_OUT.write_text(
        table.to_markdown(
            index=False,
            floatfmt=".6f",
        ),
        encoding="utf-8",
    )

    TXT_OUT.write_text(
        "Table — Economic and operational performance of FC-HMARL on the unseen TEST split\n\n"
        + table.to_string(
            index=False,
            float_format=lambda x: f"{x:,.6f}",
        )
        + "\n\n"
        + f"Power-balance violation steps: {balance_steps}\n"
        + f"Transformer violation steps: {transformer_steps}\n"
        + f"SOC lower-bound violations: {soc_low}\n"
        + f"SOC upper-bound violations: {soc_high}\n"
        + f"Maximum reserve accounting difference: {reserve_diff:.12e}\n",
        encoding="utf-8",
    )

    section("PUBLICATION TABLE")
    print(
        table.to_string(
            index=False,
            float_format=lambda x: f"{x:,.6f}",
        )
    )

    section("PHYSICAL / ACCOUNTING AUDIT")
    print(f"Power-balance violation steps : {balance_steps}")
    print(f"Transformer violation steps   : {transformer_steps}")
    print(f"SOC lower-bound violations    : {soc_low}")
    print(f"SOC upper-bound violations    : {soc_high}")
    print(
        "Max reserve identity diff    : "
        f"{reserve_diff:.12e}"
    )

    if reserve_diff > 1e-6:
        raise RuntimeError(
            "Reserve accounting identity failed."
        )

    section("STEP 7R-O3 COMPLETE")
    print("CSV:")
    print(CSV_OUT)
    print()
    print("Markdown:")
    print(MD_OUT)
    print()
    print("Text:")
    print(TXT_OUT)
    print()
    print("[OK] Locked TEST results only.")
    print("[OK] Reserve requested = feasible + curtailed.")
    print("[OK] No policy evaluation or checkpoint selection was performed.")
    print("[OK] Checkpoint 200 remains permanently locked.")


if __name__ == "__main__":
    main()
