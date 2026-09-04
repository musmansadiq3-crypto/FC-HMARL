"""
STEP 7R-O4 — BUILD FINAL TEST PUBLICATION FIGURES

Reads only the already locked FINAL TEST episode results.
Creates separate publication figures. No policy evaluation, training,
checkpoint comparison, or re-selection occurs.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
RESULT_DIR = ROOT / "outputs" / "results" / "real_fc_hmarl_final_v3_test"
EPISODE_CSV = RESULT_DIR / "final_test_episode_results.csv"
OUTPUT_DIR = RESULT_DIR / "publication_figures"

FIG1 = OUTPUT_DIR / "Figure_Final_TEST_Return_Distribution.png"
FIG2 = OUTPUT_DIR / "Figure_Final_TEST_Net_Market_Cost.png"
FIG3 = OUTPUT_DIR / "Figure_Final_TEST_Grid_Interaction.png"
FIG4 = OUTPUT_DIR / "Figure_Final_TEST_Resource_Utilization.png"


def section(title):
    print()
    print("=" * 80)
    print(title)
    print("=" * 80)


def save_current(path):
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


def main():
    section("STEP 7R-O4 — FINAL TEST PUBLICATION FIGURES")

    if not EPISODE_CSV.exists():
        raise FileNotFoundError(
            f"Locked TEST episode results not found:\n{EPISODE_CSV}"
        )

    df = pd.read_csv(EPISODE_CSV)

    if len(df) != 1268:
        raise RuntimeError(
            f"Expected 1268 locked TEST windows, found {len(df)}."
        )

    required = [
        "test_episode",
        "total_return",
        "net_market_cost_usd",
        "grid_import_kwh",
        "grid_export_kwh",
        "bess_throughput_kwh",
        "sharing_scheduled_kwh",
        "reserve_feasible_kwh",
    ]

    missing = [c for c in required if c not in df.columns]

    if missing:
        raise RuntimeError(
            "Missing required columns:\n  " + "\n  ".join(missing)
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    x = df["test_episode"].to_numpy(dtype=int)

    # ---------------------------------------------------------
    # FIGURE 1 — TEST RETURN DISTRIBUTION
    # ---------------------------------------------------------
    returns = df["total_return"].to_numpy(dtype=float)

    plt.figure(figsize=(7.5, 5.2))
    plt.hist(returns, bins=35)
    plt.axvline(
        np.mean(returns),
        linestyle="--",
        linewidth=1.5,
        label=f"Mean = {np.mean(returns):,.0f}",
    )
    plt.axvline(
        np.median(returns),
        linestyle=":",
        linewidth=1.5,
        label=f"Median = {np.median(returns):,.0f}",
    )
    plt.xlabel("Total FC-HMARL Return")
    plt.ylabel("Number of 24-h TEST Windows")
    plt.title("Distribution of Final FC-HMARL TEST Returns")
    plt.legend()
    plt.grid(True, alpha=0.25)
    save_current(FIG1)

    # ---------------------------------------------------------
    # FIGURE 2 — NET MARKET COST ACROSS TEST WINDOWS
    # ---------------------------------------------------------
    net_cost = df["net_market_cost_usd"].to_numpy(dtype=float)

    plt.figure(figsize=(8.2, 5.2))
    plt.plot(x, net_cost, linewidth=1.0)
    plt.axhline(
        np.mean(net_cost),
        linestyle="--",
        linewidth=1.5,
        label=f"Mean = ${np.mean(net_cost):,.1f}/day",
    )
    plt.xlabel("Rolling 24-h TEST Window")
    plt.ylabel("Net Market Cost (USD/day)")
    plt.title("Final FC-HMARL Net Market Cost on the TEST Split")
    plt.legend()
    plt.grid(True, alpha=0.25)
    save_current(FIG2)

    # ---------------------------------------------------------
    # FIGURE 3 — GRID IMPORT / EXPORT
    # ---------------------------------------------------------
    grid_import = df["grid_import_kwh"].to_numpy(dtype=float)
    grid_export = df["grid_export_kwh"].to_numpy(dtype=float)

    plt.figure(figsize=(8.2, 5.2))
    plt.plot(
        x,
        grid_import,
        linewidth=1.0,
        label="Grid import",
    )
    plt.plot(
        x,
        grid_export,
        linewidth=1.0,
        label="Grid export",
    )
    plt.xlabel("Rolling 24-h TEST Window")
    plt.ylabel("Energy (kWh/day)")
    plt.title("Grid Interaction of the Final FC-HMARL Controller")
    plt.legend()
    plt.grid(True, alpha=0.25)
    save_current(FIG3)

    # ---------------------------------------------------------
    # FIGURE 4 — RESOURCE UTILIZATION
    # ---------------------------------------------------------
    bess = df["bess_throughput_kwh"].to_numpy(dtype=float)
    sharing = df["sharing_scheduled_kwh"].to_numpy(dtype=float)
    reserve = df["reserve_feasible_kwh"].to_numpy(dtype=float)

    plt.figure(figsize=(8.2, 5.2))
    plt.plot(
        x,
        bess,
        linewidth=1.0,
        label="BESS throughput",
    )
    plt.plot(
        x,
        sharing,
        linewidth=1.0,
        label="Energy sharing",
    )
    plt.plot(
        x,
        reserve,
        linewidth=1.0,
        label="Feasible reserve",
    )
    plt.xlabel("Rolling 24-h TEST Window")
    plt.ylabel("Energy (kWh/day)")
    plt.title("FC-HMARL Resource Utilization on the TEST Split")
    plt.legend()
    plt.grid(True, alpha=0.25)
    save_current(FIG4)

    section("STEP 7R-O4 COMPLETE")

    print("Figure 1:")
    print(FIG1)
    print()
    print("Figure 2:")
    print(FIG2)
    print()
    print("Figure 3:")
    print(FIG3)
    print()
    print("Figure 4:")
    print(FIG4)
    print()
    print("[OK] Figures use locked TEST results only.")
    print("[OK] No model evaluation, training, or checkpoint selection was performed.")
    print("[OK] Checkpoint 200 remains permanently locked.")


if __name__ == "__main__":
    main()
