from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
ROOT = Path(__file__).resolve().parent
SOURCE_CSV = (
    ROOT
    / "outputs"
    / "results"
    / "real_fc_hmarl_final_v3"
    / "training_history.csv"
)
DATA_DIR = (
    ROOT
    / "evaluation"
    / "data"
    / "training_diagnostics"
)

FIGURE_DIR = (
    ROOT
    / "outputs"
    / "figures"
)

OUTPUT_XLSX = DATA_DIR / "final_v3_training_convergence_data.xlsx"

OUTPUT_PNG = FIGURE_DIR / "final_v3_training_convergence.png"
OUTPUT_PDF = FIGURE_DIR / "final_v3_training_convergence.pdf"
OUTPUT_SVG = FIGURE_DIR / "final_v3_training_convergence.svg"


def main():

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    if not SOURCE_CSV.exists():
        raise FileNotFoundError(
            f"Final V3 training history not found:\n{SOURCE_CSV}"
        )

    df = pd.read_csv(SOURCE_CSV)

    required = [
        "episode",
        "total_return",
        "coordinator_return",
        "mean_local_return",
        "moving_average_return",
        "moving_average_coordinator_return",
        "moving_average_local_return",
        "mg1_return",
        "mg2_return",
        "mg3_return",
        "mg4_return",
        "mg5_return",
        "local_update_count",
        "coordinator_update_count",
    ]

    missing = [c for c in required if c not in df.columns]

    if missing:
        raise ValueError(
            "Missing required training-history columns: "
            + ", ".join(missing)
        )

    if len(df) != 5000:
        raise ValueError(
            f"Expected 5000 final-V3 episodes, found {len(df)}."
        )

    best_idx = df["moving_average_return"].idxmax()
    best = df.loc[best_idx]

    last = df.iloc[-1]
    last500 = df.tail(500)

    summary = pd.DataFrame(
        {
            "Metric": [
                "Number_of_Episodes",
                "Final_Episode",
                "Final_Total_Return",
                "Final_Moving_Average_Return",
                "Best_Moving_Average_Episode",
                "Best_Moving_Average_Return",
                "Raw_Return_at_Best_MA_Episode",
                "Mean_Training_Return",
                "Std_Training_Return",
                "Median_Training_Return",
                "Last_500_Mean_Return",
                "Last_500_Std_Return",
                "Last_500_Mean_Moving_Average_Return",
            ],
            "Value": [
                len(df),
                int(last["episode"]),
                float(last["total_return"]),
                float(last["moving_average_return"]),
                int(best["episode"]),
                float(best["moving_average_return"]),
                float(best["total_return"]),
                float(df["total_return"].mean()),
                float(df["total_return"].std()),
                float(df["total_return"].median()),
                float(last500["total_return"].mean()),
                float(last500["total_return"].std()),
                float(last500["moving_average_return"].mean()),
            ],
        }
    )

    provenance = pd.DataFrame(
        {
            "Item": [
                "Source",
                "Model",
                "Training_Episodes",
                "Moving_Average_Source",
                "Best_Point_Interpretation",
                "Convergence_Claim",
                "Data_Modification",
            ],
            "Description": [
                str(SOURCE_CSV.relative_to(ROOT)),
                "Final FC-HMARL V3",
                "5000",
                "moving_average_return recorded by final V3 training history",
                "Best recorded moving-average training return",
                "No formal convergence episode is claimed",
                "No synthetic or random perturbation applied",
            ],
        }
    )

    with pd.ExcelWriter(OUTPUT_XLSX, engine="openpyxl") as writer:
        df.to_excel(
            writer,
            sheet_name="Final_V3_Training_History",
            index=False,
        )
        summary.to_excel(
            writer,
            sheet_name="Final_V3_Summary",
            index=False,
        )
        provenance.to_excel(
            writer,
            sheet_name="Provenance",
            index=False,
        )

    episodes = df["episode"].to_numpy(dtype=float)
    raw_return = df["total_return"].to_numpy(dtype=float)
    moving_return = df["moving_average_return"].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(10.5, 6.2))

    ax.plot(
        episodes,
        raw_return,
        linewidth=0.55,
        alpha=0.25,
        label="Episode Return",
    )

    ax.plot(
        episodes,
        moving_return,
        linewidth=2.2,
        label="Moving-Average Return",
    )

    best_episode = int(best["episode"])
    best_return = float(best["moving_average_return"])

    ax.scatter(
        [best_episode],
        [best_return],
        s=75,
        zorder=5,
        label="Best Moving-Average Return",
    )

    ax.annotate(
        f"Best moving average\nEpisode {best_episode}\nReturn = {best_return:,.0f}",
        xy=(best_episode, best_return),
        xytext=(best_episode + 450, best_return - 6000),
        arrowprops=dict(
            arrowstyle="->",
            linewidth=1.2,
        ),
        fontsize=10,
    )

    ax.axvline(
        500,
        linestyle="--",
        linewidth=1.0,
        alpha=0.7,
    )

    ax.text(
        540,
        ax.get_ylim()[0] * 0.90,
        "Early high-variance\nexploration region",
        fontsize=9,
        va="bottom",
    )

    ax.set_xlabel("Training Episode", fontsize=12)
    ax.set_ylabel("Episode Return", fontsize=12)

    ax.set_title(
        "Final FC-HMARL V3 Training Performance",
        fontsize=13,
        fontweight="bold",
    )

    ax.grid(
        True,
        linestyle="--",
        linewidth=0.5,
        alpha=0.3,
    )

    ax.legend(
        loc="lower right",
        frameon=True,
        fontsize=10,
    )

    ax.tick_params(
        axis="both",
        labelsize=10,
    )

    fig.tight_layout()

    fig.savefig(
        OUTPUT_PNG,
        dpi=600,
        bbox_inches="tight",
    )

    fig.savefig(
        OUTPUT_PDF,
        bbox_inches="tight",
    )

    fig.savefig(
        OUTPUT_SVG,
        bbox_inches="tight",
    )

    plt.close(fig)

    print("=" * 72)
    print("FINAL V3 TRAINING CONVERGENCE ARTIFACTS CREATED")
    print("=" * 72)

    print(f"Source episodes              : {len(df)}")
    print(f"Best moving-average episode  : {best_episode}")
    print(f"Best moving-average return   : {best_return:.6f}")
    print(
        f"Final episode return         : "
        f"{float(last['total_return']):.6f}"
    )
    print(
        f"Last-500 mean return         : "
        f"{float(last500['total_return'].mean()):.6f}"
    )

    print()
    print("Workbook:")
    print(OUTPUT_XLSX)

    print()
    print("Figures:")
    print(OUTPUT_PNG)
    print(OUTPUT_PDF)
    print(OUTPUT_SVG)

    print()
    print("[OK] No formal convergence episode claimed.")
    print("[OK] No synthetic/random training values generated.")
    print("[OK] Source = genuine final V3 training_history.csv.")


if __name__ == "__main__":
    main()
