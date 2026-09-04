"""
STEP 7R-Q2 — FINAL PUBLICATION COMPARISON FIGURES

Reads only already-locked FINAL V3 benchmark and diagnostic-ablation CSVs.
No training, policy evaluation, replay write, or checkpoint selection.

Creates:
1. Figure_Benchmark_Mean_Return.png/.pdf
2. Figure_Benchmark_Net_Market_Cost.png/.pdf
3. Figure_Benchmark_Operational_Normalized.png/.pdf
4. Figure_Diagnostic_Ablation.png/.pdf
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent

BENCH = (
    ROOT / "outputs" / "results"
    / "real_fc_hmarl_final_v3_benchmark"
    / "final_v3_benchmark_summary.csv"
)
ABL = (
    ROOT / "outputs" / "results"
    / "real_fc_hmarl_final_v3_ablation"
    / "final_v3_ablation_summary.csv"
)
LOCKED = (
    ROOT / "outputs" / "results"
    / "real_fc_hmarl_final_v3_test"
    / "final_test_summary.csv"
)
OUT = (
    ROOT / "outputs" / "results"
    / "real_fc_hmarl_final_v3_test"
    / "publication_figures"
)

DPI = 400

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
})


def save(fig, stem):
    OUT.mkdir(parents=True, exist_ok=True)
    png = OUT / f"{stem}.png"
    pdf = OUT / f"{stem}.pdf"
    fig.savefig(png, dpi=DPI, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return png, pdf


def label_bars(ax, bars, fmt="{:,.1f}"):
    for bar in bars:
        h = bar.get_height()
        ax.annotate(
            fmt.format(h),
            xy=(bar.get_x() + bar.get_width() / 2, h),
            xytext=(0, 5 if h >= 0 else -14),
            textcoords="offset points",
            ha="center",
            va="bottom" if h >= 0 else "top",
            fontsize=8,
        )


def main():
    for path in (BENCH, ABL, LOCKED):
        if not path.exists():
            raise FileNotFoundError(path)

    bench = pd.read_csv(BENCH)
    abl = pd.read_csv(ABL)
    locked = pd.read_csv(LOCKED).iloc[0]

    full_bench = float(
        bench.loc[
            bench["mode"] == "full_fc_hmarl",
            "mean_return",
        ].iloc[0]
    )
    full_abl = float(
        abl.loc[
            abl["mode"] == "full_fc_hmarl",
            "mean_return",
        ].iloc[0]
    )
    locked_return = float(locked["mean_return"])

    if abs(full_bench - locked_return) > 1e-6:
        raise RuntimeError("Benchmark does not reproduce locked TEST return.")
    if abs(full_abl - locked_return) > 1e-6:
        raise RuntimeError("Ablation does not reproduce locked TEST return.")

    # ==========================================================
    # FIGURE 1 — BENCHMARK RETURN
    # ==========================================================
    order = [
        "passive_grid_only",
        "rule_based_bess_ems",
        "full_fc_hmarl",
    ]
    labels = [
        "Passive\nGrid-only",
        "Rule-based\nBESS EMS",
        "FC-HMARL",
    ]
    b = bench.set_index("mode").loc[order]

    fig, ax = plt.subplots(figsize=(6.8, 4.6))
    x = np.arange(3)
    bars = ax.bar(x, b["mean_return"].to_numpy(float), width=0.62)
    ax.errorbar(
        x,
        b["mean_return"].to_numpy(float),
        yerr=b["std_return"].to_numpy(float),
        fmt="none",
        capsize=4,
        linewidth=1.0,
    )
    ax.set_xticks(x, labels)
    ax.set_ylabel("Mean episodic return")
    ax.set_title("Benchmark Performance on 1,268 Held-out TEST Windows")
    ax.grid(axis="y", alpha=0.25)
    label_bars(ax, bars, "{:,.0f}")
    fig.tight_layout()
    f1 = save(fig, "Figure_Benchmark_Mean_Return")

    # ==========================================================
    # FIGURE 2 — MARKET COST
    # ==========================================================
    costs = b["mean_net_market_cost_usd"].to_numpy(float)
    fc_cost = costs[2]
    red_passive = 100 * (costs[0] - fc_cost) / costs[0]
    red_rule = 100 * (costs[1] - fc_cost) / costs[1]

    fig, ax = plt.subplots(figsize=(6.8, 4.6))
    bars = ax.bar(x, costs, width=0.62)
    ax.set_xticks(x, labels)
    ax.set_ylabel("Net market cost (USD/day)")
    ax.set_title("Daily Net Market Cost")
    ax.grid(axis="y", alpha=0.25)
    label_bars(ax, bars, "${:,.0f}")
    ax.text(
        0.98, 0.96,
        f"FC-HMARL reduction:\n"
        f"{red_passive:.2f}% vs passive\n"
        f"{red_rule:.2f}% vs rule-based",
        transform=ax.transAxes,
        ha="right", va="top",
        fontsize=9,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),
    )
    fig.tight_layout()
    f2 = save(fig, "Figure_Benchmark_Net_Market_Cost")

    # ==========================================================
    # FIGURE 3 — NORMALIZED OPERATIONAL METRICS
    # Normalize each metric by the largest value among controllers.
    # This avoids mixing kWh, kW and reserve magnitudes on one axis.
    # ==========================================================
    metrics = [
        ("mean_grid_import_kwh", "Grid\nimport"),
        ("mean_peak_grid_import_kw", "Peak\nimport"),
        ("mean_bess_throughput_kwh", "BESS\nthroughput"),
        ("mean_sharing_scheduled_kwh", "Energy\nsharing"),
        ("mean_reserve_feasible_kwh", "Feasible\nreserve"),
    ]

    matrix = np.column_stack([
        b[col].to_numpy(float)
        for col, _ in metrics
    ])
    denom = np.max(np.abs(matrix), axis=0)
    denom[denom < 1e-12] = 1.0
    normalized = matrix / denom

    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    width = 0.24
    mx = np.arange(len(metrics))
    for i, controller in enumerate(labels):
        ax.bar(
            mx + (i - 1) * width,
            normalized[i],
            width=width,
            label=controller.replace("\n", " "),
        )

    ax.set_xticks(mx, [name for _, name in metrics])
    ax.set_ylabel("Normalized value (maximum = 1)")
    ax.set_ylim(0, 1.16)
    ax.set_title("Normalized Operational Utilization Across Controllers")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(ncol=3, loc="upper center")
    fig.tight_layout()
    f3 = save(fig, "Figure_Benchmark_Operational_Normalized")

    # ==========================================================
    # FIGURE 4 — DIAGNOSTIC ABLATION
    # Plot change in mean return relative to full rather than truncating
    # the raw-return y-axis. Positive means the intervention scored higher.
    # ==========================================================
    abl_order = [
        "no_confidence_awareness",
        "no_energy_sharing",
        "no_upper_level_coordination",
    ]
    abl_labels = [
        "Without confidence\nawareness",
        "Without energy\nsharing",
        "Without upper-level\ncoordination",
    ]

    a = abl.set_index("mode")
    deltas = np.array([
        float(a.loc[m, "mean_return"]) - full_abl
        for m in abl_order
    ])

    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    ax.axhline(0.0, linewidth=1.0)
    bars = ax.bar(np.arange(3), deltas, width=0.62)
    ax.set_xticks(np.arange(3), abl_labels)
    ax.set_ylabel("Change in mean return relative to full FC-HMARL")
    ax.set_title("Post-training Diagnostic Component Ablation")
    ax.grid(axis="y", alpha=0.25)
    label_bars(ax, bars, "{:+,.2f}")

    ax.text(
        0.02, 0.03,
        "Inference-time diagnostic ablations; variants were not separately retrained.",
        transform=ax.transAxes,
        ha="left", va="bottom",
        fontsize=8,
    )
    fig.tight_layout()
    f4 = save(fig, "Figure_Diagnostic_Ablation")

    print("=" * 84)
    print("STEP 7R-Q2 COMPLETE")
    print("=" * 84)
    print("Locked TEST return verified:", f"{locked_return:.6f}")
    print()
    print("Created:")
    for pair in (f1, f2, f3, f4):
        for path in pair:
            print(path)
    print()
    print("[OK] Figures use already-locked benchmark/ablation results only.")
    print("[OK] No training or policy evaluation occurred.")
    print("[OK] Ablation figure explicitly identifies inference-time diagnostics.")
    print("[OK] Small confidence/sharing effects were not artificially magnified by a truncated raw-return axis.")


if __name__ == "__main__":
    main()
