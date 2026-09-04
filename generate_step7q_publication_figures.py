"""
Step 7Q-A: Publication-quality FC-HMARL benchmark and risk figures.

Uses only already validated result files:
1) Step 7O-B three-way benchmark summary
2) Step 7P risk/robustness summary

No training, no model selection, no TEST tuning.

Outputs:
- Fig_7Q_A1_Benchmark_Comparison.png/.pdf
- Fig_7Q_A2_Risk_Comparison.png/.pdf
- Table_7Q_A_Benchmark_Risk.csv
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(r"D:\Molvi paper review\FC_HMARL")

BENCHMARK_FILE = (
    PROJECT_ROOT / "outputs" / "evaluation"
    / "step7o_rule_based_comparison"
    / "step7o_three_way_summary.csv"
)

RISK_FILE = (
    PROJECT_ROOT / "outputs" / "evaluation"
    / "step7p_risk_robustness"
    / "step7p_risk_robustness_summary.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT / "outputs" / "figures" / "step7q"
)

MODE_ORDER = [
    "passive_grid_only",
    "rule_based_bess_ems",
    "full_fc_hmarl",
]

LABELS = [
    "Passive grid-only",
    "Rule-based EMS",
    "FC-HMARL",
]


def save_figure(fig, stem):
    png = OUTPUT_DIR / f"{stem}.png"
    pdf = OUTPUT_DIR / f"{stem}.pdf"
    fig.savefig(png, dpi=600, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    print("Saved:", png)
    print("Saved:", pdf)


def ordered(df):
    out = df.set_index("mode").loc[MODE_ORDER].reset_index()
    return out


def annotate_bars(ax, bars, fmt="{:.1f}"):
    for bar in bars:
        h = bar.get_height()
        ax.annotate(
            fmt.format(h),
            xy=(bar.get_x() + bar.get_width() / 2, h),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
        )


def main():
    if not BENCHMARK_FILE.exists():
        raise FileNotFoundError(BENCHMARK_FILE)
    if not RISK_FILE.exists():
        raise FileNotFoundError(RISK_FILE)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    benchmark = ordered(pd.read_csv(BENCHMARK_FILE))
    risk = ordered(pd.read_csv(RISK_FILE))

    # Publication table
    table = pd.DataFrame({
        "Method": LABELS,
        "Mean Return": benchmark["total_return"],
        "Grid Import (kWh)": benchmark["grid_import_kwh"],
        "Net Market Cost (USD)": benchmark["net_market_cost_usd"],
        "Peak Grid Import (kW)": benchmark["peak_grid_import_kw"],
        "BESS Throughput (kWh)": benchmark["bess_throughput_kwh"],
        "Energy Sharing (kWh)": benchmark["sharing_scheduled_kwh"],
        "Cost VaR95 (USD)": risk["cost_var95_usd"],
        "Cost CVaR95 (USD)": risk["cost_cvar95_usd"],
        "Worst Cost (USD)": risk["maximum_net_market_cost_usd"],
        "Power Balance Violations":
            benchmark["power_balance_violation_steps"],
        "Transformer Violations":
            benchmark["transformer_violation_steps"],
    })

    table_file = OUTPUT_DIR / "Table_7Q_A_Benchmark_Risk.csv"
    table.to_csv(table_file, index=False)
    print("Saved:", table_file)

    x = np.arange(len(LABELS))

    # ------------------------------------------------------------
    # Figure A1: economic / operational benchmark
    # ------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.3))

    metrics = [
        ("net_market_cost_usd", "Net market cost (USD)", "{:.0f}"),
        ("grid_import_kwh", "Grid import (kWh)", "{:.0f}"),
        ("peak_grid_import_kw", "Peak grid import (kW)", "{:.0f}"),
    ]

    for ax, (column, ylabel, fmt) in zip(axes, metrics):
        values = benchmark[column].to_numpy(dtype=float)
        bars = ax.bar(x, values, width=0.62)

        ax.set_xticks(x)
        ax.set_xticklabels(LABELS, rotation=12, ha="right")
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.25)
        ax.set_axisbelow(True)

        annotate_bars(ax, bars, fmt)

        upper = max(values) * 1.14
        ax.set_ylim(0, upper)

    axes[0].set_title("(a) Economic cost")
    axes[1].set_title("(b) Grid dependence")
    axes[2].set_title("(c) Peak demand")

    fig.suptitle(
        "Held-out TEST Benchmark Comparison",
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save_figure(fig, "Fig_7Q_A1_Benchmark_Comparison")
    plt.close(fig)

    # ------------------------------------------------------------
    # Figure A2: risk comparison
    # ------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8.2, 5.2))

    width = 0.24

    mean_cost = risk["mean_net_market_cost_usd"].to_numpy(float)
    var95 = risk["cost_var95_usd"].to_numpy(float)
    cvar95 = risk["cost_cvar95_usd"].to_numpy(float)

    b1 = ax.bar(
        x - width,
        mean_cost,
        width,
        label="Mean cost",
    )
    b2 = ax.bar(
        x,
        var95,
        width,
        label=r"VaR$_{95}$",
    )
    b3 = ax.bar(
        x + width,
        cvar95,
        width,
        label=r"CVaR$_{95}$",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(LABELS)
    ax.set_ylabel("Episode net market cost (USD)")
    ax.set_title(
        "Operating-Cost Risk on Held-out TEST Episodes",
        fontweight="bold",
    )
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    ax.legend(frameon=False)

    annotate_bars(ax, b1, "{:.0f}")
    annotate_bars(ax, b2, "{:.0f}")
    annotate_bars(ax, b3, "{:.0f}")

    ymax = max(
        np.max(mean_cost),
        np.max(var95),
        np.max(cvar95),
    ) * 1.14
    ax.set_ylim(0, ymax)

    fig.tight_layout()
    save_figure(fig, "Fig_7Q_A2_Risk_Comparison")
    plt.close(fig)

    print()
    print("=" * 80)
    print("STEP 7Q-A SUMMARY")
    print("=" * 80)
    print(table.to_string(index=False))
    print()
    print("[OK] Step 7Q-A publication benchmark/risk figures complete.")


if __name__ == "__main__":
    main()
