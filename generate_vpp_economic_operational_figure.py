from pathlib import Path as FilePath

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

ROOT = FilePath(__file__).resolve().parent
DATA_PATH = ROOT / "evaluation" / "data" / "vpp_performance" / "vpp_economic_operational_performance.xlsx"
OUTPUT_DIR = ROOT / "outputs" / "figures"
BATTERY_RATED_POWER_KW = 1450.0

FONT_FAMILY = "Times New Roman"
FONT_WEIGHT = "bold"
GLOBAL_FONT_SIZE = 24

FIG_WIDTH = 16
FIG_HEIGHT = 16
FIG_DPI = 100
SUBPLOT_HSPACE = 0.7

PANEL_TITLE_SIZE = 30
AXIS_LABEL_SIZE = 30
TICK_LABEL_SIZE = 30
LEGEND_FONT_SIZE_A = 18
LEGEND_FONT_SIZE_B = 18

SPINE_LINEWIDTH = 1.5
SPINE_COLOR = "#2C3E50"
TICK_MAJOR_WIDTH = 1.0
TICK_MAJOR_LENGTH = 5.0
TICK_DIRECTION = "in"
BAR_EDGE_LINEWIDTH = 1.5

LINEWIDTH_PROFIT_MAIN = 3.0
LINEWIDTH_PROFIT_SHADOW = 8.0
LINEWIDTH_PROFIT_GLOW_1 = 15.0
LINEWIDTH_PROFIT_GLOW_2 = 10.0
LINEWIDTH_PROFIT_GLOW_3 = 5.0
LINEWIDTH_TREND_LINES = 3.0

GRID_LINEWIDTH_MAJOR = 1.5
GRID_LINEWIDTH_MINOR = 1.0
GRID_ALPHA_MAJOR = 0.3
GRID_ALPHA_MINOR = 0.15

BAR_WIDTH_PANEL_A = 0.65
BAR_WIDTH_PANEL_B = 0.8

plt.rcParams["font.family"] = FONT_FAMILY
plt.rcParams["font.size"] = GLOBAL_FONT_SIZE
plt.rcParams["font.weight"] = FONT_WEIGHT
plt.rcParams["axes.linewidth"] = SPINE_LINEWIDTH
plt.rcParams["axes.edgecolor"] = SPINE_COLOR
plt.rcParams["xtick.direction"] = TICK_DIRECTION
plt.rcParams["ytick.direction"] = TICK_DIRECTION
plt.rcParams["xtick.major.width"] = TICK_MAJOR_WIDTH
plt.rcParams["ytick.major.width"] = TICK_MAJOR_WIDTH
plt.rcParams["xtick.labelsize"] = TICK_LABEL_SIZE
plt.rcParams["ytick.labelsize"] = TICK_LABEL_SIZE

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Data file not found:\n{DATA_PATH}\n\n"
        "Place the Excel workbook at:\n"
        r"evaluation\data\vpp_performance\vpp_economic_operational_performance.xlsx"
    )

df = pd.read_excel(DATA_PATH)

required_columns = [
    "Hour",
    "Energy_Trading_USD",
    "Energy_Sharing_USD",
    "Reserve_Market_USD",
    "Net_Profit_USD",
    "Risk_Adj_Profit_USD",
    "Sharing_Ratio_Perc",
    "Grid_Dependency_Perc",
    "Self_Sufficiency_Perc",
    "Battery_Cont_kW",
]

missing_columns = [c for c in required_columns if c not in df.columns]

if missing_columns:
    raise ValueError(f"Missing columns in workbook: {missing_columns}")

for column in required_columns:
    if column == "Hour":
        continue
    df[column] = pd.to_numeric(df[column], errors="raise")

hours = pd.to_numeric(df["Hour"], errors="raise").to_numpy(dtype=float)

df["Battery_Contribution_Perc"] = (
    100.0 * df["Battery_Cont_kW"] / BATTERY_RATED_POWER_KW
)

fig, (ax1, ax2) = plt.subplots(
    2,
    1,
    figsize=(FIG_WIDTH, FIG_HEIGHT),
    dpi=FIG_DPI,
)

plt.subplots_adjust(hspace=SUBPLOT_HSPACE)

def add_3d_shadow_bars(ax, x, y, bar_width, alpha=0.15, offset=3):
    for xi, yi in zip(x, y):
        if yi > 0:
            shadow = Rectangle(
                (xi - bar_width / 2 + offset / 2, 0),
                bar_width,
                yi,
                facecolor="black",
                alpha=alpha,
                edgecolor="none",
                zorder=0,
            )
            ax.add_patch(shadow)

            side_shadow = Rectangle(
                (xi + bar_width / 2, 0),
                offset / 2,
                yi,
                facecolor="black",
                alpha=alpha * 0.5,
                edgecolor="none",
                zorder=0,
            )
            ax.add_patch(side_shadow)

def add_3d_highlight(ax, bar, height_percent=0.85, alpha=0.15):
    if bar.get_height() > 0:
        highlight = Rectangle(
            (bar.get_x(), bar.get_height() * height_percent),
            bar.get_width(),
            bar.get_height() * (1 - height_percent),
            facecolor="white",
            alpha=alpha,
            edgecolor="none",
            zorder=10,
        )
        ax.add_patch(highlight)

        gradient_line = Rectangle(
            (bar.get_x(), bar.get_height() * 0.5),
            bar.get_width(),
            0.5,
            facecolor="white",
            alpha=alpha * 0.3,
            edgecolor="none",
            zorder=9,
        )
        ax.add_patch(gradient_line)

width = BAR_WIDTH_PANEL_A

ax1.text(
    0.5,
    1.02,
    "(a) Economic Performance",
    transform=ax1.transAxes,
    fontsize=PANEL_TITLE_SIZE,
    fontweight=FONT_WEIGHT,
    fontname=FONT_FAMILY,
    ha="center",
    va="bottom",
    color="black",
)

trading_bars = ax1.bar(
    hours,
    df["Energy_Trading_USD"],
    width,
    label="Energy Trading",
    color="#4A81BF",
    edgecolor="black",
    linewidth=BAR_EDGE_LINEWIDTH,
    hatch="//",
    alpha=0.9,
)

add_3d_shadow_bars(
    ax1,
    hours,
    df["Energy_Trading_USD"],
    width,
    alpha=0.12,
)

sharing_bars = ax1.bar(
    hours,
    df["Energy_Sharing_USD"],
    width,
    bottom=df["Energy_Trading_USD"],
    label="Energy Sharing",
    color="#F2B705",
    edgecolor="black",
    linewidth=BAR_EDGE_LINEWIDTH,
    hatch="\\\\",
    alpha=0.9,
)

add_3d_shadow_bars(
    ax1,
    hours,
    df["Energy_Sharing_USD"] + df["Energy_Trading_USD"],
    width,
    alpha=0.12,
)

reserve_bars = ax1.bar(
    hours,
    df["Reserve_Market_USD"],
    width,
    bottom=df["Energy_Trading_USD"] + df["Energy_Sharing_USD"],
    label="Reserve Market",
    color="#45B058",
    edgecolor="black",
    linewidth=BAR_EDGE_LINEWIDTH,
    hatch="xx",
    alpha=0.9,
)

add_3d_shadow_bars(
    ax1,
    hours,
    df["Reserve_Market_USD"]
    + df["Energy_Trading_USD"]
    + df["Energy_Sharing_USD"],
    width,
    alpha=0.12,
)

for bar in list(trading_bars) + list(sharing_bars) + list(reserve_bars):
    add_3d_highlight(
        ax1,
        bar,
        height_percent=0.8,
        alpha=0.2,
    )

ax1.plot(
    hours,
    df["Net_Profit_USD"],
    color="black",
    linewidth=LINEWIDTH_PROFIT_GLOW_1,
    alpha=0.05,
    zorder=2,
)

ax1.plot(
    hours,
    df["Net_Profit_USD"],
    color="black",
    linewidth=LINEWIDTH_PROFIT_GLOW_2,
    alpha=0.08,
    zorder=3,
)

ax1.plot(
    hours,
    df["Net_Profit_USD"],
    color="black",
    linewidth=LINEWIDTH_PROFIT_GLOW_3,
    alpha=0.15,
    zorder=4,
)

ax1.plot(
    hours,
    df["Net_Profit_USD"],
    color="#D94333",
    linewidth=LINEWIDTH_PROFIT_MAIN,
    label="Net Profit",
    marker="o",
    markersize=12,
    markeredgecolor="black",
    markeredgewidth=2,
    markerfacecolor="#FF6B4A",
    zorder=5,
)

for i, (x, y) in enumerate(zip(hours, df["Net_Profit_USD"])):
    if i % 2 == 0:
        circle = plt.Circle(
            (x, y),
            8,
            color="#D94333",
            alpha=0.15,
            zorder=1,
        )
        ax1.add_patch(circle)

ax1.plot(
    hours,
    df["Risk_Adj_Profit_USD"],
    color="#8E44AD",
    linewidth=LINEWIDTH_PROFIT_MAIN,
    linestyle="--",
    label="Risk-Adjusted Profit",
    alpha=0.9,
    marker="s",
    markersize=8,
    markeredgecolor="black",
    markerfacecolor="#9B59B6",
    markeredgewidth=1.5,
)

ax1.plot(
    hours,
    df["Risk_Adj_Profit_USD"],
    color="black",
    linewidth=LINEWIDTH_PROFIT_SHADOW,
    alpha=0.05,
    zorder=0,
)

for i in range(1, len(hours) - 1):
    current = df["Net_Profit_USD"].iloc[i]
    previous = df["Net_Profit_USD"].iloc[i - 1]

    if current > previous + 2:
        ax1.annotate(
            "▲",
            xy=(hours[i], current + 3),
            fontsize=14,
            color="darkgreen",
            fontweight="bold",
            ha="center",
            va="bottom",
            alpha=0.7,
        )
    elif current < previous - 2:
        ax1.annotate(
            "▼",
            xy=(hours[i], current - 3),
            fontsize=14,
            color="darkred",
            fontweight="bold",
            ha="center",
            va="top",
            alpha=0.7,
        )

ax1.axhline(
    0,
    color="gray",
    linestyle="-",
    linewidth=2,
    alpha=0.3,
)

ax1.axhspan(
    -5,
    0,
    color="red",
    alpha=0.05,
)

ax1.axhspan(
    0,
    10,
    color="green",
    alpha=0.05,
)

ax1.fill_between(
    hours,
    df["Net_Profit_USD"],
    0,
    color="#D94333",
    alpha=0.08,
    zorder=1,
)

ax1.fill_between(
    hours,
    df["Risk_Adj_Profit_USD"],
    0,
    color="#8E44AD",
    alpha=0.05,
    zorder=1,
)

ax2.text(
    0.5,
    1.02,
    "(b) Operational Indices",
    transform=ax2.transAxes,
    fontsize=PANEL_TITLE_SIZE,
    fontweight=FONT_WEIGHT,
    fontname=FONT_FAMILY,
    ha="center",
    va="bottom",
    color="black",
)

bar_width = BAR_WIDTH_PANEL_B
x_positions = hours

sharing_bars = ax2.bar(
    x_positions,
    df["Sharing_Ratio_Perc"],
    bar_width,
    label="Sharing Ratio",
    color="#2C527A",
    edgecolor="black",
    linewidth=BAR_EDGE_LINEWIDTH,
    hatch="//",
    alpha=0.85,
)

add_3d_shadow_bars(
    ax2,
    x_positions,
    df["Sharing_Ratio_Perc"],
    bar_width,
    alpha=0.12,
)

grid_bars = ax2.bar(
    x_positions + bar_width / 4,
    df["Grid_Dependency_Perc"],
    bar_width * 0.6,
    label="Grid Dependency",
    color="#942115",
    edgecolor="black",
    linewidth=BAR_EDGE_LINEWIDTH,
    hatch="\\\\",
    alpha=0.85,
)

add_3d_shadow_bars(
    ax2,
    x_positions + bar_width / 4,
    df["Grid_Dependency_Perc"],
    bar_width * 0.6,
    alpha=0.12,
)

self_bars = ax2.bar(
    x_positions - bar_width / 4,
    df["Self_Sufficiency_Perc"],
    bar_width * 0.6,
    label="Self-Sufficiency",
    color="#27AE60",
    edgecolor="black",
    linewidth=BAR_EDGE_LINEWIDTH,
    hatch="xx",
    alpha=0.85,
)

add_3d_shadow_bars(
    ax2,
    x_positions - bar_width / 4,
    df["Self_Sufficiency_Perc"],
    bar_width * 0.6,
    alpha=0.12,
)

battery_bars = ax2.bar(
    x_positions + bar_width / 2,
    df["Battery_Contribution_Perc"],
    bar_width * 0.5,
    label="Battery Contribution",
    color="#D4AC0D",
    edgecolor="black",
    linewidth=BAR_EDGE_LINEWIDTH,
    hatch="++",
    alpha=0.85,
)

add_3d_shadow_bars(
    ax2,
    x_positions + bar_width / 2,
    df["Battery_Contribution_Perc"],
    bar_width * 0.5,
    alpha=0.12,
)

for bar in list(sharing_bars) + list(grid_bars) + list(self_bars) + list(battery_bars):
    add_3d_highlight(
        ax2,
        bar,
        height_percent=0.82,
        alpha=0.18,
    )

for i in range(1, len(hours) - 1):
    current = df["Sharing_Ratio_Perc"].iloc[i]
    previous = df["Sharing_Ratio_Perc"].iloc[i - 1]

    if current > previous + 3:
        ax2.annotate(
            "↑",
            xy=(hours[i], current + 4),
            fontsize=16,
            color="green",
            fontweight=FONT_WEIGHT,
            ha="center",
            va="bottom",
            alpha=0.8,
        )
    elif current < previous - 3:
        ax2.annotate(
            "↓",
            xy=(hours[i], current - 4),
            fontsize=16,
            color="red",
            fontweight=FONT_WEIGHT,
            ha="center",
            va="top",
            alpha=0.8,
        )

ax2.plot(
    hours,
    df["Sharing_Ratio_Perc"],
    color="#2C527A",
    linewidth=LINEWIDTH_TREND_LINES,
    linestyle="-",
    alpha=0.6,
    zorder=15,
)

ax2.plot(
    hours,
    df["Grid_Dependency_Perc"],
    color="#942115",
    linewidth=LINEWIDTH_TREND_LINES,
    linestyle="--",
    alpha=0.6,
    zorder=15,
)

ax2.plot(
    hours,
    df["Self_Sufficiency_Perc"],
    color="#27AE60",
    linewidth=LINEWIDTH_TREND_LINES,
    linestyle="-.",
    alpha=0.6,
    zorder=15,
)

ax2.plot(
    hours,
    df["Battery_Contribution_Perc"],
    color="#D4AC0D",
    linewidth=LINEWIDTH_TREND_LINES,
    linestyle=":",
    alpha=0.6,
    zorder=15,
)

ax2.fill_between(
    hours,
    df["Sharing_Ratio_Perc"],
    color="#2C527A",
    alpha=0.08,
    zorder=1,
)

ax2.fill_between(
    hours,
    df["Grid_Dependency_Perc"],
    100,
    color="#942115",
    alpha=0.05,
    zorder=1,
)

ax2.fill_between(
    hours,
    df["Self_Sufficiency_Perc"],
    0,
    color="#27AE60",
    alpha=0.05,
    zorder=1,
)

ax2.axhline(
    50,
    color="black",
    linestyle=":",
    linewidth=3,
    alpha=0.6,
    zorder=20,
)

ax2.axhline(
    25,
    color="gray",
    linestyle=":",
    linewidth=2,
    alpha=0.3,
)

ax2.axvspan(
    12,
    16,
    color="green",
    alpha=0.08,
    zorder=0,
)

for ax in [ax1, ax2]:
    ax.tick_params(
        axis="both",
        which="major",
        labelsize=TICK_LABEL_SIZE,
        width=TICK_MAJOR_WIDTH,
        length=TICK_MAJOR_LENGTH,
        direction=TICK_DIRECTION,
        top=True,
        right=True,
    )

    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight(FONT_WEIGHT)

    ax.grid(
        True,
        which="major",
        linestyle="--",
        alpha=GRID_ALPHA_MAJOR,
        linewidth=GRID_LINEWIDTH_MAJOR,
    )

    ax.grid(
        True,
        which="minor",
        linestyle=":",
        alpha=GRID_ALPHA_MINOR,
        linewidth=GRID_LINEWIDTH_MINOR,
    )

    ax.set_xlabel(
        "Time (hours)",
        fontsize=AXIS_LABEL_SIZE,
        fontweight=FONT_WEIGHT,
        labelpad=15,
    )

    for spine in ax.spines.values():
        spine.set_linewidth(SPINE_LINEWIDTH)
        spine.set_color(SPINE_COLOR)

    ax.set_xlim(-0.5, 24.5)
    ax.set_xticks(np.arange(0, 25, 4))
    ax.set_xticks(np.arange(0, 25, 1), minor=True)

ax1.set_ylabel(
    "Profit (USD)",
    fontsize=AXIS_LABEL_SIZE,
    fontweight=FONT_WEIGHT,
    color=SPINE_COLOR,
)

ax2.set_ylabel(
    "Operational Indices (%)",
    fontsize=AXIS_LABEL_SIZE,
    fontweight=FONT_WEIGHT,
    color=SPINE_COLOR,
)

legend1 = ax1.legend(
    loc="upper center",
    bbox_to_anchor=(0.5, -0.18),
    ncol=5,
    fontsize=LEGEND_FONT_SIZE_A,
    frameon=True,
    edgecolor=SPINE_COLOR,
    fancybox=False,
    shadow=True,
    framealpha=0.95,
    borderpad=0.8,
    handlelength=1.5,
    handletextpad=0.8,
)

legend1.get_frame().set_linewidth(2)

legend2 = ax2.legend(
    loc="upper center",
    bbox_to_anchor=(0.5, -0.18),
    ncol=4,
    fontsize=LEGEND_FONT_SIZE_B,
    frameon=True,
    edgecolor=SPINE_COLOR,
    fancybox=False,
    shadow=True,
    framealpha=0.95,
    borderpad=0.8,
    handlelength=1.5,
    handletextpad=0.8,
)

legend2.get_frame().set_linewidth(2)

for ax in [ax1, ax2]:
    ax.plot(
        [ax.get_xlim()[1] - 0.5, ax.get_xlim()[1] - 0.5],
        [ax.get_ylim()[1] - 1, ax.get_ylim()[1]],
        color=SPINE_COLOR,
        linewidth=2,
        alpha=0.3,
    )

    ax.plot(
        [ax.get_xlim()[1] - 1, ax.get_xlim()[1]],
        [ax.get_ylim()[1] - 0.5, ax.get_ylim()[1] - 0.5],
        color=SPINE_COLOR,
        linewidth=2,
        alpha=0.3,
    )

plt.subplots_adjust(bottom=0.08)
plt.tight_layout()

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

svg_path = OUTPUT_DIR / "VPP_Economic_Operational_Performance.svg"
pdf_path = OUTPUT_DIR / "VPP_Economic_Operational_Performance.pdf"
png_path = OUTPUT_DIR / "VPP_Economic_Operational_Performance.png"

plt.savefig(
    svg_path,
    bbox_inches="tight",
    dpi=300,
    format="svg",
)

plt.savefig(
    pdf_path,
    bbox_inches="tight",
    dpi=300,
    format="pdf",
)

plt.savefig(
    png_path,
    bbox_inches="tight",
    dpi=300,
    format="png",
)

plt.show()

print(f"Data loaded from: {DATA_PATH}")
print(f"SVG saved to: {svg_path}")
print(f"PDF saved to: {pdf_path}")
print(f"PNG saved to: {png_path}")
