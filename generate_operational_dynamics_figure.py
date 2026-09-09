from pathlib import Path as FilePath
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.gridspec as gridspec
ROOT = FilePath(__file__).resolve().parent
DATA_PATH = ROOT / "evaluation" / "data" / "operational" / "fc_hmarl_operational_dynamics.xlsx"
OUTPUT_DIR = ROOT / "outputs" / "figures"
FONT_FAMILY = "Times New Roman"
FONT_WEIGHT = "bold"
GLOBAL_FONT_SIZE = 26
TITLE_FONT_SIZE = 26
AXIS_LABEL_FONT_SIZE = 26
TICK_LABEL_FONT_SIZE = 26
LEGEND_FONT_SIZE = 18
ANNOTATION_FONT_SIZE = 16
STATS_BOX_FONT_SIZE = 14
FIG_WIDTH = 18
FIG_HEIGHT = 13
GS_HSPACE = 0.53
GS_WSPACE = 0.38
SPINE_LINEWIDTH = 1.5
TICK_MAJOR_WIDTH = 1.5
TICK_MAJOR_SIZE = 5.0
LINEWIDTH_MAIN = 1.5
LINEWIDTH_SECONDARY = 1.5
LINEWIDTH_THRESHOLD = 1.5
GRID_LINEWIDTH = 1.5

TITLE_PAD = 12
LEGEND_BBOX = (0.5, -0.42)
STATS_BBOX_PAD = 0.4

C_SOC_LINE = "#1A5276"
C_SOC_FILL = "#2E86C1"
C_SOC_HIGH = "#28B463"
C_SOC_LOW = "#E74C3C"
C_POWER = "#E74C3C"

C_GRID_LINE = "#D4A017"
C_GRID_EXPORT = "#FDB813"
C_GRID_IMPORT = "#E74C3C"
C_VOLT_LINE = "#28B463"
C_VOLT_LIMIT = "#FDB813"

C_FREQ_LINE = "#6C3483"
C_FREQ_FILL = "#8E44AD"
C_FREQ_LIMIT = "#E74C3C"
C_REG_LINE = "#FDB813"

C_DELAY_LINE = "#C0392B"
C_DELAY_FILL = "#E74C3C"
C_DELAY_WARN = "#FDB813"
C_DELAY_CRIT = "#E74C3C"
C_EXCHANGE_LINE = "#2E86C1"
C_EXCHANGE_TARGET = "#28B463"

plt.rcParams["font.family"] = FONT_FAMILY
plt.rcParams["font.size"] = GLOBAL_FONT_SIZE
plt.rcParams["font.weight"] = FONT_WEIGHT
plt.rcParams["axes.linewidth"] = SPINE_LINEWIDTH
plt.rcParams["axes.edgecolor"] = "black"
plt.rcParams["xtick.direction"] = "in"
plt.rcParams["ytick.direction"] = "in"
plt.rcParams["xtick.labelsize"] = TICK_LABEL_FONT_SIZE
plt.rcParams["ytick.labelsize"] = TICK_LABEL_FONT_SIZE
plt.rcParams["xtick.major.width"] = TICK_MAJOR_WIDTH
plt.rcParams["ytick.major.width"] = TICK_MAJOR_WIDTH
plt.rcParams["xtick.major.size"] = TICK_MAJOR_SIZE
plt.rcParams["ytick.major.size"] = TICK_MAJOR_SIZE
plt.rcParams["xtick.minor.visible"] = True
plt.rcParams["ytick.minor.visible"] = True

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Data file not found:\n{DATA_PATH}\n\n"
        "Place the Excel workbook at:\n"
        r"evaluation\data\operational\fc_hmarl_operational_dynamics.xlsx"
    )

df_data = pd.read_excel(DATA_PATH, sheet_name="Noisy_Operational_Data")

required_columns = [
    "Time",
    "Battery_SoC",
    "Battery_Power",
    "Grid_Exchange",
    "Voltage",
    "Frequency",
    "Regulation",
    "Coordination_Delay",
    "Exchange_Rate",
]

missing_columns = [c for c in required_columns if c not in df_data.columns]

if missing_columns:
    raise ValueError(f"Missing columns in Noisy_Operational_Data: {missing_columns}")

time = pd.to_numeric(df_data["Time"], errors="raise").to_numpy(dtype=float)
soc = pd.to_numeric(df_data["Battery_SoC"], errors="raise").to_numpy(dtype=float)
power = pd.to_numeric(df_data["Battery_Power"], errors="raise").to_numpy(dtype=float)
grid_exchange = pd.to_numeric(df_data["Grid_Exchange"], errors="raise").to_numpy(dtype=float)
voltage = pd.to_numeric(df_data["Voltage"], errors="raise").to_numpy(dtype=float)
frequency = pd.to_numeric(df_data["Frequency"], errors="raise").to_numpy(dtype=float)
regulation = pd.to_numeric(df_data["Regulation"], errors="raise").to_numpy(dtype=float)
coordination_delay = pd.to_numeric(
    df_data["Coordination_Delay"],
    errors="raise",
).to_numpy(dtype=float)
exchange_rate = pd.to_numeric(
    df_data["Exchange_Rate"],
    errors="raise",
).to_numpy(dtype=float)

lengths = {
    len(time),
    len(soc),
    len(power),
    len(grid_exchange),
    len(voltage),
    len(frequency),
    len(regulation),
    len(coordination_delay),
    len(exchange_rate),
}

if len(lengths) != 1:
    raise ValueError("All operational data series must have the same number of rows.")

fig = plt.figure(figsize=(FIG_WIDTH, FIG_HEIGHT))
gs = gridspec.GridSpec(
    2,
    2,
    figure=fig,
    hspace=GS_HSPACE,
    wspace=GS_WSPACE,
)

ax1 = fig.add_subplot(gs[0, 0])

ax1.fill_between(
    time,
    0.3,
    soc,
    alpha=0.25,
    color=C_SOC_FILL,
    label="SoC Band",
)

ax1.plot(
    time,
    soc,
    color=C_SOC_LINE,
    linewidth=LINEWIDTH_MAIN,
    label="Battery SoC",
)

ax1.axhline(
    y=0.8,
    color=C_SOC_HIGH,
    linestyle="--",
    linewidth=LINEWIDTH_THRESHOLD,
    alpha=0.7,
    label="High SoC (0.8)",
)

ax1.axhline(
    y=0.4,
    color=C_SOC_LOW,
    linestyle="--",
    linewidth=LINEWIDTH_THRESHOLD,
    alpha=0.7,
    label="Low SoC (0.4)",
)

ax1b = ax1.twinx()

ax1b.plot(
    time,
    power,
    color=C_POWER,
    linewidth=LINEWIDTH_SECONDARY,
    alpha=0.7,
    label="Battery Power",
)

ax1b.axhline(
    y=0,
    color="black",
    linewidth=GRID_LINEWIDTH,
    alpha=0.3,
)

ax1b.set_ylabel(
    "Power (p.u.)",
    fontsize=AXIS_LABEL_FONT_SIZE,
    fontweight=FONT_WEIGHT,
    color=C_POWER,
)

ax1b.tick_params(
    axis="y",
    labelcolor=C_POWER,
    labelsize=TICK_LABEL_FONT_SIZE,
)

ax1b.spines["right"].set_linewidth(SPINE_LINEWIDTH)
ax1b.spines["right"].set_color(C_POWER)

ax1.set_xlabel(
    "Time (seconds)",
    fontsize=AXIS_LABEL_FONT_SIZE,
    fontweight=FONT_WEIGHT,
)

ax1.set_ylabel(
    "State of Charge",
    fontsize=AXIS_LABEL_FONT_SIZE,
    fontweight=FONT_WEIGHT,
)

ax1.grid(
    True,
    alpha=0.12,
    linestyle="--",
    linewidth=GRID_LINEWIDTH,
)

ax1.set_title(
    "(a) Battery SoC & Power Fluctuations",
    fontsize=TITLE_FONT_SIZE,
    fontweight=FONT_WEIGHT,
    loc="left",
    pad=TITLE_PAD,
)

ax1.set_ylim([0.25, 1.0])
ax1.set_xlim([0, 300])

spike_indices = np.where(np.abs(power) > 0.4)[0]

if len(spike_indices) > 0:
    first_spike = spike_indices[0]

    if first_spike < len(time) - 1:
        ax1.annotate(
            "Power Spike",
            xy=(time[first_spike], power[first_spike]),
            xytext=(time[first_spike] + 20, power[first_spike] + 0.2),
            arrowprops=dict(
                arrowstyle="->",
                lw=2,
                color="black",
            ),
            fontsize=ANNOTATION_FONT_SIZE,
            fontweight=FONT_WEIGHT,
        )

legend_elements_a = [
    plt.Line2D(
        [0],
        [0],
        color=C_SOC_LINE,
        linewidth=LINEWIDTH_MAIN,
        label="Battery SoC",
    ),
    plt.Line2D(
        [0],
        [0],
        color=C_POWER,
        linewidth=LINEWIDTH_SECONDARY,
        label="Battery Power",
    ),
    plt.Line2D(
        [0],
        [0],
        color=C_SOC_HIGH,
        linestyle="--",
        linewidth=LINEWIDTH_THRESHOLD,
        label="High SoC (0.8)",
    ),
    plt.Line2D(
        [0],
        [0],
        color=C_SOC_LOW,
        linestyle="--",
        linewidth=LINEWIDTH_THRESHOLD,
        label="Low SoC (0.4)",
    ),
]

ax1.legend(
    handles=legend_elements_a,
    loc="lower center",
    bbox_to_anchor=LEGEND_BBOX,
    fontsize=LEGEND_FONT_SIZE,
    frameon=True,
    edgecolor="black",
    fancybox=False,
    ncol=2,
)

ax2 = fig.add_subplot(gs[0, 1])

ax2.fill_between(
    time,
    0,
    grid_exchange,
    alpha=0.2,
    color=C_GRID_EXPORT,
    where=(grid_exchange > 0),
    label="Export",
)

ax2.fill_between(
    time,
    0,
    grid_exchange,
    alpha=0.2,
    color=C_GRID_IMPORT,
    where=(grid_exchange < 0),
    label="Import",
)

ax2.plot(
    time,
    grid_exchange,
    color=C_GRID_LINE,
    linewidth=LINEWIDTH_MAIN,
    label="Grid Exchange",
)

ax2.axhline(
    y=0,
    color="black",
    linewidth=LINEWIDTH_SECONDARY,
    alpha=0.5,
)

ax2b = ax2.twinx()

ax2b.plot(
    time,
    voltage,
    color=C_VOLT_LINE,
    linewidth=LINEWIDTH_SECONDARY,
    alpha=0.8,
    label="Voltage",
)

ax2b.axhspan(
    0.95,
    1.05,
    alpha=0.08,
    color=C_VOLT_LINE,
    label="Normal Range",
)

ax2b.axhline(
    y=0.95,
    color=C_VOLT_LIMIT,
    linestyle="--",
    linewidth=LINEWIDTH_SECONDARY,
    alpha=0.6,
)

ax2b.axhline(
    y=1.05,
    color=C_VOLT_LIMIT,
    linestyle="--",
    linewidth=LINEWIDTH_SECONDARY,
    alpha=0.6,
)

ax2b.set_ylabel(
    "Voltage (p.u.)",
    fontsize=AXIS_LABEL_FONT_SIZE,
    fontweight=FONT_WEIGHT,
    color=C_VOLT_LINE,
)

ax2b.tick_params(
    axis="y",
    labelcolor=C_VOLT_LINE,
    labelsize=TICK_LABEL_FONT_SIZE,
)

ax2b.spines["right"].set_linewidth(SPINE_LINEWIDTH)
ax2b.spines["right"].set_color(C_VOLT_LINE)

ax2.set_xlabel(
    "Time (seconds)",
    fontsize=AXIS_LABEL_FONT_SIZE,
    fontweight=FONT_WEIGHT,
)

ax2.set_ylabel(
    "Grid Exchange (p.u.)",
    fontsize=AXIS_LABEL_FONT_SIZE,
    fontweight=FONT_WEIGHT,
)

ax2.grid(
    True,
    alpha=0.12,
    linestyle="--",
    linewidth=GRID_LINEWIDTH,
)

ax2.set_title(
    "(b) Grid Exchange & Voltage Variations",
    fontsize=TITLE_FONT_SIZE,
    fontweight=FONT_WEIGHT,
    loc="left",
    pad=TITLE_PAD,
)

ax2.set_xlim([0, 300])

dip_indices = np.where(voltage < 0.93)[0]

if len(dip_indices) > 0:
    first_dip = dip_indices[0]

    if first_dip < len(time) - 1:
        ax2.annotate(
            "Voltage Dip",
            xy=(time[first_dip], voltage[first_dip]),
            xytext=(time[first_dip] + 20, voltage[first_dip] - 0.05),
            arrowprops=dict(
                arrowstyle="->",
                lw=2,
                color="black",
            ),
            fontsize=ANNOTATION_FONT_SIZE,
            fontweight=FONT_WEIGHT,
        )

export_indices = np.where(grid_exchange > 0.3)[0]

if len(export_indices) > 0:
    first_export = export_indices[0]

    if first_export < len(time) - 1:
        ax2.annotate(
            "Grid Export",
            xy=(time[first_export], grid_exchange[first_export]),
            xytext=(time[first_export] + 20, grid_exchange[first_export] + 0.2),
            arrowprops=dict(
                arrowstyle="->",
                lw=2,
                color="black",
            ),
            fontsize=ANNOTATION_FONT_SIZE,
            fontweight=FONT_WEIGHT,
        )

legend_elements_b = [
    plt.Line2D(
        [0],
        [0],
        color=C_GRID_LINE,
        linewidth=LINEWIDTH_MAIN,
        label="Grid Exchange",
    ),
    plt.Line2D(
        [0],
        [0],
        color=C_VOLT_LINE,
        linewidth=LINEWIDTH_SECONDARY,
        label="Voltage",
    ),
    plt.Line2D(
        [0],
        [0],
        color=C_VOLT_LIMIT,
        linestyle="--",
        linewidth=LINEWIDTH_SECONDARY,
        label="Voltage Limits",
    ),
]

ax2.legend(
    handles=legend_elements_b,
    loc="lower center",
    bbox_to_anchor=LEGEND_BBOX,
    fontsize=LEGEND_FONT_SIZE,
    frameon=True,
    edgecolor="black",
    fancybox=False,
    ncol=2,
)

ax3 = fig.add_subplot(gs[1, 0])

ax3.fill_between(
    time,
    frequency - 0.015,
    frequency + 0.015,
    alpha=0.2,
    color=C_FREQ_FILL,
    label="Frequency Band",
)

ax3.plot(
    time,
    frequency,
    color=C_FREQ_LINE,
    linewidth=LINEWIDTH_MAIN,
    label="Frequency Deviation",
)

ax3.axhline(
    y=0,
    color="black",
    linewidth=LINEWIDTH_SECONDARY,
    alpha=0.5,
)

ax3.axhline(
    y=0.05,
    color=C_FREQ_LIMIT,
    linestyle="--",
    linewidth=LINEWIDTH_SECONDARY,
    alpha=0.6,
    label="+0.05 Hz",
)

ax3.axhline(
    y=-0.05,
    color=C_FREQ_LIMIT,
    linestyle="--",
    linewidth=LINEWIDTH_SECONDARY,
    alpha=0.6,
    label="-0.05 Hz",
)

ax3b = ax3.twinx()

ax3b.plot(
    time,
    regulation,
    color=C_REG_LINE,
    linewidth=LINEWIDTH_SECONDARY,
    alpha=0.8,
    label="Regulation Signal",
)

ax3b.axhline(
    y=0,
    color="black",
    linewidth=GRID_LINEWIDTH,
    alpha=0.3,
)

ax3b.set_ylabel(
    "Regulation (p.u.)",
    fontsize=AXIS_LABEL_FONT_SIZE,
    fontweight=FONT_WEIGHT,
    color=C_REG_LINE,
)

ax3b.tick_params(
    axis="y",
    labelcolor=C_REG_LINE,
    labelsize=TICK_LABEL_FONT_SIZE,
)

ax3b.spines["right"].set_linewidth(SPINE_LINEWIDTH)
ax3b.spines["right"].set_color(C_REG_LINE)

ax3.set_xlabel(
    "Time (seconds)",
    fontsize=AXIS_LABEL_FONT_SIZE,
    fontweight=FONT_WEIGHT,
)

ax3.set_ylabel(
    "Frequency Deviation (Hz)",
    fontsize=AXIS_LABEL_FONT_SIZE,
    fontweight=FONT_WEIGHT,
)

ax3.grid(
    True,
    alpha=0.12,
    linestyle="--",
    linewidth=GRID_LINEWIDTH,
)

ax3.set_title(
    "(c) Frequency Response & Regulation Signals",
    fontsize=TITLE_FONT_SIZE,
    fontweight=FONT_WEIGHT,
    loc="left",
    pad=TITLE_PAD,
)

ax3.set_xlim([0, 300])

legend_elements_c = [
    plt.Line2D(
        [0],
        [0],
        color=C_FREQ_LINE,
        linewidth=LINEWIDTH_MAIN,
        label="Frequency Deviation",
    ),
    plt.Line2D(
        [0],
        [0],
        color=C_REG_LINE,
        linewidth=LINEWIDTH_SECONDARY,
        label="Regulation Signal",
    ),
    plt.Line2D(
        [0],
        [0],
        color=C_FREQ_LIMIT,
        linestyle="--",
        linewidth=LINEWIDTH_SECONDARY,
        label="±0.05 Hz Limits",
    ),
]

ax3.legend(
    handles=legend_elements_c,
    loc="lower center",
    bbox_to_anchor=LEGEND_BBOX,
    fontsize=LEGEND_FONT_SIZE,
    frameon=True,
    edgecolor="black",
    fancybox=False,
    ncol=2,
)

ax4 = fig.add_subplot(gs[1, 1])

ax4.fill_between(
    time,
    coordination_delay - 5,
    coordination_delay + 5,
    alpha=0.15,
    color=C_DELAY_FILL,
    label="Delay Band",
)

ax4.plot(
    time,
    coordination_delay,
    color=C_DELAY_LINE,
    linewidth=LINEWIDTH_MAIN,
    label="Coordination Delay (ms)",
)

ax4.axhline(
    y=30,
    color=C_DELAY_WARN,
    linestyle="--",
    linewidth=LINEWIDTH_THRESHOLD,
    alpha=0.7,
    label="Warning Delay (30ms)",
)

ax4.axhline(
    y=50,
    color=C_DELAY_CRIT,
    linestyle="--",
    linewidth=LINEWIDTH_THRESHOLD,
    alpha=0.7,
    label="Critical Delay (50ms)",
)

ax4b = ax4.twinx()

ax4b.plot(
    time,
    exchange_rate,
    color=C_EXCHANGE_LINE,
    linewidth=LINEWIDTH_SECONDARY,
    alpha=0.8,
    label="Exchange Rate (pkts/s)",
)

ax4b.axhline(
    y=80,
    color=C_EXCHANGE_TARGET,
    linestyle=":",
    linewidth=LINEWIDTH_SECONDARY,
    alpha=0.6,
    label="Target Rate (80 pkts/s)",
)

ax4b.set_ylabel(
    "Exchange Rate (pkts/s)",
    fontsize=AXIS_LABEL_FONT_SIZE,
    fontweight=FONT_WEIGHT,
    color=C_EXCHANGE_LINE,
)

ax4b.tick_params(
    axis="y",
    labelcolor=C_EXCHANGE_LINE,
    labelsize=TICK_LABEL_FONT_SIZE,
)

ax4b.spines["right"].set_linewidth(SPINE_LINEWIDTH)
ax4b.spines["right"].set_color(C_EXCHANGE_LINE)

ax4.set_xlabel(
    "Time (seconds)",
    fontsize=AXIS_LABEL_FONT_SIZE,
    fontweight=FONT_WEIGHT,
)

ax4.set_ylabel(
    "Coordination Delay (ms)",
    fontsize=AXIS_LABEL_FONT_SIZE,
    fontweight=FONT_WEIGHT,
)

ax4.grid(
    True,
    alpha=0.12,
    linestyle="--",
    linewidth=GRID_LINEWIDTH,
)

ax4.set_title(
    "(d) Coordination Delay & Exchange Rate",
    fontsize=TITLE_FONT_SIZE,
    fontweight=FONT_WEIGHT,
    loc="left",
    pad=TITLE_PAD,
)

ax4.set_xlim([0, 300])

delay_spike_indices = np.where(coordination_delay > 40)[0]

if len(delay_spike_indices) > 0:
    first_spike = delay_spike_indices[0]

    if first_spike < len(time) - 1:
        ax4.annotate(
            "Delay Spike\n(Agent Sync)",
            xy=(time[first_spike], coordination_delay[first_spike]),
            xytext=(
                time[first_spike] + 20,
                coordination_delay[first_spike] + 10,
            ),
            arrowprops=dict(
                arrowstyle="->",
                lw=2,
                color="black",
            ),
            fontsize=ANNOTATION_FONT_SIZE,
            fontweight=FONT_WEIGHT,
            ha="center",
        )

dropout_indices = np.where(exchange_rate < 10)[0]

if len(dropout_indices) > 0:
    first_dropout = dropout_indices[0]

    if first_dropout < len(time) - 1:
        ax4.annotate(
            "Exchange\nDropout",
            xy=(time[first_dropout], exchange_rate[first_dropout]),
            xytext=(
                time[first_dropout] + 20,
                exchange_rate[first_dropout] + 30,
            ),
            arrowprops=dict(
                arrowstyle="->",
                lw=2,
                color="black",
            ),
            fontsize=ANNOTATION_FONT_SIZE,
            fontweight=FONT_WEIGHT,
            ha="center",
        )

legend_elements_d = [
    plt.Line2D(
        [0],
        [0],
        color=C_DELAY_LINE,
        linewidth=LINEWIDTH_MAIN,
        label="Coordination Delay",
    ),
    plt.Line2D(
        [0],
        [0],
        color=C_EXCHANGE_LINE,
        linewidth=LINEWIDTH_SECONDARY,
        label="Exchange Rate",
    ),
    plt.Line2D(
        [0],
        [0],
        color=C_DELAY_WARN,
        linestyle="--",
        linewidth=LINEWIDTH_THRESHOLD,
        label="Warning Delay (30ms)",
    ),
    plt.Line2D(
        [0],
        [0],
        color=C_DELAY_CRIT,
        linestyle="--",
        linewidth=LINEWIDTH_THRESHOLD,
        label="Critical Delay (50ms)",
    ),
]

ax4.legend(
    handles=legend_elements_d,
    loc="lower center",
    bbox_to_anchor=LEGEND_BBOX,
    fontsize=LEGEND_FONT_SIZE,
    frameon=True,
    edgecolor="black",
    fancybox=False,
    ncol=2,
)

stats_text_a = (
    f"SoC Mean: {np.mean(soc):.2f}\n"
    f"Power RMS: {np.sqrt(np.mean(power ** 2)):.2f}"
)

ax1.text(
    0.02,
    0.02,
    stats_text_a,
    transform=ax1.transAxes,
    fontsize=STATS_BOX_FONT_SIZE,
    fontweight=FONT_WEIGHT,
    bbox=dict(
        boxstyle=f"round,pad={STATS_BBOX_PAD}",
        facecolor="white",
        edgecolor="black",
        linewidth=2,
    ),
)

stats_text_b = (
    f"Grid Mean: {np.mean(grid_exchange):.2f}\n"
    f"Voltage Var: {np.var(voltage):.4f}"
)

ax2.text(
    0.02,
    0.02,
    stats_text_b,
    transform=ax2.transAxes,
    fontsize=STATS_BOX_FONT_SIZE,
    fontweight=FONT_WEIGHT,
    bbox=dict(
        boxstyle=f"round,pad={STATS_BBOX_PAD}",
        facecolor="white",
        edgecolor="black",
        linewidth=2,
    ),
)

stats_text_c = (
    f"Freq RMS: {np.sqrt(np.mean(frequency ** 2)):.3f}\n"
    f"Reg RMS: {np.sqrt(np.mean(regulation ** 2)):.3f}"
)

ax3.text(
    0.02,
    0.02,
    stats_text_c,
    transform=ax3.transAxes,
    fontsize=STATS_BOX_FONT_SIZE,
    fontweight=FONT_WEIGHT,
    bbox=dict(
        boxstyle=f"round,pad={STATS_BBOX_PAD}",
        facecolor="white",
        edgecolor="black",
        linewidth=2,
    ),
)

stats_text_d = (
    f"Delay Mean: {np.mean(coordination_delay):.1f}ms\n"
    f"Rate Mean: {np.mean(exchange_rate):.1f}pkts/s"
)

ax4.text(
    0.02,
    0.02,
    stats_text_d,
    transform=ax4.transAxes,
    fontsize=STATS_BOX_FONT_SIZE,
    fontweight=FONT_WEIGHT,
    bbox=dict(
        boxstyle=f"round,pad={STATS_BBOX_PAD}",
        facecolor="white",
        edgecolor="black",
        linewidth=2,
    ),
)

for ax in [ax1, ax2, ax3, ax4]:
    for spine in ax.spines.values():
        spine.set_linewidth(SPINE_LINEWIDTH)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

svg_path = OUTPUT_DIR / "Figure7_Operational_Dynamics.svg"
pdf_path = OUTPUT_DIR / "Figure7_Operational_Dynamics.pdf"
png_path = OUTPUT_DIR / "Figure7_Operational_Dynamics.png"

plt.tight_layout()

plt.savefig(
    svg_path,
    dpi=300,
    bbox_inches="tight",
    facecolor="white",
    edgecolor="none",
    format="svg",
)

plt.savefig(
    pdf_path,
    dpi=300,
    bbox_inches="tight",
    facecolor="white",
    edgecolor="none",
    format="pdf",
)

plt.savefig(
    png_path,
    dpi=300,
    bbox_inches="tight",
    facecolor="white",
    edgecolor="none",
    format="png",
)

plt.show()

print(f"Data loaded from: {DATA_PATH}")
print(f"SVG saved to: {svg_path}")
print(f"PDF saved to: {pdf_path}")
print(f"PNG saved to: {png_path}")
