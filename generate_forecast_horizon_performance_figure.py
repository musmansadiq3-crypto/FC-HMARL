from pathlib import Path as FilePath

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches

ROOT = FilePath(__file__).resolve().parent
DATA_PATH = ROOT / "evaluation" / "data" / "forecast_horizon" / "forecast_horizon_metrics.xlsx"
OUTPUT_DIR = ROOT / "outputs" / "figures"

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.linewidth"] = 3

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Data file not found:\n{DATA_PATH}\n\n"
        "Place the Excel workbook at:\n"
        r"evaluation\data\forecast_horizon\forecast_horizon_metrics.xlsx"
    )

forecast_df = pd.read_excel(DATA_PATH, sheet_name="Forecast_Magnitudes")
error_df = pd.read_excel(DATA_PATH, sheet_name="Forecast_Errors")

required_forecast_columns = [
    "Hour",
    "PV_Generation",
    "Load_Demand",
    "EV_Charging",
    "Electricity_Price",
]

required_error_columns = [
    "PV_RMSE",
    "PV_MAE",
    "Load_RMSE",
    "Load_MAE",
    "EV_RMSE",
    "EV_MAE",
    "Price_RMSE",
    "Price_MAE",
]

missing_forecast = [c for c in required_forecast_columns if c not in forecast_df.columns]
missing_error = [c for c in required_error_columns if c not in error_df.columns]

if missing_forecast:
    raise ValueError(f"Missing columns in Forecast_Magnitudes: {missing_forecast}")

if missing_error:
    raise ValueError(f"Missing columns in Forecast_Errors: {missing_error}")

hours = pd.to_numeric(forecast_df["Hour"], errors="raise").to_numpy(dtype=float)

PV_forecast = pd.to_numeric(forecast_df["PV_Generation"], errors="raise").to_numpy(dtype=float)
Load_forecast = pd.to_numeric(forecast_df["Load_Demand"], errors="raise").to_numpy(dtype=float)
EV_forecast = pd.to_numeric(forecast_df["EV_Charging"], errors="raise").to_numpy(dtype=float)
Price_forecast = pd.to_numeric(forecast_df["Electricity_Price"], errors="raise").to_numpy(dtype=float)

PV_RMSE = pd.to_numeric(error_df["PV_RMSE"], errors="raise").to_numpy(dtype=float)
PV_MAE = pd.to_numeric(error_df["PV_MAE"], errors="raise").to_numpy(dtype=float)
Load_RMSE = pd.to_numeric(error_df["Load_RMSE"], errors="raise").to_numpy(dtype=float)
Load_MAE = pd.to_numeric(error_df["Load_MAE"], errors="raise").to_numpy(dtype=float)
EV_RMSE = pd.to_numeric(error_df["EV_RMSE"], errors="raise").to_numpy(dtype=float)
EV_MAE = pd.to_numeric(error_df["EV_MAE"], errors="raise").to_numpy(dtype=float)
Price_RMSE = pd.to_numeric(error_df["Price_RMSE"], errors="raise").to_numpy(dtype=float)
Price_MAE = pd.to_numeric(error_df["Price_MAE"], errors="raise").to_numpy(dtype=float)

series_lengths = {
    len(hours),
    len(PV_forecast),
    len(Load_forecast),
    len(EV_forecast),
    len(Price_forecast),
    len(PV_RMSE),
    len(PV_MAE),
    len(Load_RMSE),
    len(Load_MAE),
    len(EV_RMSE),
    len(EV_MAE),
    len(Price_RMSE),
    len(Price_MAE),
}

if len(series_lengths) != 1:
    raise ValueError("Forecast magnitude and error series must have the same number of rows.")

fig, ax = plt.subplots(figsize=(20, 12), dpi=600)

width = 0.18
offset_1 = -width * 1.5
offset_2 = -width * 0.5
offset_3 = width * 0.5
offset_4 = width * 1.5

ax.bar(
    hours + offset_1,
    PV_forecast,
    width,
    color="#F4D03F",
    edgecolor="black",
    linewidth=2,
    zorder=3,
)

ax.bar(
    hours + offset_2,
    Load_forecast,
    width,
    color="#2E86C1",
    edgecolor="black",
    linewidth=2,
    zorder=3,
)

ax.bar(
    hours + offset_3,
    EV_forecast,
    width,
    color="#28B463",
    edgecolor="black",
    linewidth=2,
    zorder=3,
)

ax.bar(
    hours + offset_4,
    Price_forecast,
    width,
    color="#E74C3C",
    edgecolor="black",
    linewidth=2,
    zorder=3,
)

ax.bar(
    hours + offset_1,
    -PV_RMSE,
    width,
    color="#D4AC0D",
    edgecolor="black",
    hatch="//",
    linewidth=1.5,
    zorder=3,
)

ax.bar(
    hours + offset_2,
    -Load_RMSE,
    width,
    color="#1B4F72",
    edgecolor="black",
    hatch="//",
    linewidth=1.5,
    zorder=3,
)

ax.bar(
    hours + offset_3,
    -EV_RMSE,
    width,
    color="#1E8449",
    edgecolor="black",
    hatch="//",
    linewidth=1.5,
    zorder=3,
)

ax.bar(
    hours + offset_4,
    -Price_RMSE,
    width,
    color="#B03A2E",
    edgecolor="black",
    hatch="//",
    linewidth=1.5,
    zorder=3,
)

def plot_mae_dots(x, y, color):
    ax.plot(
        x,
        y,
        "--",
        color=color,
        linewidth=2.5,
        zorder=4,
    )

    ax.errorbar(
        x,
        y,
        yerr=0.12,
        fmt="none",
        ecolor="black",
        capsize=4,
        capthick=2,
        elinewidth=2.5,
        zorder=5,
    )

    ax.scatter(
        x,
        y,
        s=70,
        facecolors="white",
        edgecolors=color,
        linewidth=3,
        zorder=6,
        marker="o",
    )

plot_mae_dots(hours, -PV_MAE, "#D4AC0D")
plot_mae_dots(hours, -Load_MAE, "#1B4F72")
plot_mae_dots(hours, -EV_MAE, "#1E8449")
plot_mae_dots(hours, -Price_MAE, "#B03A2E")

ax.set_xlim(0.3, 24.7)
ax.set_ylim(-1.2, 1.25)

ax.axhline(
    0,
    color="black",
    linewidth=4,
    zorder=2,
)

ax.set_xlabel(
    "Prediction Horizon (h)",
    fontsize=34,
    fontweight="bold",
    labelpad=12,
)

ax.set_xticks(hours)

ax.tick_params(
    axis="x",
    labelsize=26,
    length=10,
    width=3,
)

ax.tick_params(
    axis="y",
    labelsize=26,
    length=10,
    width=3,
)

for spine in ax.spines.values():
    spine.set_visible(True)
    spine.set_linewidth(2)
    spine.set_color("black")

ax.text(
    -1.5,
    0.6,
    "Forecasted Resource Magnitude\n(Normalized)",
    rotation=90,
    va="center",
    ha="center",
    fontsize=28,
    fontweight="bold",
    color="#154360",
)

ax.text(
    -1.5,
    -0.6,
    "Forecasting Error Metric\n(Normalized)",
    rotation=90,
    va="center",
    ha="center",
    fontsize=28,
    fontweight="bold",
    color="#922B21",
)

legend_elements_upper = [
    patches.Patch(
        facecolor="#F4D03F",
        edgecolor="black",
        label="PV Generation (Forecast)",
    ),
    patches.Patch(
        facecolor="#2E86C1",
        edgecolor="black",
        label="Load Demand (Forecast)",
    ),
    patches.Patch(
        facecolor="#28B463",
        edgecolor="black",
        label="EV Charging Demand (Forecast)",
    ),
    patches.Patch(
        facecolor="#E74C3C",
        edgecolor="black",
        label="Electricity Price Signal (Forecast)",
    ),
]

leg_upper = ax.legend(
    handles=legend_elements_upper,
    loc="upper left",
    fontsize=19,
    framealpha=1,
    edgecolor="black",
    bbox_to_anchor=(0, 0.85),
)

leg_upper.get_frame().set_linewidth(2)
ax.add_artist(leg_upper)

legend_elements_lower = [
    patches.Patch(
        facecolor="#D4AC0D",
        edgecolor="black",
        hatch="//",
        label="PV Generation RMSE",
    ),
    plt.Line2D(
        [0],
        [0],
        marker="o",
        color="w",
        label="PV Generation MAE",
        markeredgecolor="#D4AC0D",
        markerfacecolor="white",
        markersize=14,
        markeredgewidth=3,
    ),
    patches.Patch(
        facecolor="#1B4F72",
        edgecolor="black",
        hatch="//",
        label="Load Demand RMSE",
    ),
    plt.Line2D(
        [0],
        [0],
        marker="o",
        color="w",
        label="Load Demand MAE",
        markeredgecolor="#1B4F72",
        markerfacecolor="white",
        markersize=14,
        markeredgewidth=3,
    ),
    patches.Patch(
        facecolor="#1E8449",
        edgecolor="black",
        hatch="//",
        label="EV Charging RMSE",
    ),
    plt.Line2D(
        [0],
        [0],
        marker="o",
        color="w",
        label="EV Charging MAE",
        markeredgecolor="#1E8449",
        markerfacecolor="white",
        markersize=14,
        markeredgewidth=3,
    ),
    patches.Patch(
        facecolor="#B03A2E",
        edgecolor="black",
        hatch="//",
        label="Market Price RMSE",
    ),
    plt.Line2D(
        [0],
        [0],
        marker="o",
        color="w",
        label="Market Price MAE",
        markeredgecolor="#B03A2E",
        markerfacecolor="white",
        markersize=14,
        markeredgewidth=3,
    ),
]

leg_lower = ax.legend(
    handles=legend_elements_lower,
    loc="lower left",
    fontsize=22,
    framealpha=1,
    edgecolor="black",
    bbox_to_anchor=(0, 0.01),
)

leg_lower.get_frame().set_linewidth(2)

plt.subplots_adjust(right=0.88)

ax.annotate(
    "",
    xy=(1.01, 0.95),
    xytext=(1.01, 0.53),
    xycoords="axes fraction",
    textcoords="axes fraction",
    arrowprops=dict(
        arrowstyle="->",
        color="#232ca7",
        lw=5,
        shrinkA=0,
        shrinkB=0,
    ),
    clip_on=False,
)

ax.text(
    1.03,
    0.74,
    "Higher Resource\n(Preferred)",
    rotation=90,
    va="center",
    ha="center",
    fontsize=24,
    color="#232ca7",
    fontweight="bold",
    transform=ax.transAxes,
    clip_on=False,
)

ax.annotate(
    "",
    xy=(1.01, 0.05),
    xytext=(1.01, 0.47),
    xycoords="axes fraction",
    textcoords="axes fraction",
    arrowprops=dict(
        arrowstyle="->",
        color="#b01616",
        lw=5,
        shrinkA=0,
        shrinkB=0,
    ),
    clip_on=False,
)

ax.text(
    1.03,
    0.26,
    "Lower Error\n(Preferred)",
    rotation=90,
    va="center",
    ha="center",
    fontsize=24,
    color="#b01616",
    fontweight="bold",
    transform=ax.transAxes,
    clip_on=False,
)

ax.axvline(
    x=7.5,
    color="black",
    linestyle="--",
    linewidth=2.5,
    zorder=1,
)

ax.axvline(
    x=16.5,
    color="black",
    linestyle="--",
    linewidth=2.5,
    zorder=1,
)

def add_zone_arrow(start, end, y_level, color, text):
    ax.annotate(
        "",
        xy=(end, y_level),
        xytext=(start, y_level),
        arrowprops=dict(
            arrowstyle="->",
            color=color,
            linestyle="--",
            linewidth=3,
        ),
    )

    ax.text(
        (start + end) / 2,
        y_level + 0.05,
        text,
        ha="center",
        va="bottom",
        fontsize=22,
        bbox=dict(
            facecolor="white",
            edgecolor="black",
            pad=5,
        ),
    )

add_zone_arrow(
    1,
    7,
    0.95,
    "#2E86C1",
    "Morning\n(Valley)",
)

add_zone_arrow(
    8,
    16,
    0.95,
    "#F4D03F",
    "Midday\n(Peak PV)",
)

add_zone_arrow(
    17,
    24,
    0.95,
    "#28B463",
    "Evening\n(High Load & Price)",
)

plt.title(
    "Forecasting Error and Forecasted Resource Magnitude Across Prediction Horizons",
    fontsize=32,
    fontweight="bold",
    pad=24,
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

svg_path = OUTPUT_DIR / "FC_HMARL_Forecasting_Performance_FINAL.svg"
pdf_path = OUTPUT_DIR / "FC_HMARL_Forecasting_Performance_FINAL.pdf"
png_path = OUTPUT_DIR / "FC_HMARL_Forecasting_Performance_FINAL.png"

plt.tight_layout()

plt.savefig(
    svg_path,
    dpi=600,
    bbox_inches="tight",
    pad_inches=0.5,
    format="svg",
)

plt.savefig(
    pdf_path,
    dpi=600,
    bbox_inches="tight",
    pad_inches=0.5,
    format="pdf",
)

plt.savefig(
    png_path,
    dpi=600,
    bbox_inches="tight",
    pad_inches=0.5,
    format="png",
)

plt.show()

print(f"Data loaded from: {DATA_PATH}")
print(f"SVG saved to: {svg_path}")
print(f"PDF saved to: {pdf_path}")
print(f"PNG saved to: {png_path}")
