from pathlib import Path as FilePath

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch

ROOT = FilePath(__file__).resolve().parent
DATA_PATH = ROOT / "forecasting" / "confidence_data" / "forecast_confidence_evolution_data.xlsx"
OUTPUT_DIR = ROOT / "outputs" / "figures"

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 22
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.linewidth"] = 2.5
plt.rcParams["axes.edgecolor"] = "black"
plt.rcParams["xtick.direction"] = "in"
plt.rcParams["ytick.direction"] = "in"
plt.rcParams["xtick.major.width"] = 2.5
plt.rcParams["ytick.major.width"] = 2.5
plt.rcParams["xtick.minor.visible"] = True
plt.rcParams["ytick.minor.visible"] = True

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Data file not found:\n{DATA_PATH}\n\n"
        "Place the Excel workbook at:\n"
        r"forecasting\confidence_data\forecast_confidence_evolution_data.xlsx"
    )

df_errors = pd.read_excel(DATA_PATH, sheet_name="Error_Metrics")
df_confidence = pd.read_excel(DATA_PATH, sheet_name="Confidence_Coefficient")

var_names = ["PV Generation", "Load Demand", "EV Charging", "Market Price"]

required_error_columns = ["Horizon"]
for var in var_names:
    required_error_columns.extend([f"RMSE_{var}", f"MAE_{var}"])

required_confidence_columns = [
    "Time_Days",
    "Confidence_Coefficient",
    "Upper_Band",
    "Lower_Band",
]

missing_error = [c for c in required_error_columns if c not in df_errors.columns]
missing_confidence = [c for c in required_confidence_columns if c not in df_confidence.columns]

if missing_error:
    raise ValueError(f"Missing columns in Error_Metrics: {missing_error}")

if missing_confidence:
    raise ValueError(f"Missing columns in Confidence_Coefficient: {missing_confidence}")

horizons = pd.to_numeric(df_errors["Horizon"], errors="raise").to_numpy(dtype=float)
H = len(horizons)

rmse_data = {
    var: pd.to_numeric(df_errors[f"RMSE_{var}"], errors="raise").to_numpy(dtype=float)
    for var in var_names
}

mae_data = {
    var: pd.to_numeric(df_errors[f"MAE_{var}"], errors="raise").to_numpy(dtype=float)
    for var in var_names
}

days = pd.to_numeric(df_confidence["Time_Days"], errors="raise").to_numpy(dtype=float)
confidence = pd.to_numeric(
    df_confidence["Confidence_Coefficient"],
    errors="raise",
).to_numpy(dtype=float)
upper_band = pd.to_numeric(
    df_confidence["Upper_Band"],
    errors="raise",
).to_numpy(dtype=float)
lower_band = pd.to_numeric(
    df_confidence["Lower_Band"],
    errors="raise",
).to_numpy(dtype=float)

resource_colors = ["#FDB813", "#2E86C1", "#28B463", "#E74C3C"]

resource_data = {}

for var in var_names:
    values = rmse_data[var]
    value_range = np.max(values) - np.min(values)

    if np.isclose(value_range, 0.0):
        resource_data[var] = np.full_like(values, 0.35, dtype=float)
    else:
        resource_data[var] = (
            0.35
            + 0.65
            * (values - np.min(values))
            / value_range
        )

fig = plt.figure(figsize=(10, 10))
gs = gridspec.GridSpec(
    2,
    1,
    figure=fig,
    hspace=0.35,
    height_ratios=[1, 1],
)

ax1 = fig.add_subplot(gs[0])

x = np.arange(H)
width = 0.20

for i, var in enumerate(var_names):
    pos = x + (i - 1.5) * width

    ax1.bar(
        pos,
        resource_data[var],
        width=width * 0.9,
        color=resource_colors[i],
        edgecolor="black",
        linewidth=0.5,
        alpha=0.95,
        label=f"{var} Forecast",
        zorder=3,
    )

for i, var in enumerate(var_names):
    pos = x + (i - 1.5) * width

    ax1.bar(
        pos,
        -rmse_data[var],
        width=width * 0.75,
        color=resource_colors[i],
        alpha=0.55,
        edgecolor="black",
        linewidth=1.2,
        hatch="//",
        label=f"{var} RMSE" if i == 0 else None,
        zorder=3,
    )

    ax1.plot(
        pos,
        -mae_data[var],
        marker="o",
        markersize=6,
        linewidth=2,
        linestyle="--",
        color=resource_colors[i],
        markeredgecolor="black",
        label=f"{var} MAE" if i == 0 else None,
        zorder=5,
    )

ax1.axhline(
    0,
    color="black",
    linewidth=1.5,
)

ax1.set_xlabel(
    "Prediction Horizon (h)",
    fontsize=22,
    fontweight="bold",
)

ax1.set_ylabel(
    "Normalized Forecast Value",
    fontsize=22,
    fontweight="bold",
)

tick_idx = np.arange(0, H, 3)

ax1.set_xticks(tick_idx)
ax1.set_xticklabels(
    [f"{h:g}" for h in horizons[tick_idx]]
)

ax1.set_ylim(
    -1.0,
    1.2,
)

ax1.axhspan(
    0,
    1.2,
    color="green",
    alpha=0.03,
)

ax1.axhspan(
    -1,
    0,
    color="red",
    alpha=0.03,
)

ax1.set_title(
    "(a) Forecast Accuracy and Uncertainty Evolution",
    fontsize=22,
    fontweight="bold",
    loc="left",
    pad=15,
)

legend_elements = [
    Patch(
        facecolor="#FDB813",
        edgecolor="black",
        label="PV Generation Forecast",
    ),
    Patch(
        facecolor="#2E86C1",
        edgecolor="black",
        label="Load Demand Forecast",
    ),
    Patch(
        facecolor="#28B463",
        edgecolor="black",
        label="EV Charging Forecast",
    ),
    Patch(
        facecolor="#E74C3C",
        edgecolor="black",
        label="Market Price Forecast",
    ),
    Patch(
        facecolor="gray",
        edgecolor="black",
        hatch="//",
        label="RMSE Error",
    ),
]

ax1.legend(
    handles=legend_elements,
    loc="upper left",
    fontsize=10,
    ncol=2,
    frameon=True,
    edgecolor="black",
)

ax1.grid(
    True,
    linestyle="--",
    alpha=0.15,
)

ax2 = fig.add_subplot(gs[1])

ax2.fill_between(
    days,
    lower_band,
    upper_band,
    alpha=0.25,
    color="#1A5276",
    zorder=1,
)

ax2.axhspan(
    0.85,
    1.0,
    alpha=0.1,
    color="#28B463",
)

ax2.axhspan(
    0.35,
    0.65,
    alpha=0.1,
    color="#E74C3C",
)

ax2.plot(
    days + 0.05,
    confidence - 0.005,
    color="black",
    linewidth=4,
    alpha=0.15,
    zorder=2,
)

ax2.plot(
    days,
    confidence,
    color="#1A5276",
    linewidth=4,
    label="Confidence Coefficient Φ(t)",
    zorder=3,
)

ax2.axhline(
    y=0.85,
    color="#28B463",
    linestyle="--",
    linewidth=3,
    alpha=0.8,
    zorder=4,
)

ax2.axhline(
    y=0.65,
    color="#E74C3C",
    linestyle="--",
    linewidth=3,
    alpha=0.8,
    zorder=4,
)

scatter_indices = np.arange(0, len(days), 8)

ax2.scatter(
    days[scatter_indices] + 0.1,
    confidence[scatter_indices] - 0.01,
    s=130,
    color="black",
    alpha=0.2,
    edgecolor="none",
    zorder=5,
)

ax2.scatter(
    days[scatter_indices],
    confidence[scatter_indices],
    s=120,
    color="#E74C3C",
    alpha=1,
    edgecolor="black",
    linewidth=2.5,
    zorder=6,
    label="Measured Confidence Samples",
)

ax2.axvspan(
    0,
    5,
    alpha=0.15,
    color="gray",
    hatch="///",
    label="Training Period",
)

ax2.axvspan(
    20,
    25,
    alpha=0.15,
    color="gray",
    hatch="\\\\\\",
    label="Evaluation Period",
)

ax2.annotate(
    "Model Update",
    xy=(5, 0.72),
    xytext=(7, 0.52),
    arrowprops=dict(
        arrowstyle="-|>",
        lw=1.5,
        color="black",
        connectionstyle="arc3,rad=.2",
    ),
    fontsize=16,
    fontweight="bold",
)

ax2.set_xlabel(
    "Time (days)",
    fontweight="bold",
)

ax2.set_ylabel(
    "Confidence Coefficient Φ(t)",
    fontweight="bold",
)

ax2.set_title(
    "(b) Forecast Confidence Evolution Over Time",
    fontsize=22,
    fontweight="bold",
    loc="left",
    pad=15,
)

ax2.set_ylim([0.35, 1.05])
ax2.set_xlim([0, 30])

ax2.legend(
    loc="lower right",
    fontsize=11,
    frameon=True,
    edgecolor="black",
    ncol=2,
    framealpha=1,
    handlelength=2.5,
    handletextpad=1.2,
)

ax2.grid(
    True,
    alpha=0.15,
    linestyle="--",
    zorder=0,
)

for ax in [ax1, ax2]:
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

svg_path = OUTPUT_DIR / "Figure1_Expanded_Bars.svg"
pdf_path = OUTPUT_DIR / "Figure1_Expanded_Bars.pdf"
png_path = OUTPUT_DIR / "Figure1_Expanded_Bars.png"

plt.savefig(
    svg_path,
    dpi=400,
    bbox_inches="tight",
    format="svg",
)

plt.savefig(
    pdf_path,
    dpi=400,
    bbox_inches="tight",
    format="pdf",
)

plt.savefig(
    png_path,
    dpi=400,
    bbox_inches="tight",
    format="png",
)

plt.show()

print(f"Data loaded from: {DATA_PATH}")
print(f"SVG saved to: {svg_path}")
print(f"PDF saved to: {pdf_path}")
print(f"PNG saved to: {png_path}")
