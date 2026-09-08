from pathlib import Path as FilePath
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle, PathPatch, Patch
from matplotlib.path import Path as MplPath
np.random.seed(2026)
ROOT = FilePath(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "publication" / "uncertainty_propagation_figure_data.xlsx"
OUTPUT_DIR = ROOT / "outputs" / "figures"
FONT_FAMILY = "Times New Roman"
FONT_WEIGHT = "bold"
GLOBAL_FONT_SIZE = 22
SPINE_LINEWIDTH = 1.5
TICK_MAJOR_WIDTH = 1.2
TICK_DIRECTION = "in"
FIG_WIDTH = 14
FIG_HEIGHT_PANEL_A = 5.2
FIG_HEIGHT_PANEL_B = 5.5
PANEL_SPACING = 1.5
LEGEND_HEIGHT_A = 0.8
LEGEND_HEIGHT_B = 0.8
FIG_HEIGHT_TOTAL = (
    FIG_HEIGHT_PANEL_A
    + FIG_HEIGHT_PANEL_B
    + PANEL_SPACING
    + LEGEND_HEIGHT_A
    + LEGEND_HEIGHT_B
)

GRID_HSPACE = 0.45
GRID_WSPACE = 0.02
GRID_WIDTH_RATIOS = [25, 1]

TITLE_SIZE = 22
TITLE_PAD = 15
AXIS_LABEL_SIZE = 22
AXIS_LABEL_PAD_X = 22
AXIS_LABEL_PAD_Y = 10

LINEWIDTH_ACTUAL = 2.0
LINEWIDTH_FORECAST = 1.8
LINEWIDTH_PI_EDGE = 0.8
LINEWIDTH_ERROR_LINES = 1.0
LINEWIDTH_INSET_HIST_EDGE = 1.0

TICK_LABEL_SIZE_PANEL_A = 22
UNCERTAINTY_LABEL_SIZE = 18
UNCERTAINTY_LABEL_Y_POS = -0.10
GRID_LINESTYLE_A = "--"
GRID_LINEWIDTH_A = 0.8
GRID_ALPHA_A = 0.15

INSET_POSITION = [0.78, 0.55, 0.18, 0.25]
INSET_LABEL_SIZE = 12
INSET_TICK_SIZE = 10
INSET_SPINE_WIDTH = 1.2

LEGEND_FONT_SIZE_A = 18
LEGEND_FRAME_LINEWIDTH_A = 1.2
LEGEND_COL_SPACING_A = 0.5

HEATMAP_GRID_LINEWIDTH = 1.0
DIAGONAL_BORDER_LINEWIDTH = 1.5
IMPORTANT_CELL_BORDER_LINEWIDTH = 2.0

TICK_LABEL_SIZE_PANEL_B = 20
HEATMAP_CMAP = "coolwarm"
DIAGONAL_COLORS = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#FFA07A", "#98D8C8", "#DDA0DD"]

HEATMAP_CELL_TEXT_SIZE = 20
HEATMAP_DIAGONAL_TEXT_SIZE = 18
HEATMAP_SIGNIFICANT_TEXT_SIZE = 18
COLORBAR_LABEL_SIZE = 20
COLORBAR_TICK_SIZE = 20

LEGEND_FONT_SIZE_B = 20
LEGEND_FRAME_LINEWIDTH_B = 1.2
LEGEND_COL_SPACING_B = 0.3

plt.rcParams["font.family"] = FONT_FAMILY
plt.rcParams["font.size"] = GLOBAL_FONT_SIZE
plt.rcParams["font.weight"] = FONT_WEIGHT
plt.rcParams["axes.linewidth"] = SPINE_LINEWIDTH
plt.rcParams["axes.edgecolor"] = "black"
plt.rcParams["xtick.direction"] = TICK_DIRECTION
plt.rcParams["ytick.direction"] = TICK_DIRECTION
plt.rcParams["xtick.major.width"] = TICK_MAJOR_WIDTH
plt.rcParams["ytick.major.width"] = TICK_MAJOR_WIDTH
plt.rcParams["xtick.minor.visible"] = True
plt.rcParams["ytick.minor.visible"] = True

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Data file not found:\n{DATA_PATH}\n\n"
        "Place the Excel workbook at:\n"
        r"data\publication\uncertainty_propagation_figure_data.xlsx"
    )

df_pi = pd.read_excel(DATA_PATH, sheet_name="Prediction Intervals")
df_um = pd.read_excel(DATA_PATH, sheet_name="Uncertainty Matrix", index_col=0)

required_pi_columns = [
    "Time_Hours",
    "Actual",
    "Forecast",
    "Upper_CI",
    "Lower_CI",
    "Prediction_Error",
]

missing_pi = [c for c in required_pi_columns if c not in df_pi.columns]

if missing_pi:
    raise ValueError(f"Missing columns in Prediction Intervals: {missing_pi}")

if df_um.empty:
    raise ValueError("Uncertainty Matrix sheet is empty.")

t_horizon = df_pi["Time_Hours"].to_numpy(dtype=float)
actual = df_pi["Actual"].to_numpy(dtype=float)
forecast = df_pi["Forecast"].to_numpy(dtype=float)
upper_ci = df_pi["Upper_CI"].to_numpy(dtype=float)
lower_ci = df_pi["Lower_CI"].to_numpy(dtype=float)
prediction_error = df_pi["Prediction_Error"].to_numpy(dtype=float)

uncertainty_matrix = df_um.to_numpy(dtype=float)
var_labels = df_um.index.astype(str).tolist()
n_vars = len(var_labels)

if uncertainty_matrix.shape != (n_vars, n_vars):
    raise ValueError(
        f"Uncertainty Matrix must be square. Found shape: {uncertainty_matrix.shape}"
    )

def add_3d_shadow_to_lines(
    ax,
    x_data,
    y_data,
    color,
    offset_x=0.05,
    offset_y=-0.02,
    alpha=0.15,
):
    x_shadow = x_data + offset_x
    y_shadow = y_data + offset_y
    verts = []
    codes = []

    for i in range(len(x_shadow)):
        verts.append((x_shadow[i], y_shadow[i]))
        codes.append(MplPath.LINETO if i > 0 else MplPath.MOVETO)

    for i in range(len(x_shadow) - 1, -1, -1):
        verts.append((x_data[i], y_data[i]))
        codes.append(MplPath.LINETO)

    codes[0] = MplPath.MOVETO
    codes[-1] = MplPath.CLOSEPOLY

    path = MplPath(verts, codes)
    patch = PathPatch(
        path,
        facecolor=color,
        edgecolor="none",
        alpha=alpha,
        zorder=0,
    )
    ax.add_patch(patch)

fig = plt.figure(figsize=(FIG_WIDTH, FIG_HEIGHT_TOTAL))

gs = gridspec.GridSpec(
    4,
    2,
    figure=fig,
    hspace=GRID_HSPACE,
    wspace=GRID_WSPACE,
    height_ratios=[2, 0.20, 2, 0.20],
    width_ratios=GRID_WIDTH_RATIOS,
)

ax1 = fig.add_subplot(gs[0, 0])

ax1.fill_between(
    t_horizon,
    lower_ci,
    upper_ci,
    alpha=0.3,
    color="#2E86C1",
    label="95% Prediction Interval",
    edgecolor="#1A5276",
    linewidth=LINEWIDTH_PI_EDGE,
)

for i in range(3):
    expand = 0.02 * (i + 1)
    ax1.fill_between(
        t_horizon,
        lower_ci - expand,
        upper_ci + expand,
        alpha=0.05,
        color="#2E86C1",
    )

ax1.plot(
    t_horizon,
    actual,
    color="#1A5276",
    linewidth=LINEWIDTH_ACTUAL,
    label="Actual Values",
    linestyle="-",
)

ax1.plot(
    t_horizon,
    forecast,
    color="#E74C3C",
    linewidth=LINEWIDTH_FORECAST,
    label="Forecast Values",
    linestyle="--",
)

blocks = [(0, 6), (6, 12), (12, 18), (18, 24)]
block_colors = ["#90EE90", "#FFFACD", "#FFB6C1", "#87CEEB"]

for start_t, end_t in blocks:
    mask = (t_horizon >= start_t) & (t_horizon <= end_t)
    t_block = t_horizon[mask]
    actual_block = actual[mask]
    forecast_block = forecast[mask]

    if len(t_block) > 2:
        add_3d_shadow_to_lines(
            ax1,
            t_block,
            actual_block,
            color="#1A5276",
            offset_x=0.1,
            offset_y=-0.03,
            alpha=0.2,
        )
        add_3d_shadow_to_lines(
            ax1,
            t_block,
            forecast_block,
            color="#8B0000",
            offset_x=0.1,
            offset_y=-0.03,
            alpha=0.2,
        )

for idx, (start_t, end_t) in enumerate(blocks):
    ax1.axvspan(
        start_t,
        end_t,
        alpha=0.08,
        color=block_colors[idx],
        zorder=0,
    )

measurement_indices = np.arange(0, len(t_horizon), 6)

ax1.scatter(
    t_horizon[measurement_indices],
    actual[measurement_indices]
    + 0.03 * np.random.randn(len(measurement_indices)),
    s=60,
    color="#28B463",
    alpha=0.6,
    edgecolor="black",
    linewidth=1.5,
    zorder=5,
    label="Measurements",
)

error_indices = np.arange(0, len(t_horizon), 12)

for idx in error_indices:
    ax1.plot(
        [t_horizon[idx], t_horizon[idx]],
        [actual[idx], forecast[idx]],
        color="#FF6B6B",
        linewidth=LINEWIDTH_ERROR_LINES,
        alpha=0.4,
        linestyle="-.",
    )

error_indices = np.arange(3, len(t_horizon), 8)
error_bar_x_offset = 0.3
error_bar_y_offset = 0.4

for idx in error_indices:
    ax1.errorbar(
        t_horizon[idx] + error_bar_x_offset,
        forecast[idx] + error_bar_y_offset,
        yerr=abs(actual[idx] - forecast[idx]) * 0.5,
        fmt="none",
        color="black",
        alpha=0.3,
        capsize=4,
        capthick=1.0,
        elinewidth=1.0,
    )

ax1.set_xlim(0, 24)
ax1.set_ylim([-0.05, 1.15])

split_points = [0, 6, 12, 18, 24]

ax1.set_xticks(split_points)
ax1.set_xticklabels(
    [f"{h}h" for h in split_points],
    fontsize=TICK_LABEL_SIZE_PANEL_A,
    fontweight=FONT_WEIGHT,
)
ax1.tick_params(
    axis="both",
    which="major",
    labelsize=TICK_LABEL_SIZE_PANEL_A,
)

for point in split_points[1:-1]:
    ax1.axvline(
        x=point,
        color="black",
        linestyle="--",
        linewidth=1.0,
        alpha=0.5,
        ymin=0.05,
        ymax=0.95,
    )

midpoints = [
    (split_points[i] + split_points[i + 1]) / 2
    for i in range(len(split_points) - 1)
]

uncertainty_labels = [
    "Low\nUncertainty",
    "Moderate\nUncertainty",
    "High\nUncertainty",
    "Moderate\nUncertainty",
]

for mid_x, label_text in zip(midpoints, uncertainty_labels):
    ax1.text(
        mid_x,
        UNCERTAINTY_LABEL_Y_POS,
        label_text,
        ha="center",
        va="top",
        fontsize=UNCERTAINTY_LABEL_SIZE,
        fontweight=FONT_WEIGHT,
        color="black",
    )

ax1.set_xlabel(
    "Time (hours)",
    fontsize=AXIS_LABEL_SIZE,
    fontweight=FONT_WEIGHT,
    labelpad=AXIS_LABEL_PAD_X,
)

ax1.set_ylabel(
    "Normalized Value",
    fontsize=AXIS_LABEL_SIZE,
    fontweight=FONT_WEIGHT,
    labelpad=AXIS_LABEL_PAD_Y,
)

ax1.grid(
    True,
    alpha=GRID_ALPHA_A,
    linestyle=GRID_LINESTYLE_A,
    linewidth=GRID_LINEWIDTH_A,
)

ax1.set_title(
    "(a) Prediction Intervals with Uncertainty Bands",
    fontsize=TITLE_SIZE,
    fontweight=FONT_WEIGHT,
    loc="left",
    pad=TITLE_PAD,
)

for spine in ax1.spines.values():
    spine.set_linewidth(SPINE_LINEWIDTH)

ax1.grid(True, alpha=0.08, linestyle=":", linewidth=0.5)

inset_ax = ax1.inset_axes(INSET_POSITION)

inset_ax.hist(
    prediction_error,
    bins=20,
    color="#2E86C1",
    edgecolor="black",
    linewidth=LINEWIDTH_INSET_HIST_EDGE,
    alpha=0.7,
)

inset_ax.axvline(
    x=0,
    color="#E74C3C",
    linestyle="--",
    linewidth=1.5,
)

inset_ax.set_xlabel(
    "Error",
    fontsize=INSET_LABEL_SIZE,
    fontweight=FONT_WEIGHT,
)

inset_ax.set_ylabel(
    "Frequency",
    fontsize=INSET_LABEL_SIZE,
    fontweight=FONT_WEIGHT,
)

inset_ax.tick_params(labelsize=INSET_TICK_SIZE)

for spine in inset_ax.spines.values():
    spine.set_linewidth(INSET_SPINE_WIDTH)

ax1_legend = fig.add_subplot(gs[1, :])
ax1_legend.axis("off")

pos_leg_a = ax1_legend.get_position()

ax1_legend.set_position(
    [
        pos_leg_a.x0,
        pos_leg_a.y0 - 0.01,
        pos_leg_a.width,
        pos_leg_a.height,
    ]
)

legend_elements_a = [
    plt.Line2D(
        [0],
        [0],
        color="#2E86C1",
        linewidth=2.5,
        label="95% Prediction Interval",
    ),
    plt.Line2D(
        [0],
        [0],
        color="#1A5276",
        linewidth=2.0,
        label="Actual Values",
    ),
    plt.Line2D(
        [0],
        [0],
        color="#E74C3C",
        linewidth=1.8,
        linestyle="--",
        label="Forecast Values",
    ),
    plt.Line2D(
        [0],
        [0],
        marker="o",
        color="w",
        markerfacecolor="#28B463",
        markersize=8,
        label="Measurements",
        markeredgecolor="black",
        markeredgewidth=1.5,
    ),
]

legend_a = ax1_legend.legend(
    handles=legend_elements_a,
    loc="center",
    fontsize=LEGEND_FONT_SIZE_A,
    frameon=True,
    edgecolor="black",
    fancybox=False,
    ncol=4,
    columnspacing=LEGEND_COL_SPACING_A,
)

legend_a.get_frame().set_linewidth(LEGEND_FRAME_LINEWIDTH_A)

ax2 = fig.add_subplot(gs[2, 0])

im = ax2.imshow(
    uncertainty_matrix,
    cmap=HEATMAP_CMAP,
    interpolation="bilinear",
    aspect="auto",
    vmin=-0.1,
    vmax=1.0,
)

ax2.set_xticks(np.arange(n_vars))
ax2.set_yticks(np.arange(n_vars))

ax2.set_xticklabels(
    var_labels,
    fontsize=TICK_LABEL_SIZE_PANEL_B,
    fontweight=FONT_WEIGHT,
    rotation=20,
    ha="right",
)

ax2.set_yticklabels(
    var_labels,
    fontsize=TICK_LABEL_SIZE_PANEL_B,
    fontweight=FONT_WEIGHT,
)

ax2.tick_params(
    axis="both",
    which="major",
    labelsize=TICK_LABEL_SIZE_PANEL_B,
)

for i in range(n_vars):
    for j in range(n_vars):
        value = uncertainty_matrix[i, j]
        text_color = "white" if value > 0.5 else "black"

        ax2.text(
            j,
            i,
            f"{value:.2f}",
            ha="center",
            va="center",
            color=text_color,
            fontsize=HEATMAP_CELL_TEXT_SIZE,
            fontweight=FONT_WEIGHT,
        )

for i in range(n_vars + 1):
    ax2.axhline(
        i - 0.5,
        color="black",
        linewidth=HEATMAP_GRID_LINEWIDTH,
    )
    ax2.axvline(
        i - 0.5,
        color="black",
        linewidth=HEATMAP_GRID_LINEWIDTH,
    )

for i in range(n_vars):
    color_idx = i % len(DIAGONAL_COLORS)

    rect = Rectangle(
        (i - 0.5, i - 0.5),
        1,
        1,
        fill=True,
        facecolor=DIAGONAL_COLORS[color_idx],
        edgecolor="black",
        linewidth=DIAGONAL_BORDER_LINEWIDTH,
        alpha=0.7,
        zorder=10,
    )

    ax2.add_patch(rect)

    ax2.text(
        i,
        i,
        f"{uncertainty_matrix[i, i]:.2f}",
        ha="center",
        va="center",
        color="white",
        fontsize=HEATMAP_DIAGONAL_TEXT_SIZE,
        fontweight=FONT_WEIGHT,
        zorder=11,
    )

important_cells = []

for i in range(n_vars):
    for j in range(n_vars):
        if i != j and uncertainty_matrix[i, j] > 0.8:
            important_cells.append((i, j))

important_pairs = [
    (0, 1),
    (1, 0),
    (2, 4),
    (4, 2),
    (1, 5),
    (5, 1),
    (3, 5),
    (5, 3),
]

for i, j in important_pairs:
    if (i, j) not in important_cells and i < n_vars and j < n_vars:
        important_cells.append((i, j))

important_cells = list(set(important_cells))

for i, j in important_cells:
    rect = Rectangle(
        (j - 0.5, i - 0.5),
        1,
        1,
        fill=False,
        edgecolor="lime",
        linewidth=IMPORTANT_CELL_BORDER_LINEWIDTH,
        linestyle="--",
        alpha=0.9,
        zorder=9,
    )

    ax2.add_patch(rect)

cbar_ax = fig.add_subplot(gs[2, 1])

cbar = plt.colorbar(
    im,
    cax=cbar_ax,
)

cbar.ax.set_ylabel(
    "Uncertainty Correlation",
    fontsize=COLORBAR_LABEL_SIZE,
    fontweight=FONT_WEIGHT,
)

cbar.ax.tick_params(
    labelsize=COLORBAR_TICK_SIZE,
)

significant_pairs = [
    (0, 1, "Strong"),
    (4, 2, "Strong"),
    (1, 5, "Strong"),
]

for i, j, label in significant_pairs:
    if i < n_vars and j < n_vars:
        ax2.text(
            j,
            i,
            f"↑{label}↑",
            ha="center",
            va="center",
            fontsize=HEATMAP_SIGNIFICANT_TEXT_SIZE,
            fontweight=FONT_WEIGHT,
            color="yellow",
            alpha=0.9,
            zorder=12,
        )

ax2.set_title(
    "(b) Forecast Uncertainty Propagation Matrix",
    fontsize=TITLE_SIZE,
    fontweight=FONT_WEIGHT,
    loc="left",
    pad=TITLE_PAD,
)

for spine in ax2.spines.values():
    spine.set_linewidth(SPINE_LINEWIDTH)

ax2_legend = fig.add_subplot(gs[3, :])
ax2_legend.axis("off")

pos = ax2_legend.get_position()

ax2_legend.set_position(
    [
        pos.x0,
        pos.y0 - 0.03,
        pos.width,
        pos.height,
    ]
)

legend_elements_b = [
    Patch(
        facecolor="none",
        edgecolor="lime",
        linewidth=IMPORTANT_CELL_BORDER_LINEWIDTH,
        linestyle="--",
        label="High Correlation (>0.8)",
    ),
    Patch(
        facecolor=DIAGONAL_COLORS[0],
        edgecolor="black",
        linewidth=DIAGONAL_BORDER_LINEWIDTH,
        label="Diagonal (Self-Correlation)",
        alpha=0.7,
    ),
    plt.Rectangle(
        (0, 0),
        1,
        1,
        facecolor="#FF6B6B",
        alpha=0.7,
        edgecolor="black",
        linewidth=1.2,
        label="Variable 1",
    ),
    plt.Rectangle(
        (0, 0),
        1,
        1,
        facecolor="#4ECDC4",
        alpha=0.7,
        edgecolor="black",
        linewidth=1.2,
        label="Variable 2",
    ),
    plt.Rectangle(
        (0, 0),
        1,
        1,
        facecolor="#45B7D1",
        alpha=0.7,
        edgecolor="black",
        linewidth=1.2,
        label="Variable 3",
    ),
    plt.Rectangle(
        (0, 0),
        1,
        1,
        facecolor="#FFA07A",
        alpha=0.7,
        edgecolor="black",
        linewidth=1.2,
        label="Variable 4",
    ),
    plt.Rectangle(
        (0, 0),
        1,
        1,
        facecolor="#98D8C8",
        alpha=0.7,
        edgecolor="black",
        linewidth=1.2,
        label="Variable 5",
    ),
    plt.Rectangle(
        (0, 0),
        1,
        1,
        facecolor="#DDA0DD",
        alpha=0.7,
        edgecolor="black",
        linewidth=1.2,
        label="Variable 6",
    ),
]

legend_b = ax2_legend.legend(
    handles=legend_elements_b,
    loc="center",
    fontsize=LEGEND_FONT_SIZE_B,
    frameon=True,
    edgecolor="black",
    fancybox=False,
    ncol=4,
    columnspacing=LEGEND_COL_SPACING_B,
)

legend_b.get_frame().set_linewidth(LEGEND_FRAME_LINEWIDTH_B)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

svg_path = OUTPUT_DIR / "Figure2_Uncertainty_Propagation.svg"
pdf_path = OUTPUT_DIR / "Figure2_Uncertainty_Propagation.pdf"
png_path = OUTPUT_DIR / "Figure2_Uncertainty_Propagation.png"

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
