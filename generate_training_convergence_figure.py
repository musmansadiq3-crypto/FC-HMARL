from pathlib import Path as FilePath

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.gridspec as gridspec
from scipy.signal import savgol_filter

ROOT = FilePath(__file__).resolve().parent
DATA_PATH = ROOT / "evaluation" / "data" / "training_diagnostics" / "fc_hmarl_training_convergence_data.xlsx"
OUTPUT_DIR = ROOT / "outputs" / "figures"

FONT_FAMILY = "Times New Roman"
FONT_WEIGHT = "bold"
GLOBAL_FONT_SIZE = 18

FIG_WIDTH = 14
FIG_HEIGHT_PANEL_A = 5
FIG_HEIGHT_PANEL_B = 4
PANEL_SPACING = 3.4
LEGEND_SPACING = 0.8
FIG_HEIGHT_TOTAL = FIG_HEIGHT_PANEL_A + FIG_HEIGHT_PANEL_B + PANEL_SPACING + LEGEND_SPACING

SPINE_LINEWIDTH = 1.5
TICK_MAJOR_WIDTH = 1.2
TICK_DIRECTION = "in"

LINEWIDTH_LOCAL_AGENTS = 1.5
LINEWIDTH_COORDINATOR = 2.5
LINEWIDTH_COORDINATOR_MA = 1.5
LINEWIDTH_CONVERGENCE_A = 1.5
LINEWIDTH_PHASE_DIVIDER = 1.5
PHASE_BOX_LINEWIDTH = 1.0

LINEWIDTH_CRITIC_LOSS = 1.8
LINEWIDTH_ACTOR_LOSS = 1.8
LINEWIDTH_POLICY_ENTROPY = 1.8
LINEWIDTH_CONVERGENCE_B = 1.5
LINEWIDTH_ANNOTATION_ARROWS = 1.2

TITLE_SIZE = 26
TITLE_PAD = 15
AXIS_LABEL_SIZE = 26
TICK_LABEL_SIZE = 26

PHASE_TEXT_SIZE = 16
ANNOTATION_TEXT_SIZE_B = 16

LEGEND_FONT_SIZE_A = 18
LEGEND_FONT_SIZE_B = 18
LEGEND_FRAME_LINEWIDTH = 1.2
LEGEND_COL_SPACING_A = 0.8

GRID_ALPHA_MAJOR = 0.15
GRID_ALPHA_MINOR = 0.08

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
        r"evaluation\data\training_diagnostics\fc_hmarl_training_convergence_data.xlsx"
    )

df_rewards = pd.read_excel(DATA_PATH, sheet_name="Training_Rewards")
df_loss = pd.read_excel(DATA_PATH, sheet_name="Loss_Entropy")

required_reward_columns = [
    "Episodes",
    "MG1_Reward",
    "MG2_Reward",
    "MG3_Reward",
    "MG4_Reward",
    "MG5_Reward",
    "VPP_Coordinator_Reward",
    "Coordinator_MA_100",
]

required_loss_columns = [
    "Critic_Loss",
    "Actor_Loss",
    "Policy_Entropy",
]

missing_rewards = [c for c in required_reward_columns if c not in df_rewards.columns]
missing_loss = [c for c in required_loss_columns if c not in df_loss.columns]

if missing_rewards:
    raise ValueError(f"Missing columns in Training_Rewards: {missing_rewards}")

if missing_loss:
    raise ValueError(f"Missing columns in Loss_Entropy: {missing_loss}")

episodes = pd.to_numeric(df_rewards["Episodes"], errors="raise").to_numpy(dtype=float)
agents = ["MG1", "MG2", "MG3", "MG4", "MG5"]

agent_rewards = {
    agent: pd.to_numeric(df_rewards[f"{agent}_Reward"], errors="raise").to_numpy(dtype=float)
    for agent in agents
}

coord_reward = pd.to_numeric(
    df_rewards["VPP_Coordinator_Reward"],
    errors="raise",
).to_numpy(dtype=float)

coord_ma_full = pd.to_numeric(
    df_rewards["Coordinator_MA_100"],
    errors="raise",
).to_numpy(dtype=float)

critic_loss = pd.to_numeric(
    df_loss["Critic_Loss"],
    errors="raise",
).to_numpy(dtype=float)

actor_loss = pd.to_numeric(
    df_loss["Actor_Loss"],
    errors="raise",
).to_numpy(dtype=float)

entropy = pd.to_numeric(
    df_loss["Policy_Entropy"],
    errors="raise",
).to_numpy(dtype=float)

if not (
    len(episodes)
    == len(coord_reward)
    == len(coord_ma_full)
    == len(critic_loss)
    == len(actor_loss)
    == len(entropy)
):
    raise ValueError("Training reward, loss, and entropy series must have the same number of rows.")

for agent in agents:
    if len(agent_rewards[agent]) != len(episodes):
        raise ValueError(f"{agent} reward length does not match Episodes.")

try:
    df_conv = pd.read_excel(DATA_PATH, sheet_name="Convergence_Metrics")
    convergence_episode = float(
        df_conv.loc[
            df_conv["Metric"] == "Convergence_Episode",
            "Value",
        ].iloc[0]
    )
    final_coord = float(
        df_conv.loc[
            df_conv["Metric"] == "Final_Coordinator_Reward",
            "Value",
        ].iloc[0]
    )
except (ValueError, KeyError, IndexError):
    convergence_episode = 2500.0
    final_coord = float(np.mean(coord_reward[-min(500, len(coord_reward)):]))

fig = plt.figure(figsize=(FIG_WIDTH, FIG_HEIGHT_TOTAL))

height_ratios = [
    FIG_HEIGHT_PANEL_A,
    LEGEND_SPACING,
    FIG_HEIGHT_PANEL_B,
]

total_height = sum(height_ratios)
normalized_ratios = [h / total_height for h in height_ratios]

gs = gridspec.GridSpec(
    3,
    1,
    figure=fig,
    hspace=PANEL_SPACING / (FIG_HEIGHT_PANEL_A + FIG_HEIGHT_PANEL_B),
    height_ratios=normalized_ratios,
)

ax1 = fig.add_subplot(gs[0])

ax1.set_ylim([-40, 80])
ax1.set_xlim([0, float(np.max(episodes))])

training_phases = [
    {"name": "Phase 1\nInitial\nExploration", "start": 0, "end": 1250, "color": "#E8F4F8", "alpha": 0.15},
    {"name": "Phase 2\nRapid\nLearning", "start": 1250, "end": 2500, "color": "#FEF9E7", "alpha": 0.15},
    {"name": "Phase 3\nStabilization", "start": 2500, "end": 3750, "color": "#EAFAF1", "alpha": 0.15},
    {"name": "Phase 4\nOptimal\nPerformance", "start": 3750, "end": 5000, "color": "#F5EEF8", "alpha": 0.15},
]

for phase in training_phases:
    ax1.axvspan(
        phase["start"],
        phase["end"],
        alpha=phase["alpha"],
        color=phase["color"],
    )

    if phase["start"] > 0:
        ax1.axvline(
            x=phase["start"],
            color="#2C3E50",
            linestyle="-",
            linewidth=LINEWIDTH_PHASE_DIVIDER,
            alpha=0.6,
        )

    mid_point = (phase["start"] + phase["end"]) / 2

    ax1.text(
        mid_point,
        66,
        phase["name"],
        ha="center",
        va="top",
        fontsize=PHASE_TEXT_SIZE,
        fontweight=FONT_WEIGHT,
        color="#2C3E50",
        bbox=dict(
            boxstyle="round,pad=0.3",
            facecolor="white",
            edgecolor="#2C3E50",
            linewidth=PHASE_BOX_LINEWIDTH,
            alpha=0.7,
        ),
    )

ax1.axhline(
    y=0,
    color="#7F8C8D",
    linestyle=":",
    linewidth=1.0,
    alpha=0.3,
)

agent_colors = ["#FDB813", "#2E86C1", "#28B463", "#E74C3C", "#8E44AD"]

tail = min(500, len(episodes))

for idx, agent in enumerate(agents):
    reward = agent_rewards[agent]

    ax1.plot(
        episodes,
        reward,
        color=agent_colors[idx],
        linewidth=LINEWIDTH_LOCAL_AGENTS,
        alpha=0.7,
        label=f"{agent} (Local)",
    )

    std_dev = np.std(reward[-tail:])

    ax1.fill_between(
        episodes,
        reward - std_dev * 0.5,
        reward + std_dev * 0.5,
        alpha=0.08,
        color=agent_colors[idx],
    )

ax1.plot(
    episodes,
    coord_reward,
    color="#1A5276",
    linewidth=LINEWIDTH_COORDINATOR,
    label="VPP Coordinator",
    linestyle="-",
)

coord_std = np.std(coord_reward[-tail:])

ax1.fill_between(
    episodes,
    coord_reward - coord_std * 0.5,
    coord_reward + coord_std * 0.5,
    alpha=0.15,
    color="#1A5276",
)

coord_ma = coord_ma_full.copy()

post_conv_start = int(np.searchsorted(episodes, convergence_episode * 1.5))

if post_conv_start < len(coord_ma):
    segment_length = len(coord_ma[post_conv_start:])
    window_size = min(51, segment_length if segment_length % 2 == 1 else segment_length - 1)

    if window_size >= 5:
        coord_ma[post_conv_start:] = savgol_filter(
            coord_ma[post_conv_start:],
            window_length=window_size,
            polyorder=2,
        )

ax1.plot(
    episodes,
    coord_ma,
    color="#E74C3C",
    linewidth=LINEWIDTH_COORDINATOR_MA,
    linestyle="--",
    alpha=0.6,
    label="Coordinator MA",
)

convergence_ep_rounded = int(round(convergence_episode))

ax1.axvline(
    x=convergence_episode,
    color="#28B463",
    linestyle="--",
    linewidth=LINEWIDTH_CONVERGENCE_A,
    alpha=0.8,
    label=f"Convergence ({convergence_ep_rounded} ep)",
)

ax1.set_xlabel(
    "Training Episodes",
    fontsize=AXIS_LABEL_SIZE,
    fontweight=FONT_WEIGHT,
)

ax1.set_ylabel(
    "Average Reward",
    fontsize=AXIS_LABEL_SIZE,
    fontweight=FONT_WEIGHT,
)

ax1.tick_params(
    axis="both",
    which="major",
    labelsize=TICK_LABEL_SIZE,
)

ax1.grid(
    True,
    alpha=GRID_ALPHA_MAJOR,
    linestyle="--",
    linewidth=1,
)

ax1.set_title(
    "(a) Training Reward Curves - Local Agents & VPP Coordinator",
    fontsize=TITLE_SIZE,
    fontweight=FONT_WEIGHT,
    loc="left",
    pad=TITLE_PAD,
)

for spine in ax1.spines.values():
    spine.set_linewidth(SPINE_LINEWIDTH)

ax1.grid(
    True,
    alpha=GRID_ALPHA_MINOR,
    linestyle=":",
    linewidth=0.5,
)

ax1_legend = fig.add_subplot(gs[1])
ax1_legend.axis("off")

legend_elements_a = [
    plt.Line2D([0], [0], color="#FDB813", linewidth=LINEWIDTH_LOCAL_AGENTS, label="MG1 (Local)"),
    plt.Line2D([0], [0], color="#2E86C1", linewidth=LINEWIDTH_LOCAL_AGENTS, label="MG2 (Local)"),
    plt.Line2D([0], [0], color="#28B463", linewidth=LINEWIDTH_LOCAL_AGENTS, label="MG3 (Local)"),
    plt.Line2D([0], [0], color="#E74C3C", linewidth=LINEWIDTH_LOCAL_AGENTS, label="MG4 (Local)"),
    plt.Line2D([0], [0], color="#8E44AD", linewidth=LINEWIDTH_LOCAL_AGENTS, label="MG5 (Local)"),
    plt.Line2D([0], [0], color="#1A5276", linewidth=LINEWIDTH_COORDINATOR, label="VPP Coordinator"),
    plt.Line2D([0], [0], color="#E74C3C", linewidth=LINEWIDTH_COORDINATOR_MA, linestyle="--", label="Coordinator MA"),
    plt.Line2D([0], [0], color="#28B463", linewidth=LINEWIDTH_CONVERGENCE_A, linestyle="--", label=f"Convergence ({convergence_ep_rounded} ep)"),
    plt.Rectangle((0, 0), 1, 1, facecolor="#E8F4F8", alpha=0.15, edgecolor="#2C3E50", linewidth=1.5, label="Phase Divisions"),
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
    handletextpad=0.8,
    borderpad=0.5,
)

legend_a.get_frame().set_linewidth(LEGEND_FRAME_LINEWIDTH)

ax2 = fig.add_subplot(gs[2])

ax2.plot(
    episodes,
    critic_loss,
    color="#E74C3C",
    linewidth=LINEWIDTH_CRITIC_LOSS,
    label="Critic Loss",
    linestyle="-",
)

ax2.plot(
    episodes,
    actor_loss,
    color="#2E86C1",
    linewidth=LINEWIDTH_ACTOR_LOSS,
    label="Actor Loss",
    linestyle="--",
)

ax2b = ax2.twinx()

ax2b.plot(
    episodes,
    entropy,
    color="#28B463",
    linewidth=LINEWIDTH_POLICY_ENTROPY,
    label="Policy Entropy",
    linestyle="-.",
)

ax2b.set_ylabel(
    "Policy Entropy",
    fontsize=AXIS_LABEL_SIZE,
    fontweight=FONT_WEIGHT,
    color="#28B463",
)

ax2b.tick_params(
    axis="y",
    labelcolor="#28B463",
    labelsize=TICK_LABEL_SIZE,
)

ax2b.spines["right"].set_linewidth(SPINE_LINEWIDTH)
ax2b.spines["right"].set_color("#28B463")

ax2.fill_between(
    episodes,
    critic_loss - 0.05,
    critic_loss + 0.05,
    alpha=0.1,
    color="#E74C3C",
)

ax2.fill_between(
    episodes,
    actor_loss - 0.04,
    actor_loss + 0.04,
    alpha=0.1,
    color="#2E86C1",
)

ax2.axvline(
    x=convergence_episode,
    color="#28B463",
    linestyle="--",
    linewidth=LINEWIDTH_CONVERGENCE_B,
    alpha=0.8,
)

ax2.annotate(
    "Loss Stabilization",
    xy=(2000, 0.12),
    xytext=(2200, 0.25),
    arrowprops=dict(
        arrowstyle="->",
        lw=LINEWIDTH_ANNOTATION_ARROWS,
        color="black",
    ),
    fontsize=ANNOTATION_TEXT_SIZE_B,
    fontweight=FONT_WEIGHT,
)

ax2b.annotate(
    "Entropy Decay",
    xy=(1000, 0.3),
    xytext=(1200, 0.5),
    arrowprops=dict(
        arrowstyle="->",
        lw=LINEWIDTH_ANNOTATION_ARROWS,
        color="#28B463",
    ),
    fontsize=ANNOTATION_TEXT_SIZE_B,
    fontweight=FONT_WEIGHT,
    color="#28B463",
)

ax2.set_xlabel(
    "Training Episodes",
    fontsize=AXIS_LABEL_SIZE,
    fontweight=FONT_WEIGHT,
)

ax2.set_ylabel(
    "Loss Value",
    fontsize=AXIS_LABEL_SIZE,
    fontweight=FONT_WEIGHT,
)

ax2.tick_params(
    axis="both",
    which="major",
    labelsize=TICK_LABEL_SIZE,
)

ax2.legend(
    loc="upper right",
    fontsize=LEGEND_FONT_SIZE_B,
    frameon=True,
    edgecolor="black",
    fancybox=False,
)

ax2.grid(
    True,
    alpha=GRID_ALPHA_MAJOR,
    linestyle="--",
    linewidth=1,
)

ax2.set_title(
    "(b) Convergence Analysis - Loss Functions & Policy Entropy",
    fontsize=TITLE_SIZE,
    fontweight=FONT_WEIGHT,
    loc="left",
    pad=TITLE_PAD,
)

ax2.set_ylim([0, 0.9])
ax2.set_xlim([0, float(np.max(episodes))])

for spine in ax2.spines.values():
    spine.set_linewidth(SPINE_LINEWIDTH)

ax2.grid(
    True,
    alpha=GRID_ALPHA_MINOR,
    linestyle=":",
    linewidth=0.5,
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

svg_path = OUTPUT_DIR / "Figure3_Training_Convergence.svg"
pdf_path = OUTPUT_DIR / "Figure3_Training_Convergence.pdf"
png_path = OUTPUT_DIR / "Figure3_Training_Convergence.png"

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
print(f"Convergence episode: {convergence_ep_rounded}")
print(f"Final coordinator reward: {final_coord:.4f}")
print(f"SVG saved to: {svg_path}")
print(f"PDF saved to: {pdf_path}")
print(f"PNG saved to: {png_path}")
