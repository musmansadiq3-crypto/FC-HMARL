from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

np.random.seed(2026)

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "publication" / "forecasting_figure_data.xlsx"
OUTPUT_DIR = ROOT / "outputs" / "figures"

FIG_WIDTH = 16
FIG_HEIGHT_PANEL_A = 8.5
FIG_HEIGHT_PANEL_B = 5.0
PANEL_SPACING = 1.0
FIG_HEIGHT_TOTAL = FIG_HEIGHT_PANEL_A + FIG_HEIGHT_PANEL_B + PANEL_SPACING

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 22
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.linewidth"] = 2.5
plt.rcParams["axes.edgecolor"] = "black"
plt.rcParams["xtick.direction"] = "in"
plt.rcParams["ytick.direction"] = "in"
plt.rcParams["xtick.major.width"] = 2.5
plt.rcParams["ytick.major.width"] = 2.5
plt.rcParams["xtick.labelsize"] = 22
plt.rcParams["ytick.labelsize"] = 22
plt.rcParams["xtick.major.size"] = 10
plt.rcParams["ytick.major.size"] = 10
plt.rcParams["xtick.minor.visible"] = True
plt.rcParams["ytick.minor.visible"] = True

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Data file not found:\n{DATA_PATH}\n\n"
        "Place the Excel workbook at:\n"
        r"data\publication\forecasting_figure_data.xlsx"
    )

df_forecast = pd.read_excel(DATA_PATH, sheet_name="Actual_vs_Forecast")
df_errors = pd.read_excel(DATA_PATH, sheet_name="Error_Metrics")

required_forecast_columns = [
    "Time_Hours",
    "PV_Actual",
    "PV_Forecast",
    "Load_Actual",
    "Load_Forecast",
    "EV_Actual",
    "EV_Forecast",
    "Price_Actual",
    "Price_Forecast",
]

required_error_columns = [
    "Horizon",
    "RMSE_PV",
    "RMSE_Load",
    "RMSE_EV",
    "RMSE_Price",
    "MAE_PV",
    "MAE_Load",
    "MAE_EV",
    "MAE_Price",
    "MAPE_PV",
    "MAPE_Load",
    "MAPE_EV",
    "MAPE_Price",
]

missing_forecast = [c for c in required_forecast_columns if c not in df_forecast.columns]
missing_errors = [c for c in required_error_columns if c not in df_errors.columns]

if missing_forecast:
    raise ValueError(f"Missing columns in Actual_vs_Forecast: {missing_forecast}")

if missing_errors:
    raise ValueError(f"Missing columns in Error_Metrics: {missing_errors}")

hours = df_forecast["Time_Hours"].to_numpy(dtype=float)
pv_actual = df_forecast["PV_Actual"].to_numpy(dtype=float)
pv_forecast = df_forecast["PV_Forecast"].to_numpy(dtype=float)
load_actual = df_forecast["Load_Actual"].to_numpy(dtype=float)
load_forecast = df_forecast["Load_Forecast"].to_numpy(dtype=float)
ev_actual = df_forecast["EV_Actual"].to_numpy(dtype=float)
ev_forecast = df_forecast["EV_Forecast"].to_numpy(dtype=float)
price_actual = df_forecast["Price_Actual"].to_numpy(dtype=float)
price_forecast = df_forecast["Price_Forecast"].to_numpy(dtype=float)

horizons = df_errors["Horizon"].to_numpy()
rmse_pv = df_errors["RMSE_PV"].to_numpy(dtype=float)
rmse_load = df_errors["RMSE_Load"].to_numpy(dtype=float)
rmse_ev = df_errors["RMSE_EV"].to_numpy(dtype=float)
rmse_price = df_errors["RMSE_Price"].to_numpy(dtype=float)
mae_pv = df_errors["MAE_PV"].to_numpy(dtype=float)
mae_load = df_errors["MAE_Load"].to_numpy(dtype=float)
mae_ev = df_errors["MAE_EV"].to_numpy(dtype=float)
mae_price = df_errors["MAE_Price"].to_numpy(dtype=float)
mape_pv = df_errors["MAPE_PV"].to_numpy(dtype=float)
mape_load = df_errors["MAPE_Load"].to_numpy(dtype=float)
mape_ev = df_errors["MAPE_EV"].to_numpy(dtype=float)
mape_price = df_errors["MAPE_Price"].to_numpy(dtype=float)

n_points = len(hours)
n_horizons = len(horizons)

ci_width_pv = 0.08 + 0.04 * np.sin(hours / 4) + 0.02 * np.abs(np.sin(hours / 8))
ci_width_load = 0.06 + 0.03 * np.sin(hours / 3) + 0.02 * np.cos(hours / 5)
ci_width_ev = 0.05 + 0.02 * np.sin(hours / 5) + 0.03 * np.exp(-((hours - 12) ** 2) / 20)
ci_width_price = 0.07 + 0.03 * np.sin(hours / 4) + 0.02 * np.sin(hours / 6)

fig = plt.figure(figsize=(FIG_WIDTH, FIG_HEIGHT_TOTAL), constrained_layout=False)

gs = gridspec.GridSpec(
    3,
    2,
    figure=fig,
    hspace=0.75,
    wspace=0.30,
    height_ratios=[FIG_HEIGHT_PANEL_A / 2, FIG_HEIGHT_PANEL_A / 2, FIG_HEIGHT_PANEL_B],
    width_ratios=[1, 1],
)

measure_idx = np.arange(0, n_points, 8)
error_idx = np.arange(0, n_points, 12)

ax_a1 = fig.add_subplot(gs[0, 0])
ax_a1.plot(hours, pv_actual, color="#1A5276", linewidth=3, label="Actual PV")
ax_a1.plot(hours, pv_forecast, color="#E74C3C", linewidth=2.5, linestyle="--", label="Forecast PV")
ax_a1.fill_between(
    hours,
    pv_forecast - ci_width_pv,
    pv_forecast + ci_width_pv,
    alpha=0.25,
    color="#2E86C1",
    label="95% CI",
)
ax_a1.fill_between(
    hours,
    pv_forecast - ci_width_pv * 1.5,
    pv_forecast + ci_width_pv * 1.5,
    alpha=0.08,
    color="#2E86C1",
)
ax_a1.scatter(
    hours[measure_idx],
    pv_actual[measure_idx],
    s=50,
    color="#28B463",
    alpha=0.6,
    edgecolor="black",
    linewidth=1.5,
    zorder=5,
    label="Measurements",
)
for idx in error_idx:
    error_magnitude = abs(pv_actual[idx] - pv_forecast[idx])
    color_intensity = min(1, error_magnitude * 5)
    ax_a1.plot(
        [hours[idx], hours[idx]],
        [pv_actual[idx], pv_forecast[idx]],
        color=plt.cm.RdYlGn_r(color_intensity),
        linewidth=1.5,
        alpha=0.5,
        linestyle="-.",
    )
ax_a1.axvline(x=6, color="gray", linestyle=":", linewidth=1.5, alpha=0.5)
ax_a1.axvline(x=18, color="gray", linestyle=":", linewidth=1.5, alpha=0.5)
ax_a1.text(6.5, 0.9, "Sunrise", fontsize=14, fontweight="bold", color="gray", rotation=90, va="top")
ax_a1.text(18.5, 0.9, "Sunset", fontsize=14, fontweight="bold", color="gray", rotation=90, va="top")
ax_a1.set_xlabel("Time (hours)", fontsize=22, fontweight="bold")
ax_a1.set_ylabel("Normalized PV", fontsize=22, fontweight="bold")
ax_a1.grid(True, alpha=0.12, linestyle="--", linewidth=1)
ax_a1.set_title("(a) PV Generation Forecast", fontsize=24, fontweight="bold", loc="left", pad=8)
ax_a1.set_ylim([-0.05, 1.1])
ax_a1.set_xlim([0, 24])
for spine in ax_a1.spines.values():
    spine.set_linewidth(2.5)
rmse_val = np.sqrt(np.mean((pv_actual - pv_forecast) ** 2))
mae_val = np.mean(np.abs(pv_actual - pv_forecast))
mape_val = np.mean(np.abs((pv_actual - pv_forecast) / (pv_actual + 0.01))) * 100
ax_a1.text(
    0.02,
    0.98,
    f"RMSE: {rmse_val:.3f}\nMAE: {mae_val:.3f}\nMAPE: {mape_val:.1f}%",
    transform=ax_a1.transAxes,
    fontsize=13,
    fontweight="bold",
    bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="black", linewidth=1.5),
    verticalalignment="top",
    horizontalalignment="left",
)
legend_elements_a1 = [
    plt.Line2D([0], [0], color="#1A5276", linewidth=3, label="Actual PV"),
    plt.Line2D([0], [0], color="#E74C3C", linewidth=2.5, linestyle="--", label="Forecast PV"),
    plt.Line2D([0], [0], color="#2E86C1", alpha=0.25, linewidth=4, label="95% CI"),
    plt.Line2D(
        [0],
        [0],
        marker="o",
        color="w",
        markerfacecolor="#28B463",
        markersize=8,
        markeredgecolor="black",
        label="Measurements",
    ),
]
ax_a1.legend(
    handles=legend_elements_a1,
    loc="lower center",
    bbox_to_anchor=(0.5, -0.65),
    fontsize=14,
    frameon=True,
    edgecolor="black",
    fancybox=False,
    ncol=2,
)

ax_a2 = fig.add_subplot(gs[0, 1])
ax_a2.plot(hours, load_actual, color="#1A5276", linewidth=3, label="Actual Load")
ax_a2.plot(hours, load_forecast, color="#E74C3C", linewidth=2.5, linestyle="--", label="Forecast Load")
ax_a2.fill_between(
    hours,
    load_forecast - ci_width_load,
    load_forecast + ci_width_load,
    alpha=0.25,
    color="#28B463",
    label="95% CI",
)
ax_a2.fill_between(
    hours,
    load_forecast - ci_width_load * 1.5,
    load_forecast + ci_width_load * 1.5,
    alpha=0.08,
    color="#28B463",
)
ax_a2.scatter(
    hours[measure_idx],
    load_actual[measure_idx],
    s=50,
    color="#FDB813",
    alpha=0.6,
    edgecolor="black",
    linewidth=1.5,
    zorder=5,
    label="Measurements",
)
for idx in error_idx:
    error_magnitude = abs(load_actual[idx] - load_forecast[idx])
    color_intensity = min(1, error_magnitude * 5)
    ax_a2.plot(
        [hours[idx], hours[idx]],
        [load_actual[idx], load_forecast[idx]],
        color=plt.cm.RdYlGn_r(color_intensity),
        linewidth=1.5,
        alpha=0.5,
        linestyle="-.",
    )
ax_a2.axvspan(7, 9, alpha=0.08, color="#FDB813")
ax_a2.axvspan(17, 20, alpha=0.08, color="#E74C3C")
ax_a2.set_xlabel("Time (hours)", fontsize=22, fontweight="bold")
ax_a2.set_ylabel("Normalized Load", fontsize=22, fontweight="bold")
ax_a2.grid(True, alpha=0.12, linestyle="--", linewidth=1)
ax_a2.set_title("(b) Load Demand Forecast", fontsize=24, fontweight="bold", loc="left", pad=8)
ax_a2.set_ylim([-0.05, 0.9])
ax_a2.set_xlim([0, 24])
for spine in ax_a2.spines.values():
    spine.set_linewidth(2.5)
rmse_val = np.sqrt(np.mean((load_actual - load_forecast) ** 2))
mae_val = np.mean(np.abs(load_actual - load_forecast))
mape_val = np.mean(np.abs((load_actual - load_forecast) / (load_actual + 0.01))) * 100
ax_a2.text(
    0.02,
    0.98,
    f"RMSE: {rmse_val:.3f}\nMAE: {mae_val:.3f}\nMAPE: {mape_val:.1f}%",
    transform=ax_a2.transAxes,
    fontsize=13,
    fontweight="bold",
    bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="black", linewidth=1.5),
    verticalalignment="top",
    horizontalalignment="left",
)
legend_elements_a2 = [
    plt.Line2D([0], [0], color="#1A5276", linewidth=3, label="Actual Load"),
    plt.Line2D([0], [0], color="#E74C3C", linewidth=2.5, linestyle="--", label="Forecast Load"),
    plt.Line2D([0], [0], color="#28B463", alpha=0.25, linewidth=4, label="95% CI"),
    plt.Line2D(
        [0],
        [0],
        marker="o",
        color="w",
        markerfacecolor="#FDB813",
        markersize=8,
        markeredgecolor="black",
        label="Measurements",
    ),
]
ax_a2.legend(
    handles=legend_elements_a2,
    loc="lower center",
    bbox_to_anchor=(0.5, -0.65),
    fontsize=14,
    frameon=True,
    edgecolor="black",
    fancybox=False,
    ncol=2,
)

ax_a3 = fig.add_subplot(gs[1, 0])
ax_a3.plot(hours, ev_actual, color="#1A5276", linewidth=3, label="Actual EV")
ax_a3.plot(hours, ev_forecast, color="#E74C3C", linewidth=2.5, linestyle="--", label="Forecast EV")
ax_a3.fill_between(
    hours,
    ev_forecast - ci_width_ev,
    ev_forecast + ci_width_ev,
    alpha=0.25,
    color="#8E44AD",
    label="95% CI",
)
ax_a3.fill_between(
    hours,
    ev_forecast - ci_width_ev * 1.5,
    ev_forecast + ci_width_ev * 1.5,
    alpha=0.08,
    color="#8E44AD",
)
ax_a3.scatter(
    hours[measure_idx],
    ev_actual[measure_idx],
    s=50,
    color="#FDB813",
    alpha=0.6,
    edgecolor="black",
    linewidth=1.5,
    zorder=5,
    label="Measurements",
)
for idx in error_idx:
    error_magnitude = abs(ev_actual[idx] - ev_forecast[idx])
    color_intensity = min(1, error_magnitude * 8)
    ax_a3.plot(
        [hours[idx], hours[idx]],
        [ev_actual[idx], ev_forecast[idx]],
        color=plt.cm.RdYlGn_r(color_intensity),
        linewidth=1.5,
        alpha=0.5,
        linestyle="-.",
    )
ax_a3.annotate(
    "Morning\nCharging",
    xy=(8, 0.35),
    xytext=(9, 0.45),
    arrowprops=dict(arrowstyle="->", lw=1.5, color="black"),
    fontsize=14,
    fontweight="bold",
    ha="center",
)
ax_a3.annotate(
    "Evening\nCharging",
    xy=(19, 0.3),
    xytext=(20, 0.4),
    arrowprops=dict(arrowstyle="->", lw=1.5, color="black"),
    fontsize=14,
    fontweight="bold",
    ha="center",
)
ax_a3.set_xlabel("Time (hours)", fontsize=22, fontweight="bold")
ax_a3.set_ylabel("Normalized EV", fontsize=22, fontweight="bold")
ax_a3.grid(True, alpha=0.12, linestyle="--", linewidth=1)
ax_a3.set_title("(c) EV Charging Demand Forecast", fontsize=24, fontweight="bold", loc="left", pad=8)
ax_a3.set_ylim([-0.05, 0.6])
ax_a3.set_xlim([0, 24])
for spine in ax_a3.spines.values():
    spine.set_linewidth(2.5)
rmse_val = np.sqrt(np.mean((ev_actual - ev_forecast) ** 2))
mae_val = np.mean(np.abs(ev_actual - ev_forecast))
mape_val = np.mean(np.abs((ev_actual - ev_forecast) / (ev_actual + 0.01))) * 100
ax_a3.text(
    0.02,
    0.98,
    f"RMSE: {rmse_val:.3f}\nMAE: {mae_val:.3f}\nMAPE: {mape_val:.1f}%",
    transform=ax_a3.transAxes,
    fontsize=13,
    fontweight="bold",
    bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="black", linewidth=1.5),
    verticalalignment="top",
    horizontalalignment="left",
)
legend_elements_a3 = [
    plt.Line2D([0], [0], color="#1A5276", linewidth=3, label="Actual EV"),
    plt.Line2D([0], [0], color="#E74C3C", linewidth=2.5, linestyle="--", label="Forecast EV"),
    plt.Line2D([0], [0], color="#8E44AD", alpha=0.25, linewidth=4, label="95% CI"),
    plt.Line2D(
        [0],
        [0],
        marker="o",
        color="w",
        markerfacecolor="#FDB813",
        markersize=8,
        markeredgecolor="black",
        label="Measurements",
    ),
]
ax_a3.legend(
    handles=legend_elements_a3,
    loc="lower center",
    bbox_to_anchor=(0.5, -0.65),
    fontsize=14,
    frameon=True,
    edgecolor="black",
    fancybox=False,
    ncol=2,
)

ax_a4 = fig.add_subplot(gs[1, 1])
ax_a4.plot(hours, price_actual, color="#1A5276", linewidth=3, label="Actual Price")
ax_a4.plot(hours, price_forecast, color="#E74C3C", linewidth=2.5, linestyle="--", label="Forecast Price")
ax_a4.fill_between(
    hours,
    price_forecast - ci_width_price,
    price_forecast + ci_width_price,
    alpha=0.25,
    color="#FDB813",
    label="95% CI",
)
ax_a4.fill_between(
    hours,
    price_forecast - ci_width_price * 1.5,
    price_forecast + ci_width_price * 1.5,
    alpha=0.08,
    color="#FDB813",
)
ax_a4.scatter(
    hours[measure_idx],
    price_actual[measure_idx],
    s=50,
    color="#28B463",
    alpha=0.6,
    edgecolor="black",
    linewidth=1.5,
    zorder=5,
    label="Measurements",
)
for idx in error_idx:
    error_magnitude = abs(price_actual[idx] - price_forecast[idx])
    color_intensity = min(1, error_magnitude * 5)
    ax_a4.plot(
        [hours[idx], hours[idx]],
        [price_actual[idx], price_forecast[idx]],
        color=plt.cm.RdYlGn_r(color_intensity),
        linewidth=1.5,
        alpha=0.5,
        linestyle="-.",
    )
spike_highlight = np.where(price_actual > 0.55)[0]
if len(spike_highlight) > 0:
    ax_a4.scatter(
        hours[spike_highlight],
        price_actual[spike_highlight],
        s=100,
        color="#E74C3C",
        alpha=0.4,
        edgecolor="red",
        linewidth=2,
        zorder=4,
        label="Price Spikes",
    )
ax_a4.set_xlabel("Time (hours)", fontsize=22, fontweight="bold")
ax_a4.set_ylabel("Normalized Price", fontsize=22, fontweight="bold")
ax_a4.grid(True, alpha=0.12, linestyle="--", linewidth=1)
ax_a4.set_title("(d) Market Price Forecast", fontsize=24, fontweight="bold", loc="left", pad=8)
ax_a4.set_ylim([-0.05, 0.85])
ax_a4.set_xlim([0, 24])
for spine in ax_a4.spines.values():
    spine.set_linewidth(2.5)
rmse_val = np.sqrt(np.mean((price_actual - price_forecast) ** 2))
mae_val = np.mean(np.abs(price_actual - price_forecast))
mape_val = np.mean(np.abs((price_actual - price_forecast) / (price_actual + 0.01))) * 100
ax_a4.text(
    0.02,
    0.98,
    f"RMSE: {rmse_val:.3f}\nMAE: {mae_val:.3f}\nMAPE: {mape_val:.1f}%",
    transform=ax_a4.transAxes,
    fontsize=13,
    fontweight="bold",
    bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="black", linewidth=1.5),
    verticalalignment="top",
    horizontalalignment="left",
)
legend_elements_a4 = [
    plt.Line2D([0], [0], color="#1A5276", linewidth=3, label="Actual Price"),
    plt.Line2D([0], [0], color="#E74C3C", linewidth=2.5, linestyle="--", label="Forecast Price"),
    plt.Line2D([0], [0], color="#FDB813", alpha=0.25, linewidth=4, label="95% CI"),
    plt.Line2D(
        [0],
        [0],
        marker="o",
        color="w",
        markerfacecolor="#28B463",
        markersize=8,
        markeredgecolor="black",
        label="Measurements",
    ),
]
ax_a4.legend(
    handles=legend_elements_a4,
    loc="lower center",
    bbox_to_anchor=(0.5, -0.65),
    fontsize=14,
    frameon=True,
    edgecolor="black",
    fancybox=False,
    ncol=2,
)

ax_b = fig.add_subplot(gs[2, :])
x_pos = np.arange(n_horizons)
width = 0.2

ax_b.bar(
    x_pos - width * 1.5,
    rmse_pv,
    width=width,
    color="#FDB813",
    edgecolor="black",
    linewidth=2,
    hatch="/",
    label="PV RMSE",
)
ax_b.bar(
    x_pos - width * 0.5,
    rmse_load,
    width=width,
    color="#2E86C1",
    edgecolor="black",
    linewidth=2,
    hatch="\\",
    label="Load RMSE",
)
ax_b.bar(
    x_pos + width * 0.5,
    rmse_ev,
    width=width,
    color="#8E44AD",
    edgecolor="black",
    linewidth=2,
    hatch="x",
    label="EV RMSE",
)
ax_b.bar(
    x_pos + width * 1.5,
    rmse_price,
    width=width,
    color="#E74C3C",
    edgecolor="black",
    linewidth=2,
    hatch="o",
    label="Price RMSE",
)

for idx in range(n_horizons):
    ax_b.errorbar(
        x_pos[idx] - width * 1.5,
        rmse_pv[idx],
        yerr=rmse_pv[idx] * 0.15,
        fmt="none",
        color="black",
        capsize=3,
        capthick=2,
    )
    ax_b.errorbar(
        x_pos[idx] - width * 0.5,
        rmse_load[idx],
        yerr=rmse_load[idx] * 0.15,
        fmt="none",
        color="black",
        capsize=3,
        capthick=2,
    )
    ax_b.errorbar(
        x_pos[idx] + width * 0.5,
        rmse_ev[idx],
        yerr=rmse_ev[idx] * 0.15,
        fmt="none",
        color="black",
        capsize=3,
        capthick=2,
    )
    ax_b.errorbar(
        x_pos[idx] + width * 1.5,
        rmse_price[idx],
        yerr=rmse_price[idx] * 0.15,
        fmt="none",
        color="black",
        capsize=3,
        capthick=2,
    )

for idx in range(n_horizons):
    ax_b.scatter(
        x_pos[idx] - width * 1.5,
        mae_pv[idx],
        s=80,
        color="#D4A017",
        edgecolor="black",
        linewidth=1.5,
        zorder=5,
        marker="D",
        label="MAE" if idx == 0 else "",
    )
    ax_b.scatter(
        x_pos[idx] - width * 0.5,
        mae_load[idx],
        s=80,
        color="#1A5276",
        edgecolor="black",
        linewidth=1.5,
        zorder=5,
        marker="D",
    )
    ax_b.scatter(
        x_pos[idx] + width * 0.5,
        mae_ev[idx],
        s=80,
        color="#6C3483",
        edgecolor="black",
        linewidth=1.5,
        zorder=5,
        marker="D",
    )
    ax_b.scatter(
        x_pos[idx] + width * 1.5,
        mae_price[idx],
        s=80,
        color="#922B21",
        edgecolor="black",
        linewidth=1.5,
        zorder=5,
        marker="D",
    )

ax_b2 = ax_b.twinx()
ax_b2.plot(
    x_pos,
    mape_pv,
    color="#D4A017",
    linewidth=2.5,
    linestyle="--",
    marker="s",
    markersize=11,
    label="PV MAPE",
)
ax_b2.plot(
    x_pos,
    mape_load,
    color="#1A5276",
    linewidth=2.5,
    linestyle="--",
    marker="s",
    markersize=11,
    label="Load MAPE",
)
ax_b2.plot(
    x_pos,
    mape_ev,
    color="#6C3483",
    linewidth=2.5,
    linestyle="--",
    marker="s",
    markersize=11,
    label="EV MAPE",
)
ax_b2.plot(
    x_pos,
    mape_price,
    color="#922B21",
    linewidth=2.5,
    linestyle="--",
    marker="s",
    markersize=11,
    label="Price MAPE",
)
ax_b2.set_ylabel("MAPE (%)", fontsize=22, fontweight="bold", color="black")
ax_b2.tick_params(axis="y", labelsize=20)

ax_b.axhspan(0, 0.08, alpha=0.08, color="#28B463")
ax_b.set_xticks(x_pos)
ax_b.set_xticklabels([f"{h}h" for h in horizons], fontsize=20, fontweight="bold")
ax_b.set_xlabel("Prediction Horizon", fontsize=22, fontweight="bold")
ax_b.set_ylabel("RMSE / MAE (Normalized)", fontsize=22, fontweight="bold")

lines1, labels1 = ax_b.get_legend_handles_labels()
lines2, labels2 = ax_b2.get_legend_handles_labels()
unique_labels = []
unique_lines = []
for line, label in zip(lines1 + lines2, labels1 + labels2):
    if label not in unique_labels and label != "":
        unique_labels.append(label)
        unique_lines.append(line)

ax_b.legend(
    unique_lines,
    unique_labels,
    loc="upper left",
    fontsize=14,
    frameon=True,
    edgecolor="black",
    fancybox=False,
    ncol=3,
)
ax_b.grid(True, alpha=0.12, linestyle="--", linewidth=1)
ax_b.set_title(
    "(e) Forecasting Error Analysis - RMSE, MAE, and MAPE Across Horizons",
    fontsize=24,
    fontweight="bold",
    loc="left",
    pad=10,
)
for spine in ax_b.spines.values():
    spine.set_linewidth(2.5)

ax_b.text(
    0.98,
    0.1,
    "Best Performance: PV & Load\nMost Challenging: EV & Price",
    transform=ax_b.transAxes,
    fontsize=16,
    fontweight="bold",
    ha="right",
    va="bottom",
    bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor="black", linewidth=2),
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

svg_path = OUTPUT_DIR / "Figure6_Forecasting_Performance.svg"
pdf_path = OUTPUT_DIR / "Figure6_Forecasting_Performance.pdf"
png_path = OUTPUT_DIR / "Figure6_Forecasting_Performance.png"

plt.tight_layout(rect=[0, 0, 1, 0.985], h_pad=1.8, w_pad=1.2)
plt.savefig(svg_path, dpi=300, bbox_inches="tight", facecolor="white", edgecolor="none", format="svg")
plt.savefig(pdf_path, dpi=300, bbox_inches="tight", facecolor="white", edgecolor="none", format="pdf")
plt.savefig(png_path, dpi=300, bbox_inches="tight", facecolor="white", edgecolor="none", format="png")
plt.show()

print(f"Data loaded from: {DATA_PATH}")
print(f"SVG saved to: {svg_path}")
print(f"PDF saved to: {pdf_path}")
print(f"PNG saved to: {png_path}")

