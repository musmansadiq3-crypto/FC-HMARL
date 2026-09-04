from pathlib import Path as FilePath

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle


FONT_FAMILY = 'Times New Roman'
FONT_WEIGHT = 'bold'
GLOBAL_FONT_SIZE = 26           

FIG_WIDTH = 20           
FIG_HEIGHT = 24          
SUBPLOT_HSPACE = 0.43             # Vertical spacing between the three panels
X_AXIS_PAD_MULTIPLIER = 1.15      # X-axis upper limit padding (115% of max value)

TITLE_FONT_SIZE = 26
TITLE_PAD = 20
AXIS_LABEL_FONT_SIZE = 26
TICK_LABEL_FONT_SIZE = 26
BAR_LABEL_FONT_SIZE = 18          # Size of the labels placed outside the bars
LEGEND_FONT_SIZE = 22
TITLE_COLOR = '#1B4F72'

SPINE_LINEWIDTH = 1.5
SPINE_COLOR = 'black'
GRID_LINEWIDTH = 1.5
BAR_EDGE_LINEWIDTH = 2.0
PANEL_BORDER_LINEWIDTH = 2.0      # Dashed boundary box line width
LEGEND_FRAME_LINEWIDTH = 0.5

TICK_MAJOR_SIZE = 10    
TICK_MAJOR_WIDTH = 2.5  
TICK_DIRECTION = 'in'

BAR_HEIGHT_AB = 0.18              # Thickness of bars in panels A and B
BAR_HEIGHT_C = 0.15               # Thickness of bars in panel C
BAR_SHADOW_OFFSET = 0.025         # Amount of shift for the 3D shadow effect
BAR_LABEL_OFFSET_REL = 0.015      # Relative space between bar end and text label
BAR_LABEL_OFFSET_ABS = 0.5        # Absolute space added for the shadow text label

COLORS_MAIN = ['#1F618D', '#1D8348', '#B7950B', '#A93226']
COLORS_3D_GRADIENT = [
    ['#2980B9', '#1F618D', '#154360'],  # Blue gradient
    ['#27AE60', '#1D8348', '#145A32'],  # Green gradient
    ['#F1C40F', '#B7950B', '#7D6608'],  # Gold gradient
    ['#E74C3C', '#A93226', '#641E16']   # Red gradient
]
HATCH_PATTERNS = ['///', '\\\\\\', 'xxx', '...']

PANEL_A_BORDER_COLOR = '#2980B9'
PANEL_B_BORDER_COLOR = '#27AE60'
PANEL_C_CMAP = 'plasma'
PANEL_C_CMAP_START = 0.0
PANEL_C_CMAP_END = 0.7

PANEL_C_ZONES = [
    (30, 60, '#FF9999'),   # Fair Zone
    (60, 85, '#FFCC99'),   # Good Zone
    (85, 110, '#99FF99')   # Excellent Zone
]

LEGEND_BBOX_AB = (0.5, -0.16)     # Position of legend below panels A and B
LEGEND_BBOX_C = (0.5, -0.16)      # Position of legend below panel C


plt.rcParams['font.family'] = FONT_FAMILY
plt.rcParams['font.size'] = GLOBAL_FONT_SIZE
plt.rcParams['font.weight'] = FONT_WEIGHT
plt.rcParams['axes.linewidth'] = SPINE_LINEWIDTH
plt.rcParams['axes.edgecolor'] = SPINE_COLOR
plt.rcParams['axes.titlepad'] = TITLE_PAD       
plt.rcParams['xtick.labelsize'] = TICK_LABEL_FONT_SIZE     
plt.rcParams['ytick.labelsize'] = TICK_LABEL_FONT_SIZE     
plt.rcParams['xtick.major.size'] = TICK_MAJOR_SIZE    
plt.rcParams['ytick.major.size'] = TICK_MAJOR_SIZE    
plt.rcParams['xtick.major.width'] = TICK_MAJOR_WIDTH  
plt.rcParams['ytick.major.width'] = TICK_MAJOR_WIDTH
plt.rcParams['xtick.direction'] = TICK_DIRECTION
plt.rcParams['ytick.direction'] = TICK_DIRECTION


ROOT = FilePath(__file__).resolve().parent
DATA_PATH = ROOT / "evaluation" / "data" / "case_studies" / "vpp_case_study_comparison_data.xlsx"
OUTPUT_DIR = ROOT / "outputs" / "figures"
file_path = DATA_PATH

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Data file not found:\n{DATA_PATH}\n\n"
        "Place the Excel workbook at:\n"
        r"evaluation\data\case_studies\vpp_case_study_comparison_data.xlsx"
    )

df_a = pd.read_excel(DATA_PATH, sheet_name="Economic", index_col=0)
df_b = pd.read_excel(DATA_PATH, sheet_name="Efficiency", index_col=0)
df_c = pd.read_excel(DATA_PATH, sheet_name="Aggregate_Scores", index_col=0)

if df_a.empty or df_b.empty or df_c.empty:
    raise ValueError("One or more required sheets are empty.")

case_studies = df_a.index.astype(str).tolist()
n_cases = len(case_studies)

if df_b.index.astype(str).tolist() != case_studies:
    raise ValueError("Case-study rows in Efficiency do not match Economic.")

if df_c.index.astype(str).tolist() != case_studies:
    raise ValueError("Case-study rows in Aggregate_Scores do not match Economic.")

for frame_name, frame in [
    ("Economic", df_a),
    ("Efficiency", df_b),
    ("Aggregate_Scores", df_c),
]:
    for column in frame.columns:
        frame[column] = pd.to_numeric(frame[column], errors="raise")

fig = plt.figure(figsize=(FIG_WIDTH, FIG_HEIGHT))

gs = gridspec.GridSpec(3, 1, figure=fig, 
                       height_ratios=[1, 1, 1.2], 
                       hspace=SUBPLOT_HSPACE)


def draw_3d_bars_horizontal(ax, df_data, use_percentage=False, panel_type='A'):
    metrics = df_data.columns.tolist()
    n_metrics = len(metrics)
    y_pos = np.arange(n_cases)
    bar_h = BAR_HEIGHT_AB
    
    max_val = df_data.max().max()
    if max_val == 0: max_val = 1
    
    for i, metric in enumerate(metrics):
        offset = (i - (n_metrics-1)/2) * bar_h
        pos = y_pos + offset
        vals = df_data[metric].values
        
        ax.barh(pos + BAR_SHADOW_OFFSET, vals, height=bar_h, 
                color='black', alpha=0.25, zorder=1, linewidth=0)
        ax.barh(pos + BAR_SHADOW_OFFSET*0.6, vals, height=bar_h, 
                color=COLORS_MAIN[i % len(COLORS_MAIN)], alpha=0.15, zorder=1, linewidth=0)
        
        main_color = COLORS_3D_GRADIENT[i % len(COLORS_3D_GRADIENT)]
        
        ax.barh(pos - bar_h*0.1, vals, height=bar_h*0.15, 
                color=main_color[0], edgecolor='none', alpha=0.4, zorder=2)
        
        bars = ax.barh(pos, vals, height=bar_h*0.7, 
                       color=main_color[1], edgecolor='black', 
                       linewidth=BAR_EDGE_LINEWIDTH, hatch=HATCH_PATTERNS[i % len(HATCH_PATTERNS)], 
                       alpha=0.95, zorder=3, label=metric)
        
        ax.barh(pos + bar_h*0.35, vals, height=bar_h*0.15, 
                color=main_color[2], edgecolor='none', alpha=0.5, zorder=2)
        
        for bar in bars:
            w = bar.get_width()
            y = bar.get_y()
            h = bar.get_height()
            
            ax.plot([w, w], [y, y + h], color='white', linewidth=1.5, alpha=0.3, zorder=4)
            ax.plot([0, 0], [y, y + h], color='black', linewidth=1, alpha=0.2, zorder=4)
        
        for bar in bars:
            w = bar.get_width()
            label_text = f'{w:.1f}'
            ax.text(w + (max_val * BAR_LABEL_OFFSET_REL) + BAR_LABEL_OFFSET_ABS, bar.get_y() + bar.get_height()/2 - 0.3, 
                    label_text, ha='left', va='center', fontsize=BAR_LABEL_FONT_SIZE, fontweight=FONT_WEIGHT,
                    color='white', alpha=0.5, zorder=5)
            ax.text(w + (max_val * BAR_LABEL_OFFSET_REL), bar.get_y() + bar.get_height()/2, 
                    label_text, ha='left', va='center', fontsize=BAR_LABEL_FONT_SIZE, fontweight=FONT_WEIGHT,
                    color='black', zorder=6)

    ax.set_xlim(0, max_val * X_AXIS_PAD_MULTIPLIER)
    ax.grid(axis='x', linestyle=':', alpha=0.5, linewidth=GRID_LINEWIDTH)
    
    ax.set_facecolor('#F8F9FA')
    
    for spine in ax.spines.values():
        spine.set_linewidth(SPINE_LINEWIDTH)
        spine.set_alpha(0.8)


ax_a = fig.add_subplot(gs[0])
draw_3d_bars_horizontal(ax_a, df_a, panel_type='A')
ax_a.set_title('(a) Comparative Economic Performance Metrics', loc='left', fontsize=TITLE_FONT_SIZE, fontweight=FONT_WEIGHT, color=TITLE_COLOR)
ax_a.set_yticks(np.arange(n_cases))
ax_a.set_yticklabels(case_studies, fontsize=TICK_LABEL_FONT_SIZE, fontweight=FONT_WEIGHT)
ax_a.set_xlabel('Performance Indicators', fontsize=AXIS_LABEL_FONT_SIZE, fontweight=FONT_WEIGHT, color=TITLE_COLOR)

rect_a = Rectangle((0, 0), 1, 1, transform=ax_a.transAxes, 
                   facecolor='none', edgecolor=PANEL_A_BORDER_COLOR, 
                   linewidth=PANEL_BORDER_LINEWIDTH, alpha=0.3, linestyle='--')
ax_a.add_patch(rect_a)

legend_a = ax_a.legend(ncol=4, fontsize=LEGEND_FONT_SIZE, loc='upper center', bbox_to_anchor=LEGEND_BBOX_AB, 
                       frameon=True, edgecolor='black', columnspacing=2.0, 
                       shadow=True, fancybox=True, facecolor='white')
legend_a.get_frame().set_linewidth(LEGEND_FRAME_LINEWIDTH)


ax_b = fig.add_subplot(gs[1])
draw_3d_bars_horizontal(ax_b, df_b, panel_type='B')
ax_b.set_title('(b) Operational Efficiency Indices', loc='left', fontsize=TITLE_FONT_SIZE, fontweight=FONT_WEIGHT, color=TITLE_COLOR)
ax_b.set_yticks(np.arange(n_cases))
ax_b.set_yticklabels(case_studies, fontsize=TICK_LABEL_FONT_SIZE, fontweight=FONT_WEIGHT)
ax_b.set_xlabel('Percentage (%)', fontsize=AXIS_LABEL_FONT_SIZE, fontweight=FONT_WEIGHT, color=TITLE_COLOR)

rect_b = Rectangle((0, 0), 1, 1, transform=ax_b.transAxes, 
                   facecolor='none', edgecolor=PANEL_B_BORDER_COLOR, 
                   linewidth=PANEL_BORDER_LINEWIDTH, alpha=0.3, linestyle='--')
ax_b.add_patch(rect_b)

legend_b = ax_b.legend(ncol=4, fontsize=LEGEND_FONT_SIZE, loc='upper center', bbox_to_anchor=LEGEND_BBOX_AB, 
                       frameon=True, edgecolor='black', columnspacing=2.0,
                       shadow=True, fancybox=True, facecolor='white')
legend_b.get_frame().set_linewidth(LEGEND_FRAME_LINEWIDTH)


ax_c = fig.add_subplot(gs[2])
metrics_c = df_c.columns.tolist()[::-1] 
y_pos_c = np.arange(len(metrics_c))
bar_h_c = BAR_HEIGHT_C

case_colors = plt.get_cmap(PANEL_C_CMAP)(np.linspace(PANEL_C_CMAP_START, PANEL_C_CMAP_END, n_cases))

for i, case in enumerate(case_studies):
    offset = (i - (n_cases-1)/2) * bar_h_c
    vals = [df_c.loc[case, m] for m in metrics_c]
    
    ax_c.barh(y_pos_c + offset - 0.005, vals, height=bar_h_c, color='black', alpha=0.15, zorder=1)
    
    ax_c.barh(y_pos_c + offset, vals, height=bar_h_c, color=case_colors[i], 
                     edgecolor='black', linewidth=BAR_EDGE_LINEWIDTH, hatch=HATCH_PATTERNS[i%4], 
                     label=case, zorder=2)
    
    for j, v in enumerate(vals):
        ax_c.text(v + 2.5, y_pos_c[j] + offset, f'{v:.1f}', va='center', 
                  ha='left', fontsize=BAR_LABEL_FONT_SIZE, fontweight=FONT_WEIGHT)

ax_c.set_title('(c) Performance Ranking and Aggregate Scoring Across Case Studies', 
               loc='left', fontsize=TITLE_FONT_SIZE, fontweight=FONT_WEIGHT, color=TITLE_COLOR)
ax_c.set_yticks(y_pos_c)
ax_c.set_yticklabels(metrics_c, fontsize=TICK_LABEL_FONT_SIZE, fontweight=FONT_WEIGHT)
ax_c.set_xlabel('Weighted Performance Score (%)', fontsize=AXIS_LABEL_FONT_SIZE, fontweight=FONT_WEIGHT, color=TITLE_COLOR)
ax_c.set_xlim(30, 110)
ax_c.grid(axis='x', linestyle=':', alpha=0.5, linewidth=GRID_LINEWIDTH)

for start_x, end_x, zone_color in PANEL_C_ZONES:
    ax_c.axvspan(start_x, end_x, color=zone_color, alpha=0.1)

legend_c = ax_c.legend(loc='upper center', bbox_to_anchor=LEGEND_BBOX_C, ncol=5, 
                       fontsize=LEGEND_FONT_SIZE, frameon=True, edgecolor='black', columnspacing=1.5,
                       shadow=True, fancybox=True, facecolor='white')
legend_c.get_frame().set_linewidth(LEGEND_FRAME_LINEWIDTH)


OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
output_dir = OUTPUT_DIR

svg_path = output_dir / "VPP_Case_Study_Comparison.svg"
pdf_path = output_dir / "VPP_Case_Study_Comparison.pdf"
png_path = output_dir / "VPP_Case_Study_Comparison.png"
0
plt.savefig(svg_path, dpi=300, bbox_inches='tight', format='svg')
plt.savefig(pdf_path, dpi=300, bbox_inches='tight', format='pdf')
plt.savefig(png_path, dpi=300, bbox_inches='tight', format='png')

print(f"Success: Graph saved as SVG at '{svg_path}'")
print(f"Success: Graph saved as PDF at '{pdf_path}'")
print(f"Success: Graph saved as PNG at '{png_path}'")

plt.show()