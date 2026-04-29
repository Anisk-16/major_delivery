"""
paper_figures_all.py
====================
Generates all publication-quality figures for the Results section:

  Fig 1 - reward_curve.png           (already exists, regenerated at 300 DPI)
  Fig 2 - orders_served_curve.png    (already exists, regenerated at 300 DPI)
  Fig 3 - ablation_bar.png           NEW: ablation study bar chart
  Fig 4 - std_reduction.png          NEW: reward & orders std reduction over training
  Fig 5 - tradeoff_scatter.png       NEW: on-time rate vs orders served trade-off
  Fig 6 - metric_comparison.png      NEW: 500k vs 1M side-by-side comparison
"""

import os, sys, math
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR  = os.path.join(BASE_DIR, "..", "paper_figures")
os.makedirs(OUT_DIR, exist_ok=True)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

DPI        = 300
BG_DARK    = "#0d1117"
BG_PANEL   = "#161b22"
GRID_COLOR = "#21262d"
SPINE_COL  = "#30363d"
WHITE      = "#e6edf3"

CYAN   = "#00e5ff"
GREEN  = "#b3f542"
CORAL  = "#ff6b6b"
GOLD   = "#ffd60a"
PURPLE = "#c084fc"
BLUE   = "#58a6ff"

def style_ax(ax, title="", xlabel="", ylabel=""):
    ax.set_facecolor(BG_PANEL)
    ax.tick_params(colors=WHITE, labelsize=9)
    ax.spines[:].set_color(SPINE_COL)
    ax.grid(True, color=GRID_COLOR, linewidth=0.7, linestyle="--", alpha=0.6)
    if title:  ax.set_title(title,  color=WHITE, fontsize=11, pad=10, fontweight="bold")
    if xlabel: ax.set_xlabel(xlabel, color=WHITE, fontsize=9)
    if ylabel: ax.set_ylabel(ylabel, color=WHITE, fontsize=9)

def save(fig, name):
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  [OK] {name}")

# ── load training data ─────────────────────────────────────────────────────────
data = np.load(os.path.join(BASE_DIR, "logs", "evaluations.npz"))
ts   = data["timesteps"].astype(float)
res  = data["results"]      # (n_ckpt, 10)
epl  = data["ep_lengths"]   # (n_ckpt, 10)

m_rew = res.mean(axis=1);  s_rew = res.std(axis=1)
m_len = epl.mean(axis=1);  s_len = epl.std(axis=1)

# add 1M live-eval point manually
ts_all   = np.append(ts,   1_000_000)
m_rew_all = np.append(m_rew, -858.78);  s_rew_all = np.append(s_rew, 20.16)
m_len_all = np.append(m_len,   13.8);   s_len_all = np.append(s_len,  0.4)

# ══════════════════════════════════════════════════════════════════════════════
# FIG 1  Reward Curve  (300 DPI redo)
# ══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(8, 4.5))
fig.patch.set_facecolor(BG_DARK)
style_ax(ax, "Fig 1 — PPO Training Reward Curve",
         "Training Timesteps (×1000)", "Mean Episode Reward")

ax.plot(ts_all/1000, m_rew_all, color=CYAN, lw=2.5, marker="o", ms=6, label="Mean Eval Reward")
ax.fill_between(ts_all/1000, m_rew_all-s_rew_all, m_rew_all+s_rew_all,
                alpha=0.18, color=CYAN, label="±1 Std Dev")

best_i = int(np.argmax(m_rew_all))
ax.annotate(f"Best: {m_rew_all[best_i]:.1f}",
            xy=(ts_all[best_i]/1000, m_rew_all[best_i]),
            xytext=(ts_all[best_i]/1000 + 60, m_rew_all[best_i] + 40),
            color=GREEN, fontsize=8.5,
            arrowprops=dict(arrowstyle="->", color=GREEN, lw=1.2))

ax.legend(facecolor=BG_PANEL, edgecolor=SPINE_COL, labelcolor=WHITE, fontsize=9)
plt.tight_layout()
save(fig, "reward_curve.png")

# ══════════════════════════════════════════════════════════════════════════════
# FIG 2  Orders Served Curve  (300 DPI redo)
# ══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(8, 4.5))
fig.patch.set_facecolor(BG_DARK)
style_ax(ax, "Fig 2 — Orders Served per Episode During Training",
         "Training Timesteps (×1000)", "Orders Served per Episode")

ax.plot(ts_all/1000, m_len_all, color=GREEN, lw=2.5, marker="s", ms=6, label="Avg Orders Served")
ax.fill_between(ts_all/1000, m_len_all-s_len_all, m_len_all+s_len_all,
                alpha=0.2, color=GREEN, label="±1 Std Dev")
ax.axhline(y=20, color=CORAL, lw=1.2, ls="--", label="Max possible (20 orders)")

ax.legend(facecolor=BG_PANEL, edgecolor=SPINE_COL, labelcolor=WHITE, fontsize=9)
plt.tight_layout()
save(fig, "orders_served_curve.png")

# ══════════════════════════════════════════════════════════════════════════════
# FIG 3  Ablation Study Bar Chart
# ══════════════════════════════════════════════════════════════════════════════
configs = [
    "Baseline PPO\n(no penalty)",
    "+ Unserved\nPenalty (P=25)",
    "+ Unserved\nPenalty (P=50)",
    "+ Speed Fix\n40 km/h, 1M steps",
]
orders_served = [5.3, 9.0, 10.0, 13.8]
on_time       = [83.3, 77.8, 80.0, 73.8]
colors_bar    = [CORAL, GOLD, PURPLE, GREEN]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))
fig.patch.set_facecolor(BG_DARK)
fig.suptitle("Fig 3 — Ablation Study: Effect of Design Decisions",
             color=WHITE, fontsize=12, fontweight="bold", y=1.01)

x = np.arange(len(configs))
bars1 = ax1.bar(x, orders_served, color=colors_bar, width=0.6, edgecolor=SPINE_COL, linewidth=0.8)
style_ax(ax1, "Orders Served per Episode", "", "Avg Orders Served / 20")
ax1.set_xticks(x); ax1.set_xticklabels(configs, fontsize=7.5, color=WHITE)
ax1.axhline(y=20, color=WHITE, lw=0.8, ls="--", alpha=0.4, label="Max (20)")
ax1.set_ylim(0, 22)
for bar, val in zip(bars1, orders_served):
    ax1.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3,
             f"{val}", ha="center", va="bottom", color=WHITE, fontsize=9, fontweight="bold")

bars2 = ax2.bar(x, on_time, color=colors_bar, width=0.6, edgecolor=SPINE_COL, linewidth=0.8)
style_ax(ax2, "On-Time Delivery Rate", "", "On-Time Rate (%)")
ax2.set_xticks(x); ax2.set_xticklabels(configs, fontsize=7.5, color=WHITE)
ax2.set_ylim(0, 100)
for bar, val in zip(bars2, on_time):
    ax2.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1.0,
             f"{val}%", ha="center", va="bottom", color=WHITE, fontsize=9, fontweight="bold")

fig.tight_layout()
save(fig, "ablation_bar.png")

# ══════════════════════════════════════════════════════════════════════════════
# FIG 4  Reward Std Dev Reduction  (convergence proof)
# ══════════════════════════════════════════════════════════════════════════════
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
fig.patch.set_facecolor(BG_DARK)
fig.suptitle("Fig 4 — Policy Convergence: Std Dev Reduction Over Training",
             color=WHITE, fontsize=12, fontweight="bold", y=1.01)

ax1.plot(ts_all/1000, s_rew_all, color=CYAN, lw=2.5, marker="o", ms=6)
ax1.fill_between(ts_all/1000, 0, s_rew_all, alpha=0.15, color=CYAN)
style_ax(ax1, "Reward Std Dev (lower = more consistent)",
         "Timesteps (×1000)", "Reward Std Dev")
ax1.annotate(f"Start: {s_rew_all[0]:.1f}", xy=(ts_all[0]/1000, s_rew_all[0]),
             xytext=(ts_all[0]/1000+60, s_rew_all[0]+3), color=CORAL, fontsize=8,
             arrowprops=dict(arrowstyle="->", color=CORAL, lw=1))
ax1.annotate(f"Final: {s_rew_all[-1]:.1f}", xy=(ts_all[-1]/1000, s_rew_all[-1]),
             xytext=(ts_all[-1]/1000-200, s_rew_all[-1]+10), color=GREEN, fontsize=8,
             arrowprops=dict(arrowstyle="->", color=GREEN, lw=1))

ax2.plot(ts_all/1000, s_len_all, color=GREEN, lw=2.5, marker="s", ms=6)
ax2.fill_between(ts_all/1000, 0, s_len_all, alpha=0.15, color=GREEN)
style_ax(ax2, "Orders Served Std Dev (lower = more stable)",
         "Timesteps (×1000)", "Orders Served Std Dev")
ax2.annotate(f"Start: {s_len_all[0]:.2f}", xy=(ts_all[0]/1000, s_len_all[0]),
             xytext=(ts_all[0]/1000+60, s_len_all[0]+0.3), color=CORAL, fontsize=8,
             arrowprops=dict(arrowstyle="->", color=CORAL, lw=1))
ax2.annotate(f"Final: {s_len_all[-1]:.1f}", xy=(ts_all[-1]/1000, s_len_all[-1]),
             xytext=(ts_all[-1]/1000-200, s_len_all[-1]+0.5), color=GREEN, fontsize=8,
             arrowprops=dict(arrowstyle="->", color=GREEN, lw=1))

fig.tight_layout()
save(fig, "std_reduction.png")

# ══════════════════════════════════════════════════════════════════════════════
# FIG 5  Trade-off Scatter: On-Time Rate vs Orders Served
# ══════════════════════════════════════════════════════════════════════════════
checkpoints = {
    "80k":   (10.9, 100.0, 1.7,  15.0),   # (orders, ontime%, std_orders, bubble_size)
    "320k":  (12.6, 80.0,  1.2,  20.0),
    "500k":  (13.0, 80.1,  8.7,  35.0),   # large bubble = high variance
    "1M":    (13.8, 73.8,  0.4,  10.0),   # small bubble = low variance
}
cols = [CORAL, GOLD, PURPLE, GREEN]
labels_ckpt = list(checkpoints.keys())

fig, ax = plt.subplots(figsize=(8, 5.5))
fig.patch.set_facecolor(BG_DARK)
style_ax(ax, "Fig 5 — Coverage-Punctuality Trade-off Across Checkpoints",
         "Orders Served per Episode", "On-Time Delivery Rate (%)")

for (label, (orders, ontime, std, bsize)), col in zip(checkpoints.items(), cols):
    ax.scatter(orders, ontime, s=bsize*30, color=col, alpha=0.85,
               edgecolors=WHITE, linewidth=0.8, zorder=5)
    ax.annotate(f"{label}\n(σ={std})", xy=(orders, ontime),
                xytext=(orders+0.1, ontime+1.5), color=col, fontsize=8.5, fontweight="bold")

ax.set_xlim(9, 15.5); ax.set_ylim(65, 110)
ax.text(13.5, 106, "Bubble size = std dev\n(smaller = more stable)",
        color=WHITE, fontsize=7.5, alpha=0.7, ha="center")

# quadrant lines
ax.axvline(x=12, color=WHITE, lw=0.6, ls=":", alpha=0.3)
ax.axhline(y=77, color=WHITE, lw=0.6, ls=":", alpha=0.3)

plt.tight_layout()
save(fig, "tradeoff_scatter.png")

# ══════════════════════════════════════════════════════════════════════════════
# FIG 6  500k vs 1M Side-by-Side Comparison Bar
# ══════════════════════════════════════════════════════════════════════════════
metrics_labels = ["Orders\nServed", "Reward\nStd Dev", "On-Time\nRate (%)",
                  "Late\nOrders", "Inference\nTime (ms)"]
vals_500k = [13.0,  21.27, 80.1, 2.6, 1.278]
vals_1m   = [13.8,  20.16, 73.8, 3.6, 1.020]

# normalize each metric to [0,1] for display (some higher=better, some lower=better)
# we'll show raw values with annotation instead
x = np.arange(len(metrics_labels))
width = 0.35

fig, ax = plt.subplots(figsize=(11, 5.5))
fig.patch.set_facecolor(BG_DARK)
style_ax(ax, "Fig 6 — 500k vs 1M Step Training: Key Metric Comparison", "", "Value")

bars_a = ax.bar(x - width/2, vals_500k, width, label="500k Steps",
                color=PURPLE, alpha=0.85, edgecolor=SPINE_COL)
bars_b = ax.bar(x + width/2, vals_1m,   width, label="1M Steps",
                color=GREEN,  alpha=0.85, edgecolor=SPINE_COL)

ax.set_xticks(x)
ax.set_xticklabels(metrics_labels, color=WHITE, fontsize=9)
ax.legend(facecolor=BG_PANEL, edgecolor=SPINE_COL, labelcolor=WHITE, fontsize=10)

# annotate bars
for bar, val in zip(bars_a, vals_500k):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3,
            f"{val}", ha="center", va="bottom", color=WHITE, fontsize=8)
for bar, val in zip(bars_b, vals_1m):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3,
            f"{val}", ha="center", va="bottom", color=WHITE, fontsize=8)

# improvement arrows & annotations
improvements = [
    (0,  "+6%", GREEN,  "higher better"),
    (1,  "-5%", GREEN,  "lower better"),
    (2,  "-8%", CORAL,  "lower (trade-off)"),
    (3,  "+38%",CORAL,  "lower better"),
    (4,  "-20%",GREEN,  "lower better"),
]
for idx, label, col, note in improvements:
    ax.text(x[idx], max(vals_500k[idx], vals_1m[idx]) + 3.5,
            label, ha="center", color=col, fontsize=8.5, fontweight="bold")

ax.set_ylim(0, 100)
plt.tight_layout()
save(fig, "metric_comparison.png")

print(f"\nAll 6 figures saved to: {OUT_DIR}")
print("\nFigure placement in paper:")
print("  Fig 1 - reward_curve.png       -> Section 4.2 (after Table 2)")
print("  Fig 2 - orders_served_curve.png-> Section 4.1 (after ablation table)")
print("  Fig 3 - ablation_bar.png       -> Section 4.1 (alongside Table 1)")
print("  Fig 4 - std_reduction.png      -> Section 4.2 (after reward curve)")
print("  Fig 5 - tradeoff_scatter.png   -> Section 4.3 (after Table 3)")
print("  Fig 6 - metric_comparison.png  -> Section 4.3 (500k vs 1M discussion)")
