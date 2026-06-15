"""Generate output/figures/glm_coef_plot.png — single-panel horizontal coefficient plot."""
import pathlib
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

plt.style.use("seaborn-v0_8-paper")
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman", "DejaVu Serif"],
        "mathtext.fontset": "cm",
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.dpi": 150,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.35,
        "grid.color": "#cccccc",
        "text.color": "black",
        "axes.edgecolor": "black",
        "axes.labelcolor": "black",
        "xtick.color": "black",
        "ytick.color": "black",
    }
)

ROOT = pathlib.Path(__file__).parent.parent
csv_path = ROOT / "output" / "glm_comparison_v2.csv"
out_path = ROOT / "output" / "figures" / "glm_coef_plot.png"

LABEL_MAP = {
    "cage_c":      "Vehicle age",
    "commercial":  "Commercial use",
    "credit_c":    "Credit score (per 100 pts)",
    "farmer":      "Farmer use",
    "log_accel_c": "Hard accel. (log)",
    "log_brake_c": "Hard braking (log)",
    "log_miles_c": "Log annual miles",
    "male":        "Male",
    "pct_c":       "Annual pct. driven",
    "senior":      "Senior driver (>65)",
    "single":      "Single",
    "young":       "Young driver (<25)",
}

SBV_COLOR  = "#aaaaaa"
SYN_COLOR  = "#333333"
LINE_COLOR = "#cccccc"

df = pd.read_csv(csv_path)
df = df[df["term"] != "Intercept"].copy()
df["label"] = df["term"].map(LABEL_MAP)
df = df.sort_values("coef_sbv").reset_index(drop=True)

fig, ax = plt.subplots(figsize=(8, 5.5))

for i, row in df.iterrows():
    ax.plot(
        [row["coef_sbv"], row["coef_synthdrive"]], [i, i],
        color=LINE_COLOR, lw=1.0, zorder=1,
    )
    ax.plot(
        row["coef_sbv"], i,
        marker="o", markersize=7,
        markerfacecolor="white", markeredgecolor=SBV_COLOR, markeredgewidth=1.5,
        linestyle="none", zorder=2,
    )
    ax.plot(
        row["coef_synthdrive"], i,
        marker="o", markersize=7,
        markerfacecolor=SYN_COLOR, markeredgecolor=SYN_COLOR,
        linestyle="none", zorder=3,
    )

ax.set_xlim(-1.0, 3.0)
ax.set_yticks(range(len(df)))
ax.set_yticklabels(df["label"].tolist(), fontsize=9)
ax.tick_params(axis="y", length=0)
ax.yaxis.set_tick_params(pad=4)
ax.set_xlabel("Poisson GLM coefficient", fontsize=11)
ax.set_ylabel("")

legend_handles = [
    mlines.Line2D([], [], marker="o", markersize=7,
                  markerfacecolor="white", markeredgecolor=SBV_COLOR,
                  markeredgewidth=1.5, linestyle="none",
                  color=SBV_COLOR, label="SBV seed"),
    mlines.Line2D([], [], marker="o", markersize=7,
                  markerfacecolor=SYN_COLOR, markeredgecolor=SYN_COLOR,
                  linestyle="none", color=SYN_COLOR, label="SynthDrive"),
]
ax.legend(handles=legend_handles, loc="lower right", frameon=True,
          framealpha=0.9, edgecolor="#cccccc")

out_path.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"Saved: {out_path}")
