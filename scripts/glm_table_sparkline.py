"""Generate output/figures/glm_table_sparkline.png — combined GLM table with dumbbell sparklines."""
import pathlib
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

plt.style.use("seaborn-v0_8-paper")
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman", "DejaVu Serif"],
    "mathtext.fontset": "cm",
    "figure.dpi": 200,
    "text.color": "black",
})

ROOT     = pathlib.Path(__file__).parent.parent
csv_path = ROOT / "output" / "glm_comparison_v2.csv"
out_path = ROOT / "output" / "figures" / "glm_table_sparkline.png"

ROW_ORDER = [
    ("pct_c",       "Annual pct. driven"),
    ("log_brake_c", "Hard braking (log)"),
    ("young",       "Young driver (<25)"),
    ("log_miles_c", "Log annual miles"),
    ("commercial",  "Commercial use"),
    ("senior",      "Senior driver (>65)"),
    ("log_accel_c", "Hard accel. (log)"),
    ("male",        "Male"),
    ("single",      "Single"),
    ("cage_c",      "Vehicle age"),
    ("credit_c",    "Credit score (per 100 pts)"),
    ("farmer",      "Farmer use"),
]
DAGGER_VARS = {"Male", "Single", "Senior driver (>65)", "Hard accel. (log)"}
SBV_COLOR   = "#aaaaaa"
SYN_COLOR   = "#333333"
LINE_COLOR  = "#cccccc"
SCALE_MIN, SCALE_MAX = -1.0, 3.0

df = pd.read_csv(csv_path).set_index("term")
rows = [
    {"label": lbl, "sbv": df.loc[t, "coef_sbv"], "syn": df.loc[t, "coef_synthdrive"]}
    for t, lbl in ROW_ORDER
]

# ── layout constants (all in inches) ─────────────────────────────
FIG_W    = 7.0
ROW_H    = 0.30
HEADER_H = 0.45
RULER_H  = 0.20
FOOTER_H = 0.65
N_ROWS   = len(rows)
FIG_H    = HEADER_H + RULER_H + N_ROWS * ROW_H + FOOTER_H   # 4.90 in

# ── column x positions (axes/figure fractions 0–1) ───────────────
# 35 % variable | 15 % SBV | 15 % SynthDrive | 35 % sparkline
C_VAR_L = 0.020    # variable name left
C_SBV_R = 0.485    # SBV value right
C_SYN_R = 0.635    # SynthDrive value right
C_SPK_L = 0.660    # sparkline left
C_SPK_R = 0.985    # sparkline right


def spk_x(v):
    return C_SPK_L + (v - SCALE_MIN) / (SCALE_MAX - SCALE_MIN) * (C_SPK_R - C_SPK_L)


def row_ytop(i):
    return 1.0 - (HEADER_H + RULER_H) / FIG_H - i * ROW_H / FIG_H


def row_ybot(i):
    return row_ytop(i) - ROW_H / FIG_H


def row_yc(i):
    return (row_ytop(i) + row_ybot(i)) / 2


fig = plt.figure(figsize=(FIG_W, FIG_H))
ax  = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")

# ── header background ────────────────────────────────────────────
hdr_bot   = 1.0 - HEADER_H / FIG_H
ruler_bot = 1.0 - (HEADER_H + RULER_H) / FIG_H
ax.add_patch(mpatches.Rectangle(
    (0, ruler_bot), 1.0, (HEADER_H + RULER_H) / FIG_H,
    facecolor="#e8e8e8", edgecolor="none", transform=ax.transAxes))
ax.add_patch(mpatches.Rectangle(
    (0, ruler_bot), 1.0, RULER_H / FIG_H,
    facecolor="white", edgecolor="none", transform=ax.transAxes))

# ── header labels ────────────────────────────────────────────────
h_yc = hdr_bot + (HEADER_H / FIG_H) / 2
ax.text(C_VAR_L, h_yc, "Variable",
        ha="left", va="center", fontsize=9, fontweight="bold",
        transform=ax.transAxes)
ax.text(C_SBV_R, h_yc, "SBV",
        ha="right", va="center", fontsize=9, fontweight="bold",
        transform=ax.transAxes)
ax.text(C_SYN_R, h_yc, "SynthDrive",
        ha="right", va="center", fontsize=9, fontweight="bold",
        transform=ax.transAxes)

# ── ruler row: scale axis between header and first data row ──────
ruler_line_y = ruler_bot + 0.008
tick_top_y   = ruler_bot + 0.016
label_y      = ruler_bot + 0.018
ax.plot([C_SPK_L, C_SPK_R], [ruler_line_y, ruler_line_y],
        color="#aaaaaa", lw=0.7, transform=ax.transAxes)
for v in [-1, 0, 1, 2, 3]:
    tx = spk_x(v)
    ax.plot([tx, tx], [ruler_line_y, tick_top_y],
            color="#aaaaaa", lw=0.7, transform=ax.transAxes)
    ax.text(tx, label_y, str(v),
            ha="center", va="bottom", fontsize=7, color="#555555",
            transform=ax.transAxes)
ax.plot([0.01, 0.99], [ruler_bot, ruler_bot], color="#cccccc", lw=0.7,
        transform=ax.transAxes)

# ── data rows ────────────────────────────────────────────────────
for i, row in enumerate(rows):
    yc  = row_yc(i)
    lbl = row["label"] + (" †" if row["label"] in DAGGER_VARS else "")

    ax.text(C_VAR_L, yc, lbl,
            ha="left", va="center", fontsize=9, transform=ax.transAxes)

    ax.text(C_SBV_R, yc, f"{row['sbv']:+.3f}",
            ha="right", va="center", fontsize=9, fontfamily="monospace",
            transform=ax.transAxes)
    ax.text(C_SYN_R, yc, f"{row['syn']:+.3f}",
            ha="right", va="center", fontsize=9, fontfamily="monospace",
            transform=ax.transAxes)

    sx, dx = spk_x(row["sbv"]), spk_x(row["syn"])
    ax.plot([sx, dx], [yc, yc], color=LINE_COLOR, lw=1.0,
            transform=ax.transAxes, zorder=2)
    ax.plot(sx, yc, marker="o", markersize=6,
            markerfacecolor="white", markeredgecolor=SBV_COLOR, markeredgewidth=1.2,
            linestyle="none", transform=ax.transAxes, zorder=3)
    ax.plot(dx, yc, marker="o", markersize=6,
            markerfacecolor=SYN_COLOR, markeredgecolor=SYN_COLOR,
            linestyle="none", transform=ax.transAxes, zorder=4)

# ── separator line ───────────────────────────────────────────────
sep_y = row_ybot(N_ROWS - 1)
ax.plot([0.01, 0.99], [sep_y, sep_y], color="#cccccc", lw=0.7,
        transform=ax.transAxes)

# ── footnote ─────────────────────────────────────────────────────
fn_y = sep_y - 0.04
ax.text(C_VAR_L, fn_y,
        "† SBV estimate below 0.05 in absolute value.",
        ha="left", va="top", fontsize=8, color="#555555",
        transform=ax.transAxes)

# ── legend ───────────────────────────────────────────────────────
leg_y  = sep_y - 0.09
leg_x1 = C_VAR_L + 0.01
ax.plot(leg_x1, leg_y, marker="o", markersize=6,
        markerfacecolor="white", markeredgecolor=SBV_COLOR, markeredgewidth=1.2,
        linestyle="none", transform=ax.transAxes)
ax.text(leg_x1 + 0.022, leg_y, "SBV seed",
        ha="left", va="center", fontsize=8, transform=ax.transAxes)

leg_x2 = C_VAR_L + 0.22
ax.plot(leg_x2, leg_y, marker="o", markersize=6,
        markerfacecolor=SYN_COLOR, markeredgecolor=SYN_COLOR,
        linestyle="none", transform=ax.transAxes)
ax.text(leg_x2 + 0.022, leg_y, "SynthDrive",
        ha="left", va="center", fontsize=8, transform=ax.transAxes)

out_path.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out_path, dpi=200, bbox_inches="tight")
print(f"Saved: {out_path}")
