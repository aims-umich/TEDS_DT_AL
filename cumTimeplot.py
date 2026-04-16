# time_models_scaled_0_500.py
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

plt.rcParams.update({
    "font.size": 16, "axes.titlesize": 20, "axes.labelsize": 18,
    "xtick.labelsize": 15, "ytick.labelsize": 15,
    "legend.fontsize": 12, "figure.titlesize": 22,
})

# ================== CONFIG ==================
X_SCALE = 10  # models per iteration

# ---------- INPUT PATHS ----------
# Keep these two as your MvG-SINDyC timings:
paths = {
    "MvG-SINDyC - Random": "/home/unabila/ghxSindy/time4rndm_4traj_10init10_CI_rmse_mae_results.csv",
    "MvG-SINDyC - AL":     "/home/unabila/ghxSindy/time4rndm_4traj_10init10_CI_rmse_mae_results.csv",  

    # NEW: deterministic SINDyC timings (put your deterministic time CSVs here)
    "SINDyC - Random":     "/home/unabila/ghxSindy/rmdm_incExp_errorAL10sindy_exp_metrics.csv",
    "SINDyC - AL":         "/home/unabila/ghxSindy/incExp_errorAL10sindy_exp_metrics.csv",


    "GRU - AL":            "/home/unabila/ghxSindy/time_gru_al_figs/active_learning_exp_metrics.csv",
    "GRU - Random":        "/home/unabila/ghxSindy/time_rndm_gru_sampling_results/incremental_metrics.csv",

    "FNN (with EXP) - AL":     "/home/unabila/ghxSindy/fnn_active_figs/active_learning_exp_metrics.csv",
    "FNN (with EXP) - Random": "/home/unabila/ghxSindy/rndmFNN_sampling_results_fnn/incremental_metrics.csv",

    "FNN (w/o EXP) - AL":      "/home/unabila/ghxSindy/WO_fnn_active_results/active_learning_exp_metrics.csv",
    "FNN (w/o EXP) - Random":  "/home/unabila/ghxSindy/timeWO_rndm_sampling_results_fnn/incremental_metrics.csv",
}

# ---------- COLORS & STYLES ----------
colors = {
    # MvG-SINDyC (keep your original palette)
    "MvG-SINDyC - Random": "#1f77b4",
    "MvG-SINDyC - AL":     "#ff7f0e",
    # Deterministic SINDyC (use distinct hues)
    "SINDyC - Random":     "#17becf",
    "SINDyC - AL":         "#bcbd22",
    # GRU/FNN (unchanged)
    "GRU - AL":            "#2ca02c",
    "GRU - Random":        "#d62728",
    "FNN (with EXP) - AL": "#9467bd",
    "FNN (with EXP) - Random": "#8c564b",
    "FNN (w/o EXP) - AL":  "#7f7f7f",
    "FNN (w/o EXP) - Random": "#e377c2",
}
linestyle = {"AL": "-", "Random": "--"}
marker    = {"AL": "o", "Random": "s"}
def style_key(lbl): return "AL" if "AL" in lbl else "Random"

LABEL_SHORT = {
    "MvG-SINDyC - Random": "MvG Rnd",
    "MvG-SINDyC - AL":     "MvG AL",
    "SINDyC - Random":     "SINDyC Rnd",
    "SINDyC - AL":         "SINDyC AL",
    "GRU - AL":            "GRU AL",
    "GRU - Random":        "GRU Rnd",
    "FNN (with EXP) - AL":     "FNN+EXP AL",
    "FNN (with EXP) - Random": "FNN+EXP Rnd",
    "FNN (w/o EXP) - AL":      "",
    "FNN (w/o EXP) - Random":  "FNN",
}

MANUAL_NUDGE = {
    "FNN (with EXP) - Random":  (-30, -6),
    "FNN (w/o EXP) - Random":   (-25,  2),
}

# ---------- MANUAL SATURATION (IN ITERATIONS; scaled by X_SCALE) ----------
# Keep your existing values for MvG; add reasonable placeholders for deterministic SINDyC (edit if you have exact).
SAT_MVG_SINDY_RANDOM     = 48
SAT_MVG_SINDY_AL         = 15
SAT_SINDY_RANDOM         = 20   # <-- adjust if your deterministic run saturates earlier/later
SAT_SINDY_AL             = 10

SAT_GRU_AL               = 20
SAT_GRU_RANDOM           = 34
SAT_FNN_EXP_AL           = 5
SAT_FNN_EXP_RANDOM       = 18
SAT_FNN_NOEXP_AL         = 3
SAT_FNN_NOEXP_RANDOM     = 3

MANUAL_SAT = {
    "MvG-SINDyC - Random": SAT_MVG_SINDY_RANDOM,
    "MvG-SINDyC - AL":     SAT_MVG_SINDY_AL,
    "SINDyC - Random":     SAT_SINDY_RANDOM,
    "SINDyC - AL":         SAT_SINDY_AL,
    "GRU - AL":            SAT_GRU_AL,
    "GRU - Random":        SAT_GRU_RANDOM,
    "FNN (with EXP) - AL": SAT_FNN_EXP_AL,
    "FNN (with EXP) - Random": SAT_FNN_EXP_RANDOM,
    "FNN (w/o EXP) - AL":  SAT_FNN_NOEXP_AL,
    "FNN (w/o EXP) - Random": SAT_FNN_NOEXP_RANDOM,
}

SAT_MARKER_SIZE = 160
SAT_LINE_WIDTH  = 1.4
SAT_LINE_ALPHA  = 0.75
SAT_TEXT_SIZE   = 12

# ================== HELPERS ==================
def load_xy_time(path):
    df = pd.read_csv(path)

    # x from iteration-like columns
    if "Iteration" in df.columns:
        x = df["Iteration"].to_numpy(dtype=float)
    elif "Selected Count" in df.columns:
        x = df["Selected Count"].to_numpy(dtype=float)
    else:
        x = np.arange(1, len(df) + 1, dtype=float)

    x = x * X_SCALE  # iterations -> models

    # y from cumulative time (prefer seconds)
    time_col = None
    for c in ("elapsed_s","cumulative_runtime_sec", "Cumulative Time (s)", "Cumulative Time (sec)", "Cumulative Time (seconds)"):
        if c in df.columns:
            time_col = c
            y_hours = df[c].astype(float).to_numpy() / 3600.0
            break
    if time_col is None:
        for c in ("Cumulative Time (hours)", "cumulative_runtime_hours"):
            if c in df.columns:
                time_col = c
                y_hours = df[c].astype(float).to_numpy()
                break
    if time_col is None:
        raise KeyError(f"No cumulative time column found in {path}. Columns: {df.columns.tolist()}")

    srt = np.argsort(x, kind="mergesort")
    return x[srt], y_hours[srt]

def smart_annotate(ax, x, y, text, color, nudge=(0, 0)):
    xlo, xhi = ax.get_xlim(); ylo, yhi = ax.get_ylim()
    dx = -18 if x > (xlo + xhi) / 2 else 18
    dy = 10
    if y >= ylo + 0.92 * (yhi - ylo):  dy = -6
    if y <= ylo + 0.02 * (yhi - ylo):  dy = max(dy, 14)
    dx += nudge[0]; dy += nudge[1]
    ax.annotate(
        text, xy=(x, y), xytext=(dx, dy), textcoords="offset points",
        ha=("right" if dx < 0 else "left"),
        va=("top" if dy < 0 else "bottom"),
        fontsize=SAT_TEXT_SIZE, color=color,
        bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.75),
        zorder=7, clip_on=True,
    )

# ================== PRELOAD SERIES ==================
series = {lbl: load_xy_time(p) for lbl, p in paths.items()}

# Collapse identical FNN (w/o EXP) AL vs Random if identical
collapse_fnn_woexp = False
l_al, l_rn = "FNN (w/o EXP) - AL", "FNN (w/o EXP) - Random"
if l_al in series and l_rn in series:
    x1, y1 = series[l_al]; x2, y2 = series[l_rn]
    if len(x1) == len(x2) and np.allclose(x1, x2, rtol=1e-6, atol=1e-9) and np.allclose(y1, y2, rtol=1e-6, atol=1e-9):
        collapse_fnn_woexp = True

# ================== PLOT ==================
fig = plt.figure(figsize=(10, 6))
ax = plt.gca()

for label in paths.keys():
    if collapse_fnn_woexp and label == l_rn:
        continue

    x, th = series[label]
    sk = style_key(label)
    plot_label = "FNN (w/o EXP) - AL & Random" if (collapse_fnn_woexp and label == l_al) else label

    ax.plot(
        x, th,
        linestyle=linestyle[sk],
        marker=marker[sk],
        markersize=4,
        linewidth=1.8,
        color=colors[label],
        label=plot_label
    )

    # Mark saturation
    x_sat_iter = MANUAL_SAT.get(label, None)
    if x_sat_iter is not None:
        x_sat_models = x_sat_iter * X_SCALE
        idx = int(np.argmin(np.abs(x - x_sat_models)))
        x0, y0 = x[idx], th[idx]

        ax.scatter([x0], [y0], s=SAT_MARKER_SIZE,
                   facecolors="none", edgecolors=colors[label],
                   linewidths=2.0, zorder=6)
        ax.axvline(x0, color=colors[label], linestyle=":",
                   linewidth=SAT_LINE_WIDTH, alpha=SAT_LINE_ALPHA, zorder=5)

        short = LABEL_SHORT.get(label, label)
        nudge = MANUAL_NUDGE.get(label, (0, 0))
        smart_annotate(ax, x0, y0, short, colors[label], nudge=nudge)

ax.set_xlabel("Cumulative Models (SINDyC) / Trajectories (ML)")
ax.set_ylabel("Time (hours)")
ax.set_xlim(0, 500)
ax.set_xticks(np.arange(0, 501, 50))
ax.xaxis.set_minor_locator(MultipleLocator(25))
ax.grid(True, which="major", alpha=0.3)
ax.legend(ncol=2, frameon=True)

for spine in ("top", "right"):
    ax.spines[spine].set_visible(False)

plt.tight_layout()
plt.savefig("cumulative_time_all_runs_0_500.png", dpi=300, bbox_inches="tight")
plt.close()
print("Saved: cumulative_time_all_runs_0_500.png")
