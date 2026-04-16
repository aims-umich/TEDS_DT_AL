# combined_4x2_rmse_trends.py
import os
import pandas as pd
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------- PATHS ----------
# MvG-SINDyC
SINDY_RANDOM = "/home/unabila/ghxSindy/Rndm_AL_10init10_CI_rmse_mae_results/Rndm4traj_10init10_CI_rmse_mae_results.csv"
SINDY_AL     = "/home/unabila/ghxSindy/Rndm_AL_10init10_CI_rmse_mae_results/AL_4traj_10init10_CI_rmse_mae_resultr.csv"

# SINDyC (non-MvG)
SINDYC_RANDOM = "/home/unabila/ghxSindy/rmdm_incExp_errorAL10sindy_exp_metrics.csv"
SINDYC_AL     = "/home/unabila/ghxSindy/incExp_errorAL10sindy_exp_metrics.csv"

# FNN
FNN_WITH_EXP_AL   = "/home/unabila/ghxSindy/fnn_active_figs/active_learning_exp_metrics.csv"
FNN_WITH_EXP_RND  = "/home/unabila/ghxSindy/rndmFNN_sampling_results_fnn/incremental_metrics.csv"
FNN_WO_EXP_AL     = "/home/unabila/ghxSindy/WO_fnn_active_results/active_learning_exp_metrics.csv"
FNN_WO_EXP_RND    = "/home/unabila/ghxSindy/WO_rndm_sampling_results_fnn/incremental_metrics.csv"

# GRU
GRU_RANDOM  = "/home/unabila/ghxSindy/rndm_gru_sampling_results/incremental_metrics10.csv"
GRU_AL      = "/home/unabila/ghxSindy/gru_al_figs/active_learning10_exp_metrics.csv"

# ---------- STYLE ----------
plt.rcParams.update({
    "font.size": 14, "axes.labelsize": 16,
    "xtick.labelsize": 13, "ytick.labelsize": 13,
    "legend.fontsize": 13, "figure.titlesize": 20,
})

COLORS = {
    # MvG-SINDyC
    "SINDY_RND":  "#1f77b4",
    "SINDY_AL":   "#ff7f0e",
    # SINDyC (plain)
    "SINDYC_RND": "#2ca02c",
    "SINDYC_AL":  "#d62728",
    # FNN
    "FNN_AL_EXP": "#1f77b4",
    "FNN_RND_EXP":"#ff7f0e",
    "FNN_AL_WO":  "#2ca02c",
    "FNN_RND_WO": "#9467bd",
    # GRU
    "GRU_RND":    "#1f77b4",
    "GRU_AL":     "#ff7f0e",
}

def pick_col(df, choices):
    for c in choices:
        if c in df.columns:
            return c
    raise KeyError(f"Need one of {choices}, got: {df.columns.tolist()}")

def load_sorted(path, x_choices):
    df = pd.read_csv(path)
    xcol = pick_col(df, x_choices)
    return df.sort_values(xcol).reset_index(drop=True), xcol

def boxify(ax):
    ax.set_frame_on(True)
    for s in ("top","right","bottom","left"):
        ax.spines[s].set_visible(True)
        ax.spines[s].set_linewidth(1.0)
    ax.tick_params(width=1.0)

# ---------- PANELS ----------
def _panel_generic(ax_m, ax_q, rnd_csv, al_csv, color_rnd, color_al,
                   x_choices_r=("selected_count","Selected Count","n_files"),
                   x_choices_a=("selected_count","Selected Count","n_files"),
                   rm_m_choices=("Average RMSE m","MeanSim RMSE m","rmse_m","mean_rmse_m"),
                   rm_q_choices=("Average RMSE Q","MeanSim RMSE Q","rmse_q","mean_rmse_q"),
                   label_rnd="Random", label_al="Active Learning"):
    rnd, x_r = load_sorted(rnd_csv, list(x_choices_r))
    al,  x_a = load_sorted(al_csv,  list(x_choices_a))

    rm_m_r = pick_col(rnd, list(rm_m_choices))
    rm_q_r = pick_col(rnd, list(rm_q_choices))
    rm_m_a = pick_col(al,  list(rm_m_choices))
    rm_q_a = pick_col(al,  list(rm_q_choices))

    ax_m.plot(rnd[x_r], rnd[rm_m_r], marker="s", linestyle="-",
              color=color_rnd, label=label_rnd)
    ax_m.plot(al[x_a],  al[rm_m_a],  marker="o", linestyle="-",
              color=color_al, label=label_al)

    ax_q.plot(rnd[x_r], rnd[rm_q_r], marker="s", linestyle="-",
              color=color_rnd, label=label_rnd)
    ax_q.plot(al[x_a],  al[rm_q_a],  marker="o", linestyle="-",
              color=color_al, label=label_al)

    for ax in (ax_m, ax_q):
        ax.grid(True, alpha=0.3)
        boxify(ax)
    ax_m.set_ylabel(r"RMSE - $\dot{m}$ (kg/s)")
    ax_q.set_ylabel("RMSE - Q (W)")

def panel_sindyc_plain(ax_m, ax_q):
    _panel_generic(ax_m, ax_q, SINDYC_RANDOM, SINDYC_AL,
                   COLORS["SINDYC_RND"], COLORS["SINDYC_AL"])

def panel_sindy_mvg(ax_m, ax_q):
    _panel_generic(ax_m, ax_q, SINDY_RANDOM, SINDY_AL,
                   COLORS["SINDY_RND"], COLORS["SINDY_AL"])

def panel_fnn(ax_m, ax_q):
    al_we, x_we = load_sorted(FNN_WITH_EXP_AL,  ["n_files"])
    rn_we, _    = load_sorted(FNN_WITH_EXP_RND, ["n_files"])
    al_wo, x_wo = load_sorted(FNN_WO_EXP_AL,   ["n_files"])
    rn_wo, _    = load_sorted(FNN_WO_EXP_RND,  ["n_files"])

    rm_m = "rmse_m"; rm_q = "rmse_q"

    ax_m.plot(al_we[x_we], al_we[rm_m], marker="o", linestyle="-",
              color=COLORS["FNN_AL_EXP"],  label="with EXP - AL")
    ax_m.plot(rn_we[x_we], rn_we[rm_m], marker="s", linestyle="--",
              color=COLORS["FNN_RND_EXP"], label="with EXP - Random")
    ax_m.plot(al_wo[x_wo], al_wo[rm_m], marker="o", linestyle="-",
              color=COLORS["FNN_AL_WO"],   label="w/o EXP - AL")
    ax_m.plot(rn_wo[x_wo], rn_wo[rm_m], marker="s", linestyle="--",
              color=COLORS["FNN_RND_WO"],  label="w/o EXP - Random")

    ax_q.plot(al_we[x_we], al_we[rm_q], marker="o", linestyle="-",
              color=COLORS["FNN_AL_EXP"],  label="with EXP - AL")
    ax_q.plot(rn_we[x_we], rn_we[rm_q], marker="s", linestyle="--",
              color=COLORS["FNN_RND_EXP"], label="with EXP - Random")
    ax_q.plot(al_wo[x_wo], al_wo[rm_q], marker="o", linestyle="-",
              color=COLORS["FNN_AL_WO"],   label="w/o EXP - AL")
    ax_q.plot(rn_wo[x_wo], rn_wo[rm_q], marker="s", linestyle="--",
              color=COLORS["FNN_RND_WO"],  label="w/o EXP - Random")

    for ax in (ax_m, ax_q):
        ax.grid(True, alpha=0.3)
        boxify(ax)
    ax_m.set_ylabel(r"RMSE - $\dot{m}$ (kg/s)")
    ax_q.set_ylabel("RMSE - Q (W)")

def panel_gru(ax_m, ax_q):
    rnd, x_r = load_sorted(GRU_RANDOM, ["n_files"])
    al,  x_a = load_sorted(GRU_AL,     ["n_files"])
    rm_m = pick_col(rnd, ["rmse_m"])
    rm_q = pick_col(rnd, ["rmse_q"])

    ax_m.plot(rnd[x_r], rnd[rm_m], marker="s", linestyle="-",
              color=COLORS["GRU_RND"], label="Random")
    ax_m.plot(al[x_a],  al[rm_m],  marker="o", linestyle="-",
              color=COLORS["GRU_AL"],  label="Active Learning")

    ax_q.plot(rnd[x_r], rnd[rm_q], marker="s", linestyle="-",
              color=COLORS["GRU_RND"], label="Random")
    ax_q.plot(al[x_a],  al[rm_q],  marker="o", linestyle="-",
              color=COLORS["GRU_AL"],  label="Active Learning")

    for ax in (ax_m, ax_q):
        ax.grid(True, alpha=0.3)
        boxify(ax)
    ax_m.set_ylabel(r"RMSE - $\dot{m}$ (kg/s)")
    ax_q.set_ylabel("RMSE - Q (W)")

# Warn if any file missing
for p in [SINDY_RANDOM, SINDY_AL, SINDYC_RANDOM, SINDYC_AL,
          FNN_WITH_EXP_AL, FNN_WITH_EXP_RND, FNN_WO_EXP_AL, FNN_WO_EXP_RND,
          GRU_RANDOM, GRU_AL]:
    if not os.path.exists(p):
        print(f"[WARN] Missing: {p}")

# ---------- FIGURE ----------
fig, axes = plt.subplots(4, 2, figsize=(16, 19), constrained_layout=False)
plt.subplots_adjust(top=0.94, wspace=0.28, hspace=0.38)

# Row 1: SINDyC (deterministic)
panel_sindyc_plain(axes[0, 0], axes[0, 1])
# Row 2: MvG-SINDyC
panel_sindy_mvg(axes[1, 0], axes[1, 1])
# Row 3: FNN
panel_fnn(axes[2, 0], axes[2, 1])
# Row 4: GRU
panel_gru(axes[3, 0], axes[3, 1])

# X-labels only on bottom row
axes[3, 0].set_xlabel("# simulation models/runs used")
axes[3, 1].set_xlabel("# simulation models/runs used")

# Legends on right plot of each row
axes[0, 1].legend(frameon=True, loc="best")
axes[1, 1].legend(frameon=True, loc="best")
axes[2, 1].legend(frameon=True, loc="best")
axes[3, 1].legend(frameon=True, loc="best")

# Row headers (aligned with new order)
fig.text(0.5, 0.946, "SINDyC",       ha="center", va="bottom", fontsize=20)
fig.text(0.5, 0.72,  "MvG-SINDyC",   ha="center", va="bottom", fontsize=20)
fig.text(0.5, 0.49,  "FNN",          ha="center", va="bottom", fontsize=20)
fig.text(0.5, 0.28,  "GRU",          ha="center", va="bottom", fontsize=20)

out_png = "rmse_trends_4x2.png"
fig.savefig(out_png, dpi=600, bbox_inches="tight")
print(f"[OK] Saved {out_png}")
plt.close(fig)
