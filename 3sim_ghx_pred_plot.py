# ghx_model_predictions_3x2.py
# 3 simulations (56, 121, 300) in a 3x2 grid
# Left column: m (kg/s), Right column: Q (W)
# Styles and layout match the user's original single-plot script.

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.size": 16,
    "axes.titlesize": 20,
    "axes.labelsize": 18,
    "xtick.labelsize": 15,
    "ytick.labelsize": 15,
    "legend.fontsize": 14,
    "figure.titlesize": 22,
})


from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent

SIM_PATHS = {
    121: {
        "SINDyC":         REPO_ROOT / "data" / "cv_holdout_preds" / "ghx_pred_index121.csv",
        "GRU":            REPO_ROOT / "data" / "gru_results" / "pred_121.csv",
        "FNN (with EXP)": REPO_ROOT / "data" / "sim_figs_fnn" / "pred_121.csv",
        "FNN (w/o EXP)":  REPO_ROOT / "data" / "wo_sim_figs_fnn" / "pred_121.csv",
    },
    176: {
        "SINDyC":         REPO_ROOT / "data" / "cv_holdout_preds" / "ghx_pred_index176.csv",
        "GRU":            REPO_ROOT / "data" / "gru_results" / "pred_176.csv",
        "FNN (with EXP)": REPO_ROOT / "data" / "sim_figs_fnn" / "pred_176.csv",
        "FNN (w/o EXP)":  REPO_ROOT / "data" / "wo_sim_figs_fnn" / "pred_176.csv",
    },
    300: {
        "SINDyC":         REPO_ROOT / "data" / "cv_holdout_preds" / "ghx_pred_index300.csv",
        "GRU":            REPO_ROOT / "data" / "gru_results" / "pred_300.csv",
        "FNN (with EXP)": REPO_ROOT / "data" / "sim_figs_fnn" / "pred_300.csv",
        "FNN (w/o EXP)":  REPO_ROOT / "data" / "wo_sim_figs_fnn" / "pred_300.csv",
    },
}

RESULTS_DIR = REPO_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# ----------- Styling (unchanged) -----------
STYLES = {
    "SINDyC":          dict(color="#2ca02c", ls="-",  marker="o"),
    "GRU":             dict(color="#1f77b4", ls="-",  marker="s"),
    "FNN (with EXP)":  dict(color="#ff7f0e", ls="-.", marker="^"),
    "FNN (w/o EXP)":   dict(color="#9467bd", ls="--", marker="D"),
}

def sparse_markevery(npts: int) -> int:
    return max(1, npts // 30)

# ----------- Helpers -----------
def _col(df, *candidates):
    lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower:
            return lower[cand.lower()]
    return None

def load_one(label: str, path: str) -> pd.DataFrame:
    """
    Return DataFrame with columns: time, m_true, m_pred, q_true, q_pred.
    """
    df = pd.read_csv(path)

    if label == "SINDyC":
        tcol = _col(df, "time_s", "time", "t", "Time_s")
        out = pd.DataFrame({
            "time":   df[tcol] if tcol else np.arange(len(df), dtype=float),
            "m_true": df[_col(df, "mflow_true_kgps")],
            "m_pred": df[_col(df, "mflow_pred_kgps")],
            "q_true": df[_col(df, "Q_true_kW", "q_true", "Q_true")],
            "q_pred": df[_col(df, "Q_pred_kW", "q_pred", "Q_pred")],
        })
    elif label == "GRU" or label.startswith("FNN"):
        tcol = _col(df, "time_s", "time", "t")
        out = pd.DataFrame({
            "time":   df[tcol] if tcol else np.arange(len(df), dtype=float),
            "m_true": df[_col(df, "m_true")],
            "m_pred": df[_col(df, "m_pred")],
            "q_true": df[_col(df, "q_true")],
            "q_pred": df[_col(df, "q_pred")],
        })
    else:
        raise ValueError(f"Unknown label: {label}")

    out = out.replace([np.inf, -np.inf], np.nan).dropna()
    out = out.sort_values("time").reset_index(drop=True)
    return out

# plot
fig, axes = plt.subplots(3, 2, figsize=(16, 14), gridspec_kw={"wspace": 0.3, "hspace": 0.45})
plt.subplots_adjust(bottom=0.16)  # room for shared legend

legend_handles = {}
legend_order   = ["True Sim", "SINDyC", "GRU", "FNN (with EXP)", "FNN (w/o EXP)"]

for row_idx, sim_id in enumerate([121,176,300]):
    paths = SIM_PATHS[sim_id]
    # Load all methods available for this sim
    loaded = {lbl: load_one(lbl, p) for lbl, p in paths.items()}

    # Choose a "truth" series to draw as the thick line
    if "SINDyC" in loaded:
        truth_df = loaded["SINDyC"][["time", "m_true", "q_true"]].copy()
    else:
        k_long = max(loaded, key=lambda k: len(loaded[k]))
        truth_df = loaded[k_long][["time", "m_true", "q_true"]].copy()

    # Left: mass flow
    ax_m = axes[row_idx, 0]
    h_truth_m, = ax_m.plot(truth_df["time"], truth_df["m_true"],
                           color="k", lw=4, solid_capstyle="round", zorder=8, label="True Sim")
    legend_handles.setdefault("True Sim", h_truth_m)

    for lbl, df in loaded.items():
        st = STYLES.get(lbl, {})
        me = sparse_markevery(len(df))
        h, = ax_m.plot(df["time"], df["m_pred"],
                       color=st.get("color"), ls=st.get("ls", "-"),
                       marker=st.get("marker"), markevery=me, markersize=6,
                       mfc="none", mec=st.get("color"), lw=3, label=lbl)
        legend_handles.setdefault(lbl, h)

    ax_m.set_ylabel(rf"Sim #{sim_id} - $\dot{{m}}_{{GHX}}$ (kg/s)")
    ax_m.grid(True, alpha=0.3)

    # Right: heat rate
    ax_q = axes[row_idx, 1]
    h_truth_q, = ax_q.plot(truth_df["time"], truth_df["q_true"],
                           color="k", lw=4, solid_capstyle="round", zorder=8, label="True Sim")
    legend_handles.setdefault("True Sim", h_truth_q)

    for lbl, df in loaded.items():
        st = STYLES.get(lbl, {})
        me = sparse_markevery(len(df))
        h, = ax_q.plot(df["time"], df["q_pred"],
                       color=st.get("color"), ls=st.get("ls", "-"),
                       marker=st.get("marker"), markevery=me, markersize=6,
                       mfc="none", mec=st.get("color"), lw=3, label=lbl)
        legend_handles.setdefault(lbl, h)

    ax_q.set_ylabel(rf"Sim #{sim_id} - $Q_{{ghx}}$ (W)")
    ax_q.grid(True, alpha=0.3)

# X labels only on the bottom row
axes[2, 0].set_xlabel("time (s)")
axes[2, 1].set_xlabel("time (s)")

# Shared legend at the bottom
handles = [legend_handles[k] for k in legend_order if k in legend_handles]
labels  = [h.get_label() for h in handles]
fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.04),
           ncol=len(handles), frameon=True, handlelength=2.2, handletextpad=0.6, columnspacing=1.2)

plt.savefig(RESULTS_DIR / "ghx3_model_predictions_3x2.png", dpi=300, bbox_inches="tight")
plt.close()
