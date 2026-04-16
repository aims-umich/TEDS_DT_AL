# ghx_exp_model_predictions_2x1_sigmaMerge.py
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

# ----------------- Paths -----------------
paths = {
    # MvG-SINDyC mean predictions (same file as before)
    "MvG-SINDyC_MEAN":  "/home/unabila/ghxSindy/4_tj_exp_pred_sindy/exp_pred_index160.csv",
    # MvG-SINDyC sigmas (time-varying)
    "MvG-SINDyC_SIGMA": "/home/unabila/ghxSindy/sigma_4_tj_exp_pred_sindy/exp_pred_index120.csv",
    # SINDy (no UQ): exp-only fit
    "SINDy":            "/home/unabila/ghxSindy/exp_only_pred_sindy/exp_pred.csv",
    # GRU / FNN
    "GRU":              "/home/unabila/ghxSindy/gru_pred_exp_results/pred_300_mq.csv",
    "FNN (with EXP)":   "/home/unabila/ghxSindy/fnn_pred_exp_results/experiment_predictions_mq.csv",
    "FNN (w/o EXP)":    "/home/unabila/ghxSindy/wo_fnn_pred_exp_results/experiment_predictions_mq.csv",
}

# ----------------- Styling -----------------
STYLES = {
    "MvG-SINDyC": dict(color="#2ca02c", ls="-",  marker="o"),      # green
    "SINDy":      dict(color="#145A32", ls="--", marker="o"),      # dark green
    "GRU":        dict(color="#1f77b4", ls="-",  marker="s"),
    "FNN (with EXP)": dict(color="#ff7f0e", ls="-.", marker="^"),
    "FNN (w/o EXP)":  dict(color="#9467bd", ls="--", marker="D"),
}

BAND_ALPHA = 0.22
SIGMA_MULT = 3    # ~95% band for Gaussian assumption
BAND_LABEL = rf"MvG-SINDyC $\pm{2}\sigma$"
GRU_SHIFT = 300

def sparse_markevery(npts: int) -> int:
    return max(1, npts // 30)

def _col(df, *cands):
    lower = {c.lower(): c for c in df.columns}
    for c in cands:
        if c and c.lower() in lower:
            return lower[c.lower()]
    return None

# ---------- Loaders ----------
def load_sindy_only(path: str) -> pd.DataFrame:
    """For exp_only_pred_sindy/exp_pred.csv: time_s, m_pred_kgps, Q_pred_kW."""
    df = pd.read_csv(path)
    t = df[_col(df, "time_s", "time", "t")].astype(float)
    m = df[_col(df, "m_pred_kgps", "mflow_pred_kgps", "mflow_GHX_bypass_kgps", "m")].astype(float)
    q = df[_col(df, "Q_pred_kW", "Q_ghx_kW", "Q", "q_pred_kW")].astype(float)
    out = pd.DataFrame({"time": t, "m_pred": m, "q_pred": q})
    return out.sort_values("time").reset_index(drop=True)

def load_mvg_sindyc_mean_sigma(mean_path: str, sigma_path: str) -> pd.DataFrame:
    df_mean  = pd.read_csv(mean_path).copy()
    df_sigma = pd.read_csv(sigma_path).copy()
    df_mean.rename(columns={_col(df_mean,  "time_s","time","t"): "time"}, inplace=True)
    df_sigma.rename(columns={_col(df_sigma, "time_s","time","t"): "time"}, inplace=True)

    # asof merge in case times are slightly off
    df = pd.merge_asof(df_mean.sort_values("time"),
                       df_sigma.sort_values("time"),
                       on="time", direction="nearest", tolerance=1e-6,
                       suffixes=("", "_sigma"))

    m_mean = df[_col(df, "mflow_GHX_bypass_kgps","mflow_pred_kgps","m_pred_kgps","m")].astype(float)
    q_mean = df[_col(df, "Q_ghx_kW","Q_pred_kW","Q","q_pred_kW")].astype(float)
    # sigma columns (name them flexibly)
    m_sig  = df[_col(df, "m_sigma","mflow_sigma_kgps","mflow_std_kgps")].astype(float)
    q_sig  = df[_col(df, "q_sigma","Q_sigma_kW","Q_std_kW")].astype(float)

    out = pd.DataFrame({"time": df["time"].astype(float),
                        "m_pred": m_mean, "q_pred": q_mean,
                        "m_sigma": m_sig, "q_sigma": q_sig})
    return out.dropna(subset=["time"]).sort_values("time").reset_index(drop=True)

def load_truth_like(label: str, path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    t = df[_col(df, "time_s","time","t")].astype(float) if _col(df,"time_s","time","t") else np.arange(len(df),dtype=float)
    out = pd.DataFrame({
        "time": t,
        "m_true": df[_col(df,"m_true")].astype(float) if _col(df,"m_true") else np.nan,
        "m_pred": df[_col(df,"m_pred")].astype(float) if _col(df,"m_pred") else np.nan,
        "q_true": df[_col(df,"q_true")].astype(float) if _col(df,"q_true") else np.nan,
        "q_pred": df[_col(df,"q_pred")].astype(float) if _col(df,"q_pred") else np.nan,
    })
    if label == "GRU":
        out["time"] = out["time"] + float(GRU_SHIFT)
    return out.replace([np.inf,-np.inf],np.nan).dropna(subset=["time"]).sort_values("time").reset_index(drop=True)

def pick_truth_df(loaded: dict) -> pd.DataFrame:
    for key in ["FNN (with EXP)", "FNN (w/o EXP)"]:
        if key in loaded:
            df = loaded[key]
            if df["m_true"].notna().any() and df["q_true"].notna().any():
                return df[["time","m_true","q_true"]].copy()
    raise ValueError("No dataset with ground-truth columns (m_true, q_true).")

# ---------- Load ----------
mvg_df   = load_mvg_sindyc_mean_sigma(paths["MvG-SINDyC_MEAN"], paths["MvG-SINDyC_SIGMA"])
sindy_df = load_sindy_only(paths["SINDy"])
others   = {k: load_truth_like(k, p) for k,p in paths.items()
            if k not in ["MvG-SINDyC_MEAN","MvG-SINDyC_SIGMA","SINDy"]}
truth_df = pick_truth_df(others)

# ----------------- Plot (2x1) -----------------
fig, axes = plt.subplots(2, 1, figsize=(5, 9), sharex=False)
plt.subplots_adjust(bottom=0.18)

handles, labels = [], []

# ===== Top: m (kg/s) =====
ax = axes[0]
h_truth_m, = ax.plot(truth_df["time"], truth_df["m_true"],
                     color="k", lw=4, solid_capstyle="round", zorder=9, label="True Exp")
handles.append(h_truth_m); labels.append("True Exp")

# MvG-SINDyC mean + band
mv_style = STYLES["MvG-SINDyC"]
me = sparse_markevery(len(mvg_df))
h_mvg_m, = ax.plot(mvg_df["time"], mvg_df["m_pred"],
                   color=mv_style["color"], ls=mv_style["ls"], marker=mv_style["marker"],
                   markevery=me, mfc="none", mec=mv_style["color"], lw=3, label="MvG-SINDyC")
if mvg_df["m_sigma"].notna().any():
    ax.fill_between(mvg_df["time"],
                    mvg_df["m_pred"] - SIGMA_MULT*mvg_df["m_sigma"],
                    mvg_df["m_pred"] + SIGMA_MULT*mvg_df["m_sigma"],
                    color=mv_style["color"], alpha=BAND_ALPHA, zorder=2, label=BAND_LABEL)

# SINDy (no-UQ)
sd_style = STYLES["SINDy"]
h_sindy_m, = ax.plot(sindy_df["time"], sindy_df["m_pred"],
                     color=sd_style["color"], ls=sd_style["ls"], marker=sd_style["marker"],
                     markevery=sparse_markevery(len(sindy_df)), mfc="none", mec=sd_style["color"],
                     lw=3, label="SINDyC")

# Other models
for lbl, df in others.items():
    st = STYLES.get(lbl, {})
    ax.plot(df["time"], df["m_pred"], color=st.get("color"),
            ls=st.get("ls","-"), marker=st.get("marker"),
            markevery=sparse_markevery(len(df)), mfc="none", mec=st.get("color"), lw=3, label=lbl)

ax.set_ylabel(r"Experiment - $\dot{m}_{GHX}$ (kg/s)")
ax.grid(True, alpha=0.3)

# ===== Bottom: Q (kW) =====
ax = axes[1]
h_truth_q, = ax.plot(truth_df["time"], truth_df["q_true"],
                     color="k", lw=4, solid_capstyle="round", zorder=9, label="True Exp")

# MvG-SINDyC
ax.plot(mvg_df["time"], mvg_df["q_pred"],
        color=mv_style["color"], ls=mv_style["ls"], marker=mv_style["marker"],
        markevery=sparse_markevery(len(mvg_df)), mfc="none", mec=mv_style["color"], lw=3, label="MvG-SINDyC")
if mvg_df["q_sigma"].notna().any():
    ax.fill_between(mvg_df["time"],
                    mvg_df["q_pred"] - 2*mvg_df["q_sigma"],
                    mvg_df["q_pred"] + 2*mvg_df["q_sigma"],
                    color=mv_style["color"], alpha=BAND_ALPHA, zorder=2, label=BAND_LABEL)

# SINDy (no-UQ)
ax.plot(sindy_df["time"], sindy_df["q_pred"],
        color=sd_style["color"], ls=sd_style["ls"], marker=sd_style["marker"],
        markevery=sparse_markevery(len(sindy_df)), mfc="none", mec=sd_style["color"], lw=3, label="SINDy")

# Others
for lbl, df in others.items():
    st = STYLES.get(lbl, {})
    ax.plot(df["time"], df["q_pred"], color=st.get("color"),
            ls=st.get("ls","-"), marker=st.get("marker"),
            markevery=sparse_markevery(len(df)), mfc="none", mec=st.get("color"), lw=3, label=lbl)

ax.set_xlabel("time (s)")
ax.set_ylabel(r"Experiment - $Q_{ghx}$ (W)")
ax.grid(True, alpha=0.3)

# Shared legend (single row)
all_handles, all_labels = axes[0].get_legend_handles_labels()
fig.legend(all_handles, all_labels,
           loc="lower center", bbox_to_anchor=(0.5, 0.02),
           ncol=len(all_labels), frameon=True,
           handlelength=2.2, handletextpad=0.6, columnspacing=1.0)
plt.subplots_adjust(bottom=0.18)

plt.savefig("Exp_ghx_model_sigmaMerge_2x1.png", dpi=300, bbox_inches="tight")
plt.close()
print("Saved: Exp_ghx_model_sigmaMerge_2x1.png")
