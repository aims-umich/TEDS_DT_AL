# -*- coding: utf-8 -*-
"""
Active-learning FNN for GHX (EXP included in sampling pool):
- Treat the EXP dataset as one more "run" in the pool (just like a sim file).
- Start with INIT_COUNT random files (could include EXP), train FNN (U->X).
- Evaluate on EXP (always), save metrics + plots.
- Score remaining files by per-file combined RMSE, add INCREMENT worst, repeat.
"""

import os, re, random
import numpy as np
import pandas as pd
import time  # <-- added

# plotting (headless)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({
    "font.size": 14, "axes.titlesize": 18, "axes.labelsize": 16,
    "xtick.labelsize": 13, "ytick.labelsize": 13, "legend.fontsize": 14,
    "figure.titlesize": 20,
})

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path

# -----------------------------
# Config
# -----------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]   # TEDS_DT_AL/
DATA_DIR = REPO_ROOT / "data" / "ghx_data_csv"    # ghx_run{idx}.csv
EXP_CSV  = REPO_ROOT / "data" / "experiment_csv" / "experiment_ghx_formatted.csv"

RESULTS_DIR = REPO_ROOT / "results"
RESULTS_CSV_DIR = RESULTS_DIR / "result_csv"
OUT_DIR = RESULTS_CSV_DIR / "fnn_active_figs"

OUT_DIR.mkdir(parents=True, exist_ok=True)


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED   = 42

INIT_COUNT = 10
INCREMENT  = 10

EPOCHS       = 40
BATCH_SIZE   = 256
LR           = 1e-3
HIDDEN       = 128
WEIGHT_DECAY = 0.0

# Controls & states (use the first 4 controls)
CTRL_NAMES  = ["opening_PV006","mflow_pump_out","T_pump_in","T_heater_out","T_chiller_after"]
CTRL4       = CTRL_NAMES[:4]
STATE_NAMES = ["mflow_GHX_bypass","qghx_kW"]  # (m, q)

OUT_DIR  = "fnn_active_figs"
os.makedirs(OUT_DIR, exist_ok=True)

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
torch.backends.cudnn.benchmark = True
if hasattr(torch, "set_float32_matmul_precision"):
    torch.set_float32_matmul_precision("high")

print(f"Using device: {DEVICE}")
if DEVICE == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))

# -----------------------------
# Data helpers
# -----------------------------
def load_all_sim_runs(dirpath):
    """Return runs, ids where each item is (U:(T,4), X:(T,2))."""
    runs, ids = [], []
    for fn in sorted(os.listdir(dirpath)):
        m = re.match(r"ghx_run(\d+)\.csv$", fn)
        if not m:
            continue
        rid = int(m.group(1))
        df = pd.read_csv(dirpath / fn)
        if not set(CTRL4).issubset(df.columns) or not set(STATE_NAMES).issubset(df.columns):
            continue
        U = df[CTRL4].to_numpy(dtype=np.float32)
        X = df[STATE_NAMES].to_numpy(dtype=np.float32)
        T = min(len(U), len(X))
        if T > 0:
            runs.append((U[:T], X[:T]))
            ids.append(rid)
    return runs, np.array(ids, dtype=int)

def load_experiment_as_run(path):
    """Load EXP as a run: (Ue, Xe, te)."""
    dfe = pd.read_csv(path)
    missing = [c for c in CTRL4 + STATE_NAMES if c not in dfe.columns]
    if len(missing):
        raise ValueError(f"Missing columns in {path}: {missing}")
    Ue = dfe[CTRL4].to_numpy(dtype=np.float32)
    Xe = dfe[STATE_NAMES].to_numpy(dtype=np.float32)
    te = dfe["time_sec"].to_numpy(dtype=float) if "time_sec" in dfe.columns else np.arange(len(Xe), dtype=float)
    mask = np.isfinite(Ue).all(axis=1) & np.isfinite(Xe).all(axis=1)
    Ue, Xe, te = Ue[mask], Xe[mask], te[mask]
    T = min(len(Ue), len(Xe))
    return Ue[:T], Xe[:T], te[:T]

# -----------------------------
# Model
# -----------------------------
class FNN(nn.Module):
    def __init__(self, in_dim=4, hidden=128, out_dim=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, out_dim)
        )
    def forward(self, x):
        return self.net(x)

# -----------------------------
# Train one FNN on a subset of runs
# -----------------------------
def train_on_runs(runs_subset):
    """Fit scalers on current subset (whatever it contains), then train FNN."""
    U_all = np.concatenate([U for (U, _) in runs_subset if len(U) > 0], axis=0)
    X_all = np.concatenate([X for (_, X) in runs_subset if len(X) > 0], axis=0)

    u_scaler = StandardScaler().fit(U_all)
    y_scaler = StandardScaler().fit(X_all)

    U_n = u_scaler.transform(U_all).astype(np.float32)
    X_n = y_scaler.transform(X_all).astype(np.float32)

    ds = TensorDataset(torch.tensor(U_n), torch.tensor(X_n))
    pin = DEVICE == "cuda"
    ld  = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, pin_memory=pin)

    model = FNN(in_dim=4, hidden=HIDDEN, out_dim=2).to(DEVICE)
    opt   = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    crit  = nn.MSELoss()

    for ep in range(1, EPOCHS+1):
        model.train(); tot = 0.0
        for xb, yb in ld:
            xb = xb.to(DEVICE, non_blocking=True)
            yb = yb.to(DEVICE, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            yp = model(xb)
            loss = crit(yp, yb)
            loss.backward()
            opt.step()
            tot += loss.item() * xb.size(0)
        print(f"  epoch {ep:02d} | train MSE={tot/len(ds):.6f}")

    return model, u_scaler, y_scaler

# -----------------------------
# Scoring helpers
# -----------------------------
def per_file_rmse(model, u_scaler, y_scaler, run):
    """Score a single run with combined RMSE on (m,q)."""
    U, X = run
    if len(U) == 0:
        return np.inf
    U_n = u_scaler.transform(U).astype(np.float32)
    with torch.no_grad():
        xb = torch.tensor(U_n, dtype=torch.float32, device=DEVICE)
        Yp_n = model(xb).cpu().numpy()
    Xp = y_scaler.inverse_transform(Yp_n)
    rmse_m = mean_squared_error(X[:,0], Xp[:,0], squared=False)
    rmse_q = mean_squared_error(X[:,1], Xp[:,1], squared=False)
    return 0.5 * (rmse_m + rmse_q)

def eval_experiment_and_plot(iter_idx, n_train_files, model, u_scaler, y_scaler, Ue, Xe, te):
    """Evaluate on EXP and save time-series + parity plots. Return metrics dict."""
    Ue_n = u_scaler.transform(Ue).astype(np.float32)
    with torch.no_grad():
        xb = torch.tensor(Ue_n, dtype=torch.float32, device=DEVICE)
        Yp_n = model(xb).cpu().numpy()
    Xp = y_scaler.inverse_transform(Yp_n)

    rmse_m = mean_squared_error(Xe[:,0], Xp[:,0], squared=False)
    rmse_q = mean_squared_error(Xe[:,1], Xp[:,1], squared=False)
    mae_m  = mean_absolute_error(Xe[:,0], Xp[:,0])
    mae_q  = mean_absolute_error(Xe[:,1], Xp[:,1])

    tag = f"iter_{iter_idx:03d}_n{n_train_files}"

    # m vs time
    plt.figure(figsize=(12,4))
    plt.plot(te, Xe[:,0], label="m true (exp)")
    plt.plot(te, Xp[:,0], label="m pred (FNN)", alpha=0.85)
    plt.xlabel("time (s)"); plt.ylabel("mflow_GHX_bypass")
    plt.title(f"Experiment m: true vs pred | {tag}")
    plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout(pad=0.6)
    #plt.savefig(os.path.join(OUT_DIR, f"{tag}_m_timeseries.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # q vs time
    plt.figure(figsize=(12,4))
    plt.plot(te, Xe[:,1], label="q true (exp)")
    plt.plot(te, Xp[:,1], label="q pred (FNN)", alpha=0.85)
    plt.xlabel("time (s)"); plt.ylabel("qghx_kW")
    plt.title(f"Experiment q: true vs pred | {tag}")
    plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout(pad=0.6)
    plt.savefig(os.path.join(OUT_DIR, f"{tag}_q_timeseries.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # Parity plots
    for j, name in enumerate(["m","q"]):
        plt.figure(figsize=(6,6))
        plt.scatter(Xe[:,j], Xp[:,j], s=18, alpha=0.5)
        lo = float(min(Xe[:,j].min(), Xp[:,j].min()))
        hi = float(max(Xe[:,j].max(), Xp[:,j].max()))
        plt.plot([lo,hi],[lo,hi],"--")
        plt.xlabel(f"{name} true (exp)"); plt.ylabel(f"{name} pred (FNN)")
        plt.title(f"Parity: {name} | {tag}")
        plt.grid(True, alpha=0.3); plt.tight_layout(pad=0.6)
        #plt.savefig(os.path.join(OUT_DIR, f"{tag}_{name}_parity.png"), dpi=300, bbox_inches="tight")
        plt.close()

    print(f"[Iter {iter_idx}] files={n_train_files} | EXP: rmse_m={rmse_m:.4f}, rmse_q={rmse_q:.4f}, "
          f"mae_m={mae_m:.4f}, mae_q={mae_q:.4f}")
    return dict(iter=iter_idx, n_files=n_train_files,
                rmse_m=float(rmse_m), rmse_q=float(rmse_q),
                mae_m=float(mae_m),   mae_q=float(mae_q))

# -----------------------------
# MAIN: Active-learning loop (EXP is in the pool)
# -----------------------------
# 1) Load sims and EXP
runs, file_ids = load_all_sim_runs(DATA_DIR)         # list of (U,X)
Ue, Xe, te = load_experiment_as_run(EXP_CSV)         # EXP as (U,X,t)

# 2) Append EXP to the pool as another "run"
EXP_IDX = len(runs)                                  # remember where EXP sits
runs.append((Ue, Xe))
# optional: a friendly label list, if you want to print identities
run_labels = list(map(str, file_ids)) + ["EXP"]

total_files = len(runs)   # includes EXP now
assert total_files > 0, "No runs found."

# 3) Initial selection (either manual or random)
all_idx = list(range(total_files))

# EITHER: manual fixed initial (must refer to indices in [0, total_files-1])
selected_idx = [14, 21, 35, 39, 42, 57, 73, 101, 245, 299]  # <- only if valid for your pool

# OR: random initial selection (includes the possibility of sampling EXP)
random.seed(SEED)
#selected_idx  = random.sample(all_idx, min(INIT_COUNT, total_files))

remaining_idx = [i for i in all_idx if i not in selected_idx]

print(f"Total files (incl. EXP): {total_files}")
print(f"Initial selected: {len(selected_idx)} | Remaining: {len(remaining_idx)}")
print(f"Selected labels: {[run_labels[i] for i in selected_idx]}")

results = []
iteration = 0

t0 = time.perf_counter()  # <-- added: global stopwatch start

while True:
    iteration += 1
    iter_start = time.perf_counter()  # <-- added: per-iteration start

    # Build current training subset (whatever is selected; EXP may or may not be inside)
    train_subset = [runs[i] for i in selected_idx]
    print(f"\n=== Iter {iteration} | training with {len(selected_idx)} files ===")

    # Train FNN (fresh) on current subset
    model, u_scaler, y_scaler = train_on_runs(train_subset)

    # Evaluate on EXP (always) + save plots
    metrics = eval_experiment_and_plot(iteration, len(selected_idx), model, u_scaler, y_scaler, Ue, Xe, te)


    # Stop if nothing remains
    if len(remaining_idx) == 0:
        break

    # Score remaining files and add worst K
    print("  Scoring remaining files (incl. EXP if not yet selected)...")
    scored = []
    for idx in remaining_idx:
        score = per_file_rmse(model, u_scaler, y_scaler, runs[idx])
        if np.isfinite(score):
            scored.append((idx, score))

    if len(scored) == 0:
        print("No valid remaining files to add. Stopping.")
        break

    scored.sort(key=lambda t: t[1], reverse=True)        # worst first
    k = min(INCREMENT, len(scored))
    new_pick = [idx for (idx, _) in scored[:k]]

    selected_idx.extend(new_pick)
    remaining_idx = [i for i in remaining_idx if i not in new_pick]
    print(f"  Added {k} worst files. Now selected={len(selected_idx)} remaining={len(remaining_idx)}")
    print(f"  Newly added labels: {[run_labels[i] for i in new_pick]}")
    
        # ---- added: timing columns
    iter_runtime = time.perf_counter() - iter_start
    cum_runtime  = time.perf_counter() - t0
    metrics["iter_runtime_sec"] = float(iter_runtime)
    metrics["cumulative_runtime_sec"] = float(cum_runtime)

    results.append(metrics)

# -----------------------------
# Save summary CSV + learning curves
# -----------------------------
plt.savefig(OUT_DIR / f"{tag}_q_timeseries.png", dpi=300, bbox_inches="tight")
csv_path = OUT_DIR / "active_learning_exp_metrics.csv"
df.to_csv(csv_path, index=False)

