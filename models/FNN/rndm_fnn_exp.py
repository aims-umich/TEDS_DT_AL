# -*- coding: utf-8 -*-
"""
Incremental random sampling with FNN (EXP is in the sampling pool):
- Treat EXP as one more "run" alongside sim files.
- Start with K random files (could include EXP), train FNN, evaluate on EXP, save metrics + plots.
- Add 'increment' more files per iteration until all are used.
- Save CSV summary of RMSE / MAE vs #files.
"""

import os, re, random
import numpy as np
import pandas as pd
import time  # <-- added for timing

# plotting (headless)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({
    "font.size": 16,
    "axes.titlesize": 20,
    "axes.labelsize": 18,
    "xtick.labelsize": 15,
    "ytick.labelsize": 15,
    "legend.fontsize": 16,
    "figure.titlesize": 22,
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
import sys
from pathlib import Path

# Find repo root by walking upward until paths.py is found
_THIS_FILE = Path(__file__).resolve()
for parent in [_THIS_FILE.parent] + list(_THIS_FILE.parents):
    if (parent / "paths.py").exists():
        REPO_ROOT = parent
        break
else:
    raise RuntimeError("Could not find repo root containing paths.py")

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
    
    
from paths import GHX_DATA_DIR, EXP_GHX_CSV, FNN_RANDOM_DIR, ensure_dirs

ensure_dirs()

DATA_DIR = GHX_DATA_DIR
EXP_CSV = EXP_GHX_CSV
OUT_DIR = FNN_RANDOM_DIR
PLOT_DIR = OUT_DIR / "plots"
PLOT_DIR.mkdir(parents=True, exist_ok=True)


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PRINT_SHAPES = True
SEED = 1234

# Random incremental sampling controls
initial_count = 10
increment     = 10

# Training hyperparams
EPOCHS      = 40
BATCH_SIZE  = 256
LR          = 1e-3
HIDDEN      = 128

# Controls: use ONLY the first 4 for training & experiment inference
CTRL_NAMES  = ["opening_PV006","mflow_pump_out","T_pump_in","T_heater_out","T_chiller_after"]
CTRL4       = CTRL_NAMES[:4]  # opening, mflow, T_pump_in, T_heater_out
STATE_NAMES = ["mflow_GHX_bypass","qghx_kW"]  # m, q

# Output dirs
RESULTS_DIR = REPO_ROOT / "results" / "result_csv"
OUT_DIR  = RESULTS_DIR / "rndmFNN_sampling_results_fnn"
PLOT_DIR = OUT_DIR / "plots"

PLOT_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)

torch.backends.cudnn.benchmark = True
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

print(f"Device: {DEVICE}")
if DEVICE == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))

# -----------------------------
# Data loading
# -----------------------------
def load_all(dirpath):
    """Return list of (rid, U:(T,4), X:(T,2)). Keeps only exact CTRL4 and STATE_NAMES."""
    runs = []
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
            runs.append((rid, U[:T], X[:T]))
    return runs

# Load all sim runs
sim_runs_all = load_all(DATA_DIR)
assert len(sim_runs_all) > 0, "No runs found in ghx_data_csv."
n_sim_files = len(sim_runs_all)
if PRINT_SHAPES:
    print(f"Loaded {n_sim_files} sim runs.")

# Load experiment once (raw) — will also be used as a "run" in the pool
dfe = pd.read_csv(EXP_CSV)
missing = [c for c in CTRL4 + STATE_NAMES if c not in dfe.columns]
if len(missing):
    raise ValueError(f"Missing columns in {EXP_CSV}: {missing}")
U_exp_4 = dfe[CTRL4].to_numpy(dtype=np.float32)
X_true  = dfe[STATE_NAMES].to_numpy(dtype=np.float32)
t_vec   = dfe["time_sec"].to_numpy(dtype=float) if "time_sec" in dfe.columns else np.arange(len(X_true), dtype=float)

mask = np.isfinite(U_exp_4).all(axis=1) & np.isfinite(X_true).all(axis=1)
U_exp_4 = U_exp_4[mask]; X_true = X_true[mask]; t_vec = t_vec[mask]
T_common = min(len(U_exp_4), len(X_true))
U_exp_4, X_true, t_vec = U_exp_4[:T_common], X_true[:T_common], t_vec[:T_common]

if PRINT_SHAPES:
    print("Experiment shapes (raw):", U_exp_4.shape, X_true.shape)

# Build a unified pool: sims + EXP as another run at the end
# sim_runs_all: list of (rid, U, X)
EXP_IDX = n_sim_files  # EXP will be the last index after append
runs_all = sim_runs_all + [("EXP", U_exp_4, X_true)]
total_files = len(runs_all)  # = n_sim_files + 1
all_indices = list(range(total_files))
print(f"Total files in pool (incl. EXP): {total_files} (EXP index = {EXP_IDX})")

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

def train_fnn(U_train_all, X_train_all):
    # Fit scalers on current training set
    u_scaler = StandardScaler().fit(U_train_all)
    y_scaler = StandardScaler().fit(X_train_all)

    U_train_n = u_scaler.transform(U_train_all).astype(np.float32)
    X_train_n = y_scaler.transform(X_train_all).astype(np.float32)

    train_ds = TensorDataset(torch.tensor(U_train_n), torch.tensor(X_train_n))
    pin = DEVICE == "cuda"
    train_ld = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, pin_memory=pin)

    model = FNN(in_dim=4, hidden=HIDDEN, out_dim=2).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    crit = nn.MSELoss()

    for ep in range(EPOCHS):
        model.train(); tot = 0.0
        for xb, yb in train_ld:
            xb = xb.to(DEVICE, non_blocking=True)
            yb = yb.to(DEVICE, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            yp = model(xb)
            loss = crit(yp, yb)
            loss.backward()
            opt.step()
            tot += loss.item() * xb.size(0)
        if PRINT_SHAPES:
            print(f"  Epoch {ep+1:02d} | train MSE={tot/len(train_ds):.6f}")

    return model, u_scaler, y_scaler

def eval_on_experiment(model, u_scaler, y_scaler):
    U_exp_n = u_scaler.transform(U_exp_4)
    with torch.no_grad():
        xb = torch.tensor(U_exp_n, dtype=torch.float32, device=DEVICE)
        Yp_n = model(xb).cpu().numpy()
    X_pred = y_scaler.inverse_transform(Yp_n)

    rmse_m = mean_squared_error(X_true[:,0], X_pred[:,0], squared=False)
    rmse_q = mean_squared_error(X_true[:,1], X_pred[:,1], squared=False)
    mae_m  = mean_absolute_error(X_true[:,0], X_pred[:,0])
    mae_q  = mean_absolute_error(X_true[:,1], X_pred[:,1])

    return X_pred, dict(rmse_m=float(rmse_m), rmse_q=float(rmse_q), mae_m=float(mae_m), mae_q=float(mae_q))

def plot_iteration(iteration, n_files, t_vec, X_true, X_pred, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)

    # m vs time
    plt.figure(figsize=(12,4))
    plt.plot(t_vec, X_true[:,0], label="m true (exp)")
    plt.plot(t_vec, X_pred[:,0], label="m pred (FNN)", alpha=0.85)
    plt.xlabel("time (s)"); plt.ylabel("mflow_GHX_bypass")
    plt.title(f"FNN: Iter {iteration} (n={n_files}) m vs time")
    plt.legend(); plt.grid(True, alpha=0.3)
    plt.tight_layout(pad=0.6)
    m_path = out_dir / f"iter_{iteration:03d}_n{n_files}_m_timeseries.png"
    #plt.savefig(m_path, dpi=300, bbox_inches="tight")
    plt.close()

    # q vs time
    plt.figure(figsize=(12,4))
    plt.plot(t_vec, X_true[:,1], label="q true (exp)")
    plt.plot(t_vec, X_pred[:,1], label="q pred (FNN)", alpha=0.85)
    plt.xlabel("time (s)"); plt.ylabel("qghx_kW")
    plt.title(f"FNN: Iter {iteration} (n={n_files}) q vs time")
    plt.legend(); plt.grid(True, alpha=0.3)
    plt.tight_layout(pad=0.6)
    q_path = out_dir / f"iter_{iteration:03d}_n{n_files}_q_timeseries.png"
    #plt.savefig(q_path, dpi=300, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(6,6))
    plt.scatter(X_true[:,0], X_pred[:,0], s=18, alpha=0.5)
    lo = float(min(X_true[:,0].min(), X_pred[:,0].min()))
    hi = float(max(X_true[:,0].max(), X_pred[:,0].max()))
    plt.plot([lo,hi],[lo,hi],"--")
    plt.xlabel("m true (exp)"); plt.ylabel("m pred (FNN)")
    plt.title(f"FNN Parity: m (n={n_files})")
    plt.grid(True, alpha=0.3)
    plt.tight_layout(pad=0.6)
    mp_path = out_dir / f"iter_{iteration:03d}_n{n_files}_m_parity.png"
    #plt.savefig(mp_path, dpi=300, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(6,6))
    plt.scatter(X_true[:,1], X_pred[:,1], s=18, alpha=0.5)
    lo = float(min(X_true[:,1].min(), X_pred[:,1].min()))
    hi = float(max(X_true[:,1].max(), X_pred[:,1].max()))
    plt.plot([lo,hi],[lo,hi],"--")
    plt.xlabel("q true (exp)"); plt.ylabel("q pred (FNN)")
    plt.title(f"FNN Parity: q (n={n_files})")
    plt.grid(True, alpha=0.3)
    plt.tight_layout(pad=0.6)
    qp_path = out_dir / f"iter_{iteration:03d}_n{n_files}_q_parity.png"
    #plt.savefig(qp_path, dpi=300, bbox_inches="tight")
    plt.close()

    return [m_path, q_path, mp_path, qp_path]

# -----------------------------
# Incremental random loop (EXP in pool)
# -----------------------------
# OPTION A: manual fixed initial indices (must be valid in [0, total_files-1])
selected_indices = [14, 21, 35, 39, 42, 57, 73, 101, 245, 299]

# OPTION B: random initial selection (could include EXP)
#selected_indices = random.sample(all_indices, min(initial_count, total_files))

remaining_indices = [i for i in all_indices if i not in selected_indices]

iteration = 0
results_rows = []

# ---- added: global stopwatch start
t0 = time.perf_counter()

while True:
    # ---- added: per-iteration start
    iter_start = time.perf_counter()

    # current selection
    sel_runs = [runs_all[i] for i in selected_indices]
    sel_ids  = [rid for (rid, _, _) in sel_runs]
    n_files  = len(sel_runs)
    print(f"\n=== Iter {iteration} | using {n_files} files ===")
    # Build big matrices from selected runs (sims and/or EXP)
    U_train = np.concatenate([U for (_,U,_) in sel_runs], axis=0)
    X_train = np.concatenate([X for (_,_,X) in sel_runs], axis=0)

    if PRINT_SHAPES:
        print("  Train shapes:", U_train.shape, X_train.shape)

    # Train FNN (fresh each iteration)
    model, u_scaler, y_scaler = train_fnn(U_train, X_train)

    # Evaluate on EXP (always)
    X_pred, metrics = eval_on_experiment(model, u_scaler, y_scaler)
    print(f"  Metrics: {metrics}")

    # Plots
    plot_paths = plot_iteration(iteration, n_files, t_vec, X_true, X_pred, PLOT_DIR)


    # done?
    if not remaining_indices:
        break

    # add 'increment' new indices from remaining
    take = min(increment, len(remaining_indices))
    new_pick = random.sample(remaining_indices, take)
    selected_indices.extend(new_pick)
    remaining_indices = [i for i in remaining_indices if i not in new_pick]
    iteration += 1
    
        # ---- added: timing metrics
    iter_runtime = time.perf_counter() - iter_start
    cum_runtime  = time.perf_counter() - t0

    # Log row
    results_rows.append({
        "iteration": iteration,
        "n_files": n_files,
        "file_ids": ";".join(map(str, sel_ids)),
        "rmse_m": metrics["rmse_m"], "mae_m": metrics["mae_m"],
        "rmse_q": metrics["rmse_q"], "mae_q": metrics["mae_q"],
        "iter_runtime_sec": float(iter_runtime),              # <-- added
        "cumulative_runtime_sec": float(cum_runtime),         # <-- added
        "plots": "|".join(plot_paths),
    })


# -----------------------------
# Save CSV summary (+ optional runtime plots)
# -----------------------------
df_out = pd.DataFrame(results_rows)
csv_path = OUT_DIR / "incremental_metrics.csv"
df_out.to_csv(csv_path, index=False)
print(f"\nSaved incremental metrics to: {csv_path}")
print(f"Plots saved in: {PLOT_DIR}")

