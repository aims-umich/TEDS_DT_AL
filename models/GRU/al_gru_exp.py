# -*- coding: utf-8 -*-
"""
Active-learning GRU (NARX-style) for GHX:
- Start with 10 random sim files, then iteratively add the 10 worst (by per-file RMSE) from the remaining pool.
- At each iteration: train on the cumulative set, evaluate on EXP (teacher-forced), save metrics + plots.
- Inputs (controls): ["opening_PV006","mflow_pump_out","T_pump_in","T_heater_out"]
- Outputs (states):  ["mflow_GHX_bypass","qghx_kW"]
- Also records per-iteration runtime and cumulative runtime.
"""

import os, re, random, math, time
import numpy as np
import pandas as pd

# headless plotting
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({
    "font.size": 14, "axes.titlesize": 18, "axes.labelsize": 16,
    "xtick.labelsize": 13, "ytick.labelsize": 13, "legend.fontsize": 14,
    "figure.titlesize": 20,
})

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# -----------------------------
# Config
# -----------------------------
DATA_DIR = "ghx_data_csv"   # folder with ghx_run{ID}.csv
EXP_CSV  = "/home/unabila/ghxSindy/experiment_csv/experiment_ghx_formatted.csv"

SEED       = 12
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
LOOKBACK   = 300
HIDDEN     = 256
BATCH      = 512
EPOCHS     = 8              # keep modest; retrains many times
LR         = 1e-3
NUM_WORKERS= 4
CLIP_Z     = 3.5
INIT_COUNT = 10
INCREMENT  = 10

CTRL_NAMES  = ["opening_PV006","mflow_pump_out","T_pump_in","T_heater_out"]
STATE_NAMES = ["mflow_GHX_bypass","qghx_kW"]

OUT_DIR = "time_gru_al_figs"
os.makedirs(OUT_DIR, exist_ok=True)

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if DEVICE == "cuda":
    torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.benchmark = True
if hasattr(torch, "set_float32_matmul_precision"):
    torch.set_float32_matmul_precision("high")

print(f"Using device: {DEVICE}")
if DEVICE == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))

# -----------------------------
# Helpers
# -----------------------------
def load_all_sim_runs(dirpath):
    runs, ids = [], []
    for fn in sorted(os.listdir(dirpath)):
        m = re.match(r"ghx_run(\d+)\.csv$", fn)
        if not m: continue
        rid = int(m.group(1))
        df = pd.read_csv(os.path.join(dirpath, fn))
        if not set(CTRL_NAMES).issubset(df.columns) or not set(STATE_NAMES).issubset(df.columns):
            continue
        U = df[CTRL_NAMES].to_numpy(dtype=np.float32)
        X = df[STATE_NAMES].to_numpy(dtype=np.float32)
        T = min(len(U), len(X))
        if T > 0:
            runs.append((U[:T], X[:T]))
            ids.append(rid)
    return runs, np.array(ids, dtype=int)

def pick(colnames, *cands):
    cl = [c.lower() for c in colnames]
    for cand in cands:
        c = cand.lower()
        for i, name in enumerate(cl):
            if name.startswith(c):
                return colnames[i]
    raise KeyError(f"None of {cands} found among {colnames}")

def mape_eps(y, yhat, eps=None):
    if eps is None:
        eps = max(1e-6, 0.02 * (np.nanmax(np.abs(y)) - np.nanmin(np.abs(y))))
    denom = np.maximum(np.abs(y), eps)
    return np.mean(np.abs(yhat - y) / denom) * 100.0

# -----------------------------
# Dataset
# -----------------------------
class WindowedSimNARX(Dataset):
    """
    Each sample:
      seq = concat([X_n[t-L:t,:], U_n[t-L:t,:]]) -> (L, 6)
      tgt = X_n[t,:]                             -> (2,)
    """
    def __init__(self, runs, u_scaler, y_scaler, lookback=300, clip_z=3.5):
        self.seq, self.tgt = [], []
        L = lookback
        for (U, X) in runs:
            if len(U) <= L: 
                continue
            Un = u_scaler.transform(U).astype(np.float32)
            Xn = y_scaler.transform(X).astype(np.float32)
            np.clip(Un, -clip_z, clip_z, out=Un)
            np.clip(Xn, -clip_z, clip_z, out=Xn)
            T = len(Un)
            for t in range(L, T):
                self.seq.append(np.concatenate([Xn[t-L:t,:], Un[t-L:t,:]], axis=1))
                self.tgt.append(Xn[t,:])
        self.seq = np.asarray(self.seq, np.float32)  # (N,L,6)
        self.tgt = np.asarray(self.tgt, np.float32)  # (N,2)

    def __len__(self): return len(self.seq)
    def __getitem__(self, i): return self.seq[i], self.tgt[i]

# -----------------------------
# Model
# -----------------------------
class GRURegressor(nn.Module):
    def __init__(self, in_dim=6, hidden=256, out_dim=2, num_layers=1, dropout=0.0):
        super().__init__()
        self.rnn  = nn.GRU(in_dim, hidden, num_layers=num_layers,
                           dropout=(dropout if num_layers>1 else 0.0),
                           batch_first=True)
        self.head = nn.Sequential(nn.ReLU(), nn.Linear(hidden, out_dim))
    def forward(self, x):            # x: (B,L,in_dim)
        out, _ = self.rnn(x)         # (B,L,H)
        hL = out[:, -1, :]           # last step
        return self.head(hL)         # (B,2)

# -----------------------------
# Train one model on a subset of runs
# -----------------------------
def train_on_runs(runs_subset, epochs=EPOCHS):
    # Fit scalers on THIS subset
    U_all = np.concatenate([U for (U,_) in runs_subset if len(U)>0], axis=0)
    X_all = np.concatenate([X for (_,X) in runs_subset if len(X)>0], axis=0)
    u_scaler = StandardScaler().fit(U_all)
    y_scaler = StandardScaler().fit(X_all)

    # Dataset & loader
    ds = WindowedSimNARX(runs_subset, u_scaler, y_scaler, LOOKBACK, CLIP_Z)
    if len(ds) == 0:
        raise ValueError("Training subset produced 0 windows (too short vs LOOKBACK).")
    pin = (DEVICE == "cuda")
    ld  = DataLoader(ds, batch_size=BATCH, shuffle=True,
                     pin_memory=pin, num_workers=NUM_WORKERS,
                     persistent_workers=(NUM_WORKERS > 0))

    # Model
    model = GRURegressor(in_dim=6, hidden=HIDDEN, out_dim=2).to(DEVICE)
    opt   = torch.optim.Adam(model.parameters(), lr=LR)
    crit  = nn.MSELoss()

    # Mixed precision if CUDA
    use_amp = (DEVICE == "cuda")
    if use_amp:
        try:
            from torch.amp import autocast as autocast_new, GradScaler as GradScalerNew
            autocast_cm = lambda: autocast_new(device_type="cuda")
            scaler      = GradScalerNew(device_type="cuda")
        except Exception:
            from torch.cuda.amp import autocast as autocast_old, GradScaler as GradScalerOld
            autocast_cm = autocast_old
            scaler      = GradScalerOld()
    else:
        from contextlib import nullcontext
        autocast_cm = nullcontext
        scaler = None

    # Train
    for ep in range(1, epochs+1):
        model.train(); tot = 0.0
        for xb, yb in ld:
            xb = xb.to(DEVICE, non_blocking=True)
            yb = yb.to(DEVICE, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            if scaler:
                with autocast_cm():
                    yp = model(xb); loss = crit(yp, yb)
                scaler.scale(loss).backward()
                scaler.step(opt); scaler.update()
            else:
                yp = model(xb); loss = crit(yp, yb)
                loss.backward(); opt.step()
            tot += loss.item() * xb.size(0)
        print(f"  epoch {ep:02d} | train MSE={tot/len(ds):.6f}")

    return model, u_scaler, y_scaler

# -----------------------------
# Evaluate per-sim-file RMSE (teacher-forced)
# -----------------------------
def per_file_rmse(model, u_scaler, y_scaler, run):
    U, X = run
    if len(U) <= LOOKBACK:
        return np.nan, np.nan, np.inf   # skip short files
    Un = u_scaler.transform(U).astype(np.float32)
    Xn = y_scaler.transform(X).astype(np.float32)
    np.clip(Un, -CLIP_Z, CLIP_Z, out=Un)
    np.clip(Xn, -CLIP_Z, CLIP_Z, out=Xn)
    L = LOOKBACK
    seqs = np.stack([np.concatenate([Xn[t-L:t,:], Un[t-L:t,:]], axis=1)
                     for t in range(L, len(Un))], axis=0)    # (T-L, L, 6)
    with torch.no_grad():
        xb   = torch.tensor(seqs, device=DEVICE)
        Yp_n = model(xb).cpu().numpy()
    Yp = y_scaler.inverse_transform(Yp_n)
    X_true = X[L:, :]
    rmse_m = mean_squared_error(X_true[:,0], Yp[:,0], squared=False)
    rmse_q = mean_squared_error(X_true[:,1], Yp[:,1], squared=False)
    combined = 0.5 * (rmse_m + rmse_q)
    return rmse_m, rmse_q, combined

# -----------------------------
# Evaluate on EXP (teacher-forced) + plots
# -----------------------------
def eval_experiment_and_plot(iter_idx, n_train_files, model, u_scaler, y_scaler):
    dfe = pd.read_csv(EXP_CSV)
    cols = list(dfe.columns)

    # robust mapping from exp headers
    col_open  = pick(cols, "opening_pv006", "opening_p")
    col_mflow = pick(cols, "mflow_pump_out", "mflow_pun")
    col_tpin  = pick(cols, "t_pump_in")
    col_theat = pick(cols, "t_heater_out", "t_heater_c")
    ctrl_exp  = [col_open, col_mflow, col_tpin, col_theat]
    for c in STATE_NAMES:
        if c not in dfe.columns:
            raise ValueError(f"Missing state column '{c}' in {EXP_CSV}")

    U_exp = dfe[ctrl_exp].to_numpy(dtype=np.float32)
    X_exp = dfe[STATE_NAMES].to_numpy(dtype=np.float32)
    t_vec = dfe["time_sec"].to_numpy(dtype=float) if "time_sec" in dfe.columns else np.arange(len(U_exp), dtype=float)

    mask = np.isfinite(U_exp).all(axis=1) & np.isfinite(X_exp).all(axis=1)
    U_exp, X_exp, t_vec = U_exp[mask], X_exp[mask], t_vec[mask]
    T = min(len(U_exp), len(X_exp))
    U_exp, X_exp, t_vec = U_exp[:T], X_exp[:T], t_vec[:T]
    if T <= LOOKBACK:
        raise ValueError(f"Experiment too short for LOOKBACK={LOOKBACK}: T={T}")

    # scale/clip
    U_exp_n = u_scaler.transform(U_exp).astype(np.float32)
    X_exp_n = y_scaler.transform(X_exp).astype(np.float32)
    np.clip(U_exp_n, -CLIP_Z, CLIP_Z, out=U_exp_n)
    np.clip(X_exp_n, -CLIP_Z, CLIP_Z, out=X_exp_n)

    L = LOOKBACK
    seqs = np.stack([np.concatenate([X_exp_n[t-L:t,:], U_exp_n[t-L:t,:]], axis=1)
                     for t in range(L, T)], axis=0)         # (T-L, L, 6)

    model.eval()
    with torch.no_grad():
        xb   = torch.tensor(seqs, device=DEVICE)
        Yp_n = model(xb).cpu().numpy()
    Yp     = y_scaler.inverse_transform(Yp_n)
    X_true = X_exp[L:, :]
    t_plot = t_vec[L:]

    # metrics
    rmse_m = mean_squared_error(X_true[:,0], Yp[:,0], squared=False)
    rmse_q = mean_squared_error(X_true[:,1], Yp[:,1], squared=False)
    mae_m  = mean_absolute_error(X_true[:,0], Yp[:,0])
    mae_q  = mean_absolute_error(X_true[:,1], Yp[:,1])
    r2_m   = r2_score(X_true[:,0], Yp[:,0])
    r2_q   = r2_score(X_true[:,1], Yp[:,1])

    print(f"[Iter {iter_idx}] files={n_train_files} | EXP: rmse_m={rmse_m:.4f} rmse_q={rmse_q:.4f} "
          f"mae_m={mae_m:.4f} mae_q={mae_q:.4f} r2_m={r2_m:.4f} r2_q={r2_q:.4f}")

    # plots
    tag = f"iter_{iter_idx:03d}_n{n_train_files}"
    os.makedirs(OUT_DIR, exist_ok=True)

    # m vs time
    plt.figure(figsize=(12,4))
    plt.plot(t_plot, X_true[:,0], label="m true (exp)")
    plt.plot(t_plot, Yp[:,0], label="m pred", alpha=0.85)
    plt.xlabel("time (s)"); plt.ylabel("mflow_GHX_bypass")
    plt.title(f"Experiment m: true vs pred | {tag}")
    plt.legend(); plt.grid(True, alpha=0.3)
    plt.tight_layout(pad=0.6)
    plt.savefig(os.path.join(OUT_DIR, f"{tag}_m_timeseries.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # q vs time
    plt.figure(figsize=(12,4))
    plt.plot(t_plot, X_true[:,1], label="q true (exp)")
    plt.plot(t_plot, Yp[:,1], label="q pred", alpha=0.85)
    plt.xlabel("time (s)"); plt.ylabel("qghx_kW")
    plt.title(f"Experiment q: true vs pred | {tag}")
    plt.legend(); plt.grid(True, alpha=0.3)
    plt.tight_layout(pad=0.6)
    plt.savefig(os.path.join(OUT_DIR, f"{tag}_q_timeseries.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # Return metrics for CSV
    return dict(
        iter=iter_idx, n_files=n_train_files,
        rmse_m=float(rmse_m), rmse_q=float(rmse_q),
        mae_m=float(mae_m),   mae_q=float(mae_q),
        r2_m=float(r2_m),     r2_q=float(r2_q)
    )

# -----------------------------
# MAIN: Active-learning loop (with timing)
# -----------------------------
runs, file_ids = load_all_sim_runs(DATA_DIR)
total_files = len(runs)
assert total_files > 0, "No simulation runs found."

all_indices      = list(range(total_files))
# selected_indices = random.sample(all_indices, min(INIT_COUNT, total_files))
selected_indices = [14, 21, 35, 39, 42, 57, 73, 101, 245, 299]
remaining_indices= [i for i in all_indices if i not in selected_indices]

print(f"Total files: {total_files}")
print(f"Initial selected: {len(selected_indices)} | Remaining: {len(remaining_indices)}")

results = []
iteration = 0

t0 = time.perf_counter()  # global start

while True:
    iter_start = time.perf_counter()

    iteration += 1
    # Build current training subset
    train_subset = [runs[i] for i in selected_indices]
    print(f"\n=== Iter {iteration} | training with {len(selected_indices)} files ===")
    model, u_scaler, y_scaler = train_on_runs(train_subset, epochs=EPOCHS)

    # Evaluate EXP & save plots/metrics
    metrics = eval_experiment_and_plot(iteration, len(selected_indices), model, u_scaler, y_scaler)



    # If no remaining files, we're done
    if len(remaining_indices) == 0:
        break

    # Score remaining files (teacher-forced) and pick worst K
    print("  Scoring remaining simulation files...")
    perfile = []
    for idx in remaining_indices:
        rmse_m, rmse_q, combined = per_file_rmse(model, u_scaler, y_scaler, runs[idx])
        perfile.append((idx, rmse_m, rmse_q, combined))

    # Filter out NaNs (too-short files) from selection
    perfile = [(i, rm, rq, c) for (i, rm, rq, c) in perfile if np.isfinite(c)]
    if len(perfile) == 0:
        print("No valid remaining files to add (likely all too short). Stopping.")
        break

    perfile.sort(key=lambda x: x[3], reverse=True)  # worst first by combined RMSE
    k = min(INCREMENT, len(perfile))
    new_pick = [i for (i, _, _, _) in perfile[:k]]

    # Update pools
    selected_indices.extend(new_pick)
    remaining_indices = [i for i in remaining_indices if i not in new_pick]

    print(f"  Added {k} worst files. Now selected={len(selected_indices)} remaining={len(remaining_indices)}")
    
        # timing
    iter_runtime = time.perf_counter() - iter_start
    cum_runtime  = time.perf_counter() - t0
    metrics["iter_runtime_sec"] = float(iter_runtime)
    metrics["cumulative_runtime_sec"] = float(cum_runtime)

    results.append(metrics)

# -----------------------------
# Save summary CSV + curves
# -----------------------------
df = pd.DataFrame(results).sort_values("n_files").reset_index(drop=True)
csv_path = os.path.join(OUT_DIR, "active_learning_exp_metrics.csv")
df.to_csv(csv_path, index=False)
print(f"\nSaved experiment metrics to {csv_path}")

# Curves: RMSE & MAE vs #files (optional – unchanged)
plt.figure(figsize=(10,5))
plt.plot(df["n_files"], df["rmse_m"], marker="o", label="RMSE m")
plt.plot(df["n_files"], df["rmse_q"], marker="o", label="RMSE q")
plt.xlabel("# simulation files used for training (cumulative)")
plt.ylabel("RMSE")
plt.title("EXP RMSE vs # training files (active learning)")
plt.grid(True, alpha=0.3); plt.legend(); plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "exp_rmse_vs_files.png"), dpi=300); plt.close()

plt.figure(figsize=(10,5))
plt.plot(df["n_files"], df["mae_m"], marker="o", label="MAE m")
plt.plot(df["n_files"], df["mae_q"], marker="o", label="MAE q")
plt.xlabel("# simulation files used for training (cumulative)")
plt.ylabel("MAE")
plt.title("EXP MAE vs # training files (active learning)")
plt.grid(True, alpha=0.3); plt.legend(); plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "exp_mae_vs_files.png"), dpi=300); plt.close()

print(f"Saved plots to {OUT_DIR}/")
