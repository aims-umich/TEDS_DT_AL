# -*- coding: utf-8 -*-
"""
Incremental random sampling:
- Start with 10 sim files, train GRU (NARX), predict experiment (teacher-forced), save metrics & plot.
- Add 10 more files from remaining, repeat until all used.
- Save a CSV summary of RMSE / MAE / R2 vs #files.
- Adds timing: iter_runtime_sec and cumulative_runtime_sec.
"""

import os
import re
import random
import time
import numpy as np
import pandas as pd

# headless plotting
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from contextlib import nullcontext

# -----------------------------
# Path bootstrap
# -----------------------------
import sys
from pathlib import Path

_THIS_FILE = Path(__file__).resolve()
for parent in [_THIS_FILE.parent] + list(_THIS_FILE.parents):
    if (parent / "paths.py").exists():
        REPO_ROOT = parent
        break
else:
    raise RuntimeError("Could not find repo root containing paths.py")

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from paths import GHX_DATA_DIR, EXP_GHX_CSV, GRU_RANDOM_DIR, ensure_dirs

ensure_dirs()

# -----------------------------
# Config
# -----------------------------
DATA_DIR = GHX_DATA_DIR
EXP_CSV = EXP_GHX_CSV

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 12
initial_count = 10
increment = 10

LOOKBACK = 300
HIDDEN = 256
BATCH = 512
EPOCHS = 8
LR = 1e-3
NUM_WORKERS = 4
CLIP_Z = 3.5

CTRL_NAMES = ["opening_PV006", "mflow_pump_out", "T_pump_in", "T_heater_out"]
STATE_NAMES = ["mflow_GHX_bypass", "qghx_kW"]

OUT_DIR = GRU_RANDOM_DIR
PLOT_DIR = OUT_DIR / "plots"
OUT_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)

torch.backends.cudnn.benchmark = True

print(f"Device: {DEVICE}")
if DEVICE == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))

# -----------------------------
# Helpers
# -----------------------------
def load_all_sim_runs_with_ids(dirpath):
    """Return list[(rid, U:(T,4), X:(T,2))]."""
    items = []
    for fn in sorted(os.listdir(dirpath)):
        m = re.match(r"ghx_run(\d+)\.csv$", fn)
        if not m:
            continue
        rid = int(m.group(1))
        df = pd.read_csv(dirpath / fn)
        if not set(CTRL_NAMES).issubset(df.columns) or not set(STATE_NAMES).issubset(df.columns):
            continue
        U = df[CTRL_NAMES].to_numpy(dtype=np.float32)
        X = df[STATE_NAMES].to_numpy(dtype=np.float32)
        T = min(len(U), len(X))
        if T > 0:
            items.append((rid, U[:T], X[:T]))
    return items

def pick(colnames, *cands):
    cl = [c.lower() for c in colnames]
    for cand in cands:
        c = cand.lower()
        for i, name in enumerate(cl):
            if name.startswith(c):
                return colnames[i]
    raise KeyError(f"None of {cands} found among {colnames}")

class WindowedSimNARX(Dataset):
    """
    Each sample:
      seq = concat([X_n[t-L:t,:], U_n[t-L:t,:]]) -> (L, 6)
      tgt = X_n[t,:]                             -> (2,)
    """
    def __init__(self, runs, u_scaler, y_scaler, lookback=300, clip_z=3.5):
        self.seq, self.tgt = [], []
        L = lookback
        for (_, U, X) in runs:
            if len(U) <= L:
                continue
            Un = u_scaler.transform(U).astype(np.float32)
            Xn = y_scaler.transform(X).astype(np.float32)
            np.clip(Un, -clip_z, clip_z, out=Un)
            np.clip(Xn, -clip_z, clip_z, out=Xn)
            T = len(Un)
            for t in range(L, T):
                self.seq.append(np.concatenate([Xn[t-L:t, :], Un[t-L:t, :]], axis=1))
                self.tgt.append(Xn[t, :])
        self.seq = np.asarray(self.seq, np.float32)  # (N,L,6)
        self.tgt = np.asarray(self.tgt, np.float32)  # (N,2)

    def __len__(self):
        return len(self.seq)

    def __getitem__(self, i):
        return self.seq[i], self.tgt[i]

class GRURegressor(nn.Module):
    def __init__(self, in_dim=6, hidden=256, out_dim=2):
        super().__init__()
        self.rnn = nn.GRU(in_dim, hidden, batch_first=True)
        self.head = nn.Sequential(nn.ReLU(), nn.Linear(hidden, out_dim))

    def forward(self, x):
        out, _ = self.rnn(x)
        return self.head(out[:, -1, :])

def train_one(model, train_ds, epochs=EPOCHS):
    pin = (DEVICE == "cuda")
    train_ld = DataLoader(
        train_ds,
        batch_size=BATCH,
        shuffle=True,
        pin_memory=pin,
        num_workers=NUM_WORKERS,
        persistent_workers=(NUM_WORKERS > 0)
    )
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    crit = nn.MSELoss()
    autocast_cm = torch.cuda.amp.autocast if DEVICE == "cuda" else nullcontext
    scaler = torch.cuda.amp.GradScaler(enabled=(DEVICE == "cuda"))

    model.train()
    for ep in range(1, epochs + 1):
        tot = 0.0
        for xb, yb in train_ld:
            xb = xb.to(DEVICE, non_blocking=True)
            yb = yb.to(DEVICE, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with autocast_cm():
                yp = model(xb)
                loss = crit(yp, yb)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            tot += loss.item() * xb.size(0)
        print(f"  epoch {ep:02d}  train_MSE={tot/len(train_ds):.6f}")

def predict_experiment(model, u_scaler, y_scaler):
    """Teacher-forced windows on experiment; return metrics and arrays."""
    dfe = pd.read_csv(EXP_CSV)
    cols = list(dfe.columns)

    # robust mapping for control headers
    col_open = pick(cols, "opening_pv006", "opening_p")
    col_mflow = pick(cols, "mflow_pump_out", "mflow_pun")
    col_tpin = pick(cols, "t_pump_in")
    col_theat = pick(cols, "t_heater_out", "t_heater_c")
    CTRL_EXP = [col_open, col_mflow, col_tpin, col_theat]

    # ensure states exist
    for c in STATE_NAMES:
        if c not in dfe.columns:
            raise ValueError(f"Missing state column '{c}' in {EXP_CSV}")

    U_exp = dfe[CTRL_EXP].to_numpy(dtype=np.float32)
    X_exp = dfe[STATE_NAMES].to_numpy(dtype=np.float32)
    t_vec = dfe["time_sec"].to_numpy(dtype=float) if "time_sec" in dfe.columns else np.arange(len(U_exp), dtype=float)

    mask = np.isfinite(U_exp).all(axis=1) & np.isfinite(X_exp).all(axis=1)
    U_exp, X_exp, t_vec = U_exp[mask], X_exp[mask], t_vec[mask]
    T = min(len(U_exp), len(X_exp))
    U_exp, X_exp, t_vec = U_exp[:T], X_exp[:T], t_vec[:T]
    if T <= LOOKBACK:
        raise ValueError(f"Experiment too short for LOOKBACK={LOOKBACK}: T={T}")

    # scale + clip inputs
    U_exp_n = u_scaler.transform(U_exp).astype(np.float32)
    X_exp_n = y_scaler.transform(X_exp).astype(np.float32)
    np.clip(U_exp_n, -CLIP_Z, CLIP_Z, out=U_exp_n)
    np.clip(X_exp_n, -CLIP_Z, CLIP_Z, out=X_exp_n)

    # teacher-forced windows: concat([X_hist, U_hist])
    L = LOOKBACK
    seqs = np.stack(
        [np.concatenate([X_exp_n[t-L:t, :], U_exp_n[t-L:t, :]], axis=1) for t in range(L, T)],
        axis=0
    )

    model.eval()
    with torch.no_grad():
        xb = torch.tensor(seqs, device=DEVICE)
        Yp_n = model(xb).cpu().numpy()
    Yp = y_scaler.inverse_transform(Yp_n)

    X_true = X_exp[L:, :]
    t_plot = t_vec[L:]

    # metrics
    m_true, m_pred = X_true[:, 0], Yp[:, 0]
    q_true, q_pred = X_true[:, 1], Yp[:, 1]
    metrics = {
        "rmse_m": float(np.sqrt(mean_squared_error(m_true, m_pred))),
        "mae_m": float(mean_absolute_error(m_true, m_pred)),
        "r2_m": float(r2_score(m_true, m_pred)),
        "rmse_q": float(np.sqrt(mean_squared_error(q_true, q_pred))),
        "mae_q": float(mean_absolute_error(q_true, q_pred)),
        "r2_q": float(r2_score(q_true, q_pred)),
    }
    return metrics, t_plot, m_true, m_pred, q_true, q_pred

def plot_timeseries(iteration, n_files, t, m_true, m_pred, q_true, q_pred):
    fig = plt.figure(figsize=(12, 6))

    ax1 = plt.subplot(2, 1, 1)
    ax1.plot(t, m_true, label="m true (exp)")
    ax1.plot(t, m_pred, label="m pred (GRU)", alpha=0.9)
    ax1.set_xlabel("time (s)")
    ax1.set_ylabel("mflow_GHX_bypass")
    ax1.set_title(f"Iter {iteration} (n={n_files}) m vs time")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    ax2 = plt.subplot(2, 1, 2)
    ax2.plot(t, q_true, label="q true (exp)")
    ax2.plot(t, q_pred, label="q pred (GRU)", alpha=0.9)
    ax2.set_xlabel("time (s)")
    ax2.set_ylabel("qghx_kW")
    ax2.set_title(f"Iter {iteration} (n={n_files}) q vs time")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    out = PLOT_DIR / f"iter_{iteration:03d}_n{n_files}_timeseries.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out

# -----------------------------
# Load sim runs
# -----------------------------
sim_runs_all = load_all_sim_runs_with_ids(DATA_DIR)
assert len(sim_runs_all) > 0, "No simulation runs found."
total_files = len(sim_runs_all)
all_indices = list(range(total_files))

# -----------------------------
# Random incremental loop (with timing)
# -----------------------------
random.seed(SEED)
# selected_files = random.sample(all_indices, min(initial_count, total_files))
selected_files = [14, 21, 35, 39, 42, 57, 73, 101, 245, 299]
remaining_indices = [i for i in all_indices if i not in selected_files]

iteration = 0
results = []

t0 = time.perf_counter()

while len(selected_files) <= total_files:
    iter_start = time.perf_counter()

    # slice selection
    sel_runs = [sim_runs_all[i] for i in selected_files]
    sel_ids = [rid for (rid, _, _) in sel_runs]
    n_files = len(sel_runs)
    print(f"\n=== Iter {iteration} | using {n_files} files ===")

    # fit scalers on current selection
    U_all = np.concatenate([U for (_, U, _) in sel_runs], axis=0)
    X_all = np.concatenate([X for (_, _, X) in sel_runs], axis=0)
    u_scaler = StandardScaler().fit(U_all)
    y_scaler = StandardScaler().fit(X_all)

    # dataset + model
    train_ds = WindowedSimNARX(sel_runs, u_scaler, y_scaler, LOOKBACK, CLIP_Z)
    print(f"Windows: {len(train_ds)} | seq {getattr(train_ds, 'seq', np.empty((0,))).shape}")

    model = GRURegressor(in_dim=6, hidden=HIDDEN, out_dim=2).to(DEVICE)
    train_one(model, train_ds, epochs=EPOCHS)

    # predict experiment + record metrics
    metrics, t, m_true, m_pred, q_true, q_pred = predict_experiment(model, u_scaler, y_scaler)
    plot_path = plot_timeseries(iteration, n_files, t, m_true, m_pred, q_true, q_pred)

    # timing
    iter_runtime = time.perf_counter() - iter_start
    cum_runtime = time.perf_counter() - t0

    results.append({
        "iteration": iteration,
        "n_files": n_files,
        "file_ids": ";".join(map(str, sel_ids)),
        "rmse_m": metrics["rmse_m"],
        "mae_m": metrics["mae_m"],
        "r2_m": metrics["r2_m"],
        "rmse_q": metrics["rmse_q"],
        "mae_q": metrics["mae_q"],
        "r2_q": metrics["r2_q"],
        "iter_runtime_sec": float(iter_runtime),
        "cumulative_runtime_sec": float(cum_runtime),
        "plot_path": str(plot_path),
    })

    # stop condition: all files used
    if not remaining_indices:
        break

    # add 'increment' new indices from remaining
    take = min(increment, len(remaining_indices))
    new_pick = random.sample(remaining_indices, take)
    selected_files.extend(new_pick)
    remaining_indices = [i for i in remaining_indices if i not in new_pick]
    iteration += 1

# -----------------------------
# Save CSV summary
# -----------------------------
df_out = pd.DataFrame(results)
csv_path = OUT_DIR / "incremental_metrics.csv"
df_out.to_csv(csv_path, index=False)

print(f"\nSaved incremental metrics to: {csv_path}")
print(f"Plots saved in: {PLOT_DIR}")