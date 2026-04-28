# -*- coding: utf-8 -*-
"""
GRU for GHX: train on simulation CSVs, test on held-out simulation files.
- Keeps (rid, U, X) with every run (fixes unpack error).
- Toggle USE_X_IN_WINDOW to include past states in the GRU input.
- Uses GPU + mixed precision if available.
- Saves time-series + parity plots for a chosen test run ID.
"""

import os
import re
import numpy as np
import pandas as pd

# Headless plotting
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
from torch.utils.data import Dataset, DataLoader

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

from paths import GHX_DATA_DIR, GRU_RESULTS_DIR, ensure_dirs

ensure_dirs()

# -----------------------------
# Config
# -----------------------------
DATA_DIR   = GHX_DATA_DIR
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
PRINT_INFO = True

# Choose controls (4 by default; add T_chiller_after if you want 5)
CTRL_NAMES  = ["opening_PV006", "mflow_pump_out", "T_pump_in", "T_heater_out"]
# CTRL_NAMES  = ["opening_PV006", "mflow_pump_out", "T_pump_in", "T_heater_out", "T_chiller_after"]
STATE_NAMES = ["mflow_GHX_bypass", "qghx_kW"]

# Windowing + training
USE_X_IN_WINDOW = True
LOOKBACK        = 120
BATCH           = 512
EPOCHS          = 10
LR              = 1e-3
NUM_WORKERS     = 4

# What test file to plot/evaluate
TEST_FILE_IDS_TO_EVAL = [176]
PLOT_FILE_ID          = 176

# Output
RESULTS_DIR = GRU_RESULTS_DIR
FIG_DIR = RESULTS_DIR
PRED_DIR = RESULTS_DIR
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Performance knobs
torch.backends.cudnn.benchmark = True
if hasattr(torch, "set_float32_matmul_precision"):
    torch.set_float32_matmul_precision("high")

if PRINT_INFO:
    print(f"Using device: {DEVICE}")
    if DEVICE == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

# -----------------------------
# Load all runs as (rid, U, X)
# -----------------------------
def load_all(dirpath):
    runs = []   # list of (rid, U:(T,nc), X:(T,2))
    for fn in sorted(os.listdir(dirpath)):
        m = re.match(r"ghx_run(\d+)\.csv$", fn)
        if not m:
            continue
        rid = int(m.group(1))
        df = pd.read_csv(dirpath / fn)
        U = df[CTRL_NAMES].to_numpy(dtype=np.float32)
        X = df[STATE_NAMES].to_numpy(dtype=np.float32)
        T = min(len(U), len(X))
        if T == 0:
            continue
        runs.append((rid, U[:T], X[:T]))
    return runs

runs = load_all(DATA_DIR)
assert len(runs) > 0, "No runs found in ghx_data_csv."
if PRINT_INFO:
    print(f"Loaded {len(runs)} runs.")

# -----------------------------
# Split by file IDs
# -----------------------------
test_indices = {12, 56, 74, 81, 121, 136, 176, 233, 266, 300}

all_ids = {rid for (rid, _, _) in runs}
missing = sorted(test_indices - all_ids)
if missing:
    raise ValueError(f"IDs not present in runs: {missing}")

train_runs = [(rid, U, X) for (rid, U, X) in runs if rid not in test_indices]
test_runs  = [(rid, U, X) for (rid, U, X) in runs if rid in test_indices]

assert not ({rid for (rid, _, _) in train_runs} & {rid for (rid, _, _) in test_runs})
print(f"Train: {len(train_runs)} | Test: {len(test_runs)}")

test_ids = {rid for (rid, _, _) in test_runs}

# -----------------------------
# Fit scalers on TRAIN only
# -----------------------------
U_train_flat = np.concatenate([U for (_, U, _) in train_runs], axis=0)
X_train_flat = np.concatenate([X for (_, _, X) in train_runs], axis=0)

u_scaler = StandardScaler().fit(U_train_flat)
y_scaler = StandardScaler().fit(X_train_flat)

# -----------------------------
# Windowed datasets
# -----------------------------
class WindowedRuns(Dataset):
    """
    If USE_X_IN_WINDOW:
        input = concat([X[t-L:t,:], U[t-L:t,:]]) -> (L, 2+nc)
    else:
        input = U[t-L:t,:] -> (L, nc)
    target = X[t,:] -> (2,)
    """
    def __init__(self, runs_with_ids, u_scaler, y_scaler, lookback=120, use_x=False):
        self.seq = []
        self.tgt = []
        L = lookback
        self.use_x = use_x

        for (_, U, X) in runs_with_ids:
            if len(U) <= L:
                continue

            U_n = u_scaler.transform(U).astype(np.float32)
            X_n = y_scaler.transform(X).astype(np.float32)
            T = len(U_n)

            for t in range(L, T):
                if use_x:
                    x_win = X_n[t-L:t, :]
                    u_win = U_n[t-L:t, :]
                    seq = np.concatenate([x_win, u_win], axis=1)
                else:
                    seq = U_n[t-L:t, :]

                self.seq.append(seq)
                self.tgt.append(X_n[t, :])

        self.seq = np.asarray(self.seq, dtype=np.float32)
        self.tgt = np.asarray(self.tgt, dtype=np.float32)

    def __len__(self):
        return len(self.seq)

    def __getitem__(self, i):
        return self.seq[i], self.tgt[i]

train_ds = WindowedRuns(train_runs, u_scaler, y_scaler, LOOKBACK, USE_X_IN_WINDOW)
pin = (DEVICE == "cuda")
train_ld = DataLoader(
    train_ds,
    batch_size=BATCH,
    shuffle=True,
    pin_memory=pin,
    num_workers=NUM_WORKERS,
    persistent_workers=(NUM_WORKERS > 0)
)

if PRINT_INFO:
    print(f"Train windows: {len(train_ds)} | seq {train_ds.seq.shape} | tgt {train_ds.tgt.shape}")

# -----------------------------
# GRU model
# -----------------------------
IN_DIM = (2 + len(CTRL_NAMES)) if USE_X_IN_WINDOW else len(CTRL_NAMES)

class GRURegressor(nn.Module):
    def __init__(self, in_dim, hidden=128, out_dim=2, num_layers=1, dropout=0.0):
        super().__init__()
        self.rnn = nn.GRU(
            input_size=in_dim,
            hidden_size=hidden,
            num_layers=num_layers,
            dropout=(dropout if num_layers > 1 else 0.0),
            batch_first=True
        )
        self.head = nn.Sequential(
            nn.ReLU(),
            nn.Linear(hidden, out_dim)
        )

    def forward(self, x):
        out, _ = self.rnn(x)
        hL = out[:, -1, :]
        return self.head(hL)

model = GRURegressor(in_dim=IN_DIM, hidden=128, out_dim=2).to(DEVICE)
opt = torch.optim.Adam(model.parameters(), lr=LR)
crit = nn.MSELoss()

# Mixed precision
use_amp = (DEVICE == "cuda")
if use_amp:
    try:
        from torch.amp import autocast as autocast_new, GradScaler as GradScalerNew
        autocast_cm = lambda: autocast_new(device_type="cuda")
        scaler = GradScalerNew(device_type="cuda")
    except Exception:
        from torch.cuda.amp import autocast as autocast_old, GradScaler as GradScalerOld
        autocast_cm = autocast_old
        scaler = GradScalerOld()
else:
    from contextlib import nullcontext
    autocast_cm = nullcontext
    scaler = None

if PRINT_INFO:
    print("Model on:", next(model.parameters()).device)
    print(f"Input dim: {IN_DIM}  (USE_X_IN_WINDOW={USE_X_IN_WINDOW})")

# -----------------------------
# Train
# -----------------------------
for ep in range(EPOCHS):
    model.train()
    tot = 0.0
    for xb, yb in train_ld:
        xb = xb.to(DEVICE, non_blocking=True)
        yb = yb.to(DEVICE, non_blocking=True)
        opt.zero_grad(set_to_none=True)

        if scaler:
            with autocast_cm():
                yp = model(xb)
                loss = crit(yp, yb)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        else:
            yp = model(xb)
            loss = crit(yp, yb)
            loss.backward()
            opt.step()

        tot += loss.item() * xb.size(0)

    if PRINT_INFO:
        print(f"Epoch {ep+1:02d} | train MSE={tot/len(train_ds):.6f}")

# -----------------------------
# Evaluate selected test files
# -----------------------------
TEST_FILE_IDS_TO_EVAL = [rid for rid in TEST_FILE_IDS_TO_EVAL if rid in test_ids]
assert len(TEST_FILE_IDS_TO_EVAL) > 0, "Requested test IDs are not in the test split."

metrics_rows  = []
all_pred_rows = []
plot_done = False

model.eval()
with torch.no_grad():
    for rid, U, X in test_runs:
        if rid not in TEST_FILE_IDS_TO_EVAL:
            continue
        if len(U) <= LOOKBACK:
            print(f"Skip file {rid}: too short (T={len(U)} <= L={LOOKBACK})")
            continue

        # normalize whole file
        U_n = u_scaler.transform(U).astype(np.float32)
        X_n = y_scaler.transform(X).astype(np.float32)

        # build windows
        seqs = []
        for t in range(LOOKBACK, len(U_n)):
            if USE_X_IN_WINDOW:
                seq = np.concatenate([X_n[t-LOOKBACK:t, :], U_n[t-LOOKBACK:t, :]], axis=1)
            else:
                seq = U_n[t-LOOKBACK:t, :]
            seqs.append(seq)
        seqs = np.asarray(seqs, dtype=np.float32)

        xb = torch.tensor(seqs, device=DEVICE)
        Yp_n = model(xb).cpu().numpy()
        Yp = y_scaler.inverse_transform(Yp_n)

        X_true = X[LOOKBACK:, :]
        T_eff = len(Yp)

        # try real time if present
        csv_path = DATA_DIR / f"ghx_run{rid}.csv"
        try:
            df_src = pd.read_csv(csv_path)
            if "time_s" in df_src.columns:
                time_s = df_src["time_s"].values[LOOKBACK:LOOKBACK + T_eff]
            else:
                time_s = np.arange(LOOKBACK, LOOKBACK + T_eff)
        except Exception:
            time_s = np.arange(LOOKBACK, LOOKBACK + T_eff)

        # save per-file predictions
        df_pred = pd.DataFrame({
            "rid": rid,
            "t_index": np.arange(LOOKBACK, LOOKBACK + T_eff, dtype=int),
            "time_s": time_s,
            "m_true": X_true[:, 0],
            "m_pred": Yp[:, 0],
            "q_true": X_true[:, 1],
            "q_pred": Yp[:, 1],
        })
        out_file = PRED_DIR / f"pred_{rid}.csv"
        df_pred.to_csv(out_file, index=False)
        all_pred_rows.append(df_pred)
        print(f"[SAVED] {out_file}")

        # metrics
        rmse_m = mean_squared_error(X_true[:, 0], Yp[:, 0], squared=False)
        rmse_q = mean_squared_error(X_true[:, 1], Yp[:, 1], squared=False)
        mae_m  = mean_absolute_error(X_true[:, 0], Yp[:, 0])
        mae_q  = mean_absolute_error(X_true[:, 1], Yp[:, 1])
        metrics_rows.append([rid, rmse_m, rmse_q, mae_m, mae_q])

        # plots
        if not plot_done and rid == PLOT_FILE_ID:
            t = time_s.astype(float)

            plt.figure(figsize=(10, 3))
            plt.plot(t, X_true[:, 0], label="m true")
            plt.plot(t, Yp[:, 0], label="m pred (GRU)", alpha=0.85)
            plt.xlabel("time (s)")
            plt.ylabel("mflow_GHX_bypass")
            plt.title(f"File {rid} m: true vs pred (GRU, L={LOOKBACK})")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(FIG_DIR / f"file_{rid}_m_timeseries.png", dpi=300)
            plt.close()

            plt.figure(figsize=(10, 3))
            plt.plot(t, X_true[:, 1], label="q true")
            plt.plot(t, Yp[:, 1], label="q pred (GRU)", alpha=0.85)
            plt.xlabel("time (s)")
            plt.ylabel("qghx_kW")
            plt.title(f"File {rid} q: true vs pred (GRU, L={LOOKBACK})")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(FIG_DIR / f"file_{rid}_q_timeseries.png", dpi=300)
            plt.close()

            plt.figure(figsize=(5, 5))
            plt.scatter(X_true[:, 0], Yp[:, 0], s=6, alpha=0.45)
            lo, hi = float(min(X_true[:, 0].min(), Yp[:, 0].min())), float(max(X_true[:, 0].max(), Yp[:, 0].max()))
            plt.plot([lo, hi], [lo, hi], "--")
            plt.xlabel("m true")
            plt.ylabel("m pred (GRU)")
            plt.title(f"File {rid} parity m")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(FIG_DIR / f"file_{rid}_m_parity.png", dpi=300)
            plt.close()

            plt.figure(figsize=(5, 5))
            plt.scatter(X_true[:, 1], Yp[:, 1], s=6, alpha=0.45)
            lo, hi = float(min(X_true[:, 1].min(), Yp[:, 1].min())), float(max(X_true[:, 1].max(), Yp[:, 1].max()))
            plt.plot([lo, hi], [lo, hi], "--")
            plt.xlabel("q true")
            plt.ylabel("q pred (GRU)")
            plt.title(f"File {rid} parity q")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(FIG_DIR / f"file_{rid}_q_parity.png", dpi=300)
            plt.close()

            plot_done = True

# -----------------------------
# Print/save metrics
# -----------------------------
if len(metrics_rows):
    print("\n== Per-file metrics (GRU) ==")
    for rid, rm_m, rm_q, ma_m, ma_q in metrics_rows:
        print(f"File {rid:>4d} | RMSE m={rm_m:.3f} q={rm_q:.3f} | MAE m={ma_m:.3f} q={ma_q:.3f}")

    df_metrics = pd.DataFrame(metrics_rows, columns=["rid", "rmse_m", "rmse_q", "mae_m", "mae_q"])
    df_metrics.to_csv(PRED_DIR / "gru_metrics.csv", index=False)
    print(f"[SAVED] {PRED_DIR / 'gru_metrics.csv'}")
else:
    print("No selected test files were evaluated. Check TEST_FILE_IDS_TO_EVAL and test split.")