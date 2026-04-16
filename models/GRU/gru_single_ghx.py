# -*- coding: utf-8 -*-
import os, re
import numpy as np
import pandas as pd

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
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from contextlib import nullcontext

# -----------------------------
# Config
# -----------------------------
DATA_DIR = "ghx_data_csv"  # folder with ghx_run{idx}.csv
DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"
PRINT_SHAPES = True

# Pick test files to visualize/evaluate
TEST_FILE_IDS_TO_EVAL = [300]
PLOT_FILE_ID          = 300

CTRL_NAMES  = ["opening_PV006","mflow_pump_out","T_pump_in","T_heater_out"]
STATE_NAMES = ["mflow_GHX_bypass","qghx_kW"]  # m, q

LOOKBACK = 120     # seconds of control history
BATCH    = 256
EPOCHS   = 30
LR       = 1e-3
HIDDEN   = 128
NUM_WORKERS = 4

# Performance knobs
torch.backends.cudnn.benchmark = True
if hasattr(torch, "set_float32_matmul_precision"):
    torch.set_float32_matmul_precision("high")

if PRINT_SHAPES:
    print(f"Using device: {DEVICE}")
    if DEVICE == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

# -----------------------------
# Load all runs
# -----------------------------
def load_all(dirpath):
    runs = []   # [(rid, U:(T,4), X:(T,2))]
    for fn in sorted(os.listdir(dirpath)):
        m = re.match(r"ghx_run(\d+)\.csv$", fn)
        if not m:
            continue
        rid = int(m.group(1))
        df = pd.read_csv(os.path.join(dirpath, fn))
        U = df[CTRL_NAMES].to_numpy(dtype=np.float32)
        X = df[STATE_NAMES].to_numpy(dtype=np.float32)
        T = min(len(U), len(X))
        runs.append((rid, U[:T], X[:T]))
    return runs

runs = load_all(DATA_DIR)
assert len(runs) > 0, "No runs found in ghx_data_csv."
if PRINT_SHAPES:
    print(f"Loaded {len(runs)} runs.")

# -----------------------------
# Split by files (80/20)
# -----------------------------
all_ids    = np.array([rid for rid,_,_ in runs], dtype=int)
sorted_ids = np.sort(all_ids)
n_train    = int(0.8 * len(sorted_ids))
train_ids  = set(sorted_ids[:n_train])
test_ids   = set(sorted_ids[n_train:])

if PRINT_SHAPES:
    print("Test IDs sample:", sorted(list(test_ids))[:10], "...", sorted(list(test_ids))[-10:])
    print("Total test files:", len(test_ids))

train_runs = [(rid,U,X) for (rid,U,X) in runs if rid in train_ids]
test_runs  = [(rid,U,X) for (rid,U,X) in runs if rid in test_ids]

if PRINT_SHAPES:
    print(f"Train files: {len(train_runs)} | Test files: {len(test_runs)}")

# -----------------------------
# Fit scalers on TRAIN only (flat)
# -----------------------------
U_train_flat = np.concatenate([U for _,U,_ in train_runs], axis=0)
X_train_flat = np.concatenate([X for _,_,X in train_runs], axis=0)

u_scaler = StandardScaler().fit(U_train_flat)
y_scaler = StandardScaler().fit(X_train_flat)

# -----------------------------
# Windowed dataset for GRU
# -----------------------------
class WindowedRuns(Dataset):
    """
    Each item: (seq, y)
      seq: (LOOKBACK, n_ctrl)  # past controls U[t-L:t]
      y  : (n_state,)          # state X[t]
    """
    def __init__(self, runs, u_scaler, y_scaler, lookback=120):
        self.seq = []
        self.tgt = []
        L = lookback
        for (_, U, X) in runs:
            if len(U) <= L:
                continue
            U_n = u_scaler.transform(U).astype(np.float32)
            X_n = y_scaler.transform(X).astype(np.float32)
            T = len(U_n)
            for t in range(L, T):
                self.seq.append(U_n[t-L:t, :])  # history [t-L, t)
                self.tgt.append(X_n[t, :])     # target at t
        self.seq = np.asarray(self.seq, dtype=np.float32)   # (N, L, n_ctrl)
        self.tgt = np.asarray(self.tgt, dtype=np.float32)   # (N, n_state)

    def __len__(self): return len(self.seq)
    def __getitem__(self, i):
        return self.seq[i], self.tgt[i]

train_ds = WindowedRuns(train_runs, u_scaler, y_scaler, LOOKBACK)
pin = (DEVICE == "cuda")
train_ld = DataLoader(
    train_ds,
    batch_size=BATCH,
    shuffle=True,
    pin_memory=pin,
    num_workers=NUM_WORKERS,
    persistent_workers=(NUM_WORKERS > 0)
)

if PRINT_SHAPES:
    print(f"Train windows: {len(train_ds)} | seq {train_ds.seq.shape} | tgt {train_ds.tgt.shape}")

# -----------------------------
# GRU model
# -----------------------------
class GRURegressor(nn.Module):
    def __init__(self, in_dim, hidden=128, out_dim=2, num_layers=1, dropout=0.0):
        super().__init__()
        self.rnn  = nn.GRU(input_size=in_dim, hidden_size=hidden,
                           num_layers=num_layers, dropout=(dropout if num_layers>1 else 0.0),
                           batch_first=True)
        self.head = nn.Sequential(
            nn.ReLU(),
            nn.Linear(hidden, out_dim)
        )
    def forward(self, x):            # x: (B, L, in_dim)
        out, _ = self.rnn(x)         # (B, L, hidden)
        hL = out[:, -1, :]           # last step summary
        return self.head(hL)         # (B, out_dim)

model = GRURegressor(in_dim=len(CTRL_NAMES), hidden=HIDDEN, out_dim=len(STATE_NAMES)).to(DEVICE)
opt    = torch.optim.Adam(model.parameters(), lr=LR)
crit   = nn.MSELoss()

# Mixed precision on CUDA
autocast = torch.cuda.amp.autocast if DEVICE == "cuda" else nullcontext
scaler   = torch.cuda.amp.GradScaler(enabled=(DEVICE=="cuda"))

if PRINT_SHAPES:
    print("Model on:", next(model.parameters()).device)

# -----------------------------
# Train
# -----------------------------
for ep in range(EPOCHS):
    model.train(); tot = 0.0
    for xb, yb in train_ld:
        xb = xb.to(DEVICE, non_blocking=True)   # (B,L,D)
        yb = yb.to(DEVICE, non_blocking=True)   # (B,2)
        opt.zero_grad(set_to_none=True)
        with autocast():
            yp = model(xb)
            loss = crit(yp, yb)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        tot += loss.item() * xb.size(0)
    if PRINT_SHAPES:
        print(f"Epoch {ep+1:02d} | train MSE={tot/len(train_ds):.6f}")

# -----------------------------
# Evaluate ONLY on selected test files
# -----------------------------
TEST_FILE_IDS_TO_EVAL = [rid for rid in TEST_FILE_IDS_TO_EVAL if rid in test_ids]
assert len(TEST_FILE_IDS_TO_EVAL) > 0, "Requested test IDs are not in the test split."

metrics_rows = []
plot_done = False
os.makedirs("figs_gru", exist_ok=True)

model.eval()
with torch.no_grad(), autocast():
    for rid, U, X in test_runs:
        if rid not in TEST_FILE_IDS_TO_EVAL:
            continue

        if len(U) <= LOOKBACK:
            print(f"Skip file {rid}: too short (T={len(U)} <= L={LOOKBACK})")
            continue

        U_n = u_scaler.transform(U).astype(np.float32)
        X_n = y_scaler.transform(X).astype(np.float32)

        # build windows on the fly for this file
        seqs = np.stack([U_n[t-LOOKBACK:t, :] for t in range(LOOKBACK, len(U_n))], axis=0)  # (T-L, L, D)
        xb   = torch.tensor(seqs, device=DEVICE)
        Yp_n = model(xb).cpu().numpy()      # (T-L, 2)
        Yp   = y_scaler.inverse_transform(Yp_n)

        X_true = X[LOOKBACK:, :]            # align ground truth

        # metrics
        rmse_m = mean_squared_error(X_true[:,0], Yp[:,0], squared=False)
        rmse_q = mean_squared_error(X_true[:,1], Yp[:,1], squared=False)
        mape_m = mean_absolute_percentage_error(X_true[:,0], Yp[:,0]) * 100.0
        mape_q = mean_absolute_percentage_error(X_true[:,1], Yp[:,1]) * 100.0
        metrics_rows.append([rid, rmse_m, rmse_q, mape_m, mape_q])

        # plots for one file
        if not plot_done and rid == PLOT_FILE_ID:
            t = np.arange(LOOKBACK, LOOKBACK + len(X_true), dtype=float)

            # m
            plt.figure(figsize=(10,3))
            plt.plot(t, X_true[:,0], label="m true")
            plt.plot(t, Yp[:,0], label="m pred (GRU)", alpha=0.85)
            plt.xlabel("time step"); plt.ylabel("mflow_GHX_bypass")
            plt.title(f"File {rid} m: true vs pred (GRU, L={LOOKBACK})")
            plt.legend(); plt.grid(True, alpha=0.3)
            plt.tight_layout(); plt.savefig(f"figs_gru/file_{rid}_m_timeseries.png", dpi=300); plt.close()

            # q
            plt.figure(figsize=(10,3))
            plt.plot(t, X_true[:,1], label="q true")
            plt.plot(t, Yp[:,1], label="q pred (GRU)", alpha=0.85)
            plt.xlabel("time step"); plt.ylabel("qghx_kW")
            plt.title(f"File {rid} q: true vs pred (GRU, L={LOOKBACK})")
            plt.legend(); plt.grid(True, alpha=0.3)
            plt.tight_layout(); plt.savefig(f"figs_gru/file_{rid}_q_timeseries.png", dpi=300); plt.close()

            # parity
            plt.figure(figsize=(5,5))
            plt.scatter(X_true[:,0], Yp[:,0], s=6, alpha=0.45)
            lo, hi = float(min(X_true[:,0].min(), Yp[:,0].min())), float(max(X_true[:,0].max(), Yp[:,0].max()))
            plt.plot([lo,hi],[lo,hi],"--")
            plt.xlabel("m true"); plt.ylabel("m pred (GRU)")
            plt.title(f"File {rid} parity m"); plt.grid(True, alpha=0.3)
            plt.tight_layout(); plt.savefig(f"figs_gru/file_{rid}_m_parity.png", dpi=300); plt.close()

            plt.figure(figsize=(5,5))
            plt.scatter(X_true[:,1], Yp[:,1], s=6, alpha=0.45)
            lo, hi = float(min(X_true[:,1].min(), Yp[:,1].min())), float(max(X_true[:,1].max(), Yp[:,1].max()))
            plt.plot([lo,hi],[lo,hi],"--")
            plt.xlabel("q true"); plt.ylabel("q pred (GRU)")
            plt.title(f"File {rid} parity q"); plt.grid(True, alpha=0.3)
            plt.tight_layout(); plt.savefig(f"figs_gru/file_{rid}_q_parity.png", dpi=300); plt.close()

            plot_done = True

# Print metrics table
if len(metrics_rows):
    print("\n== Per-file metrics (GRU) ==")
    for rid, rm_m, rm_q, mp_m, mp_q in metrics_rows:
        print(f"File {rid:>4d} | RMSE m={rm_m:.3f} q={rm_q:.3f} | MAPE m={mp_m:.2f}% q={mp_q:.2f}%")
else:
    print("No selected test files were evaluated. Check TEST_FILE_IDS_TO_EVAL and test split.")
