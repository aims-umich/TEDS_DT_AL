# -*- coding: utf-8 -*-
"""
GRU (NARX-style): uses past STATES + CONTROLS to predict next STATE.
- Train on ALL simulation CSVs (no holdout).
- Predict on experiment CSV (teacher-forced: windows use measured exp X + U).
- Controls used (order matters):
    ["opening_PV006","mflow_pump_out","T_pump_in","T_heater_out"]
- States:
    ["mflow_GHX_bypass","qghx_kW"]
"""

import os, re
import numpy as np
import pandas as pd

# headless plotting
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({
    "font.size": 16, "axes.titlesize": 20, "axes.labelsize": 18,
    "xtick.labelsize": 15, "ytick.labelsize": 15, "legend.fontsize": 16,
    "figure.titlesize": 22,
})

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# -----------------------------
# Config
# -----------------------------
DATA_DIR = "ghx_data_csv"   # folder with ghx_run{ID}.csv
EXP_CSV  = "/home/unabila/ghxSindy/experiment_csv/experiment_ghx_formatted.csv"

DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"
PRINT_INFO = True

CTRL_NAMES  = ["opening_PV006","mflow_pump_out","T_pump_in","T_heater_out"]   # 4 controls
STATE_NAMES = ["mflow_GHX_bypass","qghx_kW"]                                   # 2 states

LOOKBACK = 300          # history length (try 300–600 for slow thermals)
HIDDEN   = 256
BATCH    = 512          # tune for your GPU
EPOCHS   = 20 #8
LR       = 1e-3
NUM_WORKERS = 4
CLIP_Z   = 3.5          # clip z-scores on inputs to handle domain shift

torch.backends.cudnn.benchmark = True
if hasattr(torch, "set_float32_matmul_precision"):
    torch.set_float32_matmul_precision("high")

if PRINT_INFO:
    print(f"Using device: {DEVICE}")
    if DEVICE == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

# -----------------------------
# Utils
# -----------------------------
def load_all_sim_runs(dirpath):
    runs = []  # list[(U:(T,4), X:(T,2))]
    for fn in sorted(os.listdir(dirpath)):
        m = re.match(r"ghx_run(\d+)\.csv$", fn)
        if not m:
            continue
        df = pd.read_csv(os.path.join(dirpath, fn))
        if not set(CTRL_NAMES).issubset(df.columns) or not set(STATE_NAMES).issubset(df.columns):
            continue
        U = df[CTRL_NAMES].to_numpy(dtype=np.float32)
        X = df[STATE_NAMES].to_numpy(dtype=np.float32)
        T = min(len(U), len(X))
        if T > 0:
            runs.append((U[:T], X[:T]))
    return runs

def pick(colnames, *cands):
    """Robustly pick a column by prefix candidates (case-insensitive)."""
    cl = [c.lower() for c in colnames]
    for cand in cands:
        c = cand.lower()
        for i, name in enumerate(cl):
            if name.startswith(c):
                return colnames[i]
    raise KeyError(f"None of {cands} found among {colnames}")

# -----------------------------
# Load ALL simulation runs
# -----------------------------
sim_runs = load_all_sim_runs(DATA_DIR)
assert len(sim_runs) > 0, "No simulation runs found."
if PRINT_INFO:
    total_T = sum(min(len(U),len(X)) for (U,X) in sim_runs)
    print(f"Loaded {len(sim_runs)} sim runs, total samples ~ {total_T}")

# -----------------------------
# Fit scalers on ALL sim data
# -----------------------------
U_all = np.concatenate([U for (U,_) in sim_runs], axis=0)
X_all = np.concatenate([X for (_,X) in sim_runs], axis=0)
u_scaler = StandardScaler().fit(U_all)
y_scaler = StandardScaler().fit(X_all)

# -----------------------------
# NARX dataset (uses X_hist + U_hist)
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
            # clip inputs (not targets)
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

train_ds = WindowedSimNARX(sim_runs, u_scaler, y_scaler, LOOKBACK, CLIP_Z)
pin = (DEVICE == "cuda")
train_ld = DataLoader(
    train_ds, batch_size=BATCH, shuffle=True,
    pin_memory=pin, num_workers=NUM_WORKERS,
    persistent_workers=(NUM_WORKERS > 0)
)

if PRINT_INFO:
    print(f"Train windows: {len(train_ds)} | seq {train_ds.seq.shape} | tgt {train_ds.tgt.shape}")

# -----------------------------
# GRU model
# -----------------------------
class GRURegressor(nn.Module):
    def __init__(self, in_dim, hidden=256, out_dim=2, num_layers=1, dropout=0.0):
        super().__init__()
        self.rnn  = nn.GRU(in_dim, hidden, num_layers=num_layers,
                           dropout=(dropout if num_layers>1 else 0.0),
                           batch_first=True)
        self.head = nn.Sequential(nn.ReLU(), nn.Linear(hidden, out_dim))
    def forward(self, x):            # x: (B,L,in_dim)
        out, _ = self.rnn(x)         # (B,L,H)
        hL = out[:, -1, :]           # last step
        return self.head(hL)         # (B,2)

model = GRURegressor(in_dim=6, hidden=HIDDEN, out_dim=2).to(DEVICE)
opt    = torch.optim.Adam(model.parameters(), lr=LR)
crit   = nn.MSELoss()

# mixed precision
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

if PRINT_INFO:
    print("Model on:", next(model.parameters()).device)

# -----------------------------
# Train
# -----------------------------
for ep in range(1, EPOCHS+1):
    model.train(); tot = 0.0
    for xb, yb in train_ld:
        xb = xb.to(DEVICE, non_blocking=True)  # (B,L,6)
        yb = yb.to(DEVICE, non_blocking=True)  # (B,2)
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
    if PRINT_INFO:
        print(f"Epoch {ep:02d} | train MSE={tot/len(train_ds):.6f}")

# -----------------------------
# Predict on EXPERIMENT CSV (teacher-forced)
# -----------------------------
dfe = pd.read_csv(EXP_CSV)
cols = list(dfe.columns)

# robustly map experiment columns -> training order (guard against header drift)
col_open  = pick(cols, "opening_pv006", "opening_p")
col_mflow = pick(cols, "mflow_pump_out", "mflow_pun")
col_tpin  = pick(cols, "t_pump_in")
col_theat = pick(cols, "t_heater_out", "t_heater_c")
CTRL_EXP  = [col_open, col_mflow, col_tpin, col_theat]

# states must exist
for c in STATE_NAMES:
    if c not in dfe.columns:
        raise ValueError(f"Missing state column '{c}' in {EXP_CSV}")

U_exp = dfe[CTRL_EXP].to_numpy(dtype=np.float32)                   # (T,4)
X_exp = dfe[STATE_NAMES].to_numpy(dtype=np.float32)                # (T,2)
t_vec = dfe["time_sec"].to_numpy(dtype=float) if "time_sec" in dfe.columns else np.arange(len(U_exp), dtype=float)

mask = np.isfinite(U_exp).all(axis=1) & np.isfinite(X_exp).all(axis=1)
U_exp, X_exp, t_vec = U_exp[mask], X_exp[mask], t_vec[mask]

T = min(len(U_exp), len(X_exp))
U_exp, X_exp, t_vec = U_exp[:T], X_exp[:T], t_vec[:T]
if T <= LOOKBACK:
    raise ValueError(f"Experiment too short for LOOKBACK={LOOKBACK}: T={T}")

# scale + clip inputs (states+controls) for building windows
U_exp_n = u_scaler.transform(U_exp).astype(np.float32)
X_exp_n = y_scaler.transform(X_exp).astype(np.float32)
np.clip(U_exp_n, -CLIP_Z, CLIP_Z, out=U_exp_n)
np.clip(X_exp_n, -CLIP_Z, CLIP_Z, out=X_exp_n)

# build teacher-forced windows: concat([X_hist, U_hist]) -> (L,6)
L = LOOKBACK
seqs = np.stack([np.concatenate([X_exp_n[t-L:t,:], U_exp_n[t-L:t,:]], axis=1)
                 for t in range(L, T)], axis=0)              # (T-L, L, 6)

model.eval()
with torch.no_grad():
    xb   = torch.tensor(seqs, device=DEVICE)
    Yp_n = model(xb).cpu().numpy()                           # (T-L,2)
Yp = y_scaler.inverse_transform(Yp_n)

# align truth for metrics/plots
X_true = X_exp[L:, :]
t_plot = t_vec[L:]

# -----------------------------
# Metrics & Plots
# -----------------------------
rmse_m = mean_squared_error(X_true[:,0], Yp[:,0], squared=False)
rmse_q = mean_squared_error(X_true[:,1], Yp[:,1], squared=False)
mape_m = mean_absolute_percentage_error(X_true[:,0], Yp[:,0]) * 100.0
mape_q = mean_absolute_percentage_error(X_true[:,1], Yp[:,1]) * 100.0

print("\n=== Experiment metrics (teacher-forced) ===")
print(f"RMSE mflow_GHX_bypass = {rmse_m:.3f}")
print(f"RMSE qghx_kW          = {rmse_q:.3f}")
print(f"MAPE m (%)            = {mape_m:.2f}")
print(f"MAPE q (%)            = {mape_q:.2f}")

os.makedirs("figs_gru_exp", exist_ok=True)

plt.figure(figsize=(12,4))
plt.plot(t_plot, X_true[:,0], label="m true (exp)")
plt.plot(t_plot, Yp[:,0], label="m pred (GRU)", alpha=0.85)
plt.xlabel("time (s)"); plt.ylabel("mflow_GHX_bypass")
plt.title("Experiment: m true vs pred (GRU, X+U history)")
plt.legend(); plt.grid(True, alpha=0.3)
plt.tight_layout(pad=0.6)
plt.savefig("figs_gru_exp/m_timeseries.png", dpi=300, bbox_inches="tight")
plt.close()

plt.figure(figsize=(12,4))
plt.plot(t_plot, X_true[:,1], label="q true (exp)")
plt.plot(t_plot, Yp[:,1], label="q pred (GRU)", alpha=0.85)
plt.xlabel("time (s)"); plt.ylabel("qghx_kW")
plt.title("Experiment: q true vs pred (GRU, X+U history)")
plt.legend(); plt.grid(True, alpha=0.3)
plt.tight_layout(pad=0.6)
plt.savefig("figs_gru_exp/q_timeseries.png", dpi=300, bbox_inches="tight")
plt.close()

# parity
plt.figure(figsize=(6,6))
plt.scatter(X_true[:,0], Yp[:,0], s=18, alpha=0.5)
lo, hi = float(min(X_true[:,0].min(), Yp[:,0].min())), float(max(X_true[:,0].max(), Yp[:,0].max()))
plt.plot([lo,hi],[lo,hi],"--")
plt.xlabel("m true (exp)"); plt.ylabel("m pred (GRU)")
plt.title("Experiment parity: m")
plt.grid(True, alpha=0.3); plt.tight_layout(pad=0.6)
plt.savefig("figs_gru_exp/m_parity.png", dpi=300, bbox_inches="tight")
plt.close()

plt.figure(figsize=(6,6))
plt.scatter(X_true[:,1], Yp[:,1], s=18, alpha=0.5)
lo, hi = float(min(X_true[:,1].min(), Yp[:,1].min())), float(max(X_true[:,1].max(), Yp[:,1].max()))
plt.plot([lo,hi],[lo,hi],"--")
plt.xlabel("q true (exp)"); plt.ylabel("q pred (GRU)")
plt.title("Experiment parity: q")
plt.grid(True, alpha=0.3); plt.tight_layout(pad=0.6)
plt.savefig("figs_gru_exp/q_parity.png", dpi=300, bbox_inches="tight")
plt.close()

np.savez("ghx_gru_experiment_pred.npz",
         y_pred=Yp, y_true=X_true, U_exp=U_exp[L:], t=t_plot)
print('Saved predictions to "ghx_gru_experiment_pred.npz" and plots to "figs_gru_exp/".')
