# -*- coding: utf-8 -*-
import os, re
import numpy as np
import pandas as pd

# plotting (headless)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# --- Bigger fonts everywhere ---
plt.rcParams.update({
    "font.size": 16,        # base font
    "axes.titlesize": 20,   # subplot titles
    "axes.labelsize": 18,   # x/y labels
    "xtick.labelsize": 15,
    "ytick.labelsize": 15,
    "legend.fontsize": 16,
    "figure.titlesize": 22  # figure suptitle
})

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
    
    
from paths import GHX_DATA_DIR, FNN_SIM_DIR, ensure_dirs

ensure_dirs()

DATA_DIR = GHX_DATA_DIR
OUT_DIR = FNN_SIM_DIR

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PRINT_SHAPES = True

# Choose which test files to evaluate (IDs come from ghx_run{ID}.csv)

TEST_FILE_IDS_TO_EVAL = [176]     # <-- change this list as you like
PLOT_FILE_ID = 176               # which one to plot (must be in TEST_FILE_IDS_TO_EVAL)

#CTRL_NAMES  = ["opening_PV006","mflow_pump_out","T_pump_in","T_heater_out","T_chiller_after"]
CTRL_NAMES  = ["opening_PV006","mflow_pump_out","T_pump_in","T_heater_out"]
STATE_NAMES = ["mflow_GHX_bypass","qghx_kW"]  # m, q

# -----------------------------
# Load all runs
# -----------------------------
def load_all(dirpath):
    runs = []   # [(rid, U:(T,5), X:(T,2))]
    for fn in sorted(os.listdir(dirpath)):
        m = re.match(r"ghx_run(\d+)\.csv$", fn)
        if not m:
            continue
        rid = int(m.group(1))
        df = pd.read_csv(dirpath / fn)
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
# -----------------------------

test_indices = {12, 56, 74, 81, 121, 136, 176, 233, 266, 300}

# sanity check that all fixed ids exist
all_ids = {rid for (rid, _, _) in runs}
missing = sorted(test_indices - all_ids)
if missing:
    raise ValueError(f"IDs not present in runs: {missing}")

# split
train_runs = [(rid, U, X) for (rid, U, X) in runs if rid not in test_indices]
test_runs  = [(rid, U, X) for (rid, U, X) in runs if rid in  test_indices]

# safety
assert not ({rid for (rid,_,_) in train_runs} & {rid for (rid,_,_) in test_runs})
print(f"Train: {len(train_runs)} | Test: {len(test_runs)}")

# NEW: needed for filtering TEST_FILE_IDS_TO_EVAL
test_ids = {rid for (rid, _, _) in test_runs}

# -----------------------------
# Build big matrices for training
# -----------------------------
U_train = np.concatenate([U for (_,U,_) in train_runs], axis=0)
X_train = np.concatenate([X for (_,_,X) in train_runs], axis=0)

# Fit scalers on TRAIN only
u_scaler = StandardScaler().fit(U_train)
y_scaler = StandardScaler().fit(X_train)

# Prepare torch DataLoader for training (supervised, pointwise: u(t) -> x(t))
U_train_n = u_scaler.transform(U_train).astype(np.float32)
X_train_n = y_scaler.transform(X_train).astype(np.float32)

train_ds = TensorDataset(torch.tensor(U_train_n), torch.tensor(X_train_n))
pin = True if DEVICE == "cuda" else False
train_ld = DataLoader(train_ds, batch_size=256, shuffle=True, pin_memory=pin)

# -----------------------------
# Define & Train FNN
# -----------------------------
class FNN(nn.Module):
    def __init__(self, in_dim=4, hidden=128, out_dim=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, out_dim)
        )
    def forward(self, x): return self.net(x)

model = FNN().to(DEVICE)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
crit = nn.MSELoss()

for ep in range(40):
    model.train(); tot = 0.0
    for xb, yb in train_ld:
        xb = xb.to(DEVICE, non_blocking=True)
        yb = yb.to(DEVICE, non_blocking=True)
        opt.zero_grad()
        yp = model(xb)
        loss = crit(yp, yb)
        loss.backward()
        opt.step()
        tot += loss.item() * xb.size(0)
    if PRINT_SHAPES:
        print(f"Epoch {ep+1:02d} | train MSE={tot/len(train_ds):.6f}")

