# -*- coding: utf-8 -*-
import os
import re
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
    "font.size": 16,
    "axes.titlesize": 20,
    "axes.labelsize": 18,
    "xtick.labelsize": 15,
    "ytick.labelsize": 15,
    "legend.fontsize": 16,
    "figure.titlesize": 22
})

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

from paths import GHX_DATA_DIR, EXP_GHX_CSV, FNN_WO_SIM_DIR, ensure_dirs

ensure_dirs()

# -----------------------------
# Config
# -----------------------------
DATA_DIR = GHX_DATA_DIR
EXP_CSV = EXP_GHX_CSV
OUT_DIR = FNN_WO_SIM_DIR

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PRINT_SHAPES = True

# Toggle: include experiment rows in training
INCLUDE_EXPERIMENT_IN_TRAIN = False

# Choose which test files to evaluate
TEST_FILE_IDS_TO_EVAL = [176]
PLOT_FILE_ID = 176

CTRL_NAMES = ["opening_PV006", "mflow_pump_out", "T_pump_in", "T_heater_out"]
STATE_NAMES = ["mflow_GHX_bypass", "qghx_kW"]

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
# Train/test split
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
# Build training matrices
# -----------------------------
U_train = np.concatenate([U for (_, U, _) in train_runs], axis=0)
X_train = np.concatenate([X for (_, _, X) in train_runs], axis=0)

# -----------------------------
# Load experiment
# -----------------------------
dfe = pd.read_csv(EXP_CSV)

missing = [c for c in CTRL_NAMES + STATE_NAMES if c not in dfe.columns]
if len(missing):
    raise ValueError(f"Missing columns in {EXP_CSV}: {missing}")

U_exp_4 = dfe[CTRL_NAMES].to_numpy(dtype=np.float32)
X_true  = dfe[STATE_NAMES].to_numpy(dtype=np.float32)
t_vec   = dfe["time_sec"].to_numpy(dtype=float) if "time_sec" in dfe.columns else np.arange(len(X_true), dtype=float)

mask = np.isfinite(U_exp_4).all(axis=1) & np.isfinite(X_true).all(axis=1)
U_exp_4 = U_exp_4[mask]
X_true  = X_true[mask]
t_vec   = t_vec[mask]

T_common = min(len(U_exp_4), len(X_true))
U_exp_4, X_true, t_vec = U_exp_4[:T_common], X_true[:T_common], t_vec[:T_common]

if PRINT_SHAPES:
    print("\nExperiment shapes (raw):")
    print("  U_exp_4:", U_exp_4.shape)
    print("  X_true :", X_true.shape)

# -----------------------------
# Optionally include experiment in training
# -----------------------------
if INCLUDE_EXPERIMENT_IN_TRAIN:
    U_train_all = np.concatenate([U_train, U_exp_4], axis=0)
    X_train_all = np.concatenate([X_train, X_true], axis=0)
    if PRINT_SHAPES:
        print("\nIncluding EXP rows in training:")
        print("  U_train_all:", U_train_all.shape, "| X_train_all:", X_train_all.shape)
else:
    U_train_all, X_train_all = U_train, X_train

# -----------------------------
# Fit scalers on training only
# -----------------------------
u_scaler = StandardScaler().fit(U_train_all)
y_scaler = StandardScaler().fit(X_train_all)

U_train_n = u_scaler.transform(U_train_all).astype(np.float32)
X_train_n = y_scaler.transform(X_train_all).astype(np.float32)

train_ds = TensorDataset(torch.tensor(U_train_n), torch.tensor(X_train_n))
pin = DEVICE == "cuda"
train_ld = DataLoader(train_ds, batch_size=256, shuffle=True, pin_memory=pin)

# -----------------------------
# Define & train FNN
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

model = FNN().to(DEVICE)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
crit = nn.MSELoss()

for ep in range(40):
    model.train()
    tot = 0.0
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

# -----------------------------
# Evaluate only on selected test files
# -----------------------------
TEST_FILE_IDS_TO_EVAL = [rid for rid in TEST_FILE_IDS_TO_EVAL if rid in test_ids]
assert len(TEST_FILE_IDS_TO_EVAL) > 0, "Requested test IDs are not in the test split."

metrics_rows = []
plot_done = False

for rid, U, X in test_runs:
    if rid not in TEST_FILE_IDS_TO_EVAL:
        continue

    U_n = u_scaler.transform(U).astype(np.float32)
    with torch.no_grad():
        xb = torch.tensor(U_n, device=DEVICE)
        Yp_n = model(xb).cpu().numpy()
    Yp = y_scaler.inverse_transform(Yp_n)

    df_pred = pd.DataFrame({
        "rid": rid,
        "m_true": X[:, 0],
        "m_pred": Yp[:, 0],
        "q_true": X[:, 1],
        "q_pred": Yp[:, 1],
    })
    out_file = OUT_DIR / f"pred_{rid}.csv"
    df_pred.to_csv(out_file, index=False)
    print(f"[SAVED] {out_file}")

    rmse_m = mean_squared_error(X[:, 0], Yp[:, 0], squared=False)
    rmse_q = mean_squared_error(X[:, 1], Yp[:, 1], squared=False)
    mae_m = mean_absolute_error(X[:, 0], Yp[:, 0])
    mae_q = mean_absolute_error(X[:, 1], Yp[:, 1])

    metrics_rows.append([rid, rmse_m, rmse_q, mae_m, mae_q])

    if not plot_done and rid == PLOT_FILE_ID:
        t = np.arange(len(X), dtype=float)

        # m
        plt.figure(figsize=(10, 3))
        plt.plot(t, X[:, 0], label="m true")
        plt.plot(t, Yp[:, 0], label="m pred", alpha=0.8)
        plt.xlabel("time step")
        plt.ylabel("mflow_GHX_bypass")
        plt.title(f"File {rid} m: true vs pred")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(OUT_DIR / f"4_file_{rid}_m_timeseries.png", dpi=300)
        plt.close()

        # q
        plt.figure(figsize=(10, 3))
        plt.plot(t, X[:, 1], label="q true")
        plt.plot(t, Yp[:, 1], label="q pred", alpha=0.8)
        plt.xlabel("time step")
        plt.ylabel("qghx_kW")
        plt.title(f"File {rid} q: true vs pred")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(OUT_DIR / f"4_file_{rid}_q_timeseries.png", dpi=300)
        plt.close()

        plot_done = True

# -----------------------------
# Print metrics
# -----------------------------
if len(metrics_rows):
    print("\n== Per-file metrics ==")
    for rid, rm_m, rm_q, mp_m, mp_q in metrics_rows:
        print(f"File {rid:>4d} | RMSE m={rm_m:.3f} q={rm_q:.3f} | MAE m={mp_m:.2f} q={mp_q:.2f}")
else:
    print("No selected test files were evaluated. Check TEST_FILE_IDS_TO_EVAL and test split.")