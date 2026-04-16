# Load python packages
# Data processing
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sdf
import json,pickle
from itertools import combinations
import random
from scipy.stats import t

from scipy.signal import savgol_filter

# Optimization
from scipy import optimize,interpolate
from scipy.interpolate import interp1d
from scipy.integrate import solve_ivp
from scipy import interpolate
from scipy.spatial import distance
import timeit
import math

from scipy.stats import norm

# Machine learning
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import pysindy as ps

# Multiprocessing
import joblib
from scipy.stats import norm

# misc
from tqdm import tqdm
from shutil import copyfile
import os,sys
import seaborn as sns

# Load several functions to process the results
# ---
def get_confidence_interval_indicator_ghx(x_ghx_sim_experiment, valid_indices, plot_index):
    """
    Debug-friendly version: computes 95% CI coverage and prints info for verification.
    """
    print(f"\n--- Computing CI for GHX variable {plot_index} ---")
    print(f"Number of valid indices: {len(valid_indices)}")

    experience = savgol_filter(x_ghx_experiment_conv[plot_index, :], 1200, 3)
    print(f"Smoothed experimental shape: {experience.shape}")

    try:
        if len(valid_indices) == 0:
            print("No valid indices. Returning NaN.")
            return 0.0, np.nan, np.nan, np.nan, np.nan

        valid_sim_data = []
        for i in valid_indices:
            sim = x_ghx_sim_experiment[i]
            if sim.ndim == 2 and sim.shape[1] > plot_index:
                valid_sim_data.append(sim[:, plot_index])
            else:
                print(f"Skipping model {i} shape invalid: {sim.shape}")

        if len(valid_sim_data) == 0:
            print("No valid simulation data collected.")
            return 0.0, np.nan, np.nan, np.nan, np.nan

        full_data = np.array(valid_sim_data).T
        print(f"Full data shape (time, models): {full_data.shape}")

        credible_intervals_t = []
        confidence_interval_indicator = 0.0
        error_indicator = 0.0
        error_indicator_max = []
        confidence_interval_indicator_max = []

        for i, sample in enumerate(full_data):
            sample_mean = np.mean(sample)
            sample_std = np.std(sample, ddof=1)
            n = len(sample)

            if n <= 1 or np.isnan(sample_std):
                margin_of_error = 0.0
            else:
                df = n - 1
                t_critical = t.ppf(0.975, df)
                margin_of_error = t_critical * (sample_std / np.sqrt(n))

            lower = sample_mean - margin_of_error
            upper = sample_mean + margin_of_error

            credible_intervals_t.append([lower, upper])
            confidence_interval_indicator += (upper - lower)

            abs_error = abs(experience[i] - sample_mean)
            error_indicator += abs_error
            error_indicator_max.append(abs_error)
            confidence_interval_indicator_max.append(upper - lower)

            # Print example values for first few time steps
            if i % 1000 == 0:
                print(f"t={i:4d} | mean={sample_mean:.3f}, MoE={margin_of_error:.3f}, CI=({lower:.3f}, {upper:.3f}), exp={experience[i]:.3f}, error={abs_error:.3f}")

        confidence_interval_indicator /= full_data.shape[0]
        error_indicator /= full_data.shape[0]

        within_confidence = [
            lower <= exp_val <= upper
            for exp_val, (lower, upper) in zip(experience, credible_intervals_t)
        ]
        coverage = within_confidence.count(True) / len(within_confidence)

        print(f"\n--- CI Summary for variable {plot_index} ---")
        print(f"Coverage: {coverage:.4f}")
        print(f"Avg Error: {error_indicator:.4f}")
        print(f"Avg CI Width: {confidence_interval_indicator:.4f}")
        print(f"Max Error: {np.max(error_indicator_max):.4f}")
        print(f"CI Width @ Max Error: {confidence_interval_indicator_max[np.argmax(error_indicator_max)]:.4f}")

        return (
            coverage,
            error_indicator,
            confidence_interval_indicator,
            np.max(error_indicator_max),
            confidence_interval_indicator_max[np.argmax(error_indicator_max)]
        )

    except Exception as e:
        print(f"[ERROR] CI computation failed for plot_index={plot_index}: {e}")
        return 0.0, np.nan, np.nan, np.nan, np.nan



    
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.signal import savgol_filter
import pandas as pd

def plot_simulation_and_metrics(x_ghx_sim_experiment, selected_model_len):
    """
    Save per-iteration EXP prediction CSV using mean_sim (mean over valid sims)
    for m and q, then proceed with plotting and metric computation as before.
    Also saves the experimental (smoothed) series in the same CSV.
    """
    import os
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    from scipy.signal import savgol_filter

    labels = [r'$\dot{m}_{ghx,bypass}$ (kg/s)', r'$Q_{ghx}$ (W)']
    fig, ax = plt.subplots(2, 2, figsize=(15, 15))

    valid_ghx = []
    results = {}

    rmse_mean_sim_list = []
    mae_mean_sim_list = []

    # placeholders to capture final mean predictions for both variables
    mean_pred_m = None  # for m (index 0)
    mean_pred_q = None  # for q (index 1)

    # time vector (must match sim length)
    try:
        time_s_full = np.asarray(tm2[:-1], dtype=float)
    except Exception:
        time_s_full = np.arange(x_ghx_sim_experiment[0].shape[0], dtype=float)

    for row, plot_indices in enumerate([0, 1]):
        valid_indices = []
        x_ghx_sim_forplot = []
        rmse_values = []
        mae_values = []

        # screen valid sims by bounds
        for p in range(len(x_ghx_sim_experiment)):
            data = x_ghx_sim_experiment[p][:, plot_indices]
            if plot_indices == 0:
                condition = np.max(data) <= 1 and data[-1] >= 0 and np.min(data) >= 0
            else:  # plot_indices == 1 (Q)
                condition = np.max(data) <= 450 and -50 <= data[-1] <= 200 and np.min(data) >= -50
            if condition:
                valid_indices.append(p)

        # Fallback: if nothing passed the screen, use ALL sims
        if len(valid_indices) == 0:
            valid_indices = list(range(len(x_ghx_sim_experiment)))

        valid_ghx.append(valid_indices)
        results[f"valid_indices_{plot_indices}"] = valid_indices
        print(f"Valid indices for variable {plot_indices}: {valid_indices}")

        # collect curves and per-sim errors
        for p in valid_indices:
            ax[row][0].plot(x_ghx_sim_experiment[p][:, plot_indices], '-*', label=f'Sampled {p}')
            x_ghx_sim_forplot.append(x_ghx_sim_experiment[p])
            rmse = np.sqrt(mean_squared_error(x_ghx_experiment_conv[plot_indices, :],
                                              x_ghx_sim_experiment[p][:, plot_indices]))
            mae = mean_absolute_error(x_ghx_experiment_conv[plot_indices, :],
                                      x_ghx_sim_experiment[p][:, plot_indices])
            rmse_values.append(rmse)
            mae_values.append(mae)

        # Average over all individual RMSE/MAE
        avg_rmse = np.mean(rmse_values) if rmse_values else np.nan
        avg_mae  = np.mean(mae_values)  if mae_values  else np.nan

        # Mean and std of simulations (over the valid ones)
        sim_data_array = np.array(x_ghx_sim_forplot)  # shape: (P_valid, T, 2)
        mean_sim = np.mean(sim_data_array, axis=0)[:, plot_indices]  # (T,)
        std_sim  = np.std(sim_data_array,  axis=0)[:, plot_indices]  # (T,)

        # capture for saving after loop
        if plot_indices == 0:
            mean_pred_m = mean_sim.copy()
        else:
            mean_pred_q = mean_sim.copy()

        # RMSE/MAE between mean_sim and experiment
        rmse_mean_sim = np.sqrt(mean_squared_error(x_ghx_experiment_conv[plot_indices, :], mean_sim))
        mae_mean_sim  = mean_absolute_error(x_ghx_experiment_conv[plot_indices, :], mean_sim)

        rmse_mean_sim_list.append(rmse_mean_sim)
        mae_mean_sim_list.append(mae_mean_sim)

        if plot_indices == 0:
            results["rmse0"] = avg_rmse
            results["mae0"] = avg_mae
            results["rmse_mean_sim0"] = rmse_mean_sim
            results["mae_mean_sim0"]  = mae_mean_sim
        else:
            results["rmse1"] = avg_rmse
            results["mae1"] = avg_mae
            results["rmse_mean_sim1"] = rmse_mean_sim
            results["mae_mean_sim1"]  = mae_mean_sim

        print(f"[{plot_indices}] Avg RMSE: {avg_rmse:.3f}, MAE: {avg_mae:.3f}")
        print(f"[{plot_indices}] RMSE mean_sim: {rmse_mean_sim:.3f}, MAE mean_sim: {mae_mean_sim:.3f}")

        # Original + smoothed experiment overlays
        ax[row][0].plot(x_ghx_experiment[plot_indices, :], '-k', label='original experiment', alpha=0.5)
        ax[row][0].plot(savgol_filter(x_ghx_experiment[plot_indices, :], 1200, 3), '-', color='green', label='smoothed experiment')
        ax[row][0].set_title(f'Sampled Models: Valid {len(valid_indices)}/{len(x_ghx_sim_experiment)}')
        ax[row][0].set_ylabel(labels[plot_indices], fontsize=15)
        ax[row][0].set_xlabel('time (s)', fontsize=15)

        # Mean, CI, max, min
        ax[row][1].plot(mean_sim, '-*', color='red', label='mean')
        ax[row][1].fill_between(time_s_full, mean_sim + 2 * std_sim, mean_sim - 2 * std_sim, alpha=0.5)
        ax[row][1].plot(np.max(sim_data_array, axis=0)[:, plot_indices], '-', label='max', linewidth=1.2)
        ax[row][1].plot(np.min(sim_data_array, axis=0)[:, plot_indices], '-', label='min', linewidth=1.2)
        ax[row][1].plot(x_ghx_experiment[plot_indices, :], '-k', label='original experiment', alpha=0.5)
        ax[row][1].plot(savgol_filter(x_ghx_experiment[plot_indices, :], 1200, 3), '-', color='green', label='smoothed experiment')
        ax[row][1].set_ylabel(labels[plot_indices], fontsize=15)
        ax[row][1].set_xlabel('time (s)', fontsize=15)
        ax[row][1].legend()

    # -------- save ONE CSV with mean_sim predictions + experiment (smoothed) --------
    if (mean_pred_m is not None) and (mean_pred_q is not None):
        L = min(len(time_s_full), len(mean_pred_m), len(mean_pred_q),
                x_ghx_experiment_conv.shape[1])
        out_df = pd.DataFrame({
            "time_s":                 time_s_full[:L],
            "m_pred_kgps":            mean_pred_m[:L],
            "Q_pred_kW":              mean_pred_q[:L],
            "m_exp_smooth_kgps":      x_ghx_experiment_conv[0, :L],
            "Q_exp_smooth_kW":        x_ghx_experiment_conv[1, :L],
        })
        out_dir = "1tj_err_exp_pred_sindy"
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"exp_pred_index{int(selected_model_len):03d}.csv")
        out_df.to_csv(out_path, index=False)
        print(f"[SAVE] mean_sim predictions + exp -> {out_path}")

    plt.tight_layout()
    plt.close()

    return (
        valid_ghx,
        results["rmse0"], results["mae0"],
        results["rmse1"], results["mae1"],
        results["rmse_mean_sim0"], results["mae_mean_sim0"],
        results["rmse_mean_sim1"], results["mae_mean_sim1"]
    )


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx


def finer(xold,yold,xnew):
    f = interpolate.interp1d(xold,yold)
    ynew = f(xnew)
    return ynew


def generate_input_control_tes(df_teds_dc_class):
    """
    Generate input and control time-series for training the surrogate for TES

    :param df_teds_dc_class: (list) List of DataFrame, each DataFrame being one Dymola post-processed run.

    return x_tes_train,u_tes_train, tm2: input and control, respectively, and time

    """
    x_tes_train = []
    u_tes_train = []
    for i in range(len(df_teds_dc_class)):
        tm = df_teds_dc_class[i]['time_sec'].values-df_teds_dc_class[i]['time_sec'].values[0]
        tm2 = np.linspace(0,tm[-1],int(tm[-1])+1)
        u_PV = df_teds_dc_class[i]['opening_PV050'].values
        u_PVB = df_teds_dc_class[i]['opening_PV006'].values
        u_Tin_dc = df_teds_dc_class[i]['T_TES_in'].values
        u_mpump_dc = df_teds_dc_class[i]['mflow_pump_out'].values
        u_Tpump_dc = df_teds_dc_class[i]['T_pump_in'].values
        u_Tchiller_dc = df_teds_dc_class[i]['T_chiller_after'].values
        u_Theatout_dc = df_teds_dc_class[i]['T_heater_out'].values

        x_min_dc = df_teds_dc_class[i]['mflow_TES_in'].values
        x_Tout_dc = df_teds_dc_class[i]['T_TES_out'].values
        x_Ttop_dc = df_teds_dc_class[i]['TES_node50'].values
        x_Tmid_dc = df_teds_dc_class[i]['TES_node100'].values
        x_Tbot_dc = df_teds_dc_class[i]['TES_node150'].values
        x_mout_dc = df_teds_dc_class[i]['mflow_TES_out'].values

        u_PV2 = finer(tm,u_PV,tm2)
        u_PVB2 = finer(tm,u_PVB,tm2)
        u_Tin_dc2 = finer(tm,u_Tin_dc,tm2)
        u_mpump_dc2 = finer(tm,u_mpump_dc,tm2)
        u_Tpump_dc2 = finer(tm,u_Tpump_dc,tm2)
        u_Tchiller_dc2 = finer(tm,u_Tchiller_dc,tm2)
        u_Theatout_dc2 = finer(tm,u_Theatout_dc,tm2)

        x_min_dc2 = finer(tm,x_min_dc,tm2)
        x_Tout_dc2 = finer(tm,x_Tout_dc,tm2)
        x_Ttop_dc2 = finer(tm,x_Ttop_dc,tm2)
        x_Tbot_dc2 = finer(tm,x_Tbot_dc,tm2)
        x_Tmid_dc2 = finer(tm,x_Tmid_dc,tm2)
        x_mout_dc2 = finer(tm,x_mout_dc,tm2)

        x_tes_dc = np.vstack([x_min_dc2, x_Tout_dc2, x_Ttop_dc2, x_Tmid_dc2, x_Tbot_dc2]).transpose()
        u_tes_dc = np.vstack([u_PVB2, u_mpump_dc2, u_Tpump_dc2,u_Theatout_dc2]).transpose()

        x_tes_train.append(x_tes_dc)
        u_tes_train.append(u_tes_dc)
    return x_tes_train,u_tes_train,tm2


def generate_input_control_ghx(df_teds_dc_class):
    """
    Generate input and control time-series for training the surrogate for GHX

    :param df_teds_dc_class: (list) List of DataFrame, each DataFrame being one Dymola post-processed run.

    return x_ghx_train,u_ghx_train, tm2: input and control, respectively, and time

    """
    x_ghx_train = []
    u_ghx_train = []

    for i in range(len(df_teds_dc_class)):

        tm = df_teds_dc_class[i]['time_sec'].values-df_teds_dc_class[i]['time_sec'].values[0]
        tm2 = np.linspace(0,tm[-1],int(tm[-1])+1)

        u_mpump_dc = df_teds_dc_class[i]['mflow_pump_out'].values
        u_Tpump_dc = df_teds_dc_class[i]['T_pump_in'].values
        u_Tchiller_dc = df_teds_dc_class[i]['T_chiller_after'].values

        x_bpflow_dc = df_teds_dc_class[i]['mflow_GHX_bypass'].values
        x_bpT_dc = df_teds_dc_class[i]['T_GHX_bypass'].values
        x_mghx_dc = df_teds_dc_class[i]['mflow_GHX_in'].values
        x_Qout_dc = df_teds_dc_class[i]['qghx'].values

        u_mpump_dc2 = finer(tm,u_mpump_dc,tm2)
        u_Tpump_dc2 = finer(tm,u_Tpump_dc,tm2)
        u_Tchiller_dc2 = finer(tm,u_Tchiller_dc,tm2) # used as input conditions from TES and heater

        u_PVB = df_teds_dc_class[i]['opening_PV006'].values
        u_Theatout_dc = df_teds_dc_class[i]['T_heater_out'].values

        u_PVB2 = finer(tm,u_PVB,tm2)
        u_Theatout_dc2 = finer(tm,u_Theatout_dc,tm2)


        x_bpflow_dc2 = finer(tm,x_bpflow_dc,tm2)
        #x_bpT_dc2 = finer(tm,x_bpT_dc,tm2)
        x_mghx_dc2 = finer(tm,x_mghx_dc,tm2)
        x_Qout_dc2 = finer(tm,x_Qout_dc,tm2)

        x_ghx_dc = np.vstack([x_bpflow_dc2, x_Qout_dc2/1000]).transpose()
        #u_ghx_dc = np.vstack([u_mpump_dc2, u_Tpump_dc2, u_Tchiller_dc2]).transpose()
        u_ghx_dc = np.vstack([u_PVB2, u_mpump_dc2, u_Tpump_dc2, u_Theatout_dc2, u_Tchiller_dc2]).transpose()

        x_ghx_train.append(x_ghx_dc)
        u_ghx_train.append(u_ghx_dc)

    return x_ghx_train,u_ghx_train,tm2


def SINDyC_fit(x_tes_train,u_tes_train,tm2,threshold=1e-3,alpha=1e-3,name="SinDyc",verbose=True):
    """
    Fit a SINDyC model based on input and control time-series and return it

    :param x_tes_train: (list) input time-serie
    :param u_tes_train: (list) control time-serie
    :param tm2: (list) time steps
    :param threshold: (float) Hard thresholding parameter in the SLTSQ algorithm
    :param alpha: (float) Sparsity promoting parameter in the SLLTSQ algorithm
    :param name: (str) Name of the file in which the SINDyC model will be dumped. It should also contain the path in which we want to save the data.
    return model_tes: SINDy model
    """
    poly_library = ps.PolynomialLibrary(degree=1, include_interaction= False) # F/True

    max_iter= 100 ########### default 20
    stlsq_optimizer = ps.STLSQ(threshold=threshold, alpha=alpha, max_iter= max_iter, normalize_columns= False, verbose=verbose)
    model_tes = ps.SINDy(feature_library=poly_library, optimizer=stlsq_optimizer)
    model_tes.fit(x=x_tes_train, u=u_tes_train, t=tm2[1]-tm2[0])# multiple_trajectories=True)

    with open("{}.pkl".format(name), "wb") as outfile: #save the model
        pickle.dump(model_tes, file=outfile)

    return model_tes
    

def process_model(t, x, u,model_tes,model_ghx):
    """
    Compute the derivative of the SINDyC models to solve the ODE

    :param t: (dummy) dummy variable for solve_ivp
    :param x: (list) state initial condition
    :param u: (list) control
    :param model_tes: (SINDyC model) SINDyC model of the TES
    :param model_ghx: (SONDyC model) SINDyc model of the GHX

    :return dxdt: (list) Derivative of the state
    """

    # Controls
    # ---
    mout_pump = u[1]
    Tout_pump = u[2]
    Tout_heater = u[3]
    PV_tes = u[0]

    # Variables
    # ---
    # TES
    m_tes = x[0]
    Tout_tes = x[1]
    Ttop_tes = x[2]
    Tmid_tes = x[3]
    Tbot_tes = x[4]
    # GHX
    mbp_ghx = x[5]
    Q_ghx = x[6]

    m_heater = mout_pump - m_tes
    T_chiller = (m_tes*Tout_tes+m_heater*Tout_heater)/mout_pump

    # Compute derivatives
    # ---

    dmin_tes_dt = (model_tes.coefficients()[0][0]
                   + model_tes.coefficients()[0][1]*m_tes
                   + model_tes.coefficients()[0][2]*Tout_tes
                   + model_tes.coefficients()[0][3]*Ttop_tes
                   + model_tes.coefficients()[0][4]*Tmid_tes
                   + model_tes.coefficients()[0][5]*Tbot_tes
                   # control actions
                   + model_tes.coefficients()[0][6]*PV_tes
                   + model_tes.coefficients()[0][7]*mout_pump
                   + model_tes.coefficients()[0][8]*Tout_pump
                   + model_tes.coefficients()[0][9]*Tout_heater
                   #+ model_tes.coefficients()[0][7]*Tin_tes
                  )

    dTout_tes_dt = (model_tes.coefficients()[1][0]
                    + model_tes.coefficients()[1][1]*m_tes
                    + model_tes.coefficients()[1][2]*Tout_tes
                    + model_tes.coefficients()[1][3]*Ttop_tes
                    + model_tes.coefficients()[1][4]*Tmid_tes
                    + model_tes.coefficients()[1][5]*Tbot_tes
                    # control actions
                    + model_tes.coefficients()[1][6]*PV_tes
                    + model_tes.coefficients()[1][7]*mout_pump
                    + model_tes.coefficients()[1][8]*Tout_pump
                    + model_tes.coefficients()[1][9]*Tout_heater
                    #+ model_tes.coefficients()[1][7]*Tin_tes
                   )

    dTtop_tes_dt = (model_tes.coefficients()[2][0]
                    + model_tes.coefficients()[2][1]*m_tes
                    + model_tes.coefficients()[2][2]*Tout_tes
                    + model_tes.coefficients()[2][3]*Ttop_tes
                    + model_tes.coefficients()[2][4]*Tmid_tes
                    + model_tes.coefficients()[2][5]*Tbot_tes
                    # control actions
                    + model_tes.coefficients()[2][6]*PV_tes
                    + model_tes.coefficients()[2][7]*mout_pump
                    + model_tes.coefficients()[2][8]*Tout_pump
                    + model_tes.coefficients()[2][9]*Tout_heater
                    #+ model_tes.coefficients()[2][7]*Tin_tes
                   )

    dTmid_tes_dt = (model_tes.coefficients()[3][0]
                   + model_tes.coefficients()[3][1]*m_tes
                   + model_tes.coefficients()[3][2]*Tout_tes + model_tes.coefficients()[3][3]*Ttop_tes
                   + model_tes.coefficients()[3][4]*Tmid_tes + model_tes.coefficients()[3][5]*Tbot_tes
                   # control actions
                   + model_tes.coefficients()[3][6]*PV_tes
                   + model_tes.coefficients()[3][7]*mout_pump
                   + model_tes.coefficients()[3][8]*Tout_pump
                   + model_tes.coefficients()[3][9]*Tout_heater
                    #+ model_tes.coefficients()[3][7]*Tin_tes
                   )

    dTbot_tes_dt = (model_tes.coefficients()[4][0]
                   + model_tes.coefficients()[4][1]*m_tes
                   + model_tes.coefficients()[4][2]*Tout_tes + model_tes.coefficients()[4][3]*Ttop_tes
                   + model_tes.coefficients()[4][4]*Tmid_tes + model_tes.coefficients()[4][5]*Tbot_tes
                   # control actions
                   + model_tes.coefficients()[4][6]*PV_tes
                   + model_tes.coefficients()[4][7]*mout_pump
                   + model_tes.coefficients()[4][8]*Tout_pump
                   + model_tes.coefficients()[4][9]*Tout_heater
                    #+ model_tes.coefficients()[4][7]*Tin_tes
                   )

    dmbp_ghx_dt = (model_ghx.coefficients()[0][0]
                   + model_ghx.coefficients()[0][1]*mbp_ghx
                   + model_ghx.coefficients()[0][2]*Q_ghx
                   # control actions
                   + model_ghx.coefficients()[0][3]*PV_tes
                   + model_ghx.coefficients()[0][4]*mout_pump
                   + model_ghx.coefficients()[0][5]*Tout_pump
                   + model_ghx.coefficients()[0][6]*Tout_heater
                   #+ model_ghx.coefficients()[0][7]*T_chiller
                  )

    dQ_ghx_dt = (model_ghx.coefficients()[1][0]
                   + model_ghx.coefficients()[1][1]*mbp_ghx
                   + model_ghx.coefficients()[1][2]*Q_ghx
                   # control actions
                   + model_ghx.coefficients()[1][3]*PV_tes
                   + model_ghx.coefficients()[1][4]*mout_pump
                   + model_ghx.coefficients()[1][5]*Tout_pump
                   + model_ghx.coefficients()[1][6]*Tout_heater
                   #+ model_ghx.coefficients()[1][7]*T_chiller
                 )

    dxdt = np.concatenate(
        (
            dmin_tes_dt, dTout_tes_dt, dTtop_tes_dt, dTmid_tes_dt, dTbot_tes_dt,
            dmbp_ghx_dt, dQ_ghx_dt
        ),axis = None)

    return dxdt

def simulate_sindyc(model_tes,model_ghx,x_tes,x_ghx,u_ghx_train,t_hat):
    """
    Integrate the SINDyC model

    :param model_tes: (SINDyC model) SINDyC model of the TES
    :param model_ghx: (SINDyC model) SINDyC model of the GHX
    :param x_tes: (list) Provide initial conditions of the state for TES
    :param x_ghx: (list) Provide initial conditions of the state for GHX
    :param u_tes_train: (list) Control input
    :param t_hat: (list) Time steps for integration

    :return y_hat: (list) Prediction of the SINDyC model
    """
    # Initialize integrator keywords for solve_ivp to replicate the odeint defaults
    # ---
    integrator_keywords = {}
    integrator_keywords['rtol'] = 1e-12
    integrator_keywords['method'] = 'LSODA'#RK45
    integrator_keywords['atol'] = 1e-12

    # Initialize the starting point
    # ---
    y_hat = []
    x0 = np.concatenate((x_tes[0,:],x_ghx[0,:]),axis = None)

    y_hat.append(x0)
    # Perform the integration step by steps
    # ---
    for i in range(0,len(t_hat)-2):
        u_hat = np.concatenate(
            (u_ghx_train[i,0],u_ghx_train[i,1],
             u_ghx_train[i,2], u_ghx_train[i,3],
            ),
            axis = None)
        sol = solve_ivp(process_model, t_hat[i:i+2], y_hat[i],
                        args=(u_hat,model_tes,model_ghx),**integrator_keywords)
        y_hat.append(sol.y[:,-1])

    return y_hat





def eval_experiment_sindyc_deterministic(model_tes_cv, model_ghx_cv, selected_model_len: int):
    """
    Deterministic eval on experiment with pooled SINDyC models.
    Returns (valid_ghx_placeholder, avg_rmse0, avg_mae0, avg_rmse1, avg_mae1,
             rmse0_mean, mae0_mean, rmse1_mean, mae1_mean)

    Notes:
      - valid_ghx_placeholder keeps the return signature identical to your old code.
      - We use your smoothed experiment arrays x_ghx_experiment_conv / x_tes_experiment_conv.
    """
    # controls for GHX are 4-wide (PV006, m_pump, T_pump_in, T_heater_out)
    u_ghx_exp_4 = u_experiment_conv_2[:4, :].T  # shape (T,4)

    # simulate deterministically
    pred_list = simulate_sindyc(
        model_tes_cv, model_ghx_cv,
        x_tes_experiment_conv.T,           # (T,5)
        x_ghx_experiment_conv.T,           # (T,2)
        u_ghx_exp_4,                       # (T,4)
        tm2
    )
    pred_arr = np.asarray(pred_list)       # (T, 7) in your setup
    pred_ghx = pred_arr[:, 5:]             # (T, 2) -> [m, Q]

    # align to experiment length
    T = min(pred_ghx.shape[0], x_ghx_experiment_conv.shape[1])
    y_true_m = x_ghx_experiment_conv[0, :T]
    y_true_q = x_ghx_experiment_conv[1, :T]
    y_pred_m = pred_ghx[:T, 0]
    y_pred_q = pred_ghx[:T, 1]

    # metrics (identical components you tracked before)
    rmse_m  = mean_squared_error(y_true_m, y_pred_m, squared=False)
    mae_m   = mean_absolute_error(y_true_m, y_pred_m)
    rmse_q  = mean_squared_error(y_true_q, y_pred_q, squared=False)
    mae_q   = mean_absolute_error(y_true_q, y_pred_q)

    # keep your mean_sim naming so downstream code remains unchanged
    rmse0_mean, mae0_mean = rmse_m, mae_m
    rmse1_mean, mae1_mean = rmse_q, mae_q
    avg_rmse0, avg_mae0   = rmse_m, mae_m  # no ensemble; avg == single
    avg_rmse1, avg_mae1   = rmse_q, mae_q

    # save a CSV exactly like before
    try:
        time_s_full = np.asarray(tm2[:-1], dtype=float)
    except Exception:
        time_s_full = np.arange(T, dtype=float)

    L = min(len(time_s_full), T, x_ghx_experiment_conv.shape[1])
    out_df = pd.DataFrame({
        "time_s":            time_s_full[:L],
        "m_pred_kgps":       y_pred_m[:L],
        "Q_pred_kW":         y_pred_q[:L],
        "m_exp_smooth_kgps": x_ghx_experiment_conv[0, :L],
        "Q_exp_smooth_kW":   x_ghx_experiment_conv[1, :L],
    })
    out_dir = "1tj_err_exp_pred_sindy"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"exp_pred_index{int(selected_model_len):03d}.csv")
    out_df.to_csv(out_path, index=False)
    print(f"[SAVE] deterministic exp prediction -> {out_path}")

    # return signature matches your prior caller
    valid_ghx_placeholder = [[], []]  # no screening list in deterministic case
    return (valid_ghx_placeholder,
            avg_rmse0, avg_mae0, avg_rmse1, avg_mae1,
            rmse0_mean, mae0_mean, rmse1_mean, mae1_mean)

    
#save_dir = f'/home/nabiu/sindyC/newbounds500_training_data_random/'
save_dir = "/home/unabila/ghxSindy/newbounds500_training_data_random/"

def load_data(i):

    """
    Load data from Dymola post-process output (i.e., *.pkl files)

    :param i: (int) Reference to the saved model.

    return x_tes_train_temp, x_ghx_train_temp, u_tes_train_temp, u_ghx_train_temp,tm2: TES, GHX states, TES, GHX actuators, and time
    """

    with open('%ssaved_%d.pkl'%(save_dir,i),'rb') as infile:
        df_teds_dc_class_all = pickle.load(infile)

    try:
        x_tes_train_temp,u_tes_train_temp,tm2=generate_input_control_tes(df_teds_dc_class_all)
        x_ghx_train_temp,u_ghx_train_temp,tm2=generate_input_control_ghx(df_teds_dc_class_all)
    except:
        print("Failed",i)
        x_tes_train_temp, x_ghx_train_temp, u_tes_train_temp, u_ghx_train_temp,tm2 = [],[],[],[],[]
    return x_tes_train_temp, x_ghx_train_temp, u_tes_train_temp, u_ghx_train_temp,tm2


# Check that all indices are accounted for when the data were generated
# Indeed, we will access the files by indices
# ---

valid_indices = []

for subdir, dirs, files in os.walk(save_dir):
    for file in files:
        valid_indices.append(int(file.strip('.pkl').strip('saved_')))

print(len(sorted(valid_indices)))


#Load generated data from *.mat files
# ---


#model_dir = '/home/nabiu/ghx_sindy/newbounds_1samplestotrain_model_directory/'
model_dir = "/home/unabila/ghxSindy/newbounds_12tj_samplestotrain_model_directory/"
    
# ---
if not os.path.exists(model_dir):
    os.mkdir(model_dir)

# Load the data in parallel
# ---

ncores=20 # number of processors to run in parallel to load data faster
if ncores > 1:
    with joblib.Parallel(n_jobs=ncores) as parallel:
        fitness=parallel(joblib.delayed(load_data)(count)for count in range(500))

###################################
x_tes_train_all,u_tes_train_all=[],[]
x_ghx_train_all,u_ghx_train_all=[],[]
failed = 0

# Print the shape of fitness
print("Shape of fitness:", (len(fitness), len(fitness[0])))

for elem in fitness:
    #print("elem[0]:",elem[0])
    #print("Shape of elem[0]:", np.shape(elem[0]))
    if elem[0] != []:
        #print("elem[0][0]:", elem[0][0])
        #print("Shape of elem[0][0]:", np.shape(elem[0][0]))
        #if elem[0][0] != []:# and elem[1] != [] and elem[2] != [] and elem[3] != [] and elem[4] != []:

        x_tes_train_all.append(elem[0][0])
        x_ghx_train_all.append(elem[1][0])
        u_tes_train_all.append(elem[2][0])
        u_ghx_train_all.append(elem[3][0])
        tm2 = elem[4]
        try:
            print(np.array(x_tes_train_all).shape)

        except:

            x_tes_train_all.pop(-1)
            x_ghx_train_all.pop(-1)
            u_tes_train_all.pop(-1)
            u_ghx_train_all.pop(-1)
            failed +=1

    #else:
        #failed +=1
print("Number of failed dymolas ",failed)

# Load Experimental data
# ---


dummy_1,dummy_2,dummy_3,dummy_4,tm2=load_data(10)

with open('TES_experiment_data.pkl', 'rb') as f:
    x_tes_experiment = pickle.load(f)

with open('GHX_experiment_data.pkl', 'rb') as f:
    x_ghx_experiment = pickle.load(f)

with open("{}.pkl".format('U_control_experiment_data'), "rb") as f: #save the model
    u_experiment = pickle.load(f)


tm2 = np.linspace(0, 5251, 5251)

window_len = 1200
polyorder = 3

x_ghx_experiment_smooth = savgol_filter(x_ghx_experiment, window_len, polyorder, axis=1)
x_tes_experiment_smooth = savgol_filter(x_tes_experiment, window_len, polyorder, axis=1)

x_tes_experiment_conv = finer(np.linspace(0.0, tm2[-1], x_tes_experiment_smooth.shape[1]), x_tes_experiment_smooth, tm2[:-1])
x_ghx_experiment_conv = finer(np.linspace(0.0, tm2[-1], x_ghx_experiment_smooth.shape[1]), x_ghx_experiment_smooth, tm2[:-1])
u_experiment_conv = finer(np.linspace(0.0,tm2[-1],u_experiment.shape[1]),u_experiment,tm2[:])

u_experiment_conv_2 = np.full_like(u_experiment_conv,1)
u_experiment_conv_2[0] = u_experiment_conv[3]
u_experiment_conv_2[1] = u_experiment_conv[0]
u_experiment_conv_2[2] = u_experiment_conv[1]
u_experiment_conv_2[3] = u_experiment_conv[2]


#TES
transposedTES_array = x_tes_experiment_conv.T  # Shape will be (5250, 5)

# Step 3: Vertical stack
paddedTES_array = np.pad(transposedTES_array, ((0, 1), (0, 0)), mode='edge')  # Padding to make it (5251, 5)

# Step 4: Expand dimensions to get (1, 5251, 2)
x_tes_experiment_new = np.expand_dims(paddedTES_array, axis=0)  # Shape will be (1, 5251, 5)

print(x_tes_experiment_new.shape)

# U EXP
# Step 2: Transpose the array (swap rows and columns)
transposedU_array = u_experiment_conv_2.T  # Shape will be (5250, 2)
u_experiment_new = np.expand_dims(transposedU_array, axis=0)  # Shape will be (1, 5251, 2)

print(u_experiment_new.shape)

#GHX
transposed_array = x_ghx_experiment_conv.T  # Shape will be (5250, 2)
padded_array = np.pad(transposed_array, ((0, 1), (0, 0)), mode='edge')  # Padding to make it (5251, 2)

# Step 4: Expand dimensions to get (1, 5251, 2)
x_ghx_experiment_new = np.expand_dims(padded_array, axis=0)  # Shape will be (1, 5251, 2)

print(x_ghx_experiment_new.shape)

# Generate and save exp models
# ---

# TES
threshold=1e-6
alpha=1e-3

model_tes_exp = SINDyC_fit(x_tes_experiment_new,u_experiment_new,tm2,threshold=threshold,alpha=alpha,verbose=False)#threshold=1e-6,no alpha

model_para_tes_experiment = model_tes_exp.coefficients().T.flatten()
print(model_para_tes_experiment.shape)


print(model_tes_exp.get_feature_names())

model_tes_exp.print()


# GHX
tm2 = np.linspace(0.0, 5251, 5251)

threshold=1e-8
alpha=1e-6

model_ghx_exp = SINDyC_fit(x_ghx_experiment_new,u_experiment_new,tm2,threshold=threshold,alpha=alpha,verbose=False)#threshold=1e-6,no alpha

model_para_experiment = model_ghx_exp.coefficients().T.flatten()
print(model_para_experiment.shape)


print(model_ghx_exp.get_feature_names())

model_ghx_exp.print()


random.seed(23)
numb_indices = 500  #500
final_set = []
# --- 1) Append EXP trajectory to the pools (match sim shapes) ---
x_tes_exp_traj = x_tes_experiment_new[0]          # (5251, 5)
u_tes_exp_traj = u_experiment_new[0]              # (5251, 4)
x_ghx_exp_traj = x_ghx_experiment_new[0]          # (5251, 2)

# Sim GHX controls have 5 columns: [PV006, m_pump, T_pump_in, T_heater_out, T_chiller_after]
# EXP has only 4; pad a dummy 5th col (unused later because we slice to :4 when fitting GHX).
pad_col = np.zeros((u_tes_exp_traj.shape[0], 1), dtype=u_tes_exp_traj.dtype)
u_ghx_exp_traj5 = np.hstack([u_tes_exp_traj, pad_col])   # (5251, 5)

x_tes_train_all.append(x_tes_exp_traj)
u_tes_train_all.append(u_tes_exp_traj)
x_ghx_train_all.append(x_ghx_exp_traj)
u_ghx_train_all.append(u_ghx_exp_traj5)

EXP_INDEX = len(x_tes_train_all) - 1
print(f"[INFO] Experiment appended as pool index {EXP_INDEX}")

# --- 2) Build final_set as before (unchanged) ---
final_set = []
numb_trajectories = 1
if numb_trajectories > 1:
    for i in range(numb_indices):
        tmp = []
        for _ in range(numb_trajectories):
            elem = random.randint(0, len(x_tes_train_all)-1)
            while elem in tmp:
                elem = random.randint(0, len(x_tes_train_all)-1)
            tmp.append(elem)
        final_set.append(tmp)
else:
    for i in range(numb_indices - 126):
        final_set.append([i])

# --- 3) Fit per-entry SINDyC models (use lists; no np.array(...).reshape(...)) ---
model_tes_all, model_ghx_all = [], []
model_para_tes_all, model_para_ghx_all = [], []

for incr, elem in enumerate(final_set):
    # gather trajectories as lists to keep heterogeneity safe
    x_tes_train = [x_tes_train_all[e] for e in elem]           # each (5251, 5)
    u_tes_train = [u_tes_train_all[e] for e in elem]           # each (5251, 4)
    x_ghx_train = [x_ghx_train_all[e] for e in elem]           # each (5251, 2)
    u_ghx_train = [u_ghx_train_all[e] for e in elem]           # each (5251, 5)

    # TES fit
    threshold, alpha = 1e-6, 1e-3
    model_tes = SINDyC_fit(x_tes_train, u_tes_train, tm2,
                           threshold=threshold, alpha=alpha,
                           name=f'{model_dir}/TES_{incr+1}', verbose=False)
    model_tes_all.append(model_tes)
    mp_tes = model_tes.coefficients().transpose().reshape(-1, 10, 5)
    model_para_tes_all.append(mp_tes)

    # GHX fit (use only first 4 controls)
    threshold, alpha = 1e-8, 1e-6
    u_ghx_train_4 = [u[:, :4] for u in u_ghx_train]
    model_ghx = SINDyC_fit(x_ghx_train, u_ghx_train_4, tm2,
                           threshold=threshold, alpha=alpha,
                           name=f'{model_dir}/GHX_{incr+1}', verbose=False)
    model_ghx_all.append(model_ghx)
    mp_ghx = model_ghx.coefficients().transpose().reshape(-1, 7, 2)
    model_para_ghx_all.append(mp_ghx)

# Stack parameter banks
model_para_tes = np.vstack(model_para_tes_all)    # (N, 10, 5)
model_para_ghx = np.vstack(model_para_ghx_all)    # (N, 7, 2)

# --- 4) Also append single EXP models to banks (so AL can select them later) ---
model_tes_all.append(model_tes_exp)
model_ghx_all.append(model_ghx_exp)

model_para_tes = np.vstack([model_para_tes, model_tes_exp.coefficients().T.reshape(1, 10, 5)])
model_para_ghx = np.vstack([model_para_ghx, model_ghx_exp.coefficients().T.reshape(1, 7, 2)])

print(f"Shapes -> TES: {model_para_tes.shape}, GHX: {model_para_ghx.shape}")


# Save all coefficients
# ---

with open('%sTES_UQ_all_coefficients.pkl'%(model_dir), 'wb') as f:
    pickle.dump(model_para_tes,f)

with open('%sGHX_UQ_all_coefficients.pkl'%(model_dir), 'wb') as f:
    pickle.dump(model_para_ghx,f)

with open('%sEXP_UQ_all_coefficients.pkl'%(model_dir), 'wb') as f:
    pickle.dump(model_para_experiment,f)


#################################### %%%%%%%%%%%%%%%%%%%%%%% AL %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% ###############################
# Initial setup
# ========== ERROR-BASED ACTIVE LEARNING (SINDyC) ==========
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np, pandas as pd, joblib, random, time

# ------------------- Config -------------------
INIT_COUNT  = 10         # start with 10 trajectories
INCREMENT   = 10           # add 10 worst per iteration

ncores      = 20            # parallelism for scoring and MC
OUT_METRICS = "rmdm_incExp_errorAL10sindy_exp_metrics.csv"

# ------------------- Helpers -------------------
def _fit_pooled_models(selected_idx):
    """Fit fresh pooled SINDyC models on the current selection (TES & GHX)."""
    x_tes_train = [x_tes_train_all[i] for i in selected_idx]          # (5251,5) each
    u_tes_train = [u_tes_train_all[i] for i in selected_idx]          # (5251,4)
    x_ghx_train = [x_ghx_train_all[i] for i in selected_idx]          # (5251,2)
    u_ghx_train = [u_ghx_train_all[i][:, :4] for i in selected_idx]   # (5251,4)

    # your standard hyperparams
    threshold_tes, alpha_tes = 1e-6, 1e-3
    threshold_ghx, alpha_ghx = 1e-8, 1e-6

    model_tes_cv = SINDyC_fit(x_tes_train, u_tes_train, tm2,
                              threshold=threshold_tes, alpha=alpha_tes,
                              name=f'{model_dir}/TES_pooled', verbose=False)
    model_ghx_cv = SINDyC_fit(x_ghx_train, u_ghx_train, tm2,
                              threshold=threshold_ghx, alpha=alpha_ghx,
                              name=f'{model_dir}/GHX_pooled', verbose=False)
    return model_tes_cv, model_ghx_cv

def _score_one(idx, model_tes_cv, model_ghx_cv):
    """
    Deterministic prediction on a single remaining trajectory with the pooled models,
    return combined error (0.5*(RMSE_m + RMSE_q)) plus components.
    """
    try:
        x_tes = x_tes_train_all[idx]
        x_ghx = x_ghx_train_all[idx]
        u_ghx = u_ghx_train_all[idx][:, :4]     # 4 controls

        pred_list = simulate_sindyc(model_tes_cv, model_ghx_cv,
                                    x_tes, x_ghx, u_ghx, tm2)
        pred_arr = np.asarray(pred_list)[:, 5:]  # (T,2) -> GHX [m, Q]
        T = pred_arr.shape[0]
        true_ = x_ghx[:T, :]                     # align

        rmse_m = mean_squared_error(true_[:, 0], pred_arr[:, 0], squared=False)
        rmse_q = mean_squared_error(true_[:, 1], pred_arr[:, 1], squared=False)
        score  = 0.5 * (rmse_m + rmse_q)
        return (idx, float(score), float(rmse_m), float(rmse_q))
    except Exception as e:
        # invalid -> push to the end
        return (idx, float('inf'), float('inf'), float('inf'))



# ------------------- Loop -------------------
EXP_INDEX = len(x_tes_train_all) - 1

total_models   = len(x_ghx_train_all)     # use ALL loaded sim trajectories
all_indices    = list(range(total_models))
random.seed(23)


#exclude EXP from the initial draw
sim_only     = [i for i in all_indices if i != EXP_INDEX]
random.seed(23)
selected_idx  = random.sample(sim_only, min(INIT_COUNT, len(sim_only)))
remaining_idx = [i for i in all_indices if i not in selected_idx]   # EXP_INDEX is here
print(f"[AL] total={total_models} | init={len(selected_idx)} | remain={len(remaining_idx)} | EXP={EXP_INDEX}")


results_rows = []
iteration = 0
seed=23
rng = random.Random(seed)
t0 = time.perf_counter()
exp_included_iter = None

while True:
    iteration += 1
    print(f"\n=== [AL] Iter {iteration} | selected={len(selected_idx)} | remaining={len(remaining_idx)} ===")

    # 1) Fit pooled models on currently selected
    model_tes_cv, model_ghx_cv = _fit_pooled_models(selected_idx)

    # 2) Deterministic eval on experiment (no MVG)
    print("  - Evaluating experiment with deterministic SINDyC...")
    (valid_ghx, avg_rmse0, avg_mae0, avg_rmse1, avg_mae1,
     rmse0_mean, mae0_mean, rmse1_mean, mae1_mean) = eval_experiment_sindyc_deterministic(
        model_tes_cv, model_ghx_cv, selected_model_len=len(selected_idx)
    )

    # (Optional) CI placeholders to preserve your CSV schema
    CI_m = (np.nan,)*5
    CI_Q = (np.nan,)*5

    # 3) Log metrics
    row = {
        "iter": iteration,
        "selected_count": len(selected_idx),
        "added_indices": str(new_pick) if 'new_pick' in locals() else "[]",
        "avg_rmse_m": avg_rmse0, "avg_mae_m": avg_mae0,
        "avg_rmse_q": avg_rmse1, "avg_mae_q": avg_mae1,
        "mean_rmse_m": rmse0_mean, "mean_mae_m": mae0_mean,
        "mean_rmse_q": rmse1_mean, "mean_mae_q": mae1_mean,
        "CI_m": str(CI_m),
        "CI_q": str(CI_Q),
        "elapsed_s": time.perf_counter() - t0,
        "exp_included_iter": exp_included_iter
    }
    results_rows.append(row)
    pd.DataFrame(results_rows).to_csv(OUT_METRICS, index=False)
    print(f"  - Wrote metrics to {OUT_METRICS}")

    # 4) randomly add INCREMENT from remaining
    if len(remaining_idx) == 0:
        results_rows.append(row)
        pd.DataFrame(results_rows).to_csv(OUT_METRICS, index=False)
        print("\n[RANDOM] Finished: all trajectories consumed.")
        break

    k = min(INCREMENT, len(remaining_idx))
    

    new_pick = rng.sample(remaining_idx, k)
    row["added_indices"] = str(new_pick)
    results_rows.append(row)
    pd.DataFrame(results_rows).to_csv(OUT_METRICS, index=False)
    print(f"  - Randomly added {k}: {new_pick}")

    # update pools
    selected_idx.extend(new_pick)
    remaining_idx = [i for i in remaining_idx if i not in new_pick]

print("\n[AL] Finished: all trajectories consumed.")






