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

def plot_simulation_and_metrics(x_ghx_sim_experiment, selected_model_len):
    labels = [r'$\dot{m}_{ghx,bypass}$ (kg/s)', r'$Q_{ghx}$ (W)']
    fig, ax = plt.subplots(2, 2, figsize=(15, 15))

    valid_ghx = []
    results = {}

    rmse_mean_sim_list = []
    mae_mean_sim_list = []

    for row, plot_indices in enumerate([0, 1]):
        valid_indices = []
        x_ghx_sim_forplot = []
        rmse_values = []
        mae_values = []

        for p in range(rand_numb):
            data = x_ghx_sim_experiment[p][:, plot_indices]
            if plot_indices == 0:
                condition = np.max(data) <= 1 and data[-1] >= 0 and np.min(data) >= 0
            else:  # plot_indices == 1
                condition = np.max(data) <= 450 and -50 <= data[-1] <= 200 and np.min(data) >= -50

            if condition:
                valid_indices.append(p)

        valid_ghx.append(valid_indices)
        results[f"valid_indices_{plot_indices}"] = valid_indices

        print(f"Valid indices for variable {plot_indices}: {valid_indices}")

        for p in valid_indices:
            ax[row][0].plot(x_ghx_sim_experiment[p][:, plot_indices], '-*', label=f'Sampled {p}')
            x_ghx_sim_forplot.append(x_ghx_sim_experiment[p])

            rmse = np.sqrt(mean_squared_error(x_ghx_experiment_conv[plot_indices, :], x_ghx_sim_experiment[p][:, plot_indices]))
            mae = mean_absolute_error(x_ghx_experiment_conv[plot_indices, :], x_ghx_sim_experiment[p][:, plot_indices])

            rmse_values.append(rmse)
            mae_values.append(mae)

        # Average over all individual RMSE/MAE
        avg_rmse = np.mean(rmse_values)
        avg_mae = np.mean(mae_values)

        # Mean and std of simulations
        sim_data_array = np.array(x_ghx_sim_forplot)
        mean_sim = np.mean(sim_data_array, axis=0)[:, plot_indices]
        std_sim = np.std(sim_data_array, axis=0)[:, plot_indices]

        # RMSE/MAE between mean_sim and experiment
        rmse_mean_sim = np.sqrt(mean_squared_error(x_ghx_experiment_conv[plot_indices, :], mean_sim))
        mae_mean_sim = mean_absolute_error(x_ghx_experiment_conv[plot_indices, :], mean_sim)

        rmse_mean_sim_list.append(rmse_mean_sim)
        mae_mean_sim_list.append(mae_mean_sim)

        if plot_indices == 0:
            results["rmse0"] = avg_rmse
            results["mae0"] = avg_mae
            results["rmse_mean_sim0"] = rmse_mean_sim
            results["mae_mean_sim0"] = mae_mean_sim
        else:
            results["rmse1"] = avg_rmse
            results["mae1"] = avg_mae
            results["rmse_mean_sim1"] = rmse_mean_sim
            results["mae_mean_sim1"] = mae_mean_sim

        print(f"[{plot_indices}] Avg RMSE: {avg_rmse:.3f}, MAE: {avg_mae:.3f}")
        print(f"[{plot_indices}] RMSE mean_sim: {rmse_mean_sim:.3f}, MAE mean_sim: {mae_mean_sim:.3f}")

        # Plotting original and smoothed experiment data
        ax[row][0].plot(x_ghx_experiment[plot_indices, :], '-k', label='original experiment', alpha=0.5)
        ax[row][0].plot(savgol_filter(x_ghx_experiment[plot_indices, :], 1200, 3), '-', color='green', label='smoothed experiment')
        ax[row][0].set_title(f'Sampled Models: Valid {len(valid_indices)}/{rand_numb}')
        ax[row][0].set_ylabel(labels[plot_indices], fontsize=15)
        ax[row][0].set_xlabel('time (s)', fontsize=15)

        # Plot mean, CI, max, min
        ax[row][1].plot(mean_sim, '-*', color='red', label='mean')
        ax[row][1].fill_between(tm2[:-1], mean_sim + 2 * std_sim, mean_sim - 2 * std_sim, alpha=0.5)
        ax[row][1].plot(np.max(sim_data_array, axis=0)[:, plot_indices], '-', label='max', linewidth=1.2)
        ax[row][1].plot(np.min(sim_data_array, axis=0)[:, plot_indices], '-', label='min', linewidth=1.2)
        ax[row][1].plot(x_ghx_experiment[plot_indices, :], '-k', label='original experiment', alpha=0.5)
        ax[row][1].plot(savgol_filter(x_ghx_experiment[plot_indices, :], 1200, 3), '-', color='green', label='smoothed experiment')
        ax[row][1].set_ylabel(labels[plot_indices], fontsize=15)
        ax[row][1].set_xlabel('time (s)', fontsize=15)
        ax[row][1].legend()

    plt.tight_layout()
    plt.savefig(f"try4Tj_10init10_summary_selected_{selected_model_len}.png", dpi=300)
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



def simulate(model_tes,model_ghx, mean, covariance, mean_ghx, covariance_ghx):
    """
    Integrate the SINDyC model from experimental's initial conditions

    :param model_tes: (SINDyC model) SINDyC model of the TES
    :param model_ghx: (SINDyC model) SINDyC model of the GHX

    :return results: (list) Prediction for 7 quantities of interest
    """
    # Sample coefficients from the fitted distribution
    # ---

    # TES
    model_tes.model[-1].coef_ = np.random.multivariate_normal(mean=mean,cov=covariance).reshape(model_tes.model[-1].coef_.shape[0],model_tes.model[-1].coef_.shape[1])
    # GHX
    model_ghx.model[-1].coef_ = np.random.multivariate_normal(mean=mean_ghx,cov=covariance_ghx).reshape(model_ghx.model[-1].coef_.shape[0],model_ghx.model[-1].coef_.shape[1])

    results = simulate_sindyc(model_tes,model_ghx,x_tes_experiment_conv.T,x_ghx_experiment_conv.T,u_experiment_conv_2.T,tm2)
    return results


    
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
model_dir = "/home/unabila/ghxSindy/newbounds_4tj_samplestotrain_model_directory/"
    
# ---
if not os.path.exists(model_dir):
    os.mkdir(model_dir)

# Load the data in parallel
# ---

ncores=10 # number of processors to run in parallel to load data faster
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


numb_trajectories = 4 # number of trajectories to fit each model 12
if numb_trajectories > 1:
    for i in range(numb_indices):
        final_set_temp = []
        for _ in range(numb_trajectories): # Choose numb_trajectories elements out of which we we will perform the fitting and evaluate the gaussian
            elem = random.randint(0, np.array(x_tes_train_all).shape[0]-1)
            if elem not in final_set:
                final_set_temp.append(elem)
            else:
                while elem in final_set:
                    elem = random.randint(0,np.array(x_tes_train_all).shape[0]-1)
                final_set_temp.append(elem)
        final_set.append(final_set_temp)
else:
    for i in range(numb_indices - 126):  #####  manually  add failed
        final_set.append([i])


model_tes_all, model_ghx_all = [], []
model_para_tes_all, model_para_ghx_all = [], []

for incr, elem in enumerate(final_set):

    x_tes_train = np.array(x_tes_train_all)[elem].reshape(-1, 5251, 5)
    u_tes_train = np.array(u_tes_train_all)[elem].reshape(-1, 5251, 4)
    x_ghx_train = np.array(x_ghx_train_all)[elem].reshape(-1, 5251, 2)
    u_ghx_train = np.array(u_ghx_train_all)[elem].reshape(-1, 5251, 5)

    # TES
    threshold = 1e-6
    alpha = 1e-3
    model_tes = SINDyC_fit(list(x_tes_train), list(u_tes_train), tm2,
                           threshold=threshold, alpha=alpha,
                           name=f'{model_dir}/TES_{incr+1}', verbose=False)
    model_tes_all.append(model_tes)
    model_para = model_tes.coefficients().transpose()
    model_para_tes_all.append(model_para.reshape(-1, 10, 5))

    # GHX
    threshold = 1e-8
    alpha = 1e-6
    model_ghx = SINDyC_fit(list(x_ghx_train), list(u_ghx_train[:, :, :-1]), tm2,
                           threshold=threshold, alpha=alpha,
                           name=f'{model_dir}/GHX_{incr+1}', verbose=False)
    model_ghx_all.append(model_ghx)
    model_para = model_ghx.coefficients().transpose()
    model_para_ghx_all.append(model_para.reshape(-1, 7, 2))

# Stack final arrays
model_para_tes = np.vstack(model_para_tes_all)  # shape: (N, 10, 5)
model_para_ghx = np.vstack(model_para_ghx_all)  # shape: (N, 7, 2)


print(f"Shape of model_para_tes: {model_para_tes.shape}")

print(f"Shape of model_para_ghx: {model_para_ghx.shape}")



#################################### %%%%%%%%%%%%%%%%%%%%%%% AL %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% ###############################
# Initial setup

total_models = model_para_ghx.shape[0]
all_indices = list(range(total_models))
initial_count = 10
increment = 100
rand_numb = 100
ncores = 10

# Initial random selection


# Step 1: Flatten GHX model parameters
model_para_ghx_flat = model_para_ghx.reshape(total_models, -1)

# Step 2: Compute Mahalanobis distances for all models
inv_cov_matrix_m = np.linalg.inv(np.cov(model_para_ghx_flat.T))
mahalanobis_all = [
    distance.mahalanobis(model_para_experiment, model, inv_cov_matrix_m)
    for model in model_para_ghx_flat
]

# Step 3: Get indices of worst 100 models (largest distances)
worst_100_indices = np.argsort(mahalanobis_all)[-100:]

# Step 4: Select initial_count randomly from worst 100
selected_models = random.sample(list(worst_100_indices), initial_count)

# Step 5: Define remaining indices (all others except selected)
remaining_indices = list(set(all_indices) - set(selected_models))


##################################
remaining_indices = list(set(all_indices) - set(selected_models))

# Flatten GHX model parameters
model_para_ghx_flat = model_para_ghx.reshape(total_models, -1)

# Compute Mahalanobis distances once for remaining models
remaining_flat = model_para_ghx_flat[remaining_indices]

inv_cov_matrix_m = np.linalg.inv(np.cov(model_para_ghx_flat.T))  # You had typo with backslash earlier

mahalanobis_distances = [
    distance.mahalanobis(model_para_experiment, model, inv_cov_matrix_m)
    for model in remaining_flat
]
sorted_remain_idx = [remaining_indices[i] for i in np.argsort(mahalanobis_distances)]

# ACTIVE LEARNING LOOP
iteration = 0
results = []

while len(selected_models) <= total_models:
    # TES stats
    parameters_tes = pd.DataFrame()
    for j in range(model_para_tes.shape[2]):
        for i in range(model_para_tes.shape[1]):
            parameters_tes[f'A_{j+1}{i}'] = model_para_tes[selected_models, i, j]
    mean = parameters_tes.mean()
    covariance = np.cov(parameters_tes.T)

    # GHX stats
    parameters_ghx = pd.DataFrame()
    for j in range(model_para_ghx.shape[2]):
        for i in range(model_para_ghx.shape[1]):
            parameters_ghx[f'A_{j+1}{i}'] = model_para_ghx[selected_models, i, j]
    mean_ghx = parameters_ghx.mean()
    covariance_ghx = np.cov(parameters_ghx.T)

    # Simulate
    if ncores > 1:
        with joblib.Parallel(n_jobs=ncores) as parallel:
            fitness = parallel(joblib.delayed(simulate)(
                model_tes_all[random.choice(selected_models)],
                model_ghx_all[random.choice(selected_models)],
                mean, covariance, mean_ghx, covariance_ghx
            ) for _ in range(rand_numb))

    x_test_sim_experiment, x_ghx_sim_experiment = [], []
    for results_i in fitness:
        x_test_sim_experiment.append(np.array(results_i)[:, :5])
        x_ghx_sim_experiment.append(np.array(results_i)[:, 5:])


    
    # Unpack all 9 return values
    valid_ghx, avg_rmse0, avg_mae0, avg_rmse1, avg_mae1, rmse0_mean, mae0_mean, rmse1_mean, mae1_mean = plot_simulation_and_metrics(
        x_ghx_sim_experiment, selected_model_len=len(selected_models)
    )
    
    # Evaluate confidence intervals
    valid_indices_m, valid_indices_q = valid_ghx  # from plot_simulation_and_metrics()
    
    CI_m = get_confidence_interval_indicator_ghx(x_ghx_sim_experiment, valid_indices_m, 0)
    CI_Q = get_confidence_interval_indicator_ghx(x_ghx_sim_experiment, valid_indices_q, 1)

    
    # Save results
    results.append({
        'Selected Count': len(selected_models),
        'Valid Portion 0': CI_m,
        'Valid Portion 1': CI_Q,
        'Valid Indices': str(valid_ghx),
        'Average RMSE m': avg_rmse0,
        'Average MAE m': avg_mae0,
        'Average RMSE Q': avg_rmse1,
        'Average MAE Q': avg_mae1,
        'MeanSim RMSE m': rmse0_mean,
        'MeanSim MAE m': mae0_mean,
        'MeanSim RMSE Q': rmse1_mean,
        'MeanSim MAE Q': mae1_mean,
    })


    # Stop if finished
    if len(selected_models) == total_models:
        break

    # Select next batch of best models
    new_indices = sorted_remain_idx[iteration * increment: (iteration + 1) * increment]
    selected_models.extend(new_indices)
    iteration += 1


#Final check
print(f"Total selected models: {len(selected_models)} (Expected: {total_models})")


# Save all results
results_df = pd.DataFrame(results)
results_df.to_csv("try_4tj_CI_rmse_mae_results10init10_cor.csv", index=False)

# Extract metrics
selected_counts = results_df['Selected Count']
rmse_m = results_df['Average RMSE m']
mae_m = results_df['Average MAE m']
rmse_q = results_df['Average RMSE Q']
mae_q = results_df['Average MAE Q']

# Plot for 'm'
plt.figure(figsize=(8, 5))
plt.plot(selected_counts, rmse_m, label='RMSE - m', marker='o')
plt.plot(selected_counts, mae_m, label='MAE - m', marker='x')
plt.xlabel('Number of Selected Models')
plt.ylabel('Error')
plt.title('RMSE and MAE Trends for m')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("try_4tj_10int10_rmse_mae_trend_m.png", dpi=300)

# Plot for 'Q'
plt.figure(figsize=(8, 5))
plt.plot(selected_counts, rmse_q, label='RMSE - Q', marker='s')
plt.plot(selected_counts, mae_q, label='MAE - Q', marker='^')
plt.xlabel('Number of Selected Models')
plt.ylabel('Error')
plt.title('RMSE and MAE Trends for Q')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("try_4tj_10int10_rmse_mae_trend_q_cor.png", dpi=300)


## CI plot
print(results_df['Valid Portion 0'].iloc[0], type(results_df['Valid Portion 0'].iloc[0]))

CI_m = results_df['Valid Portion 0'].apply(lambda x: x[0])

CI_q = results_df['Valid Portion 1'].apply(lambda x: x[0])  # CI for Q

# Plot for CI of variable m (mass flow)
plt.figure(figsize=(8, 5))
plt.plot(selected_counts, CI_m, label='CI - $\dot{m}_{ghx,bypass}$', marker='o')
plt.xlabel('Number of Selected Models')
plt.ylabel('CI Coverage (95%)')
plt.title('Confidence Interval Coverage for Mass Flow')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("try_4tj_10int10_ci_coverage_massflow.png", dpi=300)

# Plot for CI of variable Q (heat)
plt.figure(figsize=(8, 5))
plt.plot(selected_counts, CI_q, label='CI - $Q_{ghx}$', marker='s')
plt.xlabel('Number of Selected Models')
plt.ylabel('CI Coverage (95%)')
plt.title('Confidence Interval Coverage for Heat')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("try_4tj_10int10_ci_coverage_Q.png", dpi=300)

# --- Mass Flow Plot (m) ---
plt.figure(figsize=(8, 5))
plt.plot(results_df['Selected Count'], results_df['MeanSim RMSE m'], label='RMSE', marker='o')
plt.plot(results_df['Selected Count'], results_df['MeanSim MAE m'], label='MAE', marker='x')
plt.xlabel('Number of Selected Models')
plt.ylabel('Error')
plt.title(r'MeanSim Errors for $\dot{m}_{ghx,bypass}$')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(f'AL_{numb_trajectories}traj_{initial_count}init{increment}_meansim_error_m.png', dpi=300)

# --- Heat Plot (Q) ---
plt.figure(figsize=(8, 5))
plt.plot(results_df['Selected Count'], results_df['MeanSim RMSE Q'], label='RMSE', marker='s')
plt.plot(results_df['Selected Count'], results_df['MeanSim MAE Q'], label='MAE', marker='^')
plt.xlabel('Number of Selected Models')
plt.ylabel('Error')
plt.title(r'MeanSim Errors for $Q_{ghx}$')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(f'AL_{numb_trajectories}traj_{initial_count}init{increment}_meansim_error_q.png', dpi=300)








