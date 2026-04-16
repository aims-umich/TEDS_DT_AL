# Load python packages
# Data processing
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sdf
import json,pickle
from itertools import combinations
import random

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
    #plt.savefig(f"rndm_12Tj_10init10_summary_selected_{selected_model_len}.png", dpi=300)
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
model_dir = "/home/unabila/ghxSindy/newbounds_8_samplestotrain_model_directory/"
    
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

# ================== MANUAL HOLDOUT (incl. 300), TRAIN, AND PREDICT 363 ==================


# --- 0) Build the set of "valid" GHX trajectories using your bounds ---
valid_ghx_idx = []
valid_ghx_idx = list(range(len(x_ghx_train_all)))

print(f"[INFO] Using ALL loaded trajectories as 'valid': {len(valid_ghx_idx)}")

# --- 1) Manual TEST set of 10 (must be subset of valid), including 363 ---
test_indices = [12, 56, 74, 81, 121, 136, 176, 233, 266, 300]
missing = [i for i in test_indices if i not in valid_ghx_idx]
if missing:
    raise ValueError(f"These test_indices are not valid per bounds: {missing}")
target_idx = 300  # ##############################################################################  we will predict this one

# --- 2) TRAIN set = valid minus manual test set ---
train_indices = [i for i in valid_ghx_idx if i not in test_indices]
print(f"[INFO] Train size: {len(train_indices)} | Test size: {len(test_indices)}")

# --- 3) Assemble training lists (shapes SINDyC_fit expects) ---
x_tes_train = [x_tes_train_all[i] for i in train_indices]         # list of (5251, 5)
u_tes_train = [u_tes_train_all[i] for i in train_indices]         # list of (5251, 4)
x_ghx_train = [x_ghx_train_all[i] for i in train_indices]         # list of (5251, 2)
u_ghx_train = [u_ghx_train_all[i][:, :4] for i in train_indices]  # list of (5251, 4)

# --- 4) Fit fresh SINDyC models on TRAIN set ---
threshold_tes, alpha_tes = 1e-6, 1e-3
threshold_ghx, alpha_ghx = 1e-8, 1e-6

model_tes_cv = SINDyC_fit(x_tes_train, u_tes_train, tm2, threshold=threshold_tes, alpha=alpha_tes,
                          name=f'{model_dir}/TES_cv', verbose=False)
model_ghx_cv = SINDyC_fit(x_ghx_train, u_ghx_train, tm2, threshold=threshold_ghx, alpha=alpha_ghx,
                          name=f'{model_dir}/GHX_cv', verbose=False)

# --- 5) Treat index 363 as *experiment* so existing code keeps working ---


x_ghx_experiment = x_ghx_train_all[target_idx].T                 # (2, 5251)
x_ghx_experiment_conv = x_ghx_experiment[:, :-1]                 # (2, 5250)

x_tes_experiment = x_tes_train_all[target_idx].T                 # (5, 5251)
x_tes_experiment_conv = x_tes_experiment[:, :-1]                 # (5, 5250)

u_experiment_conv_2 = u_ghx_train_all[target_idx][:, :4].T       # (4, 5251)

# (Optional) if you want smoothed plots to resemble prior look:
# window_len, polyorder = 1200, 3
# x_ghx_experiment_sm = savgol_filter(x_ghx_experiment, window_len, polyorder, axis=1)

# --- 6) Predict m & Q for index 300 using the CV models ---
x_tes_test = x_tes_train_all[target_idx]              # (5251, 5)
x_ghx_test = x_ghx_train_all[target_idx]              # (5251, 2)  (truth)
u_ghx_test = u_ghx_train_all[target_idx][:, :4]       # (5251, 4)

pred_list = simulate_sindyc(model_tes_cv, model_ghx_cv, x_tes_test, x_ghx_test, u_ghx_test, tm2)
pred_arr  = np.array(pred_list)       # (5250, 7)
pred_ghx  = pred_arr[:, 5:]           # (5250, 2) -> [mflow_pred, Q_pred]

# Align truth to prediction length
T_pred = pred_ghx.shape[0]
true_ghx = x_ghx_test[:T_pred, :]     # (5250, 2)

# --- 7) Metrics ---
rmse_m = np.sqrt(mean_squared_error(true_ghx[:, 0], pred_ghx[:, 0]))
mae_m  = mean_absolute_error(true_ghx[:, 0], pred_ghx[:, 0])
rmse_q = np.sqrt(mean_squared_error(true_ghx[:, 1], pred_ghx[:, 1]))
mae_q  = mean_absolute_error(true_ghx[:, 1], pred_ghx[:, 1])

print(f"[METRICS] idx=363 | mflow: RMSE={rmse_m:.4f}, MAE={mae_m:.4f} | "
      f"Q: RMSE={rmse_q:.4f}, MAE={mae_q:.4f}")

# --- 8) Save predicted vs true series + one-row metrics CSV ---
os.makedirs("cv_holdout_preds", exist_ok=True)
time_s = np.asarray(tm2[:-1])[:T_pred]

pd.DataFrame({
    "time_s": time_s,
    "mflow_true_kgps": true_ghx[:, 0],
    "mflow_pred_kgps": pred_ghx[:, 0],
    "Q_true_kW": true_ghx[:, 1],
    "Q_pred_kW": pred_ghx[:, 1],
}).to_csv("cv_holdout_preds/ghx_pred_index363.csv", index=False)

pd.DataFrame([{
    "index": target_idx,
    "rmse_mflow": rmse_m,
    "mae_mflow": mae_m,
    "rmse_Q": rmse_q,
    "mae_Q": mae_q,
    "train_size": len(train_indices),
    "num_test": len(test_indices)
}]).to_csv("cv_holdout_preds/ghx_metrics_index363.csv", index=False)

print("[SAVED] cv_holdout_preds/ghx_pred_index363.csv")
print("[SAVED] cv_holdout_preds/ghx_metrics_index363.csv")

# --- 1) Time-series overlays ---
# mflow (kg/s)
plt.figure(figsize=(9, 5))
plt.plot(time_s, true_ghx[:, 0], label="True mflow (kg/s)")
plt.plot(time_s, pred_ghx[:, 0], label="Pred mflow (kg/s)", linestyle="--")
plt.title(f"Index {target_idx} mflow: True vs Pred")
plt.xlabel("time (s)")
plt.ylabel("mflow_GHX_bypass (kg/s)")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig("cv_holdout_preds/idx363_mflow_true_vs_pred_timeseries.png", dpi=200)
plt.close()

# Q (kW)
plt.figure(figsize=(9, 5))
plt.plot(time_s, true_ghx[:, 1], label="True Q (kW)")
plt.plot(time_s, pred_ghx[:, 1], label="Pred Q (kW)", linestyle="--")
plt.title(f"Index {target_idx} Q: True vs Pred")
plt.xlabel("time (s)")
plt.ylabel("Q_ghx (kW)")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig("cv_holdout_preds/idx363_Q_true_vs_pred_timeseries.png", dpi=200)
plt.close()





