from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent

# Core folders
DATA_DIR = REPO_ROOT / "data"
MODELS_DIR = REPO_ROOT / "models"
RESULTS_DIR = REPO_ROOT / "results"
RESULTS_CSV_DIR = RESULTS_DIR / "result_csv"

# -----------------------------
# Data locations
# -----------------------------
GHX_DATA_DIR = DATA_DIR / "ghx_data_csv"
EXPERIMENT_CSV_DIR = DATA_DIR / "experiment_csv"

TES_EXP_PKL = DATA_DIR / "TES_experiment_data.pkl"
GHX_EXP_PKL = DATA_DIR / "GHX_experiment_data.pkl"
U_EXP_PKL = DATA_DIR / "U_control_experiment_data.pkl"
EXP_GHX_CSV = EXPERIMENT_CSV_DIR / "experiment_ghx_formatted.csv"

CV_HOLDOUT_DIR = DATA_DIR / "cv_holdout_preds"
NEWBOUNDS_TRAIN_DIR = DATA_DIR / "newbounds500_training_data_random"
MODEL_STORE_DIR = DATA_DIR / "newbounds_samplestotrain_model_directory"

# -----------------------------
# FNN result folders
# -----------------------------
FNN_SIM_DIR = RESULTS_CSV_DIR / "sim_figs_fnn"
FNN_WO_SIM_DIR = RESULTS_CSV_DIR / "wo_sim_figs_fnn"

FNN_AL_DIR = RESULTS_CSV_DIR / "fnn_active_figs"
FNN_WO_AL_DIR = RESULTS_CSV_DIR / "WO_fnn_active_results"

FNN_RANDOM_DIR = RESULTS_CSV_DIR / "rndmFNN_sampling_results_fnn"
FNN_WO_RANDOM_DIR = RESULTS_CSV_DIR / "timeWO_rndm_sampling_results_fnn"

FNN_PRED_EXP_DIR = RESULTS_CSV_DIR / "fnn_pred_exp_results"
FNN_WO_PRED_EXP_DIR = RESULTS_CSV_DIR / "wo_fnn_pred_exp_results"

# -----------------------------
# GRU result folders
# -----------------------------
GRU_RESULTS_DIR = RESULTS_CSV_DIR / "gru_results"
GRU_EXP_DIR = RESULTS_CSV_DIR / "gru_pred_exp_results"

GRU_AL_DIR = RESULTS_CSV_DIR / "time_gru_al_figs"
GRU_RANDOM_DIR = RESULTS_CSV_DIR / "time_rndm_gru_sampling_results"

# -----------------------------
# SINDy / MvG-SINDy result folders
# -----------------------------
SINDY_EXP_PRED_DIR = RESULTS_CSV_DIR / "1tj_err_exp_pred_sindy"

# -----------------------------
# Final top-level figures in results/
# -----------------------------
EXP_FIG = RESULTS_DIR / "Exp_ghx_model_sigmaMerge_2x1.png"
TIME_FIG = RESULTS_DIR / "cumulative_time_all_runs_0_500.png"
PRED_FIG = RESULTS_DIR / "ghx3_model_predictions_3x2.png"
RMSE_FIG = RESULTS_DIR / "rmse_trends_4x2.png"

# -----------------------------
# Ensure folders exist
# -----------------------------
def ensure_dirs():
    for d in [
        RESULTS_DIR,
        RESULTS_CSV_DIR,

        CV_HOLDOUT_DIR,
        MODEL_STORE_DIR,

        FNN_SIM_DIR,
        FNN_WO_SIM_DIR,
        FNN_AL_DIR,
        FNN_WO_AL_DIR,
        FNN_RANDOM_DIR,
        FNN_WO_RANDOM_DIR,
        FNN_PRED_EXP_DIR,
        FNN_WO_PRED_EXP_DIR,

        GRU_RESULTS_DIR,
        GRU_EXP_DIR,
        GRU_AL_DIR,
        GRU_RANDOM_DIR,

        SINDY_EXP_PRED_DIR,
    ]:
        d.mkdir(parents=True, exist_ok=True)