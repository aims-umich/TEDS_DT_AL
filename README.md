# TEDS_DT_AL

This repository contains code and results for **active learning (AL)** applied to **digital twin (DT)** surrogate modeling of the **Thermal Energy Distribution System (TEDS)**.

## Overview

### TEDS
TEDS is a thermal energy distribution test platform with key components such as:
- Thermal Energy Storage (TES)
- Glycol Heat Exchanger (GHX)
- Heater and control inputs

In this project, the focus is on the **GHX subsystem** during discharge operation.

### Digital Twin (DT)
A digital twin is a fast surrogate representation of the physical system that can support:
- real-time prediction,
- control-oriented analysis,
- uncertainty-aware decision support.

Because experimental data are limited, simulation data and surrogate models are used to build the DT.

### Active Learning (AL)
Active learning is used to select the **most informative trajectories** for training, instead of randomly using all trajectories.  
This improves:
- data efficiency,
- convergence speed,
- practical surrogate training for DT applications.

## Surrogate Models

This repository compares four surrogate model types:

1. **SINDyC**  
   Sparse Identification of Nonlinear Dynamics with Control  
   - interpretable
   - equation-based
   - fast for control-oriented applications

2. **MvG-SINDyC**  
   Multivariate-Gaussian SINDyC  
   - probabilistic extension of SINDyC
   - provides uncertainty quantification
   
3. **FNN**  
   Feedforward Neural Network  
   - flexible nonlinear baseline
   - no temporal memory

4. **GRU**  
   Gated Recurrent Unit  
   - sequence-based neural network
   - strong performance for transient dynamic prediction

---
### Environment Setup

Clone the repository and move into the project folder:

git clone https://github.com/aims-umich/TEDS_DT_AL.git
cd TEDS_DT_AL

Create the Conda environment from the provided environment file:

conda env create -f environment_torch.yml
conda activate torchgpu

### Example 1: FNN with experimental data

Run:

python models/FNN/al_fnn_exp.py

This script trains or evaluates the FNN model that uses experimental information.

What to expect
terminal output showing training or evaluation progress
metrics saved in the repository results folders
optional plots saved if plotting is enabled in the script
Expected output location

Typical outputs for this workflow are stored under:

results/result_csv/fnn_active_figs/

For example:

results/result_csv/fnn_active_figs/active_learning_exp_metrics.csv

### Example 2: Experimental prediction comparison plot

Run:

python virPred_exp_plot.py

This script compares experimental prediction performance for:

MvG-SINDyC
SINDyC
GRU
FNN (with EXP)
FNN (w/o EXP)

Expected output: The comparison figure is typically saved in:

results/

For example:

results/Exp_ghx_model_sigmaMerge_2x1.png












