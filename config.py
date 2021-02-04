"""
config.py

Written by Sindre Stenen Blakseth, 2021.

Main configuration file.
"""

########################################################################################################################
# Package imports.

import numpy as np
import os
import torch

########################################################################################################################
# Reproducibility.

torch.manual_seed(0)
np.random.seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

########################################################################################################################
# Run configuration.

run_name = "dataset_test1"
system   = 1
data_tag = "dataset_test1"

augment_training_data = False

ensemble_size = 10

do_train = False
do_test  = False

load_model_from_save      = False
resume_training_from_save = False
model_load_path           = None #os.path.join(cp_load_dir, 'G_150000.pth')

########################################################################################################################
# Environment configuration.

base_dir     = '/home/sindre/msc_thesis/data-driven_corrections'
datasets_dir = os.path.join(base_dir,    'datasets')
results_dir  = os.path.join(base_dir,    'results')
run_dir      = os.path.join(results_dir, run_name)
cp_main_dir  = os.path.join(base_dir,    'checkpoints')
cp_load_dir  = os.path.join(cp_main_dir, '')
cp_save_dir  = os.path.join(cp_main_dir, run_name)

########################################################################################################################
# Domain configuration.

t_end     = 0.1
x_a       = 0.0
x_b       = 1.0
A         = 1.0

if system == 1:
    k_ref = 2500.0
    cV_ref = 200.0
    rho = 1000.0
    q_hat_ref = 100000.0
    T_a = 200
    T_b = 300
    def get_k(x):
        return k_ref * (1 + 2*x + np.sin(3*np.pi*x) + 0.8*np.cos(20*np.pi*x))
    def get_cV(x):
        return cV_ref * np.ones_like(x)
    def get_q_hat(x):
        return q_hat_ref * np.exp(-100*(x - 0.5*(x_b - x_a))**2)
    def get_T0(x):
        return 200 + 100*x + 100*np.sin(2*np.pi*x)
elif system == 2:
    k_ref = 2000.0
    cV_ref = 500.0
    rho = 1000.0
    q_hat_ref = 0.0
    T_a = 1000
    T_b = 250
    def get_k(x):
        return 0.5 * k_ref * np.exp(2*x)
    def get_cV(x):
        return cV_ref * (1 + 4*np.heaviside(x + 0.5*(x_b - x_a), 0.5))
    def get_q_hat(x):
        return q_hat_ref * np.ones_like(x)
    def get_T0(x):
        return 1000/((x+1)**2) + 50*np.sin(4*np.pi*x)
else:
    raise Exception("Invalid domain selection.")


########################################################################################################################
# Discretization.

# Coarse spatial discretization.
N_coarse = 20
dx_coarse = (x_b - x_a) / N_coarse
faces_coarse = np.linspace(x_a, x_b, num=N_coarse + 1, endpoint=True)
nodes_coarse = np.zeros(N_coarse + 2)
nodes_coarse[0] = x_a
nodes_coarse[1:-1] = faces_coarse[:-1] + dx_coarse / 2
nodes_coarse[-1] = x_b

# Fine spatial discretization.
N_fine = 4860
dx_fine = (x_b - x_a) / N_fine
faces_fine = np.linspace(x_a, x_b, num=N_fine + 1, endpoint=True)
nodes_fine = np.zeros(N_fine + 2)
nodes_fine[0] = x_a
nodes_fine[1:-1] = faces_fine[:-1] + dx_fine / 2
nodes_fine[-1] = x_b

# Temporal discretization.
dt_fine   = 0.000125
dt_coarse = 0.001
Nt_fine    = int(t_end / dt_fine) + 1
Nt_coarse  = int(t_end / dt_coarse) + 1

########################################################################################################################
# Data configuration.

# Dataset sizes (unaugmented).
N_train_examples = int(0.6*Nt_coarse)
N_val_examples   = int(0.2*Nt_coarse)
N_test_examples  = int(0.2*Nt_coarse)

# Parameters for shift data augmentation.
N_shift_steps   = 5
shift_step_size = 100

########################################################################################################################
# Model configuration.

act_type = 'lrelu'
act_param = 0.01

use_dropout = False
dropout_prop = 0.1