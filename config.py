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

run_name = "my_run"
system   = 2
data_tag = "my_data"

ensemble_size = 10

do_train = True
do_test  = True

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

t_end     = 5.0
x_a       = 0.0
x_b       = 1.0
A         = 1.0

if system == 1:
    k_ref = 2500.0
    cV_ref = 200.0
    rho = 1000.0
    q_hat_ref = 100000.0
    def get_k(x):
        return k_ref * (1 + 2*x + np.sin(3*np.pi*x) + 0.8*np.cos(20*np.pi*x))
    def get_cV(x):
        return cV_ref * np.ones_like(x)
    def get_q_hat(x):
        return q_hat_ref * np.exp(-100*(x - 0.5*(x_b - x_a))**2)
elif system == 2:
    k_ref = 2000.0
    cV_ref = 500.0
    rho = 1000.0
    q_hat_ref = 0.0
    def get_k(x):
        return 0.5 * k_ref * np.exp(2*x)
    def get_cV(x):
        return cV_ref * (1 + 4*np.heaviside(x + 0.5*(x_b - x_a), 0.5))
    def get_q_hat(x):
        return q_hat_ref * np.ones_like(x)
else:
    raise Exception("Invalid domain selection.")


########################################################################################################################
# Discretization.

# Coarse spatial discretization.
N_coarse = 25
dx_coarse = (x_b - x_a) / N_coarse
faces_coarse = np.linspace(x_a, x_b, num=N_coarse + 1, endpoint=True)
nodes_coarse = np.zeros(N_coarse + 2)
nodes_coarse[0] = x_a
nodes_coarse[1:-1] = faces_coarse[:-1] + dx_coarse / 2
nodes_coarse[-1] = x_b

# Fine spatial discretization.
N_fine = 250
dx_fine = (x_b - x_a) / N_fine
faces_fine = np.linspace(x_a, x_b, num=N_fine + 1, endpoint=True)
nodes_fine = np.zeros(N_fine + 2)
nodes_fine[0] = x_a
nodes_fine[1:-1] = faces_fine[:-1] + dx_fine / 2
nodes_fine[-1] = x_b

# Temporal discretization.
dt_fine   = 1.0
dt_coarse = 1.0

########################################################################################################################
# Data configuration.