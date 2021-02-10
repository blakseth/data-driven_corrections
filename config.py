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

run_name  = "smoothIC11"
system    = 3
data_tag  = "smoothIC11"
model_key = 0
model_is_hybrid = True

model_types = [
    'GlobalDense',
    'GlobalCNN',
    'LocalDense',
]
model_name = model_types[model_key]

augment_training_data = False

ensemble_size = 2

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

if system == 1:
    t_end = 4.0
    x_a = 0.0
    x_b = 1.0
    A = 1.0
    k_ref     = 2500.0
    cV_ref    = 200.0
    rho       = 1000.0
    q_hat_ref = 100000.0
    T_a       = 200
    T_b       = 300
    def get_k(x):
        return k_ref * (1 + 2*x + np.sin(3*np.pi*x) + 0.8*np.cos(20*np.pi*x))
    def get_cV(x):
        return cV_ref * np.ones_like(x)
    def get_q_hat(x, t):
        return q_hat_ref * np.exp(-100*(x - 0.5*(x_b - x_a))**2)
    def get_T0(x):
        return 200 + 100*x + 100*np.sin(2*np.pi*x)
elif system == 2:
    t_end = 4.0
    x_a = 0.0
    x_b = 1.0
    A = 1.0
    k_ref     = 2000.0
    cV_ref    = 500.0
    rho       = 1000.0
    q_hat_ref = 0.0
    T_a       = 1000
    T_b       = 250
    def get_k(x):
        return 0.5 * k_ref * np.exp(2*x)
    def get_cV(x):
        return cV_ref * (1 + 4*np.heaviside(x + 0.5*(x_b - x_a), 0.5))
    def get_q_hat(x, t):
        return q_hat_ref * np.ones_like(x)
    def get_T0(x):
        return 1000/((x+1)**2) + 50*np.sin(4*np.pi*x)
elif system == 3:
    t_end = 3.0
    x_a = 0.0
    x_b = 1.0
    A = 1.0
    k_ref = 2500
    cV_ref = 200
    rho = 200
    q_hat_ref = 0
    T_a = 250
    T_b = 400
    k_discont_nodes = np.asarray([0.0, 0.12, 0.28, 0.52, 0.68, 0.72, 0.88, 1.0])
    k_prefactors = np.asarray([1.0, 5.0, 1.0, 0.1, 1.0, 10.0, 1.0])
    k_int_values = np.zeros(k_discont_nodes.shape[0])
    def get_k(x):
        if type(x) is np.ndarray:
            ks = []
            for i, x_value in enumerate(x):
                for j in range(k_prefactors.shape[0]):
                    if k_discont_nodes[j] <= x_value <= k_discont_nodes[j + 1]:
                        ks.append(k_ref * k_prefactors[j])
            return np.asarray(ks)
        else:
            for j in range(k_prefactors.shape[0]):
                if k_discont_nodes[j] <= x < k_discont_nodes[j + 1]:
                    return k_ref * k_prefactors[j]
    def get_k_approx(x):
        return np.ones_like(x) * k_ref
    def get_cV(x):
        return np.ones_like(x) * cV_ref
    def get_q_hat(x, t):
        return np.zeros_like(x)
    def get_q_hat_approx(x, t):
        return np.zeros_like(x)
    def get_T0(x):
        return T_a + (T_b - T_a) * (x + 0.5 * np.sin(2 * np.pi * x))
elif system == 4:
    t_end = 0.1
    x_a = 0.0
    x_b = 1.0
    A = 1.0
    rho = 1.0
    k_ref = 1.0
    cV_ref = 1.0
    T_a = 0.0
    T_b = 1.0
    q_hat_ref = 6.0
    def get_k(x):
        return np.ones_like(x) * k_ref
    def get_cV(x):
        return np.ones_like(x) * cV_ref
    def get_q_hat(x, t):
        return x * q_hat_ref
    def get_T0(x):
        return x
elif system == 5:
    t_end = 0.1
    x_a = 0.0
    x_b = 1.0
    A = 1.0
    rho = 1.0
    k_ref = 1.0
    cV_ref = 1.0
    T_a = 0.0
    T_b = 0.0
    def get_k(x):
        return np.ones_like(x) * k_ref
    def get_cV(x):
        return np.ones_like(x) * cV_ref
    def get_q_hat(x, t):
        return (1 + 4*np.pi**2)*np.sin(2*np.pi*x)*1
    def get_T0(x):
        return x
else:
    raise Exception("Invalid domain selection.")


########################################################################################################################
# Discretization.

# Coarse spatial discretization.
N_coarse = 20
dx_coarse = (x_b - x_a) / N_coarse
faces_coarse = np.linspace(x_a, x_b, num = N_coarse + 1, endpoint=True)
nodes_coarse = np.zeros(N_coarse + 2)
nodes_coarse[0] = x_a
nodes_coarse[1:-1] = faces_coarse[:-1] + dx_coarse / 2
nodes_coarse[-1] = x_b

# Fine spatial discretization.
N_fine = 4860
dx_fine = (x_b - x_a) / N_fine
faces_fine = np.linspace(x_a, x_b, num = N_fine + 1, endpoint=True)
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
N_train_examples = int(0.8*Nt_coarse)
N_val_examples   = int(0.1*Nt_coarse)
N_test_examples  = int(0.1*Nt_coarse)

# Parameters for shift data augmentation.
N_shift_steps   = 5
shift_step_size = 100

# Test iterations at which temperature profiles are saved.
profile_save_steps = np.asarray([1, int(np.sqrt(N_test_examples)), N_test_examples]) - 1

########################################################################################################################
# Model configuration.

num_layers = 3

hidden_layer_size = 100

loss_func = 'MSE'

optimizer = 'adam'
learning_rate = 1e-4

act_type = 'lrelu'
act_param = 0.2

use_dropout = False
dropout_prop = 0.1

########################################################################################################################
# Training configuration.

max_train_it = int(2e4)

print_train_loss_period = int(1e2)    # Number of training iterations per print of training losses.
save_model_period       = int(5e10)   # Number of training iterations per model save.
validation_period       = int(1e2)    # Number of training iterations per validation.

batch_size_train = 64
batch_size_val   = N_val_examples
batch_size_test  = N_test_examples

overfit_limit = 1e3