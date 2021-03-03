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
# Configuration parameters

group_name = "testing_several_runs_at_once"
run_names  = [["testing_several_runs_at_once_s1", "testing_several_runs_at_once_s2A"]]
systems    = ["1", "2A"]
data_tags  = ["test1", "test2A"]
model_keys = [4]
assert len(systems) == len(data_tags) == len(run_names[0])
assert len(run_names) == len(model_keys)


########################################################################################################################
# Create config object.

class Config:
    def __init__(self, group_name, run_name, system, data_tag, model_key, do_train, do_test):
        # ---------------------------------------------------------------------------------------------------------------
        # Run configuration.
        self.group_name = group_name
        self.run_name   = run_name
        self.system     = system
        self.data_tag   = data_tag
        self.model_key  = model_key
        self.model_is_hybrid = True

        model_types = [
            'GlobalDense',
            'GlobalCNN',
            'LocalDense',
            'EnsembleLocalDense',
            'EnsembleGlobalCNN'
        ]
        self.model_name = model_types[model_key]

        self.augment_training_data = False

        self.ensemble_size = 2

        self.do_train = do_train
        self.do_test = do_test

        self.load_model_from_save = False
        self.resume_training_from_save = False
        self.model_load_path = None  # os.path.join(cp_load_dir, 'G_150000.pth')

        #---------------------------------------------------------------------------------------------------------------
        # Environment configuration.

        self.base_dir     = '/home/sindre/msc_thesis/data-driven_corrections'
        self.datasets_dir = os.path.join(self.base_dir, 'datasets')
        self.results_dir  = os.path.join(self.base_dir, 'results')
        self.group_dir    = os.path.join(self.results_dir, group_name)
        self.run_dir      = os.path.join(self.group_dir, run_name)
        self.cp_main_dir  = os.path.join(self.base_dir, 'checkpoints')
        self.cp_load_dir  = os.path.join(self.cp_main_dir, '')
        self.cp_save_dir  = os.path.join(self.cp_main_dir, run_name)

        #---------------------------------------------------------------------------------------------------------------
        # Domain configuration.
        if self.system == "1":
            exact_solution_available = True
            t_end     = 5.0
            x_a       = 0.0
            x_b       = 1.0
            A         = 1.0
            rho       = 1.0
            k_ref     = 1.0
            cV_ref    = 1.0
            q_hat_ref = 1.0
            def get_T_exact(x, t):
                return t + 0.5 * (x ** 2)
            def get_T0(x):
                return get_T_exact(x, 0)
            def get_T_a(t):
                return get_T_exact(x_a, t)
            def get_T_b(t):
                return get_T_exact(x_b, t)
            def get_q_hat(x, t):
                return -2 * np.ones_like(x)
            def get_q_hat_approx(x, t):
                return 0.8 * get_q_hat(x, t)
            def get_k(x):
                return np.ones_like(x) * k_ref
            def get_k_approx(x):
                return get_k(x)
            def get_cV(x):
                return np.ones_like(x) * cV_ref
        elif self.system == "2A":
            exact_solution_available = True
            t_end     = 5.0
            x_a       = 0.0
            x_b       = 1.0
            A         = 1.0
            rho       = 1.0
            k_ref     = 1.0
            cV_ref    = 1.0
            q_hat_ref = 1.0
            def get_T_exact(x, t):
                return np.sqrt(t) + 10 * (x ** 2) * (x - 1) * (x + 2)
            def get_T0(x):
                return get_T_exact(x, 0)
            def get_T_a(t):
                return get_T_exact(x_a, t)
            def get_T_b(t):
                return get_T_exact(x_b, t)
            def get_q_hat(x, t):
                return -1 / (2 * np.sqrt(t)) - 20 * (6 * (x ** 2) + 3 * x - 2)
            def get_q_hat_approx(x, t):
                return 0.8 * get_q_hat(x, t)
            def get_k(x):
                return np.ones_like(x) * k_ref
            def get_k_approx(x):
                return get_k(x)
            def get_cV(x):
                return np.ones_like(x) * cV_ref
        elif self.system == "2B":
            exact_solution_available = True
            t_end     = 5.0
            x_a       = 0.0
            x_b       = 1.0
            A         = 1.0
            rho       = 1.0
            k_ref     = 1.0
            cV_ref    = 1.0
            q_hat_ref = 1.0
            def get_T_exact(x, t):
                return 2 * np.sqrt(t) + 7 * (x ** 2) * (x - 1) * (x + 2)
            def get_T0(x):
                return get_T_exact(x, 0)
            def get_T_a(t):
                return get_T_exact(x_a, t)
            def get_T_b(t):
                return get_T_exact(x_b, t)
            def get_q_hat(x, t):
                return -1 / (np.sqrt(t)) - 14 * (6 * (x ** 2) + 3 * x - 2)
            def get_q_hat_approx(x, t):
                return 0.8 * get_q_hat(x, t)
            def get_k(x):
                return np.ones_like(x) * k_ref
            def get_k_approx(x):
                return get_k(x)
            def get_cV(x):
                return np.ones_like(x) * cV_ref
        elif self.system == "3":
            exact_solution_available = True
            t_end = 5.0
            x_a = 0.0
            x_b = 1.0
            A = 1.0
            rho = 1.0
            k_ref = 1.0
            cV_ref = 1.0
            q_hat_ref = 1.0
            def get_T_exact(x, t):
                return 2 * (x ** 2) - (t ** 2) * x * (x - 1)
            def get_T0(x):
                return get_T_exact(x, 0)
            def get_T_a(t):
                return get_T_exact(x_a, t)
            def get_T_b(t):
                return get_T_exact(x_b, t)
            def get_q_hat(x, t):
                return 2 * t * x * (x - 1) + 2 * (t ** 2) - 4
            def get_q_hat_approx(x, t):
                return 0.8 * get_q_hat(x, t)
            def get_k(x):
                return np.ones_like(x) * k_ref
            def get_k_approx(x):
                return get_k(x)
            def get_cV(x):
                return np.ones_like(x) * cV_ref
        elif self.system == "4":
            exact_solution_available = True
            t_end = 1.0
            x_a = 0.0
            x_b = 1.0
            A = 1.0
            rho = 1.0
            k_ref = 1.0
            cV_ref = 1.0
            q_hat_ref = 1.0
            def get_T_exact(x, t):
                return np.sin(2 * np.pi * x) * np.exp(-t)
            def get_T0(x):
                return get_T_exact(x, 0)
            def get_T_a(t):
                return get_T_exact(x_a, t)
            def get_T_b(t):
                return get_T_exact(x_b, t)
            def get_q_hat(x, t):
                return (1 + 4 * (np.pi ** 2)) * np.sin(2 * np.pi * x) * np.exp(-t)
            def get_q_hat_approx(x, t):
                return 0.8 * get_q_hat(x, t)
            def get_k(x):
                return np.ones_like(x) * k_ref
            def get_k_approx(x):
                return get_k(x)
            def get_cV(x):
                return np.ones_like(x) * cV_ref
        elif self.system == "5A":
            exact_solution_available = True
            t_end = 5.0
            x_a = 0.0
            x_b = 1.0
            A = 1.0
            rho = 1.0
            k_ref = 1.0
            cV_ref = 1.0
            q_hat_ref = 1.0
            def get_T_exact(x, t):
                return -2 * (x ** 3) * (x - 1) / (t + 0.5)
            def get_T0(x):
                return get_T_exact(x, 0)
            def get_T_a(t):
                return get_T_exact(x_a, t)
            def get_T_b(t):
                return get_T_exact(x_b, t)
            def get_q_hat(x, t):
                return (24 * (x ** 2) - 12 * x) / (t + 0.5) - (2 * (x ** 4) - 2 * (x ** 3)) / ((t + 0.5) ** 2)
            def get_q_hat_approx(x, t):
                return 0.8 * get_q_hat(x, t)
            def get_k(x):
                return np.ones_like(x) * k_ref
            def get_k_approx(x):
                return get_k(x)
            def get_cV(x):
                return np.ones_like(x) * cV_ref
        elif self.system == "5B":
            exact_solution_available = True
            t_end = 5.0
            x_a = 0.0
            x_b = 1.0
            A = 1.0
            rho = 1.0
            k_ref = 1.0
            cV_ref = 1.0
            q_hat_ref = 1.0
            def get_T_exact(x, t):
                return -(x ** 3) * (x - 1) / (t + 0.1)
            def get_T0(x):
                return get_T_exact(x, 0)
            def get_T_a(t):
                return get_T_exact(x_a, t)
            def get_T_b(t):
                return get_T_exact(x_b, t)
            def get_q_hat(x, t):
                return (12 * (x ** 2) - 6 * x) / (t + 0.1) - ((x ** 4) - (x ** 3)) / ((t + 0.1) ** 2)
            def get_q_hat_approx(x, t):
                return 0.8 * get_q_hat(x, t)
            def get_k(x):
                return np.ones_like(x) * k_ref
            def get_k_approx(x):
                return get_k(x)
            def get_cV(x):
                return np.ones_like(x) * cV_ref
        elif self.system == "6":
            exact_solution_available = True
            t_end = 5.0
            x_a = 0.0
            x_b = 1.0
            A = 1.0
            rho = 1.0
            k_ref = 1.0
            cV_ref = 1.0
            q_hat_ref = 1.0
            def get_T_exact(x, t):
                return 2 + (x - 1) * np.tanh(x / (t + 0.1))
            def get_T0(x):
                return get_T_exact(x, 0)
            def get_T_a(t):
                return get_T_exact(x_a, t)
            def get_T_b(t):
                return get_T_exact(x_b, t)
            def get_q_hat(x, t):
                return (x * (x - 1) - 2 * ((x - 1) * np.tanh(x / (t + 0.1)) + t + 0.1)) / (
                            ((t + 0.1) * np.cosh(x / (t + 0.1))) ** 2)
            def get_q_hat_approx(x, t):
                return 0.8 * get_q_hat(x, t)
            def get_k(x):
                return np.ones_like(x) * k_ref
            def get_k_approx(x):
                return get_k(x)
            def get_cV(x):
                return np.ones_like(x) * cV_ref
        elif self.system == "7":
            exact_solution_available = True
            t_end = 5.0
            x_a = 0.0
            x_b = 1.0
            A = 1.0
            rho = 1.0
            k_ref = 1.0
            cV_ref = 1.0
            q_hat_ref = 1.0
            def get_T_exact(x, t):
                return np.sin(2 * np.pi * t) + np.sin(2 * np.pi * x)
            def get_T0(x):
                return get_T_exact(x, 0)
            def get_T_a(t):
                return get_T_exact(x_a, t)
            def get_T_b(t):
                return get_T_exact(x_b, t)
            def get_q_hat(x, t):
                return 4 * (np.pi ** 2) * np.sin(2 * np.pi * x) - 2 * np.pi * np.cos(2 * np.pi * t)
            def get_q_hat_approx(x, t):
                return 0.8 * get_q_hat(x, t)
            def get_k(x):
                return np.ones_like(x) * k_ref
            def get_k_approx(x):
                return get_k(x)
            def get_cV(x):
                return np.ones_like(x) * cV_ref
        elif self.system == "8A":
            exact_solution_available = True
            t_end = 5.0
            x_a = 0.0
            x_b = 1.0
            A = 1.0
            rho = 1.0
            k_ref = 1.0
            cV_ref = 1.0
            q_hat_ref = 1.0
            def get_T_exact(x, t):
                return 1 + np.sin(2 * np.pi * t) * np.cos(2 * np.pi * x)
            def get_T0(x):
                return get_T_exact(x, 0)
            def get_T_a(t):
                return get_T_exact(x_a, t)
            def get_T_b(t):
                return get_T_exact(x_b, t)
            def get_q_hat(x, t):
                return 4 * (np.pi ** 2) * np.sin(2 * np.pi * t) * np.cos(2 * np.pi * x) - 2 * np.pi * np.cos(
                    2 * np.pi * t) * np.cos(2 * np.pi * x)
            def get_q_hat_approx(x, t):
                return 0.8 * get_q_hat(x, t)
            def get_k(x):
                return np.ones_like(x) * k_ref
            def get_k_approx(x):
                return get_k(x)
            def get_cV(x):
                return np.ones_like(x) * cV_ref
        elif self.system == "8B":
            exact_solution_available = True
            t_end = 5.0
            x_a = 0.0
            x_b = 1.0
            A = 1.0
            rho = 1.0
            k_ref = 1.0
            cV_ref = 1.0
            q_hat_ref = 1.0
            def get_T_exact(x, t):
                return 1 + np.sin(3 * np.pi * t) * np.cos(4 * np.pi * x)
            def get_T0(x):
                return get_T_exact(x, 0)
            def get_T_a(t):
                return get_T_exact(x_a, t)
            def get_T_b(t):
                return get_T_exact(x_b, t)
            def get_q_hat(x, t):
                return 16 * (np.pi ** 2) * np.sin(3 * np.pi * t) * np.cos(4 * np.pi * x) - 3 * np.pi * np.cos(
                    3 * np.pi * t) * np.cos(4 * np.pi * x)
            def get_q_hat_approx(x, t):
                return 0.8 * get_q_hat(x, t)
            def get_k(x):
                return np.ones_like(x) * k_ref
            def get_k_approx(x):
                return get_k(x)
            def get_cV(x):
                return np.ones_like(x) * cV_ref
        elif self.system == "9":
            exact_solution_available = True
            t_end = 2.0
            x_a = 0.0
            x_b = 1.0
            A = 1.0
            rho = 1.0
            k_ref = 1.0
            cV_ref = 1.0
            q_hat_ref = 1.0
            def get_T_exact(x, t):
                return 1 + np.sin(2 * np.pi * x * (t ** 2))
            def get_T0(x):
                return get_T_exact(x, 0)
            def get_T_a(t):
                return get_T_exact(x_a, t)
            def get_T_b(t):
                return get_T_exact(x_b, t)
            def get_q_hat(x, t):
                return 4 * (np.pi ** 2) * (t ** 4) * np.sin(2 * np.pi * x * (t ** 2)) - 4 * np.pi * x * t * np.cos(
                    2 * np.pi * x * (t ** 2))
            def get_q_hat_approx(x, t):
                return 0.5 * get_q_hat(x, t)
            def get_k(x):
                return np.ones_like(x) * k_ref
            def get_k_approx(x):
                return get_k(x)
            def get_cV(x):
                return np.ones_like(x) * cV_ref
        elif self.system == "10":
            exact_solution_available = True
            t_end = 5.0
            x_a = 0.0
            x_b = 1.0
            A = 1.0
            rho = 1.0
            k_ref = 1.0
            cV_ref = 1.0
            q_hat_ref = 1.0
            def get_T_exact(x, t):
                return 5 + x * (x - 1) / (t + 0.1) + 0.1 * t * np.sin(2 * np.pi * x)
            def get_T0(x):
                return get_T_exact(x, 0)
            def get_T_a(t):
                return get_T_exact(x_a, t)
            def get_T_b(t):
                return get_T_exact(x_b, t)
            def get_q_hat(x, t):
                return x * (x - 1) / ((t + 0.1) ** 2) - 2 / (t + 0.1) - (0.1 - 0.2 * np.pi * t) * np.sin(2 * np.pi * x)
            def get_q_hat_approx(x, t):
                return 0.8 * get_q_hat(x, t)
            def get_k(x):
                return np.ones_like(x) * k_ref
            def get_k_approx(x):
                return get_k(x)
            def get_cV(x):
                return np.ones_like(x) * cV_ref
        elif self.system == "11":
            exact_solution_available = True
            t_end = 5.0
            x_a = 0.0
            x_b = 1.0
            A = 1.0
            rho = 1.0
            k_ref = 1.0
            cV_ref = 1.0
            q_hat_ref = 1.0
            def get_T_exact(x, t):
                return 1 + np.sin(5 * x * t) * np.exp(-0.2 * x * t)
            def get_T0(x):
                return get_T_exact(x, 0)
            def get_T_a(t):
                return get_T_exact(x_a, t)
            def get_T_b(t):
                return get_T_exact(x_b, t)
            def get_q_hat(x, t):
                return ((2 * (t ** 2) - 5 * x) * np.cos(5 * x * t) + (0.2 * x + 24.96 * (t ** 2)) * np.sin(
                    5 * x * t)) * np.exp(-0.2 * x * t)
            def get_q_hat_approx(x, t):
                return 0.8 * get_q_hat(x, t)
            def get_k(x):
                return np.ones_like(x) * k_ref
            def get_k_approx(x):
                return get_k(x)
            def get_cV(x):
                return np.ones_like(x) * cV_ref
        elif self.system == "12":
            exact_solution_available = True
            t_end = 5.0
            x_a = 0.0
            x_b = 1.0
            A = 1.0
            rho = 1.0
            k_ref = 1.0
            cV_ref = 1.0
            q_hat_ref = 1.0
            def get_T_exact(x, t):
                return 5 * t * (x ** 2) * np.sin(10 * np.pi * t) + np.sin(2 * np.pi * x) / (t + 0.2)
            def get_T0(x):
                return get_T_exact(x, 0)
            def get_T_a(t):
                return get_T_exact(x_a, t)
            def get_T_b(t):
                return get_T_exact(x_b, t)
            def get_q_hat(x, t):
                return (4 * (np.pi ** 2) + 1 / (t + 0.2)) * np.sin(2 * np.pi * x) / (t + 0.2) - (
                            5 * (x ** 2) + 10 * t) * np.sin(10 * np.pi * t) - 50 * np.pi * (x ** 2) * (t ** 2) * np.cos(
                    10 * np.pi * t)
            def get_q_hat_approx(x, t):
                return 0.8 * get_q_hat(x, t)
            def get_k(x):
                return np.ones_like(x) * k_ref
            def get_k_approx(x):
                return get_k(x)
            def get_cV(x):
                return np.ones_like(x) * cV_ref
        elif self.system == "13":
            exact_solution_available = True
            t_end = 5.0
            x_a = 0.0
            x_b = 1.0
            A = 1.0
            rho = 1.0
            k_ref = 1.0
            cV_ref = 1.0
            q_hat_ref = 1.0
            def get_T_exact(x, t):
                return 1 + t / (1 + ((x - 0.5) ** 2))
            def get_T0(x):
                return get_T_exact(x, 0)
            def get_T_a(t):
                return get_T_exact(x_a, t)
            def get_T_b(t):
                return get_T_exact(x_b, t)
            def get_q_hat(x, t):
                return (2 * x + 2 * t - 1) / ((1 + (x - 0.5) ** 2) ** 2) - (8 * t * (x - 0.5) ** 2) / (
                            (1 + (x - 0.5) ** 2) ** 3)
            def get_q_hat_approx(x, t):
                return 0.8 * get_q_hat(x, t)
            def get_k(x):
                return np.ones_like(x) * k_ref
            def get_k_approx(x):
                return get_k(x)
            def get_cV(x):
                return np.ones_like(x) * cV_ref
        elif self.system == "14":
            exact_solution_available = True
            t_end = 5.0
            x_a = 0.0
            x_b = 1.0
            A = 1.0
            rho = 1.0
            k_ref = 1.0
            cV_ref = 1.0
            q_hat_ref = 1.0
            def get_T_exact(x, t):
                return 1 + t * np.exp(-1000 * (x - 0.5) ** 2)
            def get_T0(x):
                return get_T_exact(x, 0)
            def get_T_a(t):
                return get_T_exact(x_a, t)
            def get_T_b(t):
                return get_T_exact(x_b, t)
            def get_q_hat(x, t):
                return np.exp(-1000 * (x - 0.5) ** 2) * (4000000 * t * (x - (x ** 2) - 0.2495) - 1)
            def get_q_hat_approx(x, t):
                return 0.8 * get_q_hat(x, t)
            def get_k(x):
                return np.ones_like(x) * k_ref
            def get_k_approx(x):
                return get_k(x)
            def get_cV(x):
                return np.ones_like(x) * cV_ref
        elif self.system == "sp1":
            t_end = 3.0
            x_a = 0.0
            x_b = 1.0
            A = 1.0
            k_ref = 2500
            cV_ref = 200
            rho = 200
            q_hat_ref = 0
            def get_T_a(t):
                return 250
            def get_T_b(x):
                return 400
            k_discont_nodes = np.asarray([0.0, 0.12, 0.28, 0.52, 0.68, 0.72, 0.88, 1.0])
            k_prefactors = np.asarray([1.0, 5.0, 1.0, 0.1, 1.0, 10.0, 1.0])
            k_int_values = np.zeros(k_discont_nodes.shape[0])
            exact_solution_available = False
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
                return get_T_a(0) + (get_T_b(0) - get_T_a(0)) * (x + 0.5 * np.sin(2 * np.pi * x))
        else:
            raise Exception("Invalid domain selection.")

        self.exact_solution_available = exact_solution_available
        self.t_end            = t_end
        self.x_a              = x_a
        self.x_b              = x_b
        self.A                = A
        self.k_ref            = k_ref
        self.cV_ref           = cV_ref
        self.rho              = rho
        self.q_hat_ref        = q_hat_ref
        self.get_T_exact      = None
        if self.exact_solution_available:
            self.get_T_exact      = get_T_exact
        self.get_T0           = get_T0
        self.get_T_a          = get_T_a
        self.get_T_b          = get_T_b
        self.get_q_hat        = get_q_hat
        self.get_q_hat_approx = get_q_hat_approx
        self.get_k            = get_k
        self.get_k_approx     = get_k_approx
        self.get_cV           = get_cV

        #---------------------------------------------------------------------------------------------------------------
        # Discretization.

        # Coarse spatial discretization.
        self.N_coarse = 20
        self.dx_coarse = (self.x_b - self.x_a) / self.N_coarse
        self.faces_coarse = np.linspace(self.x_a, self.x_b, num=self.N_coarse + 1, endpoint=True)
        self.nodes_coarse = np.zeros(self.N_coarse + 2)
        self.nodes_coarse[0] = self.x_a
        self.nodes_coarse[1:-1] = self.faces_coarse[:-1] + self.dx_coarse / 2
        self.nodes_coarse[-1] = self.x_b

        # Fine spatial discretization.
        self.N_fine = 20
        self.dx_fine = (self.x_b - self.x_a) / self.N_fine
        self.faces_fine = np.linspace(self.x_a, self.x_b, num=self.N_fine + 1, endpoint=True)
        self.nodes_fine = np.zeros(self.N_fine + 2)
        self.nodes_fine[0] = self.x_a
        self.nodes_fine[1:-1] = self.faces_fine[:-1] + self.dx_fine / 2
        self.nodes_fine[-1] = self.x_b

        # Temporal discretization.
        self.dt_fine = 0.001
        self.dt_coarse = 0.001
        self.Nt_fine = int(self.t_end / self.dt_fine) + 1
        self.Nt_coarse = int(self.t_end / self.dt_coarse) + 1

        ########################################################################################################################
        # Data configuration.

        self.do_simulation_test = False

        # Dataset sizes (unaugmented).
        self.train_examples_ratio = 0.8
        self.val_examples_ratio = 0.1
        self.test_examples_ratio = 0.1
        assert np.around(self.train_examples_ratio + self.val_examples_ratio + self.test_examples_ratio) == 1.0
        self.N_train_examples = int(self.train_examples_ratio * self.Nt_coarse)
        self.N_val_examples = int(self.val_examples_ratio * self.Nt_coarse)
        self.N_test_examples = int(self.test_examples_ratio * self.Nt_coarse)

        # Parameters for shift data augmentation.
        self.N_shift_steps = 5
        self.shift_step_size = 5

        # Test iterations at which temperature profiles are saved.
        self.profile_save_steps = np.asarray([1, int(np.sqrt(self.N_test_examples)), self.N_test_examples]) - 1

        ########################################################################################################################
        # Model configuration.

        self.num_layers = 5

        self.hidden_layer_size = 9

        self.loss_func = 'MSE'

        self.optimizer = 'adam'
        self.learning_rate = 1e-4

        self.act_type = 'lrelu'
        self.act_param = 0.2

        self.use_dropout = False
        self.dropout_prob = 0.1

        ########################################################################################################################
        # Training configuration.

        self.max_train_it = int(1e3)
        self.min_train_it = int(1e2)

        self.print_train_loss_period = int(1e2)  # Number of training iterations per print of training losses.
        self.save_model_period = int(5e10)  # Number of training iterations per model save.
        self.validation_period = int(1e2)  # Number of training iterations per validation.

        self.batch_size_train = 32
        self.batch_size_val = self.N_val_examples
        self.batch_size_test = self.N_test_examples

        self.overfit_limit = 10



########################################################################################################################
# Save configuration
"""
def save_config():
    os.makedirs(run_dir, exist_ok=True)
    config_save_file = open(os.path.join(run_dir, "config_save.txt"), "w")
    config_save_file.write(
        f"Run configuration----------------------------------------------\n"
        f"run_name                  = {run_name}\n"
        f"systems                   = {system}\n"
        f"data_tag                  = {data_tag}\n"
        f"model_name                = {model_name}\n"
        f"model_is_hybrid           = {model_is_hybrid}\n"
        f"augment_training_data     = {augment_training_data}\n"
        f"ensemble_size             = {ensemble_size}\n"
        f"do_train                  = {do_train}\n"
        f"do_test                   = {do_test}\n"
        f"load_model_from_save      = {load_model_from_save}\n"
        f"resume_training_from_save = {resume_training_from_save}\n"
        f"model_load_path           = {model_load_path}\n"
        f"\n"
        f"Environment configuration--------------------------------------\n"
        f"base_dir     = {base_dir}\n"
        f"datasets_dir = {datasets_dir}\n"
        f"results_dir  = {results_dir}\n"
        f"run_dir      = {run_dir}\n"
        f"cp_main_dir  = {cp_main_dir}\n"
        f"cp_load_dir  = {cp_load_dir}\n"
        f"cp_save_dir  = {cp_save_dir}\n"
        f"\n"
        f"Simulation configuration---------------------------------------\n"
        f"x_a                       = {x_a}\n"
        f"x_b                       = {x_b}\n"
        f"N_coarse                  = {N_coarse}\n"
        f"dx_coarse                 = {dx_coarse}\n"
        f"nodes_coarse              = {nodes_coarse}\n"
        f"nodes_coarse              = {nodes_coarse}\n"
        f"faces_coarse              = {faces_coarse}\n"
        f"is_steady                 = {is_steady}\n"
        f"exact_solution_available  = {exact_solution_available}\n"
        f"dt_coarse                 = {dt_coarse}\n"
        f"dt                        = {dt}\n"
        f"\n"
        f"Data configuration---------------------------------------------\n"
        f"data_tag                  = {data_tag}\n"
        f"T_min_train               = {T_min_train}\n"
        f"T_max_train               = {T_max_train}\n"
        f"T_min_test                = {T_min_test}\n"
        f"T_max_test                = {T_max_test}\n"
        f"T_0                       = {get_T0(nodes_coarse)}\n"
        f"N_train_examples          = {N_train_examples}\n"
        f"N_val_examples            = {N_val_examples}\n"
        f"N_test_examples           = {N_test_examples}\n"
        f"do_simulation_test        = {do_simulation_test}\n"
        f"Physics configuration------------------------------------------\n"
        f"area                      = {area}\n"
        f"c_v                       = {c_V}\n"
        f"rho                       = {rho}\n"
        f"q_hat                     = {q_hat}\n"
        f"k_profile_type            = {k_profile_type}\n"
        f"k                         = {get_k(nodes_coarse)}\n"
        f"Architecture configuration-------------------------------------\n"
        f"model name                = {model_name}\n"
        f"input size                = {input_size}\n"
        f"output_size               = {output_size}\n"
        f"hidden_size               = {hidden_size}\n"
        f"num_layers                = {num_layers}\n"
        f"use_dropout               = {use_dropout}\n"
        f"act_type                  = {act_type}\n"
        f"model_type                = {model_type}\n"
        f"Training configuration-----------------------------------------\n"
        f"num_train_it              = {num_train_it}\n"
        f"print_train_loss_period   = {print_train_loss_period}\n"
        f"save_model_period         = {save_model_period}\n"
        f"validation_period         = {validation_period}\n"
        f"train_batch_size          = {train_batch_size}\n"
        f"optimizer_name            = {optimizer_name}\n"
        f"learning_rate             = {learning_rate}\n"
        f"use_shift_data_aug        = {use_shift_data_augmentation}\n"
        f"N_shift_steps             = {N_shift_steps}\n"
        f"use_mirror_data_aug       = {use_mirror_data_augmentation}\n"
        f"normalize_loss_by_target  = {normalize_loss_by_target}\n"
        f"normalize_loss_by_guess   = {normalize_loss_by_guess}\n"
    )
    config_save_file.close()
"""
########################################################################################################################