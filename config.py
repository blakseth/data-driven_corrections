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

use_GPU    = True
group_name = "2021-04-30_HAM_missing_conductivity_errors_fixed"
run_names  = [["GlobalDense_HAM_k2", "GlobalDense_HAM_k5", "GlobalDense_HAM_k6"]]
systems    = ["k2", "k5", "k6"]
data_tags  = ["k2_50cells", "k5_50cells", "k6_50cells"]
model_keys = [0]
assert len(systems) == len(data_tags) == len(run_names[0])
assert len(run_names) == len(model_keys)
synthesize_modelling_error = True


########################################################################################################################
# Create config object.

class Config:
    def __init__(self, use_GPU, group_name, run_name, system, data_tag, model_key, do_train, do_test):
        # ---------------------------------------------------------------------------------------------------------------
        # Run configuration.
        self.use_GPU    = use_GPU
        self.device     = torch.device("cuda" if (torch.cuda.is_available() and use_GPU) else "cpu")
        self.group_name = group_name
        self.run_name   = run_name
        self.system     = system
        self.data_tag   = data_tag
        self.model_key  = model_key
        self.model_type = 'data' # Can be 'hybrid', 'residual', 'end-to-end' or 'data'

        self.synthesize_mod_error = synthesize_modelling_error

        model_names = [
            'GlobalDense',
            'GlobalCNN',
            'LocalDense',
            'EnsembleLocalDense',
            'EnsembleGlobalCNN'
        ]
        self.model_name = model_names[model_key]

        self.augment_training_data = False

        self.parametrized_system = True

        self.ensemble_size = 1

        self.do_train = do_train
        self.do_test = do_test

        self.load_model_from_save = False
        self.resume_training_from_save = False
        self.model_load_path = None  # os.path.join(cp_load_dir, 'G_150000.pth')

        self.run_vars = set([attr for attr in dir(self) if not callable(getattr(self, attr)) and not attr.startswith("__")])
        other_vars = self.run_vars

        #---------------------------------------------------------------------------------------------------------------
        # Environment configuration.

        self.base_dir     = '/home/sindre/msc_thesis/data-driven_corrections'
        #self.base_dir     = '/lustre1/work/sindresb/msc_thesis/data-driven_corrections/'
        #self.base_dir      = '/content/gdrive/My Drive/msc_thesis/data-driven_corrections'
        self.datasets_dir = os.path.join(self.base_dir, 'datasets')
        self.results_dir  = os.path.join(self.base_dir, 'results')
        self.group_dir    = os.path.join(self.results_dir, group_name)
        self.run_dir      = os.path.join(self.group_dir, run_name)
        self.cp_main_dir  = os.path.join(self.base_dir, 'checkpoints')
        self.cp_load_dir  = os.path.join(self.cp_main_dir, '')
        self.cp_save_dir  = os.path.join(self.cp_main_dir, run_name)

        self.env_vars = set([attr for attr in dir(self) if not callable(getattr(self, attr)) and not attr.startswith("__")]) - other_vars
        other_vars = other_vars.union(self.env_vars)

        #---------------------------------------------------------------------------------------------------------------
        # Domain configuration.

        if self.parametrized_system:
            lin_alphas   = np.linspace(0.1, 2.0, 20, endpoint=True)
            permutation  = np.random.RandomState(seed=42).permutation(lin_alphas.shape[0])
            lin_alphas   = lin_alphas[permutation]
            extra_alphas = np.asarray([-0.5, 2.5])
            self.alphas  = np.concatenate((lin_alphas, extra_alphas), axis=0)
        else:
            self.alphas = np.asarray([1.0])

        if self.system == "k1":
            exact_solution_available = True
            t_end     = 5.0
            x_a       = 0.0
            x_b       = 1.0
            A         = 1.0
            rho       = 1.0
            k_ref     = 1.0
            cV_ref    = 1.0
            q_hat_ref = 1.0
            def get_T_exact(x, t, alpha):
                return t + alpha*x
            def get_T0(x, alpha):
                return get_T_exact(x, 0, alpha)
            def get_T_a(t, alpha):
                return get_T_exact(x_a, t, alpha)
            def get_T_b(t, alpha):
                return get_T_exact(x_b, t, alpha)
            def get_q_hat(x, t, alpha):
                return (1 - alpha) * np.ones_like(x)
            def get_k(x, t, alpha):
                return 1 + x
            def get_k_approx(x):
                return np.ones_like(x) * k_ref
            def get_cV(x):
                return np.ones_like(x) * cV_ref
        elif self.system == "k2":
            exact_solution_available = True
            t_end     = 5.0
            x_a       = 0.0
            x_b       = 1.0
            A         = 1.0
            rho       = 1.0
            k_ref     = 1.0
            cV_ref    = 1.0
            q_hat_ref = 1.0
            def get_T_exact(x, t, alpha):
                return 5 - alpha*x/(1 + t)
            def get_T0(x, alpha):
                return get_T_exact(x, 0, alpha)
            def get_T_a(t, alpha):
                return get_T_exact(x_a, t, alpha)
            def get_T_b(t, alpha):
                return get_T_exact(x_b, t, alpha)
            def get_q_hat(x, t, alpha):
                return alpha*(x - alpha)/((1 + t)**2)
            def get_k(x, t, alpha):
                return get_T_exact(x, t, alpha)
            def get_k_approx(x):
                return np.ones_like(x) * k_ref
            def get_cV(x):
                return np.ones_like(x) * cV_ref
        elif self.system == "k3":
            exact_solution_available = True
            t_end     = 5.0
            x_a       = 0.0
            x_b       = 1.0
            A         = 1.0
            rho       = 1.0
            k_ref     = 1.0
            cV_ref    = 1.0
            q_hat_ref = 1.0
            def get_T_exact(x, t, alpha):
                if type(x) == np.ndarray:
                    T = np.zeros_like(x)
                    for x_index, x_val in enumerate(x):
                        if x_val <= 0.5:
                            T[x_index] = alpha + 2*x_val
                        else:
                            T[x_index] = alpha + 0.75 + 0.5*x_val
                else:
                    if x <= 0.5:
                        T = alpha + 2*x
                    else:
                        T = alpha + 0.75 + 0.5*x
                return np.exp(-t) * T
            def get_T0(x, alpha):
                return get_T_exact(x, 0, alpha)
            def get_T_a(t, alpha):
                return get_T_exact(x_a, t, alpha)
            def get_T_b(t, alpha):
                return get_T_exact(x_b, t, alpha)
            def get_q_hat(x, t, alpha):
                return - get_T_exact(x, t, alpha)
            def get_k(x, t, alpha):
                k = np.ones_like(x)
                for i in range(k.shape[0]):
                    if x[i] <= 0.5:
                        k[i] /= 2.0
                    else:
                        k[i] *= 2.0
                return k
            def get_k_approx(x):
                return np.ones_like(x) * k_ref
            def get_cV(x):
                return np.ones_like(x) * cV_ref
        elif self.system == "k4":
            exact_solution_available = True
            t_end = 5.0
            x_a = 1.0
            x_b = 2.0
            A = 1.0
            rho = 1.0
            k_ref = 1.0
            cV_ref = 1.0
            q_hat_ref = 1.0
            def get_T_exact(x, t, alpha):
                return alpha*(t + 1)*np.cos(2*np.pi*x)
            def get_T0(x, alpha):
                return get_T_exact(x, 0, alpha)
            def get_T_a(t, alpha):
                return get_T_exact(x_a, t, alpha)
            def get_T_b(t, alpha):
                return get_T_exact(x_b, t, alpha)
            def get_q_hat(x, t, alpha):
                return alpha*np.cos(2*np.pi*x)*(1 + 8*(np.pi**2)*(t + 1)*np.sin(2*np.pi*x))
            def get_k(x, t, alpha):
                return np.sin(2*np.pi*x)
            def get_k_approx(x):
                return np.ones_like(x) * k_ref
            def get_cV(x):
                return np.ones_like(x) * cV_ref
        elif self.system == "k5":
            exact_solution_available = True
            t_end     = 5.0
            x_a       = 0.0
            x_b       = 1.0
            A         = 1.0
            rho       = 1.0
            k_ref     = 1.0
            cV_ref    = 1.0
            q_hat_ref = 1.0
            def get_T_exact(x, t, alpha):
                return alpha*t*(x+1)
            def get_T0(x, alpha):
                return get_T_exact(x, 0, alpha)
            def get_T_a(t, alpha):
                return get_T_exact(x_a, t, alpha)
            def get_T_b(t, alpha):
                return get_T_exact(x_b, t, alpha)
            def get_q_hat(x, t, alpha):
                return alpha*(x - alpha*(t**2))
            def get_k(x, t, alpha):
                return get_T_exact(x, t, alpha)
            def get_k_approx(x):
                return np.ones_like(x) * k_ref
            def get_cV(x):
                return np.ones_like(x) * cV_ref
        elif self.system == "k6":
            exact_solution_available = True
            t_end     = 5.0
            x_a       = 0.0
            x_b       = 1.0
            A         = 1.0
            rho       = 1.0
            k_ref     = 1.0
            cV_ref    = 1.0
            q_hat_ref = 1.0
            def get_T_exact(x, t, alpha):
                return (x + alpha)/(t + 1)
            def get_T0(x, alpha):
                return get_T_exact(x, 0, alpha)
            def get_T_a(t, alpha):
                return get_T_exact(x_a, t, alpha)
            def get_T_b(t, alpha):
                return get_T_exact(x_b, t, alpha)
            def get_q_hat(x, t, alpha):
                return -(x + alpha + 1)/((t + 1)**2)
            def get_k(x, t, alpha):
                return get_T_exact(x, t, alpha)
            def get_k_approx(x):
                return np.ones_like(x) * k_ref
            def get_cV(x):
                return np.ones_like(x) * cV_ref
        elif self.system == "k7":
            exact_solution_available = True
            t_end     = 5.0
            x_a       = 0.0
            x_b       = 1.0
            A         = 1.0
            rho       = 1.0
            k_ref     = 1.0
            cV_ref    = 1.0
            q_hat_ref = 1.0
            def get_T_exact(x, t, alpha):
                return 4*(x**3) - 4*(x**2) + alpha*(t + 1)
            def get_T0(x, alpha):
                return get_T_exact(x, 0, alpha)
            def get_T_a(t, alpha):
                return get_T_exact(x_a, t, alpha)
            def get_T_b(t, alpha):
                return get_T_exact(x_b, t, alpha)
            def get_q_hat(x, t, alpha):
                return alpha - 36*(x**2) - 8*x + 8
            def get_k(x, t, alpha):
                return 1 + x
            def get_k_approx(x):
                return np.ones_like(x) * k_ref
            def get_cV(x):
                return np.ones_like(x) * cV_ref
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
        self.get_k            = get_k
        self.get_k_approx     = get_k_approx
        self.get_cV           = get_cV

        self.dom_vars = set([attr for attr in dir(self) if
                             not callable(getattr(self, attr)) and not attr.startswith("__")]) - other_vars
        other_vars = other_vars.union(self.dom_vars)

        #---------------------------------------------------------------------------------------------------------------
        # Discretization.

        # Coarse spatial discretization.
        self.N_coarse = 50
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

        self.disc_vars = set([attr for attr in dir(self) if
                             not callable(getattr(self, attr)) and not attr.startswith("__")]) - other_vars
        other_vars = other_vars.union(self.disc_vars)

        #---------------------------------------------------------------------------------------------------------------
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
        if self.parametrized_system:
            self.N_train_examples *= lin_alphas.shape[0]
            self.N_val_examples *= lin_alphas.shape[0]
            self.N_test_examples *= lin_alphas.shape[0]
            self.N_test_examples += self.Nt_coarse * extra_alphas.shape[0]
        self.N_train_alphas = int(self.train_examples_ratio * lin_alphas.shape[0])
        self.N_val_alphas   = int(self.val_examples_ratio   * lin_alphas.shape[0])
        self.N_test_alphas  = int(self.test_examples_ratio  * lin_alphas.shape[0]) + extra_alphas.shape[0]
        print("N_train_alphas", self.N_train_alphas)
        print("N_val_alphas", self.N_val_alphas)
        print("N_test_alphas", self.N_test_alphas)

        # Parameters for shift data augmentation.
        self.N_shift_steps = 5
        self.shift_step_size = 5

        # Test iterations at which temperature profiles are saved.
        if self.parametrized_system:
            base = self.Nt_coarse - 1
        else:
            base = self.N_test_examples
        self.profile_save_steps = np.asarray([
            1,
            int(base**(1/5)),
            int(base**(1/3)),
            int(np.sqrt(base)),
            int(base**(5/7)),
            int(base**(6/7)),
            base
        ]) - 1

        self.data_vars = set([attr for attr in dir(self) if
                             not callable(getattr(self, attr)) and not attr.startswith("__")]) - other_vars
        other_vars = other_vars.union(self.data_vars)

        #---------------------------------------------------------------------------------------------------------------
        # Model configuration.

        self.loss_func = 'MSE'

        self.optimizer = 'adam'
        self.learning_rate = 1e-5

        self.act_type = 'lrelu'
        self.act_param = 0.01

        self.use_dropout = True
        self.dropout_prob = 0.0

        self.model_specific_params = []
        if self.model_name == 'GlobalDense':
            self.num_layers = 6
            self.hidden_layer_size = 100
            # [No. fc layers, No. nodes in each hidden layer]
            self.model_specific_params = [self.num_layers, self.hidden_layer_size]
            def get_model_specific_params():
                return [self.num_layers, self.hidden_layer_size]
        elif self.model_name == 'LocalDense':
            self.num_layers = 5
            self.hidden_layer_size = 9
            # [No. layers in each network, No. nodes in each hidden layer of each network]
            self.model_specific_params = [self.num_layers, self.hidden_layer_size]
            def get_model_specific_params():
                return [self.num_layers, self.hidden_layer_size]
        elif self.model_name == 'GlobalCNN':
            self.num_conv_layers = 5
            self.kernel_size = 3
            self.num_channels = 80
            self.num_fc_layers = 1
            self.model_specific_params = [self.num_conv_layers, self.kernel_size, self.num_channels, self.num_fc_layers]
            def get_model_specific_params():
                return [self.num_conv_layers, self.kernel_size, self.num_channels, self.num_fc_layers]
        elif self.model_name == 'EnsembleLocalDense':
            self.num_layers = 5
            self.hidden_layer_size = 9
            self.num_networks = self.N_coarse
            self.model_specific_params = [self.num_networks, self.num_layers, self.hidden_layer_size]
            def get_model_specific_params():
                return [self.num_networks, self.num_layers, self.hidden_layer_size]
        elif self.model_name == 'EnsembleGlobalCNN':
            self.num_conv_layers = 5
            self.kernel_size = 3
            self.num_channels = 20
            self.num_fc_layers = 1
            self.num_networks = self.N_coarse
            self.model_specific_params = [self.num_networks, self.num_conv_layers, self.kernel_size, self.num_channels, self.num_fc_layers]
            def get_model_specific_params():
                return [self.num_networks, self.num_conv_layers, self.kernel_size, self.num_channels, self.num_fc_layers]
        else:
            raise Exception("Invalid model selection.")

        self.get_model_specific_params = get_model_specific_params

        self.mod_vars = set([attr for attr in dir(self) if
                             not callable(getattr(self, attr)) and not attr.startswith("__")]) - other_vars
        other_vars = other_vars.union(self.mod_vars)

        #---------------------------------------------------------------------------------------------------------------
        # Training configuration.

        self.max_train_it = int(1e6)
        self.min_train_it = int(2e4)

        self.save_train_loss_period = int(1e2)  # Number of training iterations per save of training losses.
        self.print_train_loss_period = int(4e2) # Number of training iterations per save of training losses.
        self.save_model_period = int(5e10)  # Number of training iterations per model save.
        self.validation_period = int(1e2)  # Number of training iterations per validation.

        self.batch_size_train = 32
        self.batch_size_val = self.N_val_examples
        self.batch_size_test = self.N_test_examples

        self.overfit_limit = 10
        if self.learning_rate == 1e-5:
            self.overfit_limit = 20

        self.train_vars = set([attr for attr in dir(self) if
                             not callable(getattr(self, attr)) and not attr.startswith("__")]) - other_vars


########################################################################################################################
# Save configuration.

def save_config(cfg):
    names_of_var_lists = {
        "run_vars",
        "env_vars",
        "dom_vars",
        "disc_vars",
        "data_vars",
        "mod_vars",
        "train_vars"
    }

    variable_names = [attr for attr in dir(cfg) if not callable(getattr(cfg, attr)) and not attr.startswith("__")]
    length_of_longest_var_name = len(max(variable_names, key=len))
    format_str = "{:<" + str(length_of_longest_var_name) + "}"

    cfg_string = "Configuration of run " + cfg.run_name + "\n"

    cfg_string += "\nRun configuration--------------------------------------\n"
    run_vars = sorted(list(cfg.run_vars - names_of_var_lists))
    for var in run_vars:
        cfg_string += format_str.format(var) + " = " + str(getattr(cfg, var)) + "\n"

    cfg_string += "\nEnvironment configuration------------------------------\n"
    env_vars = sorted(list(cfg.env_vars - names_of_var_lists))
    for var in env_vars:
        cfg_string += format_str.format(var) + " = " + str(getattr(cfg, var)) + "\n"

    cfg_string += "\nDomain configuration-----------------------------------\n"
    dom_vars = sorted(list(cfg.dom_vars - names_of_var_lists))
    for var in dom_vars:
        cfg_string += format_str.format(var) + " = " + str(getattr(cfg, var)) + "\n"

    cfg_string += "\nDiscretization configuration---------------------------\n"
    disc_vars = sorted(list(cfg.disc_vars - names_of_var_lists))
    for var in disc_vars:
        cfg_string += format_str.format(var) + " = " + str(getattr(cfg, var)) + "\n"

    cfg_string += "\nData configuration-------------------------------------\n"
    data_vars = sorted(list(cfg.data_vars - names_of_var_lists))
    for var in data_vars:
        cfg_string += format_str.format(var) + " = " + str(getattr(cfg, var)) + "\n"

    cfg_string += "\nModel configuration------------------------------------\n"
    mod_vars = sorted(list(cfg.mod_vars - names_of_var_lists))
    for var in mod_vars:
        cfg_string += format_str.format(var) + " = " + str(getattr(cfg, var)) + "\n"

    cfg_string += "\nTraining configuration---------------------------------\n"
    train_vars = sorted(list(cfg.train_vars - names_of_var_lists))
    for var in train_vars:
        cfg_string += format_str.format(var) + " = " + str(getattr(cfg, var)) + "\n"

    os.makedirs(cfg.run_dir, exist_ok=True)
    with open(os.path.join(cfg.run_dir, "config_save.txt"), "w") as f:
        f.write(cfg_string)

########################################################################################################################

def main():
    lin_alphas = np.linspace(0.1, 2.0, 20, endpoint=True)
    permutation = np.random.RandomState(seed=42).permutation(lin_alphas.shape[0])
    lin_alphas = lin_alphas[permutation]
    extra_alphas = np.asarray([-0.5, 2.5])
    alphas = np.concatenate((lin_alphas, extra_alphas), axis=0)
    print(alphas)

########################################################################################################################

if __name__ == "__main__":
    main()

########################################################################################################################