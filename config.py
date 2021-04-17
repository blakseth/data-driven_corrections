"""
config.py

Written by Sindre Stenen Blakseth, 2021.

Main configuration file.
"""

########################################################################################################################
# Package imports.

import numpy as np
import os
import scipy.special
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
group_name = "2021-04-15_trial_euler_manufactured"
run_names  = [["trial1"]]
systems    = ["ManSol1"]
data_tags  = ["ManSol1"]
model_type = 'hybrid'
model_keys = [7]
assert len(systems) == len(data_tags) == len(run_names[0])
assert len(run_names) == len(model_keys)
N_x = 100


########################################################################################################################
# Create config object.

class Config:
    def __init__(self, use_GPU, group_name, run_name, system, data_tag, model_key, do_train, do_test, N_x, model_type):
        # ---------------------------------------------------------------------------------------------------------------
        # Run configuration.
        self.use_GPU    = use_GPU
        self.device     = torch.device("cuda" if (torch.cuda.is_available() and use_GPU) else "cpu")
        self.group_name = group_name
        self.run_name   = run_name
        self.system     = system
        self.data_tag   = data_tag
        self.model_key  = model_key
        self.model_type = model_type # Can be 'hybrid', 'residual', 'end-to-end' or 'data'

        model_names = [
            'GlobalDense',
            'GlobalCNN',
            'LocalDense',
            'EnsembleLocalDense',
            'EnsembleGlobalCNN',
            'Dense2D',
            'CNN2D',
            'DenseEuler',
            'LocalEuler'
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
            train_alphas, _ = scipy.special.roots_legendre(16)
            train_alphas = train_alphas * 0.5 + 0.5
            val_alphas = np.asarray([0.25, 0.75])
            test_alphas = np.asarray([0.1, 0.5, 0.9, 1.5])
            self.train_alphas = train_alphas
            self.val_alphas = val_alphas
            self.test_alphas = test_alphas
            self.alphas = np.concatenate((self.train_alphas, self.val_alphas, self.test_alphas), axis=0)
        else:
            self.alphas = np.asarray([1.0])

        if self.system == "ManSol1":
            exact_solution_available = True
            x_a = 0.0
            x_b = 1.0
            t_end = 1.0
            CFL = 0.99
            dt = 2.5e-3
            x_split = 0.5
            c_V = 2.5
            gamma = 1.4
            p_ref = 1.5
            def get_T(x, t, alpha):
                return np.ones_like(x)
            def get_p(x, t, alpha):
                if t == 0.0:
                    return get_p0(x, alpha)
                else:
                    return p_ref + alpha*np.tanh((x_split - x)/t)
            def get_rho(x, t, alpha):
                return get_p(x, t, alpha) / (c_V * gamma * get_T(x, t, alpha))
            def get_c(x, t, alpha):
                return np.sqrt((gamma - 1)*gamma*c_V*get_T(x, t, alpha))
            def get_u(x, t, alpha):
                return get_c(x, t, alpha) / (1 + np.exp(-10*alpha*x*(t**(1/4))))
            def get_T0(x, alpha):
                return get_T(x, 0, alpha)
            def get_p0(x, alpha):
                p0 = np.zeros_like(x)
                for x_index, x_value in enumerate(x):
                    if x_value <= x_split:
                        p0[x_index] = p_ref + alpha
                    else:
                        p0[x_index] = p_ref - alpha
                return p0
            def get_rho0(x, alpha):
                return get_rho(x, 0, alpha)
            def get_u0(x, alpha):
                return get_u(x, 0, alpha)
        else:
            raise Exception("Invalid domain selection.")

        """
        if self.system == "SOD":
            exact_solution_available = True
            t_end     = 0.2
            dt        = 5.0e-4
            x_a       = 0.0
            x_b       = 1.0
            x_split   = 0.5
            CFL       = 0.9
            gamma     = 1.4
            c_V       = 2.5
            init_rho1 = 1.0
            init_p1   = 1.0
            init_u1   = 0.0
            init_T1   = init_p1 / (init_rho1*c_V*(gamma - 1))
            init_rho2 = 0.125
            init_p2   = 0.1
            init_u2   = 0.0
            init_T2   = init_p2 / (init_rho2*c_V*(gamma - 1))
        elif self.system == "123":
            exact_solution_available = True
            t_end     = 0.15
            dt        = 2e-3
            x_a       = 0.0
            x_b       = 1.0
            x_split   = 0.5
            CFL       = 0.9
            gamma     = 1.4
            c_V       = 2.5
            init_rho1 = 1.0
            init_p1   = 0.4
            init_u1   = -2.0
            init_T1   = init_p1 / (init_rho1*c_V*(gamma - 1))
            init_rho2 = 1.0
            init_p2   = 0.4
            init_u2   = 2.0
            init_T2   = init_p2 / (init_rho2*c_V*(gamma - 1))
        elif self.system == "LWC":
            exact_solution_available = True
            t_end     = 0.012
            dt        = 1e-4
            x_a       = 0.0
            x_b       = 1.0
            x_split   = 0.5
            CFL       = 0.9
            gamma     = 1.4
            c_V       = 2.5
            init_rho1 = 1.0
            init_p1   = 1000.0
            init_u1   = 0.0
            init_T1   = init_p1 / (init_rho1*c_V*(gamma - 1))
            init_rho2 = 1.0
            init_p2   = 0.01
            init_u2   = 0.0
            init_T2   = init_p2 / (init_rho2*c_V*(gamma - 1))
        elif self.system == "Test4":
            exact_solution_available = True
            t_end     = 0.035
            dt        = 3e-4
            x_a       = 0.0
            x_b       = 1.0
            x_split   = 0.4
            CFL       = 0.9
            gamma     = 1.4
            c_V       = 2.5
            init_rho1 = 6.0
            init_p1   = 461.0
            init_u1   = 19.6
            init_T1   = init_p1 / (init_rho1*c_V*(gamma - 1))
            init_rho2 = 6.0
            init_p2   = 46.1
            init_u2   = -6.2
            init_T2   = init_p2 / (init_rho2*c_V*(gamma - 1))
        elif self.system == "Test5":
            exact_solution_available = True
            t_end     = 0.012
            dt        = 1e-4
            x_a       = 0.0
            x_b       = 1.0
            x_split   = 0.8
            CFL       = 0.9
            gamma     = 1.4
            c_V       = 2.5
            init_rho1 = 1.0
            init_p1   = 1000.0
            init_u1   = -20.0
            init_T1   = init_p1 / (init_rho1*c_V*(gamma - 1))
            init_rho2 = 6.0
            init_p2   = 0.01
            init_u2   = -20.0
            init_T2   = init_p2 / (init_rho2*c_V*(gamma - 1))
        elif self.system == "StatCDisc":
            exact_solution_available = True
            t_end     = 2.0
            dt        = 4e-3
            x_a       = 0.0
            x_b       = 1.0
            x_split   = 0.5
            CFL       = 0.9
            gamma     = 1.4
            c_V       = 2.5
            init_rho1 = 1.4
            init_p1   = 1.0
            init_u1   = 0.0
            init_T1   = init_p1 / (init_rho1*c_V*(gamma - 1))
            init_rho2 = 1.0
            init_p2   = 1.0
            init_u2   = 0.0
            init_T2   = init_p2 / (init_rho2*c_V*(gamma - 1))
        elif self.system == "MovCDisc":
            exact_solution_available = True
            t_end     = 1.3
            dt        = 5e-3
            x_a       = 0.0
            x_b       = 1.0
            x_split   = 0.5
            CFL       = 0.9
            gamma     = 1.4
            c_V       = 2.5
            init_rho1 = 1.4
            init_p1   = 1.0
            init_u1   = 0.1
            init_T1   = init_p1 / (init_rho1*c_V*(gamma - 1))
            init_rho2 = 1.0
            init_p2   = 1.0
            init_u2   = 0.1
            init_T2   = init_p2 / (init_rho2*c_V*(gamma - 1))
        elif self.system == "Thermo":
            exact_solution_available = True
            t_end     = 5e-4
            dt        = 1e-5
            x_a       = 0.0
            x_b       = 1.0
            x_split   = 0.5
            CFL       = 0.9
            gamma     = 1.4
            c_V       = 2.5
            init_rho1 = 1.0595
            init_p1   = 1.0e5
            init_u1   = 0.0
            init_T1   = init_p1 / (init_rho1*c_V*(gamma - 1))
            init_rho2 = 0.1324275
            init_p2   = 1.0e4
            init_u2   = 0.0
            init_T2   = init_p2 / (init_rho2*c_V*(gamma - 1))
        else:
            raise Exception("Invalid domain selection.")
        """

        self.exact_solution_available = exact_solution_available
        self.t_end     = t_end
        self.x_a       = x_a
        self.x_b       = x_b
        self.x_split   = x_split
        self.CFL       = CFL
        self.get_T     = get_T
        self.get_p     = get_p
        self.get_u     = get_u
        self.get_rho   = get_rho
        self.get_T0    = get_T0
        self.get_p0    = get_p0
        self.get_u0    = get_u0
        self.get_rho0  = get_rho0
        self.gamma     = gamma
        self.c_V       = c_V
        self.dt        = dt

        self.dom_vars = set([attr for attr in dir(self) if
                             not callable(getattr(self, attr)) and not attr.startswith("__")]) - other_vars
        other_vars = other_vars.union(self.dom_vars)

        #---------------------------------------------------------------------------------------------------------------
        # Discretization.

        # Coarse spatial discretization.
        self.N_x = N_x          # Excluding boundary nodes.
        self.NJ  = self.N_x + 2 # Including boundary nodes.
        self.dx  = (self.x_b - self.x_a) / self.N_x
        self.x_faces = np.linspace(self.x_a, self.x_b, num=self.N_x + 1, endpoint=True)
        self.x_nodes = np.zeros(self.N_x + 2)
        self.x_nodes[0]    = self.x_a
        self.x_nodes[1:-1] = self.x_faces[:-1] + self.dx / 2
        self.x_nodes[-1]   = self.x_b

        # Temporal discretization.
        self.N_t = int(self.t_end / self.dt) + 1

        self.disc_vars = set([attr for attr in dir(self) if
                             not callable(getattr(self, attr)) and not attr.startswith("__")]) - other_vars
        other_vars = other_vars.union(self.disc_vars)

        #---------------------------------------------------------------------------------------------------------------
        # Data configuration.

        self.do_simulation_test = False

        # Dataset sizes (unaugmented).
        self.N_train_examples = int(self.train_alphas.shape[0] * (self.N_t-2))
        self.N_val_examples = int(self.val_alphas.shape[0] * (self.N_t-2))
        self.N_test_examples = int(self.test_alphas.shape[0] * (self.N_t-2))
        self.N_train_alphas = self.train_alphas.shape[0]
        self.N_val_alphas   = self.val_alphas.shape[0]
        self.N_test_alphas  = self.test_alphas.shape[0]
        print("N_train_alphas", self.N_train_alphas)
        print("N_val_alphas", self.N_val_alphas)
        print("N_test_alphas", self.N_test_alphas)

        # Parameters for shift data augmentation.
        self.N_shift_steps = 5
        self.shift_step_size = 5

        # Test iterations at which temperature profiles are saved.
        if self.parametrized_system:
            base = self.N_t - 2
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

        self.loss_func = 'L1'

        self.optimizer = 'adam'
        self.learning_rate = 1e-4

        self.act_type = 'lrelu'
        self.act_param = 0.01

        self.use_dropout = True
        self.dropout_prob = 0.2

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
            self.num_networks = self.N_x
            self.model_specific_params = [self.num_networks, self.num_layers, self.hidden_layer_size]
            def get_model_specific_params():
                return [self.num_networks, self.num_layers, self.hidden_layer_size]
        elif self.model_name == 'EnsembleGlobalCNN':
            self.num_conv_layers = 5
            self.kernel_size = 3
            self.num_channels = 20
            self.num_fc_layers = 1
            self.num_networks = self.N_x
            self.model_specific_params = [self.num_networks, self.num_conv_layers, self.kernel_size, self.num_channels, self.num_fc_layers]
            def get_model_specific_params():
                return [self.num_networks, self.num_conv_layers, self.kernel_size, self.num_channels, self.num_fc_layers]
        elif self.model_name == 'Dense2D':
            self.num_layers = 6
            self.hidden_layer_size = 400
            # [No. fc layers, No. nodes in each hidden layer]
            self.model_specific_params = [self.num_layers, self.hidden_layer_size]
            def get_model_specific_params():
                return [self.num_layers, self.hidden_layer_size]
        elif self.model_name == 'CNN2D':
            self.num_conv_layers = 5
            self.kernel_size = 3
            self.num_channels = 50
            self.model_specific_params = [self.num_conv_layers, self.kernel_size, self.num_channels]
            def get_model_specific_params():
                return [self.num_conv_layers, self.kernel_size, self.num_channels]
        elif self.model_name == 'DenseEuler':
            self.num_layers = 5
            self.hidden_layer_size = 100
            # [No. fc layers, No. nodes in each hidden layer]
            self.model_specific_params = [self.num_layers, self.hidden_layer_size]
            def get_model_specific_params():
                return [self.num_layers, self.hidden_layer_size]
        elif self.model_name == 'LocalEuler':
            self.num_layers = 5
            self.hidden_layer_size = 25
            # [No. fc layers, No. nodes in each hidden layer]
            self.model_specific_params = [self.num_layers, self.hidden_layer_size]
            def get_model_specific_params():
                return [self.num_layers, self.hidden_layer_size]
        else:
            raise Exception("Invalid model selection.")

        self.get_model_specific_params = get_model_specific_params

        self.mod_vars = set([attr for attr in dir(self) if
                             not callable(getattr(self, attr)) and not attr.startswith("__")]) - other_vars
        other_vars = other_vars.union(self.mod_vars)

        #---------------------------------------------------------------------------------------------------------------
        # Training configuration.

        self.max_train_it = int(1e6)
        self.min_train_it = int(1e4)

        self.save_train_loss_period = int(1e2)  # Number of training iterations per save of training losses.
        self.print_train_loss_period = int(4e2) # Number of training iterations per save of training losses.
        self.save_model_period = int(5e10)  # Number of training iterations per model save.
        self.validation_period = int(1e2)  # Number of training iterations per validation.

        self.batch_size_train = 8
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
    cfg = Config(
        use_GPU=use_GPU,
        group_name=group_name,
        run_name=run_names[0][0],
        system=systems[0],
        data_tag=data_tags[0],
        model_key=model_keys[0],
        do_train=False,
        do_test=False,
        N_x=N_x,
        model_type=model_type,
    )
    print(cfg.alphas)
    print(cfg.N_train_examples)

########################################################################################################################

if __name__ == "__main__":
    main()

########################################################################################################################