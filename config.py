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
group_name = "2021-04-25_2D_experiment_redo_s4"
run_names  = [["2D_GlobalDense_s4_DDM"]]
systems    = ["4"]
data_tags  = ["2D_s4"]
model_type = 'data'
model_keys = [5]
assert len(systems) == len(data_tags) == len(run_names[0])
assert len(run_names) == len(model_keys)
N_x = 20


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
            'CNN2D'
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

        #self.base_dir     = '/home/sindre/msc_thesis/data-driven_corrections'
        #self.base_dir     = '/lustre1/work/sindresb/msc_thesis/data-driven_corrections/'
        self.base_dir      = '/content/gdrive/My Drive/msc_thesis/data-driven_corrections'
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

        if self.system == "1":
            exact_solution_available = True
            t_end     = 5.0
            x_a       = 0.0
            x_b       = 1.0
            y_c       = 0.0
            y_d       = 1.0
            A         = 1.0
            rho       = 1.0
            k_ref     = 1.0
            cV_ref    = 1.0
            q_hat_ref = 1.0
            def get_T_exact(x, y, t, alpha):
                def local_T(x, y, t, alpha):
                    return t + 0.5 * alpha * ((x ** 2) + (y ** 2)) + 1.0*x
                if type(x) is np.ndarray and type(y) is np.ndarray:
                    T = np.zeros((x.shape[0], y.shape[0]))
                    for i, y_ in enumerate(y):
                        for j, x_ in enumerate(x):
                            T[j, i] = local_T(x_, y_, t, alpha)
                    return T
                elif type(x) is np.ndarray:
                    T = np.zeros(x.shape[0])
                    for j, x_ in enumerate(x):
                        T[j] = local_T(x_, y, t, alpha)
                    return T
                elif type(y) is np.ndarray:
                    T = np.zeros(y.shape[0])
                    for i, y_ in enumerate(y):
                        T[i] = local_T(x, y_, t, alpha)
                    return T
                else:
                    return local_T(x, y, t, alpha)
            def get_T0(x, y, alpha):
                return get_T_exact(x, y, 0, alpha)
            def get_T_a(y, t, alpha):
                return get_T_exact(x_a, y, t, alpha)
            def get_T_b(y, t, alpha):
                return get_T_exact(x_b, y, t, alpha)
            def get_T_c(x, t, alpha):
                return get_T_exact(x, y_c, t, alpha)
            def get_T_d(x, t, alpha):
                return get_T_exact(x, y_d, t, alpha)
            def get_q_hat(x, y, t, alpha):
                return (1 - 2*alpha) * np.ones((x.shape[0], y.shape[0]))
            def get_q_hat_approx(x, y, t, alpha):
                return np.zeros((x.shape[0], y.shape[0]))
            def get_k(x, y):
                return np.ones((x.shape[0], y.shape[0])) * k_ref
            def get_k_approx(x, y):
                return get_k(x, y)
            def get_cV(x, y):
                return np.ones((x.shape[0], y.shape[0])) * cV_ref
        elif self.system == "2":
            exact_solution_available = True
            t_end     = 5.0
            x_a       = 0.0
            x_b       = 1.0
            y_c       = 0.0
            y_d       = 1.0
            A         = 1.0
            rho       = 1.0
            k_ref     = 1.0
            cV_ref    = 1.0
            q_hat_ref = 1.0
            def get_T_exact(x, y, t, alpha):
                def local_T(x, y, t, alpha):
                    return np.sqrt(t + alpha + 1) + x*(x - 1)*y*(y -1)
                if type(x) is np.ndarray and type(y) is np.ndarray:
                    T = np.zeros((x.shape[0], y.shape[0]))
                    for i, y_ in enumerate(y):
                        for j, x_ in enumerate(x):
                            T[j, i] = local_T(x_, y_, t, alpha)
                    return T
                elif type(x) is np.ndarray:
                    T = np.zeros(x.shape[0])
                    for j, x_ in enumerate(x):
                        T[j] = local_T(x_, y, t, alpha)
                    return T
                elif type(y) is np.ndarray:
                    T = np.zeros(y.shape[0])
                    for i, y_ in enumerate(y):
                        T[i] = local_T(x, y_, t, alpha)
                    return T
                else:
                    return local_T(x, y, t, alpha)
            def get_T0(x, y, alpha):
                return get_T_exact(x, y, 0, alpha)
            def get_T_a(y, t, alpha):
                return get_T_exact(x_a, y, t, alpha)
            def get_T_b(y, t, alpha):
                return get_T_exact(x_b, y, t, alpha)
            def get_T_c(x, t, alpha):
                return get_T_exact(x, y_c, t, alpha)
            def get_T_d(x, t, alpha):
                return get_T_exact(x, y_d, t, alpha)
            def get_q_hat(x, y, t, alpha):
                return -2*(1/(4*np.sqrt(t + alpha + 1)) + x**2 + x + y**2 + y)
            def get_q_hat_approx(x, y, t, alpha):
                return np.zeros((x.shape[0], y.shape[0]))
            def get_k(x, y):
                return np.ones((x.shape[0], y.shape[0])) * k_ref
            def get_k_approx(x, y):
                return get_k(x, y)
            def get_cV(x, y):
                return np.ones((x.shape[0], y.shape[0])) * cV_ref
        elif self.system == "3":
            exact_solution_available = True
            t_end     = 5.0
            x_a       = 0.0
            x_b       = 1.0
            y_c       = 0.0
            y_d       = 1.0
            A         = 1.0
            rho       = 1.0
            k_ref     = 1.0
            cV_ref    = 1.0
            q_hat_ref = 1.0
            def get_T_exact(x, y, t, alpha):
                def local_T(x, y, t, alpha):
                    return 2 + alpha*np.tanh(x*y/(t + 0.1))
                if type(x) is np.ndarray and type(y) is np.ndarray:
                    T = np.zeros((x.shape[0], y.shape[0]))
                    for i, y_ in enumerate(y):
                        for j, x_ in enumerate(x):
                            T[j, i] = local_T(x_, y_, t, alpha)
                    return T
                elif type(x) is np.ndarray:
                    T = np.zeros(x.shape[0])
                    for j, x_ in enumerate(x):
                        T[j] = local_T(x_, y, t, alpha)
                    return T
                elif type(y) is np.ndarray:
                    T = np.zeros(y.shape[0])
                    for i, y_ in enumerate(y):
                        T[i] = local_T(x, y_, t, alpha)
                    return T
                else:
                    return local_T(x, y, t, alpha)
            def get_T0(x, y, alpha):
                return get_T_exact(x, y, 0, alpha)
            def get_T_a(y, t, alpha):
                return get_T_exact(x_a, y, t, alpha)
            def get_T_b(y, t, alpha):
                return get_T_exact(x_b, y, t, alpha)
            def get_T_c(x, t, alpha):
                return get_T_exact(x, y_c, t, alpha)
            def get_T_d(x, t, alpha):
                return get_T_exact(x, y_d, t, alpha)
            def get_q_hat(x, y, t, alpha):
                return alpha*(2*(x**2 + y**2)*np.tanh(x*y/(t + 0.1)) - x*y)/(((t + 0.1)*np.cosh(x*y/(t + 0.1)))**2)
            def get_q_hat_approx(x, y, t, alpha):
                return np.zeros((x.shape[0], y.shape[0]))
            def get_k(x, y):
                return np.ones((x.shape[0], y.shape[0])) * k_ref
            def get_k_approx(x, y):
                return get_k(x, y)
            def get_cV(x, y):
                return np.ones((x.shape[0], y.shape[0])) * cV_ref
        elif self.system == "4":
            exact_solution_available = True
            t_end     = 5.0
            x_a       = 0.0
            x_b       = 1.0
            y_c       = 0.0
            y_d       = 1.0
            A         = 1.0
            rho       = 1.0
            k_ref     = 1.0
            cV_ref    = 1.0
            q_hat_ref = 1.0
            def get_T_exact(x, y, t, alpha):
                def local_T(x, y, t, alpha):
                    return 1 + np.sin(2*np.pi*t + alpha)*np.cos(2*np.pi*x)*np.cos(2*np.pi*y)
                if type(x) is np.ndarray and type(y) is np.ndarray:
                    T = np.zeros((x.shape[0], y.shape[0]))
                    for i, y_ in enumerate(y):
                        for j, x_ in enumerate(x):
                            T[j, i] = local_T(x_, y_, t, alpha)
                    return T
                elif type(x) is np.ndarray:
                    T = np.zeros(x.shape[0])
                    for j, x_ in enumerate(x):
                        T[j] = local_T(x_, y, t, alpha)
                    return T
                elif type(y) is np.ndarray:
                    T = np.zeros(y.shape[0])
                    for i, y_ in enumerate(y):
                        T[i] = local_T(x, y_, t, alpha)
                    return T
                else:
                    return local_T(x, y, t, alpha)
            def get_T0(x, y, alpha):
                return get_T_exact(x, y, 0, alpha)
            def get_T_a(y, t, alpha):
                return get_T_exact(x_a, y, t, alpha)
            def get_T_b(y, t, alpha):
                return get_T_exact(x_b, y, t, alpha)
            def get_T_c(x, t, alpha):
                return get_T_exact(x, y_c, t, alpha)
            def get_T_d(x, t, alpha):
                return get_T_exact(x, y_d, t, alpha)
            def get_q_hat(x, y, t, alpha):
                return 2*np.pi*np.cos(2*np.pi*x)*np.cos(2*np.pi*y)*(np.cos(2*np.pi*t + alpha) + 8*np.pi*np.sin(2*np.pi*t + alpha))
            def get_q_hat_approx(x, y, t, alpha):
                return np.zeros((x.shape[0], y.shape[0]))
            def get_k(x, y):
                return np.ones((x.shape[0], y.shape[0])) * k_ref
            def get_k_approx(x, y):
                return get_k(x, y)
            def get_cV(x, y):
                return np.ones((x.shape[0], y.shape[0])) * cV_ref
        elif self.system == "5":
            exact_solution_available = True
            t_end     = 5.0
            x_a       = 0.0
            x_b       = 1.0
            y_c       = 0.0
            y_d       = 1.0
            A         = 1.0
            rho       = 1.0
            k_ref     = 1.0
            cV_ref    = 1.0
            q_hat_ref = 1.0
            def get_T_exact(x, y, t, alpha):
                def local_T(x, y, t, alpha):
                    return np.exp(-100*alpha*((x-0.5)**2)) * np.exp(-10*((y-0.5)**2)) * np.exp(-0.1*t)
                if type(x) is np.ndarray and type(y) is np.ndarray:
                    T = np.zeros((x.shape[0], y.shape[0]))
                    for i, y_ in enumerate(y):
                        for j, x_ in enumerate(x):
                            T[j, i] = local_T(x_, y_, t, alpha)
                    return T
                elif type(x) is np.ndarray:
                    T = np.zeros(x.shape[0])
                    for j, x_ in enumerate(x):
                        T[j] = local_T(x_, y, t, alpha)
                    return T
                elif type(y) is np.ndarray:
                    T = np.zeros(y.shape[0])
                    for i, y_ in enumerate(y):
                        T[i] = local_T(x, y_, t, alpha)
                    return T
                else:
                    return local_T(x, y, t, alpha)
            def get_T0(x, y, alpha):
                return get_T_exact(x, y, 0, alpha)
            def get_T_a(y, t, alpha):
                return get_T_exact(x_a, y, t, alpha)
            def get_T_b(y, t, alpha):
                return get_T_exact(x_b, y, t, alpha)
            def get_T_c(x, t, alpha):
                return get_T_exact(x, y_c, t, alpha)
            def get_T_d(x, t, alpha):
                return get_T_exact(x, y_d, t, alpha)
            def get_q_hat(x, y, t, alpha):
                return (-0.1 - (200*alpha*(x-0.5))**2 + 200*alpha - (20*(y-0.5))**2 + 20) * get_T_exact(x, y, t, alpha)
            def get_q_hat_approx(x, y, t, alpha):
                return np.zeros((x.shape[0], y.shape[0]))
            def get_k(x, y):
                return np.ones((x.shape[0], y.shape[0])) * k_ref
            def get_k_approx(x, y):
                return get_k(x, y)
            def get_cV(x, y):
                return np.ones((x.shape[0], y.shape[0])) * cV_ref
        else:
            raise Exception("Invalid domain selection.")

        self.exact_solution_available = exact_solution_available
        self.t_end            = t_end
        self.x_a              = x_a
        self.x_b              = x_b
        self.y_c              = y_c
        self.y_d              = y_d
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
        self.get_T_c          = get_T_c
        self.get_T_d          = get_T_d
        self.get_q_hat        = get_q_hat
        self.get_q_hat_approx = get_q_hat_approx
        self.get_k            = get_k
        self.get_k_approx     = get_k_approx
        self.get_cV           = get_cV

        self.dom_vars = set([attr for attr in dir(self) if
                             not callable(getattr(self, attr)) and not attr.startswith("__")]) - other_vars
        other_vars = other_vars.union(self.dom_vars)

        #---------------------------------------------------------------------------------------------------------------
        # Discretization.

        # Coarse spatial discretization.
        self.N_x = N_x
        self.N_y = N_x # x and y use same discr.
        self.dx  = (self.x_b - self.x_a) / self.N_x
        self.dy  = (self.y_d - self.y_c) / self.N_y
        self.x_faces = np.linspace(self.x_a, self.x_b, num=self.N_x + 1, endpoint=True)
        self.y_faces = np.linspace(self.y_c, self.y_d, num=self.N_y + 1, endpoint=True)
        self.x_nodes = np.zeros(self.N_x + 2)
        self.x_nodes[0]    = self.x_a
        self.x_nodes[1:-1] = self.x_faces[:-1] + self.dx / 2
        self.x_nodes[-1]   = self.x_b
        self.y_nodes = np.zeros(self.N_y + 2)
        self.y_nodes[0]    = self.y_c
        self.y_nodes[1:-1] = self.y_faces[:-1] + self.dy / 2
        self.y_nodes[-1]   = self.y_d

        # Temporal discretization.
        self.dt = 1e-3
        self.N_t = int(self.t_end / self.dt) + 1

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
        self.N_train_examples = int(self.train_examples_ratio * self.N_t)
        self.N_val_examples = int(self.val_examples_ratio * self.N_t)
        self.N_test_examples = int(self.test_examples_ratio * self.N_t)
        if self.parametrized_system:
            self.N_train_examples *= lin_alphas.shape[0]
            self.N_val_examples *= lin_alphas.shape[0]
            self.N_test_examples *= lin_alphas.shape[0]
            self.N_test_examples += self.N_t * extra_alphas.shape[0]
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
            base = self.N_t - 1
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
            self.num_networks = self.N_x * self.N_y
            self.model_specific_params = [self.num_networks, self.num_layers, self.hidden_layer_size]
            def get_model_specific_params():
                return [self.num_networks, self.num_layers, self.hidden_layer_size]
        elif self.model_name == 'EnsembleGlobalCNN':
            self.num_conv_layers = 5
            self.kernel_size = 3
            self.num_channels = 20
            self.num_fc_layers = 1
            self.num_networks = self.N_x * self.N_y
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
            self.num_channels = 80
            self.model_specific_params = [self.num_conv_layers, self.kernel_size, self.num_channels]
            def get_model_specific_params():
                return [self.num_conv_layers, self.kernel_size, self.num_channels]
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