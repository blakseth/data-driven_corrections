"""
feature_engineering.py

Written by Sindre Stenen Blakseth, 2021.

Feature engineering for correcting the 1D Euler equations.
"""

########################################################################################################################
# Package imports.
import math

import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import torch


plt.rcParams.update({
    "font.family": "DejaVu Serif",
    "font.serif": ["Computer Modern Roman"],
})

########################################################################################################################
# File imports.

import config
import datasets
import exact_solver
import physics
import util

########################################################################################################################
# Helper functions.

def get_last_left_state_index(xs, loc):
    for i in range(xs.shape[0]):
        if xs[i] <= loc < xs[i + 1]:
            return i

########################################################################################################################
# Dataset creation.

def create_dataset(cfg):
    train_alphas  = cfg.train_alphas
    train_targets = np.zeros((train_alphas.shape[0], 4))
    for a in range(train_alphas.shape[0]):
        cfg.init_u1 = cfg.init_u2 = 0.1 * train_alphas[a]
        ref_V0 = physics.get_init_V_mtx(cfg)
        ref_V1 = exact_solver.exact_solver(cfg, cfg.dt)
        src    = physics.get_corr_src_term(cfg, ref_V0, ref_V1, 'LxF')
        train_targets[a, :2] = np.amax(src, axis=1)[1:]
        train_targets[a, 2:] = np.amin(src, axis=1)[1:]

    val_alphas = cfg.val_alphas
    val_targets = np.zeros((val_alphas.shape[0], 4))
    for a in range(val_alphas.shape[0]):
        cfg.init_u1 = cfg.init_u2 = 0.1 * val_alphas[a]
        ref_V0 = physics.get_init_V_mtx(cfg)
        ref_V1 = exact_solver.exact_solver(cfg, cfg.dt)
        src    = physics.get_corr_src_term(cfg, ref_V0, ref_V1, 'LxF')
        val_targets[a, :2] = np.amax(src, axis=1)[1:]
        val_targets[a, 2:] = np.amin(src, axis=1)[1:]

    train_target_means = np.mean(train_targets, axis=0)
    train_target_stds  = np.std( train_targets, axis=0)
    assert train_target_means.shape[0] == train_target_stds.shape[0] == 4

    train_targets_normal = util.z_normalize_componentwise(train_targets, train_target_means, train_target_stds, axis=1)
    val_targets_normal   = util.z_normalize_componentwise(val_targets,   train_target_means, train_target_stds, axis=1)

    print("train_targets", train_targets)
    print("train_targets_normal", train_targets_normal)
    print("train_means", train_target_means)
    print("train_stds:", train_target_stds)
    #raise Exception

    train_alphas_tensor  = torch.from_numpy(train_alphas)
    train_targets_tensor = torch.from_numpy(train_targets_normal)
    stats_train          = np.zeros((train_alphas.shape[0], 4))
    stats_train[0,:]     = train_target_means
    stats_train[1,:]     = train_target_stds
    stats_train_tensor   = torch.from_numpy(stats_train)

    val_alphas_tensor    = torch.from_numpy(val_alphas)
    val_targets_tensor   = torch.from_numpy(val_targets_normal)
    stats_val            = np.zeros((val_alphas.shape[0], 4))
    stats_val[0, :]      = train_target_means
    stats_val[1, :]      = train_target_stds
    stats_val_tensor     = torch.from_numpy(stats_val)

    dataset_train = torch.utils.data.TensorDataset(train_alphas_tensor, train_targets_tensor, stats_train_tensor)
    dataset_val   = torch.utils.data.TensorDataset(val_alphas_tensor,   val_targets_tensor,   stats_val_tensor)

    return dataset_train, dataset_val

########################################################################################################################
# Model definition.

class DenseLayerWithAct(torch.nn.Module):
    def __init__(self, size, dropout_prob):
        super(DenseLayerWithAct, self).__init__()
        self.layer      = torch.nn.Linear(size, size)
        self.activation = torch.nn.LeakyReLU()
        self.dropout    = torch.nn.Dropout(dropout_prob)

    def forward(self, x):
        return self.dropout(self.activation(self.layer(x)))

class DenseModule(torch.nn.Module):
    def __init__(self, num_layers, hidden_layer_size, dropout_prob):
        super(DenseModule, self).__init__()

        # Defining input layer.
        first_layer      = torch.nn.Linear(1, hidden_layer_size)
        first_activation = torch.nn.LeakyReLU()

        # Defining hidden layers.
        hidden_block = [
            DenseLayerWithAct(hidden_layer_size, dropout_prob) for hidden_layer in range(num_layers - 2)
        ]

        # Defining output layer.
        last_layer = torch.nn.Linear(hidden_layer_size, 4)

        # Defining full architecture.
        self.net = torch.nn.Sequential(
            first_layer,
            first_activation,
            *hidden_block,
            last_layer
        ).double()

    def forward(self, x):
        return self.net(x)

class Model:
    def __init__(self, depth, width, dp, loss_func, lr, optim):
        # Defining network.
        self.net = DenseModule(depth, width, dp)

        # Defining loss function.
        if loss_func == 'MSE':
            self.loss = torch.nn.MSELoss(reduction='mean')
        elif loss_func == 'L1':
            self.loss = torch.nn.L1Loss(reduction='sum')
        else:
            raise Exception("Invalid loss function selection.")

        # Defining learning rate and optimizer.
        self.learning_rate = lr
        if optim == 'adam':
            self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.learning_rate)
        else:
            raise Exception("Invalid optimizer selection.")

        # Lists for storing training losses and corresponding training iteration numbers.
        self.train_losses     = []
        self.train_iterations = []
        self.val_losses       = []
        self.val_iterations   = []

########################################################################################################################
# Training.

def train(model, dataset_train, dataset_val):
    max_iter      = 1e6
    min_iter      = 1e2
    overfit_limit = 20
    val_epoch_since_improvement = torch.tensor(0.0)
    lowest_val_los = torch.tensor(1e30)

    print_train_loss_period = 100
    save_train_loss_period  = 10
    validation_period       = 100

    dataloader_train = torch.utils.data.DataLoader(
        dataset     = dataset_train,
        batch_size  = 8,
        shuffle     = True,
        num_workers = 0,
        pin_memory  = True
    )
    dataloader_val = torch.utils.data.DataLoader(
        dataset     = dataset_val,
        batch_size  = 2,
        shuffle     = True,
        num_workers = 0,
        pin_memory  = True
    )

    it = 0
    for epoch in range(int(max_iter)):
        flag = True
        for i, data in enumerate(dataloader_train):
            if it >= max_iter or (val_epoch_since_improvement >= overfit_limit and it >= min_iter):
                break
            it += 1

            input  = torch.unsqueeze(data[0], dim=1)
            target = data[1]
            output = model.net(input)

            loss = model.loss(output, target)
            if it % print_train_loss_period == 0:
                print(it, loss.item())
            if it % save_train_loss_period == 0:
                model.train_losses.append(loss.item())
                model.train_iterations.append(it)

            model.net.zero_grad()

            loss.backward()

            model.optimizer.step()

            if it % validation_period == 0:
                model.net.eval()
                with torch.no_grad():
                    for j, val_data in enumerate(dataloader_val):
                        val_input  = torch.unsqueeze(val_data[0], dim=1)
                        val_target = val_data[1]
                        val_output = model.net(val_input)
                        val_loss   = model.loss(val_output, val_target)

                        model.val_losses.append(val_loss.item())
                        model.val_iterations.append(it)
                        if val_loss < lowest_val_los:
                            lowest_val_los = val_loss
                            val_epoch_since_improvement = 0
                        else:
                            val_epoch_since_improvement += 1
            flag = False
            #print("it", it)
        #print("epoch:", epoch)
        if flag:
            break

    data_dict = dict()
    data_dict['Training loss'] = np.asarray([model.train_iterations, model.train_losses])
    data_dict['Validation loss'] = np.asarray([model.val_iterations, model.val_losses])

    return data_dict

########################################################################################################################
# Testing.

def test(cfg, model, dataset_train):
    means = dataset_train[0][2].detach().numpy()[:4]
    stds  = dataset_train[1][2].detach().numpy()[:4]
    final_profiles = {
        'unc': np.zeros((cfg.test_alphas.shape[0], 3, cfg.x_nodes.shape[0])),
        'cor': np.zeros((cfg.test_alphas.shape[0], 3, cfg.x_nodes.shape[0])),
        'ref': np.zeros((cfg.test_alphas.shape[0], 3, cfg.x_nodes.shape[0]))
    }
    L2_errors_unc = np.zeros((cfg.test_alphas.shape[0], cfg.N_t - 1))
    L2_errors_cor = np.zeros((cfg.test_alphas.shape[0], cfg.N_t - 1))

    abcs = pickle.load(open("targets.pkl", 'rb'))
    loaded_maxs = abcs['maxs']
    loaded_mins = abcs['mins']

    for a, alpha in enumerate(cfg.test_alphas):
        print("alpha:", alpha)
        u = cfg.init_u1 = cfg.init_u2 = 0.1*alpha

        old_time = 0.0
        old_unc_V = physics.get_init_V_mtx(cfg)
        old_cor_V = physics.get_init_V_mtx(cfg)
        old_ref_V = physics.get_init_V_mtx(cfg)
        old_cd_loc = cfg.x_split
        old_cd_index = get_last_left_state_index(cfg.x_nodes, old_cd_loc)

        print("tensor alpha:", torch.tensor(alpha).reshape((1,1)))
        alpha_in = torch.tensor(alpha).reshape((1,1)).double()
        predictions = util.z_unnormalize_componentwise(model.net(alpha_in).detach().numpy(), means, stds, axis=1)
        maxs = np.zeros(3)
        mins = np.zeros(3)
        maxs[0] = 0.2 - 0.01*alpha
        mins[0] = -0.2 - 0.01*alpha
        maxs[1:] = np.squeeze(predictions)[:2]
        mins[1:] = np.squeeze(predictions)[2:]

        print("loaded_maxs:", loaded_maxs[18 + a])
        print("maxs", maxs)
        print("loaded_mins:", loaded_mins[18 + a])
        print("mins", mins)

        for i in range(1, cfg.N_t):
            new_time = np.around(old_time + cfg.dt, decimals=10)

            new_cd_loc = old_cd_loc + u*cfg.dt
            new_cd_index = get_last_left_state_index(cfg.x_nodes, new_cd_loc)

            src = np.zeros((3, cfg.N_x))
            if math.isclose(new_cd_loc, cfg.x_nodes[new_cd_index]):
                #print("0")
                src[:, new_cd_index-2] = maxs
                src[:, new_cd_index-1] = mins
            elif math.isclose(old_cd_loc, cfg.x_nodes[old_cd_index]):
                #print("3")
                src[:, new_cd_index - 2] = src[:, new_cd_index - 1] = maxs
            elif new_cd_index == old_cd_index:
                #print("1")
                #print("src[:,new_cd_index-1].shape", src[:,new_cd_index-1].shape)
                #print("maxs.shape", maxs.shape)
                src[:,new_cd_index-1] = maxs
                src[:,new_cd_index]   = mins
            else:
                #print("2")
                src[:,new_cd_index-2] = src[:,new_cd_index-1] = maxs

            new_cor_V = physics.get_new_state(cfg, old_cor_V, src, 'LxF')

            new_unc_V = physics.get_new_state(cfg, old_unc_V, np.zeros_like(src), 'LxF')
            new_ref_V = exact_solver.exact_solver(cfg, new_time)
            #src_exact = physics.get_corr_src_term(cfg, old_exact_V, new_exact_V, 'LxF')

            ref_norm = util.get_disc_L2_norm(new_ref_V)
            L2_errors_unc[a][i-1] = util.get_disc_L2_norm(new_unc_V - new_ref_V) / ref_norm
            L2_errors_cor[a][i-1] = util.get_disc_L2_norm(new_cor_V - new_ref_V) / ref_norm

            print("L2_errors_unc[a][i-1].shape:", L2_errors_unc[a][i-1])

            old_unc_V    = new_unc_V
            old_cor_V    = new_cor_V
            old_ref_V    = new_ref_V
            old_time     = new_time
            old_cd_loc   = new_cd_loc
            old_cd_index = new_cd_index

        final_profiles['unc'][a] = old_unc_V
        final_profiles['cor'][a] = old_cor_V
        final_profiles['ref'][a] = old_ref_V

    error_dict = {
        'unc': L2_errors_unc,
        'cor': L2_errors_cor
    }

    return error_dict, final_profiles

########################################################################################################################
# Validate that perfectly learnt corrective source terms will yield zero error in the corrected solution.

def validate():
    cfg = config.Config(
        use_GPU=config.use_GPU,
        group_name=config.group_name,
        run_name=config.run_names[0][0],
        system=config.systems[0],
        data_tag=config.data_tags[0],
        model_key=config.model_keys[0],
        do_train=False,
        do_test=False,
        N_x=100,
        model_type=config.model_type,
    )

    abcs = pickle.load(open("targets.pkl", 'rb'))
    maxs = abcs['maxs']
    mins = abcs['mins']

    final_Vs     = []
    final_ref_Vs = []
    for a, alpha in enumerate(cfg.alphas):
        print("alpha:", alpha)
        u = cfg.init_u1 = cfg.init_u2 = 0.1*alpha
        if a > 15:
            u = np.around(u, decimals=5)
            cfg.init_u1 = np.around(cfg.init_u1, decimals=5)
            cfg.init_u2 = np.around(cfg.init_u2, decimals=5)

        old_time = 0.0
        old_V = physics.get_init_V_mtx(cfg)
        old_exact_V = physics.get_init_V_mtx(cfg)
        old_cd_loc = cfg.x_split
        old_cd_index = get_last_left_state_index(cfg.x_nodes, old_cd_loc)

        for i in range(1, cfg.N_t):
            new_time = np.around(old_time + cfg.dt, decimals=10)
            new_cd_loc = np.around(old_cd_loc + u*cfg.dt, decimals=10)
            #print("new_cd_loc", new_cd_loc)
            #print("old_cd_loc", old_cd_loc)
            new_cd_index = get_last_left_state_index(cfg.x_nodes, new_cd_loc)
            src = np.zeros((3, cfg.N_x))
            if math.isclose(new_cd_loc, cfg.x_nodes[new_cd_index]):
                #print("0")
                src[:, new_cd_index-2] = maxs[a, :]
                src[:, new_cd_index-1] = mins[a, :]
            elif math.isclose(old_cd_loc, cfg.x_nodes[old_cd_index]):
                #print("3")
                src[:, new_cd_index - 2] = src[:, new_cd_index - 1] = maxs[a, :]
            elif new_cd_index == old_cd_index:
                #print("1")
                src[:,new_cd_index-1] = maxs[a,:]
                src[:,new_cd_index]   = mins[a,:]
            else:
                #print("2")
                src[:,new_cd_index-2] = src[:,new_cd_index-1] = maxs[a,:]
            new_V = physics.get_new_state(cfg, old_V, src, 'LxF')
            new_exact_V = exact_solver.exact_solver(cfg, new_time)
            src_exact = physics.get_corr_src_term(cfg, old_exact_V, new_exact_V, 'LxF')
            cor_V = physics.get_new_state(cfg, old_exact_V, src_exact, 'LxF')

            #print("x_nodes:", cfg.x_nodes)
            #print("new_cd_index:", new_cd_index)
            #print("argmax src:", np.argmax(src[0]))
            #print("argmax src_exact:", np.argmax(src_exact[0]))
            #print("src:", src)
            #print("src_exact:", src_exact)

            #print("new_V:", new_V)
            #print("new_V_exact", new_exact_V)
            #np.testing.assert_allclose(cor_V, new_exact_V, rtol=1e-8, atol=1e-8)
            #np.testing.assert_allclose(src, src_exact, atol=1e-8)
            #np.testing.assert_allclose(new_V, new_exact_V, rtol=1e-8, atol=1e-8)

            old_V = new_V
            old_time = new_time
            old_exact_V = new_exact_V
            old_cd_loc = new_cd_loc
            old_cd_index = new_cd_index

        final_Vs.append(old_V)
        final_ref_Vs.append(old_exact_V)

    for i in range(len(final_Vs)):
        print("alpha:", cfg.alphas[i])
        print("cor_V:", final_Vs[i])
        print("ref_V:", final_ref_Vs[i])

    print("Success")

########################################################################################################################

def main():
    print("\nEXECUTION INITIATED\n")
    print("--------------------------------------")
    print("Concept validation initiated.")
    #validate()
    print("Concept validation completed.")
    print("--------------------------------------\n")

    # Create configuration object.
    cfg = config.Config(
        use_GPU     = config.use_GPU,
        group_name  = config.group_name,
        run_name    = config.run_names[0][0],
        system      = config.systems[0],
        data_tag    = config.data_tags[0],
        model_key   = config.model_keys[0],
        do_train    = False,
        do_test     = False,
        N_x         = 100,
        model_type  = config.model_type,
    )
    #os.makedirs(cfg.run_dir, exist_ok=False)

    """

    print("--------------------------------------")
    print("Dataset creation initiated.")
    dataset_train, dataset_val = create_dataset(cfg)
    print("Dataset creation completed.")
    print("--------------------------------------\n")

    print("--------------------------------------")
    print("Model creation initiated.")
    model = Model(
        depth     = 15,
        width     = 20,
        dp        = 0.0,
        loss_func = 'MSE',
        lr        = 1e-4,
        optim     = 'adam'
    )
    print("Model creation completed.")
    print("--------------------------------------\n")

    print("--------------------------------------")
    print("Training initiated.")
    train_data = train(model, dataset_train, dataset_val)
    print("Training completed.")
    print("--------------------------------------\n")

    print("--------------------------------------")
    print("Testing initiated.")
    errors, profiles = test(cfg, model, dataset_train)
    print("Errors unc:", errors['unc'])
    print("Errors cor:", errors['cor'])

    print("Testing completed.")
    print("--------------------------------------\n")

    print("--------------------------------------")
    print("Data saving initiated.")
    with open(os.path.join(cfg.run_dir, "plot_data_raw" + ".pkl"), "wb") as f:
        pickle.dump(profiles, f)
    with open(os.path.join(cfg.run_dir, "error_data_raw" + ".pkl"), "wb") as f:
        pickle.dump(errors, f)
    with open(os.path.join(cfg.run_dir, "train_data_raw" + ".pkl"), "wb") as f:
        pickle.dump(train_data, f)
    print("Data saving completed.")
    print("--------------------------------------")
    
    """

    with open(os.path.join("/home/sindre/msc_thesis/data-driven_corrections/results/2021-04-17_Euler_contact_disc/Euler_Dense_feature_engineered/plot_data_raw.pkl"), "rb") as f:
        profiles = pickle.load(f)
    with open(os.path.join("/home/sindre/msc_thesis/data-driven_corrections/results/2021-04-17_Euler_contact_disc/Euler_Dense_feature_engineered/error_data_raw.pkl"), "rb") as f:
        errors = pickle.load(f)

    my_path = "/home/sindre/msc_thesis/data-driven_corrections/thesis_figures/2021-06-19_euler"
    os.makedirs(my_path, exist_ok=True)

    print("--------------------------------------")
    print("Data visualization initiated.")
    for a, alpha in enumerate(cfg.test_alphas):
        # Profiles.
        unc = np.append(profiles['unc'][a], np.reshape(profiles['unc'][a][0] / (cfg.c_V * (cfg.gamma - 1) * profiles['unc'][a][2]), (1, cfg.x_nodes.shape[0])), axis=0)
        cor = np.append(profiles['cor'][a], np.reshape(profiles['cor'][a][0] / (cfg.c_V * (cfg.gamma - 1) * profiles['cor'][a][2]), (1, cfg.x_nodes.shape[0])), axis=0)
        ref = np.append(profiles['ref'][a], np.reshape(profiles['ref'][a][0] / (cfg.c_V * (cfg.gamma - 1) * profiles['ref'][a][2]), (1, cfg.x_nodes.shape[0])), axis=0)
        fig, axs = plt.subplots(2, 2)
        ylabels = [r"$p$ (Pa)", r"$v$ (m/s)", r"$T$ (°C)", r"$\rho$ (kg/m$^3$)"]
        for j in range(4):
            axs[j // 2][j % 2].scatter(cfg.x_nodes, unc[j], s=10, marker='o', facecolors='none', edgecolors='r', linewidths = 0.5, label='LxF')
            axs[j // 2][j % 2].scatter(cfg.x_nodes, cor[j], s=10, marker='D', facecolors='none', edgecolors='g', linewidths = 0.5, label='HAM')
            axs[j // 2][j % 2].plot(cfg.x_nodes, ref[j], 'k-', label='Exact')
            #axs[j].legend()
            axs[j // 2][j % 2].set_xlabel(r'$x$ (m)')
            axs[j // 2][j % 2].set_ylabel(ylabels[j])
            axs[j // 2][j % 2].set_axisbelow(True)
            axs[j // 2][j % 2].grid()
            #axs[j // 2][j % 2].label_outer()
        plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.35, wspace=0.4)
        plt.savefig(os.path.join(my_path, "profiles_alpha" + str(np.around(alpha, decimals=5)) + "t" + str(
            np.around(cfg.t_end, decimals=5)) + ".pdf"), bbox_inches='tight')
        plt.close()

        # Errors.
        print("errors['unc'].shape", errors['unc'].shape)
        iterations = np.arange(1, errors['unc'].shape[1] + 1, 1)
        print("iterations:", iterations)
        plt.figure()
        plt.semilogy(iterations, errors['unc'][a], 'r-', linewidth=2.0, label="LxF")
        plt.semilogy(iterations, errors['cor'][a], 'g-', linewidth=2.0, label="HAM")
        plt.xlim([0, len(errors['unc'][a])])
        plt.xlabel("Time level", fontsize=20)
        plt.ylabel(r"Relative $\ell_2$-error", fontsize=20)
        plt.xticks(fontsize=17)
        plt.yticks(fontsize=17)
        plt.grid()
        #plt.legend(prop={'size': 17})
        plt.savefig(os.path.join(my_path, "errors_alpha" + str(np.around(alpha, decimals=5)) + "t" + str(
            np.around(cfg.t_end, decimals=5)) + ".pdf"), bbox_inches='tight')
        plt.close()
    """
    # Losses
    plt.figure()
    plt.semilogy(train_data['Training loss'][0], train_data['Training loss'][1], label='train')
    plt.semilogy(train_data['Validation loss'][0], train_data['Validation loss'][1], label='val')
    plt.xlabel("Iterations")
    plt.ylabel("MSE loss")
    plt.legend()
    plt.savefig(os.path.join(cfg.run_dir, "losses.pdf"), bbox_inches='tight')
    plt.close()
    """
    print("Data visualization completed.")
    print("--------------------------------------\n")

    print("EXECUTION COMPLETED")

########################################################################################################################

if __name__ == "__main__":
    main()

########################################################################################################################