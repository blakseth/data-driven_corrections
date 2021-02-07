"""
test.py

Written by Sindre Stenen Blakseth, 2021.

Testing if ML-model successfully corrects 1D heat equation simulations.
"""

########################################################################################################################
# Package imports.

import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import torch

########################################################################################################################
# File imports.

import config
from datasets import load_datasets
import models
import physics
import util

########################################################################################################################
# helper function for visualizing test data.

def visualize_test_data(error_stats_dict, plot_stats_dict):
    # Visualize error stats.
    iterations = np.arange(1, len(error_stats_dict['unc']) + 1, 1)

    plt.figure()
    plt.semilogy(iterations, error_stats_dict['unc'], 'r-', linewidth=2.0, label="Uncorrected")
    plt.semilogy(iterations, error_stats_dict['cor_mean'], 'b-', linewidth=2.0, label="Corrected, mean")
    plt.fill_between(iterations,
                     error_stats_dict['cor_mean'] + error_stats_dict['cor_std'],
                     error_stats_dict['cor_mean'] - error_stats_dict['cor_std'],
                     facecolor='yellow', alpha=0.5, label="Corrected, std.dev.")
    plt.xlim([0, len(error_stats_dict['unc'])])
    plt.xlabel("Test iterations", fontsize=20)
    plt.ylabel(r"$L_2$ Error")
    plt.xticks(fontsize=17)
    plt.yticks(fontsize=17)
    plt.grid()
    plt.legend(prop={'size': 19})
    plt.savefig(os.path.join(config.run_dir, "error_stats.pdf"), bbox_inches='tight')

    # Visualize temperature profiles.
    for i in range(plot_stats_dict['unc'].shape[0]):
        plt.figure()
        plt.plot(plot_stats_dict['x'], plot_stats_dict['unc'][i], 'r-', linewidth=2.0, label="Uncorrected")
        plt.plot(plot_stats_dict['x'], plot_stats_dict['cor_mean'][i], 'b-', linewidth=2.0, label="Corrected, mean")
        plt.plot(plot_stats_dict['x'], plot_stats_dict['ref'][i], 'k-', linewidth=2.0, label="Reference")
        plt.fill_between(plot_stats_dict['x'],
                         plot_stats_dict['cor_mean'][i] + plot_stats_dict['cor_std'][i],
                         plot_stats_dict['cor_mean'][i] - plot_stats_dict['cor_std'][i],
                         facecolor='yellow', alpha=0.5, label="Corrected, std.dev.")
        plt.xlim([plot_stats_dict['x'][0], plot_stats_dict['x'][-1]])
        plt.xlabel(r"$x$ (m)", fontsize=20)
        plt.ylabel(r"$T$ (K))")
        plt.xticks(fontsize=17)
        plt.yticks(fontsize=17)
        plt.grid()
        plt.legend(prop={'size': 19})
        plt.savefig(os.path.join(config.run_dir, "profiles" + str(i) + ".pdf"), bbox_inches='tight')

    plt.show()


########################################################################################################################
# Helper function for saving test data.

def save_test_data(error_dicts, plot_data_dicts):
    # Pickle raw data.
    with open(os.path.join(config.run_dir, "error_data_raw" + ".pkl"), "wb") as f:
        pickle.dump(error_dicts, f)
    with open(os.path.join(config.run_dir, "plot_data_raw" + ".pkl"), "wb") as f:
        pickle.dump(plot_data_dicts, f)

    # Save raw error data in a text file for easy manual inspection.
    with open(os.path.join(config.run_dir, "error_data_raw" + ".txt"), "w") as f:
        f.write("L2 error (corrected)\t\tL2 error (uncorrected)\n")
        for num, error_dict in enumerate(error_dicts):
            f.write("\nModel instance " + str(num) + "\n")
            for i in range(error_dict['unc'].shape[0]):
                f.write(str(i) + ": " + str(error_dict['cor'][i]) + "\t\t" + str(error_dict['unc'][i]) + "\n")
        f.close()

    # Calculate statistical properties of errors.
    cor_errors = np.asarray([error_dicts[i]['cor'] for i in range(len(error_dicts))])
    cor_error_mean = np.mean(cor_errors, axis=0)
    cor_error_std  = np.std(cor_errors,  axis=0)

    # Calculate statistical properties of plot data
    cor_plots = np.asarray([plot_data_dicts[i]['cor'] for i in range(len(plot_data_dicts))])
    cor_plot_mean = np.mean(cor_plots, axis=0)
    cor_plot_std  = np.std(cor_plots,  axis=0)

    # Pickle statistical properties.
    error_stats_dict = {
        'cor_mean': cor_error_mean,
        'cor_std':  cor_error_std,
        'unc':      error_dicts[0]['unc']
    }
    plot_stats_dict = {
        'cor_mean': cor_plot_mean,
        'cor_std':  cor_plot_std,
        'unc':      plot_data_dicts[0]['unc'], # These are the same for all model instances, ...
        'ref':      plot_data_dicts[0]['ref'], # ... so the choice of retrieving them from ...
        'x':        plot_data_dicts[0]['x']    # ... the first instance is arbitrary.
    }
    with open(os.path.join(config.run_dir, "error_data_stats" + ".pkl"), "wb") as f:
        pickle.dump(error_stats_dict, f)
    with open(os.path.join(config.run_dir, "plot_data_stats" + ".pkl"), "wb") as f:
        pickle.dump(plot_stats_dict, f)

    return error_stats_dict, plot_stats_dict


########################################################################################################################
# Testing ML-model.

def simulation_test(model, num):
    model.net.eval()

    # Get target temperature profile of last validation example. This will be IC for test simulation.
    _, dataset_val, dataset_test = load_datasets(False, True, True)

    # Get stats used for normalization/unnormalization.
    stats = dataset_test[:6][3].detach().numpy()

    unc_mean = stats[0]
    unc_std  = stats[3]
    ref_mean = stats[1]
    ref_std  = stats[4]
    src_mean = stats[2]
    src_std  = stats[5]

    # Use last reference data example of validation set as IC for test simulation.
    IC = util.z_unnormalize(dataset_val[-1][0].detach().numpy(), ref_mean, ref_std)

    L2_errors_unc = np.zeros(config.N_test_examples)
    L2_errors_cor = np.zeros(config.N_test_examples)

    plot_steps = config.profile_save_steps
    plot_data_dict = {
        'x': config.nodes_coarse,
        'unc': np.zeros((plot_steps.shape[0], config.nodes_coarse.shape[0])),
        'ref': np.zeros((plot_steps.shape[0], config.nodes_coarse.shape[0])),
        'cor': np.zeros((plot_steps.shape[0], config.nodes_coarse.shape[0]))
    }
    plot_num = 0
    old_unc = IC
    old_cor = IC
    for i in range(config.N_test_examples):
        # ref at new time given ref at old time is stored in the test set.
        new_ref_tensor = dataset_test[i][1]
        new_ref = util.z_unnormalize(new_ref_tensor.detach().numpy(), ref_mean, ref_std)

        # new_unc  = new uncorrected profile given old uncorrected profile.
        # new_unc_ = new uncorrected profile given old   corrected profile.
        if i == 0:
            new_unc_ = dataset_test[i][0]
            new_unc  = util.z_unnormalize(new_unc_.detach().numpy(), unc_mean, unc_std)
        else:
            new_unc = physics.simulate(
                config.nodes_coarse, config.faces_coarse,
                old_unc, config.T_a, config.T_b,
                config.get_k, config.get_cV, config.rho, config.A,
                config.get_q_hat, np.zeros(config.N_coarse),
                config.dt_coarse, config.dt_coarse, False
            )
            new_unc_ = torch.from_numpy(util.z_normalize(
                physics.simulate(
                    config.nodes_coarse, config.faces_coarse,
                    old_cor, config.T_a, config.T_b,
                    config.get_k, config.get_cV, config.rho, config.A,
                    config.get_q_hat, np.zeros(config.N_coarse),
                    config.dt_coarse, config.dt_coarse, False
                ), unc_mean, unc_std
            ))

        new_cor = np.zeros_like(new_ref)
        if config.model_is_hybrid:
            new_src = util.z_unnormalize(model.net(new_unc_).detach().numpy(), src_mean, src_std)
            new_cor = physics.simulate(
                config.nodes_coarse, config.faces_coarse,
                old_cor, config.T_a, config.T_b,
                config.get_k, config.get_cV, config.rho, config.A,
                config.get_q_hat, new_src,
                config.dt_coarse, config.dt_coarse, False
            )
        else:
            new_cor[0]  = new_ref[0]   # Since BCs are not ...
            new_cor[-1] = new_ref[-1]  # predicted by the NN.
            new_cor[1:-1] = util.z_unnormalize(model.net(new_unc_).detach().numpy(), ref_mean, ref_std)

        lin_unc = lambda x: util.linearize_between_nodes(x, config.nodes_coarse, new_unc)
        lin_ref = lambda x: util.linearize_between_nodes(x, config.nodes_coarse, new_ref)
        lin_cor = lambda x: util.linearize_between_nodes(x, config.nodes_coarse, new_cor)

        ref_norm = util.get_L2_norm(config.faces_coarse, lin_ref)
        unc_error_norm = util.get_L2_norm(config.faces_coarse, lambda x: lin_unc(x) - lin_ref(x)) / ref_norm
        cor_error_norm = util.get_L2_norm(config.faces_coarse, lambda x: lin_cor(x) - lin_ref(x)) / ref_norm

        L2_errors_unc[i] = unc_error_norm
        L2_errors_cor[i] = cor_error_norm

        if i in plot_steps:
            plot_data_dict['unc'][plot_num] = new_unc
            plot_data_dict['ref'][plot_num] = new_ref
            plot_data_dict['cor'][plot_num] = new_cor
            plot_num += 1

        old_unc = new_unc
        old_cor = new_cor

    error_dict = {
        'unc': L2_errors_unc,
        'cor': L2_errors_cor
    }

    return error_dict, plot_data_dict

########################################################################################################################

def main():
    model = models.create_new_model()
    num = 0
    simulation_test(model, num)

########################################################################################################################

if __name__ == "__main__":
    main()

########################################################################################################################