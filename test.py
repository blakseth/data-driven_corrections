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
    plt.figure()
    plt.xlabel("Test iterations", fontsize=20)
    plt.ylabel(r"$L_2$ Error")

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
    unc_errors = np.asarray([error_dicts[i]['unc'] for i in range(len(error_dicts))])
    cor_errors = np.asarray([error_dicts[i]['cor'] for i in range(len(error_dicts))])
    unc_error_mean = np.mean(unc_errors, axis=0)
    unc_error_std  = np.std(unc_errors,  axis=0)
    cor_error_mean = np.mean(cor_errors, axis=0)
    cor_error_std  = np.std(cor_errors,  axis=0)

    # Calculate statistical properties of plot data.
    unc_plots = np.asarray([plot_data_dicts[i]['unc'] for i in range(len(plot_data_dicts))])
    cor_plots = np.asarray([plot_data_dicts[i]['cor'] for i in range(len(plot_data_dicts))])
    unc_plot_mean = np.mean(unc_plots, axis=0)
    unc_plot_std  = np.std(unc_plots,  axis=0)
    cor_plot_mean = np.mean(cor_plots, axis=0)
    cor_plot_std  = np.std(cor_plots,  axis=0)

    # Pickle statistical properties.
    error_stats_dict = {
        'unc_mean': unc_error_mean,
        'unc_std':  unc_error_std,
        'cor_mean': cor_error_mean,
        'cor_std':  cor_error_std,
    }
    plot_stats_dict = {
        'unc_mean': unc_plot_mean,
        'unc_std':  unc_plot_std,
        'cor_mean': cor_plot_mean,
        'cor_std':  cor_plot_std,
        'ref':      plot_data_dicts[0]['ref'], # These are the same for all model instances, so the choice of ...
        'x':        plot_data_dicts[0]['x']    # ... retrieving them from the first instance is arbitrary.
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
    _, _, dataset_test = load_datasets(False, False, True)

    # Get stats used for normalization/unnormalization.
    stats = dataset_test[:6][3].detach().numpy()

    unc_mean = stats[0]
    unc_std  = stats[3]
    ref_mean = stats[1]
    ref_std  = stats[4]
    src_mean = stats[2]
    src_std  = stats[5]

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
    for i in range(config.N_test_examples):

        new_unc_tensor = dataset_test[i][0]
        new_ref_tensor = dataset_test[i][1]

        new_unc = util.z_unnormalize(new_unc_tensor.detach().numpy(), unc_mean, unc_std)
        new_ref = util.z_unnormalize(new_ref_tensor.detach().numpy(), ref_mean, ref_std)

        new_cor = np.zeros_like(new_ref)
        if config.model_is_hybrid:
            new_src = util.z_unnormalize(model.net(new_unc_tensor).detach().numpy(), src_mean, src_std)
            new_cor = physics.simulate(
                config.nodes_coarse, config.faces_coarse,
                config.get_T0(config.nodes_coarse), config.T_a, config.T_b,
                config.get_k, config.get_cV, config.rho, config.A,
                config.get_q_hat, new_src,
                config.dt_coarse, config.dt_coarse, False
            )
        else:
            new_cor[0]  = new_ref[0]   # Since BCs are not ...
            new_cor[-1] = new_ref[-1]  # predicted by the NN.
            new_cor[1:-1] = util.z_unnormalize(model.net(new_unc_tensor).detach().numpy(), ref_mean, ref_std)

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