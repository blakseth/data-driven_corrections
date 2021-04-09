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

from datasets import load_datasets
import models
import physics
import util

########################################################################################################################
# helper function for visualizing test data.

def visualize_test_data(cfg, error_stats_dict, plot_stats_dict):
    # Visualize error stats.
    iterations = np.arange(1, error_stats_dict['unc_L2'].shape[1] + 1, 1)
    for a, alpha in enumerate(plot_stats_dict['alphas']):
        plt.figure()
        plt.semilogy(iterations, error_stats_dict['unc_L2'][a], 'r-', linewidth=2.0, label="Uncorrected")
        plt.semilogy(iterations, error_stats_dict['cor_mean_L2'][a], 'b-', linewidth=2.0, label="Corrected, mean")
        plt.fill_between(iterations,
                         error_stats_dict['cor_mean_L2'][a] + error_stats_dict['cor_std_L2'][a],
                         error_stats_dict['cor_mean_L2'][a] - error_stats_dict['cor_std_L2'][a],
                         facecolor='yellow', alpha=0.5, label="Corrected, std.dev.")
        plt.xlim([0, len(error_stats_dict['unc_L2'][a])])
        plt.xlabel("Test iterations", fontsize=20)
        plt.ylabel(r"$l_2$ Error", fontsize=20)
        plt.xticks(fontsize=17)
        plt.yticks(fontsize=17)
        plt.grid()
        plt.legend(prop={'size': 17})
        plt.savefig(os.path.join(cfg.run_dir, "l2_error_stats_alpha" + str(np.around(alpha, decimals=5)) + ".pdf"), bbox_inches='tight')
        plt.close()

        plt.figure()
        plt.semilogy(iterations, error_stats_dict['unc_Linfty'][a], 'r-', linewidth=2.0, label="Uncorrected")
        plt.semilogy(iterations, error_stats_dict['cor_mean_Linfty'][a], 'b-', linewidth=2.0, label="Corrected, mean")
        plt.fill_between(iterations,
                         error_stats_dict['cor_mean_Linfty'][a] + error_stats_dict['cor_std_Linfty'][a],
                         error_stats_dict['cor_mean_Linfty'][a] - error_stats_dict['cor_std_Linfty'][a],
                         facecolor='yellow', alpha=0.5, label="Corrected, std.dev.")
        plt.xlim([0, len(error_stats_dict['unc_Linfty'][a])])
        plt.xlabel("Test iterations", fontsize=20)
        plt.ylabel(r"$l_2$ Error", fontsize=20)
        plt.xticks(fontsize=17)
        plt.yticks(fontsize=17)
        plt.grid()
        plt.legend(prop={'size': 17})
        plt.savefig(os.path.join(cfg.run_dir, "linfty_error_stats_alpha" + str(np.around(alpha, decimals=5)) + ".pdf"), bbox_inches='tight')
        plt.close()

    t0 = (cfg.train_examples_ratio + cfg.test_examples_ratio)*cfg.t_end
    x_dense = np.linspace(cfg.x_a, cfg.x_b, num=1001, endpoint=True)
    y_dense = np.linspace(cfg.y_c, cfg.y_d, num=1001, endpoint=True)
    plot_times = plot_stats_dict['time']
        #print("plot_times:", plot_times)

    if 'cor_means_mean' in error_stats_dict.keys():
        labels = ['ML-corrected', 'Uncorrected']
        avgs = [error_stats_dict['cor_means_mean'], error_stats_dict['unc_means_mean']]
        devs = [error_stats_dict['cor_stds_mean'], error_stats_dict['unc_stds_mean']]
        #print(labels)
        #print("avgs:", avgs)
        #print("devs:", devs)

        x = np.arange(len(labels))  # the label locations
        width = 0.4  # the width of the bars

        fig, ax = plt.subplots()
        fig.set_figheight(3)
        fig.set_figwidth(8)
        ax.yaxis.grid(zorder=-1)
        # ax.set_aspect(2)
        rects1 = ax.bar(x - width / 2, avgs, width, label='Average', zorder=3)
        rects2 = ax.bar(x + width / 2, devs, width, label='Standard deviation', zorder=3)

        # Add some text for labels, title and custom x-axis tick labels, etc.
        ax.set_ylabel(r'$L_2$ error', fontsize=15)
        # ax.set_title('Scores by group and gender')
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=12)
        ax.legend(prop={'size': 11})
        ax.set_yscale('log')
        #ax.set_ylim((0, 0.15))
        fig.tight_layout()

        plt.savefig(os.path.join(cfg.run_dir, "histogram.pdf"), bbox_inches='tight')
        plt.close()
        with open(os.path.join(cfg.run_dir, "histogram_data" + ".txt"), "w") as f:
            f.write(str(labels) + "\n")
            f.write("avgs: " + str(avgs) + "\n")
            f.write("devs: " + str(devs) + "\n")

    # Visualize temperature profiles.
    for a, alpha in enumerate(plot_stats_dict['alphas']):
        for i in range(plot_stats_dict['unc'][a].shape[0]):
            unc_field = plot_stats_dict['unc'][a][i]
            cor_field = plot_stats_dict['cor_mean'][a][i]
            ref_field = plot_stats_dict['ref'][a][i]
            ref_dense = cfg.get_T_exact(x_dense, y_dense, plot_times[i], alpha)
            minmin = np.min([np.amin(unc_field), np.amin(cor_field), np.amin(ref_field), np.amin(ref_dense)])
            maxmax = np.min([np.amax(unc_field), np.amax(cor_field), np.amax(ref_field), np.amax(ref_dense)])
            fig, axs = plt.subplots(2, 2)
            im = axs[0, 0].imshow(np.flip(np.swapaxes(ref_field, 0, 1), 0), vmin=minmin, vmax=maxmax,
                             extent=[cfg.x_a - 0.5*cfg.dx, cfg.x_b + 0.5*cfg.dx,
                                     cfg.y_c - 0.5*cfg.dy, cfg.y_d + 0.5*cfg.dy])
            axs[0, 0].set_title('Reference')
            surf = axs[0, 1].contourf(x_dense, y_dense, np.swapaxes(ref_dense, 0, 1), vmin=minmin, vmax=maxmax, levels=100)
            for c in surf.collections:
                c.set_edgecolor("face")
            axs[0, 1].set_title('Reference')
            axs[1, 0].imshow(np.flip(np.swapaxes(unc_field, 0, 1), 0), vmin=minmin, vmax=maxmax,
                             extent=[cfg.x_a - 0.5*cfg.dx, cfg.x_b + 0.5*cfg.dx,
                                     cfg.y_c - 0.5*cfg.dy, cfg.y_d + 0.5*cfg.dy])
            axs[1, 0].set_title('Uncorrected')
            axs[1, 1].imshow(np.flip(np.swapaxes(cor_field, 0, 1), 0), vmin=minmin, vmax=maxmax,
                             extent=[cfg.x_a - 0.5*cfg.dx, cfg.x_b + 0.5*cfg.dx,
                                     cfg.y_c - 0.5*cfg.dy, cfg.y_d + 0.5*cfg.dy])
            axs[1, 1].set_title('Corrected')
            for ax in fig.get_axes():
                ax.set_xlim((cfg.x_a, cfg.x_b))
                ax.set_ylim((cfg.y_c, cfg.y_d))
                ax.set_xlabel(r'$x$ (m)')
                ax.set_ylabel(r'$y$ (m)')
                ax.label_outer()
            fig.colorbar(im, ax=axs.ravel().tolist())
            plt.savefig(os.path.join(cfg.run_dir, "profiles_alpha" + str(np.around(alpha, decimals=5)) + "t" + str(
                np.around(plot_times[i], decimals=5)) + ".pdf"), bbox_inches='tight')
            plt.close()
    """
    # Visualize correction source terms (if applicable).
    if 'src_mean' in plot_stats_dict.keys():
        for a, alpha in enumerate(plot_stats_dict['alphas']):
            for i in range(plot_stats_dict['src_mean'][a].shape[0]):
                plt.figure()
                plt.plot(plot_stats_dict['x'][1:-1], plot_stats_dict['src_mean'][a][i], 'b-', linewidth=2.0,
                         label="Corrected, mean")
                plt.plot(x_dense,
                         cfg.get_q_hat(x_dense, plot_times[i], alpha) - cfg.get_q_hat_approx(x_dense, plot_times[i], alpha),
                         'k-', linewidth=2.0, label="Reference")
                plt.fill_between(plot_stats_dict['x'][1:-1],
                                 plot_stats_dict['src_mean'][a][i] + plot_stats_dict['src_std'][a][i],
                                 plot_stats_dict['src_mean'][a][i] - plot_stats_dict['src_std'][a][i],
                                 facecolor='yellow', alpha=0.5, label="Corrected, std.dev.")
                plt.xlim([plot_stats_dict['x'][0], plot_stats_dict['x'][-1]])
                plt.xlabel(r"$x$ (m)", fontsize=20)
                plt.ylabel(r"$T$ (K)", fontsize=20)
                plt.xticks(fontsize=17)
                plt.yticks(fontsize=17)
                plt.grid()
                plt.legend(prop={'size': 17})
                plt.savefig(os.path.join(cfg.run_dir, "src_profiles_alpha" + str(np.around(alpha, decimals=5)) + "t" + str(np.around(plot_times[i], decimals=5)) + ".pdf"),
                            bbox_inches='tight')
                plt.close()
    """

########################################################################################################################
# Helper function for saving test data.

def save_test_data(cfg, error_dicts, plot_data_dicts):
    # Pickle raw data.
    with open(os.path.join(cfg.run_dir, "error_data_raw" + ".pkl"), "wb") as f:
        pickle.dump(error_dicts, f)
    with open(os.path.join(cfg.run_dir, "plot_data_raw" + ".pkl"), "wb") as f:
        pickle.dump(plot_data_dicts, f)

    # Save raw error data in a text file for easy manual inspection.
    with open(os.path.join(cfg.run_dir, "error_data_raw" + ".txt"), "w") as f:
        if cfg.parametrized_system:
            for a, alpha in enumerate(plot_data_dicts[0]['alphas']):
                f.write("alpha: " + str(np.around(alpha, decimals=5)) + "\n")
                f.write("l2 error (corrected)\t\tl2 error (uncorrected)\t\tl_inf error (corrected)\t\tl_inf error (uncorrected)\n")
                for num, error_dict in enumerate(error_dicts):
                    f.write("\nModel instance " + str(num) + "\n")
                    for i in range(error_dict['unc_L2'][a].shape[0]):
                        f.write(str(i) + ": " + str(error_dict['cor_L2'][a][i]) + "\t\t" + str(error_dict['unc_L2'][a][i]) + "\t\t" + str(error_dict['cor_Linfty'][a][i]) + "\t\t" + str(error_dict['unc_Linfty'][a][i]) + "\n")
                f.write("\n")
        else:
            f.write("l2 error (corrected)\t\tl2 error (uncorrected)\t\tl_inf error (corrected)\t\tl_inf error (uncorrected)\n")
            for num, error_dict in enumerate(error_dicts):
                f.write("\nModel instance " + str(num) + "\n")
                for i in range(error_dict['unc_L2'].shape[0]):
                    f.write(str(i) + ": " + str(error_dict['cor_L2'][i]) + "\t\t" + str(error_dict['unc_L2'][i]) + "\t\t" + str(error_dict['cor_Linfty'][i]) + "\t\t" + str(error_dict['unc_Linfty'][i]) +"\n")

    if cfg.parametrized_system:
        cor_L2_error_means_list = []
        cor_L2_error_stds_list  = []
        cor_Linfty_error_means_list = []
        cor_Linfty_error_stds_list = []
        cor_plot_means_list  = []
        cor_plot_stds_list   = []
        cor_plot_bests_list  = []
        if 'src' in plot_data_dicts[0].keys():
            src_plot_means_list = []
            src_plot_stds_list  = []
        for a, alpha in enumerate(plot_data_dicts[0]['alphas']):
            # Calculate statistical properties of errors.
            cor_L2_errors = np.asarray([error_dicts[i]['cor_L2'][a] for i in range(len(error_dicts))])
            best_system_index = np.argmin(cor_L2_errors[:,-1])
            cor_L2_error_means_list.append(np.mean(cor_L2_errors, axis=0))
            cor_L2_error_stds_list.append(np.std(cor_L2_errors, axis=0))
            cor_Linfty_errors = np.asarray([error_dicts[i]['cor_Linfty'][a] for i in range(len(error_dicts))])
            cor_Linfty_error_means_list.append(np.mean(cor_Linfty_errors, axis=0))
            cor_Linfty_error_stds_list.append(np.std(cor_Linfty_errors, axis=0))

            # Calculate statistical properties of plot data
            cor_plots = np.asarray([plot_data_dicts[i]['cor'][a] for i in range(len(plot_data_dicts))])
            cor_plot_means_list.append(np.mean(cor_plots, axis=0))
            cor_plot_stds_list.append(np.std(cor_plots, axis=0))
            cor_plot_bests_list.append(plot_data_dicts[best_system_index]['cor'][a])
            if 'src' in plot_data_dicts[0].keys():
                src_plots = np.asarray([plot_data_dicts[i]['src'][a] for i in range(len(plot_data_dicts))])
                src_plot_means_list.append(np.mean(src_plots, axis=0))
                src_plot_stds_list.append(np.std(src_plots, axis=0))

        # Pickle statistical properties.
        error_stats_dict = {
            'cor_mean_L2': np.asarray(cor_L2_error_means_list),
            'cor_std_L2': np.asarray(cor_L2_error_stds_list),
            'unc_L2': error_dicts[0]['unc_L2'],
            'cor_mean_Linfty': np.asarray(cor_Linfty_error_means_list),
            'cor_std_Linfty': np.asarray(cor_Linfty_error_stds_list),
            'unc_Linfty': error_dicts[0]['unc_Linfty']
        }
        plot_stats_dict = {
            'cor_mean': np.asarray(cor_plot_means_list),
            'cor_best': np.asarray(cor_plot_bests_list),
            'cor_std': np.asarray(cor_plot_stds_list),
            'unc': np.asarray([plot_data_dicts[0]['unc'][a] for a in range(plot_data_dicts[0]['alphas'].shape[0])]),  # These are the same for all model instances, ...
            'ref': np.asarray([plot_data_dicts[0]['ref'][a] for a in range(plot_data_dicts[0]['alphas'].shape[0])]),  # ... so the choice of retrieving them from ...
            'x': plot_data_dicts[0]['x'],      # ... the first instance is arbitrary.
            'time': plot_data_dicts[0]['time'],
            'alphas': plot_data_dicts[0]['alphas']
        }
        if 'src' in plot_data_dicts[0].keys():
            plot_stats_dict['src_mean'] = np.asarray(src_plot_means_list)
            plot_stats_dict['src_std'] = np.asarray(src_plot_stds_list)
    else:
        # Calculate statistical properties of errors.
        cor_L2_errors = np.asarray([error_dicts[i]['cor'] for i in range(len(error_dicts))])
        cor_L2_error_mean = np.mean(cor_L2_errors, axis=0)
        cor_L2_error_std  = np.std(cor_L2_errors,  axis=0)
        cor_Linfty_errors = np.asarray([error_dicts[i]['cor'] for i in range(len(error_dicts))])
        cor_Linfty_error_mean = np.mean(cor_Linfty_errors, axis=0)
        cor_Linfty_error_std = np.std(cor_Linfty_errors, axis=0)

        if 'cor_mean' in error_dicts[0].keys():
            cor_means = np.asarray([error_dicts[i]['cor_mean'] for i in range(len(error_dicts))])
            cor_means_mean = np.mean(cor_means, axis=0)
        if 'cor_std' in error_dicts[0].keys():
            cor_stds = np.asarray([error_dicts[i]['cor_std'] for i in range(len(error_dicts))])
            cor_stds_mean = np.mean(cor_stds, axis=0)

        # Calculate statistical properties of plot data
        cor_plots = np.asarray([plot_data_dicts[i]['cor'] for i in range(len(plot_data_dicts))])
        cor_plot_mean = np.mean(cor_plots, axis=0)
        cor_plot_std  = np.std(cor_plots,  axis=0)
        if 'src' in plot_data_dicts[0].keys():
            src_plots = np.asarray([plot_data_dicts[i]['src'] for i in range(len(plot_data_dicts))])
            src_plot_mean = np.mean(src_plots, axis=0)
            src_plot_std  = np.std(src_plots,  axis=0)

        # Pickle statistical properties.
        error_stats_dict = {
            'cor_mean_L2': cor_L2_error_mean,
            'cor_std_L2':  cor_L2_error_std,
            'cor_mean_Linfty': cor_Linfty_error_mean,
            'cor_std_Linfty': cor_Linfty_error_std,
            'unc_L2':      error_dicts[0]['unc_L2'],
            'unc_Linfty': error_dicts[0]['unc_Linfty']
        }
        if 'cor_mean' in error_dicts[0].keys():
            error_stats_dict['cor_means_mean'] = cor_means_mean
        if 'cor_std' in error_dicts[0].keys():
            error_stats_dict['cor_stds_mean'] = cor_stds_mean
        if 'unc_mean' in error_dicts[0].keys():
            error_stats_dict['unc_means_mean'] = error_dicts[0]['unc_mean']
        if 'unc_std' in error_dicts[0].keys():
            error_stats_dict['unc_stds_mean'] = error_dicts[0]['unc_std']
        plot_stats_dict = {
            'cor_mean': cor_plot_mean,
            'cor_std':  cor_plot_std,
            'unc':      plot_data_dicts[0]['unc'], # These are the same for all model instances, ...
            'ref':      plot_data_dicts[0]['ref'], # ... so the choice of retrieving them from ...
            'x':        plot_data_dicts[0]['x'],   # ... the first instance is arbitrary.
            'y':        plot_data_dicts[0]['y'],
            'time':     plot_data_dicts[0]['time'],
        }
        if 'src' in plot_data_dicts[0].keys():
            plot_stats_dict['src_mean'] = src_plot_mean
            plot_stats_dict['src_std']  = src_plot_std

    with open(os.path.join(cfg.run_dir, "error_data_stats" + ".pkl"), "wb") as f:
        pickle.dump(error_stats_dict, f)
    with open(os.path.join(cfg.run_dir, "plot_data_stats" + ".pkl"), "wb") as f:
        pickle.dump(plot_stats_dict, f)

    return error_stats_dict, plot_stats_dict


########################################################################################################################
# Testing ML-model.

def parametrized_simulation_test(cfg, model):
    model.net.eval()

    if cfg.model_name[:8] == "Ensemble":
        for m in range(len(model.nets)):
            model.nets[m].net.eval()
    else:
        model.net.eval()

    _, _, dataset_test = load_datasets(cfg, False, False, True)

    num_param_values = cfg.N_test_alphas
    stats      = dataset_test[:8][3].detach().numpy()
    ICs        = dataset_test[:][4].detach().numpy()
    times      = dataset_test[:][5].detach().numpy()
    alphas     = dataset_test[:num_param_values][6].detach().numpy()
    #print("alphas", alphas)

    unc_mean = stats[0]
    unc_std  = stats[4]
    ref_mean = stats[1]
    ref_std  = stats[5]
    res_mean = stats[2]
    res_std  = stats[6]
    src_mean = stats[3]
    src_std  = stats[7]

    L2_errors_unc = np.zeros((num_param_values, cfg.N_t - 1))
    L2_errors_cor = np.zeros((num_param_values, cfg.N_t - 1))
    Linfty_errors_unc = np.zeros((num_param_values, cfg.N_t - 1))
    Linfty_errors_cor = np.zeros((num_param_values, cfg.N_t - 1))

    num_profile_plots = cfg.profile_save_steps.shape[0]
    plot_data_dict = {
        'x': cfg.x_nodes,
        'y': cfg.y_nodes,
        'unc': np.zeros((num_param_values, num_profile_plots, cfg.x_nodes.shape[0], cfg.y_nodes.shape[0])),
        'ref': np.zeros((num_param_values, num_profile_plots, cfg.x_nodes.shape[0], cfg.y_nodes.shape[0])),
        'cor': np.zeros((num_param_values, num_profile_plots, cfg.x_nodes.shape[0], cfg.y_nodes.shape[0])),
        'time': np.zeros(num_profile_plots),
        'alphas': alphas
    }
    if cfg.model_type == 'hybrid' and cfg.exact_solution_available:
        plot_data_dict['src'] = np.zeros((num_param_values, num_profile_plots, cfg.N_x, cfg.N_y))

    for a, alpha in enumerate(alphas):
        IC = ICs[a * (cfg.N_t - 1)]
        #print("IC:", IC)
        old_unc = IC
        old_cor = IC
        plot_num = 0
        for i in range(cfg.N_t - 1):
            index = a * (cfg.N_t - 1) + i
            old_time = np.around(times[index] - cfg.dt, decimals=10)
            new_time = np.around(times[index], decimals=10)

            new_unc = physics.simulate_2D(
                cfg, old_unc, old_time, new_time, alpha, cfg.get_q_hat_approx, np.zeros((cfg.N_x, cfg.N_y))
            )
            #if i == 0:
            #    print("First unc:", new_unc)
            new_unc_ = physics.simulate_2D(
                cfg, old_cor, old_time, new_time, alpha, cfg.get_q_hat_approx, np.zeros((cfg.N_x, cfg.N_y))
            )
            new_unc_tensor_ = torch.from_numpy(util.z_normalize(new_unc_, unc_mean, unc_std))
            old_cor_tensor  = torch.from_numpy(util.z_normalize(old_cor,  ref_mean, ref_std))

            if cfg.exact_solution_available:
                new_ref = cfg.get_T_exact(cfg.x_nodes, cfg.y_nodes, new_time, alpha)
                #print("new_ref:", new_ref)
            else:
                raise Exception("Invalid config.")

            new_cor = np.zeros_like(new_unc)
            if cfg.model_type == 'hybrid':
                new_src = util.z_unnormalize(
                    model.net(new_unc_tensor_.to(cfg.device)).detach().cpu().numpy(), src_mean, src_std
                )
                new_cor = physics.simulate_2D(
                    cfg, old_cor, old_time, new_time, alpha, cfg.get_q_hat_approx, new_src
                )
            elif cfg.model_type == 'residual':
                new_res = np.zeros(new_unc.shape)
                unnomralized_res = model.net(new_unc_tensor_.to(cfg.device)).detach().cpu().numpy()
                new_res[1:-1, 1:-1] = util.z_unnormalize(model.net(new_unc_tensor_.to(cfg.device)).detach().cpu().numpy(), res_mean, res_std)
                new_cor = new_unc_ + new_res
                #print("unnormalized_res:", unnomralized_res)
                #print("new_res:", new_res)
                #print("new_unc_:", new_unc_)
                #print("new_cor:", new_cor)
            elif cfg.model_type == 'end-to-end':
                new_cor[0, :] = cfg.get_T_a(cfg.y_nodes, new_time, alpha)  # Since BCs are not ...
                new_cor[-1, :] = cfg.get_T_b(cfg.y_nodes, new_time, alpha)  # predicted by the NN.
                new_cor[:, 0] = cfg.get_T_c(cfg.x_nodes, new_time, alpha)
                new_cor[:, -1] = cfg.get_T_d(cfg.x_nodes, new_time, alpha)
                new_cor[1:-1, 1:-1] = util.z_unnormalize(model.net(new_unc_tensor_.to(cfg.device)).detach().cpu().numpy(), ref_mean, ref_std)
            elif cfg.model_type == 'data':
                new_cor[0, :] = cfg.get_T_a(cfg.y_nodes, new_time, alpha)  # Since BCs are not ...
                new_cor[-1, :] = cfg.get_T_b(cfg.y_nodes, new_time, alpha)  # predicted by the NN.
                new_cor[:, 0] = cfg.get_T_c(cfg.x_nodes, new_time, alpha)
                new_cor[:, -1] = cfg.get_T_d(cfg.x_nodes, new_time, alpha)
                new_cor[1:-1] = util.z_unnormalize(model.net(old_cor_tensor.to(cfg.device)).detach().cpu().numpy(), ref_mean, ref_std)

            #if i == 0:
            #    print("First cor:", new_cor)

            ref_norm_L2 = util.get_disc_L2_norm(new_ref)
            unc_error_norm_L2 = util.get_disc_L2_norm(new_unc - new_ref) / ref_norm_L2
            cor_error_norm_L2 = util.get_disc_L2_norm(new_cor - new_ref) / ref_norm_L2
            ref_norm_Linfty = util.get_disc_Linfty_norm(new_ref)
            unc_error_norm_Linfty = util.get_disc_Linfty_norm(new_unc - new_ref) / ref_norm_Linfty
            cor_error_norm_Linfty = util.get_disc_Linfty_norm(new_cor - new_ref) / ref_norm_Linfty

            L2_errors_unc[a][i] = unc_error_norm_L2
            L2_errors_cor[a][i] = cor_error_norm_L2
            Linfty_errors_unc[a][i] = unc_error_norm_Linfty
            Linfty_errors_cor[a][i] = cor_error_norm_Linfty

            if i in cfg.profile_save_steps:
                plot_data_dict['unc'][a][plot_num] = new_unc
                plot_data_dict['ref'][a][plot_num] = new_ref
                plot_data_dict['cor'][a][plot_num] = new_cor
                if cfg.model_type == 'hybrid' and cfg.exact_solution_available:
                    plot_data_dict['src'][a][plot_num] = new_src
                if a == 0:
                    plot_data_dict['time'][plot_num] = new_time
                plot_num += 1

            old_cor = new_cor
            old_unc = new_unc

    error_dict = {
        'unc_L2': L2_errors_unc,
        'cor_L2': L2_errors_cor,
        'unc_Linfty': Linfty_errors_unc,
        'cor_Linfty': Linfty_errors_cor,
        'alphas': alphas
    }

    return error_dict, plot_data_dict

########################################################################################################################

def main():
    pass

########################################################################################################################

if __name__ == "__main__":
    main()

########################################################################################################################