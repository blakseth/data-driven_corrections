"""
paper_plots.py

Written by Sindre Stenen Blakseth, 2021.

Script for producing the plots for HAM-CoSTA-paper.
"""

########################################################################################################################
# File imports.

import util

########################################################################################################################
# Package imports.

import os
import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib.colors as cs

plt.rcParams.update({
    "font.family": "DejaVu Serif",
    "font.serif": ["Computer Modern Roman"],
})

########################################################################################################################
# Exact solutions.

def get_exact_solution(system_number, alpha, t):
    if system_number == 1:
        def local_T(x, y, tt, alphaa):
            return tt + alphaa*x + y**2
    elif system_number == 2:
        def local_T(x, y, tt, alphaa):
            return (x + y + alphaa)/(tt + 1)
    elif system_number == 3:
        def local_T(x, y, tt, alphaa):
            if x <= 0.5:
                return y * np.exp(-tt) * (alphaa + 2 * x)
            else:
                return y * np.exp(-tt) * (alphaa + 0.75 + 0.5 * x)
    elif system_number == 4:
        def local_T(x, y, tt, alphaa):
            return alphaa + (tt + 1)*np.cos(2*np.pi*x)*np.cos(4*np.pi*y)
    else:
        print("No exact solution was chosen for system_number =", system_number)
        raise Exception

    def get_T_exact(x, y, tt, alphaa):
        if type(x) is np.ndarray and type(y) is np.ndarray:
            T = np.zeros((x.shape[0], y.shape[0]))
            for i, y_ in enumerate(y):
                for j, x_ in enumerate(x):
                    T[j, i] = local_T(x_, y_, tt, alphaa)
            return T
        elif type(x) is np.ndarray:
            T = np.zeros(x.shape[0])
            for j, x_ in enumerate(x):
                T[j] = local_T(x_, y, tt, alphaa)
            return T
        elif type(y) is np.ndarray:
            T = np.zeros(y.shape[0])
            for i, y_ in enumerate(y):
                T[i] = local_T(x, y_, tt, alphaa)
            return T
        else:
            return local_T(x, y, tt, alphaa)

    return lambda x,y: get_T_exact(x, y, t, alpha)

########################################################################################################################
# Visualizing data.

def visualize_error_data(iterations, unc_errors, end_errors, hyb_errors, output_dir, filename):
    plt.figure()
    plt.semilogy(iterations, unc_errors, 'r-', linewidth=2.0, label="Uncorrected")
    plt.semilogy(iterations, end_errors, 'b-', linewidth=2.0, label="End-to-end")
    plt.semilogy(iterations, hyb_errors, 'g-', linewidth=2.0, label="CoSTA")
    plt.xlim([0, len(unc_errors)])
    plt.xlabel("Test Iterations", fontsize=20)
    plt.ylabel(r"Relative $l_2$ Error", fontsize=20)
    plt.xticks(fontsize=17)
    plt.yticks(fontsize=17)
    plt.grid()
    plt.legend(prop={'size': 17})
    plt.savefig(os.path.join(output_dir, filename + ".pdf"), bbox_inches='tight')
    plt.close()

def visualize_profile(x, unc_profile, end_profile, hyb_profile, exact_callable, output_dir, filename):
    plt.figure()
    plt.scatter(x, unc_profile, s=40, facecolors='none', edgecolors='r', label="Uncorrected")
    plt.scatter(x, end_profile, s=40, facecolors='none', edgecolors='b', label="End-to-end")
    plt.scatter(x, hyb_profile, s=40, facecolors='none', edgecolors='g', label="CoSTA")
    x_dense = np.linspace(x[0], x[-1], 1001, endpoint=True)
    plt.plot(x_dense, exact_callable(x_dense), 'k-', linewidth=2.0, label="Exact")
    plt.xlim(x[0], x[-1])
    plt.xlabel(r"$x$ (m)", fontsize=20)
    plt.ylabel(r"$T$ (K)", fontsize=20)
    plt.xticks(fontsize=17)
    plt.yticks(fontsize=17)
    plt.grid()
    plt.legend(prop={'size': 17})
    plt.savefig(os.path.join(output_dir, filename + ".pdf"), bbox_inches='tight')
    plt.close()

def visualize_error_data_combined(iterations, unc_errors, end_errors_FCNN, end_errors_CNN, hyb_errors_FCNN, hyb_errors_CNN, res_errors_FCNN, res_errors_CNN, dat_errors_FCNN, dat_errors_CNN, output_dir, filename, y_lim):
    plt.figure()
    plt.semilogy(iterations, unc_errors, 'r-', linewidth=2.0, label="PBM")
    if dat_errors_FCNN is not None:
        plt.semilogy(iterations, dat_errors_FCNN, 'b-',  linewidth=2.0, label="DDM")
    if dat_errors_CNN is not None:
        plt.semilogy(iterations, dat_errors_CNN,  'y--', linewidth=2.0, label="DDM CNN")
    if end_errors_FCNN is not None:
        plt.semilogy(iterations, end_errors_FCNN, 'c-',  linewidth=2.0, label="End-to-end FCNN")
    if end_errors_CNN is not None:
        plt.semilogy(iterations, end_errors_CNN,  'c--', linewidth=2.0, label="End-to-end CNN")
    if hyb_errors_FCNN is not None:
        plt.semilogy(iterations, hyb_errors_FCNN, 'g-',  linewidth=2.0, label="HAM")
    if hyb_errors_CNN is not None:
        plt.semilogy(iterations, hyb_errors_CNN,  'g--', linewidth=2.0, label="CoSTA CNN")
    if res_errors_FCNN is not None:
        plt.semilogy(iterations, res_errors_FCNN, 'y-',  linewidth=2.0, label="Residual FCNN")
    if res_errors_CNN is not None:
        plt.semilogy(iterations, res_errors_CNN,  'y--', linewidth=2.0, label="Residual CNN")
    plt.xlim([0, len(unc_errors)])
    plt.ylim(y_lim)
    plt.xlabel("Test Iterations", fontsize=20)
    plt.ylabel(r"Relative $l_2$ Error", fontsize=20)
    plt.xticks(fontsize=17)
    plt.yticks(fontsize=17)
    plt.grid()
    #plt.legend(prop={'size': 17})
    plt.savefig(os.path.join(output_dir, filename + ".pdf"), bbox_inches='tight')
    plt.close()

def visualize_profile_combined(x, y, PBM_field, DDM_field, HAM_field, exact_callable, output_dir, filename):
    x_a = x[0]
    x_b = x[-1]
    y_c = y[0]
    y_d = y[-1]
    dx  = x[2] - x[1]
    dy  = y[2] - y[1]

    exact_field = exact_callable(x, y)
    x_dense = np.linspace(x_a, x_b, num=200, endpoint=True)
    y_dense = np.linspace(y_c, y_d, num=200, endpoint=True)
    exact_field_dense = exact_callable(x_dense, y_dense)

    PBM_diff_field = PBM_field - exact_field
    DDM_diff_field = DDM_field - exact_field
    HAM_diff_field = HAM_field - exact_field

    maxmax = np.amax(np.asarray([np.amax(np.abs(PBM_diff_field)), np.amax(np.abs(DDM_diff_field)), np.amax(np.abs(HAM_diff_field))]))
    minmin = -maxmax

    threshold = 1e-4

    fig, axs = plt.subplots(2, 2)

    surf_map = plt.get_cmap('plasma')

    surf = axs[0, 0].contourf(x_dense, y_dense, np.swapaxes(exact_field_dense, 0, 1), levels=100, cmap=surf_map)
    for c in surf.collections:
        c.set_edgecolor("face")
    axs[0, 0].set_title('Exact')

    # sample the colormaps that you want to use. Use 128 from each so we get 256
    # colors in total
    colors1 = plt.cm.hot(np.linspace(0, 1, 128))
    colorsmid = plt.cm.RdGy(np.linspace(0.5, 0.6, 25))
    colors2 = plt.cm.twilight(np.linspace(0, 0.5, 103))

    # combine them and build a new colormap
    colors = np.vstack((colors1, colorsmid, colors2))
    mymap = cs.LinearSegmentedColormap.from_list('my_colormap', colors)

    diff_map = plt.get_cmap('seismic')

    im2 = axs[0, 1].imshow(np.flip(np.swapaxes(PBM_diff_field, 0, 1), 0), norm=cs.SymLogNorm(threshold), vmin=minmin, vmax=maxmax,
                          extent=[x_a - 0.5 * dx, x_b + 0.5 * dx,
                                  y_c - 0.5 * dy, y_d + 0.5 * dy], cmap= mymap)
    axs[0, 1].set_title('PBM')

    im3 = axs[1, 0].imshow(np.flip(np.swapaxes(DDM_diff_field, 0, 1), 0), norm=cs.SymLogNorm(threshold), vmin=minmin, vmax=maxmax,
                     extent=[x_a - 0.5 * dx, x_b + 0.5 * dx,
                             y_c - 0.5 * dy, y_d + 0.5 * dy], cmap= mymap)
    axs[1, 0].set_title('DDM')
    im4 = axs[1, 1].imshow(np.flip(np.swapaxes(HAM_diff_field, 0, 1), 0), norm=cs.SymLogNorm(threshold), vmin=minmin, vmax=maxmax,
                     extent=[x_a - 0.5 * dx, x_b + 0.5 * dx,
                             y_c - 0.5 * dy, y_d + 0.5 * dy], cmap= mymap)
    axs[1, 1].set_title('HAM')
    for ax in fig.get_axes():
        ax.set_xlim((x_a, x_b))
        ax.set_ylim((y_c, y_d))
        ax.set_xlabel(r'$x$ (m)')
        ax.set_ylabel(r'$y$ (m)')
        ax.label_outer()
    fig.colorbar(surf, ax=axs[0, 0])
    fig.colorbar(im2,  ax=axs[0, 1])
    fig.colorbar(im3,  ax=axs[1, 0])
    fig.colorbar(im4,  ax=axs[1, 1])
    plt.savefig(os.path.join(output_dir, filename + "_1.pdf"), bbox_inches='tight')
    plt.close()


    ########################################

    fig, axs = plt.subplots(1, 3)

    # sample the colormaps that you want to use. Use 128 from each so we get 256
    # colors in total
    colors1 = plt.cm.hot(np.linspace(0, 1, 128))
    colorsmid = plt.cm.RdGy(np.linspace(0.5, 0.6, 25))
    colors2 = plt.cm.twilight(np.linspace(0, 0.5, 103))

    # combine them and build a new colormap
    colors = np.vstack((colors1, colorsmid, colors2))
    mymap = cs.LinearSegmentedColormap.from_list('my_colormap', colors)

    diff_map = plt.get_cmap('seismic')

    im1 = axs[0].imshow(np.flip(np.swapaxes(PBM_diff_field, 0, 1), 0), norm=cs.SymLogNorm(threshold), vmin=minmin,
                           vmax=maxmax,
                           extent=[x_a - 0.5 * dx, x_b + 0.5 * dx,
                                   y_c - 0.5 * dy, y_d + 0.5 * dy], cmap=mymap)
    axs[0].set_title('PBM')

    im2 = axs[1].imshow(np.flip(np.swapaxes(DDM_diff_field, 0, 1), 0), norm=cs.SymLogNorm(threshold), vmin=minmin,
                           vmax=maxmax,
                           extent=[x_a - 0.5 * dx, x_b + 0.5 * dx,
                                   y_c - 0.5 * dy, y_d + 0.5 * dy], cmap=mymap)
    axs[1].set_title('DDM')

    im3 = axs[2].imshow(np.flip(np.swapaxes(HAM_diff_field, 0, 1), 0), norm=cs.SymLogNorm(threshold), vmin=minmin,
                           vmax=maxmax,
                           extent=[x_a - 0.5 * dx, x_b + 0.5 * dx,
                                   y_c - 0.5 * dy, y_d + 0.5 * dy], cmap=mymap)
    axs[2].set_title('HAM')
    for ax in fig.get_axes():
        ax.set_xlim((x_a, x_b))
        ax.set_ylim((y_c, y_d))
        ax.set_xlabel(r'$x$ (m)')
        ax.set_ylabel(r'$y$ (m)')
        ax.label_outer()
    fig.colorbar(im3, ax=axs.ravel().tolist(), shrink=.4)
    plt.savefig(os.path.join(output_dir, filename + "_2.pdf"), bbox_inches='tight')
    plt.close()

    ################################

    maxmax = np.amax(
        np.asarray([np.amax(PBM_field), np.amax(DDM_field), np.amax(HAM_field), np.amax(exact_field)]))
    minmin = np.amin(
        np.asarray([np.amin(PBM_field), np.amin(DDM_field), np.amin(HAM_field), np.amin(exact_field)]))

    threshold = 1e-4

    fig, axs = plt.subplots(2, 2)

    surf_map = plt.get_cmap('plasma')

    surf = axs[0, 0].contourf(x_dense, y_dense, np.swapaxes(exact_field_dense, 0, 1), vmin=minmin, vmax=maxmax, levels=100, cmap=surf_map)
    for c in surf.collections:
        c.set_edgecolor("face")
    axs[0, 0].set_title('Exact')

    # sample the colormaps that you want to use. Use 128 from each so we get 256
    # colors in total
    colors1 = plt.cm.hot(np.linspace(0, 1, 128))
    colorsmid = plt.cm.RdGy(np.linspace(0.5, 0.6, 25))
    colors2 = plt.cm.twilight(np.linspace(0, 0.5, 103))

    # combine them and build a new colormap
    colors = np.vstack((colors1, colorsmid, colors2))
    mymap = cs.LinearSegmentedColormap.from_list('my_colormap', colors)

    diff_map = plt.get_cmap('seismic')

    print("Exact:", exact_field)
    print("HAM:", HAM_field)

    im2 = axs[0, 1].imshow(np.flip(np.swapaxes(PBM_field, 0, 1), 0), vmin=minmin,
                           vmax=maxmax,
                           extent=[x_a - 0.5 * dx, x_b + 0.5 * dx,
                                   y_c - 0.5 * dy, y_d + 0.5 * dy], cmap=surf_map)
    axs[0, 1].set_title('PBM')

    im3 = axs[1, 0].imshow(np.flip(np.swapaxes(DDM_field, 0, 1), 0), vmin=minmin,
                           vmax=maxmax,
                           extent=[x_a - 0.5 * dx, x_b + 0.5 * dx,
                                   y_c - 0.5 * dy, y_d + 0.5 * dy], cmap=surf_map)
    axs[1, 0].set_title('DDM')
    im4 = axs[1, 1].imshow(np.flip(np.swapaxes(HAM_field, 0, 1), 0), vmin=minmin,
                           vmax=maxmax,
                           extent=[x_a - 0.5 * dx, x_b + 0.5 * dx,
                                   y_c - 0.5 * dy, y_d + 0.5 * dy], cmap=surf_map)
    axs[1, 1].set_title('HAM')
    for ax in fig.get_axes():
        ax.set_xlim((x_a, x_b))
        ax.set_ylim((y_c, y_d))
        ax.set_xlabel(r'$x$ (m)')
        ax.set_ylabel(r'$y$ (m)')
        ax.label_outer()
    fig.colorbar(im3, ax=axs.ravel().tolist())
    plt.savefig(os.path.join(output_dir, filename + "_3.pdf"), bbox_inches='tight')
    plt.close()
    """
    plt.figure()
    plt.scatter(x, unc_profile, s=40, facecolors='none', edgecolors='r', label="PBM")
    if dat_profile_FCNN is not None:
        plt.scatter(x, dat_profile_FCNN, s=40, marker='s', facecolors='none', edgecolors='b', label="DDM")
    if dat_profile_CNN is not None:
        plt.scatter(x, dat_profile_CNN,  s=40, marker='^', facecolors='none', edgecolors='b', label="DDM CNN")
    if end_profile_FCNN is not None:
        plt.scatter(x, end_profile_FCNN, s=40, marker='o', facecolors='none', edgecolors='b', label="End-to-end FCNN")
    if end_profile_CNN is not None:
        plt.scatter(x, end_profile_CNN,  s=40, marker='^', facecolors='none', edgecolors='b', label="End-to-end CNN")
    if hyb_profile_FCNN is not None:
        plt.scatter(x, hyb_profile_FCNN, s=40, marker='D', facecolors='none', edgecolors='g', label="HAM")
    if hyb_profile_CNN is not None:
        plt.scatter(x, hyb_profile_CNN,  s=40, marker='^', facecolors='none', edgecolors='g', label="CoSTA CNN")
    if res_profile_FCNN is not None:
        plt.scatter(x, res_profile_FCNN, s=40, marker='o', facecolors='none', edgecolors='y', label="Residual FCNN")
    if res_profile_CNN is not None:
        plt.scatter(x, res_profile_CNN,  s=40, marker='^', facecolors='none', edgecolors='y', label="Residual CNN")
    x_dense = np.linspace(x[0], x[-1], 1001, endpoint=True)
    plt.plot(x_dense, exact_callable(x_dense), 'k-', linewidth=2.0, label="Exact")
    plt.xlim(x[0], x[-1])
    plt.xlabel(r"$x$ (m)", fontsize=20)
    plt.ylabel(r"$T$ (K)", fontsize=20)
    plt.xticks(fontsize=17)
    plt.yticks(fontsize=17)
    plt.grid()
    #plt.legend(prop={'size': 17})
    plt.savefig(os.path.join(output_dir, filename + ".pdf"), bbox_inches='tight')
    plt.close()
    """

########################################################################################################################

def main():
    hybrid_CNN_dir  = ""
    hybrid_FCNN_dir = "/home/sindre/msc_thesis/data-driven_corrections/results/2021-04-26_2Dk_HAM/2D_GlobalDense_k"
    end_CNN_dir     = ""
    end_FCNN_dir    = ""
    res_CNN_dir     = ""
    res_FCNN_dir    = ""
    dat_CNN_dir     = ""
    dat_FCNN_dir    = "/home/sindre/msc_thesis/data-driven_corrections/results/2021-04-26_2Dk_DDM/2D_GlobalDense_k"
    output_dir      = "/home/sindre/msc_thesis/data-driven_corrections/thesis_figures/2Dk_V4"

    use_CNN_results   = False
    use_FCNN_results  = True
    use_end_results   = False
    use_hyb_results   = True
    use_res_results   = False
    use_dat_results   = True

    os.makedirs(output_dir, exist_ok=False)

    num_systems_studied = 14
    systems_to_include = [1, 2, 3, 4]

    y_lims_interp = [[1e-9, 1e-2], [1e-9, 1e2], [5e-9, 1e2], [7e-9, 1e2]]
    y_lims_extrap = [[1e-9, 1e2],  [1e-9, 1e2], [5e-9, 3e2], [7e-9, 1e2]]

    for s in range(num_systems_studied):
        system_number = s + 1
        if system_number not in systems_to_include:
            continue


        # Plotting profiles.

        if use_FCNN_results and use_hyb_results:
            with open(os.path.join(hybrid_FCNN_dir + str(system_number) + "_HAM", "plot_data_stats.pkl"), "rb") as f:
                hybrid_FCNN_plot_dict = pickle.load(f)
        if use_FCNN_results and use_end_results:
            with open(os.path.join(end_FCNN_dir + str(system_number), "plot_data_stats.pkl"), "rb") as f:
                end_FCNN_plot_dict = pickle.load(f)
        if use_FCNN_results and use_res_results:
            with open(os.path.join(res_FCNN_dir + str(system_number), "plot_data_stats.pkl"), "rb") as f:
                res_FCNN_plot_dict = pickle.load(f)
        if use_FCNN_results and use_dat_results:
            with open(os.path.join(dat_FCNN_dir + str(system_number) + "_DDM", "plot_data_stats.pkl"), "rb") as f:
                dat_FCNN_plot_dict = pickle.load(f)
        print("Successfully loaded FCNN plot dicts.")

        if use_CNN_results and use_hyb_results:
            with open(os.path.join(hybrid_CNN_dir + str(system_number), "plot_data_stats.pkl"), "rb") as f:
                hybrid_CNN_plot_dict = pickle.load(f)
        if use_CNN_results and use_end_results:
            with open(os.path.join(end_CNN_dir + str(system_number), "plot_data_stats.pkl"), "rb") as f:
                end_CNN_plot_dict = pickle.load(f)
        if use_CNN_results and use_res_results:
            with open(os.path.join(res_CNN_dir + str(system_number), "plot_data_stats.pkl"), "rb") as f:
                res_CNN_plot_dict = pickle.load(f)
        if use_CNN_results and use_dat_results:
            with open(os.path.join(dat_CNN_dir + str(system_number), "plot_data_stats.pkl"), "rb") as f:
                dat_CNN_plot_dict = pickle.load(f)
        print("Successfully loaded CNN plot dicts.")

        alphas     = hybrid_FCNN_plot_dict['alphas']
        plot_times = hybrid_FCNN_plot_dict['time']
        plot_times = [plot_times[2], plot_times[-1]]
        x          = hybrid_FCNN_plot_dict['x']
        y          = hybrid_FCNN_plot_dict['x'] # Only works since I have used a square grid in all experiments.

        print("Plot times:", plot_times)

        #np.testing.assert_allclose(alphas, end_FCNN_plot_dict['alphas'], rtol=1e-10, atol=1e-10)
        #np.testing.assert_allclose(plot_times, end_FCNN_plot_dict['time'], rtol=1e-10, atol=1e-10)
        #np.testing.assert_allclose(alphas, end_CNN_plot_dict['alphas'], rtol=1e-10, atol=1e-10)
        #np.testing.assert_allclose(alphas, hybrid_CNN_plot_dict['alphas'], rtol=1e-10, atol=1e-10)
        #np.testing.assert_allclose(plot_times, end_CNN_plot_dict['time'], rtol=1e-10, atol=1e-10)
        #np.testing.assert_allclose(plot_times, hybrid_CNN_plot_dict['time'], rtol=1e-10, atol=1e-10)

        plot_dir = os.path.join(output_dir, "plots")
        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir, exist_ok=True)

        for a, alpha in enumerate(alphas):
            for plot_num, t in enumerate(plot_times):
                unc_profile = hybrid_FCNN_plot_dict['unc'][a][plot_num]
                #np.testing.assert_allclose(unc_profile, end_FCNN_plot_dict['unc'][a][plot_num], rtol=1e-10, atol=1e-10)
                #np.testing.assert_allclose(unc_profile, end_CNN_plot_dict['unc'][a][plot_num], rtol=1e-10, atol=1e-10)
                #np.testing.assert_allclose(unc_profile, hybrid_CNN_plot_dict['unc'][a][plot_num], rtol=1e-10, atol=1e-10)

                if use_FCNN_results and use_end_results:
                    end_profile_FCNN = end_FCNN_plot_dict['cor_mean'][a][plot_num]
                else:
                    end_profile_FCNN = None
                if use_FCNN_results and use_hyb_results:
                    hyb_profile_FCNN = hybrid_FCNN_plot_dict['cor_mean'][a][plot_num]
                else:
                    hyb_profile_FCNN = None
                if use_FCNN_results and use_res_results:
                    res_profile_FCNN = res_FCNN_plot_dict['cor_mean'][a][plot_num]
                else:
                    res_profile_FCNN = None
                if use_FCNN_results and use_dat_results:
                    dat_profile_FCNN = dat_FCNN_plot_dict['cor_mean'][a][plot_num]
                else:
                    dat_profile_FCNN = None
                if use_CNN_results and use_end_results:
                    end_profile_CNN = end_CNN_plot_dict['cor_mean'][a][plot_num]
                else:
                    end_profile_CNN = None
                if use_CNN_results and use_hyb_results:
                    hyb_profile_CNN = hybrid_CNN_plot_dict['cor_mean'][a][plot_num]
                else:
                    hyb_profile_CNN = None
                if use_CNN_results and use_res_results:
                    res_profile_CNN = res_CNN_plot_dict['cor_mean'][a][plot_num]
                else:
                    res_profile_CNN = None
                if use_CNN_results and use_dat_results:
                    dat_profile_CNN = dat_CNN_plot_dict['cor_mean'][a][plot_num]
                else:
                    dat_profile_CNN = None

                exact_callable = get_exact_solution(system_number, alpha, t)

                #np.testing.assert_allclose(hybrid_FCNN_plot_dict['ref'][a][plot_num], exact_callable(x), rtol=1e-10, atol=1e-10)
                #np.testing.assert_allclose(end_FCNN_plot_dict['ref'][a][plot_num], exact_callable(x), rtol=1e-10,
                #                           atol=1e-10)
                #np.testing.assert_allclose(res_FCNN_plot_dict['ref'][a][plot_num], exact_callable(x), rtol=1e-10,
                #                           atol=1e-10)

                filename = "profiles_s" + str(system_number) + "_alpha" + str(np.around(alpha, decimals=5)) + "_time" + str(np.around(t, decimals=5))
                visualize_profile_combined(x, y, PBM_field=unc_profile, DDM_field=dat_profile_FCNN, HAM_field=hyb_profile_FCNN, exact_callable=exact_callable, output_dir=plot_dir, filename=filename)
                print("Successfully plotted profiles for system " + str(system_number) + ", alpha" + str(np.around(alpha, decimals=5)) + ", time" + str(np.around(t, decimals=5)))


        # Plotting l2 errors.

        if use_FCNN_results and use_hyb_results:
            with open(os.path.join(hybrid_FCNN_dir  + str(system_number) + "_HAM", "error_data_stats.pkl"), "rb") as f:
                hybrid_FCNN_error_dict = pickle.load(f)
        if use_FCNN_results and use_end_results:
            with open(os.path.join(end_FCNN_dir + str(system_number), "error_data_stats.pkl"), "rb") as f:
                end_FCNN_error_dict = pickle.load(f)
        if use_FCNN_results and use_res_results:
            with open(os.path.join(res_FCNN_dir + str(system_number), "error_data_stats.pkl"), "rb") as f:
                res_FCNN_error_dict = pickle.load(f)
        if use_FCNN_results and use_dat_results:
            with open(os.path.join(dat_FCNN_dir + str(system_number) + "_DDM", "error_data_stats.pkl"), "rb") as f:
                dat_FCNN_error_dict = pickle.load(f)
        print("Successfully loaded FCNN error dicts.")

        if use_CNN_results and use_hyb_results:
            with open(os.path.join(hybrid_CNN_dir + str(system_number), "error_data_stats.pkl"), "rb") as f:
                hybrid_CNN_error_dict = pickle.load(f)
        if use_CNN_results and use_end_results:
            with open(os.path.join(end_CNN_dir + str(system_number), "error_data_stats.pkl"), "rb") as f:
                end_CNN_error_dict = pickle.load(f)
        if use_CNN_results and use_res_results:
            with open(os.path.join(res_CNN_dir + str(system_number), "error_data_stats.pkl"), "rb") as f:
                res_CNN_error_dict = pickle.load(f)
        if use_CNN_results and use_dat_results:
            with open(os.path.join(dat_CNN_dir + str(system_number), "error_data_stats.pkl"), "rb") as f:
                dat_CNN_error_dict = pickle.load(f)
        print("Successfully loaded CNN error dicts.")

        #assert alphas.shape[0] == hybrid_FCNN_error_dict['unc_L2'].shape[0]
        #assert alphas.shape[0] == end_FCNN_error_dict['unc_L2'].shape[0]
        #assert hybrid_FCNN_error_dict['unc_L2'].shape[1] == end_FCNN_error_dict['unc_L2'].shape[1]
        #assert alphas.shape[0] == hybrid_FCNN_error_dict['unc_L2'].shape[0]
        #assert alphas.shape[0] == end_FCNN_error_dict['unc_L2'].shape[0]
        #assert hybrid_CNN_error_dict['unc_L2'].shape[1] == end_CNN_error_dict['unc_L2'].shape[1]

        error_dir = os.path.join(output_dir, "errors")
        if not os.path.exists(error_dir):
            os.makedirs(error_dir, exist_ok=False)


        iterations = np.arange(1, hybrid_FCNN_error_dict['unc_L2'].shape[1] + 1, 1)
        for a, alpha in enumerate(alphas):
            unc_errors = hybrid_FCNN_error_dict['unc_L2'][a]
            #np.testing.assert_allclose(unc_errors, end_FCNN_error_dict['unc_L2'][a], rtol=1e-10, atol=1e-10)
            #np.testing.assert_allclose(unc_errors, hybrid_CNN_error_dict['unc_L2'][a], rtol=1e-10, atol=1e-10)
            #np.testing.assert_allclose(unc_errors, end_CNN_error_dict['unc_L2'][a], rtol=1e-10, atol=1e-10)

            if use_FCNN_results and use_end_results:
                end_errors_FCNN = end_FCNN_error_dict['cor_mean_L2'][a]
            else:
                end_errors_FCNN = None
            if use_FCNN_results and use_hyb_results:
                hyb_errors_FCNN = hybrid_FCNN_error_dict['cor_mean_L2'][a]
            else:
                hyb_errors_FCNN = None
            if use_FCNN_results and use_res_results:
                res_errors_FCNN = res_FCNN_error_dict['cor_mean_L2'][a]
            else:
                res_errors_FCNN = None
            if use_FCNN_results and use_dat_results:
                dat_errors_FCNN = dat_FCNN_error_dict['cor_mean_L2'][a]
            else:
                dat_errors_FCNN = None
            if use_CNN_results and use_end_results:
                end_errors_CNN = end_CNN_error_dict['cor_mean_L2'][a]
            else:
                end_errors_CNN = None
            if use_CNN_results and use_hyb_results:
                hyb_errors_CNN = hybrid_CNN_error_dict['cor_mean_L2'][a]
            else:
                hyb_errors_CNN = None
            if use_CNN_results and use_res_results:
                res_errors_CNN = res_CNN_error_dict['cor_mean_L2'][a]
            else:
                res_errors_CNN = None
            if use_CNN_results and use_dat_results:
                dat_errors_CNN = dat_CNN_error_dict['cor_mean_L2'][a]
            else:
                dat_errors_CNN = None

            if 0.0 < alpha < 2.2:
                y_lims = y_lims_interp
            else:
                y_lims = y_lims_extrap

            filename = "errors_s" + str(system_number) + "_alpha" + str(np.around(alpha, decimals=5))
            visualize_error_data_combined(iterations, unc_errors, end_errors_FCNN, end_errors_CNN, hyb_errors_FCNN, hyb_errors_CNN, res_errors_FCNN, res_errors_CNN, dat_errors_FCNN, dat_errors_CNN, error_dir, filename, y_lims[s])
            print("Successfully plotted FCNN errors for system " + str(system_number) + ", alpha" + str(np.around(alpha, decimals=5)))

        """
        print("alpha:", alphas[1])
        last_unc = hybrid_FCNN_plot_dict['unc'][1][-1]
        last_ref = hybrid_FCNN_plot_dict['ref'][1][-1]
        last_hyb = hybrid_FCNN_plot_dict['cor_mean'][1][-1]
        last_end = end_FCNN_plot_dict['cor_mean'][1][-1]
        last_res = res_FCNN_plot_dict['cor_mean'][1][-1]
        print("last_unc", last_unc)
        print("last_ref", last_ref)
        print("Last_hyb", last_hyb)
        print("Last_end", last_end)
        print("Last_res", last_res)
        ref_norm = util.get_disc_L2_norm(last_ref)
        unc_error = util.get_disc_L2_norm(last_unc - last_ref) / ref_norm
        hyb_error = util.get_disc_L2_norm(last_hyb - last_ref) / ref_norm
        end_error = util.get_disc_L2_norm(last_end - last_ref) / ref_norm
        res_error = util.get_disc_L2_norm(last_res - last_ref) / ref_norm
        print("unc_error calculated", unc_error)
        print("unc_error recorded", hybrid_FCNN_error_dict['unc_L2'][1][-1])
        print("hyb_error calculated", hyb_error)
        print("hyb_error recorded", hybrid_FCNN_error_dict['cor_mean_L2'][1][-1])
        print("end_error calculated", end_error)
        print("end_error recorded", end_FCNN_error_dict['cor_mean_L2'][1][-1])
        print("res_error calculated", res_error)
        print("res_error recorded", res_FCNN_error_dict['cor_mean_L2'][1][-1])

        with open(os.path.join(end_FCNN_dir + str(system_number), "plot_data_raw.pkl"), "rb") as f:
            raw_end_FCNN_plot_dicts = pickle.load(f)

        raw_profiles = []
        errors = []
        for i, raw_plot_dict in enumerate(raw_end_FCNN_plot_dicts):
            raw_profile = raw_plot_dict['cor'][1][-1]
            error = util.get_disc_L2_norm(raw_profile - last_ref) / ref_norm
            raw_profiles.append(raw_profile)
            errors.append(error)
            print("Profile " + str(i) + ":", raw_profile)
            print("Error " + str(i) + ":", error)
        raw_profiles = np.asarray(raw_profiles)
        errors = np.asarray(errors)
        mean_profile = np.zeros(raw_profiles.shape[1])
        for j in range(raw_profiles.shape[1]):
            for i in range(raw_profiles.shape[0]):
                mean_profile[j] += (raw_profiles[i][j] / raw_profiles.shape[0])
        mean_error = 0.0
        for i in range(errors.shape[0]):
            mean_error += (errors[i] / errors.shape[0])
        print("Calculated mean error:", mean_error)
        print("Recorded mean error:", end_FCNN_error_dict['cor_mean_L2'][1][-1])
        print("Calculated mean profile:", mean_profile)
        print("Recorded mean profile:", end_FCNN_plot_dict['cor_mean'][1][-1])
        print("Error of mean profile:", util.get_disc_L2_norm(mean_profile - last_ref) / ref_norm)

        plt.figure()
        for i in range(raw_profiles.shape[0]):
            print("x:", x)
            print("raw_profiles", raw_profiles[i])
            plt.plot(x, raw_profiles[i], label=str(i))
        plt.plot(x, mean_profile, label='mean')
        plt.legend()
        plt.show()
        """


########################################################################################################################

if __name__ == "__main__":
    main()

########################################################################################################################