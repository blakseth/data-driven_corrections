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
import matplotlib as mpl
from scipy.integrate import fixed_quad

plt.rcParams.update({
    "font.family": "DeJavu Serif",
    "font.serif": ["Computer Modern Roman"],
    "mathtext.fontset": 'cm'
})

########################################################################################################################
# Exact solutions.

def get_exact_solution(system_number, alpha, t):
    if system_number == 1:
        return lambda x: t + alpha*x

    if system_number == 2:
        return lambda x: 5 - alpha*x/(1 + t)

    if system_number == 3:
        def get_T_exact(x, tt, alphaa):
            if type(x) == np.ndarray:
                T = np.zeros_like(x)
                for x_index, x_val in enumerate(x):
                    if x_val <= 0.5:
                        T[x_index] = alphaa + 2 * x_val
                    else:
                        T[x_index] = alphaa + 0.75 + 0.5 * x_val
            else:
                if x <= 0.5:
                    T = alphaa + 2 * x
                else:
                    T = alphaa + 0.75 + 0.5 * x
            return np.exp(-tt) * T
        return lambda x: get_T_exact(x, t, alpha)

    if system_number == 4:
        return lambda x: alpha*(t + 1)*np.cos(2*np.pi*x)

    if system_number == 5:
        return lambda x: alpha*t*(x+1)

    if system_number == 6:
        return lambda x: (x + alpha)/(t + 1)

    if system_number == 7:
        return lambda x: 4*(x**3) - 4*(x**2) + alpha*(t + 1)

    print("No exact solution was chosen for system_number =", system_number)

def get_exact_conductivity(system_number, alpha, t):
    if system_number == 1:
        return lambda x: 1 + x

    if system_number == 2:
        return get_exact_solution(system_number, alpha, t)

    if system_number == 3:
        def get_k_exact(x, tt, alphaa):
            if type(x) == np.ndarray:
                k = np.ones_like(x)
                for i in range(k.shape[0]):
                    if x[i] <= 0.5:
                        k[i] /= 2.0
                    else:
                        k[i] *= 2.0
            else:
                if x <= 0.5:
                    k = 0.5
                else:
                    k = 2.0
            return k
        return lambda x: get_k_exact(x, t, alpha)

    if system_number == 4:
        return lambda x: np.sin(2*np.pi*x)

    if system_number == 5:
        return get_exact_solution(system_number, alpha, t)

    if system_number == 6:
        return get_exact_solution(system_number, alpha, t)

    if system_number == 7:
        return lambda x: 1 + x

    print("No exact solution was chosen for system_number =", system_number)

########################################################################################################################
# Visualizing data.

def visualize_error_data(iterations, unc_errors, end_errors, hyb_errors, output_dir, filename):
    plt.figure()
    plt.semilogy(iterations, unc_errors, 'r-', linewidth=2.0, label="Uncorrected")
    plt.semilogy(iterations, end_errors, 'b-', linewidth=2.0, label="End-to-end")
    plt.semilogy(iterations, hyb_errors, 'g-', linewidth=2.0, label="CoSTA")
    plt.xlim([0, len(unc_errors)])
    plt.xlabel("Test Iterations", fontsize=20)
    plt.ylabel(r"Relative $\mathcal{l}_2$ Error", fontsize=20)
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

def visualize_profile_combined(x, unc_profile, end_profile_FCNN, end_profile_CNN, hyb_profile_FCNN, hyb_profile_CNN, res_profile_FCNN, res_profile_CNN, dat_profile_FCNN, dat_profile_CNN, exact_callable, output_dir, filename):
    plt.figure()

    plt.scatter(x, unc_profile, s=35, facecolors='none', edgecolors='r', label="PBM")
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
    """
    plt.plot(x, unc_profile, 'r-', linewidth=6.0, label="PBM")
    plt.plot(x, hyb_profile_FCNN, 'g-', linewidth=3.7, label="HAM")
    plt.plot(x, dat_profile_FCNN, 'b-', linewidth=2.0, label="DDM")
    """
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

def visualize_src_terms(x, src, alpha, output_dir, filename):
    plt.figure()
    plt.scatter(x, src, s=40, marker='D', facecolor='none', edgecolors='g', label=r"$\hat{\sigma}$")
    if alpha is not None:
        plt.plot(x, np.ones_like(x) - alpha, 'k-', linewidth=2.0, label=r"$1-\alpha$")
    plt.xlabel(r"$x$ (m)", fontsize=20)
    plt.ylabel(r"$\hat{\sigma}\ \left(\mathrm{J/m}^3\right)$", fontsize=20)
    plt.xticks(fontsize=17)
    plt.yticks(fontsize=17)
    plt.grid()
    # plt.legend(prop={'size': 17})
    plt.savefig(os.path.join(output_dir, filename + ".pdf"), bbox_inches='tight')
    plt.close()

def visualize_conductivity(nodes, src, T, k_callable, output_dir, filename):
    poly_deg = 3
    # Assumes equidistant grid.
    src_coeffs = np.polyfit(nodes[1:-1], src, poly_deg)
    T_coeffs   = np.polyfit(nodes,       T,   poly_deg)
    def get_poly(x, coeffs):
        poly = np.zeros_like(x)
        for i in range(coeffs.shape[0]):
            poly += (x**i)*coeffs[-1 - i]
        return poly

    eps_k    = np.zeros_like(nodes)
    T_poly   = np.poly1d(T_coeffs)
    src_poly = np.poly1d(src_coeffs)
    T_sample = T_poly(nodes)

    #src_extended = np.zeros_like(nodes)
    #src_extended[0] = src[0] - 0.5*(src[1] - src[0])
    #print("src_extended[0]", src_extended[0])
    #src_extended[1:-1] = src
    #src_extended[-1] = src[-1] + 0.5*(src[-1] - src[-2])
    #src_lin = lambda x: util.linearize_between_nodes(x, nodes, src_extended)

    x_dense = np.linspace(nodes[0], nodes[-1], 1001, endpoint=True)
    plt.figure()
    plt.plot(x_dense, src_poly(x_dense))
    plt.plot(nodes[1:-1], src, 'ko')
    plt.savefig(os.path.join(output_dir, "src_" + filename + ".pdf"))
    plt.close()

    x_dense = np.linspace(nodes[0], nodes[-1], 1001, endpoint=True)
    plt.figure()
    plt.plot(x_dense, T_poly(x_dense))
    plt.plot(nodes, T_sample, 'bo')
    plt.plot(nodes, T, 'k.')
    plt.savefig(os.path.join(output_dir, "T_" + filename + ".pdf"))
    plt.close()

    dT = np.gradient(T_sample, nodes)

    for i in range(eps_k.shape[0]):
        #if i == 0:
        #    dT = (T_sample[1] - T_sample[0])/(nodes[1] - nodes[0])
        #elif i == eps_k.shape[0] - 1:
        #    dT = (T_sample[-1] - T_sample[-2])/(nodes[-1] - nodes[-2])
        #else:
        #    dT = (T_sample[i+1] - 2*T_sample[i] + T_sample[i-1])/((nodes[i] - nodes[i-1])**2)
        integral = fixed_quad(func = src_poly, a = nodes[0], b = nodes[i], n = 5)[0]
        print("integral:", integral)
        print("dT:", dT[i])
        eps_k[i] = integral/dT[i]

    k_PBM = 1.0
    k_corrected = k_PBM + eps_k
    k_corrected = k_corrected - k_corrected[0] + k_callable(nodes[0])

    x_dense = np.linspace(nodes[0], nodes[-1], 1001, endpoint=True)

    plt.figure()
    plt.scatter(nodes, k_corrected, s=40, marker='D', facecolor='none', edgecolors='g', label=r"$\hat{\sigma}$")
    plt.plot(x_dense, k_callable(x_dense), 'k-', linewidth=2.0, label=r"$k$")
    plt.xlabel(r"$x$ (m)", fontsize=20)
    plt.ylabel(r"$k\ \left(\mathrm{W/Km}\right)$", fontsize=20)
    plt.xticks(fontsize=17)
    plt.yticks(fontsize=17)
    plt.grid()
    # plt.legend(prop={'size': 17})
    plt.savefig(os.path.join(output_dir, filename + ".pdf"), bbox_inches='tight')
    plt.close()

########################################################################################################################

def main():
    hybrid_CNN_dir  = ""
    hybrid_FCNN_dir = "/home/sindre/msc_thesis/data-driven_corrections/results/2021-04-30_HAM_missing_conductivity_errors_fixed/GlobalDense_HAM_k"
    end_CNN_dir     = ""
    end_FCNN_dir    = ""
    res_CNN_dir     = ""
    res_FCNN_dir    = ""
    dat_CNN_dir     = ""
    dat_FCNN_dir    = "/home/sindre/msc_thesis/data-driven_corrections/results/2021-04-30_DDM_missing_conductivity_errors_fixed/GlobalDense_DDM_k"
    output_dir      = "/home/sindre/msc_thesis/data-driven_corrections/thesis_figures/missing_conductivity_1D_trail2"

    visualize_profiles = False
    visualize_errors   = False
    visualize_src      = True
    visualize_k        = True


    use_CNN_results   = False
    use_FCNN_results  = True

    use_end_results   = False
    use_hyb_results   = True
    use_res_results   = False
    use_dat_results   = True

    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=False)

    num_systems_studied = 14
    systems_to_include = [2, 5, 6]

    y_lims_interp = [None, [1e-7, 2e-1], [1e-9, 1e2], [1e-6, 3e-1], [1e-9, 1e2], [1e-9, 1e2], [4e-7, 2e0], [1e-6, 5e-1]]
    y_lims_extrap = [None, [1e-5, 2e0],  [1e-9, 1e2], [1e-5, 1e0],  [1e-9, 1e2], [1e-9, 1e2], [1e-5, 1e0], [1e-6, 4e-1]]

    for s in range(-1, num_systems_studied):
        system_number = s + 1
        if system_number not in systems_to_include:
            continue


        # Plotting temperature profiles.

        if use_FCNN_results and use_hyb_results:
            with open(os.path.join(hybrid_FCNN_dir + str(system_number), "plot_data_stats.pkl"), "rb") as f:
                hybrid_FCNN_plot_dict = pickle.load(f)
        if use_FCNN_results and use_end_results:
            with open(os.path.join(end_FCNN_dir + str(system_number), "plot_data_stats.pkl"), "rb") as f:
                end_FCNN_plot_dict = pickle.load(f)
        if use_FCNN_results and use_res_results:
            with open(os.path.join(res_FCNN_dir + str(system_number), "plot_data_stats.pkl"), "rb") as f:
                res_FCNN_plot_dict = pickle.load(f)
        if use_FCNN_results and use_dat_results:
            with open(os.path.join(dat_FCNN_dir + str(system_number), "plot_data_stats.pkl"), "rb") as f:
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
        x          = hybrid_FCNN_plot_dict['x']

        print("Plot times:", plot_times)

        #np.testing.assert_allclose(alphas, end_FCNN_plot_dict['alphas'], rtol=1e-10, atol=1e-10)
        #np.testing.assert_allclose(plot_times, end_FCNN_plot_dict['time'], rtol=1e-10, atol=1e-10)
        #np.testing.assert_allclose(alphas, end_CNN_plot_dict['alphas'], rtol=1e-10, atol=1e-10)
        #np.testing.assert_allclose(alphas, hybrid_CNN_plot_dict['alphas'], rtol=1e-10, atol=1e-10)
        #np.testing.assert_allclose(plot_times, end_CNN_plot_dict['time'], rtol=1e-10, atol=1e-10)
        #np.testing.assert_allclose(plot_times, hybrid_CNN_plot_dict['time'], rtol=1e-10, atol=1e-10)

        if visualize_profiles:
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
                    visualize_profile_combined(x, unc_profile, end_profile_FCNN, end_profile_CNN, hyb_profile_FCNN, hyb_profile_CNN, res_profile_FCNN, res_profile_CNN, dat_profile_FCNN, dat_profile_CNN, exact_callable, plot_dir, filename)
                    print("Successfully plotted profiles for system " + str(system_number) + ", alpha" + str(np.around(alpha, decimals=5)) + ", time" + str(np.around(t, decimals=5)))

        if visualize_src:
            src_dir = os.path.join(output_dir, "src")
            if not os.path.exists(src_dir):
                os.makedirs(src_dir, exist_ok=False)

            print("keys:", hybrid_FCNN_plot_dict.keys())
            # Plotting corrective source terms, if applicable.
            if use_hyb_results and use_FCNN_results:
                for a, alpha in enumerate(alphas):
                    for plot_num, t in enumerate(plot_times):
                        src = hybrid_FCNN_plot_dict['src_mean'][a][plot_num]
                        filename = "src_s" + str(system_number) + "_alpha" + str(np.around(alpha, decimals=5)) + "_time" + str(np.around(t, decimals=5))
                        visualize_src_terms(x[1:-1], src, None, src_dir, filename)

        if visualize_k:
            k_dir = os.path.join(output_dir, "k")
            if not os.path.exists(k_dir):
                os.makedirs(k_dir, exist_ok=False)

            if use_hyb_results and use_FCNN_results:
                for a, alpha in enumerate(alphas):
                    for plot_num, t in enumerate(plot_times):
                        src = hybrid_FCNN_plot_dict['src_mean'][a][plot_num]
                        T   = hybrid_FCNN_plot_dict['cor_mean'][a][plot_num]
                        k_callable = get_exact_conductivity(system_number, alpha, t)
                        filename = "k_s" + str(system_number) + "_alpha" + str(np.around(alpha, decimals=5)) + "_time" + str(np.around(t, decimals=5))
                        visualize_conductivity(x, src, T, k_callable, k_dir, filename)

        # Plotting l2 errors.

        if use_FCNN_results and use_hyb_results:
            with open(os.path.join(hybrid_FCNN_dir  + str(system_number), "error_data_stats.pkl"), "rb") as f:
                hybrid_FCNN_error_dict = pickle.load(f)
        if use_FCNN_results and use_end_results:
            with open(os.path.join(end_FCNN_dir + str(system_number), "error_data_stats.pkl"), "rb") as f:
                end_FCNN_error_dict = pickle.load(f)
        if use_FCNN_results and use_res_results:
            with open(os.path.join(res_FCNN_dir + str(system_number), "error_data_stats.pkl"), "rb") as f:
                res_FCNN_error_dict = pickle.load(f)
        if use_FCNN_results and use_dat_results:
            with open(os.path.join(dat_FCNN_dir + str(system_number), "error_data_stats.pkl"), "rb") as f:
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

        if visualize_errors:
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
                visualize_error_data_combined(iterations, unc_errors, end_errors_FCNN, end_errors_CNN, hyb_errors_FCNN, hyb_errors_CNN, res_errors_FCNN, res_errors_CNN, dat_errors_FCNN, dat_errors_CNN, error_dir, filename, y_lims[system_number])
                print("Successfully plotted FCNN errors for system " + str(system_number) + ", alpha" + str(np.around(alpha, decimals=5)))
    """
    src_dir = os.path.join(output_dir, "srcs")
    if not os.path.exists(src_dir):
        os.makedirs(src_dir, exist_ok=False)

    with open(os.path.join(hybrid_FCNN_dir + str(1), "plot_data_stats.pkl"), "rb") as f:
        hybrid_FCNN_plot_dict = pickle.load(f)
        print("Value, alpha=0.7:", hybrid_FCNN_plot_dict['src_mean'][1][3][10])
        print("Value, alpha=1.5:", hybrid_FCNN_plot_dict['src_mean'][0][3][10])
        x = hybrid_FCNN_plot_dict['x']
        visualize_src_terms(x[1:-1], hybrid_FCNN_plot_dict['src_mean'][0][-1], 1.5, src_dir, "src_alpha1.5.pdf")
        visualize_src_terms(x[1:-1], hybrid_FCNN_plot_dict['src_mean'][1][-1], 0.7, src_dir, "src_alpha0.7.pdf")
    """
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