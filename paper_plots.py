"""
paper_plots.py

Written by Sindre Stenen Blakseth, 2021.

Script for producing the plots for HAM-CoSTA-paper.
"""

########################################################################################################################
# Package imports.

import os
import numpy as np
import pickle
import matplotlib.pyplot as plt

########################################################################################################################
# Exact solutions.

def get_exact_solution(system_number, alpha, t):
    if system_number == 1:
        return lambda x: t + 0.5 * alpha * (x ** 2)

    if system_number == 2:
        return lambda x: np.sqrt(t + alpha + 1) + 10 * (x ** 2) * (x - 1) * (x + 2)

    if system_number == 3:
        return lambda x: 2 * (x ** (alpha + 4)) - (t ** 2) * x * (x - 1)

    if system_number == 4:
        return lambda x: np.sin(2 * np.pi * x) * np.exp(-alpha * t)

    if system_number == 5:
        return lambda x: -2 * (x ** 3) * (x - alpha) / (t + 0.5)

    if system_number == 6:
        return lambda x: 2 + alpha * (x - 1) * np.tanh(x / (t + 0.1))

    if system_number == 7:
        return lambda x: np.sin(2 * np.pi * t) + alpha * np.sin(2 * np.pi * x)

    if system_number == 8:
        return lambda x: 1 + np.sin(2 * np.pi * t + alpha) * np.cos(2 * np.pi * x)

    if system_number == 9:
        return lambda x: 1 + alpha * np.cos(2 * np.pi * x * (t ** 2))

    if system_number == 10:
        return lambda x: 5 + x * (x - 1) / (t + 0.1) + 0.1 * t * np.sin(2 * np.pi * x + alpha)

    if system_number == 11:
        return lambda x: 1 + np.sin(5 * x * t) * np.exp(-0.2 * x * t) + alpha * (x ** 3)

    if system_number == 12:
        return lambda x: 5 * t * (x ** 2) * np.sin(10 * np.pi * t) + np.sin(2 * np.pi * alpha * x) / (t + 0.2)

    if system_number == 13:
        return lambda x: 1 + t / (1 + ((x - 0.5 * alpha) ** 2))

    if system_number == 14:
        return lambda x: 1 + t * np.exp(-1000 * (alpha + 1) * (x - 0.5) ** 2)

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

########################################################################################################################

def main():
    hybrid_CNN_dir  = ""
    hybrid_FCNN_dir = "/home/sindre/msc_thesis/data-driven_corrections/results/2021-03-24_hybrid_GlobalDense_rerun/GlobalDense_s"
    end_CNN_dir     = ""
    end_FCNN_dir    = "/home/sindre/msc_thesis/data-driven_corrections/results/2021-03-24_end_GlobalDense_rerun/GlobalDense_s"
    output_dir      = "/home/sindre/msc_thesis/data-driven_corrections/paper_figures/plots_2021-03-24_v2"

    os.makedirs(output_dir, exist_ok=False)

    num_systems_studied = 14
    systems_to_include = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]

    for s in range(num_systems_studied):
        system_number = s + 1
        if system_number not in systems_to_include:
            continue


        # Plotting profiles for FCNNs.
        with open(os.path.join(hybrid_FCNN_dir + str(system_number), "plot_data_stats.pkl"), "rb") as f:
            hybrid_FCNN_plot_dict = pickle.load(f)
        with open(os.path.join(end_FCNN_dir + str(system_number), "plot_data_stats.pkl"), "rb") as f:
            end_FCNN_plot_dict = pickle.load(f)
        print("Successfully loaded FCNN plot dicts.")

        alphas     = hybrid_FCNN_plot_dict['alphas']
        plot_times = hybrid_FCNN_plot_dict['time']
        x          = hybrid_FCNN_plot_dict['x']
        np.testing.assert_allclose(alphas, end_FCNN_plot_dict['alphas'], rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(plot_times, end_FCNN_plot_dict['time'], rtol=1e-10, atol=1e-10)

        FCNN_plot_dir = os.path.join(output_dir, "FCNN_plots")
        if not os.path.exists(FCNN_plot_dir):
            os.makedirs(FCNN_plot_dir, exist_ok=True)

        for a, alpha in enumerate(alphas):
            for plot_num, t in enumerate(plot_times):
                unc_profile = hybrid_FCNN_plot_dict['unc'][a][plot_num]
                np.testing.assert_allclose(unc_profile, end_FCNN_plot_dict['unc'][a][plot_num], rtol=1e-10, atol=1e-10)
                end_profile = end_FCNN_plot_dict['cor_mean'][a][plot_num]
                hyb_profile = hybrid_FCNN_plot_dict['cor_mean'][a][plot_num]
                exact_callable = get_exact_solution(system_number, alpha, t)
                filename = "profiles_FCNN_s" + str(system_number) + "_alpha" + str(np.around(alpha, decimals=5)) + "_time" + str(np.around(t, decimals=5))
                visualize_profile(x, unc_profile, end_profile, hyb_profile, exact_callable, FCNN_plot_dir, filename)
                print("Successfully plotted FCNN profiles for system " + str(system_number) + ", alpha" + str(np.around(alpha, decimals=5)) + ", time" + str(np.around(t, decimals=5)))


        # Plotting profiles for CNNs.
        if hybrid_CNN_dir is not "" and end_CNN_dir is not "":
            with open(os.path.join(hybrid_CNN_dir + str(system_number), "plot_data_stats.pkl"), "rb") as f:
                hybrid_CNN_plot_dict = pickle.load(f)
            with open(os.path.join(end_CNN_dir + str(system_number), "plot_data_stats.pkl"), "rb") as f:
                end_CNN_plot_dict = pickle.load(f)
            print("Successfully loaded CNN plot dicts.")

            np.testing.assert_allclose(alphas, end_CNN_plot_dict['alphas'], rtol=1e-10, atol=1e-10)
            np.testing.assert_allclose(alphas, hybrid_CNN_plot_dict['alphas'], rtol=1e-10, atol=1e-10)
            np.testing.assert_allclose(plot_times, end_CNN_plot_dict['time'], rtol=1e-10, atol=1e-10)
            np.testing.assert_allclose(plot_times, hybrid_CNN_plot_dict['time'], rtol=1e-10, atol=1e-10)

            CNN_plot_dir = os.path.join(output_dir, "CNN_plots")
            if not os.path.exists(CNN_plot_dir):
                os.makedirs(CNN_plot_dir, exist_ok=True)

            for a, alpha in enumerate(alphas):
                for plot_num, t in enumerate(plot_times):
                    unc_profile = hybrid_CNN_plot_dict['unc'][a][plot_num]
                    np.testing.assert_allclose(unc_profile, end_CNN_plot_dict['unc'][a][plot_num], rtol=1e-10, atol=1e-10)
                    end_profile = end_CNN_plot_dict['cor_mean'][a][plot_num]
                    hyb_profile = hybrid_CNN_plot_dict['cor_mean'][a][plot_num]
                    exact_callable = get_exact_solution(system_number, alpha, t)
                    filename = "profiles_CNN_s" + str(system_number) + "_alpha" + str(np.around(alpha, decimals=5)) + "_time" + str(np.around(t, decimals=5))
                    visualize_profile(x, unc_profile, end_profile, hyb_profile, exact_callable, CNN_plot_dir, filename)
                    print("Successfully plotted CNN profiles for system " + str(system_number) + ", alpha" + str(np.around(alpha, decimals=5)) + ", time" + str(np.around(t, decimals=5)))


        # Plotting l2 errors for FCNNs.
        with open(os.path.join(hybrid_FCNN_dir + str(system_number), "error_data_stats.pkl"), "rb") as f:
            hybrid_FCNN_error_dict = pickle.load(f)
        with open(os.path.join(end_FCNN_dir + str(system_number), "error_data_stats.pkl"), "rb") as f:
            end_FCNN_error_dict = pickle.load(f)
        print("Successfully loaded FCNN error dicts.")

        assert alphas.shape[0] == hybrid_FCNN_error_dict['unc_L2'].shape[0]
        assert alphas.shape[0] == end_FCNN_error_dict['unc_L2'].shape[0]
        assert hybrid_FCNN_error_dict['unc_L2'].shape[1] == end_FCNN_error_dict['unc_L2'].shape[1]

        FCNN_error_dir = os.path.join(output_dir, "FCNN_errors")
        if not os.path.exists(FCNN_error_dir):
            os.makedirs(FCNN_error_dir, exist_ok=False)

        iterations = np.arange(1, hybrid_FCNN_error_dict['unc_L2'].shape[1] + 1, 1)
        for a, alpha in enumerate(alphas):
            unc_errors = hybrid_FCNN_error_dict['unc_L2'][a]
            np.testing.assert_allclose(unc_errors, end_FCNN_error_dict['unc_L2'][a], rtol=1e-10, atol=1e-10)
            end_errors = end_FCNN_error_dict['cor_mean_L2'][a]
            hyb_errors = hybrid_FCNN_error_dict['cor_mean_L2'][a]
            filename = "errors_FCNN_s" + str(system_number) + "_alpha" + str(np.around(alpha, decimals=5))
            visualize_error_data(iterations, unc_errors, end_errors, hyb_errors, FCNN_error_dir, filename)
            print("Successfully plotted FCNN errors for system " + str(system_number) + ", alpha" + str(np.around(alpha, decimals=5)))


        # Plotting l2 errors for FCNNs.
        if hybrid_CNN_dir is not "" and end_CNN_dir is not "":
            with open(os.path.join(hybrid_FCNN_dir + str(system_number), "error_data_stats.pkl"), "rb") as f:
                hybrid_FCNN_error_dict = pickle.load(f)
            with open(os.path.join(end_FCNN_dir + str(system_number), "error_data_stats.pkl"), "rb") as f:
                end_FCNN_error_dict = pickle.load(f)
            print("Successfully loaded CNN error dicts.")

            assert alphas.shape[0] == hybrid_FCNN_error_dict['unc_L2'].shape[0]
            assert alphas.shape[0] == end_FCNN_error_dict['unc_L2'].shape[0]
            assert hybrid_FCNN_error_dict['unc_L2'].shape[1] == end_FCNN_error_dict['unc_L2'].shape[1]

            CNN_error_dir = os.path.join(output_dir, "CNN_errors")
            if not os.path.exists(CNN_error_dir):
                os.makedirs(CNN_error_dir, exist_ok=False)

            iterations = np.arange(1, hybrid_FCNN_error_dict['unc_L2'].shape[1] + 1, 1)
            for a, alpha in enumerate(alphas):
                unc_errors = hybrid_FCNN_error_dict['unc_L2'][a]
                np.testing.assert_allclose(unc_errors, end_FCNN_error_dict['unc_L2'][a], rtol=1e-10, atol=1e-10)
                end_errors = end_FCNN_error_dict['cor_mean_L2'][a]
                hyb_errors = hybrid_FCNN_error_dict['cor_mean_L2'][a]
                filename = "errors_FCNN_s" + str(system_number) + "_alpha" + str(np.around(alpha, decimals=5))
                visualize_error_data(iterations, unc_errors, end_errors, hyb_errors, CNN_error_dir, filename)
                print("Successfully plotted CNN errors for system " + str(system_number) + ", alpha" + str(np.around(alpha, decimals=5)))


########################################################################################################################

if __name__ == "__main__":
    main()

########################################################################################################################