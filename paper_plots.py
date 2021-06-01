"""
paper_plots.py

Written by Sindre Stenen Blakseth, 2021.

Displaying the results of grid refinement studies.
"""

########################################################################################################################

import matplotlib.pyplot as plt
import numpy as np
import os
import pickle

########################################################################################################################

def visualize_grid_refinement(NJs, PBM_errors, DDM_errors, HAM_errors, output_dir, filename):
    plt.figure()
    plt.scatter(NJs, PBM_errors, s=35, marker='o', facecolors='none', edgecolors='r')
    plt.scatter(NJs, DDM_errors, s=40, marker='s', facecolors='none', edgecolors='b')
    plt.scatter(NJs, HAM_errors, s=40, marker='D', facecolors='none', edgecolors='g')
    plt.plot(np.asarray([NJs[0], NJs[-1]]), PBM_errors[0]*(NJs[0]**2)*np.asarray([NJs[0], NJs[-1]])**(-2.), 'k--', linewidth=2.0)
    #plt.plot(np.asarray([NJs[0], NJs[-1]]), PBM_errors[0]*(NJs[0]**2.5)*np.asarray([NJs[0], NJs[-1]])**(-2.5), 'y--', linewidth=2.0)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlim(NJs[0]/1.5, NJs[-1]*1.5)
    plt.xlabel(r"$N_j$", fontsize=20)
    plt.ylabel(r"Relative $\ell_2$ Error", fontsize=20)
    plt.xticks(fontsize=17)
    plt.yticks(fontsize=17)
    plt.grid()
    plt.savefig(os.path.join(output_dir, filename + ".pdf"), bbox_inches='tight')
    plt.close()

########################################################################################################################

def main():
    main_dir   = "/home/sindre/msc_thesis/data-driven_corrections/results/2021-05-07_grs"
    output_dir = "/home/sindre/msc_thesis/data-driven_corrections/thesis_figures/final_grs"

    os.makedirs(output_dir, exist_ok=True)

    system_names = ["1", "6"]

    for system_number, system_name in enumerate(system_names):
        data_dir = os.path.join(main_dir, "grid_arch0_sys" + str(system_number))

        with open(os.path.join(data_dir, "grid_refinement_study" + ".pkl"), "rb") as f:
            data_dict = pickle.load(f)

            alphas = data_dict['alphas']
            NJs    = data_dict['NJs']

            for a, alpha in enumerate(alphas):
                PBM_errors = data_dict['PBM'][:,a]
                DDM_errors = data_dict['DDM'][:,a]
                HAM_errors = data_dict['HAM'][:,a]

                filename = "grs_s" + system_name + "a_" + str(np.around(alpha, decimals=5))
                visualize_grid_refinement(NJs, PBM_errors, DDM_errors, HAM_errors, output_dir, filename)

########################################################################################################################

if __name__ == "__main__":
    main()

########################################################################################################################
