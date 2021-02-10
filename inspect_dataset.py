"""
inspect_dataset.py

Written by Sindre Stenen Blakseth, 2021.

Script for inspecting datasets used for training and evaluating ML correction models.
"""

########################################################################################################################
# Package imports.

import joblib
import matplotlib.pyplot as plt
import numpy as np
import numpy_indexed as npi
import os

########################################################################################################################
# File imports.

import config
import physics

########################################################################################################################

def main():
    if config.system in [1,2]:
        indices = [0, 100, 200, 400, 800, 1600, 4000]
    elif config.system == 3:
        indices = [0, 100, 200, 400, 800, 1500, 3000]
    elif config.system == 4:
        indices = [0,9,19,29,39,49,59,69,79,89, 99]
    elif config.system == 5:
        indices = [0, 10, 100, 1000]
    else:
        raise Exception("Invalid system selection.")
    save_filepath = os.path.join(config.datasets_dir, config.data_tag + ".sav")
    if os.path.exists(save_filepath):
        simulation_data = joblib.load(save_filepath)
    else:
        raise Exception("No matching dataset found.")

    plt.figure()
    plt.title("Uncorrected")
    for index in indices:
        plt.plot(simulation_data['x'], simulation_data['unc'][index], label=index)
    plt.legend()
    plt.grid()
    #plt.savefig(os.path.join(config.results_dir, config.data_tag + '/unc.pdf'), bbox_inches='tight')

    plt.figure()
    plt.title("Reference")
    for index in indices:
        plt.plot(simulation_data['x'], simulation_data['ref'][index], label=index)
    plt.legend()
    plt.grid()
    #plt.savefig(os.path.join(config.results_dir, config.data_tag + '/ref.pdf'), bbox_inches='tight')

    plt.figure()
    plt.title("Source")
    for index in indices:
        plt.plot(simulation_data['x'][1:-1], simulation_data['src'][index], label=index)
    plt.legend()
    plt.grid()
    #plt.savefig(os.path.join(config.results_dir, config.data_tag + '/src.pdf'), bbox_inches='tight')

    if config.exact_solution_available:
        plt.figure()
        plt.title("Exact vs ref")
        plt.plot(simulation_data['x'], simulation_data['ref'][-1], '.', label="ref")
        plt.plot(simulation_data['x'], config.get_T_exact(config.nodes_coarse, config.t_end), label="exact")
        plt.legend()
        plt.grid()

    plt.show()

    T0 = simulation_data['unc'][0]
    t = 0
    index = 0
    T_old = T0
    errors = []
    while t < config.t_end:
        T_new = physics.simulate(
            config.nodes_coarse, config.faces_coarse,
            T_old, config.T_a, config.T_b,
            config.get_k_approx, config.get_cV, config.rho, config.A,
            config.get_q_hat_approx, simulation_data['src'][index+1],
            config.dt_coarse, t, np.around(t + config.dt_coarse, decimals=10), False
        )
        index += 1
        t = np.around(t + config.dt_coarse, decimals=10)
        if index < simulation_data['ref'].shape[0]:
            errors.append(np.sqrt(np.sum((T_new - simulation_data['ref'][index])**2)))
        T_old = T_new

    print("Corrected:", T_old)
    print("Reference:", simulation_data['ref'][-1])
    print("End diff:", T_old - simulation_data['ref'][-1])

    plt.figure()
    plt.title("Errors")
    plt.plot(np.arange(len(errors)), errors)
    plt.grid()
    # plt.savefig(os.path.join(config.results_dir, config.data_tag + '/src.pdf'), bbox_inches='tight')
    plt.show()

########################################################################################################################'

if __name__ == "__main__":
    main()

########################################################################################################################