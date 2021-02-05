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
import util
import physics

########################################################################################################################
# Testing ML-model.

def simulation_test(model, num):
    model.net.eval()

    # Get target temperature profile of last validation example. This will be IC for test simulation.
    _, dataset_val, dataset_test = load_datasets(False, True, False)

    dataloader_val = torch.utils.data.DataLoader(
        dataset     = dataset_val,
        batch_size  = config.N_val_examples,
        shuffle     = False,
        num_workers = 0,
        pin_memory  = True
    )


    IC = np.zeros(config.nodes_coarse.shape[0])
    for i, data in enumerate(dataloader_val):
        IC[0]  = data[0][-1][0]
        IC[-1] = data[0][-1][-1]
        if config.model_type == "end-to-end":
            IC[1:-1] = data[1][-1]
            IC = util.unnormalize(IC, config.T_min_train, config.T_max_train)
        if config.model_type == "source":
            old = util.unnormalize(data[2][-1], config.T_min_train, config.T_max_train)
            correct_source = util.unnormalize(data[1][-1], config.source_min_train, config.source_max_train)
            IC = physics.simulate(
                config.nodes_coarse, config.faces_coarse, old, old[0], old[-1],
                config.k_ref, config.c_V, config.rho, config.area, config.q_hat + correct_source,
                config.dt_coarse, config.dt_coarse, False, True
            )

    T_uncorrected = IC
    T_corrected   = IC
    T_exact       = IC

    T_min = config.T_min_train
    T_max = config.T_max_train

    src_min = config.source_min_train
    src_max = config.source_max_train

    L2_errors_uncorrected = []
    L2_errors_corrected   = []

    plot_steps = [0, 4, 19, 99, 999]
    for i in range(config.N_test_examples):

        new_uncorrected = physics.simulate(
            config.nodes_coarse, config.faces_coarse, T_uncorrected, T_uncorrected[0], T_uncorrected[-1],
            config.k_ref, config.c_V, config.rho, config.area, config.q_hat,
            config.dt_coarse, config.dt_coarse, False, True
        )
        new_exact = physics.simulate(
            config.nodes_coarse, config.faces_coarse, T_exact, T_exact[0], T_exact[-1],
            config.get_k, config.c_V, config.rho, config.area, config.q_hat,
            config.dt, config.dt_coarse, False, True
        )

        if config.model_type == "end-to-end":
            new_corrected = T_corrected
            new_corrected[1:-1] = util.unnormalize(
                model.net(
                    torch.from_numpy(
                        util.normalize(T_corrected, T_min, T_max)
                    )
                ).detach().numpy(),
                T_min, T_max
            )
        elif config.model_type == "source":
            new_correction_src = util.unnormalize(
                model.net(
                    torch.from_numpy(
                        util.normalize(new_uncorrected, src_min, src_max)
                    )
                ).detach().numpy(),
                src_min, src_max
            )
            new_corrected = physics.simulate(
                config.nodes_coarse, config.faces_coarse, T_corrected, T_corrected[0], T_corrected[-1],
                config.get_k, config.c_V, config.rho, config.area, config.q_hat + new_correction_src,
                config.dt_coarse, config.dt_coarse, False, True
            )
        else:
            print("Model type:", config.model_type)
            raise Exception

        if i in plot_steps:
            plt.figure()
            if config.model_type == "end-to-end":
                plt.title("End-to-end simulation test, model #" + str(num+1) + ", iteration " + str(i + 1))
            if config.model_type == "source":
                plt.title("Hybrid simulation test, model #" + str(num + 1) + ", iteration " + str(i + 1))
            plt.xlabel(r"$x$ (m)")
            plt.ylabel(r"$T$ (K)")
            plt.plot(config.nodes_coarse, new_uncorrected, 'r-',  linewidth=2.0, label="Uncorrected")
            plt.plot(config.nodes_coarse, new_corrected,   'k.',  linewidth=2.3, label="Corrected")
            plt.plot(config.nodes_coarse, new_corrected,   'k--', linewidth=0.9)
            plt.plot(config.nodes_coarse, new_exact,       'g-',  linewidth=1.3, label="Target")
            plt.legend()
            plt.grid()
            plt.savefig(
                os.path.join(config.run_dir, "Simulation test, model #" + str(num+1) + ", iteration " + str(i+1) + ".pdf"),
                bbox_inches='tight'
            )

            data_dict = dict()
            data_dict['Uncorrected'] = np.asarray([config.nodes_coarse, new_uncorrected])
            data_dict['Corrected']   = np.asarray([config.nodes_coarse, new_corrected])
            data_dict['Target']      = np.asarray([config.nodes_coarse, new_exact])

            pickle.dump(
                data_dict, open(os.path.join(config.run_dir, "plot_data_example" + "_" + str(num+1) + "_" + str(i+1) + ".pkl"), "wb")
            )

        T_uncorrected = new_uncorrected
        T_corrected   = new_corrected
        T_exact       = new_exact

        linearized_uncorrected = lambda x_lam: util.linearized_T_num(x_lam, config.nodes_coarse, new_uncorrected)
        linearized_corrected   = lambda x_lam: util.linearized_T_num(x_lam, config.nodes_coarse, new_corrected)
        linearized_exact       = lambda x_lam, t: util.linearized_T_num(x_lam, config.nodes_coarse, new_exact)
        L2_errors_uncorrected.append(
            util.get_L2_norm(config.faces_coarse, np.infty, linearized_uncorrected, linearized_exact)
        )
        L2_errors_corrected.append(
            util.get_L2_norm(config.faces_coarse, np.infty, linearized_corrected, linearized_exact)
        )

    loss_data_dict = dict()
    loss_data_dict['L2_uncorrected'] = L2_errors_uncorrected
    loss_data_dict['L2_corrected']   = L2_errors_corrected
    f = open(os.path.join(config.run_dir, "hybrid_modelling_simulation_test" + str(num) + ".txt"), "w")
    f.write("L2 error (corrected)")
    for i in range(len(L2_errors_corrected)):
        f.write(str(i) + ": " + str(L2_errors_corrected[i]) + "\n")
    f.write("\n")
    f.write("L2 error (uncorrected)")
    for i in range(len(L2_errors_uncorrected)):
        f.write(str(i) + ": " + str(L2_errors_uncorrected[i]) + "\n")
    f.close()

def test(model, num):
    if config.do_simulation_test:
        simulation_test(model, num)
    elif config.model_type == "end-to-end":
        test_end_to_end(model, num)
    elif config.model_type == "source":
        test_source(model, num)

########################################################################################################################