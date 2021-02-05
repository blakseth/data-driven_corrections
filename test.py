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

def test_end_to_end(model, num):
    model.net.eval()

    _, _, dataset_test = load_datasets(False, False, True)

    dataloader_test = torch.utils.data.DataLoader(
        dataset=dataset_test,
        batch_size=config.N_test_examples,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    random_indices = [4, 12, 28, 49]
    #random_indices = [0, 150, 300, 499]

    model.net.eval()

    for i, data in enumerate(dataloader_test):
        #print("i:", i)

        real_coarse = data[0]
        real_fine = data[1]

        fake_fine = model.net(real_coarse)

        fake_fine_full = np.zeros(real_coarse.shape)
        fake_fine_full[:,0] = real_coarse[:,0]
        fake_fine_full[:,1:-1] = fake_fine.detach().numpy()
        fake_fine_full[:,-1] = real_coarse[:,-1]
        fake_fine_full_unnorm = util.unnormalize(fake_fine_full, config.T_min_train, config.T_max_train)

        real_fine_full = np.zeros(real_coarse.shape)
        real_fine_full[:, 0] = real_coarse[:, 0]
        real_fine_full[:, 1:-1] = real_fine.detach().numpy()
        real_fine_full[:, -1] = real_coarse[:, -1]
        real_fine_full_unnorm = util.unnormalize(real_fine_full, config.T_min_train, config.T_max_train)

        test_loss = model.loss_fn(fake_fine, real_fine).item()

        T_min = config.T_min_train
        T_max = config.T_max_train

        x = config.nodes_coarse
        x_exact = np.linspace(0, 1, 1000)

        #print("Shape: ", fake_fine_full.shape)
        L2_error = 0.0
        L2_error_uncorrected = 0.0
        plot_count = 0
        for j in range(fake_fine_full.shape[0]):
            if j in random_indices:
                plt.figure()
                plt.title("End-to-End Learning for Steady Problem,\nModel " + str(num + 1)+ ", Corrected Solution #" + str(j + 1),fontsize=20)
                plt.xlabel(r"$x$", fontsize=16)
                plt.ylabel(r"$T_c$", fontsize=16)
                plt.plot(x, util.unnormalize(real_coarse[j], T_min, T_max), 'r-', linewidth=2.0,label="Uncorrected")
                if config.exact_solution_available:
                    plt.plot(
                        x_exact,
                        util.unnormalize(util.T_exact(x_exact, real_coarse[j][0], real_coarse[j][-1]), T_min,T_max),
                        'g-', linewidth=1.3, label="Exact"
                    )
                else:
                    plt.plot(x, real_fine_full_unnorm[j], 'g-', linewidth=1.3,label="Target")
                plt.plot(x, fake_fine_full_unnorm[j], 'k.', linewidth=2.3, label="Corrected")
                plt.plot(x, fake_fine_full_unnorm[j], 'k--', linewidth=1.3)
                plt.legend()
                plt.grid()
                plt.savefig(os.path.join(config.run_dir, "end-to-end_steady" + str(num) + str(plot_count) + ".pdf"), bbox_inches='tight')

                data_dict = dict()
                data_dict['Uncorrected'] = np.asarray([x, util.unnormalize(real_coarse[j], T_min, T_max)])
                if config.exact_solution_available:
                    data_dict['Target'] = np.asarray([x_exact, util.unnormalize(util.T_exact(x_exact, real_coarse[j][0], real_coarse[j][-1]), T_min,T_max)])
                else:
                    data_dict['Target'] = np.asarray([x, real_fine_full_unnorm[j]])
                data_dict['Corrected'] = np.asarray([x, fake_fine_full_unnorm[j]])
                pickle.dump(
                    data_dict,
                    open(os.path.join(config.run_dir, "plot_data_example" + "_" + str(num) + "_" + str(random_indices[plot_count])+ ".pkl"), "wb")
                )
                plot_count += 1


            T_L = util.unnormalize(real_coarse[j][0].numpy(), T_min, T_max)
            T_R = util.unnormalize(real_coarse[j][-1].numpy(), T_min, T_max)
            if config.exact_solution_available:
                exact_ = lambda x_lam,t: util.T_exact(x_lam, T_L, T_R)
            else:
                exact_ = lambda x_lam,t: util.linearized_T_num(x_lam, config.nodes_coarse, real_fine_full_unnorm[j])
            linearized_ = lambda x_lam: util.linearized_T_num(x_lam, config.nodes_coarse, fake_fine_full_unnorm[j])
            linearized_uncorrected_ = lambda x_lam: util.linearized_T_num(x_lam, config.nodes_coarse, util.unnormalize(real_coarse[j], T_min, T_max))
            L2_error_ = util.get_L2_norm(config.faces_coarse, np.infty, linearized_, exact_)
            L2_error_uncorrected_ = util.get_L2_norm(config.faces_coarse, np.infty, linearized_uncorrected_, exact_)
            L2_error += L2_error_ / config.N_test_examples
            L2_error_uncorrected += L2_error_uncorrected_ / config.N_test_examples
            #print(j, L2_error_)
            if j == 0:
                print("First test input:", util.unnormalize(real_coarse[j], T_min, T_max))
                print("First corrected:", fake_fine_full_unnorm[j])
                print("First exact:", real_fine_full_unnorm[j])
                x_test = np.linspace(config.x_a, config.x_b, 20)
                print("Linearized uncorrected:", linearized_uncorrected_(x_test))
                print("Linearized corrected:", linearized_(x_test))
                print("Linearized exact:", exact_(x_test, None))

        f = open(os.path.join(config.run_dir, "end-to-end_steady" + str(num) + ".txt"), "w")
        f.write("Test loss: " + str(test_loss) + "\n")
        f.write("L2 error: " + str(L2_error) + "\n")
        f.write("L2 error (uncorrected): " + str(L2_error_uncorrected) + "\n")
        f.close()
        print("Test loss: " + str(test_loss))
        print("L2 error: " + str(L2_error))
        print("L2 error (uncorrected): " + str(L2_error_uncorrected) + "\n")

def test_source(model, num):
    model.net.eval()

    _, _, dataset_test = load_datasets(False, False, True)

    dataloader_test = torch.utils.data.DataLoader(
        dataset=dataset_test,
        batch_size=config.N_test_examples,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    random_indices = [4, 12, 28, 49]
    #random_indices = [0, 150, 300, 499]

    min_source = config.source_min_train
    max_source = config.source_max_train

    T_min = config.T_min_train
    T_max = config.T_max_train

    x = config.nodes_coarse
    x_exact = np.linspace(0, 1, 1000)

    plot_count = 0
    L2_error = 0.0
    L2_error_uncorrected = 0.0

    for i, data in enumerate(dataloader_test):
        print("i:", i)

        coarse_temps = data[0]
        real_sources = data[1]
        if not config.is_steady:
            old_temps    = data[2]

        fake_sources = model.net(coarse_temps)

        test_loss = model.loss_fn(fake_sources, real_sources).item()

        fake_sources = util.unnormalize(fake_sources.detach().numpy(), min_source, max_source)
        real_sources = util.unnormalize(real_sources.detach().numpy(), min_source, max_source)
        coarse_temps = util.unnormalize(coarse_temps.detach().numpy(), T_min, T_max)
        if not config.is_steady:
            old_temps = util.unnormalize(old_temps.detach().numpy(),    T_min, T_max)

        for j in range(data[0].shape[0]):
            #print("Correction source term:", fake_sources[j])
            #print("Fasit:", util.unnormalize(real_sources[j].detach().numpy(), min_source, max_source))
            T_L = coarse_temps[j][0]
            T_R = coarse_temps[j][-1]
            if config.is_steady:
                T_corrected = physics.simulate(
                    config.nodes_coarse, config.faces_coarse, None, T_L, T_R,
                    config.k_ref, config.c_V, config.rho, config.area, config.q_hat + fake_sources[j],
                    np.infty, np.infty, True, True)
                T_exact = physics.simulate(
                    config.nodes_coarse, config.faces_coarse, None, T_L, T_R,
                    config.k_ref, config.c_V, config.rho, config.area, config.q_hat + real_sources[j],
                    np.infty, np.infty, True, True)
            else:
                T_corrected = physics.simulate(
                    config.nodes_coarse, config.faces_coarse, old_temps[j], T_L, T_R,
                    config.k_ref, config.c_V, config.rho, config.area, config.q_hat + fake_sources[j],
                    config.dt_coarse, config.dt_coarse, False, True)
                T_exact = physics.simulate(
                    config.nodes_coarse, config.faces_coarse, old_temps[j], T_L, T_R,
                    config.k_ref, config.c_V, config.rho, config.area, config.q_hat + real_sources[j],
                    config.dt_coarse, config.dt_coarse, False, True)

            if j in random_indices:
                plt.figure()
                plt.title("Hybrid Modelling for Steady Problem,\nModel " + str(num+1)+ ", Corrected Solution #" + str(j+1), fontsize=20)
                plt.xlabel(r"$x$", fontsize=16)
                plt.ylabel(r"$T_c$", fontsize=16)
                plt.plot(x, coarse_temps[j], 'r-', linewidth=2.0, label="Uncorrected")
                if config.exact_solution_available:
                    plt.plot(
                        x_exact,
                        util.T_exact(x_exact, T_L, T_R),
                        'g-', linewidth=1.3, label="Exact"
                    )
                else:
                    plt.plot(
                        x, T_exact, 'g-', linewidth=1.3, label="Exact"
                    )
                plt.plot(x, T_corrected, 'k.', linewidth=2.3, label="Corrected")
                plt.plot(x, T_corrected, 'k--', linewidth=0.9)
                plt.legend()
                plt.grid()
                plt.savefig(os.path.join(config.run_dir, "hybrid_modelling_steady" + str(num) + str(plot_count) + ".pdf"),
                            bbox_inches='tight')

                data_dict = dict()
                data_dict['Uncorrected'] = np.asarray([x, coarse_temps[j]])
                if config.exact_solution_available:
                    data_dict['Target'] = np.asarray([x_exact, util.T_exact(x_exact, T_L, T_R)])
                else:
                    data_dict['Target'] = np.asarray([x, T_exact])
                data_dict['Corrected'] = np.asarray([x, T_corrected])
                pickle.dump(
                    data_dict,
                    open(os.path.join(config.run_dir, "plot_data_example" + "_" + str(num) + "_" + str(random_indices[plot_count]) + ".pkl"), "wb")
                )

                plot_count += 1

            if config.exact_solution_available:
                exact_ = lambda x_lam, t: util.T_exact(x_lam, T_L, T_R)
            else:
                exact_ = lambda x_lam, t: util.linearized_T_num(x_lam, config.nodes_coarse, T_exact)
            linearized_ = lambda x_lam: util.linearized_T_num(x_lam, config.nodes_coarse, T_corrected)
            linearized_uncorrected_ = lambda x_lam: util.linearized_T_num(x_lam, config.nodes_coarse, coarse_temps[j])
            L2_error_ = util.get_L2_norm(config.faces_coarse, np.infty, linearized_, exact_)
            L2_error_uncorrected_ = util.get_L2_norm(config.faces_coarse, np.infty, linearized_uncorrected_, exact_)
            L2_error += L2_error_ / config.N_test_examples
            L2_error_uncorrected += L2_error_uncorrected_ / config.N_test_examples
            # print(j, L2_error_)

            if j == 0:
                print("First test input:", coarse_temps[j])
                print("First corrected:", T_corrected)
                print("First exact:", T_exact)
                print("First target:", real_sources[j])

        f = open(os.path.join(config.run_dir, "hybrid_modelling_steady" + str(num) + ".txt"), "w")
        f.write("Test loss: " + str(test_loss) + "\n")
        f.write("L2 error (corrected): " + str(L2_error) + "\n")
        f.write("L2 error (uncorrected): " + str(L2_error_uncorrected) + "\n")
        f.close()
        print("Test loss: " + str(test_loss))
        print("L2 error: " + str(L2_error))
        print("L2 error (uncorrected): " + str(L2_error_uncorrected))

def simulation_test(model, num):
    model.net.eval()

    # Get target temperature profile of last validation example. This will be IC for test simulation.
    _, dataset_val, dataset_test = load_datasets(False, True, False)

    dataloader_val = torch.utils.data.DataLoader(
        dataset=dataset_val,
        batch_size=config.N_val_examples,
        shuffle=False,
        num_workers=0,
        pin_memory=True
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