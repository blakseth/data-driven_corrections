"""
calculate_deltaT.py

Written by Sindre Stenen Blakseth, 2021.

Script for determining the impact of the source term for different physical systems.
"""

########################################################################################################################
# Package imports.

import matplotlib.pyplot as plt
import numpy as np
import pickle
import time

########################################################################################################################
# File imports.

import config
import physics
import util

########################################################################################################################

def get_delta_norm(u1, u2):
    return np.sqrt((util.get_disc_L2_norm(u1) - util.get_disc_L2_norm(u2))**2) / np.sqrt((util.get_disc_L2_norm(u1)**2 + util.get_disc_L2_norm(u2)**2))

def get_rel_delta_norm(u1, u2, ref):
    ref_norm = util.get_disc_L2_norm(ref)
    return np.sqrt((util.get_disc_L2_norm(u1)/ref_norm - util.get_disc_L2_norm(u2)/ref_norm)**2) / np.sqrt((util.get_disc_L2_norm(u1)/ref_norm)**2 + (util.get_disc_L2_norm(u2)/ref_norm)**2)

def get_delta_norms():
    print("EXECUTION INITIATED\n")
    systems = ["1", "2A", "6", "8A"]
    for system in systems:
        print("SYSTEM:", system)
        cfg = config.Config(
            use_GPU=config.use_GPU,
            group_name=config.group_name,
            run_name=config.run_names[0][0],
            system=system,
            data_tag=config.data_tags[0],
            model_key=config.model_keys[0],
            do_train=False,
            do_test=False
        )
        deltaT = np.zeros((cfg.alphas.shape[0], cfg.Nt_coarse))
        deltaT_rel = np.zeros((cfg.alphas.shape[0], cfg.Nt_coarse))
        DeltaT = np.zeros((cfg.alphas.shape[0], cfg.Nt_coarse))
        for a, alpha in enumerate(cfg.alphas):
            print("alpha:", alpha)
            old_time = 0
            old_T_wo_src = cfg.get_T0(cfg.nodes_coarse, alpha)
            old_T_w_src  = cfg.get_T0(cfg.nodes_coarse, alpha)
            for i in range(1, cfg.Nt_coarse):
                new_time = np.around(old_time + cfg.dt_coarse, decimals=10)
                T_wo_src = physics.simulate(
                    cfg.nodes_coarse, cfg.faces_coarse, old_T_wo_src,
                    lambda t: cfg.get_T_a(t, alpha), lambda t: cfg.get_T_b(t, alpha),
                    cfg.get_k_approx, cfg.get_cV, cfg.rho, cfg.A,
                    lambda x, t: cfg.get_q_hat_approx(x, t, alpha),
                    np.zeros_like(cfg.nodes_coarse[1:-1]),
                    cfg.dt_coarse, old_time, new_time, False
                )
                T_w_src = physics.simulate(
                    cfg.nodes_coarse, cfg.faces_coarse, old_T_w_src,
                    lambda t: cfg.get_T_a(t, alpha), lambda t: cfg.get_T_b(t, alpha),
                    cfg.get_k_approx, cfg.get_cV, cfg.rho, cfg.A,
                    lambda x, t: cfg.get_q_hat(x, t, alpha),
                    np.zeros_like(cfg.nodes_coarse[1:-1]),
                    cfg.dt_coarse, old_time, new_time, False
                )
                T_exact = cfg.get_T_exact(cfg.nodes_coarse, new_time, alpha)
                deltaT[a][i] = get_delta_norm(T_wo_src, T_w_src)
                deltaT_rel[a][i] = get_rel_delta_norm(T_wo_src, T_w_src, T_exact)
                DeltaT[a][i] = get_delta_norm(T_wo_src - T_exact, T_w_src - T_exact)
                old_time = new_time
                old_T_w_src = T_w_src
                old_T_wo_src = T_wo_src

                if i == cfg.Nt_coarse - 1 and np.around(alpha, decimals=1) == 0.7:
                    plt.figure()
                    plt.grid()
                    plt.plot(cfg.nodes_coarse, T_wo_src, '.',  label="Wo src")
                    plt.plot(cfg.nodes_coarse, T_w_src,  '--', label="W/ src")
                    plt.plot(cfg.nodes_coarse, T_exact,  '-',  label="Exact")
                    plt.legend()
                    plt.savefig("profiles" + system + ".pdf")
                    plt.close()

        deltaT_mean = np.mean(deltaT, axis=1)
        deltaT_mean = np.append(deltaT_mean, np.mean(deltaT_mean, axis=0))
        DeltaT_mean = np.mean(DeltaT, axis=1)
        DeltaT_mean = np.append(DeltaT_mean, np.mean(DeltaT_mean, axis=0))
        print("deltaT_mean:", deltaT_mean)
        print("DeltaT_mean:", DeltaT_mean)
        deltaT_last = deltaT[:,-1]
        deltaT_last = np.append(deltaT_last, np.mean(deltaT_last, axis=0))
        DeltaT_last = DeltaT[:,-1]
        DeltaT_last = np.append(DeltaT_last, np.mean(DeltaT_last, axis=0))
        print("deltaT_last:", deltaT_last)
        print("DeltaT_last:", DeltaT_last)
        print("")
    print("EXECUTION COMPLETED")

########################################################################################################################

def get_modelling_and_discretization_error():
    print("EXECUTION INITIATED\n")
    systems = ["8A"]
    data_dict = {
        'system_labels': systems,
        'avg_mod_errors': np.zeros(4),
        'avg_discr_errors': np.zeros(4),
    }
    for system_index, system in enumerate(systems):
        print("SYSTEM:", system)
        cfg = config.Config(
            use_GPU=config.use_GPU,
            group_name=config.group_name,
            run_name=config.run_names[0][0],
            system=system,
            data_tag=config.data_tags[0],
            model_key=config.model_keys[0],
            do_train=False,
            do_test=False
        )
        modelling_errors = []
        discretization_errors = []
        for a, alpha in enumerate(cfg.alphas):
            print("ALPHA:", alpha)
            old_profile_with_discr_error = cfg.get_T0(cfg.nodes_coarse, alpha)
            for i in range(cfg.Nt_coarse):
                old_time = cfg.dt_coarse * (i-1)
                new_time = cfg.dt_coarse * i
                new_profile_with_discr_error = physics.simulate(
                    cfg.nodes_coarse, cfg.faces_coarse, old_profile_with_discr_error,
                    lambda t: cfg.get_T_a(t, alpha), lambda t: cfg.get_T_b(t, alpha),
                    cfg.get_k_approx, cfg.get_cV, cfg.rho, cfg.A,
                    lambda x, t: cfg.get_q_hat(x, t, alpha),
                    np.zeros_like(cfg.nodes_coarse[1:-1]),
                    cfg.dt_coarse, old_time, new_time, False
                )
                ref_coarse = cfg.get_T_exact(cfg.nodes_coarse, new_time, alpha)

                ref_norm = util.get_disc_L2_norm(ref_coarse)
                diff_norm = util.get_disc_L2_norm(new_profile_with_discr_error - ref_coarse)
                discretization_errors.append(diff_norm / ref_norm)

                old_profile_with_discr_error = new_profile_with_discr_error

            print("Discretization errors calculated.")

            start = time.time()
            old_profile_with_mod_error = cfg.get_T0(cfg.nodes_fine, alpha)
            for i in range(cfg.Nt_fine):
                old_time = cfg.dt_fine * (i-1)
                new_time = cfg.dt_fine * i
                new_profile_with_mod_error = physics.simulate(
                    cfg.nodes_fine, cfg.faces_fine, old_profile_with_mod_error,
                    lambda t: cfg.get_T_a(t, alpha), lambda t: cfg.get_T_b(t, alpha),
                    cfg.get_k_approx, cfg.get_cV, cfg.rho, cfg.A,
                    lambda x, t: cfg.get_q_hat_approx(x, t, alpha),
                    np.zeros_like(cfg.nodes_fine[1:-1]),
                    cfg.dt_fine, old_time, new_time, False
                )
                ref_fine = cfg.get_T_exact(cfg.nodes_fine, new_time, alpha)

                ref_norm = util.get_disc_L2_norm(ref_fine)
                diff_norm = util.get_disc_L2_norm(new_profile_with_mod_error - ref_fine)
                modelling_errors.append(diff_norm / ref_norm)

                old_profile_with_mod_error = new_profile_with_mod_error

            print("Modelling errors calculated.")
            print("Time:", time.time() - start)

        modelling_errors = np.asarray(modelling_errors)
        discretization_errors = np.asarray(discretization_errors)

        avg_modelling_error = np.mean(modelling_errors)
        avg_discretization_error = np.mean(discretization_errors)

        data_dict['avg_mod_errors'][system_index] = avg_modelling_error
        data_dict['avg_discr_errors'][system_index] = avg_discretization_error

        print("")
        print("Average modelling error:", avg_modelling_error)
        print("Average discretization error", avg_discretization_error)
        print("Relative modelling error:", avg_modelling_error / (avg_modelling_error + avg_discretization_error))
        print("Relative discretization error:", avg_discretization_error / (avg_modelling_error + avg_discretization_error))
        print("")

    with open("error_measures.pkl", "wb") as f:
        pickle.dump(data_dict, f)

    print("EXECUTION COMPLETED")

########################################################################################################################

def main():
    get_modelling_and_discretization_error()

########################################################################################################################

if __name__ == "__main__":
    main()

########################################################################################################################