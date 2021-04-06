""""
grid_refinement.py

Written by Sindre Stenen Blakseth, 2021.

Script for performing grid refinement studies of PBM, DDM and HAM simulation methods.
"""

########################################################################################################################
# Package imports.

import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle

########################################################################################################################
# File imports.

import config
import datasets
import models
import test
import train

########################################################################################################################
# Grid refinement study.

def grid_refinement_study(model_key, sys_num, NJs, create_datasets, run_name, save_dir, verbose):
    PBM_results = np.zeros((NJs.shape[0], 4))
    HAM_results = np.zeros((NJs.shape[0], 4))
    DDM_results = np.zeros((NJs.shape[0], 4))
    alphas = np.zeros(4)
    for res_num, N_j in enumerate(NJs):
        print("---------------------------------")
        print("---------------------------------")
        print("N_j = ", N_j, "\n")
        data_tag = "sys" + str(config.systems[sys_num]) + "_Nj" + str(N_j)

        # Create data if necessary.
        if create_datasets:
            print("Creating datasets")
            dataset_cfg = config.Config(
                use_GPU    = config.use_GPU,
                group_name = config.group_name,
                run_name   = run_name,
                system     = config.systems[sys_num],
                data_tag   = data_tag,
                model_key  = model_key,
                do_train   = True,
                do_test    = True,
                N_j        = N_j,
                model_type = 'data' # Datasets are the same for all model types, so this is just a placeholder value.
            )
            datasets.main(dataset_cfg)
            print("Created datasets\n")

        # Train HAM.
        print("HAM init")
        HAM_cfg = config.Config(
            use_GPU     = config.use_GPU,
            group_name  = config.group_name,
            run_name    = run_name,
            system      = config.systems[sys_num],
            data_tag    = data_tag,
            model_key   = model_key,
            do_train    = True,
            do_test     = True,
            N_j         = N_j,
            model_type  = 'data'  # Datasets are the same for all model types, so this is just a placeholder value.
        )
        HAM_model = models.create_new_model(HAM_cfg, HAM_cfg.model_specific_params)
        if verbose:
            print("\n" + HAM_cfg.model_name + "\n")
            if HAM_cfg.model_name[:8] == 'Ensemble':
                print("Ensemble model containing " + str(len(HAM_model.nets)) + " networks as shown below.")
                print(HAM_model.nets[0].net)
            elif HAM_cfg.model_name[:5] == 'Local':
                print("Ensemble model containing " + str(len(HAM_model.net.nets)) + " networks as shown below.")
                print(HAM_model.net.nets[0])
            else:
                print(HAM_model.net)
        _ = train.train(HAM_cfg, HAM_model, 0)
        HAM_error_dict, _ = test.parametrized_simulation_test(HAM_cfg, HAM_model)
        alphas = HAM_error_dict['alphas']
        HAM_final_errors = HAM_error_dict['cor_L2'][:, -1] * np.sqrt(HAM_cfg.dx_coarse)
        print("HAM errors:", HAM_final_errors)
        PBM_final_errors = HAM_error_dict['unc_L2'][:, -1] * np.sqrt(HAM_cfg.dx_coarse)
        print("PBM errors:", PBM_final_errors)
        assert HAM_final_errors.shape == alphas.shape
        print("HAM complete\n")


        # Train DDM.
        print("DDM init")
        DDM_cfg = config.Config(
            use_GPU=config.use_GPU,
            group_name=config.group_name,
            run_name=run_name,
            system=config.systems[sys_num],
            data_tag=data_tag,
            model_key=model_key,
            do_train=True,
            do_test=True,
            N_j=N_j,
            model_type='data'  # Datasets are the same for all model types, so this is just a placeholder value.
        )
        DDM_model = models.create_new_model(DDM_cfg, DDM_cfg.model_specific_params)
        if verbose:
            print("\n" + HAM_cfg.model_name + "\n")
            if DDM_cfg.model_name[:8] == 'Ensemble':
                print("Ensemble model containing " + str(len(DDM_model.nets)) + " networks as shown below.")
                print(DDM_model.nets[0].net)
            elif DDM_cfg.model_name[:5] == 'Local':
                print("Ensemble model containing " + str(len(DDM_model.net.nets)) + " networks as shown below.")
                print(DDM_model.net.nets[0])
            else:
                print(DDM_model.net)
        _ = train.train(DDM_cfg, DDM_model, 0)
        DDM_error_dict, _ = test.parametrized_simulation_test(DDM_cfg, DDM_model)
        DDM_final_errors = DDM_error_dict['cor_L2'][:, -1] * np.sqrt(DDM_cfg.dx_coarse)
        print("DDM errors:", DDM_final_errors)
        print("DDM complete")

        PBM_results[res_num,:] = PBM_final_errors
        HAM_results[res_num,:] = HAM_final_errors
        DDM_results[res_num,:] = DDM_final_errors

    print("PDB results:", PBM_results)
    print("HAM results:", HAM_results)
    print("DDM results:", DDM_results)

    print("Plot")
    for a, alpha in enumerate(alphas):
        plt.figure()
        ax = plt.gca()
        ax.set_yscale('log')
        plt.scatter(NJs, PBM_results[:, a], s=40, marker='o', facecolors='none', edgecolors='r', label="PBM")
        plt.scatter(NJs, DDM_results[:, a], s=40, marker='s', facecolors='none', edgecolors='b', label="DDM")
        plt.scatter(NJs, HAM_results[:, a], s=40, marker='D', facecolors='none', edgecolors='g', label="HAM")
        plt.plot(np.asarray([NJs[0], NJs[-1]]), PBM_results[0, a]*np.asarray([1.0, (NJs[0]/NJs[-1])**2]), 'k--', label="2nd order")
        plt.xlabel(r"$N_j$", fontsize=20)
        plt.ylabel(r"Relative $l_2$ error", fontsize=20)
        plt.xticks(fontsize=17)
        plt.yticks(fontsize=17)
        plt.legend()
        plt.grid()
        plt.savefig(os.path.join(HAM_cfg.run_dir, "grid_refine_alpha" + str(np.around(alpha, decimals=5)) + ".pdf"), bbox_inches='tight')

    print("Save")
    data_dict = {
        'PBM': PBM_results,
        'HAM': HAM_results,
        'DDM': DDM_results,
        'NJs': NJs,
        'alphas': alphas
    }
    with open(os.path.join(HAM_cfg.run_dir, "grid_refinement_study" + ".pkl"), "wb") as f:
        pickle.dump(data_dict, f)

########################################################################################################################

def main():
    print("EXECUTION INITIATED\n")
    parser = argparse.ArgumentParser(description="Set purpose of run.")
    parser.add_argument("--dataset", default=False, action="store_true", help="Create new datasets from raw data.")
    parser.add_argument("--verbose", default=False, action="store_true", help="Toggle verbose output.")
    args = parser.parse_args()
    spatial_resolutions = np.asarray([15])
    if args.dataset:
        create_datasets = True
    else:
        create_datasets = False
    model_keys = [0]
    #base_dir     = '/home/sindre/msc_thesis/data-driven_corrections'
    base_dir     = '/content/gdrive/My Drive/msc_thesis/data-driven_corrections'
    results_dir  = os.path.join(base_dir, 'results')
    main_run_dir = os.path.join(results_dir, config.group_name)
    for model_key in model_keys:
        for sys_num in range(len(config.systems)):
            run_name = "grid_arch" + str(model_key) + "_sys" + str(sys_num)
            os.makedirs(os.path.join(main_run_dir, run_name), exist_ok=False)
            grid_refinement_study(model_key, sys_num, spatial_resolutions, create_datasets, run_name, main_run_dir, args.verbose)
    print("\nEXECUTION COMPLETED\n")

########################################################################################################################

if __name__ == "__main__":
    main()

########################################################################################################################