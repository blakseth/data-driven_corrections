"""
main.py

Written by Sindre Stenen Blakseth, 2020.

Main entry point for ML-correcting heat equation.
"""

########################################################################################################################
# Package imports

import argparse
import os
import torch

########################################################################################################################
# File imports

import config
import datasets
import test
import train
import models
import parameter_grid_search

########################################################################################################################

def main():
    #-------------------------------------------------------------------------------------------------------------------
    # Parse arguments.
    parser = argparse.ArgumentParser(description="Set purpose of run.")
    parser.add_argument("--dataset",  default=False, action="store_true", help="Create new datasets from raw data.")
    parser.add_argument("--train",    default=False, action="store_true", help="Train ML model.")
    parser.add_argument("--test",     default=False, action="store_true", help="Test pre-trained ML model.")
    parser.add_argument("--use",      default=False, action="store_true", help="Use pre-trained ML model on new data.")
    parser.add_argument("--grs",      default=False, action="store_true", help="Perform parameter grid search.")
    args = parser.parse_args()

    print("\nEXECUTION INITIATED\n")

    if args.grs:
        print("-----------------------------------------------")
        print("-----------------------------------------------")
        print("Initiating parameter grid search.\n")
        for model_num in range(len(config.model_keys)):
            cfg = config.Config(
                use_GPU    = config.use_GPU,
                group_name = config.group_name,
                run_name   = config.run_names[model_num][0],
                system     = config.systems[0],
                data_tag   = config.data_tags[0],
                model_key  = config.model_keys[model_num],
                do_train   = False,
                do_test    = False,
                N_x        = config.N_x,
                model_type = config.model_type,
            )
            print("- - - - - - - - - - - - - - - - - - - - - - - -")
            print("- - - - - - - - - - - - - - - - - - - - - - - -")
            print("Finding optimal parameters for model " + cfg.model_name)
            parameter_grid_search.grid_search(cfg)
            print("")
        print("Completed parameter grid search.\n\nEXECUTION COMPLETED")
        return

    group_name = config.group_name
    for model_num in range(len(config.model_keys)):
        for sys_num in range(len(config.systems)):
            print("\n********************************************************")
            print("Model  number:", model_num)
            print("System number:", sys_num)
            print("********************************************************\n")

            # -------------------------------------------------------------------------------------------------------------------
            # Configuration setup.
            cfg = config.Config(
                use_GPU    = config.use_GPU,
                group_name = group_name,
                run_name   = config.run_names[model_num][sys_num],
                system     = config.systems[sys_num],
                data_tag   = config.data_tags[sys_num],
                model_key  = config.model_keys[model_num],
                do_train   = args.train,
                do_test    = args.test,
                N_x        = config.N_x,
                model_type = config.model_type
            )

            #-------------------------------------------------------------------------------------------------------------------
            # Ensure directories exist.
            #os.makedirs(config.datasets_dir, exist_ok=True)
            #os.makedirs(config.raw_data_dir, exist_ok=True)
            #os.makedirs(config.results_dir,  exist_ok=True)
            #os.makedirs(cfg.group_dir,    exist_ok=True)
            os.makedirs(cfg.run_dir,      exist_ok=False)
            #os.makedirs(config.tb_dir,       exist_ok=True)
            #if config.is_train:
            #    os.makedirs(config.tb_run_dir,   exist_ok=False)
            #os.makedirs(config.cp_load_dir,  exist_ok=True)
            #os.makedirs(config.cp_save_dir,  exist_ok=True)
            #os.makedirs(config.eval_im_dir,  exist_ok=True)
            #os.makedirs(config.metrics_dir,  exist_ok=True)

            #-------------------------------------------------------------------------------------------------------------------
            # Save configurations.
            config.save_config(cfg)

            #-------------------------------------------------------------------------------------------------------------------
            # Create datasets.
            if model_num == 0:
                if args.dataset:
                    print("----------------------------")
                    print("Initiating dataset creation.\n")
                    print("Data tag:", cfg.data_tag)
                    datasets.main(cfg)
                    print("\nCompleted dataset creation.")
                    print("----------------------------\n")

            #-------------------------------------------------------------------------------------------------------------------
            # Define network model(s).

            ensemble = []
            print("----------------------------")
            print("Initiating model definition.")
            for i in range(cfg.ensemble_size):
                model = models.create_new_model(cfg, cfg.model_specific_params)
                ensemble.append(model)
                if i == 0 and sys_num == 0:
                    print("\n" + cfg.model_name + "\n")
                    if cfg.model_name[:8] == 'Ensemble':
                        print("Ensemble model containing " + str(len(model.nets)) + " networks as shown below.")
                        print(model.nets[0].net)
                    elif cfg.model_name[:10] == 'LocalDense':
                        print("Ensemble model containing " + str(len(model.net.nets)) + " networks as shown below.")
                        print(model.net.nets[0])
                    else:
                        print(model.net)
            print("\nCompleted model definition.")
            print("----------------------------\n")


            # -------------------------------------------------------------------------------------------------------------------
            # Train model(s).

            if args.train:
                print("----------------------------")
                print("Initiating training")
                for i, model in enumerate(ensemble):
                    print("\nTraining instance " + str(i))
                    _ = train.train(cfg, model, i)
                print("\nCompleted training.")
                print("----------------------------\n")

            #-------------------------------------------------------------------------------------------------------------------
            # Test model(s).

            if args.test:
                print("----------------------------")
                print("Initiating testing.")
                error_dicts = []
                plot_data_dicts = []
                for i, model in enumerate(ensemble):
                    print("\nTesting instance " + str(i))
                    if cfg.do_simulation_test:
                        error_dict, plot_data_dict = test.simulation_test(cfg, model, i)
                    elif cfg.parametrized_system:
                        error_dict, plot_data_dict = test.parametrized_simulation_test(cfg, model)
                    else:
                        error_dict, plot_data_dict = test.single_step_test(cfg, model, i)
                    error_dicts.append(error_dict)
                    plot_data_dicts.append(plot_data_dict)
                print("")
                error_stats_dict, plot_stats_dict = test.save_test_data(cfg, error_dicts, plot_data_dicts)
                test.visualize_test_data(cfg, error_stats_dict, plot_stats_dict)
                print("\nCompleted testing.")
                print("----------------------------\n")

            # ------------------------------------------------------------------------------------------------------------------
            # Use pre-trained network to make predictions.

            if args.use:
                print("Prediction is currently not implemented.") # TODO: Implement prediction in 'predict.py'

    print("EXECUTION COMPLETED\n")

########################################################################################################################

if __name__ == "__main__":
    main()

########################################################################################################################