"""
main.py

Written by Sindre Stenen Blakseth, 2020.

Main entry point for ML-correcting heat equation.
"""

########################################################################################################################
# Package imports

import argparse
import os

########################################################################################################################
# File imports

import config
import datasets
import test
import train
import models

########################################################################################################################

def main():
    # Parse arguments.
    parser = argparse.ArgumentParser(description="Set purpose of run.")
    parser.add_argument("--dataset",  default=False, action="store_true", help="Create new datasets from raw data.")
    parser.add_argument("--train",    default=False, action="store_true", help="Train ESRGAN.")
    parser.add_argument("--test",     default=False, action="store_true", help="Test pre-trained ESRGAN.")
    parser.add_argument("--use",      default=False, action="store_true", help="Use pre-trained ESRGAN on LR data.")
    args = parser.parse_args()

    # Check validity of arguments.
    if args.train != config.do_train or args.test != config.do_test:
       raise ValueError("Invalid configuration.")

    # -------------------------------
    # Ensure directories exist.
    #os.makedirs(config.datasets_dir, exist_ok=True)
    #os.makedirs(config.raw_data_dir, exist_ok=True)
    #os.makedirs(config.results_dir,  exist_ok=True)
    os.makedirs(config.run_dir,      exist_ok=False)
    #os.makedirs(config.tb_dir,       exist_ok=True)
    #if config.is_train:
    #    os.makedirs(config.tb_run_dir,   exist_ok=False)
    #os.makedirs(config.cp_load_dir,  exist_ok=True)
    #os.makedirs(config.cp_save_dir,  exist_ok=True)
    #os.makedirs(config.eval_im_dir,  exist_ok=True)
    #os.makedirs(config.metrics_dir,  exist_ok=True)

    #-------------------------------------------------------------------------------------------------------------------
    # Create datasets.
    if args.dataset:
        print("Initiating dataset creation.")
        datasets.main()
        print("Completed dataset creation.")

    #-------------------------------------------------------------------------------------------------------------------
    # Train network.

    ensemble = []
    for i in range(config.ensemble_size):
        model = models.create_new_model()
        ensemble.append(model)

    if args.train:
        print("Initiating training.")
        for i, model in enumerate(ensemble):
            train.train(model, i)
        print("Completed training.")

    #-------------------------------------------------------------------------------------------------------------------
    # Test network.

    if args.test:
        print("Initiating testing.")
        error_dicts = []
        plot_data_dicts = []
        for i, model in enumerate(ensemble):
            error_dict, plot_data_dict = test.simulation_test(model, i)
            error_dicts.append(error_dict)
            plot_data_dicts.append(plot_data_dict)
        error_stats_dict, plot_stats_dict = test.save_test_data(error_dicts, plot_data_dicts)
        test.visualize_test_data(error_stats_dict, plot_stats_dict)
        print("Completed testing.")

    # ------------------------------------------------------------------------------------------------------------------
    # Use pre-trained network to make predictions.

    if args.use:
        print("Prediction is currently not implemented.") # TODO: Implement prediction in 'predict.py'

########################################################################################################################

if __name__ == "__main__":
    main()

########################################################################################################################