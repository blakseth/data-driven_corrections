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

########################################################################################################################

def main():
    #-------------------------------------------------------------------------------------------------------------------
    # Parse arguments.
    parser = argparse.ArgumentParser(description="Set purpose of run.")
    parser.add_argument("--dataset",  default=False, action="store_true", help="Create new datasets from raw data.")
    parser.add_argument("--train",    default=False, action="store_true", help="Train ESRGAN.")
    parser.add_argument("--test",     default=False, action="store_true", help="Test pre-trained ESRGAN.")
    parser.add_argument("--use",      default=False, action="store_true", help="Use pre-trained ESRGAN on LR data.")
    args = parser.parse_args()

    #-------------------------------------------------------------------------------------------------------------------
    # Configuration setup.

    cfg = config.Config(
        run_name  = config.run_names[0][0],
        system    = config.systems[0],
        data_tag  = config.data_tags[0],
        model_key = config.model_keys[0],
        do_train  = args.train,
        do_test   = args.test
    )

    #-------------------------------------------------------------------------------------------------------------------
    # Ensure directories exist.
    #os.makedirs(config.datasets_dir, exist_ok=True)
    #os.makedirs(config.raw_data_dir, exist_ok=True)
    #os.makedirs(config.results_dir,  exist_ok=True)
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
    #config.save_config()

    #-------------------------------------------------------------------------------------------------------------------
    # Create datasets.
    if args.dataset:
        print("Initiating dataset creation.")
        datasets.main(cfg)
        print("Completed dataset creation.")

    #-------------------------------------------------------------------------------------------------------------------
    # Train network.

    ensemble = []
    for i in range(cfg.ensemble_size):
        model_specific_params = []
        if cfg.model_name == 'GlobalDense':
            model_specific_params = [cfg.num_layers, cfg.hidden_layer_size]
        elif cfg.model_name == 'LocalDense':
            model_specific_params = [cfg.num_layers, 9]
        elif cfg.model_name == 'GlobalCNN':
            model_specific_params = [cfg.num_layers, 3, 20, 1]
        elif cfg.model_name == 'EnsembleLocalDense':
            model_specific_params = [cfg.N_coarse, cfg.num_layers, cfg.hidden_layer_size]
        elif cfg.model_name == 'EnsembleGlobalCNN':
            # [No. networks, No. conv layers, Kernel size, No. channels, No. FC layers at end]
            model_specific_params = [cfg.N_coarse, cfg.num_layers, 3, 20, 1]
        model = models.create_new_model(cfg, model_specific_params)
        ensemble.append(model)

    if args.train:
        dataset_train, dataset_val, _ = datasets.load_datasets(cfg, True, True, False)

        dataloader_train = torch.utils.data.DataLoader(
            dataset=dataset_train,
            batch_size=cfg.batch_size_train,
            shuffle=True,
            num_workers=0,
            pin_memory=True
        )
        dataloader_val = torch.utils.data.DataLoader(
            dataset=dataset_val,
            batch_size=cfg.batch_size_val,
            shuffle=True,
            num_workers=0,
            pin_memory=True
        )
        print("Initiating training.")
        for i, model in enumerate(ensemble):
            _ = train.train(cfg, model, i, dataloader_train, dataloader_val)
        print("Completed training.")

    #-------------------------------------------------------------------------------------------------------------------
    # Test network.

    if args.test:
        print("Initiating testing.")
        error_dicts = []
        plot_data_dicts = []
        for i, model in enumerate(ensemble):
            if cfg.do_simulation_test:
                error_dict, plot_data_dict = test.simulation_test(cfg, model, i)
            else:
                error_dict, plot_data_dict = test.single_step_test(cfg, model, i)
            error_dicts.append(error_dict)
            plot_data_dicts.append(plot_data_dict)
        error_stats_dict, plot_stats_dict = test.save_test_data(cfg, error_dicts, plot_data_dicts)
        test.visualize_test_data(cfg, error_stats_dict, plot_stats_dict)
        print("Completed testing.")

    # ------------------------------------------------------------------------------------------------------------------
    # Use pre-trained network to make predictions.

    if args.use:
        print("Prediction is currently not implemented.") # TODO: Implement prediction in 'predict.py'

########################################################################################################################

if __name__ == "__main__":
    main()

########################################################################################################################