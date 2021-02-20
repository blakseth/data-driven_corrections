"""
parameter_grid_search.py

Written by Sindre Stenen Blakseth, 2021.

Script for performing parameter grid search.
"""

########################################################################################################################
# Package imports.

import numpy as np
import os
import torch

########################################################################################################################
# File imports.

import config
from datasets import load_datasets_from_path
import models
import train

########################################################################################################################
# Parameter space.

learning_rates = [1e-4, 5e-4, 1e-5]
dropout_probs = [0.0, 0.1, 0.2]
widths = config.N_coarse * [0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
depths = [2, 3, 4, 5, 6, 7, 8, 9, 10]


########################################################################################################################

def main():
    # Load datasets and create dataloaders.
    dataset_paths = [
        [os.path.join(config.datasets_dir, 'system2B_train.pt'),
        os.path.join(config.datasets_dir, 'system2B_val.pt'),
        os.path.join(config.datasets_dir, 'system2B_test.pt')],
        [os.path.join(config.datasets_dir, 'system5B_train.pt'),
        os.path.join(config.datasets_dir, 'system5B_val.pt'),
        os.path.join(config.datasets_dir, 'system5B_test.pt')],
        [os.path.join(config.datasets_dir, 'system8B_train.pt'),
        os.path.join(config.datasets_dir, 'system8B_val.pt'),
        os.path.join(config.datasets_dir, 'system8B_test.pt')]
    ]
    datasets = [[load_datasets_from_path(dataset_paths[i][0], dataset_paths[i][1], None)] for i in range(len(dataset_paths))]
    dataloaders = []
    for i in range(len(dataset_paths)):
        dataloader_train = torch.utils.data.DataLoader(
            dataset     = datasets[i][0],
            batch_size  = config.batch_size_train,
            shuffle     = True,
            num_workers = 0,
            pin_memory  = True
        )
        dataloader_val = torch.utils.data.DataLoader(
            dataset     = datasets[i][1],
            batch_size  = config.batch_size_val,
            shuffle     = True,
            num_workers = 0,
            pin_memory  = True
        )
        dataloader_test = torch.utils.data.DataLoader(
            dataset     = datasets[i][2],
            batch_size  = config.batch_size_test,
            shuffle     = True,
            num_workers = 0,
            pin_memory  = True
        )
        dataloaders.append([dataloader_train, dataloader_val, dataloader_test])

    search_data = []
    for lr in learning_rates:
        for dp in dropout_probs:
            for w in widths:
                for d in depths:
                    param_str = "lr: " + str(lr) + ", dp: " + str(dp) + ", w: " + str(w) + ", d: " + str(d)
                    final_val_losses = np.zeros(range(len(dataloaders)))
                    for system_num in range(len(dataloaders)):
                        model = models.create_new_model(lr, dp, [d, w])
                        data_dict = train.train(model, 0, dataloaders[system_num][0], dataloaders[system_num][1])
                        final_val_losses[system_num] = data_dict["Validation loss"][0]
                    final_val_losses_sum = np.sum(final_val_losses)
                    search_data.append({"str": param_str, "sum": final_val_losses_sum, "losses": final_val_losses})

    lowest_loss = np.inf
    best_params = ""
    for data_dict in search_data:
        print("Parameters:\t", data_dict["str"])
        print("Losses:\t",     data_dict["losses"])
        print("Loss sum:\t",   data_dict["sum"])
        print("\n")
        if data_dict["sum"] < lowest_loss:
            lowest_loss = data_dict["sum"]
            best_params = data_dict["str"]

    print("BEST PARAMETERS:")
    print(best_params)

########################################################################################################################

if __name__ == "__main__":
    main()

########################################################################################################################