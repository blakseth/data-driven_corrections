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

learning_rates = [1e-4, 1e-5]
dropout_probs = [0.0, 0.1, 0.2]
widths = (config.N_coarse * np.asarray([0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])).astype(int)
depths = [2, 3, 4, 5, 6, 7, 8, 9, 10]


########################################################################################################################

def main():
    os.makedirs(config.run_dir, exist_ok=False)
    # Load datasets and create dataloaders.
    dataset_paths = [
        [os.path.join(config.datasets_dir, 'system2B_sst_train.pt'),
        os.path.join(config.datasets_dir,  'system2B_sst_val.pt'),
        os.path.join(config.datasets_dir,  'system2B_sst_test.pt')],
        [os.path.join(config.datasets_dir, 'system5B_sst_train.pt'),
        os.path.join(config.datasets_dir,  'system5B_sst_val.pt'),
        os.path.join(config.datasets_dir,  'system5B_sst_test.pt')],
        [os.path.join(config.datasets_dir, 'system8B_sst_train.pt'),
        os.path.join(config.datasets_dir,  'system8B_sst_val.pt'),
        os.path.join(config.datasets_dir,  'system8B_sst_test.pt')]
    ]
    print("Paths:", dataset_paths)
    print("Paths[0][0]:", dataset_paths[0][0])
    dataloaders = []
    for i in range(len(dataset_paths)):
        train_set, val_set, test_set = load_datasets_from_path(dataset_paths[i][0], dataset_paths[i][1], dataset_paths[i][2])
        dataloader_train = torch.utils.data.DataLoader(
            dataset     = train_set,
            batch_size  = config.batch_size_train,
            shuffle     = True,
            num_workers = 0,
            pin_memory  = True
        )
        dataloader_val = torch.utils.data.DataLoader(
            dataset     = val_set,
            batch_size  = config.batch_size_val,
            shuffle     = True,
            num_workers = 0,
            pin_memory  = True
        )
        dataloader_test = torch.utils.data.DataLoader(
            dataset     = test_set,
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
                    print("Params:", param_str)
                    final_val_losses = np.zeros(len(dataloaders))
                    for system_num in range(len(dataloaders)):
                        print("System num:", system_num)
                        model = models.create_new_model(lr, dp, [d, w])
                        data_dict = train.train(model, 0, dataloaders[system_num][0], dataloaders[system_num][1])
                        final_val_losses[system_num] = data_dict["Validation loss"][1][-1]
                    final_val_losses_sum = np.sum(final_val_losses)
                    search_data.append({"str": param_str, "sum": final_val_losses_sum, "losses": final_val_losses})

    lowest_loss = np.inf
    best_params = ""
    with open(os.path.join(config.run_dir, "grid_search_results" + ".txt"), "w") as f:
        for data_dict in search_data:
            print("Parameters:\t", data_dict["str"])
            print("Losses:\t",     data_dict["losses"])
            print("Loss sum:\t",   data_dict["sum"])
            print("\n")
            f.write("Parameters:\t" + str(data_dict["str"])    + "\n")
            f.write("Losses:\t"     + str(data_dict["losses"]) + "\n")
            f.write("Loss sum:\t"   + str(data_dict["sum"])    + "\n")
            f.write("\n")
            if data_dict["sum"] < lowest_loss:
                lowest_loss = data_dict["sum"]
                best_params = data_dict["str"]

        print("BEST PARAMETERS:")
        print(best_params)

########################################################################################################################

if __name__ == "__main__":
    main()

########################################################################################################################