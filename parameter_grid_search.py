"""
parameter_grid_search.py

Written by Sindre Stenen Blakseth, 2021.

Script for performing parameter grid search.
"""

########################################################################################################################
# Package imports.

import os
import torch

########################################################################################################################
# File imports.

import config
import datasets

########################################################################################################################
# Parameter space.

learning_rates = [1e-4, 3e-4, 1e-5]
dropout_probs = [0.0, 0.1, 0.2]


########################################################################################################################

def main():
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

    for system_index in range(len(dataset_paths)):
        system_dataset_paths = dataset_paths[system_index]
        dataset_train, dataset_val, _ = datasets.load_datasets_from_path(system_dataset_paths[0], system_dataset_paths[1], None)
        dataloader_train = torch.utils.data.DataLoader(
            dataset=dataset_train,
            batch_size=config.batch_size_train,
            shuffle=True,
            num_workers=0,
            pin_memory=True
        )
        dataloader_val = torch.utils.data.DataLoader(
            dataset=dataset_val,
            batch_size=config.batch_size_val,
            shuffle=True,
            num_workers=0,
            pin_memory=True
        )


########################################################################################################################

if __name__ == "__main__":
    main()

########################################################################################################################