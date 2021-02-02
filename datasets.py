"""
datasets.py

Written by Sindre Stenen Blakseth

Creating training, validation and test datasets for ML correction of 1D heat conduction simulations.
"""

########################################################################################################################
# Package imports.

import matplotlib.pyplot as plt
import numpy as np
import os
import torch
import torch.utils.data
import physics

########################################################################################################################
# File imports.

import config
import util

########################################################################################################################
# Data configurations.

datasets_location = config.datasets_dir
data_tag = config.data_tag

########################################################################################################################
# Create training and test datasets.

def create_datasets():
    """
    Purpose: Create datasets for supervised learning of data-driven correction models for the 1D heat equation.
    :return: dataset_train, dataset_val, dataset_test
    """
    # Load reference data.

    # Load uncorrected data.

    # Normalize data.

    # Split data into training, validation and test set.

    # Augment training data.

    # Convert data from Numpy array to Torch tensor.

    # Create datasets.

    return

########################################################################################################################
# Writing and loading datasets to/from disk.

def save_datasets(dataset_train, dataset_val, dataset_test):
    if dataset_train is not None:
        torch.save(dataset_train, os.path.join(datasets_location, data_tag + '_train.pt'))
    print("Saved training set to:", os.path.join(datasets_location, data_tag + '_train.pt'))
    if dataset_val is not None:
        torch.save(dataset_val,   os.path.join(datasets_location, data_tag + '_val.pt'  ))
    print("Saved validation set to:", os.path.join(datasets_location, data_tag + '_val.pt'))
    if dataset_test is not None:
        torch.save(dataset_test,  os.path.join(datasets_location, data_tag + '_test.pt' ))
    print("Saved test set to:", os.path.join(datasets_location, data_tag + '_test.pt'))

def load_datasets(load_train, load_val, load_test):
    if load_train:
        try:
            dataset_train = torch.load(os.path.join(datasets_location, data_tag + '_train.pt'))
        except:
            print("WARNING: Training dataset not found.")
            dataset_train = None
    else:
        dataset_train = None
    if load_val:
        try:
            dataset_val   = torch.load(os.path.join(datasets_location, data_tag + '_val.pt'  ))
        except:
            print("WARNING: Validation dataset not found.")
            dataset_val = None
    else:
        dataset_val = None
    if load_test:
        try:
            dataset_test  = torch.load(os.path.join(datasets_location, data_tag + '_test.pt' ))
        except:
            print("WARNING: Testing dataset not found.")
            dataset_test = None
    else:
        dataset_test = None
    return dataset_train, dataset_val, dataset_test

########################################################################################################################

def main():
    """
    plt.figure()
    plt.title("Temperature profiles from coarse-scale and fine-scale \n discretizations for " + r"$T_L = 250$ and $T_R = 400$")
    x = np.linspace(0, 1, endpoint = True)
    plt.plot(x, util.T_coarse(x, 250, 400), label='coarse')
    plt.plot(x, util.T_exact(x, 250, 400), label='fine')
    plt.xlabel('x')
    plt.ylabel('T')
    plt.legend()
    plt.grid()
    plt.savefig('ML_exp1_profiles.pdf')
    """
    #if config.data_tag == 'all_black':
    #    dataset_train, dataset_val, dataset_test = create_colour_datasets(colour = 'black')
    #elif config.data_tag == 'all_grey':
    #    dataset_train, dataset_val, dataset_test = create_colour_datasets(colour = 'grey')
    #elif config.data_tag == 'gradient':
    #    dataset_train, dataset_val, dataset_test = create_gradient_dataset()
    #elif config.data_tag == 'flickr_15k':
    #    dataset_train, dataset_val, dataset_test = create_flickr_dataset()
    #else:
    dataset_train, dataset_val, dataset_test = create_datasets()
    save_datasets(dataset_train, dataset_val, dataset_test)
    """
    dataset_train, _, dataset_test = load_datasets(True, False, True)
    dataloader_train = torch.utils.data.DataLoader(
        dataset=dataset_train,
        batch_size=8,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )
    for i, data in enumerate(dataloader_train):
        coarse = data[0]
        fine = data[1]
        scaled_coarse = util.unnormalize(coarse.data.numpy(), 250, 400)
        scaled_fine = util.unnormalize(fine.data.numpy(), 250, 400)
        #print("Coarse:", scaled_coarse)
        #print("Fine:", scaled_fine)
    """

########################################################################################################################

if __name__ == "__main__":
    main()

########################################################################################################################