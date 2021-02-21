"""
datasets.py

Written by Sindre Stenen Blakseth

Creating training, validation and test datasets for ML correction of 1D heat conduction simulations.
"""

########################################################################################################################
# Package imports.

import matplotlib.pyplot as plt
import numpy as np
import numpy_indexed as npi
import os
import joblib
import torch
import torch.utils.data

########################################################################################################################
# File imports.

import config
import physics
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
    # Load pickled simulation data, or create and pickle new data if none exists already.
    save_filepath = os.path.join(datasets_location, data_tag + ".sav")
    if os.path.exists(save_filepath):
        simulation_data = joblib.load(save_filepath)
    else:
        unc_Ts    = np.zeros((config.Nt_coarse, config.N_coarse + 2))
        unc_Ts[0] = config.get_T0(config.nodes_coarse)
        ref_Ts    = np.zeros((config.Nt_coarse, config.N_coarse + 2))
        ref_Ts[0] = config.get_T0(config.nodes_coarse)
        ref_Ts_full    = np.zeros((config.Nt_coarse, config.N_fine + 2))
        ref_Ts_full[0] = config.get_T0(config.nodes_fine)
        idx = npi.indices(np.around(config.nodes_fine,   decimals=10),
                          np.around(config.nodes_coarse, decimals=10))
        for i in range(1, config.Nt_coarse):
            old_time = np.around(config.dt_coarse*(i-1), decimals=10)
            new_time = np.around(config.dt_coarse*i,     decimals=10)
            if i <= config.Nt_coarse * (config.N_train_examples + config.N_val_examples):
                unc_IC = ref_Ts[i-1]
            else:
                unc_IC = unc_Ts[i-1]
            unc_Ts[i] = physics.simulate(
                config.nodes_coarse, config.faces_coarse,
                unc_IC, config.get_T_a, config.get_T_b,
                config.get_k_approx, config.get_cV, config.rho, config.A,
                config.get_q_hat_approx, np.zeros_like(config.nodes_coarse[1:-1]),
                config.dt_coarse, old_time, new_time, False
            )
            if config.exact_solution_available:
                ref_Ts[i] = config.get_T_exact(config.nodes_coarse, new_time)
            else:
                ref_Ts_full[i] = physics.simulate(
                    config.nodes_fine, config.faces_fine,
                    ref_Ts_full[i-1], config.get_T_a, config.get_T_b,
                    config.get_k, config.get_cV, config.rho, config.A,
                    config.get_q_hat, np.zeros_like(config.nodes_fine[1:-1]),
                    config.dt_fine, old_time, new_time, False
                )
                for j in range(config.N_coarse + 2):
                    ref_Ts[i][j] = ref_Ts_full[i][idx[j]]

        # Calculate correction source terms.
        sources = np.zeros((config.Nt_coarse, config.N_coarse))
        for i in range(1, config.Nt_coarse): # Intentionally leaves the first entry all-zeros.
            old_time = np.around(config.dt_coarse * (i - 1), decimals=10)
            new_time = np.around(config.dt_coarse * i, decimals=10)
            sources[i] = physics.get_corrective_src_term(
                config.nodes_coarse, config.faces_coarse,
                ref_Ts[i], ref_Ts[i-1],
                config.get_T_a, config.get_T_b,
                config.get_k_approx, config.get_cV, config.rho, config.A, config.get_q_hat_approx,
                config.dt_coarse, old_time, False
            )
            corrected = physics.simulate(
                config.nodes_coarse, config.faces_coarse,
                ref_Ts[i-1], config.get_T_a, config.get_T_b,
                config.get_k_approx, config.get_cV, config.rho, config.A,
                config.get_q_hat_approx, sources[i],
                config.dt_coarse, old_time, new_time, False
            )
            np.testing.assert_allclose(corrected, ref_Ts[i], rtol=1e-10, atol=1e-15)
        print("Correction source terms generated and verified.")

        # Store data
        simulation_data = {
            'x':   config.nodes_coarse,
            'unc': unc_Ts,
            'ref': ref_Ts,
            'src': sources
        }
        joblib.dump(simulation_data, save_filepath)

    # Remove IC from data.
    simulation_data['unc'] = simulation_data['unc'][1:,:]
    simulation_data['ref'] = simulation_data['ref'][1:,:]
    simulation_data['src'] = simulation_data['src'][1:,:] # The entry removed here is all-zeros.

    # Split data into training, validation and test set.
    train_unc = simulation_data['unc'][:config.N_train_examples,:]
    train_ref = simulation_data['ref'][:config.N_train_examples,:]
    train_src = simulation_data['src'][:config.N_train_examples,:]

    val_unc   = simulation_data['unc'][config.N_train_examples:config.N_train_examples + config.N_val_examples,:]
    val_ref   = simulation_data['ref'][config.N_train_examples:config.N_train_examples + config.N_val_examples,:]
    val_src   = simulation_data['src'][config.N_train_examples:config.N_train_examples + config.N_val_examples,:]

    test_unc  = simulation_data['unc'][config.N_train_examples + config.N_val_examples:,:]
    test_ref  = simulation_data['ref'][config.N_train_examples + config.N_val_examples:,:]
    test_src  = simulation_data['src'][config.N_train_examples + config.N_val_examples:,:]

    assert train_unc.shape[0] == config.N_train_examples
    assert train_ref.shape[0] == config.N_train_examples
    assert train_src.shape[0] == config.N_train_examples

    assert val_unc.shape[0]   == config.N_val_examples
    assert val_ref.shape[0]   == config.N_val_examples
    assert val_src.shape[0]   == config.N_val_examples

    assert test_unc.shape[0]  == config.N_test_examples
    assert test_ref.shape[0]  == config.N_test_examples
    assert test_src.shape[0]  == config.N_test_examples

    # Calculate statistical properties of training data.
    train_unc_mean = np.mean(train_unc)
    train_ref_mean = np.mean(train_ref)
    train_src_mean = np.mean(train_src)

    train_unc_std  = np.std(train_unc)
    train_ref_std  = np.std(train_ref)
    train_src_std  = np.std(train_src)

    # Augment training data.
    if config.augment_training_data:
        # Shift augmentation.

        train_unc_orig = train_unc
        train_ref_orig = train_ref
        train_src_orig = train_src
        for i in range(config.N_shift_steps):
            # Uncorrected temperature
            train_unc_aug = train_unc_orig + (i + 1) * config.shift_step_size
            train_unc = np.concatenate((train_unc, train_unc_aug), axis=0)

            # Reference temperature
            train_ref_aug = train_ref_orig + (i + 1) * config.shift_step_size
            train_ref = np.concatenate((train_ref, train_ref_aug), axis=0)

            # Correction source term
            train_src = np.concatenate((train_src, train_src_orig), axis=0)

        # Mirror augmentation.

        # Uncorrected temperature
        train_unc_mirror = np.flip(train_unc, axis=1)
        train_unc = np.concatenate((train_unc, train_unc_mirror), axis=0)

        # Reference temperature
        train_ref_mirror = np.flip(train_ref, axis=1)
        train_ref = np.concatenate((train_ref, train_ref_mirror), axis=0)

        # Correction source term
        train_src_mirror = -np.flip(train_src, axis=1)
        train_src = np.concatenate((train_src, train_src_mirror), axis=0)

    # z_normalize data.
    train_unc_normalized = util.z_normalize(train_unc, train_unc_mean, train_unc_std)
    val_unc_normalized   = util.z_normalize(val_unc,   train_unc_mean, train_unc_std)
    test_unc_normalized  = util.z_normalize(test_unc,  train_unc_mean, train_unc_std)

    train_ref_normalized = util.z_normalize(train_ref, train_ref_mean, train_ref_std)
    val_ref_normalized   = util.z_normalize(val_ref,   train_ref_mean, train_ref_std)
    test_ref_normalized  = util.z_normalize(test_ref,  train_ref_mean, train_ref_std)

    train_src_normalized = util.z_normalize(train_src, train_src_mean, train_src_std)
    val_src_normalized   = util.z_normalize(val_src,   train_src_mean, train_src_std)
    test_src_normalized  = util.z_normalize(test_src,  train_src_mean, train_src_std)

    # Convert data from Numpy array to Torch tensor.
    train_unc_tensor = torch.from_numpy(train_unc_normalized)
    train_ref_tensor = torch.from_numpy(train_ref_normalized)
    train_src_tensor = torch.from_numpy(train_src_normalized)

    val_unc_tensor   = torch.from_numpy(val_unc_normalized)
    val_ref_tensor   = torch.from_numpy(val_ref_normalized)
    val_src_tensor   = torch.from_numpy(val_src_normalized)

    test_unc_tensor  = torch.from_numpy(test_unc_normalized)
    test_ref_tensor  = torch.from_numpy(test_ref_normalized)
    test_src_tensor  = torch.from_numpy(test_src_normalized)

    # Create array to store stats used for normalization.
    stats = np.asarray([
        train_unc_mean,
        train_ref_mean,
        train_src_mean,
        train_unc_std,
        train_ref_std,
        train_src_std
    ])

    # Pad with zeros to satisfy requirements of Torch's TensorDataset.
    # (Assumes that all datasets contain 6 or more data examples.)
    stats_train = np.zeros(train_unc.shape[0])
    stats_val   = np.zeros(val_unc.shape[0])
    stats_test  = np.zeros(test_unc.shape[0])
    stats_train[:6] = stats
    stats_val[:6]   = stats
    stats_test[:6]  = stats

    # Convert stats arrays to tensors
    stats_train_tensor = torch.from_numpy(stats_train)
    stats_val_tensor = torch.from_numpy(stats_val)
    stats_test_tensor = torch.from_numpy(stats_test)

    # Create datasets.
    dataset_train = torch.utils.data.TensorDataset(train_unc_tensor, train_ref_tensor, train_src_tensor, stats_train_tensor)
    dataset_val   = torch.utils.data.TensorDataset(val_unc_tensor,   val_ref_tensor,   val_src_tensor,   stats_val_tensor)
    dataset_test  = torch.utils.data.TensorDataset(test_unc_tensor,  test_ref_tensor,  test_src_tensor,  stats_test_tensor)

    return dataset_train, dataset_val, dataset_test

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

def load_datasets_from_path(train_path, val_path, test_path):
    if train_path is not None:
        try:
            dataset_train = torch.load(train_path)
        except:
            print("WARNING: Training dataset not found.")
            dataset_train = None
    if val_path is not None:
        try:
            dataset_val = torch.load(val_path)
        except:
            print("WARNING: Training dataset not found.")
            dataset_val = None
    if test_path is not None:
        try:
            dataset_test = torch.load(test_path)
        except:
            print("WARNING: Training dataset not found.")
            dataset_train = None
    return dataset_train, dataset_val, dataset_test


########################################################################################################################

def main():
    dataset_train, dataset_val, dataset_test = create_datasets()
    save_datasets(dataset_train, dataset_val, dataset_test)

########################################################################################################################

if __name__ == "__main__":
    main()

########################################################################################################################