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

import pandas as pd
import ppscore as pps

########################################################################################################################
# File imports.

import config
import exact_solver
import physics
import util

########################################################################################################################
# Create training and test datasets.

def create_parametrized_datasets(cfg):
    # Data config.
    datasets_location = cfg.datasets_dir
    data_tag = cfg.data_tag

    print("ALPHAS:", cfg.alphas)

    unc_Vs  = np.zeros((cfg.alphas.shape[0], cfg.N_t, 3, cfg.NJ ))
    ref_Vs  = np.zeros((cfg.alphas.shape[0], cfg.N_t, 3, cfg.NJ ))
    res_Vs  = np.zeros((cfg.alphas.shape[0], cfg.N_t, 3, cfg.NJ ))
    old_Vs  = np.zeros((cfg.alphas.shape[0], cfg.N_t, 3, cfg.NJ ))
    sources = np.zeros((cfg.alphas.shape[0], cfg.N_t, 3, cfg.N_x))
    sources_old = np.zeros((cfg.alphas.shape[0], cfg.N_t, 3, cfg.N_x))
    for a, alpha in enumerate(cfg.alphas):
        cfg.init_u2 = 0.1*alpha
        cfg.init_u1 = 0.1*alpha
        #print("V_mtx2:", physics.get_init_V_mtx(cfg))
        unc_Vs[a][0] = physics.get_init_V_mtx(cfg)
        ref_Vs[a,0,:,:] = physics.get_init_V_mtx(cfg)
        old_Vs[a][0] = physics.get_init_V_mtx(cfg)
        """
        print("ref_Vs[a][0]:", ref_Vs[a][0])
        print("ref_Vs[a][0].shape", ref_Vs[a][0].shape)
        print("ref_Vs[a,0,0,:]:", ref_Vs[a,0,0,:])
        print("ref_Vs[a][0][-1]:", ref_Vs[a][0][-1])
        print("ref_Vs[a][0][0][0]:", ref_Vs[a][0][0][0])
        print("ref_Vs[a][0]:", ref_Vs[a][0])
        plt.figure()
        plt.title("Initial condition")
        plt.plot(cfg.x_nodes, ref_Vs[a][0][0], label='p')
        plt.plot(cfg.x_nodes, ref_Vs[a][0][1], label='u')
        plt.plot(cfg.x_nodes, ref_Vs[a][0][2], label='T')
        plt.legend()
        plt.show()
        """

        # Residual at first time level is already set to zero, which is the correct value.
        for i in range(1, cfg.N_t):
            new_time = np.around(cfg.dt * i, decimals=10) # Assumes time start at 0.
            old_Vs[a][i] = ref_Vs[a][i-1]
            sources_old[a][i] = sources[a][i - 1]
            unc_Vs[a][i] = physics.get_new_state(cfg, old_Vs[a][i], np.zeros((3, cfg.N_x)), 'LxF')
            if cfg.exact_solution_available:
                ref_Vs[a][i] = exact_solver.exact_solver(cfg, new_time)
            else:
                raise Exception("Parametrized datasets currently require the exact solution to be available.")
            res_Vs[a][i] = ref_Vs[a][i] - unc_Vs[a][i]

            sources[a][i] = physics.get_corr_src_term(cfg, ref_Vs[a][i-1], ref_Vs[a][i], 'LxF')
            if i %50== 1 and a == 0:
                print("time:", new_time)
                plt.figure()
                plt.title("Corrective source terms")
                plt.plot(cfg.x_nodes[1:-1], sources[a][i][0], label='p')
                plt.plot(cfg.x_nodes[1:-1], sources[a][i][1], label='u')
                plt.plot(cfg.x_nodes[1:-1], sources[a][i][2], label='T')
                plt.figure()
                plt.title("Reference profiles")
                plt.plot(cfg.x_nodes, ref_Vs[a][i][0], label='p')
                plt.plot(cfg.x_nodes, ref_Vs[a][i][1], label='u')
                plt.plot(cfg.x_nodes, ref_Vs[a][i][2], label='T')
            corrected = physics.get_new_state(cfg, old_Vs[a][i], sources[a][i], 'LxF')
            np.testing.assert_allclose(corrected, ref_Vs[a][i], rtol=1e-10, atol=1e-10)
    plt.show()
    # Remove data for t=0 from datasets.
    ICs = np.delete(old_Vs,  obj=[0,1], axis=1)
    unc = np.delete(unc_Vs,  obj=[0,1], axis=1)
    ref = np.delete(ref_Vs,  obj=[0,1], axis=1)
    res = np.delete(res_Vs,  obj=[0,1], axis=1)
    src = np.delete(sources, obj=[0,1], axis=1)
    src_old = np.delete(sources_old, obj=[0,1], axis=1)
    times = np.linspace(cfg.dt*2, cfg.t_end, cfg.N_t - 2, endpoint=True)
    print("time[1]", times[1])
    print("3*dt", 3*cfg.dt)
    assert np.around(times[1], 10) == np.around(3 * cfg.dt, 10)
    assert src.shape == (cfg.alphas.shape[0], cfg.N_t - 2, 3, cfg.N_x)

    print("times.shape", times.shape)
    print("unc.shape", unc.shape)
    print("cfg.N_t", cfg.N_t)
    print("N_train_examples:", cfg.N_train_examples)

    # Split data into training, validation and testing sets.

    train_ICs = np.zeros((cfg.N_train_examples, 3, cfg.NJ ))
    train_unc = np.zeros((cfg.N_train_examples, 3, cfg.NJ ))
    train_ref = np.zeros((cfg.N_train_examples, 3, cfg.NJ ))
    train_res = np.zeros((cfg.N_train_examples, 3, cfg.NJ ))
    train_src = np.zeros((cfg.N_train_examples, 3, cfg.N_x))
    train_src_old = np.zeros((cfg.N_train_examples, 3, cfg.N_x))
    train_times = np.zeros(cfg.N_train_examples)
    train_alphas = []
    for a in range(cfg.N_train_alphas):
        print("a", a)
        print("ICs slice shape:", ICs[a, :, :, :].shape)
        print("diff:", (cfg.N_t - 2) * (a + 1) - (cfg.N_t - 2) * (a + 0))
        train_ICs[(cfg.N_t - 2) * (a + 0):(cfg.N_t - 2) * (a + 1), :, :] = ICs[a, :, :, :]
        train_unc[(cfg.N_t - 2) * (a + 0):(cfg.N_t - 2) * (a + 1), :, :] = unc[a, :, :, :]
        train_ref[(cfg.N_t - 2) * (a + 0):(cfg.N_t - 2) * (a + 1), :, :] = ref[a, :, :, :]
        train_res[(cfg.N_t - 2) * (a + 0):(cfg.N_t - 2) * (a + 1), :, :] = res[a, :, :, :]
        train_src[(cfg.N_t - 2) * (a + 0):(cfg.N_t - 2) * (a + 1), :, :] = src[a, :, :, :]
        train_src_old[(cfg.N_t - 2) * (a + 0):(cfg.N_t - 2) * (a + 1), :, :] = src_old[a, :, :, :]
        train_times[(cfg.N_t - 2) * (a + 0):(cfg.N_t - 2) * (a + 1)] = times
        train_alphas.append(cfg.alphas[a])
    train_alphas = np.asarray(train_alphas)
    print("train_ref[-1]:", train_ref[-1])

    val_ICs = np.zeros((cfg.N_val_examples, 3, cfg.NJ ))
    val_unc = np.zeros((cfg.N_val_examples, 3, cfg.NJ ))
    val_ref = np.zeros((cfg.N_val_examples, 3, cfg.NJ ))
    val_res = np.zeros((cfg.N_val_examples, 3, cfg.NJ ))
    val_src = np.zeros((cfg.N_val_examples, 3, cfg.N_x))
    val_src_old = np.zeros((cfg.N_val_examples, 3, cfg.N_x))
    val_times = np.zeros(cfg.N_val_examples)
    val_alphas = []
    for a in range(cfg.N_val_alphas):
        print("a", a)
        offset = cfg.N_train_alphas
        val_ICs[(cfg.N_t - 2) * (a + 0):(cfg.N_t - 2) * (a + 1), :, :] = ICs[a + offset, :, :, :]
        val_unc[(cfg.N_t - 2) * (a + 0):(cfg.N_t - 2) * (a + 1), :, :] = unc[a + offset, :, :, :]
        val_ref[(cfg.N_t - 2) * (a + 0):(cfg.N_t - 2) * (a + 1), :, :] = ref[a + offset, :, :, :]
        val_res[(cfg.N_t - 2) * (a + 0):(cfg.N_t - 2) * (a + 1), :, :] = res[a + offset, :, :, :]
        val_src[(cfg.N_t - 2) * (a + 0):(cfg.N_t - 2) * (a + 1), :, :] = src[a + offset, :, :, :]
        val_src_old[(cfg.N_t - 2) * (a + 0):(cfg.N_t - 2) * (a + 1), :, :] = src_old[a + offset, :, :, :]
        val_times[(cfg.N_t - 2) * (a + 0):(cfg.N_t - 2) * (a + 1)] = times
        val_alphas.append(cfg.alphas[a + offset])
    val_alphas = np.asarray(val_alphas)

    test_ICs = np.zeros((cfg.N_test_examples, 3, cfg.NJ ))
    test_unc = np.zeros((cfg.N_test_examples, 3, cfg.NJ ))
    test_ref = np.zeros((cfg.N_test_examples, 3, cfg.NJ ))
    test_res = np.zeros((cfg.N_test_examples, 3, cfg.NJ ))
    test_src = np.zeros((cfg.N_test_examples, 3, cfg.N_x))
    test_src_old = np.zeros((cfg.N_test_examples, 3, cfg.N_x))
    test_times = np.zeros(cfg.N_test_examples)
    test_alphas = []
    for a in range(cfg.N_test_alphas):
        print("a", a)
        offset = cfg.N_train_alphas + cfg.N_val_alphas
        test_ICs[(cfg.N_t - 2) * (a + 0):(cfg.N_t - 2) * (a + 1), :, :] = ICs[a + offset, :, :, :]
        test_unc[(cfg.N_t - 2) * (a + 0):(cfg.N_t - 2) * (a + 1), :, :] = unc[a + offset, :, :, :]
        test_ref[(cfg.N_t - 2) * (a + 0):(cfg.N_t - 2) * (a + 1), :, :] = ref[a + offset, :, :, :]
        test_res[(cfg.N_t - 2) * (a + 0):(cfg.N_t - 2) * (a + 1), :, :] = res[a + offset, :, :, :]
        test_src[(cfg.N_t - 2) * (a + 0):(cfg.N_t - 2) * (a + 1), :, :] = src[a + offset, :, :, :]
        test_src_old[(cfg.N_t - 2) * (a + 0):(cfg.N_t - 2) * (a + 1), :, :] = src_old[a + offset, :, :, :]
        test_times[(cfg.N_t - 2) * (a + 0):(cfg.N_t - 2) * (a + 1)] = times
        test_alphas.append(cfg.alphas[a + offset])
    test_alphas = np.asarray(test_alphas)
    print("train_alphas", train_alphas)
    print("val_alphas", val_alphas)
    print("test_alphas", test_alphas)

    # Augment training data.

    if cfg.augment_training_data:
        # Shift augmentation.

        train_ICs_orig = train_ICs.copy()
        train_unc_orig = train_unc.copy()
        train_ref_orig = train_ref.copy()
        train_res_orig = train_res.copy()
        train_src_orig = train_src.copy()
        train_src_old_orig = train_src_old.copy()
        train_times_orig = train_times.copy()
        for i in range(cfg.N_shift_steps):
            # IC temperature
            train_ICs_aug = train_ICs_orig + (i + 1) * cfg.shift_step_size
            train_ICs = np.concatenate((train_ICs, train_ICs_aug), axis=0)

            # Uncorrected temperature
            train_unc_aug = train_unc_orig + (i + 1) * cfg.shift_step_size
            train_unc = np.concatenate((train_unc, train_unc_aug), axis=0)

            # Reference temperature
            train_ref_aug = train_ref_orig + (i + 1) * cfg.shift_step_size
            train_ref = np.concatenate((train_ref, train_ref_aug), axis=0)

            # Residual temperature
            train_res_aug = train_res_orig + (i + 1) * cfg.shift_step_size
            train_res = np.concatenate((train_res, train_res_aug), axis=0)

            # Correction source term
            train_src = np.concatenate((train_src, train_src_orig), axis=0)
            train_src_old = np.concatenate((train_src_old, train_src_old_orig), axis=0)

            # Time levels
            train_times = np.concatenate((train_times, train_times_orig), axis=0)

        # Mirror augmentation.

        # IC temperature
        train_ICs_mirror = np.flip(train_ICs, axis=1).copy()
        train_ICs = np.concatenate((train_ICs, train_ICs_mirror), axis=0)

        # Uncorrected temperature
        train_unc_mirror = np.flip(train_unc, axis=1).copy()
        train_unc = np.concatenate((train_unc, train_unc_mirror), axis=0)

        # Reference temperature
        train_ref_mirror = np.flip(train_ref, axis=1).copy()
        train_ref = np.concatenate((train_ref, train_ref_mirror), axis=0)

        # Residual temperature
        train_res_mirror = np.flip(train_res, axis=1).copy()
        train_res = np.concatenate((train_res, train_res_mirror), axis=0)

        # Correction source term
        train_src_mirror = np.flip(train_src, axis=1).copy()
        train_src = np.concatenate((train_src, train_src_mirror), axis=0)
        train_src_old_mirror = np.flip(train_src_old, axis=1).copy()
        train_src_old = np.concatenate((train_src_old, train_src_old_mirror), axis=0)

        # Time levels
        train_times = np.concatenate((train_times, train_times), axis=0)

    # Calculate statistical properties of training data.
    ref_means = []
    src_means = []
    ref_stds  = []
    src_stds  = []
    for i in range(3):
        ref_means.append(0.0)
        src_means.append(0.0)
        ref_stds.append(1.0)
        src_stds.append(1.0)

    plt.figure()
    plt.title("Source before normalization")
    plt.plot(cfg.x_nodes[1:-1], train_src[0, 0, :], label='p')
    plt.plot(cfg.x_nodes[1:-1], train_src[0, 1, :], label='u')
    plt.plot(cfg.x_nodes[1:-1], train_src[0, 2, :], label='T')
    plt.legend()

    # z_normalize data.
    train_unc_normalized = util.z_normalize_componentwise(train_unc, ref_means, ref_stds)
    val_unc_normalized   = util.z_normalize_componentwise(val_unc,   ref_means, ref_stds)
    test_unc_normalized  = util.z_normalize_componentwise(test_unc,  ref_means, ref_stds)

    train_ref_normalized = util.z_normalize_componentwise(train_ref, ref_means, ref_stds)
    val_ref_normalized   = util.z_normalize_componentwise(val_ref,   ref_means, ref_stds)
    test_ref_normalized  = util.z_normalize_componentwise(test_ref,  ref_means, ref_stds)

    train_res_normalized = util.z_normalize_componentwise(train_res, ref_means, ref_stds)
    val_res_normalized   = util.z_normalize_componentwise(val_res,   ref_means, ref_stds)
    test_res_normalized  = util.z_normalize_componentwise(test_res,  ref_means, ref_stds)

    train_src_normalized = util.z_normalize_componentwise(train_src, src_means, src_stds)
    val_src_normalized   = util.z_normalize_componentwise(val_src,   src_means, src_stds)
    test_src_normalized  = util.z_normalize_componentwise(test_src,  src_means, src_stds)

    print("\n\nSRC MEAN:", src_means, "\n\n")

    plt.figure()
    plt.title("Source after normalization")
    plt.plot(cfg.x_nodes[1:-1], train_src_normalized[0, 0, :], label='p')
    plt.plot(cfg.x_nodes[1:-1], train_src_normalized[0, 1, :], label='u')
    plt.plot(cfg.x_nodes[1:-1], train_src_normalized[0, 2, :], label='T')
    plt.legend()

    train_src_old_normalized = util.z_normalize_componentwise(train_src_old, src_means, src_stds)
    val_src_old_normalized = util.z_normalize_componentwise(val_src_old, src_means, src_stds)
    test_src_old_normalized = util.z_normalize_componentwise(test_src_old, src_means, src_stds)

    # Note that the ICs are not to be used in conjunction with the NN directly,
    # so there is no need to normalize them. Same goes for time levels.

    # Convert data from Numpy array to Torch tensor.
    train_ICs_tensor = torch.from_numpy(train_ICs)
    train_unc_tensor = torch.from_numpy(train_unc_normalized)
    train_ref_tensor = torch.from_numpy(train_ref_normalized)
    train_res_tensor = torch.from_numpy(train_res_normalized)
    train_src_tensor = torch.from_numpy(train_src_normalized)
    train_src_old_tensor = torch.from_numpy(train_src_old_normalized)
    train_times_tensor = torch.from_numpy(train_times)

    val_ICs_tensor = torch.from_numpy(val_ICs)
    val_unc_tensor = torch.from_numpy(val_unc_normalized)
    val_ref_tensor = torch.from_numpy(val_ref_normalized)
    val_res_tensor = torch.from_numpy(val_res_normalized)
    val_src_tensor = torch.from_numpy(val_src_normalized)
    val_src_old_tensor = torch.from_numpy(val_src_old_normalized)
    val_times_tensor = torch.from_numpy(val_times)

    test_ICs_tensor = torch.from_numpy(test_ICs)
    test_unc_tensor = torch.from_numpy(test_unc_normalized)
    test_ref_tensor = torch.from_numpy(test_ref_normalized)
    test_res_tensor = torch.from_numpy(test_res_normalized)
    test_src_tensor = torch.from_numpy(test_src_normalized)
    test_src_old_tensor = torch.from_numpy(test_src_old_normalized)
    test_times_tensor = torch.from_numpy(test_times)

    # Create array to store stats used for normalization.
    stats = np.asarray([
        np.asarray(ref_means),
        np.asarray(ref_stds),
        np.asarray(src_means),
        np.asarray(src_stds),
    ])

    # Pad with zeros to satisfy requirements of Torch's TensorDataset.
    # (Assumes that all datasets contain 6 or more data examples.)
    assert train_unc.shape[0] >= stats.shape[0] and val_unc.shape[0] >= stats.shape[0] and test_unc.shape[0] >= stats.shape[0]
    stats_train = np.zeros((train_unc.shape[0], 3))
    stats_val = np.zeros((val_unc.shape[0], 3))
    stats_test = np.zeros((test_unc.shape[0], 3))
    stats_train[:stats.shape[0],:] = stats
    stats_val[:stats.shape[0],:] = stats
    stats_test[:stats.shape[0],:] = stats

    # Convert stats arrays to tensors
    stats_train_tensor = torch.from_numpy(stats_train)
    stats_val_tensor = torch.from_numpy(stats_val)
    stats_test_tensor = torch.from_numpy(stats_test)

    assert train_unc.shape[0] >= train_alphas.shape[0] and val_unc.shape[0] >= val_alphas.shape[0] and test_unc.shape[0] >= test_alphas.shape[0]
    train_alphas_padded = np.zeros(train_unc.shape[0])
    val_alphas_padded   = np.zeros(val_unc.shape[0])
    test_alphas_padded  = np.zeros(test_unc.shape[0])
    train_alphas_padded[:train_alphas.shape[0]] = train_alphas
    val_alphas_padded[:val_alphas.shape[0]]     = val_alphas
    test_alphas_padded[:test_alphas.shape[0]]   = test_alphas
    print("test_alphas_padded[:10]", test_alphas_padded[:10])

    # Convert alpha arrays to tensors
    train_alphas_tensor = torch.from_numpy(train_alphas_padded)
    val_alphas_tensor   = torch.from_numpy(val_alphas_padded)
    test_alphas_tensor  = torch.from_numpy(test_alphas_padded)

    # Create datasets.
    dataset_train = torch.utils.data.TensorDataset(train_unc_tensor,    train_ref_tensor, train_src_tensor,
                                                   stats_train_tensor,  train_ICs_tensor, train_times_tensor,
                                                   train_alphas_tensor, train_res_tensor, train_src_old_tensor)
    dataset_val = torch.utils.data.TensorDataset(  val_unc_tensor,      val_ref_tensor,   val_src_tensor,
                                                   stats_val_tensor,    val_ICs_tensor,   val_times_tensor,
                                                   val_alphas_tensor,   val_res_tensor,   val_src_old_tensor)
    dataset_test = torch.utils.data.TensorDataset( test_unc_tensor,     test_ref_tensor,  test_src_tensor,
                                                   stats_test_tensor,   test_ICs_tensor,  test_times_tensor,
                                                   test_alphas_tensor,  test_res_tensor,  test_src_old_tensor)

    train_src = train_src_tensor.detach().numpy()
    train_src_old = train_src_old_tensor.detach().numpy()

    ppscores = []
    for i in range(0, train_src.shape[0], 10):
        train_src_flat = train_src[i].flatten()
        train_src_old_flat = train_src_old[i].flatten()
        df = pd.DataFrame()
        df["x"] = train_src_flat.tolist()
        df["y"] = train_src_old_flat.tolist()
        ppscores.append(pps.score(df, "x", "y")['ppscore'])
    print("Mean ppscore =", np.mean(np.asarray(ppscores)))

    plt.figure()
    plt.title("Source normalized")
    plt.plot(cfg.x_nodes[1:-1], train_src[0,0,:], label='p')
    plt.plot(cfg.x_nodes[1:-1], train_src[0,1,:], label='u')
    plt.plot(cfg.x_nodes[1:-1], train_src[0,2,:], label='T')
    plt.legend()
    plt.figure()
    plt.title("Source old normalized")
    plt.plot(cfg.x_nodes[1:-1], train_src_old[0, 0, :], label='p')
    plt.plot(cfg.x_nodes[1:-1], train_src_old[0, 1, :], label='u')
    plt.plot(cfg.x_nodes[1:-1], train_src_old[0, 2, :], label='T')
    plt.legend()

    train_src_unnorm = util.z_unnormalize_componentwise(train_src, src_means, src_stds)
    train_src_old_unnorm = util.z_unnormalize_componentwise(train_src_old, src_means, src_stds)

    plt.figure()
    plt.title("Source")
    plt.plot(cfg.x_nodes[1:-1], train_src_unnorm[0, 0, :], label='p')
    plt.plot(cfg.x_nodes[1:-1], train_src_unnorm[0, 1, :], label='u')
    plt.plot(cfg.x_nodes[1:-1], train_src_unnorm[0, 2, :], label='T')
    plt.legend()
    plt.figure()
    plt.title("Source old")
    plt.plot(cfg.x_nodes[1:-1], train_src_old_unnorm[0, 0, :], label='p')
    plt.plot(cfg.x_nodes[1:-1], train_src_old_unnorm[0, 1, :], label='u')
    plt.plot(cfg.x_nodes[1:-1], train_src_old_unnorm[0, 2, :], label='T')
    plt.legend()

    plt.show()

    return dataset_train, dataset_val, dataset_test

def create_datasets(cfg):
    """
    Purpose: Create datasets for supervised learning of data-driven correction models for the 1D heat equation.
    :return: dataset_train, dataset_val, dataset_test
    """

    # Data config.
    datasets_location = cfg.datasets_dir
    data_tag = cfg.data_tag

    # Load pickled simulation data, or create and pickle new data if none exists already.
    save_filepath = os.path.join(datasets_location, data_tag + ".sav")
    if os.path.exists(save_filepath) and False:
        simulation_data = joblib.load(save_filepath)
    else:
        unc_Ts    = np.zeros((cfg.Nt_coarse, cfg.N_coarse + 2))
        unc_Ts[0] = cfg.get_T0(cfg.nodes_coarse)
        ref_Ts    = np.zeros((cfg.Nt_coarse, cfg.N_coarse + 2))
        ref_Ts[0] = cfg.get_T0(cfg.nodes_coarse)
        IC_Ts     = np.zeros((cfg.Nt_coarse, cfg.N_coarse + 2))
        IC_Ts[0]  = cfg.get_T0(cfg.nodes_coarse)
        ref_Ts_full    = np.zeros((cfg.Nt_coarse, cfg.N_fine + 2))
        ref_Ts_full[0] = cfg.get_T0(cfg.nodes_fine)
        idx = npi.indices(np.around(cfg.nodes_fine,   decimals=10),
                          np.around(cfg.nodes_coarse, decimals=10))
        for i in range(1, cfg.Nt_coarse):
            old_time = np.around(cfg.dt_coarse*(i-1), decimals=10)
            new_time = np.around(cfg.dt_coarse*i,     decimals=10)
            if i <= cfg.Nt_coarse * (cfg.N_train_examples + cfg.N_val_examples) or (not cfg.do_simulation_test):
                unc_IC = ref_Ts[i-1]
            else:
                unc_IC = unc_Ts[i-1]
            IC_Ts[i]  = unc_IC
            unc_Ts[i] = physics.simulate(
                cfg.nodes_coarse, cfg.faces_coarse,
                unc_IC, cfg.get_T_a, cfg.get_T_b,
                cfg.get_k_approx, cfg.get_cV, cfg.rho, cfg.A,
                cfg.get_q_hat_approx, np.zeros_like(cfg.nodes_coarse[1:-1]),
                cfg.dt_coarse, old_time, new_time, False
            )
            if cfg.exact_solution_available:
                ref_Ts[i] = cfg.get_T_exact(cfg.nodes_coarse, new_time)
            else:
                ref_Ts_full[i] = physics.simulate(
                    cfg.nodes_fine, cfg.faces_fine,
                    ref_Ts_full[i-1], cfg.get_T_a, cfg.get_T_b,
                    cfg.get_k, cfg.get_cV, cfg.rho, cfg.A,
                    cfg.get_q_hat, np.zeros_like(cfg.nodes_fine[1:-1]),
                    cfg.dt_fine, old_time, new_time, False
                )
                for j in range(cfg.N_coarse + 2):
                    ref_Ts[i][j] = ref_Ts_full[i][idx[j]]

        # Calculate correction source terms.
        sources = np.zeros((cfg.Nt_coarse, cfg.N_coarse))
        for i in range(1, cfg.Nt_coarse): # Intentionally leaves the first entry all-zeros.
            old_time = np.around(cfg.dt_coarse * (i - 1), decimals=10)
            new_time = np.around(cfg.dt_coarse * i, decimals=10)
            sources[i] = physics.get_corrective_src_term(
                cfg.nodes_coarse, cfg.faces_coarse,
                ref_Ts[i], ref_Ts[i-1],
                cfg.get_T_a, cfg.get_T_b,
                cfg.get_k_approx, cfg.get_cV, cfg.rho, cfg.A, cfg.get_q_hat_approx,
                cfg.dt_coarse, old_time, False
            )
            corrected = physics.simulate(
                cfg.nodes_coarse, cfg.faces_coarse,
                ref_Ts[i-1], cfg.get_T_a, cfg.get_T_b,
                cfg.get_k_approx, cfg.get_cV, cfg.rho, cfg.A,
                cfg.get_q_hat_approx, sources[i],
                cfg.dt_coarse, old_time, new_time, False
            )
            np.testing.assert_allclose(corrected, ref_Ts[i], rtol=1e-10, atol=1e-10)
        print("Correction source terms generated and verified.")

        # Store data
        simulation_data = {
            'x':   cfg.nodes_coarse,
            'ICs': IC_Ts,
            'unc': unc_Ts,
            'ref': ref_Ts,
            'src': sources
        }
        joblib.dump(simulation_data, save_filepath)

    # Remove data for t=0 from datasets.
    ICs = simulation_data['ICs'][1:,:]
    unc = simulation_data['unc'][1:,:]
    ref = simulation_data['ref'][1:,:]
    src = simulation_data['src'][1:,:] # The entry removed here is all-zeros.
    times = np.linspace(cfg.dt_coarse, cfg.t_end, cfg.Nt_coarse - 1, endpoint=True)
    assert times[1] == 2*cfg.dt_coarse

    # Shuffle data.
    assert ICs.shape[0] == unc.shape[0] == ref.shape[0] == src.shape[0] == times.shape[0]
    permutation = np.random.permutation(ICs.shape[0])
    ICs = ICs[permutation]
    unc = unc[permutation]
    ref = ref[permutation]
    src = src[permutation]
    times = times[permutation]

    # Split data into training, validation and test set.
    train_ICs = ICs[:cfg.N_train_examples, :]
    train_unc = unc[:cfg.N_train_examples,:]
    train_ref = ref[:cfg.N_train_examples,:]
    train_src = src[:cfg.N_train_examples,:]
    train_times = times[:cfg.N_train_examples]

    val_ICs   = ICs[cfg.N_train_examples:cfg.N_train_examples + cfg.N_val_examples,:]
    val_unc   = unc[cfg.N_train_examples:cfg.N_train_examples + cfg.N_val_examples,:]
    val_ref   = ref[cfg.N_train_examples:cfg.N_train_examples + cfg.N_val_examples,:]
    val_src   = src[cfg.N_train_examples:cfg.N_train_examples + cfg.N_val_examples,:]
    val_times = times[cfg.N_train_examples:cfg.N_train_examples + cfg.N_val_examples]

    test_ICs  = ICs[cfg.N_train_examples + cfg.N_val_examples:,:]
    test_unc  = unc[cfg.N_train_examples + cfg.N_val_examples:,:]
    test_ref  = ref[cfg.N_train_examples + cfg.N_val_examples:,:]
    test_src  = src[cfg.N_train_examples + cfg.N_val_examples:,:]
    test_times = times[cfg.N_train_examples + cfg.N_val_examples:]

    assert train_ICs.shape[0] == cfg.N_train_examples
    assert train_unc.shape[0] == cfg.N_train_examples
    assert train_ref.shape[0] == cfg.N_train_examples
    assert train_src.shape[0] == cfg.N_train_examples
    assert train_times.shape[0] == cfg.N_train_examples

    assert val_ICs.shape[0]   == cfg.N_val_examples
    assert val_unc.shape[0]   == cfg.N_val_examples
    assert val_ref.shape[0]   == cfg.N_val_examples
    assert val_src.shape[0]   == cfg.N_val_examples
    assert val_times.shape[0] == cfg.N_val_examples

    assert test_ICs.shape[0]  == cfg.N_test_examples
    assert test_unc.shape[0]  == cfg.N_test_examples
    assert test_ref.shape[0]  == cfg.N_test_examples
    assert test_src.shape[0]  == cfg.N_test_examples
    assert test_times.shape[0] == cfg.N_test_examples

    # Augment training data.
    if cfg.augment_training_data:
        # Shift augmentation.

        train_ICs_orig = train_ICs.copy()
        train_unc_orig = train_unc.copy()
        train_ref_orig = train_ref.copy()
        train_src_orig = train_src.copy()
        train_times_orig = train_times.copy()
        for i in range(cfg.N_shift_steps):
            # IC temperature
            train_ICs_aug = train_ICs_orig + (i + 1) * cfg.shift_step_size
            train_ICs = np.concatenate((train_ICs, train_ICs_aug), axis=0)

            # Uncorrected temperature
            train_unc_aug = train_unc_orig + (i + 1) * cfg.shift_step_size
            train_unc = np.concatenate((train_unc, train_unc_aug), axis=0)

            # Reference temperature
            train_ref_aug = train_ref_orig + (i + 1) * cfg.shift_step_size
            train_ref = np.concatenate((train_ref, train_ref_aug), axis=0)

            # Correction source term
            train_src = np.concatenate((train_src, train_src_orig), axis=0)

            # Time levels
            train_times = np.concatenate((train_times, train_times_orig), axis=0)

        # Mirror augmentation.

        # IC temperature
        train_ICs_mirror = np.flip(train_ICs, axis=1).copy()
        train_ICs = np.concatenate((train_ICs, train_ICs_mirror), axis=0)

        # Uncorrected temperature
        train_unc_mirror = np.flip(train_unc, axis=1).copy()
        train_unc = np.concatenate((train_unc, train_unc_mirror), axis=0)

        # Reference temperature
        train_ref_mirror = np.flip(train_ref, axis=1).copy()
        train_ref = np.concatenate((train_ref, train_ref_mirror), axis=0)

        # Correction source term
        train_src_mirror = np.flip(train_src, axis=1).copy()
        train_src = np.concatenate((train_src, train_src_mirror), axis=0)

        # Time levels
        train_times = np.concatenate((train_times, train_times), axis=0)

    # Calculate statistical properties of training data.
    train_unc_mean = np.mean(train_unc)
    train_ref_mean = np.mean(train_ref)
    train_src_mean = np.mean(train_src)

    train_unc_std = np.std(train_unc)
    train_ref_std = np.std(train_ref)
    train_src_std = np.std(train_src)

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

    # Note that the ICs are not to be used in conjunction with the NN directly,
    # so there is no need to normalize them. Same goes for time levels.

    # Convert data from Numpy array to Torch tensor.
    train_ICs_tensor = torch.from_numpy(train_ICs)
    train_unc_tensor = torch.from_numpy(train_unc_normalized)
    train_ref_tensor = torch.from_numpy(train_ref_normalized)
    train_src_tensor = torch.from_numpy(train_src_normalized)
    train_times_tensor = torch.from_numpy(train_times)

    val_ICs_tensor   = torch.from_numpy(val_ICs)
    val_unc_tensor   = torch.from_numpy(val_unc_normalized)
    val_ref_tensor   = torch.from_numpy(val_ref_normalized)
    val_src_tensor   = torch.from_numpy(val_src_normalized)
    val_times_tensor = torch.from_numpy(val_times)

    test_ICs_tensor  = torch.from_numpy(test_ICs)
    test_unc_tensor  = torch.from_numpy(test_unc_normalized)
    test_ref_tensor  = torch.from_numpy(test_ref_normalized)
    test_src_tensor  = torch.from_numpy(test_src_normalized)
    test_times_tensor = torch.from_numpy(test_times)

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
    assert train_unc.shape[0] >= 6 and val_unc.shape[0] >= 6 and test_unc.shape[0] >= 6
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
    dataset_train = torch.utils.data.TensorDataset(train_unc_tensor,   train_ref_tensor, train_src_tensor,
                                                   stats_train_tensor, train_ICs_tensor, train_times_tensor)
    dataset_val   = torch.utils.data.TensorDataset(val_unc_tensor,     val_ref_tensor,   val_src_tensor,
                                                   stats_val_tensor,   val_ICs_tensor,   val_times_tensor)
    dataset_test  = torch.utils.data.TensorDataset(test_unc_tensor,    test_ref_tensor,  test_src_tensor,
                                                   stats_test_tensor,  test_ICs_tensor,  test_times_tensor)

    return dataset_train, dataset_val, dataset_test

########################################################################################################################
# Writing and loading datasets to/from disk.

def save_datasets(cfg, dataset_train, dataset_val, dataset_test):
    datasets_location = cfg.datasets_dir
    data_tag = cfg.data_tag
    if dataset_train is not None:
        torch.save(dataset_train, os.path.join(datasets_location, data_tag + '_train.pt'))
    print("Saved training set to:", os.path.join(datasets_location, data_tag + '_train.pt'))
    if dataset_val is not None:
        torch.save(dataset_val,   os.path.join(datasets_location, data_tag + '_val.pt'  ))
    print("Saved validation set to:", os.path.join(datasets_location, data_tag + '_val.pt'))
    if dataset_test is not None:
        torch.save(dataset_test,  os.path.join(datasets_location, data_tag + '_test.pt' ))
    print("Saved test set to:", os.path.join(datasets_location, data_tag + '_test.pt'))

def load_datasets(cfg, load_train, load_val, load_test):
    datasets_location = cfg.datasets_dir
    data_tag = cfg.data_tag
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
            dataset_test = None
    return dataset_train, dataset_val, dataset_test


########################################################################################################################

def main(cfg):
    if cfg.parametrized_system:
        dataset_train, dataset_val, dataset_test = create_parametrized_datasets(cfg)
    else:
        dataset_train, dataset_val, dataset_test = create_datasets(cfg)
    save_datasets(cfg, dataset_train, dataset_val, dataset_test)

########################################################################################################################

if __name__ == "__main__":
    configuration = config.Config(None)
    main(configuration)

########################################################################################################################