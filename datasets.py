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
# Create training and test datasets.

def create_parametrized_datasets(cfg):
    # Data config.
    datasets_location = cfg.datasets_dir
    data_tag = cfg.data_tag

    print("ALPHAS:", cfg.alphas)

    unc_Ts = np.zeros((cfg.alphas.shape[0], cfg.N_t, cfg.N_x + 2, cfg.N_y + 2))
    ref_Ts = np.zeros((cfg.alphas.shape[0], cfg.N_t, cfg.N_x + 2, cfg.N_y + 2))
    res_Ts = np.zeros((cfg.alphas.shape[0], cfg.N_t, cfg.N_x + 2, cfg.N_y + 2))
    IC_Ts = np.zeros((cfg.alphas.shape[0],  cfg.N_t, cfg.N_x + 2, cfg.N_y + 2))
    sources = np.zeros((cfg.alphas.shape[0], cfg.N_t, cfg.N_x, cfg.N_y))
    for a, alpha in enumerate(cfg.alphas):
        unc_Ts[a][0] = cfg.get_T0(cfg.x_nodes, cfg.y_nodes, alpha)
        ref_Ts[a][0] = cfg.get_T0(cfg.x_nodes, cfg.y_nodes, alpha)
        IC_Ts[a][0]  = cfg.get_T0(cfg.x_nodes, cfg.y_nodes, alpha)
        # Residual at first time level is already set to zero, which is the correct value.
        for i in range(1, cfg.N_t):
            old_time = np.around(cfg.dt * (i - 1), decimals=10)
            new_time = np.around(cfg.dt * i, decimals=10)
            unc_IC = ref_Ts[a][i-1]
            IC_Ts[a][i] = unc_IC
            unc_Ts[a][i] = physics.simulate_2D(
                cfg, unc_IC, old_time, new_time, alpha,
                cfg.get_q_hat_approx,
                np.zeros((cfg.x_nodes[1:-1].shape[0], cfg.y_nodes[1:-1].shape[0]))
            )
            if cfg.exact_solution_available:
                ref_Ts[a][i] = cfg.get_T_exact(cfg.x_nodes, cfg.y_nodes, new_time, alpha)
            else:
                raise Exception("Parametrized datasets currently requires the exact solution to be available.")
            res_Ts[a][i] = ref_Ts[a][i] - unc_Ts[a][i]

        for i in range(1, cfg.N_t): # Intentionally leaves the first entry all-zeros.
            old_time = np.around(cfg.dt * (i - 1), decimals=10)
            new_time = np.around(cfg.dt * i, decimals=10)
            sources[a][i] = physics.get_corrective_src_term_2D(
                cfg, ref_Ts[a][i-1], ref_Ts[a][i], old_time, alpha, cfg.get_q_hat_approx
            )
            corrected = physics.simulate_2D(
                cfg, ref_Ts[a][i-1], old_time, new_time, alpha, cfg.get_q_hat_approx, sources[a][i]
            )
            np.testing.assert_allclose(corrected, ref_Ts[a][i], rtol=1e-10, atol=1e-10)

    # Store data
    simulation_data = {
        'x': cfg.x_nodes,
        'y': cfg.y_nodes,
        'ICs': IC_Ts,
        'unc': unc_Ts,
        'ref': ref_Ts,
        'res': res_Ts,
        'src': sources,
        'alphas': cfg.alphas
    }
    print()
    save_filepath = os.path.join(datasets_location, data_tag + ".sav")
    #joblib.dump(simulation_data, save_filepath)

    # Remove data for t=0 from datasets.
    ICs = np.delete(simulation_data['ICs'], obj=0, axis=1)
    unc = np.delete(simulation_data['unc'], obj=0, axis=1)
    ref = np.delete(simulation_data['ref'], obj=0, axis=1)
    res = np.delete(simulation_data['res'], obj=0, axis=1)
    src = np.delete(simulation_data['src'], obj=0, axis=1)
    times = np.linspace(cfg.dt, cfg.t_end, cfg.N_t - 1, endpoint=True)
    assert times[1] == 2 * cfg.dt

    # Split data into training, validation and testing sets.

    train_ICs = np.zeros((cfg.N_train_examples, cfg.N_x + 2, cfg.N_y + 2))
    train_unc = np.zeros((cfg.N_train_examples, cfg.N_x + 2, cfg.N_y + 2))
    train_ref = np.zeros((cfg.N_train_examples, cfg.N_x + 2, cfg.N_y + 2))
    train_res = np.zeros((cfg.N_train_examples, cfg.N_x + 2, cfg.N_y + 2))
    train_src = np.zeros((cfg.N_train_examples, cfg.N_x    , cfg.N_y    ))
    train_times = np.zeros(cfg.N_train_examples)
    train_alphas = []
    for a in range(cfg.N_train_alphas):
        print("a", a)
        train_ICs[(cfg.N_t - 1) * (a + 0):(cfg.N_t - 1) * (a + 1), :, :] = ICs[a, :, :, :]
        train_unc[(cfg.N_t - 1) * (a + 0):(cfg.N_t - 1) * (a + 1), :, :] = unc[a, :, :, :]
        train_ref[(cfg.N_t - 1) * (a + 0):(cfg.N_t - 1) * (a + 1), :, :] = ref[a, :, :, :]
        train_res[(cfg.N_t - 1) * (a + 0):(cfg.N_t - 1) * (a + 1), :, :] = res[a, :, :, :]
        train_src[(cfg.N_t - 1) * (a + 0):(cfg.N_t - 1) * (a + 1), :, :] = src[a, :, :, :]
        train_times[(cfg.N_t - 1) * (a + 0):(cfg.N_t - 1) * (a + 1)] = times
        train_alphas.append(cfg.alphas[a])
    train_alphas = np.asarray(train_alphas)

    val_ICs = np.zeros((cfg.N_val_examples, cfg.N_x + 2, cfg.N_y + 2))
    val_unc = np.zeros((cfg.N_val_examples, cfg.N_x + 2, cfg.N_y + 2))
    val_ref = np.zeros((cfg.N_val_examples, cfg.N_x + 2, cfg.N_y + 2))
    val_res = np.zeros((cfg.N_val_examples, cfg.N_x + 2, cfg.N_y + 2))
    val_src = np.zeros((cfg.N_val_examples, cfg.N_x    , cfg.N_y    ))
    val_times = np.zeros(cfg.N_val_examples)
    val_alphas = []
    for a in range(cfg.N_val_alphas):
        print("a", a)
        offset = cfg.N_train_alphas
        val_ICs[(cfg.N_t - 1) * (a + 0):(cfg.N_t - 1) * (a + 1), :, :] = ICs[a + offset, :, :, :]
        val_unc[(cfg.N_t - 1) * (a + 0):(cfg.N_t - 1) * (a + 1), :, :] = unc[a + offset, :, :, :]
        val_ref[(cfg.N_t - 1) * (a + 0):(cfg.N_t - 1) * (a + 1), :, :] = ref[a + offset, :, :, :]
        val_res[(cfg.N_t - 1) * (a + 0):(cfg.N_t - 1) * (a + 1), :, :] = res[a + offset, :, :, :]
        val_src[(cfg.N_t - 1) * (a + 0):(cfg.N_t - 1) * (a + 1), :, :] = src[a + offset, :, :, :]
        val_times[(cfg.N_t - 1) * (a + 0):(cfg.N_t - 1) * (a + 1)] = times
        val_alphas.append(cfg.alphas[a + offset])
    val_alphas = np.asarray(val_alphas)

    test_ICs = np.zeros((cfg.N_test_examples, cfg.N_x + 2, cfg.N_y + 2))
    test_unc = np.zeros((cfg.N_test_examples, cfg.N_x + 2, cfg.N_y + 2))
    test_ref = np.zeros((cfg.N_test_examples, cfg.N_x + 2, cfg.N_y + 2))
    test_res = np.zeros((cfg.N_test_examples, cfg.N_x + 2, cfg.N_y + 2))
    test_src = np.zeros((cfg.N_test_examples, cfg.N_x    , cfg.N_y    ))
    test_times = np.zeros(cfg.N_test_examples)
    test_alphas = []
    for a in range(cfg.N_test_alphas):
        print("a", a)
        offset = cfg.N_train_alphas + cfg.N_val_alphas
        test_ICs[(cfg.N_t - 1) * (a + 0):(cfg.N_t - 1) * (a + 1), :, :] = ICs[a + offset, :, :, :]
        test_unc[(cfg.N_t - 1) * (a + 0):(cfg.N_t - 1) * (a + 1), :, :] = unc[a + offset, :, :, :]
        test_ref[(cfg.N_t - 1) * (a + 0):(cfg.N_t - 1) * (a + 1), :, :] = ref[a + offset, :, :, :]
        test_res[(cfg.N_t - 1) * (a + 0):(cfg.N_t - 1) * (a + 1), :, :] = res[a + offset, :, :, :]
        test_src[(cfg.N_t - 1) * (a + 0):(cfg.N_t - 1) * (a + 1), :, :] = src[a + offset, :, :, :]
        test_times[(cfg.N_t - 1) * (a + 0):(cfg.N_t - 1) * (a + 1)] = times
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

        # Time levels
        train_times = np.concatenate((train_times, train_times), axis=0)

    # Calculate statistical properties of training data.
    train_unc_mean = np.mean(train_unc)
    train_ref_mean = np.mean(train_ref)
    train_res_mean = np.mean(train_res)
    train_src_mean = np.mean(train_src)

    train_unc_std = np.std(train_unc)
    train_ref_std = np.std(train_ref)
    train_res_std = np.std(train_res)
    train_src_std = np.std(train_src)

    # z_normalize data.
    train_unc_normalized = util.z_normalize(train_unc, train_unc_mean, train_unc_std)
    val_unc_normalized   = util.z_normalize(val_unc,   train_unc_mean, train_unc_std)
    test_unc_normalized  = util.z_normalize(test_unc,  train_unc_mean, train_unc_std)

    train_ref_normalized = util.z_normalize(train_ref, train_ref_mean, train_ref_std)
    val_ref_normalized   = util.z_normalize(val_ref,   train_ref_mean, train_ref_std)
    test_ref_normalized  = util.z_normalize(test_ref,  train_ref_mean, train_ref_std)

    train_res_normalized = util.z_normalize(train_res, train_res_mean, train_res_std)
    val_res_normalized   = util.z_normalize(val_res,   train_res_mean, train_res_std)
    test_res_normalized  = util.z_normalize(test_res,  train_res_mean, train_res_std)

    train_src_normalized = util.z_normalize(train_src, train_src_mean, train_src_std)
    val_src_normalized   = util.z_normalize(val_src,   train_src_mean, train_src_std)
    test_src_normalized  = util.z_normalize(test_src,  train_src_mean, train_src_std)

    # Note that the ICs are not to be used in conjunction with the NN directly,
    # so there is no need to normalize them. Same goes for time levels.

    # Convert data from Numpy array to Torch tensor.
    train_ICs_tensor = torch.from_numpy(train_ICs)
    train_unc_tensor = torch.from_numpy(train_unc_normalized)
    train_ref_tensor = torch.from_numpy(train_ref_normalized)
    train_res_tensor = torch.from_numpy(train_res_normalized)
    train_src_tensor = torch.from_numpy(train_src_normalized)
    train_times_tensor = torch.from_numpy(train_times)

    val_ICs_tensor = torch.from_numpy(val_ICs)
    val_unc_tensor = torch.from_numpy(val_unc_normalized)
    val_ref_tensor = torch.from_numpy(val_ref_normalized)
    val_res_tensor = torch.from_numpy(val_res_normalized)
    val_src_tensor = torch.from_numpy(val_src_normalized)
    val_times_tensor = torch.from_numpy(val_times)

    test_ICs_tensor = torch.from_numpy(test_ICs)
    test_unc_tensor = torch.from_numpy(test_unc_normalized)
    test_ref_tensor = torch.from_numpy(test_ref_normalized)
    test_res_tensor = torch.from_numpy(test_res_normalized)
    test_src_tensor = torch.from_numpy(test_src_normalized)
    test_times_tensor = torch.from_numpy(test_times)

    # Create array to store stats used for normalization.
    stats = np.asarray([
        train_unc_mean,
        train_ref_mean,
        train_res_mean,
        train_src_mean,
        train_unc_std,
        train_ref_std,
        train_res_std,
        train_src_std
    ])

    # Pad with zeros to satisfy requirements of Torch's TensorDataset.
    # (Assumes that all datasets contain 6 or more data examples.)
    assert train_unc.shape[0] >= stats.shape[0] and val_unc.shape[0] >= stats.shape[0] and test_unc.shape[0] >= stats.shape[0]
    stats_train = np.zeros(train_unc.shape[0])
    stats_val = np.zeros(val_unc.shape[0])
    stats_test = np.zeros(test_unc.shape[0])
    stats_train[:stats.shape[0]] = stats
    stats_val[:stats.shape[0]] = stats
    stats_test[:stats.shape[0]] = stats

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
                                                   train_alphas_tensor, train_res_tensor)
    dataset_val = torch.utils.data.TensorDataset(  val_unc_tensor,      val_ref_tensor,   val_src_tensor,
                                                   stats_val_tensor,    val_ICs_tensor,   val_times_tensor,
                                                   val_alphas_tensor,   val_res_tensor)
    dataset_test = torch.utils.data.TensorDataset( test_unc_tensor,     test_ref_tensor,  test_src_tensor,
                                                   stats_test_tensor,   test_ICs_tensor,  test_times_tensor,
                                                   test_alphas_tensor,  test_res_tensor)

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