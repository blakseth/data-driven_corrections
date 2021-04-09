"""
train.py

Written by Sindre Stenen Blakseth, 2021.

Script for training models for data-driven corrections of the 1D heat equation.
"""

########################################################################################################################
# Package imports.

import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import time
import torch

########################################################################################################################
# File imports.

import datasets
import util

########################################################################################################################
# Training ML-model.

def train(cfg, model, num):
    print("CUDA availability:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("Current device:", torch.cuda.current_device(),
              "- num devices:", torch.cuda.device_count(),
              "- device name:", torch.cuda.get_device_name(0))

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

    it_per_epoch = len(dataloader_train)
    num_epochs = cfg.max_train_it // it_per_epoch + 1

    it = 0

    lowest_val_los = torch.tensor(float("Inf")).to(cfg.device)
    val_epoch_since_improvement = torch.tensor(0.0).to(cfg.device)

    stats = dataset_train[:8][3].detach().numpy()
    ref_mean = stats[1]
    ref_std = stats[5]

    start = time.time()

    for epoch in range(num_epochs):
        for i, data in enumerate(dataloader_train):
            if it >= cfg.max_train_it or (val_epoch_since_improvement >= cfg.overfit_limit and it >= cfg.min_train_it):
                break

            it += 1

            unc_data = data[0].to(cfg.device) # unc = uncorrected.
            #print("unc_shape train.py:", unc_data.shape)
            ref_data = data[1].to(cfg.device) # ref = reference.
            src_data = data[2].to(cfg.device) # src = source.
            res_data = data[7].to(cfg.device) # res = residual.
            old_data = util.z_normalize(data[4], ref_mean, ref_std).to(cfg.device) # old = reference at previous time level.

            model.net.train()

            if cfg.model_type == 'data':
                #print("Data pass")
                out_data = model.net(old_data)
            else:
                out_data = model.net(unc_data) # out = output (corrected profile or predicted correction source term).

            if cfg.model_type == 'hybrid':
                loss = model.loss(out_data, src_data)
            elif cfg.model_type == 'residual':
                loss = model.loss(out_data, res_data[:, 1:-1, 1:-1])
            elif cfg.model_type == 'end-to-end':
                loss = model.loss(out_data, ref_data[:, 1:-1, 1:-1])
            elif cfg.model_type == 'data':
                #print("Data loss")
                loss = model.loss(out_data, ref_data[:, 1:-1, 1:-1])

            """
            if it == cfg.num_train_it:
                print("unc_data:", unc_data)
                print("ref_data:", ref_data)
                print("src_data:", src_data)
                print("out_data:", out_data)
                print("loss:", loss)
            """

            if it % cfg.print_train_loss_period == 0:
                print(it, loss.item())
            if it % cfg.save_train_loss_period == 0:
                model.train_losses.append(loss.item())
                model.train_iterations.append(it)

            model.net.zero_grad()

            loss.backward()

            model.optimizer.step()

            if it % cfg.validation_period == 0:
                print("val it:", it)
                model.net.eval()
                with torch.no_grad():
                    for j, val_data in enumerate(dataloader_val):
                        print("j:", j)
                        unc_data_val = val_data[0].to(cfg.device)
                        ref_data_val = val_data[1].to(cfg.device)
                        res_data_val = val_data[7].to(cfg.device)
                        src_data_val = val_data[2].to(cfg.device)
                        if cfg.model_type == 'data':
                            #print("Data val pass")
                            old_data_val = util.z_normalize(val_data[4], ref_mean, ref_std).to(cfg.device)
                            out_data_val = model.net(old_data_val)
                        else:
                            out_data_val = model.net(unc_data_val)

                        if cfg.model_type == 'hybrid':
                            val_loss = model.loss(out_data_val, src_data_val)
                        elif cfg.model_type == 'residual':
                            val_loss = model.loss(out_data_val, res_data_val[:, 1:-1, 1:-1])
                        elif cfg.model_type == 'end-to-end':
                            val_loss = model.loss(out_data_val, ref_data_val[:, 1:-1, 1:-1])
                        elif cfg.model_type == 'data':
                            #print("Data val loss")
                            val_loss = model.loss(out_data_val, ref_data_val[:, 1:-1, 1:-1])
                        model.val_losses.append(val_loss.item())
                        model.val_iterations.append(it)
                        if val_loss < lowest_val_los:
                            lowest_val_los = val_loss
                            val_epoch_since_improvement = 0
                        else:
                            val_epoch_since_improvement += 1
    if torch.cuda.is_available():
        torch.cuda.synchronize(cfg.device)
    end = time.time()
    print ("Time elapsed:", end - start)

    if cfg.model_name[:8] == "Ensemble":
        train_losses = np.zeros((len(model.nets), len(model.nets[0].train_losses)))
        for m in range(len(model.nets)):
            train_losses[m] = model.nets[m].train_losses
        train_losses = np.sum(train_losses, axis=0)
        #print("train_losses.shape:", train_losses.shape)
        val_losses = np.zeros((len(model.nets), len(model.nets[0].val_losses)))
        for m in range(len(model.nets)):
            #print("model.nets[m].val_losses:", model.nets[m].val_losses)
            val_losses[m] = model.nets[m].val_losses
        val_losses = np.sum(val_losses, axis=0)
        #print("val_losses.shape:", val_losses.shape)
        train_iterations = model.nets[0].train_iterations
        val_iterations = model.nets[0].val_iterations
        #print("len(val_iterations):", len(val_iterations))
    else:
        train_losses = model.train_losses
        val_losses = model.val_losses
        train_iterations = model.train_iterations
        val_iterations = model.val_iterations

    fig1, ax1 = plt.subplots()
    ax1.set_title("Training and Validation Loss, Model " + str(num))
    ax1.set_xlabel("Iterations")
    ax1.set_ylabel("MSELoss")
    ax1.plot(train_iterations, train_losses, label='Training loss')
    ax1.plot(val_iterations, val_losses, label='Validation loss')
    ax1.set_yscale('log')
    ax1.set_yticks([1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2])
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(cfg.run_dir, "training_and_val_loss" + str(num) + ".pdf"))
    plt.close()
    #print("Ticks:", np.arange(np.log10(min_loss), np.log(max_loss) + 1, 1.0))

    data_dict = dict()
    data_dict['Training loss'] = np.asarray([train_iterations, train_losses])
    data_dict['Validation loss'] = np.asarray([val_iterations, val_losses])
    pickle.dump(data_dict, open(os.path.join(cfg.run_dir, "plot_data_loss_training_and_val" + str(num) + ".pkl"), "wb"))

    return data_dict

########################################################################################################################