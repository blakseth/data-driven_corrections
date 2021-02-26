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
import torch

########################################################################################################################
# File imports.
import config
from datasets import load_datasets

########################################################################################################################
# Training ML-model.

def train(model, num, dataloader_train, dataloader_val):
    it_per_epoch = len(dataloader_train)
    num_epochs = config.max_train_it // it_per_epoch + 1

    it = 0

    lowest_val_los = np.inf
    val_epoch_since_improvement = 0

    for epoch in range(num_epochs):
        for i, data in enumerate(dataloader_train):
            if it >= config.max_train_it or (val_epoch_since_improvement >= config.overfit_limit and it >= config.min_train_it):
                break

            model.net.train()
            it += 1

            unc_data = data[0] # unc = uncorrected.
            ref_data = data[1] # ref = reference.
            src_data = data[2] # src = source.

            out_data = model.net(unc_data) # out = output (corrected profile or predicted correction source term).

            if config.model_is_hybrid:
                loss = model.loss(out_data, src_data)
            else:
                loss = model.loss(out_data, ref_data[:, 1:-1])

            """
            if it == config.num_train_it:
                print("unc_data:", unc_data)
                print("ref_data:", ref_data)
                print("src_data:", src_data)
                print("out_data:", out_data)
                print("loss:", loss)
            """

            if it % config.print_train_loss_period == 0:
                print(it, loss.item())
                model.train_losses.append(loss.item())
                model.train_iterations.append(it)

            model.net.zero_grad()

            loss.backward()

            model.optimizer.step()

            if it % config.validation_period == 0:
                model.net.eval()
                with torch.no_grad():
                    for j, val_data in enumerate(dataloader_val):
                        unc_data_val = val_data[0]
                        ref_data_val = val_data[1]
                        src_data_val = val_data[2]
                        out_data_val = model.net(unc_data_val)
                        if config.model_is_hybrid:
                            val_loss = model.loss(out_data_val, src_data_val)
                        else:
                            val_loss = model.loss(out_data_val, ref_data_val[:, 1:-1])
                        model.val_losses.append(val_loss.item())
                        model.val_iterations.append(it)
                        if val_loss < lowest_val_los:
                            lowest_val_los = val_loss
                            val_epoch_since_improvement = 0
                        else:
                            val_epoch_since_improvement += 1

    fig1, ax1 = plt.subplots()
    ax1.set_title("Training and Validation Loss, Model " + str(num))
    ax1.set_xlabel("Iterations")
    ax1.set_ylabel("MSELoss")
    max_loss = max(np.amax(model.train_losses), np.amax(model.val_losses))
    min_loss = min(np.amin(model.train_losses), np.amin(model.val_losses))
    ax1.plot(model.train_iterations, model.train_losses, label='Training loss')
    ax1.plot(model.val_iterations, model.val_losses, label='Validation loss')
    ax1.set_yscale('log')
    ax1.set_yticks([1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2])
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(config.run_dir, "training_and_val_loss" + str(num) + ".pdf"))
    #print("Ticks:", np.arange(np.log10(min_loss), np.log(max_loss) + 1, 1.0))

    data_dict = dict()
    data_dict['Training loss'] = np.asarray([model.train_iterations, model.train_losses])
    data_dict['Validation loss'] = np.asarray([model.val_iterations, model.val_losses])
    pickle.dump(data_dict, open(os.path.join(config.run_dir, "plot_data_loss_training_and_val" + str(num) + ".pkl"), "wb"))

    return data_dict

########################################################################################################################