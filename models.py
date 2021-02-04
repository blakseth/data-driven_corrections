"""
models.py

Written by Sindre Stenen Blakseth, 2021

Machine learning models for data-driven corrections of 1D heat conduction problems.
"""

########################################################################################################################
# Package imports.

import torch

########################################################################################################################
# File imports.

import config

########################################################################################################################
# MyModel.

class Hidden_layer_with_act(torch.nn.Module):
    def __init__(self):
        super(Hidden_layer_with_act, self).__init__()
        self.layer = torch.nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = None
        self.dropout = torch.nn.Dropout(0.0)
        if config.act_type == 'lrelu':
            self.activation = torch.nn.LeakyReLU(0.2)
        if config.use_dropout:
            self.dropout = torch.nn.Dropout(0.1)

    def forward(self, x):
        return self.dropout(self.activation(self.layer(x)))

class MyModel:
    def __init__(self):
        # Defining network architecture.
        first_layer = torch.nn.Linear(config.input_size, config.hidden_size)
        first_activation = None
        if config.act_type == 'lrelu':
            first_activation = torch.nn.LeakyReLU(0.2)

        hidden_block = [
            Hidden_layer_with_act() for hidden_layer in range(config.num_layers - 2)
        ]

        last_layer = torch.nn.Linear(config.hidden_size, config.output_size)

        self.net = torch.nn.Sequential(
            first_layer,
            first_activation,
            *hidden_block,
            last_layer
        ).double()

        # Defining loss function.
        """
        self.basic_loss = torch.nn.MSELoss(reduction='sum')

        if config.normalize_loss_by_target:

            def MSELoss_normalized_by_target(fake, real):
                print("Real: ", real)
                normalized_fake = torch.div(fake, real)
                print("Norm fake: ", real)
                normalized_loss = self.basic_loss(normalized_fake, torch.ones_like(real))
                print(normalized_loss)
                return normalized_loss
            self.loss_fn = MSELoss_normalized_by_target
        else:
        """
        self.loss_fn = torch.nn.MSELoss(reduction='sum')

            # Defining learning parameters.
        self.learning_rate = config.learning_rate
        if config.optimizer_name == 'adam':
            self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.learning_rate)

        # Lists for storing training losses and corresponding training iteration numbers.
        self.train_losses     = []
        self.train_iterations = []
        self.val_losses       = []
        self.val_iterations   = []

def source_loss():
    # Compute residual of fine-scale
    fine_residual = 0
    coarse_residual = 0


    return torch.abs(fine_residual - coarse_residual)

########################################################################################################################
# Creating a new model.

def create_new_model():
    if config.model_name == "MyModel":
        return MyModel()

########################################################################################################################