"""
models.py

Written by Sindre Stenen Blakseth, 2021.

Machine learning models for data-driven corrections of 1D heat conduction problems.
"""

########################################################################################################################
# Package imports.

import torch

########################################################################################################################
# File imports.

import config

########################################################################################################################
# Modules.

# Dense layer with activation function and (possibly) dropout.
class DenseLayerWithAct(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super(DenseLayerWithAct, self).__init__()
        self.layer = torch.nn.Linear(input_size, output_size)
        self.activation = None
        self.dropout = torch.nn.Dropout(0.0)
        if config.act_type == 'lrelu':
            self.activation = torch.nn.LeakyReLU(config.act_param)
        else:
            raise Exception("Invalid loss function selection.")
        if config.use_dropout:
            self.dropout = torch.nn.Dropout(config.dropout_prop)

    def forward(self, x):
        return self.dropout(self.activation(self.layer(x)))

# Network of dense layers.
class DenseModule(torch.nn.Module):
    def __init__(self, num_layers, input_size, output_size, hidden_size = 0):
        assert num_layers >= 2
        assert num_layers == 2 or hidden_size > 0
        assert input_size > 0 and output_size > 0
        super(DenseModule, self).__init__()

        # Defining input layer.
        first_layer = torch.nn.Linear(input_size, hidden_size)
        first_activation = None
        if config.act_type == 'lrelu':
            first_activation = torch.nn.LeakyReLU(config.act_param)

        # Defining hidden layers.
        hidden_block = [
            DenseLayerWithAct(hidden_size, hidden_size) for hidden_layer in range(num_layers - 2)
        ]

        # Defining output layer.
        last_layer = torch.nn.Linear(hidden_size, output_size)

        # Defining full architecture.
        self.net = torch.nn.Sequential(
            first_layer,
            first_activation,
            *hidden_block,
            last_layer
        ).double()

    def forward(self, x):
        return self.net(x)

# Network of 1D convolution layers, possibly with dense layers at the end.
class ConvolutionModule(torch.nn.Module):
    def __init__(self, num_layers, kernel_size, padding, stride, num_features, num_dense, dense_size):
        super(ConvolutionModule, self).__init__()
        # TODO: Define this module.

# Ensemble of dense networks.
class EnsembleDenseModule(torch.nn.Module):
    def __init__(self, num_layers, input_size, output_size):
        super(EnsembleDenseModule, self).__init__()

        # Define one locally-correcting network per output node.
        self.local_networks = [DenseModule(num_layers, 3, 3, 1) for i in range(output_size)]

    def forward(self, x):
        # TODO: Is it possible to make this more efficient?
        # TODO: This probably does not work for batch_size > 1.
        output = torch.zeros_like(x)
        for i in range(1, len(self.local_networks)):
            output[i] = self.local_networks[i](x[i-1:i+2]) # Pass elements i-1, i and i+1 of x.
        return output

########################################################################################################################
# Full model, consisting of network, loss function, optimizer and information storage facilitation.

class Model:
    def __init__(self, module_name, num_layers, input_size, output_size, hidden_size = 0):
        assert num_layers >= 2
        assert num_layers == 2 or hidden_size > 0
        assert input_size > 0 and output_size > 0

        # Defining network architecture.
        if module_name == 'DenseModule':
            self.net = DenseModule(num_layers, input_size, output_size, hidden_size)
        else:
            raise Exception("Invalid module selection.")

        # Defining loss function.
        if config.loss_func == 'MSE':
            self.loss = torch.nn.MSELoss(reduction='mean')
        else:
            raise Exception("Invalid loss function selection.")

        # Defining learning parameters.
        self.learning_rate = config.learning_rate
        if config.optimizer == 'adam':
            self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.learning_rate)
        else:
            raise Exception("Invalid optimizer selection.")

        # Lists for storing training losses and corresponding training iteration numbers.
        self.train_losses     = []
        self.train_iterations = []
        self.val_losses       = []
        self.val_iterations   = []

########################################################################################################################
# Creating a new model.

def create_new_model():
    if config.model_name == 'GlobalDense':
        return Model('DenseModule', config.num_layers, config.N_coarse + 2, config.N_coarse, config.hidden_layer_size)
    elif config.model_name == 'CNNModule':
        return Model('CNNModule', config.num_layers, config.N_coarse + 2, config.N_coarse, config.N_coarse + 2)
    elif config.model_name == 'LocalDense':
        return Model('EnsembleDenseModule', config.num_layers, config.N_coarse + 2, config.N_coarse, config.N_coarse + 2)
    else:
        raise Exception("Invalid model selection.")

########################################################################################################################