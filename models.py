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
    def __init__(self, input_size, output_size, dropout_probs):
        super(DenseLayerWithAct, self).__init__()
        self.layer = torch.nn.Linear(input_size, output_size)
        self.activation = None
        self.dropout = torch.nn.Dropout(dropout_probs)
        if config.act_type == 'lrelu':
            self.activation = torch.nn.LeakyReLU(config.act_param)
        else:
            raise Exception("Invalid loss function selection.")

    def forward(self, x):
        return self.dropout(self.activation(self.layer(x)))

# Convolution layer with activation function and (possibly) dropout.
class ConvLayerWithAct(torch.nn.Module):
    def __init__(self, num_in_ch, num_out_ch, kernel_size):
        super(ConvLayerWithAct, self).__init__()
        padding = kernel_size / 2 - 1
        self.layer = torch.nn.Conv2d(
            in_channels  = num_in_ch,
            out_channels = num_out_ch,
            kernel_size  = kernel_size,
            padding      = padding,
            stride       = 1
        )
        self.activation = None
        if config.act_type == 'lrelu':
            self.activation = torch.nn.LeakyReLU(config.act_param)
        else:
            raise Exception("Invalid loss function selection.")

    def forward(self, x):
        return self.activation(self.layer(x))

# Network of dense layers.
class DenseModule(torch.nn.Module):
    def __init__(self, num_layers, input_size, output_size, hidden_size = 0, dropout_prob = 0):
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
            DenseLayerWithAct(hidden_size, hidden_size, dropout_prob) for hidden_layer in range(num_layers - 2)
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
    def __init__(self, num_conv_layers, output_size, kernel_size, num_filters, num_fc_layers):
        super(ConvolutionModule, self).__init__()
        assert num_conv_layers >  0
        assert num_fc_layers   >  0
        assert output_size     >  0
        assert kernel_size     >= 3
        assert num_filters     >  0

        # Defining input layer.
        padding = kernel_size / 2 - 1
        first_layer = torch.nn.Conv2d(1, num_filters, kernel_size, padding, 1)
        first_activation = None
        if config.act_type == 'lrelu':
            first_activation = torch.nn.LeakyReLU(config.act_param)

        # Defining hidden conv layers.
        hidden_conv_block = [
            ConvLayerWithAct(num_filters, num_filters, kernel_size) for hidden_layer in range(num_conv_layers - 1)
        ]

        # Defining first dense layer.
        first_fc = DenseLayerWithAct((output_size + 2) * num_filters, output_size, 0.0)

        # Defining other dense layers.
        dense_block = [
            DenseLayerWithAct(output_size, output_size, 0.0) for fc_layer in range(num_fc_layers - 1)
        ]

        # Defining full architecture.
        self.net = torch.nn.Sequential(
            first_layer,
            first_activation,
            *hidden_conv_block,
            first_fc,
            *dense_block
        ).double()

    def forward(self, x):
        return self.net(x)


# Ensemble of dense networks.
class EnsembleDenseModule(torch.nn.Module):
    def __init__(self, num_layers, output_size, network_width, dropout_prob):
        super(EnsembleDenseModule, self).__init__()

        # Define one locally-correcting network per output node.
        self.nets = [DenseModule(num_layers, 3, 1, network_width, dropout_prob) for i in range(output_size)]

        print("\n\nNUMBER OF NETS:", len(self.nets))

    def forward(self, x):
        # TODO: Is it possible to make this more efficient?
        output = torch.zeros((x.shape[0], x.shape[1] - 2), requires_grad=True).double()
        for i in range(0, len(self.nets)):
            stencil = x[:,i:i+3].clone()
            output[:,i] = torch.squeeze(self.nets[i](stencil), 1)
        return output

########################################################################################################################
# Full model, consisting of network, loss function, optimizer and information storage facilitation.

class Model:
    def __init__(self, module_name, learning_rate, dropout_prob, model_specific_params):

        # Defining network architecture.
        input_size = config.N_coarse + 2
        output_size = config.N_coarse
        if module_name == 'DenseModule':
            num_layers = model_specific_params[0]
            hidden_size = model_specific_params[1]
            assert num_layers >= 2
            assert num_layers == 2 or hidden_size > 0
            self.net = DenseModule(num_layers, input_size, output_size, hidden_size, dropout_prob)
        elif module_name == 'EnsembleDenseModule':
            num_layers = model_specific_params[0]
            network_width = model_specific_params[1]
            self.net = EnsembleDenseModule(num_layers, output_size, network_width, dropout_prob)
        else:
            raise Exception("Invalid model selection.")

        # Defining loss function.
        if config.loss_func == 'MSE':
            self.loss = torch.nn.MSELoss(reduction='mean')
        else:
            raise Exception("Invalid loss function selection.")

        # Defining learning parameters.
        if module_name == 'EDenseModule':
            params = self.net.parameters()
        elif module_name == 'EnsembleDenseModule':
            params = []
            for i in range(len(self.net.nets)):
                params += list(self.net.nets[i].parameters())
            for param in params:
                param.requires_grad = True
        else:
            raise Exception("Invalid model selection.")
        self.learning_rate = config.learning_rate
        if config.optimizer == 'adam':
            self.optimizer = torch.optim.Adam(params, lr=learning_rate)
        else:
            raise Exception("Invalid optimizer selection.")

        # Lists for storing training losses and corresponding training iteration numbers.
        self.train_losses     = []
        self.train_iterations = []
        self.val_losses       = []
        self.val_iterations   = []

########################################################################################################################
# Creating a new model.

def create_new_model(learning_rate, dropout_prob, model_specific_params):
    if config.model_name == 'GlobalDense':
        return Model('DenseModule', learning_rate, dropout_prob, model_specific_params)
    elif config.model_name == 'CNNModule':
        return Model('CNNModule', config.num_layers, config.N_coarse + 2, config.N_coarse, config.N_coarse + 2)
    elif config.model_name == 'LocalDense':
        return Model('EnsembleDenseModule', learning_rate, dropout_prob, model_specific_params)
    else:
        raise Exception("Invalid model selection.")

########################################################################################################################