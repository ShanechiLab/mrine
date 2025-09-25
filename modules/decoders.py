import torch
import torch.nn  as nn


class DecoderPoisson(nn.Module):
    '''
    Decoder module for Poisson distributed modalities
    '''

    def __init__(self, **kwargs):
        '''
        Initializes the DecoderPoisson object
        '''
        
        super(DecoderPoisson, self).__init__()

        self.n_a = kwargs.pop('n_a', None)
        self.n_s = kwargs.pop('n_s', None)
        self.layer_list_s = kwargs.pop('layer_list_s', None)
        self.kernel_initializer_fn = kwargs.pop('kernel_initializer_fn', torch.nn.init.xavier_normal_)
        self.activation_fn = kwargs.pop('activation_fn', torch.nn.Tanh())

        self.layers = nn.ModuleList()
        current_dim = self.n_a
        for dim in self.layer_list_s:
            self.layers.append(nn.Linear(current_dim, dim))
            self.kernel_initializer_fn(self.layers[-1].weight)
            current_dim = dim
        # The last layer reconstructs the log firing rates
        self.logfr_layer = nn.Linear(current_dim, self.n_s)
        self.kernel_initializer_fn(self.logfr_layer.weight)

    def forward(self, x):
        '''
        Performs  forward pass through the decoder network

        x: Filtered/smoothed/predicted multiscale or single-scale embedding factors to reconstruct Poisson distribution's mean, i.e., firing rates, (num_seq*num_steps, n_a)
        '''

        for layer in self.layers:
            x = layer(x)
            x = self.activation_fn(x)
        logfr = self.logfr_layer(x)
        logfr = torch.clamp(logfr, min=-5, max=5) # clip the log firing rates if large
        fr =  torch.exp(logfr)
        return fr


class DecoderGaussian(nn.Module):
    '''
    Decoder module for Gaussian distributed modalities
    '''

    def __init__(self, **kwargs):
        '''
        Initializes the DecoderGaussian object
        '''

        super(DecoderGaussian, self).__init__()

        self.n_a = kwargs.pop('n_a', None)
        self.n_y = kwargs.pop('n_y', None)
        self.layer_list_y = kwargs.pop('layer_list_y', None)
        self.kernel_initializer_fn = kwargs.pop('kernel_initializer_fn', nn.init.xavier_normal_)
        self.activation_fn = kwargs.pop('activation_fn', nn.Tanh())

        self.layers = nn.ModuleList()
        current_dim = self.n_a
        for dim in self.layer_list_y:
            self.layers.append(nn.Linear(current_dim, dim))
            self.kernel_initializer_fn(self.layers[-1].weight)
            current_dim = dim
        
        # Mean layer
        self.mean_layer = nn.Linear(current_dim, self.n_y)
        self.kernel_initializer_fn(self.mean_layer.weight)
        
        # If variance is learned, we learn global variance instead of learning the variance at each time-step. By default, we use constant variance instead of learning.
        self.logvar = nn.Parameter(torch.randn(self.n_y), requires_grad=True)

    def forward(self, x):
        '''
        Performs  forward pass through the decoder network

        x: Filtered/smoothed/predicted multiscale or single-scale embedding factors to reconstruct Gaussian distribution's mean and variance, (num_seq*num_steps, n_a)
        '''

        for layer in self.layers:
            x = layer(x)
            x = self.activation_fn(x)
        mean = self.mean_layer(x)
        var = torch.exp(self.logvar[None, :]) * torch.ones_like(mean) # unused when learn_x_var in the config is set to False (default setting is False).
        return mean, var

