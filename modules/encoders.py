import torch
import torch.nn as tnn
from modules.LDM import LDM
from nn import init_ss_parameters

class Encoder(tnn.Module):
    '''
    Encoder module used for single-scale networks, or modality-specific and fusion network architectures of MultiscaleEncoder
    '''
    
    def __init__(self, **kwargs):
        '''
        initializes the input and hidden layers of the encoder
        '''
        
        super(Encoder, self).__init__()
        
        self.input_dim = kwargs.pop('input_dim', None)
        self.output_dim = kwargs.pop('output_dim', None)
        self.layer_list = kwargs.pop('layer_list', None)
        self.kernel_initializer_fn = kwargs.pop('kernel_initializer_fn', tnn.init.xavier_normal_)
        self.activation_fn = kwargs.pop('activation_fn', tnn.Tanh())
        self.layers = tnn.ModuleList()
        
        current_dim = self.input_dim
        for dim in self.layer_list:
            self.layers.append(tnn.Linear(current_dim, dim))
            self.kernel_initializer_fn(self.layers[-1].weight)
            current_dim = dim

        self.final_layer = torch.nn.Linear(current_dim, self.output_dim)
        self.kernel_initializer_fn(self.final_layer.weight)
        
    def forward(self, x):
        '''
        Performs forward pass through the MLP encoder network

        x: torch.Tensor, data to forward pass through the network, (num_seq*num_steps, n_s or n_y or 2*n_a)
        '''

        for layer in self.layers:
            x = layer(x)
            x = self.activation_fn(x)
        x = self.final_layer(x)
        return x
    

class MultiscaleEncoder(tnn.Module):
    '''
    Multiscale encoder module for MRINE
    '''

    def __init__(self, **kwargs):
        '''
        Initializes modality-specific networks and LDMs, and fusion network that overall define the MultiscaleEncoder
        '''

        super(MultiscaleEncoder, self).__init__()
        
        self.n_s = kwargs.pop('n_s', None)
        self.n_y = kwargs.pop('n_y', None)
        self.n_a = kwargs.pop('n_a', None)
        self.n_x = kwargs.pop('n_x', None)
        self.layer_list_s = kwargs.pop('layer_list_s', None)
        self.layer_list_y = kwargs.pop('layer_list_y', None)
        self.layer_list_m = kwargs.pop('layer_list_m', None)
        self.kernel_initializer_fn = kwargs.pop('kernel_initializer_fn', tnn.init.xavier_normal_)
        self.activation_fn = kwargs.pop('activation_fn', tnn.Tanh())
        self.init_A_scale = kwargs.pop('init_A_scale', None)
        self.init_C_scale = kwargs.pop('init_C_scale', None)
        self.init_Q_scale = kwargs.pop('init_Q_scale', None)
        self.init_R_scale = kwargs.pop('init_R_scale', None)
        self.init_cov_scale = kwargs.pop('init_cov_scale', None)

        # Modality-specific network for modality s (spiking activity/discrete observations)
        self.encoder_s = Encoder(input_dim=self.n_s,
                                 output_dim=self.n_a,
                                 layer_list=self.layer_list_s,
                                 kernel_initializer_fn=self.kernel_initializer_fn,
                                 activation_fn=self.activation_fn)
        # Modality-specific network for modality y (LFP/continuous observations)
        self.encoder_y = Encoder(input_dim=self.n_y,
                                 output_dim=self.n_a,
                                 layer_list=self.layer_list_y,
                                 kernel_initializer_fn=self.kernel_initializer_fn,
                                 activation_fn=self.activation_fn)
        # Fusion network
        self.encoder_m = Encoder(input_dim=2*self.n_a,
                                 output_dim=self.n_a,
                                 layer_list=self.layer_list_m,
                                 kernel_initializer_fn=self.kernel_initializer_fn,
                                 activation_fn=self.activation_fn)
        # LDM parameters of modality-specific LDM for modality s
        A_s, C_s, Q_log_diag_s, R_log_diag_s, Sigma_0_s, x_0_s = init_ss_parameters(n_x=self.n_x, n_a=self.n_a, 
                                                                                    init_A_scale=self.init_A_scale, init_C_scale=self.init_C_scale, 
                                                                                    init_Q_scale=self.init_Q_scale, init_R_scale=self.init_R_scale, 
                                                                                    init_cov_scale=self.init_cov_scale)
        # LDM parameters of modality-specific LDM for modality y
        A_y, C_y, Q_log_diag_y, R_log_diag_y, Sigma_0_y, x_0_y = init_ss_parameters(n_x=self.n_x, n_a=self.n_a, 
                                                                                    init_A_scale=self.init_A_scale, init_C_scale=self.init_C_scale, 
                                                                                    init_Q_scale=self.init_Q_scale, init_R_scale=self.init_R_scale, 
                                                                                    init_cov_scale=self.init_cov_scale)
        # Modality-specific LDM for modality s
        self.ldm_s = LDM(n_x=self.n_x, n_a=self.n_a, 
                         A=A_s, C=C_s, Q_log_diag=Q_log_diag_s, R_log_diag=R_log_diag_s,
                         Sigma_0=Sigma_0_s, x_0=x_0_s)
        # Modality-specific LDM for modality y
        self.ldm_y = LDM(n_x=self.n_x, n_a=self.n_a, 
                         A=A_y, C=C_y, Q_log_diag=Q_log_diag_y, R_log_diag=R_log_diag_y,
                         Sigma_0=Sigma_0_y, x_0=x_0_y)
    
    def forward(self, s, y, m_s, m_y):
        '''
        Performs forward pass through the MultiscaleEncoder network

        s: torch.Tensor, spiking activity, or discrete observations. (num_seq*num_steps, n_s)
        y: torch.Tensor, LFP, or continuous signals. (num_seq*num_steps, n_y)
        m_s: torch.Tensor, mask tensor for s denoting whether s is available for a time-step or not. (num_seq*num_steps)
        m_y: torch.Tensor, mask tensor for y denoting whether y is available for a time-step or not. (num_seq*num_steps)
        '''

        num_steps = m_s.shape[1]
        # Obtain initial modality-specific representation for modality s
        a_s = self.encoder_s(s).reshape(-1, num_steps, self.n_a)
        # Obtain initial modality-specific representation for modality y
        a_y = self.encoder_y(y).reshape(-1, num_steps, self.n_a)

        # Perform filtering with modality-specific LDM of modality s to learn within-modality dynamics and account for missing samples by intrinsic dynamics
        _, x_filter_s, _, _, _, _ = self.ldm_s(a=a_s, mask=m_s, do_smoothing=False)
        # Perform filtering with modality-specific LDM of modality y to learn within-modality dynamics and account for missing samples by intrinsic dynamics
        _, x_filter_y, _, _, _, _ = self.ldm_y(a=a_y, mask=m_y, do_smoothing=False)

        # Obtain modality-specific embedding factors
        a_filter_s = (self.ldm_s.C[None, None, :, :] @ x_filter_s[:, :, :, None]).squeeze(dim=-1)
        a_filter_y = (self.ldm_y.C[None, None, :, :] @ x_filter_y[:, :, :, None]).squeeze(dim=-1)

        # Concatenation and fusion 
        a_m = torch.cat([a_filter_s, a_filter_y], dim=-1).reshape(-1, 2*self.n_a)
        a_m = self.encoder_m(a_m)
        return a_m
