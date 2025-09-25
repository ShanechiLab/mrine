from modules.decoders import *
from modules.encoders import *
from modules.LDM import *

from nn import get_kernel_initializer_function, log_likelihood_gaussian, log_likelihood_poisson, get_activation_function, compute_sm_reg_loss, time_dropout
from python_utils import extract_dict_from_key

import torch
import torch.nn as nn

class MRINE(nn.Module):
    '''
    Multiscale Real-Time Inference of Nonlinear Embedddings (MRINE) 
    '''
    def __init__(self, config):
        '''
        Initializes MRINE object

        config: CfgNode, config denoting the hyperparameters of MRINE
        '''
        super(MRINE, self).__init__()

        # Get the config and dimension parameters
        self.config = config

        # Set the seed 
        torch.manual_seed(self.config.seed)

        self.activation_fn = get_activation_function(self.config.model.activation)
        self.kernel_initializer_fn = get_kernel_initializer_function(self.config.model.kernel_initializer)

        # For LDMs, initialize SS parameters
        A, C, Q_log_diag, R_log_diag, Sigma_0, x_0 = init_ss_parameters(n_x=self.config.model.n_x, n_a=self.config.model.n_a,
                                                                        init_A_scale=self.config.model.init_A_scale, init_C_scale=self.config.model.init_C_scale,
                                                                        init_Q_scale=self.config.model.init_Q_scale, init_R_scale=self.config.model.init_R_scale, 
                                                                        init_cov_scale=self.config.model.init_cov_scale)

        # Initialize the LDM
        self.ldm = LDM(n_x=self.config.model.n_x, n_a=self.config.model.n_a,
                       apply_normalize_A=self.config.model.self.apply_normalize_A,
                       A=A, C=C, Q_log_diag=Q_log_diag, R_log_diag=R_log_diag,
                       Sigma_0=Sigma_0, x_0=x_0)
        
        # Initialize the single-scale or multiscale encoder and decoder(s)
        self.encoder = self._get_encoder()
        self.decoder_dict = self._get_decoder_dict()

        # We apply regular dropout for inputs and outputs of the encoder 
        self.inp_dropout = torch.nn.Dropout(self.config.model.dropout_rate)
        self.a_hat_dropout = torch.nn.Dropout(self.config.model.dropout_rate)


    def _get_encoder(self):
        '''
        Returns the encoder module
        '''
        if self.config.model.model_type.lower() == 'multi':
            encoder = MultiscaleEncoder(n_s=self.config.model.n_s,
                                        n_y=self.config.model.n_y,
                                        n_a=self.config.model.n_a,
                                        n_x=self.config.model.n_x,
                                        layer_list_s=self.config.model.layer_list_s,
                                        layer_list_y=self.config.model.layer_list_y,
                                        layer_list_m=self.config.model.layer_list_m,
                                        kernel_initializer_fn=self.kernel_initializer_fn,
                                        activation_fn=self.activation_fn,
                                        init_A_scale=self.config.model.init_A_scale,
                                        init_C_scale=self.config.model.init_C_scale,
                                        init_Q_scale=self.config.model.init_Q_scale,
                                        init_R_scale=self.config.model.init_R_scale,
                                        init_cov_scale=self.config.model.init_cov_scale)
        # single-scale encoder designs are the same for Poisson and Gaussian modalities
        elif self.config.model.model_type.lower() == 'single-poisson':
            encoder = Encoder(input_dim=self.config.model.n_s,
                              output_dim=self.config.model.n_a,
                              layer_list=self.config.model.layer_list_s,
                              kernel_initializer_fn=self.kernel_initializer_fn,
                              activation_fn=self.activation_fn)
        elif self.config.model.model_type.lower() == 'single-gaussian':
            encoder = Encoder(input_dim=self.config.model.n_y,
                              output_dim=self.config.model.n_a,
                              layer_list=self.config.model.layer_list_y,
                              kernel_initializer_fn=self.kernel_initializer_fn,
                              activation_fn=self.activation_fn)
        return encoder


    def _get_decoder_dict(self):
        '''
        Creates the dictionary of modality-specific decoders for modalities to be used for learning
        '''

        decoder_dict = nn.ModuleDict()
        
        if self.config.model.model_type.lower() == 'single-poisson':
            decoder_dict['s_poisson'] = DecoderPoisson(n_a=self.config.model.n_a, 
                                               n_s=self.config.model.n_s,
                                               layer_list_s=self.config.model.layer_list_s[::-1],
                                               kernel_initializer_fn=self.kernel_initializer_fn,
                                               activation_fn=self.activation_fn)
        elif self.config.model.model_type.lower() == 'single-gaussian':
            decoder_dict['y_gaussian'] = DecoderGaussian(n_a=self.config.model.n_a, 
                                                n_y=self.config.model.n_y,
                                                layer_list_y=self.config.model.layer_list_y[::-1],
                                                kernel_initializer_fn=self.kernel_initializer_fn,
                                                activation_fn=self.activation_fn)
        elif self.config.model.model_type.lower() == 'multi':
            for d, likelihood, n, layer_list in zip(['s', 'y'],
                                                    [self.config.model.likelihood_s, self.config.model.likelihood_y], 
                                                    [self.config.model.n_s, self.config.model.n_y], 
                                                    [self.config.model.layer_list_s, self.config.model.layer_list_y]):
                if likelihood.lower() == 'poisson':
                    decoder_dict[f'{d}_{likelihood}'] = DecoderPoisson(n_a=self.config.model.n_a, 
                                                                   n_s=n,
                                                                   layer_list_s=layer_list[::-1],
                                                                   kernel_initializer_fn=self.kernel_initializer_fn,
                                                                   activation_fn=self.activation_fn)
                else:
                    decoder_dict[f'{d}_{likelihood}'] = DecoderGaussian(n_a=self.config.model.n_a, 
                                                                    n_y=n,
                                                                    layer_list_y=layer_list[::-1],
                                                                    kernel_initializer_fn=self.kernel_initializer_fn,
                                                                    activation_fn=self.activation_fn)
        return decoder_dict

    
    def forward(self, s=None, y=None, m_s=None, m_y=None):
        '''
        Performs forward pass on MRINE model

        s: torch.Tensor, spiking activity, or discrete observations. (num_seq, num_steps, n_s)
        y: torch.Tensor, LFP, or continuous signals. (num_seq, num_steps, n_y)
        m_s: torch.Tensor, mask tensor for s denoting whether s is available for a time-step or not. (num_seq, num_steps)
        m_y: torch.Tensor, mask tensor for y denoting whether y is available for a time-step or not. (num_seq, num_steps)
        '''
        
        # If the model is in traning mode and time dropout probability is not zero, apply time-dropout
        if self.training and self.config.model.td_rate > 0:
            m_s_td = time_dropout(m_s, keep_prob=1-self.config.model.td_rate) if m_s is not None else None
            m_y_td = time_dropout(m_y, keep_prob=1-self.config.model.td_rate) if m_y is not None else None
        else:
            m_s_td = m_s
            m_y_td = m_y
        
        if self.config.model.model_type.lower() == 'multi':
            # Input dropouts
            s = self.inp_dropout(s)
            y = self.inp_dropout(y)
            num_seq, num_steps, _ = s.shape

            # Initial representation of multiscale embedding factors
            a_hat = self.encoder(s=s.reshape(-1, self.config.model.n_s), 
                                 y=y.reshape(-1, self.config.model.n_y), 
                                 m_s=m_s_td,
                                 m_y=m_y_td)
            # As missing observations are handled by their modality-specific LDMs, we set mask tensor of multiscale LDM as ones tensor
            mask_kf = torch.ones(num_seq, num_steps, device=a_hat.device)
        # For single-scale networks, LDM masks are the masks of corresponding modalities
        elif self.config.model.model_type.lower() == 'single-poisson':
            s = self.inp_dropout(s)
            num_seq, num_steps, _ = s.shape
            a_hat = self.encoder(x=s.reshape(-1, self.config.model.n_s))
            mask_kf = m_s_td 
        elif self.config.model.model_type.lower() == 'single-gaussian':
            y = self.inp_dropout(y)
            num_seq, num_steps, _ = y.shape
            a_hat = self.encoder(x=y.reshape(-1, self.config.model.n_y))
            mask_kf = m_y_td
        
        a_hat = a_hat.reshape(-1, num_steps, self.config.model.n_a)

        # Run multiscale (single-scale for single-scale networks) LDM, obtain filtering and smoothing outputs
        a_hat = self.a_hat_dropout(a_hat)
        x_pred, x_filter, x_smooth, Sigma_pred, Sigma_filter, Sigma_smooth = self.ldm(a=a_hat, mask=mask_kf, do_smoothing=True)

        a_pred = (self.ldm.C[None, None, :, :] @ x_pred[:, :, :, None]).squeeze(dim=-1)
        a_filter = (self.ldm.C[None, None, :, :] @ x_filter[:, :, :, None]).squeeze(dim=-1)
        a_smooth = (self.ldm.C[None, None, :, :] @ x_smooth[:, :, :, None]).squeeze(dim=-1)

        # Obtain filtering and smoothing likelihood distribution parameters
        data_dict_pred = dict()
        data_dict_filter = dict()
        data_dict_smooth = dict()

        for key, decoder in self.decoder_dict.items():
            d, likelihood = key.split('_')
            data_dict_pred[d] = dict()
            data_dict_filter[d] = dict()
            data_dict_smooth[d] = dict()

            if likelihood.lower() == 'gaussian':
                y_pred, y_var_pred = decoder(a_pred.view(-1, self.config.model.n_a))
                y_filter, y_var_filter = decoder(a_filter.view(-1, self.config.model.n_a))
                y_smooth, y_var_smooth = decoder(a_smooth.view(-1, self.config.model.n_a))
                n = y_filter.shape[-1]

                y_pred = y_pred.view(-1, num_steps, n)
                y_var_pred = y_var_pred.view(-1, num_steps, n)

                y_filter = y_filter.view(-1, num_steps, n)
                y_var_filter = y_var_filter.view(-1, num_steps, n)

                y_smooth = y_smooth.view(-1, num_steps, n)
                y_var_smooth = y_var_smooth.view(-1, num_steps, n)
                
                data_dict_pred[d]['mu_pred'] = y_pred
                data_dict_filter[d]['mu_filter'] = y_filter
                data_dict_smooth[d]['mu_smooth'] = y_smooth
                
                # If variances is not learned, we use constant variance, e.g., ones tensor multiplied by config.model.y_var (we set it to 1)
                if self.config.model.learn_y_var:
                    data_dict_pred[d]['var_pred'] = y_var_pred
                    data_dict_filter[d]['var_filter'] = y_var_filter
                    data_dict_smooth[d]['var_smooth'] = y_var_smooth
                else:
                    data_dict_pred[d]['var_pred'] = self.config.model.y_var * torch.ones_like(y_var_pred)
                    data_dict_filter[d]['var_filter'] = self.config.model.y_var * torch.ones_like(y_var_filter)
                    data_dict_smooth[d]['var_smooth'] = self.config.model.y_var * torch.ones_like(y_var_smooth) 

            elif likelihood.lower() == 'poisson':
                fr_pred = decoder(a_pred.view(-1, self.config.model.n_a)) 
                fr_filter = decoder(a_filter.view(-1, self.config.model.n_a)) 
                fr_smooth = decoder(a_smooth.view(-1, self.config.model.n_a)) 
                n = fr_filter.shape[-1]

                fr_pred = fr_pred.reshape(-1, num_steps, n)
                fr_filter = fr_filter.reshape(-1, num_steps, n)
                fr_smooth = fr_smooth.reshape(-1, num_steps, n)

                data_dict_pred[d]['fr_pred'] = fr_pred
                data_dict_filter[d]['fr_filter'] = fr_filter
                data_dict_smooth[d]['fr_smooth'] = fr_smooth
        
        # Put all inference variables into a dictionary to return
        model_vars = dict(m_s_td=m_s_td, m_y_td=m_y_td,
                          x_pred=x_pred, x_filter=x_filter, x_smooth=x_smooth,
                          Sigma_pred=Sigma_pred, Sigma_filter=Sigma_filter, Sigma_smooth=Sigma_smooth,
                          a_pred=a_hat, a_smooth=a_smooth, a_filter=a_filter,
                          data_dict_pred=data_dict_pred, data_dict_filter=data_dict_filter, data_dict_smooth=data_dict_smooth)
        return model_vars


    def get_k_step_ahead_prediction(self, model_vars, k):
        '''
        Performs k-step-ahead prediction to obtain predicted latent factors and likelihood distribution parameters

        model_vars: dict, inference variables returned by forward() function
        k: int, future horizon to predict
        '''

        # Start from the filtered latent factors
        x_filter = model_vars['x_filter']
        num_seq = x_filter.shape[0]

        # Forward iterate k-times to obtain k-step-ahead predicted latent factors
        if k > 0:
            x_pred_k = x_filter[:, :-k, :] # [x_k|0, x_{k+1}|1, ..., x_{T}|{T-k}]
            for _ in range(k):
                x_pred_k = (self.ldm.A[None, None, :, :] @ x_pred_k[:, :, :, None]).squeeze(dim=-1) 
            a_pred_k = (self.ldm.C[None, None, :, :] @ x_pred_k[:, :, :, None]).squeeze(dim=-1)
        elif k == 0:
            # k=0 means smoothing
            a_pred_k = model_vars['a_smooth']

        # After obtaining k-step-ahead predicted embedding factors, obtain predicted likelihood distribution parameters
        data_dict_pred_k = {}
        for key, decoder in self.decoder_dict.items():
            d, likelihood = key.split('_')
            data_dict_pred_k[d] = {}
            if likelihood.lower() == 'gaussian':
                y_pred_k, y_var_pred_k = decoder(a_pred_k.view(-1, self.config.model.n_a))
                n = y_pred_k.shape[-1]
                if not self.config.model.learn_y_var:
                    y_var_pred_k = self.config.model.y_var * torch.ones_like(y_var_pred_k)

                data_dict_pred_k[d]['mu_pred'] = y_pred_k.reshape(num_seq, -1, n)
                data_dict_pred_k[d]['var_pred'] = y_var_pred_k.reshape(num_seq, -1, n)
            
            elif likelihood.lower() == 'poisson':
                fr_pred_k = decoder(a_pred_k.view(-1, self.config.model.n_a)) 
                n = fr_pred_k.shape[-1]
                data_dict_pred_k[d]['fr_pred'] = fr_pred_k.reshape(num_seq, -1, n)
        return data_dict_pred_k


    def get_k_step_ahead_prediction_likelihood(self, model_vars, k, s=None, y=None, m_s=None, m_y=None):
        '''
        Computes k-step-ahead predicted likelihoods after computing predicted likelihood distribution parameters

        model_vars: dict, inference variables returned by forward() function
        k: int, future horizon to predict
        s: torch.Tensor, spiking activity, or discrete observations. (num_seq, num_steps, n_s)
        y: torch.Tensor, LFP, or continuous signals. (num_seq, num_steps, n_y)
        m_s: torch.Tensor, mask tensor for s denoting whether s is available for a time-step or not. (num_seq, num_steps)
        m_y: torch.Tensor, mask tensor for y denoting whether y is available for a time-step or not. (num_seq, num_steps)
        '''

        # Get k step ahead prediction
        data_dict_pred_k = self.get_k_step_ahead_prediction(model_vars=model_vars, k=k)

        # Iterate over modalities
        log_ll_pred_k_dict = {}
        scaled_log_ll_pred_k_sum = 0
        for key in self.decoder_dict.keys():
            d, likelihood = key.split('_')
            data = s if d == 's' else y
            mask = m_s if d == 's' else m_y
            n = data.shape[-1]

            data_flat = data[:, k:, :].reshape(-1,  n)
            mask_flat = mask[:, k:].reshape(-1, 1)
            
            if likelihood.lower() == 'gaussian':
                y_pred_k = data_dict_pred_k[d]['mu_pred']
                y_var_pred_k = data_dict_pred_k[d]['var_pred']
                
                log_ll_pred_k = log_likelihood_gaussian(y_flat=data_flat, 
                                                          mu_flat=y_pred_k.reshape(-1, n),
                                                          var_flat=y_var_pred_k.reshape(-1, n),
                                                          mask_flat=mask_flat)
                scaled_log_ll_pred_k = log_ll_pred_k 
            elif likelihood.lower() == 'poisson':
                fr_pred_k = data_dict_pred_k[d]['fr_pred']
                log_ll_pred_k = log_likelihood_poisson(s_flat=data_flat,
                                                         fr_flat=fr_pred_k.reshape(-1, n),
                                                         mask_flat=mask_flat)
            
            # If multiscale network is being trained, we scale the likelihood of s to approximately make it around similar scales with that of y
            if self.config.model.model_type == 'multi' and d == 's':
                scaled_log_ll_pred_k = log_ll_pred_k * self.config.loss.tau 
            else:
                scaled_log_ll_pred_k = log_ll_pred_k 

            # Likelihood sum for k-step-ahead prediction across modalities
            scaled_log_ll_pred_k_sum += scaled_log_ll_pred_k

            # Dump each modality's predicted likelihood to dictionary to be logged further in the traning
            log_ll_pred_k_dict[d] = log_ll_pred_k 
        return log_ll_pred_k_dict, scaled_log_ll_pred_k_sum


    def compute_loss(self, model_vars, s=None, y=None, m_s=None, m_y=None):
        '''
        Computes all the loss values for parameter learning 

        model_vars: dict, inference variables returned by forward() function
        s: torch.Tensor, spiking activity, or discrete observations. (num_seq, num_steps, n_s)
        y: torch.Tensor, LFP, or continuous signals. (num_seq, num_steps, n_y)
        m_s: torch.Tensor, mask tensor for s denoting whether s is available for a time-step or not. (num_seq, num_steps)
        m_y: torch.Tensor, mask tensor for y denoting whether y is available for a time-step or not. (num_seq, num_steps)
        '''

        # Dump individual loss values for logging or Tensorboard
        loss_dict = dict()

        # Compute k-step-ahead and smoothing (0-step) likelihoods
        scaled_log_ll_pred_sum = 0  # total predicted likelihood for all future step ahead predictions
        for k in self.config.loss.steps_ahead:
            if k > 0:
                loss_dict_key = f'log_ll_pred_{k}'
            elif k == 0:
                loss_dict_key = 'log_ll_smoothing'

            loss_dict[loss_dict_key], scaled_log_ll_pred_k_sum = self.get_k_step_ahead_prediction_likelihood(model_vars=model_vars,
                                                                                                             k=k,
                                                                                                             s=s, y=y,
                                                                                                             m_s=model_vars['m_s_td'], m_y=model_vars['m_y_td'])
            scaled_log_ll_pred_sum += scaled_log_ll_pred_k_sum
            # unflatten the loss_dict for loss_dict_key
            loss_dict = extract_dict_from_key(loss_dict, loss_dict_key)
        model_loss = -scaled_log_ll_pred_sum  # we want to maximize the total future step ahead predicted likelihoods 
        loss_dict['model_loss'] = model_loss

        # L2 regularization loss 
        l2_reg = 0
        for name, param in self.named_parameters():
            if 'weight' in name:
                l2_reg = l2_reg + self.config.loss.scale_l2 * torch.norm(param)
        loss_dict['l2_reg'] = l2_reg

        # Smoothness regularization loss over consecutive time-steps for smoothed decoder outputs
        scaled_sm_reg_loss_dec_sum = 0

        for key in self.decoder_dict.keys():
            d, likelihood = key.split('_')
            mask = m_s if d == 's' else m_y
            scale_sm_reg = self.config.loss.scale_sm_reg_s if d == 's' else self.config.loss.scale_sm_reg_y

            data_dict_smooth = model_vars['data_dict_smooth']
            if likelihood.lower() == 'poisson':
                s_smooth = data_dict_smooth[d]['fr_smooth']
                sm_reg_loss = compute_sm_reg_loss(mean=s_smooth, likelihood=likelihood, mask=mask) 
            elif likelihood.lower() == 'gaussian':
                y_smooth = data_dict_smooth[d]['mu_smooth']
                y_var_smooth = data_dict_smooth[d]['var_smooth']
                sm_reg_loss = compute_sm_reg_loss(mean=y_smooth, var=y_var_smooth, likelihood=likelihood, mask=mask)
            
            scaled_sm_reg_loss_dec_sum += sm_reg_loss * scale_sm_reg
            
            # Dump each smoothness regularization loss into a dictionary to be logged
            loss_dict[f'sm_reg_loss_{d}'] = sm_reg_loss 

        # Smoothness regularization loss over consecutive time-steps for x_smooth
        x_smooth = model_vars['x_smooth'][:, :, :self.config.model.n_x//2]
        Sigma_smooth_diag = torch.diagonal(model_vars['Sigma_smooth'], dim1=-1, dim2=-2)[:, :, :self.config.model.n_x//2]

        sm_reg_loss_x = compute_sm_reg_loss(mean=x_smooth, var=Sigma_smooth_diag, likelihood='gaussian')
        loss_dict['sm_reg_loss_x'] = sm_reg_loss_x

        # Final loss is summation of model loss (k-steps ahead + smoothing log likelihood), L2 regularization loss and smoothness regularizations on smoothed decoder outputs and multiscale (single-scale) latent factors
        loss = model_loss + l2_reg + scaled_sm_reg_loss_dec_sum + self.config.loss.scale_sm_reg_x * sm_reg_loss_x
        loss_dict['total_loss'] = loss
        return loss, loss_dict


    def step(self, s=None, y=None, m_s=None, m_y=None):
        '''
        Step function that performs forward pass and loss calculation

        model_vars: dict, inference variables returned by forward() function
        s: torch.Tensor, spiking activity, or discrete observations. (num_seq, num_steps, n_s)
        y: torch.Tensor, LFP, or continuous signals. (num_seq, num_steps, n_y)
        m_s: torch.Tensor, mask tensor for s denoting whether s is available for a time-step or not. (num_seq, num_steps)
        m_y: torch.Tensor, mask tensor for y denoting whether y is available for a time-step or not. (num_seq, num_steps)
        '''
        
        model_vars = self.forward(s=s, y=y, 
                                  m_s=m_s, m_y=m_y)
        loss, loss_dict = self.compute_loss(model_vars, 
                                            s=s, y=y, 
                                            m_s=m_s, m_y=m_y)
        return loss, loss_dict, model_vars
        


                    
 
        
        

