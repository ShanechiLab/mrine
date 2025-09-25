import torch 
import torch.nn as nn

def compute_sm_reg_loss(mean, var=None, likelihood='poisson', mask=None):
    '''
    Computes smoothness regularization loss for specified likelihood distribution

    mean: torch.Tensor, mean of the distribution to compute smoothness regularization, (num_seq, num_steps, n_x)
    var: torch.Tensor, variance of the distribution to compute smoothness regularization, (num_seq, num_steps, n_x). None by default, ones tensor is used instead (unit_variance)
    likelihood: str, likelihood to compute KL divergence. Options are 'poisson' and 'gaussian'
    mask: torch.Tensor, mask tensor denoting the availability of time-steps, (num_seq, num_steps)
    '''
    
    num_seq, num_steps, dim_x = mean.shape
    if var is None:
        var = torch.ones_like(mean, dtype=torch.float32).to(mean.device)
    if mask is None:
        mask = torch.ones((num_seq, num_steps), dtype=torch.float32).to(mean.device)
    var[var<0] = 1e-4 # just in case for numerical stability

    try: # fails when number of missing samples across batches are different
        mask_bool = mask.type(torch.bool).unsqueeze(dim=-1).tile(1, 1, dim_x)
        masked_mean = mean[mask_bool].reshape(num_seq, -1, dim_x)
        masked_var = var[mask_bool].reshape(num_seq, -1, dim_x)

        if likelihood.lower() == 'poisson':
            smoothness_loss = (masked_mean[:, :-1, :] * torch.log(masked_mean[:, :-1, :] / masked_mean[:, 1:, :]) + masked_mean[:, 1:, :] - masked_mean[:, :-1, :]).mean() 
        elif likelihood.lower() == 'gaussian':
            smoothness_loss = (0.5 * torch.log(masked_var[:, 1:, :] / masked_var[:, :-1, :]) + (torch.square(torch.diff(masked_mean, dim=1)) + masked_var[:, :-1, :]) / (2*masked_var[:, 1:, :]) - 1/2 ).mean() 
        return smoothness_loss
    except: # instead, compute it across each trial
        smoothness_loss = []
        for i in range(num_seq):
            mask_bool = mask[i, :, :].type(torch.bool).unsqueeze(dim=-1).tile(1, dim_x)
            masked_mean = mean[i, :, :][mask_bool].reshape(-1, dim_x)
            masked_var = var[i, :, :][mask_bool].reshape(-1, dim_x)
            if likelihood.lower() == 'poisson':
                smoothness_loss_i = (masked_mean[:-1, :] * torch.log(masked_mean[:-1, :] / masked_mean[1:, :]) + masked_mean[1:, :] - masked_mean[:-1, :]).mean() 
            elif likelihood.lower() == 'gaussian':
                smoothness_loss_i = (0.5 * torch.log(masked_var[1:, :] / masked_var[:-1, :]) + (torch.square(torch.diff(masked_mean, dim=0)) + masked_var[:-1, :]) / (2*masked_var[1:, :]) - 1/2 ).mean() 
            smoothness_loss.append(smoothness_loss_i)
        return smoothness_loss_i.sum() / num_seq


def time_dropout(mask, keep_prob):
    '''
    Applies time dropout to mask matrices

    mask: torch.Tensor, mask denoting the availability of observations at time-steps, (num_seq, num_steps)
    keep_prob: float, keep probability of a time-step
    '''

    td_mask = (torch.bernoulli(torch.full_like(mask, keep_prob) * mask)).float().to(mask.device)
    return td_mask


def init_ss_parameters(n_x, n_a, init_A_scale, init_C_scale, init_Q_scale, init_R_scale, init_cov_scale, device='cpu'):
    A = init_A_scale * torch.eye(n_x, dtype=torch.float32, device=device) 
    C = init_C_scale * torch.randn(n_a, n_x, dtype=torch.float32, device=device) 

    Q_log_diag = torch.log(init_Q_scale * torch.ones(n_x, dtype=torch.float32, device=device))
    R_log_diag = torch.log(init_R_scale * torch.ones(n_a, dtype=torch.float32, device=device))
    Sigma_0 = init_cov_scale * torch.eye(n_x, dtype=torch.float32, device=device)

    x_0 = torch.zeros(n_x, dtype=torch.float32, device=device)
    return A, C, Q_log_diag, R_log_diag, Sigma_0, x_0


def log_likelihood_gaussian(y_flat, mu_flat, var_flat=None, mask_flat=None):    
    '''
    Returns average Gaussian log likelihood (with mean-field approximation assumption (independent across last dimension)) 

    x_flat: torch.Tensor, data to compute whose likelihood, (num_seq*num_steps, n_y)
    mu_flat: torch.Tensor, mean of the distribution, (num_seq*num_steps, n_y)
    var_flat: torch.Tensor, variance of the distribution, (num_seq*num_steps, n_y)
    mask_flat: torch.Tensor, mask to compute log likelihood loss, (num_seq*num_steps, 1)
    '''

    if mask_flat is None: 
        mask_flat = torch.ones(y_flat.shape[:-1], dtype=torch.float32)
    
    if var_flat is None:
        var_flat = torch.ones_like(mu_flat, dtype=torch.float32)
    
    if len(mask_flat.shape) != len(y_flat.shape):
        mask_flat = mask_flat.unsqueeze(dim=-1)

    mu_flat = torch.nan_to_num(mu_flat, nan=0, posinf=0, neginf=0) # in case undesired values occur
    var_flat = torch.nan_to_num(var_flat, nan=0, posinf=0, neginf=0) # in case undesired values occur

    dist = torch.distributions.Normal(loc=mu_flat, scale=torch.sqrt(var_flat))
    log_lik = dist.log_prob(y_flat)
    
    # Detach the gradients from missing time-steps
    log_lik_bp = torch.mul(mask_flat, log_lik)
    log_lik_mask = torch.mul(1-mask_flat, log_lik).detach()
    log_lik = log_lik_bp + log_lik_mask

    # After missing time-steps loss is detached, recompute the masked loss
    log_lik = torch.mul(mask_flat, log_lik)

    if mask_flat.shape[-1] != mu_flat.shape[-1]: # which means shape of mask_flat is of dimension 1
        num_el = mask_flat.sum() * mu_flat.shape[-1]
    else:
        num_el = mask_flat.sum()
    
    log_lik = log_lik.sum() / num_el
    return log_lik


def log_likelihood_poisson(s_flat, fr_flat, mask_flat=None):
    '''
    Returns average Poisson log likelihood  

    s_flat: torch.Tensor, data to compute whose likelihood, (num_samp, n_s)
    fr_flat: torch.Tensor, mean of the distribution, (num_samp, n_s)
    mask_flat: torch.Tensor, mask to compute log likelihood loss, (num_samp, 1)
    '''

    if mask_flat is None: 
        mask_flat = torch.ones(s_flat.shape[:-1], dtype=torch.float32)
    
    if len(mask_flat.shape) != len(s_flat.shape):
        mask_flat = mask_flat.unsqueeze(dim=-1)

    fr_flat = torch.nan_to_num(fr_flat, nan=0, posinf=0, neginf=0) # in case undesired values occur

    dist = torch.distributions.Poisson(rate=fr_flat)
    log_lik = dist.log_prob(s_flat)
    
    # detach the gradients from missing time-steps
    log_lik_bp = torch.mul(mask_flat, log_lik)
    log_lik_mask = torch.mul(1-mask_flat, log_lik).detach()
    log_lik = log_lik_bp + log_lik_mask

    # after missing time-steps loss is detached, recompute the masked loss
    log_lik = torch.mul(mask_flat, log_lik)

    if mask_flat.shape[-1] != fr_flat.shape[-1]: # which means shape of mask_flat is of dimension 1
        num_el = mask_flat.sum() * fr_flat.shape[-1]
    else:
        num_el = mask_flat.sum()
    
    log_lik = log_lik.sum() / num_el
    return log_lik


def get_activation_function(activation_str):
    '''
    Returns activation function given the activation function's name

    activation_str: str, activation function's name
    '''

    if activation_str.lower().lower() == 'elu':
        return nn.ELU()
    elif activation_str.lower().lower() == 'hardtanh':
        return nn.Hardtanh()
    elif activation_str.lower().lower() == 'leakyrelu':
        return nn.LeakyReLU()
    elif activation_str.lower().lower() == 'relu':
        return nn.ReLU()
    elif activation_str.lower().lower() == 'rrelu':
        return nn.RReLU()
    elif activation_str.lower().lower() == 'sigmoid':
        return nn.Sigmoid()
    elif activation_str.lower().lower() == 'mish':
        return nn.Mish()
    elif activation_str.lower().lower() == 'tanh':
        return nn.Tanh()
    elif activation_str.lower().lower() == 'tanhshrink':
        return nn.Tanhshrink()
    elif activation_str.lower().lower() == 'linear':
        return lambda x:x


def get_kernel_initializer_function(kernel_initializer_str):
    '''
    Returns kernel initialization function given the kernel initialization function's name

    kernel_initializer_str: str, kernel initialization function's name
    '''
    
    if kernel_initializer_str.lower().lower() == 'uniform':
        return nn.init.uniform_
    elif kernel_initializer_str.lower().lower() == 'normal':
        return nn.init.normal_
    elif kernel_initializer_str.lower().lower() == 'xavier_uniform':
        return nn.init.xavier_uniform_
    elif kernel_initializer_str.lower().lower() == 'xavier_normal': # good for tanh and sigmoid activation
        return nn.init.xavier_normal_
    elif kernel_initializer_str.lower().lower() == 'kaiming_uniform': 
        return nn.init.kaiming_uniform_
    elif kernel_initializer_str.lower().lower() == 'kaiming_normal': # good for relu activation
        return nn.init.kaiming_normal_
    elif kernel_initializer_str.lower().lower() == 'orthogonal':
        return nn.init.orthogonal_
    elif kernel_initializer_str.lower().lower() == 'default':
        return lambda x:x