from python_utils import convert_to_tensor
import torch
import numpy as np
from copy import deepcopy
from scipy.stats import pearsonr
from nn import log_likelihood_poisson, log_likelihood_gaussian

def create_kfold_inds(num_seq, num_folds=5, fold=1):
    '''
    Creates training and validation trial indices 
    num_seq: number of sequences (trials)
    num_folds: number of folds for which k-fold CV is performed
    fold: fold number for which training and validation indices will be returned
    '''
    num_seq_per_fold = num_seq // num_folds
    for i in range(1, num_folds+1):        
        if i == num_folds:
            index_valid = torch.arange( (i-1) * num_seq_per_fold, num_seq)
        else:    
            index_valid = torch.arange( (i-1) * num_seq_per_fold, i * num_seq_per_fold )
        index_train = torch.tensor([j for j in range(num_seq) if j not in index_valid])
        
        if i == fold:
            break
    return index_train, index_valid


def get_tau(s=None, y=None, m_s=None, m_y=None, likelihood_s=None, likelihood_y=None, use_gaus_var=False):
    '''
    Automatically sets tau (scaling parameter between different likelihood distributions). Computes likelihoods
    for available time-steps.

    s: torch.Tensor, spiking activity, or discrete observations. (num_seq, num_steps, n_s)
    y: torch.Tensor, LFP, or continuous signals. (num_seq, num_steps, n_y)
    m_s: torch.Tensor, mask tensor for s denoting whether s is available for a time-step or not. (num_seq, num_steps)
    m_y: torch.Tensor, mask tensor for y denoting whether y is available for a time-step or not. (num_seq, num_steps)
    likelihood_s: str, likelihood for modality s
    likelihood_y: str, likelihood for modality y
    use_gaus_var = bool, Either to use variance of Gaussian signal  while computing mean likelihood, False by default
    '''
    if s is None or y is None:
        return 1
    
    mean_lls = []
    for data, mask, likelihood in zip([s, y], [m_s, m_y], [likelihood_s, likelihood_y]):
        data_flat = data.reshape(-1, data.shape[-1])
        mask_flat = mask.reshape(-1, 1)
        mask_flat_bool = mask_flat.bool().tile(1, data.shape[-1])

        data_flat_masked = data_flat[mask_flat_bool].reshape(-1, data.shape[-1])
        data_mean_flat = data_flat_masked.mean(dim=0).tile(data_flat.shape[0], 1)

        if likelihood.lower() == 'gaussian':
            if use_gaus_var:
                data_var_flat = data_flat_masked.var(dim=0).tile(data_flat.shape[0], 1)
            else:
                data_var_flat = None  
 
            mean_ll = log_likelihood_gaussian(y_flat=data_flat, 
                                              mu_flat=data_mean_flat,
                                              var_flat=data_var_flat,
                                              mask_flat=mask_flat)
        else:
            mean_ll = log_likelihood_poisson(s_flat=data_flat, 
                                             fr_flat=data_mean_flat,
                                             mask_flat=mask_flat)
        mean_lls.append(mean_ll)
    
    mean_lls = torch.tensor(mean_lls).abs()

    # The first element was likelihood of s, second element was that of y. We use tau to scale s.
    return float(mean_lls[1] / mean_lls[0])



def get_pearson_cc(x, xhat):
    '''
    Computes Pearson correlation coefficient (CC) between 2 tensors. If tensors are 3D, 
    they are concatenated across first 2 dimensions before computing CC. Returns CC values 
    across each n_x  dimension and their average.

    x: torch.Tensor, True data tensor. (num_seq, num_steps, n_x) or (num_steps, n_x)
    xhat: torch.Tensor, Reconstructed data tensor. (num_seq, num_steps, n_x) or (num_steps, n_x)
    '''
    assert x.shape == xhat.shape, f'dimensions of x {x.shape} and xhat {xhat.shape} do not match'

    if len(x.shape) == 3:
        dim_x = x.shape[-1]
        x = x.reshape(-1, dim_x)
        xhat = xhat.reshape(-1, dim_x)

    x = convert_to_tensor(x).detach().cpu().numpy() # make sure every array/tensor has .numpy() function, pearsonr works on ndarrays
    xhat = convert_to_tensor(xhat).detach().cpu().numpy()

    ccs = []
    for dim in range(x.shape[-1]):
        cc, _ = pearsonr(x[:, dim], xhat[:, dim])
        ccs.append(cc)
    ccs = torch.tensor(ccs, dtype=torch.float32)
    ccs_mean = torch.nanmean(ccs.nan_to_num(posinf=torch.nan, neginf=torch.nan))
    return ccs, ccs_mean


def z_score_tensor(x, fit=True, **kwargs):
    '''
    Performs z-scoring on x either by using mean and std provided as key arguments (fit should be False), or computes 
    statistics over x. If x is 3D, it is concatenated across first 2 dimensions. 

    x: torch.Tensor, tensor to z-score, (num_seq, num_steps, n_x) or (num_steps, n_x)
    '''
    with torch.no_grad():
        x = convert_to_tensor(x)

        x_resh = x.reshape(-1, x.shape[-1])

        if fit:
            mean = torch.mean(x_resh, dim=0)
            std = torch.std(x_resh, dim=0)
        else:
            mean = kwargs.pop('mean')
            std = kwargs.pop('std')  

        # to prevent nan values
        std[std==0] = 1
        x_resh = (x_resh - mean) / std
        x_z_scored = x_resh.reshape(x.shape)
        return x_z_scored, mean, std  


def z_score_train_valid_tensor(x, index_train, index_valid, mask=None, x_for_stats=None):
    '''
    Performs z-scoring on x by calculating statistics over trials denoted  by index_train, and applies 
    the calculated statistics on trials denoted by index_valid.  

    x: torch.Tensor, tensor to z-score, (num_seq, num_steps, n_x) or (num_steps, n_x)
    index_train: torch.Tensor, trial indices for training data 
    index_valid: torch.Tensor, trial indices for validation data
    mask: torch.Tensor, mask tensor used for masking while calculating statistics, (num_seq, num_steps) or (num_steps)
    x_for_stats: torch.Tensor, if provided, z-score statistics will be computed by using this tensor and applied on x, (num_seq, num_steps, n_x) or (num_steps, n_x)
    '''
    num_seq, num_steps, dim_x = x.shape
    if mask is None:
        mask = torch.ones((num_seq, num_steps), dtype=torch.float32)

    mask_bool = mask.type(torch.bool).unsqueeze(dim=-1).tile(1, 1, dim_x)

    if x_for_stats is None:
        x_for_stats = deepcopy(x)

    x_for_stats = x_for_stats[mask_bool].reshape(num_seq, -1, dim_x)

    x_for_stats_train = x_for_stats[index_train, :, :]
    _, mean, std = z_score_tensor(x_for_stats_train, fit=True)
    x_zs, _, _ = z_score_tensor(x, mean=mean, std=std, fit=False)   

    x_train = x_zs[index_train, :, :]
    x_valid = x_zs[index_valid, :, :]
    return x_train, x_valid, mean, std 


def get_mask(data_shape, ds_rate=1):
    '''
    Creates the mask tensor 

    data_shape: list/torch.Tensor, 2D tensor/list denoting [num_seq, num_steps]
    ds_rate: int, downsampling rate on the tensor. For instance, for 50 ms LFP signals, ds_rate is set to 5, 1 out of every 5 observation is used
    '''
    if len(data_shape) < 2:
        assert False, 'During mask generation, data with dimension less than 2 encountered!'
    ds_rate = int(ds_rate)

    mask = torch.zeros(data_shape[0], data_shape[1], dtype=torch.float32)
    mask[:, ::ds_rate] = 1
    return mask


def get_dropped_mask(mask, sample_drop_per):
    '''
    Creates sample dropped mask to be used in inference

    mask: torch.Tensor, original mask tensor before sample dropping, (num_seq, num_steps) or (num_steps)
    sample_drop_per: float, sample dropping probability
    '''
    dropped_mask = deepcopy(mask)
    for i in range(mask.shape[0]):
        ones_inds = np.where(mask[i] == 1)[0]
        dropped_mask[i, ones_inds] = torch.tensor(np.random.choice([0, 1], size=ones_inds.shape[0], p=[sample_drop_per, 1-sample_drop_per]), dtype=torch.float32)
    return dropped_mask
