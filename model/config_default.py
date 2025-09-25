from yacs.config import CfgNode as CN
import torch

from python_utils import flatten_dict, unflatten_dict

## Initialize default config
## Set device, seed
_config = CN() 
_config.device = 'cpu'
_config.seed = int(torch.randint(low=0, high=100000, size=(1,)))

# Dump model related settings
_config.model = CN() 

_config.model.model_type = 'multi' # please choose 'multi' or 'single-poisson' or 'single-gaussian'
_config.model.likelihood_s = 'poisson' # it can be 'poisson' or 'gaussian'
_config.model.likelihood_y = 'gaussian' # it can be 'poisson' or 'gaussian'
_config.model.dropout_rate = 0.1 # regular dropout rate for encoder inputs and outputs
_config.model.td_rate = 0.3 # time dropout rate

_config.model.layer_list_s = [128,128,128] # modality-specific encoder network (or single-scale encoder network) architecture of modality s
_config.model.layer_list_y = [128,128,128] # modality-specific encoder network (or single-scale encoder network) architecture of modality y
_config.model.layer_list_m = [128] # fusion network architecture

_config.model.activation = 'tanh' # activation function of MLP hidden layers. see nn.get_activation_function() for options 
_config.model.learn_y_var = False # whether to learn variance of gaussian modality or not. if learnt, we learn global variance across time and batches. False by default
_config.model.y_var = 1 # constant variance scale of modality y

_config.model.n_s = -1 # number of channels (observations dimensions) of modality s
_config.model.n_y = -1 # number of channels (observations dimensions) of modality y
_config.model.n_a = 64 # dimension of multiscale or single-scale embedding factors 
_config.model.n_x = 64 # dimension of multiscale or single-scale latent factors 
_config.model.apply_normalize_A = False # normalizes A by limiting maximum singular value, can stabilize training if it produces NaNs
_config.model.init_A_scale = 0.9 # initialization scale for identity initialized transition matrices
_config.model.init_C_scale = 0.9 # initialization scale for randomly initialized observation matrices
_config.model.init_Q_scale = 0.5 # initialization of state noise covariance matrices
_config.model.init_R_scale = 1 # initialization of observation noise covariance matrices
_config.model.init_cov_scale = 1 # initialization of initial predicted error covariance of Kalman filter
_config.model.kernel_initializer = 'xavier_normal' # initialization for MLP layers' weights. see nn.get_kernel_initializer_function() for options

_config.model.save_dir = './results' # save directory where checkpoints, plots, log files, encoding and decoding results are saved
_config.model.save_steps = 100 # frequency to save checkpoints (every save_steps epochs)

# Dump loss related settings
_config.loss = CN()
_config.loss.tau = 3 # likelihood scaling parameter
_config.loss.scale_l2 = 1 # L2 regularization scale
_config.loss.steps_ahead = [0,1,2,3,4] # future step aheads to optimize MRINE. 0 means smoothing
_config.loss.scale_sm_reg_s = 250 # scale for smoothness regularization on smoothed decoder outputs for modality s
_config.loss.scale_sm_reg_y = 10 # scale for smoothness regularization on smoothed decoder outputs for modality y
_config.loss.scale_sm_reg_x = 30 # scale for smoothness regularization on smoothed multiscale or single-scale latent factors (applied on half of the dimensions)

# Dump training related settings
_config.train = CN()
_config.train.batch_size = 32 # number of trials/segments in a batch
_config.train.num_epochs = 500 # number of epochs to train the model
_config.train.valid_step = 1 # frequency to perform evaluation on the validation set
_config.train.plot_save_steps = 50 # frequency to save training/validation plots

# Dump loading settings
_config.load = CN()
_config.load.resume_train = False # if True and appropriate file_name is provided, the training will start from where it was left 
_config.load.file_name = "" # file name  of the model checkpoint to load, without .pth extension

# Dump learning rate related settings
_config.lr = CN()

_config.lr = CN()
_config.lr.init_lr = 0.01 # initial learning rate of the optimizer
_config.lr.base_lr = 0.001 # base learning rate of CyclicLR, see https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.CyclicLR.html for details
_config.lr.max_lr = 0.01 # maximum learning rate of CyclicLR
_config.lr.gamma = 0.99 # exponential decay scalar for maximum learning rate
_config.lr.mode = 'exp_range' # if gamma is less than 1, set this to 'exp_range'. if gamma is 1, set it to triangular
_config.lr.step_size_up = 10 # number of steps (epochs) for learning rate increase from base_lr to max_lr

# Dump optimizer related parameters
_config.optim = CN()
_config.optim.grad_clip = 0.1 # gradient clipping norm


def get_cfg_defaults():
    '''
    Returns the default config
    '''
    return _config.clone()


def update_config(config, new_config):
    '''
    Updates the config from new config

    config: CfgNode, config to update
    new_config: CfgNode, config with values to be used for update 
    '''
    flat_config = flatten_dict(config)
    flat_new_config = flatten_dict(new_config)
    flat_config.update(flat_new_config)
    return CN(unflatten_dict(flat_config))


def create_config_from_args(args):
    '''
    Creates the config from arguments passed through parser

    args: Namespace, optional arguments for main functions. see run_kfold_mrine_lorenz.py and run_kfold_mrine_nhp.py 
    '''

    config = get_cfg_defaults()
    args_dict = vars(args)
    for key, val in args_dict.items():
        if val is not None:
            if key in config.model.keys():
                config.model[key] = val
            elif key in config.loss.keys():
                config.loss[key] = val
            elif key in config.train.keys():
                config.train[key] = val
            elif key in config.load.keys():
                config.load[key] = val
            elif key in config.lr.keys():
                config.lr[key] = val
            elif key in config.optim.keys():
                config.optim[key] = val
            elif key in config.keys():
                config[key] = val
    if config.lr.gamma == 1:
        config.lr.mode = 'triangular'
    return config

