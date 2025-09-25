from kfold_cv import run_kfold_cv
from time_series_utils import get_mask, get_dropped_mask
import model.config_default as cfg

import argparse
import torch
import yacs 


def main(args):
    print(f'-------------------------------------analysis starts with arguments: {args} ----------------------------------------------')

    # Create K-Fold settings
    kfold_settings = {}
    kfold_settings['num_folds'] = args.num_folds
    kfold_settings['which_folds'] = args.which_folds
    kfold_settings['z_score_data'] = args.z_score_data
    kfold_settings['autoset_tau'] = args.autoset_tau

    for seshnum in args.seshnum_list:
        # Load the data provided in ./data.
        load_path = f'{args.data_dir}/nhp/{args.dataset_name}/session_{seshnum}.pt'
        data = torch.load(load_path)
        s = data['s'][:, :, :args.n_s]
        y = data['y'][:, :, :args.n_y]

        if len(y.shape) == 4: # means LFP power signals
            num_seq, num_timesteps, _, _ = y.shape
            y = y.reshape(num_seq, num_timesteps, -1)

        target = data['kinem'] # Will be decoded from latent factors
        
        # Get modalities based on what model to train, set config path and save directory
        if args.model_type == 'multi':
            m_s_zs = get_mask(s.shape, ds_rate=1)
            if args.sample_drop_per_s != 0:
                m_s = get_dropped_mask(m_s_zs, sample_drop_per=args.sample_drop_per_s)
            else:
                m_s = m_s_zs
            
            m_y_zs = get_mask(y.shape, ds_rate=args.timescale_diff)
            if args.sample_drop_per_y != 0:
                m_y = get_dropped_mask(m_y_zs, sample_drop_per=args.sample_drop_per_y)
            else:
                m_y = m_y_zs

            timescale_str = 'same_timescale_10ms' if args.timescale_diff == 1 else f'timescale_diff_10ms_{(args.timescale_diff*10):.0f}ms'
            save_dir = f'{args.save_dir}/{args.dataset_name}/multi/{timescale_str}/session_{seshnum}/n_s{args.n_s}-n_y{args.n_y}'
            config_path = f'./configs/nhp/{args.dataset_name}/multi/{timescale_str}.yaml'
        elif args.model_type == 'single-poisson':
            m_s_zs = get_mask(s.shape, ds_rate=1)
            if args.sample_drop_per_s != 0:
                m_s = get_dropped_mask(m_s_zs, sample_drop_per=args.sample_drop_per_s)
            else:
                m_s = m_s_zs
            y, m_y, m_y_zs = None, None, None
            save_dir = f'{args.save_dir}/{args.dataset_name}/single-poisson/10ms/session_{seshnum}/n_s{args.n_s}'
            config_path = f'./configs/nhp/{args.dataset_name}/single-poisson.yaml'
        elif args.model_type == 'single-gaussian':
            y = y[:, ::args.timescale_diff, :]
            target = target[:, ::args.timescale_diff, :]
            m_y_zs = get_mask(y.shape, ds_rate=1)
            if args.sample_drop_per_y != 0:
                m_y = get_dropped_mask(m_y_zs, sample_drop_per=args.sample_drop_per_y)
            else:
                m_y = m_y_zs
            s, m_s, m_s_zs = None, None, None
            save_dir = f'{args.save_dir}/{args.dataset_name}/single-gaussian/{(args.timescale_diff*10):.0f}ms/session_{seshnum}/n_y{args.n_y}'
            config_path = f'./configs/nhp/{args.dataset_name}/single-gaussian/{(args.timescale_diff*10):.0f}ms.yaml'
            
        # Create configs for MRINE from args, or load the config provided in ./configs 
        if not args.load_config:
            config = cfg.create_config_from_args(args)
        else:
            with open(config_path, 'r') as cfg_f:
                config = yacs.config.load_cfg(cfg_f)
            config.device = args.device
            default_config = cfg.get_cfg_defaults()
            config = cfg.update_config(default_config, config)

        # Update the save directory
        config.model.save_dir = save_dir
        
        # Run K-Fold CV
        run_kfold_cv(train_mrine=args.train_mrine, 
                     config=config, kfold_settings=kfold_settings,
                     sample_drop_per_s=args.sample_drop_per_s, sample_drop_per_y=args.sample_drop_per_y,
                     s=s, y=y, m_s=m_s, m_y=m_y,
                     m_s_zs=m_s_zs, m_y_zs=m_y_zs,
                     target=target,
                     do_decode_target=True, decode_target_ds_rate=args.timescale_diff,
                     which_latents=['x_filter', 'x_smooth'], compute_cc_flat=True)


if __name__.lower() == '__main__':  
    parser = argparse.ArgumentParser(description='MRINE on the NHP Dataset')

    # Experiment related settings
    parser.add_argument('--dataset_name', default='center_out_reaching', help='Which NHP dataset to run. Options are grid_reaching (NHP grid reaching dataset) and center_out_reaching (NHP center-out reaching dataset).')
    parser.add_argument('--model_type', default='multi', help='Which model to run. Options are multi (MRINE), single-poisson and single-gaussian (single-scale networks).')
    parser.add_argument('--n_s', type=int, default=20, help='Number of channels for spiking activity')
    parser.add_argument('--n_y', type=int, default=20, help='Number of channels for LFP')
    parser.add_argument('--timescale_diff', type=int, default=5, help='Timescale difference between LFP and spiking activity. 1 means same timescale (10 ms), 5 means 50 ms for LFP and 10 ms for spikes.')
    parser.add_argument('--seshnum_list', nargs='+', type=int, default=[1], help='Which sessions to run.')

    # Save path
    parser.add_argument('--save_dir', type=str, default='./results/nhp', help='Main saving directory for results')  
    parser.add_argument('--data_dir', type=str, default='./data', help='Main directory where data is saved')
    parser.add_argument('--load_config', required=False, default=True, help='Whether to load config provided in ./configs. True by default.') 

    # K-Fold Settings
    parser.add_argument('--num_folds', type=int, default=5, help='Number of folds for k-fold CV') 
    parser.add_argument('--which_folds', nargs='+', type=int, default=[1,2,2,4,5], help='Which folds to run the model')
    parser.add_argument('--z_score_data', required=False, default=True, action='store_true', help='If True, z-scoring will be applied to CONTINUOUS modality. True by default')

    # Model related settings
    parser.add_argument('--device', type=str, default='cuda', help='Device to run the model on')
    parser.add_argument('--seed', type=int, default=1, help='Seed for reproducibility')
    parser.add_argument('--likelihood_s', type=str, default='poisson', help='Likelihood of s (spike in this case)')
    parser.add_argument('--likelihood_y', type=str, default='gaussian', help='Likelihood of y (LFP in this case)')
    parser.add_argument('--layer_list_s',  nargs='+', type=int, default=[128,128,128], help='Modality-specific encoder hidden layers for s')
    parser.add_argument('--layer_list_y',  nargs='+', type=int, default=[128,128,128], help='Modality-specific encoder hidden layers for y')
    parser.add_argument('--layer_list_m',  nargs='+', type=int, default=[128], help='Fusion network hidden layers')

    parser.add_argument('--activation', type=str, default='tanh', help='Activation function used in MLP hidden layers')
    parser.add_argument('--n_a', type=int, default=64, help='Dimension of multiscale embedding factors')
    parser.add_argument('--n_x', type=int, default=64, help='Dimension of multiscale latent factors')

    parser.add_argument('--td_rate', type=float, default=0.3, help='Time dropout rate')
    parser.add_argument('--dropout_rate', type=float, default=0.1, help='Dropout rate')
    parser.add_argument('--kernel_initializer', type=str, default='xavier_normal', help='Kernel initializer function for encoder/decoder parameters') 

    # Loss related settings
    parser.add_argument('--tau', type=float, default=3, help='Scaling hyperparameter for scale difference of different likelihoods')
    parser.add_argument('--autoset_tau', required=False, default=True, action='store_true', help='If True, tau will be computed automatically as described in the manuscript. True by default.')
    parser.add_argument('--scale_l2', type=float, default= 1e-3, help='L2 regularization MLP weights of MRINE')
    parser.add_argument('--steps_ahead', nargs='+', type=int, default=[0,1,2,3,4], help='Future steps list for which k-step-ahead loss is optimized. 0 means smoothing.')    
    parser.add_argument('--scale_sm_reg_s', type=float, default=250, help='Scale of smoothness regularization on smoothed firing rates')    
    parser.add_argument('--scale_sm_reg_y', type=float, default=5, help='Scale of smoothness regularization on smoothed mean of gaussian modality') 
    parser.add_argument('--scale_sm_reg_x', type=int, default=30, help='Scale of smoothness regularization on multiscale dynamic factors')   

    # Training related settings
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for MRINE')
    parser.add_argument('--num_epochs', type=int, default=500, help='Number of epochs for which MRINE is trained')

    # Load related settings
    parser.add_argument('--resume_train', required=False, default=False, action='store_true', help='If True, training will resume starting from the provided checkpoint, otherwise, loaded model from ckpt will be trained for given num_epochs')
    parser.add_argument('--file_name', type=str, default='', help='Which checkpoint to load, provide checkpoint filename without including file extension (.pth)')
    parser.add_argument('--train_mrine', required=False, default=True, action='store_true', help='If True, MRINE model will be trained, otherwise, only decoding and encoding result saving will be performed after loading the last ckpt. True by default.')
 
    # Learning rate/scheduler related settings
    parser.add_argument('--init_lr', type=float, default=0.01, help='Initial learning rate')
    parser.add_argument('--base_lr', type=float, default=0.001, help='Base LR for Cyclic LR Scheduler')
    parser.add_argument('--max_lr', type=float, default=0.01, help='Max LR for Cyclic LR Scheduler')
    parser.add_argument('--gamma', type=float, default=0.99, help='Exponential envelope exponent for Cyclic LR Scheduler')
    parser.add_argument('--step_size_up', type=int, default=10, help='Number of steps to reach max LR for Cyclic LR Scheduler')
    parser.add_argument('--grad_clip', type=float, default=0.1, help='Gradient clipping norm') 

    # Other settings
    parser.add_argument('--sample_drop_per_s', type=float, default=0, help='Sample dropping probability for s')
    parser.add_argument('--sample_drop_per_y', type=float, default=0, help='Sample dropping probability for y')

    # Parse the arguments 
    args = parser.parse_args()

    # Run the main method
    main(args)
