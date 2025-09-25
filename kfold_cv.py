import os
import torch
from torch.utils.data import DataLoader

from python_utils import convert_to_tensor
from time_series_utils import z_score_train_valid_tensor, create_kfold_inds, get_tau
from trainer.TrainerMRINE import TrainerMRINE
from torch_datasets import MRINEDataset


def run_kfold_cv(train_mrine, config, kfold_settings, 
                 sample_drop_per_s=None, sample_drop_per_y=None, 
                 s=None, y=None, m_s=None, m_y=None, 
                 m_s_zs=None, m_y_zs=None, target=None, 
                 do_decode_target=False, decode_target_ds_rate=False,
                 which_latents=['x_smooth'], compute_cc_flat=True):
    '''z
    Runs k-fold cross validation (CV) on the data and trains the models, performs inference

    train_mrine: bool, Boolean to determine whether to train models or just load the model from the last checkpoint (if available)
    config: CfgNode, yacs config denoting the hyperparameters used for training
    kfold_settings: dict, dictionary to set k-fold CV settings. It should have keys 'which_folds', 'autoset_tau', 'num_folds', 'z_score_data'. 
    sample_drop_per_s: float, Sample dropping probability of modality s during inference. 0 by default.
    sample_drop_per_y: float, Sample dropping probability of modality y during inference. 0 by default.
    s: torch.Tensor, spiking activity, or discrete observations. (num_seq, num_steps, n_s)
    y: torch.Tensor, LFP, or continuous signals. (num_seq, num_steps, n_y)
    m_s: torch.Tensor, mask tensor for s denoting whether s is available for a time-step or not. (num_seq, num_steps)
    m_y: torch.Tensor, mask tensor for y denoting whether y is available for a time-step or not. (num_seq, num_steps)
    m_s_zs: torch.Tensor, mask tensor to compute z-score statistics for modality s. Ignored for Poisson likelihoods.
            It is used when sample dropping is studied, z-scoring statistics are computed over original masks to apply same statistics as data used for model training. (num_seq, num_steps)
    m_s_zs: torch.Tensor, mask tensor to compute z-score statistics for modality s. Ignored for Poisson likelihoods.
            It is used when sample dropping is studied, z-scoring statistics are computed over original masks to apply same statistics as data used for model training. (num_seq, num_steps)
    target: torch.Tensor,  target variable to decode from inferred latent factors. (num_seq, num_steps)
    do_decode_target: bool, target decoding will be performed if True.
    decode_target_ds_rate: bool, if True, target will be decoded after downsampling both latent factors and target. Used when comparing MRINE trained with different timescale s and y, and single-scale 
                                network trained on slower timescale y.
    which_latents: list, list of strings that denote which latent factors to perform target decoding for. Options are 'x_filter', 'x_smooth'. 'a_smooth' and 'a_pred' is same as their 
                        'x_filter', 'x_smooth' as their relationship is linear. 
    compute_cc_flat: bool, if True, Pearson correlation coeffient (CC) will be computed over continuous time-series instead of computing across trials and averaging.
    '''
    
    # If z-scoring masks are None, use original masks
    if m_s_zs is None:
        m_s_zs = m_s
    if m_y_zs is None:
        m_y_zs = m_y
    
    # Main save directory where each fold results will be saved under .../{main_save_dir}/fold_{fold}
    main_save_dir = config.model.save_dir

    config.model.n_s = s.shape[-1] if s is not None else -1
    config.model.n_y = y.shape[-1] if y is not None else -1

    if sample_drop_per_s is None:
        sample_drop_per_s = 0
    if sample_drop_per_y is None:
        sample_drop_per_y = 0

    for fold in kfold_settings['which_folds']:
        # Set the fold save directory: e.g. for fold 1, {main_save_dir}/fold_1/{drop_str}
        fold_str = f'fold_{fold}'
        drop_str = '' # To save decoding results of sample dropping scenarios in distinct folders

        if config.model.model_type == 'multi':
            if sample_drop_per_s + sample_drop_per_y != 0:
                drop_str += 'dropPers'
                drop_str += f'_{sample_drop_per_s:.1e}_{sample_drop_per_y:.1e}'
        elif config.model.model_type == 'single-poisson':
            if sample_drop_per_s != 0:
                drop_str += 'dropPers'
                drop_str += f'_{sample_drop_per_s:.1e}'
        elif config.model.model_type == 'single-gaussian':
            if sample_drop_per_y != 0:
                drop_str += 'dropPers'
                drop_str += f'_{sample_drop_per_y:.1e}'

        config.load.ckpt_load_dir = f'{main_save_dir}/{fold_str}' # if file_name is not empty ('')
        config.model.save_dir = f'{main_save_dir}/{fold_str}/{drop_str}'; os.makedirs(config.model.save_dir, exist_ok=True)

        # Get the k-fold data
        data_fold = create_kfold_data(s=s, y=y,
                                      m_s=m_s, m_y=m_y, 
                                      m_s_zs=m_s_zs, m_y_zs=m_y_zs,
                                      target=target, 
                                      fold=fold,
                                      kfold_settings=kfold_settings,
                                      likelihood_s=config.model.likelihood_s,
                                      likelihood_y=config.model.likelihood_y)

        # Save data_fold so that we can access the data and data settings we have passed thru model
        torch.save(data_fold, f'{config.model.save_dir}/data_and_settings.pt')
        print(f'save directory: {config.model.save_dir}')

        # To automatically set the log likelihoods scales across modalities by mean
        if kfold_settings['autoset_tau']:
            config.loss.tau = get_tau(s=data_fold['train']['s'], y=data_fold['train']['y'], 
                                      m_s=data_fold['train']['m_s_zs'], m_y=data_fold['train']['m_y_zs'], 
                                      likelihood_s=config.model.likelihood_s, likelihood_y=config.model.likelihood_y)

        # Create the dataset and dataloaders
        train_dataset = MRINEDataset(s=data_fold['train']['s'],
                                     y=data_fold['train']['y'],
                                     m_s=data_fold['train']['m_s'],
                                     m_y=data_fold['train']['m_y'],
                                     target=data_fold['train']['target'])
        valid_dataset = MRINEDataset(s=data_fold['valid']['s'],
                                     y=data_fold['valid']['y'],
                                     m_s=data_fold['valid']['m_s'],
                                     m_y=data_fold['valid']['m_y'],
                                     target=data_fold['valid']['target'])

        train_loader = DataLoader(train_dataset, batch_size=config.train.batch_size, shuffle=True)
        valid_loader = DataLoader(valid_dataset, batch_size=config.train.batch_size, shuffle=True)

        # Train the MRINE model
        if train_mrine:
            trainer = TrainerMRINE(config=config)
            trainer.train(train_loader=train_loader, valid_loader=valid_loader)
        else:
            print(f'Training is skipped, make sure that {config.train.num_epochs}_ckpt.pt exist!')

        # Load final ckpt
        config_mrine_inf = config.clone()
        config_mrine_inf.device = 'cpu'
        config_mrine_inf.load.file_name = str(config.train.num_epochs)
        config_mrine_inf.load.resume_train = False

        trainer = TrainerMRINE(config=config_mrine_inf, save_config=False)

        # To save results, recreate data loaders without shuffling  
        train_loader = DataLoader(train_dataset, batch_size=config_mrine_inf.train.batch_size, shuffle=False)
        valid_loader = DataLoader(valid_dataset, batch_size=config_mrine_inf.train.batch_size, shuffle=False)

        # Save inference results such as inferred latent factors, etc.
        trainer.save_encoding_results(train_loader=train_loader,
                                      valid_loader=valid_loader)

        # Fit LR decoders for target decoding and save decoding results
        if target is not None and do_decode_target:
            trainer.decode_target(train_loader=train_loader, valid_loader=valid_loader, z_score_inp=False, decode_target_ds_rate=1, which_latents=which_latents, compute_cc_flat=compute_cc_flat)
            if decode_target_ds_rate != 1 and config.model.model_type == 'multi': # so we do decoding at each modalities timescale if they are different
                trainer.decode_target(train_loader=train_loader, valid_loader=valid_loader, z_score_inp=False, decode_target_ds_rate=decode_target_ds_rate, which_latents=which_latents, compute_cc_flat=compute_cc_flat)


def create_kfold_data(fold, kfold_settings, s=None, y=None, 
                      m_s=None, m_y=None, m_s_zs=None,  m_y_zs=None,
                      target=None, likelihood_s='poisson', likelihood_y='gaussian'):
    '''
    Splits data into train for k-fold CV

    fold: int, fold number to split the data into training and validation sets
    kfold_settings: dict, dictionary to set k-fold CV settings. It should have keys 'which_folds', 'autoset_tau', 'num_folds', 'z_score_data'. 
    s: torch.Tensor, spiking activity, or discrete observations. (num_seq, num_steps, n_s)
    y: torch.Tensor, LFP, or continuous signals. (num_seq, num_steps, n_y)
    m_s: torch.Tensor, mask tensor for s denoting whether s is available for a time-step or not. (num_seq, num_steps)
    m_y: torch.Tensor, mask tensor for y denoting whether y is available for a time-step or not. (num_seq, num_steps)
    m_s_zs: torch.Tensor, mask tensor to compute z-score statistics for modality s. Ignored for Poisson likelihoods.
            It is used when sample dropping is studied, z-scoring statistics are computed over original masks to apply same statistics as data used for model training. (num_seq, num_steps)
    m_s_zs: torch.Tensor, mask tensor to compute z-score statistics for modality s. Ignored for Poisson likelihoods.
            It is used when sample dropping is studied, z-scoring statistics are computed over original masks to apply same statistics as data used for model training. (num_seq, num_steps)
    target: torch.Tensor,  target variable to decode from inferred latent factors. (num_seq, num_steps)
    likelihood_s: str, likelihood for modality s
    likelihood_y: str, likelihood for modality y
    
    '''
    kfold_settings['which_folds'] = fold

    num_folds = kfold_settings['num_folds']
    num_seq, num_steps, _ = s.shape if s is not None else y.shape
    
    # Get training and validation trial/segment indices
    index_train, index_valid = create_kfold_inds(num_seq=num_seq, num_folds=num_folds, fold=fold)
    data_fold = {}
    data_fold['settings'] = kfold_settings

    data_fold['train'] = {}
    data_fold['valid'] = {}
    
    data_fold['train']['index'] = index_train
    data_fold['valid']['index'] = index_valid

    for d, data, mask, mask_zs, likelihood in zip(['s', 'y'], [s, y], [m_s, m_y], [m_s_zs, m_y_zs], [likelihood_s, likelihood_y]):
        if data is not None:
            if mask_zs is None:
                mask_zs = mask

            # Split mask tensors into training and validation
            data_fold['train'][f'm_{d}'] = convert_to_tensor(mask[index_train, ...])
            data_fold['valid'][f'm_{d}'] = convert_to_tensor(mask[index_valid, ...])
            data_fold['train'][f'm_{d}_zs'] = convert_to_tensor(mask_zs[index_train, ...])
            data_fold['valid'][f'm_{d}_zs'] = convert_to_tensor(mask_zs[index_valid, ...])

            if likelihood == 'poisson':
                # We don't apply z-scoring for discrete modalities
                data_fold['train'][d] = convert_to_tensor(data[index_train, ...])
                data_fold['valid'][d] = convert_to_tensor(data[index_valid, ...])
            elif likelihood == 'gaussian':
                # Perform z-scoring if asked
                if kfold_settings['z_score_data']:
                    train_data_zs, valid_data_zs, mean_data, std_data = z_score_train_valid_tensor(x=data,
                                                                                                   mask=mask_zs,
                                                                                                   index_train=index_train, 
                                                                                                   index_valid=index_valid)
                    data_fold['train'][d] = train_data_zs
                    data_fold['valid'][d] = valid_data_zs
                    data_fold['train'][f'mean_{d}'] = mean_data
                    data_fold['valid'][f'std_{d}'] = std_data
                else:
                    data_fold['train'][d] = convert_to_tensor(data[index_train, ...])
                    data_fold['valid'][d] = convert_to_tensor(data[index_valid, ...])
            
            print(f"{d}, shape_train {data_fold['train'][d].shape} shape_valid {data_fold['valid'][d].shape}")
        else:
            data_fold['train'][d] = None
            data_fold['valid'][d] = None
            data_fold['train'][f'm_{d}'] = None
            data_fold['valid'][f'm_{d}'] = None
            data_fold['train'][f'm_{d}_zs'] = None
            data_fold['valid'][f'm_{d}_zs'] = None

    if target is not None:
        # Split target variables into training and validation sets
        data_fold['train']['target'] = convert_to_tensor(target[index_train, ...])
        data_fold['valid']['target'] = convert_to_tensor(target[index_valid, ...])
        print(f"target, shape_train {data_fold['train']['target'].shape} shape_valid {data_fold['valid']['target'].shape}")
    else:
        # Set to None if no target is presented. Dataset object creates ones tensor if this is the case
        data_fold['train']['target'] = None
        data_fold['valid']['target'] = None
    return data_fold
