import os
import torch
torch.set_printoptions(precision=3)
import numpy as np
np.set_printoptions(precision=3)

from tqdm import tqdm
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')
from torch.nn.utils import clip_grad_norm_

from sklearn.linear_model import LinearRegression

from trainer.BaseTrainer import BaseTrainer
from model.MRINE import MRINE 
from python_utils import carry_to_device
from time_series_utils import get_pearson_cc, z_score_tensor
from metrics import Mean


class TrainerMRINE(BaseTrainer):
    '''
    Trainer class for MRINE. Overwrites BaseTrainer class.
    '''
    def __init__(self, config, save_config=True):
        '''
        Initializes TrainerMRINE object

        config: CfgNode, config denoting the hyperparameters of MRINE
        save_config: bool, if True, config will be save in the save directory denoted in the config
        '''
        super(TrainerMRINE, self).__init__(config)

        # Initialize logger 
        self.logger = self._get_logger(prefix='mrine')

        # Set device
        self.device = 'cpu' if self.config.device.lower() == 'cpu' or not torch.cuda.is_available() else self.config.device
        self.config.device = self.device 

        # Initialize the model, optimizer and learning rate scheduler
        self.mrine = MRINE(self.config); self.mrine.to(self.device)
        self.optimizer = self._get_optimizer(params=self.mrine.parameters())
        self.lr_scheduler = self._get_lr_scheduler()

        # Load ckpt if asked
        if self.config.load.file_name != '':
            if 'ckpt_load_dir' in self.config.load:
                ckpt_load_dir =  self.config.load.ckpt_load_dir
            else:
                ckpt_load_dir = None

            self.mrine, self.optimizer, self.lr_scheduler = self._load_ckpt(model=self.mrine,
                                                                             optimizer=self.optimizer,
                                                                             lr_scheduler=self.lr_scheduler, 
                                                                             load_dir=ckpt_load_dir,
                                                                             file_name=self.config.load.file_name)
        
        # Get the metrics 
        self.metric_names, self.metrics = self._get_metrics()

        # Save the config
        if save_config:
            self._save_config()


    def _get_metrics(self):
        '''
        Creates metrics to log during training and to tensorboard.
        '''
        metric_names = []
        for k in self.config.loss.steps_ahead:
            if k > 0:
                metric_names.append(f'log_ll_pred_{k}_s')
                metric_names.append(f'log_ll_pred_{k}_y')
            else:
                metric_names.append('log_ll_smoothing_s')
                metric_names.append('log_ll_smoothing_y')

        metric_names.append('sm_reg_loss_s')
        metric_names.append('sm_reg_loss_y')
        metric_names.append('sm_reg_loss_x')

        metric_names.append('model_loss')
        metric_names.append('l2_reg')
        metric_names.append('total_loss')
        
        metrics = {}
        metrics['train'] = {}
        metrics['valid'] = {}

        for key in metric_names:
            metrics['train'][key] = Mean()
            metrics['valid'][key] = Mean()
        
        return metric_names, metrics
    

    def _get_log_str(self, epoch, train_valid='train'):
        '''
        Creates logging string printed on the console

        epoch: int, epoch number for which the metrics will be logged
        train_valid: str, Training or validation string for which the metrics will be logged
        '''
        log_str = f'Epoch {epoch}, {train_valid.upper()} '

        if self.config.model.model_type in ['multi', 'single-poisson']:
            log_str += '\n log ll s|' 
            for k in self.config.loss.steps_ahead:
                if k > 0:
                    log_str += f" {k}-step: {self.metrics[train_valid][f'log_ll_pred_{k}_s'].compute():.3f},"
                elif k == 0:
                    log_str += f" smoothing: {self.metrics[train_valid][f'log_ll_smoothing_s'].compute():.3f},"

        if self.config.model.model_type in ['multi', 'single-gaussian']:
            log_str += '\n log ll y|' 
            for k in self.config.loss.steps_ahead:
                if k > 0:
                    log_str += f" {k}-step: {self.metrics[train_valid][f'log_ll_pred_{k}_y'].compute():.3f},"
                elif k == 0:
                    log_str += f" smoothing: {self.metrics[train_valid][f'log_ll_smoothing_y'].compute():.3f},"

        log_str += '\nl2_reg: {:.3f}, scale_l2: {:.3f}, '.format(self.metrics[train_valid]['l2_reg'].compute(), self.config.loss.scale_l2)
        
        log_str += '\nsmoothness regularization losses|'
        log_str += '\n\tOn smoothed decoder outputs|'

        if self.config.model.model_type in ['multi', 'single-gaussian']:
            log_str += f" \n\t\ty: {self.metrics[train_valid]['sm_reg_loss_y'].compute():.5f}, scale_sm_reg_y: {self.config.loss.scale_sm_reg_y}"
        if self.config.model.model_type in ['multi', 'single-poisson']:
            log_str += f" \n\t\ts: {self.metrics[train_valid]['sm_reg_loss_s'].compute():.5f} scale_sm_reg_s: {self.config.loss.scale_sm_reg_s}"

        log_str += f"\n\tOn smoothed x: {self.metrics[train_valid]['sm_reg_loss_x'].compute():.5f}, scale_sm_reg_x: {self.config.loss.scale_sm_reg_x}"

        log_str += '\nmodel_loss: {:.3f}, total_loss: {:.3f}'.format(self.metrics[train_valid]['model_loss'].compute(), self.metrics[train_valid]['total_loss'].compute())
        return log_str


    def train_epoch(self, epoch, train_loader):
        '''
        Performs one epoch forward and backward passes (training) on the data 

        epoch: int, epoch number for which the training will be performed
        train_loader: torch.utils.data.DataLoader, training dataloader object
        '''

        # Put the model in training mode
        self.mrine.train()

        # Reset the metrics in the beginning of each epoch
        self._reset_metrics(train_valid='train')

        # Start iterating over batches
        step = (epoch - 1) * len(train_loader) + 1
        with tqdm(train_loader, unit='batch') as tepoch:
            for b, batch in enumerate(tepoch):
                tepoch.set_description(f"Epoch {epoch}, TRAIN")

                # Carry data to device
                s, y, m_s, m_y, _ = carry_to_device(data=batch, device=self.device)

                # Perform forward pass and compute loss
                loss, loss_dict, model_vars = self.mrine.step(s=s, y=y,
                                                              m_s=m_s, m_y=m_y)

                # Create plots of the variables obtained with the initialized model
                if epoch == self.start_epoch and b == 0:
                    self.create_plots(s, y, m_s, m_y, model_vars, 'init', prefix='train')

                # Update model parameters
                self.optimizer.zero_grad()
                loss.backward()

                # Log gradient norms to tensorboard before and after gradient clipping
                self.write_model_gradients(self.mrine, step=step, prefix='unclipped')
                clip_grad_norm_(self.mrine.parameters(), self.config.optim.grad_clip)
                self.write_model_gradients(self.mrine, step=step, prefix='clipped')

                # Update parameters
                self.optimizer.step()

                # Notification if any parameter has invalid (nan) gradient. check parameters and restart training 
                for n, p in self.mrine.named_parameters():
                    if p.grad is not None and p.grad.isnan().any():
                        print(n)

                # Update metrics
                cur_batch_size = s.shape[0] if s is not None else y.shape[0]
                self._update_metrics(loss_dict=loss_dict, batch_size=cur_batch_size, 
                                     train_valid='train', verbose=False)

                # Update the step 
                step += 1
        
        # Save model and optimizer
        if epoch % self.config.model.save_steps == 0 or epoch == 1 or epoch == self.config.train.num_epochs:
            self._save_ckpt(epoch=epoch, model=self.mrine, optimizer=self.optimizer, lr_scheduler=self.lr_scheduler)

        # Create and save plots from the last batch
        if epoch % self.config.train.plot_save_steps == 0 or epoch == 1 or epoch == self.config.train.num_epochs:
            self.create_plots(s, y, m_s, m_y, model_vars, epoch, prefix='train')

        # Logging the training step information for last batch
        log_str = self._get_log_str(epoch=epoch, train_valid='train')
        self.logger.info(log_str)

        # Update LR 
        self.lr_scheduler.step()


    def valid_epoch(self, epoch, valid_loader):
        '''
        Performs one epoch evaluation on the data 

        epoch: int, epoch number for which the validation will be performed
        valid_loader: torch.utils.data.DataLoader, validation dataloader object
        '''

        # Put model to evaluation mode
        self.mrine.eval()

        # Disable gradient computation 
        with torch.no_grad():
            # Reset metrics in the beginning of each epoch
            self._reset_metrics(train_valid='valid')

            with tqdm(valid_loader, unit='batch') as tepoch:
                for _, batch in enumerate(tepoch):
                    tepoch.set_description(f"Epoch {epoch}, VALID")

                    # Carry data to device
                    s, y, m_s, m_y, _ = carry_to_device(data=batch, device=self.device)

                    # Perform forward pass and compute loss
                    _, loss_dict, model_vars = self.mrine.step(s=s, y=y,
                                                            m_s=m_s, m_y=m_y)

                    # Update metrics 
                    cur_batch_size = s.shape[0] if s is not None else y.shape[0]
                    self._update_metrics(loss_dict=loss_dict, batch_size=cur_batch_size, 
                                        train_valid='valid', verbose=False)

            # Create and save plots from last batch
            if epoch % self.config.train.plot_save_steps == 0 or epoch == 1 or epoch == self.config.train.num_epochs:
                self.create_plots(s, y, m_s, m_y, model_vars, epoch, prefix='valid')

            # Logging the training step information for last batch
            log_str = self._get_log_str(epoch=epoch, train_valid='valid')
            self.logger.info(log_str)


    def train(self, train_loader, valid_loader=None):
        '''
        Performs training for number of epochs denoted in config
        
        train_loader: torch.utils.data.DataLoader, training dataloader object
        valid_loader: torch.utils.data.DataLoader, validation dataloader object
        '''

        for epoch in range(self.start_epoch, self.config.train.num_epochs + 1):
            # Training
            self.train_epoch(epoch, train_loader)

            # Validation
            if (epoch % self.config.train.valid_step == 0 and isinstance(valid_loader, torch.utils.data.dataloader.DataLoader)) or epoch == self.start_epoch:
                self.valid_epoch(epoch, valid_loader)

            # write tensorboard summaries    
            self.write_summary(epoch, prefix='train')
            self.write_summary(epoch, prefix='valid')


    def save_encoding_results(self, train_loader, valid_loader=None, save_enc_results=True):
        '''
        Saves encoding (inference) results in the save directory denoted in the config

        train_loader: torch.utils.data.DataLoader, training dataloader object for which the inference is performed
        valid_loader: torch.utils.data.DataLoader, validation dataloader object for which the inference is performed
        save_enc_results: bool, whether to save the results
        '''
        # Put model to evaluation mode
        self.mrine.eval()

        # Disable gradient computation
        with torch.no_grad():

            # Create empty dicts for logging
            encoding_dict = {}
            encoding_dict['x_pred'] = dict(train=[], valid=[])
            encoding_dict['x_filter'] = dict(train=[], valid=[])
            encoding_dict['x_smooth'] = dict(train=[], valid=[])

            encoding_dict['a_pred'] = dict(train=[], valid=[])
            encoding_dict['a_filter'] = dict(train=[], valid=[])
            encoding_dict['a_smooth'] = dict(train=[], valid=[])
        
            for key in self.mrine.decoder_dict.keys():
                d, likelihood = key.split('_')
                if likelihood.lower() == 'poisson':
                    encoding_dict[f'{d}_fr_pred'] = dict(train=[], valid=[])
                    encoding_dict[f'{d}_fr_filter'] = dict(train=[], valid=[])
                    encoding_dict[f'{d}_fr_smooth'] = dict(train=[], valid=[])
                elif likelihood.lower() == 'gaussian':
                    encoding_dict[f'{d}_mu_pred'] = dict(train=[], valid=[])
                    encoding_dict[f'{d}_mu_filter'] = dict(train=[], valid=[])
                    encoding_dict[f'{d}_mu_smooth'] = dict(train=[], valid=[])
                encoding_dict[d] = dict(train=[], valid=[])
                encoding_dict[f'm_{d}'] = dict(train=[], valid=[])

            # Iterate over training and validation dataloaders
            for train_valid, loader in zip(['train', 'valid'], [train_loader, valid_loader]):
                if isinstance(loader, torch.utils.data.dataloader.DataLoader):
                    # Iterate over batches
                    for _, batch in enumerate(loader):
                        s, y, m_s, m_y, _ = carry_to_device(data=batch, device=self.device)
                        model_vars = self.mrine(s=s, y=y,
                                                m_s=m_s, m_y=m_y)
                        
                        # Append inference results
                        encoding_dict['x_pred'][train_valid].append(model_vars['x_pred'])
                        encoding_dict['x_filter'][train_valid].append(model_vars['x_filter'])
                        encoding_dict['x_smooth'][train_valid].append(model_vars['x_smooth'])
    
                        encoding_dict['a_pred'][train_valid].append(model_vars['a_pred'])
                        encoding_dict['a_filter'][train_valid].append(model_vars['a_filter'])
                        encoding_dict['a_smooth'][train_valid].append(model_vars['a_smooth'])

                        # Append reconstructions/predictions depending on whether the model type is multi, single-poisson or single-gaussian
                        for key in self.mrine.decoder_dict.keys():
                            d, likelihood = key.split('_')
                            if likelihood.lower() == 'poisson':
                                encoding_dict[f'{d}_fr_pred'][train_valid].append(model_vars['data_dict_pred'][d]['fr_pred'])
                                encoding_dict[f'{d}_fr_filter'][train_valid].append(model_vars['data_dict_filter'][d]['fr_filter'])
                                encoding_dict[f'{d}_fr_smooth'][train_valid].append(model_vars['data_dict_smooth'][d]['fr_smooth'])
                            elif likelihood.lower() == 'gaussian':
                                encoding_dict[f'{d}_mu_pred'][train_valid].append(model_vars['data_dict_pred'][d]['mu_pred'])
                                encoding_dict[f'{d}_mu_filter'][train_valid].append(model_vars['data_dict_filter'][d]['mu_filter'])
                                encoding_dict[f'{d}_mu_smooth'][train_valid].append(model_vars['data_dict_smooth'][d]['mu_smooth'])
                            
                            if d == 's':
                                encoding_dict['s'][train_valid].append(s)
                                encoding_dict['m_s'][train_valid].append(m_s)
                            elif d =='y':
                                encoding_dict['y'][train_valid].append(y)
                                encoding_dict['m_y'][train_valid].append(m_y)

                    # Convert lists to tensors
                    for d_key in encoding_dict.keys():
                        encoding_dict[d_key][train_valid] = torch.cat(encoding_dict[d_key][train_valid], dim=0)
            
            # Save results
            if save_enc_results:
                enc_path = f'{self.config.model.save_dir}/encoding_results.pt'
                torch.save(encoding_dict,  enc_path)
            

    def decode_target(self, train_loader, valid_loader=None, z_score_inp=True, save_dec_results=True, 
                      decode_target_ds_rate=1, which_latents=['x_smooth'], compute_cc_flat=True):
        '''
        Performs target variable decoding from inferred latent factors

        train_loader: torch.utils.data.DataLoader, training dataloader object for which the inference is performed
        valid_loader: torch.utils.data.DataLoader, validation dataloader object for which the inference is performed
        z_score_inp: bool, if True, inferred latent factors will be z-scored before decoding
        save_dec_results: bool, whether to save the results
        decode_target_ds_rate: bool, if True, target will be decoded after downsampling both latent factors and target. Used when comparing MRINE trained with different timescale s and y, and single-scale 
                                    network trained on slower timescale y.
        which_latents: list, list of strings that denote which latent factors to perform target decoding for. Options are 'x_filter', 'x_smooth'. 'a_smooth' and 'a_pred' is same as their 
                            'x_filter', 'x_smooth' as their relationship is linear. 
        compute_cc_flat: bool, if True, Pearson correlation coeffient (CC) will be computed over continuous time-series instead of computing across trials and averaging.
        '''

        # Put the model in evaluation mode
        self.mrine.eval()

        # Linear regression is same for x and a as their relation is linear
        for i, feat in enumerate(which_latents):
            if feat.startswith('a_'):
                which_latents[i] = 'x_' +  feat[2:]

        # Get variables for which decoding will be performed
        x_dict = {}
        for feat in which_latents:
            x_dict[feat] = dict(train=[], valid=[])

        for key in self.mrine.decoder_dict.keys():
            d, _ = key.split('_')
            x_dict[f"{d}_smooth"] = dict(train=[], valid=[])
            x_dict[f"{d}_filter"] = dict(train=[], valid=[])

        target_dict = dict(train=[], valid=[])

        # Iterate over dataloaders
        loaders = dict(train=train_loader, valid=valid_loader)
        for train_valid, loader in loaders.items():
            if isinstance(loader, torch.utils.data.dataloader.DataLoader):
                # Iterate over batches 
                for _, batch in enumerate(loader):
                    s, y, m_s, m_y, target = carry_to_device(data=batch, device=self.device)
                    model_vars = self.mrine(s=s, y=y,
                                            m_s=m_s, m_y=m_y)
                    
                    # Append target variables and inferred latent factors
                    target_dict[train_valid].append(target.detach())
                    _, num_steps,  n_z = target.shape
                    for feat in which_latents:
                        x_dict[feat][train_valid].append(model_vars[feat])
                    
                    # Perform decoding also from reconstructed outputs
                    for key in self.decoder_dict.keys():
                        d, likelihood = key.split('_')

                        if likelihood == 'poisson':
                            x_dict[f"{d}_smooth"][train_valid].append(model_vars['data_dict_smooth'][d]['fr_smooth'])
                            x_dict[f"{d}_filter"][train_valid].append(model_vars['data_dict_filter'][d]['fr_filter'])
                        else:
                            x_dict[f"{d}_smooth"][train_valid].append(model_vars['data_dict_smooth'][d]['mu_smooth'])
                            x_dict[f"{d}_filter"][train_valid].append(model_vars['data_dict_filter'][d]['mu_filter'])
                
                # Convert lists to tensors
                target_dict[train_valid] = torch.cat(target_dict[train_valid], dim=0).reshape(-1, n_z)
                for feat in x_dict.keys():
                    n_f =  x_dict[feat][train_valid][0].shape[-1]
                    x_dict[feat][train_valid] = torch.cat(x_dict[feat][train_valid], dim=0).reshape(-1, n_f)
        
        # If downsampling rate is not equal to 1, downsample both target variables and latent factors
        if decode_target_ds_rate != 1:
            target_dict['train'] = target_dict['train'][::decode_target_ds_rate, :]
            target_dict['valid'] = target_dict['valid'][::decode_target_ds_rate, :]
            extra_save_str = f'_ds_{decode_target_ds_rate}'
        else:
            extra_save_str = ''

        # Start decoding
        num_steps = int(num_steps / decode_target_ds_rate)
        decoding_dict = {}

        for key, feat in x_dict.items():
            # Set dimensions and save dir of config
            n_f = feat['train'].shape[-1]

            if decode_target_ds_rate != 1:
                feat_train = feat['train'][::decode_target_ds_rate, :]
                feat_valid = feat['valid'][::decode_target_ds_rate, :]
            else:
                feat_train = feat['train']
                feat_valid = feat['valid']

            # Z-score input if asked
            if z_score_inp:
                feat_train, feat_mean, feat_std = z_score_tensor(feat_train, fit=True)
                if isinstance(valid_loader, torch.utils.data.dataloader.DataLoader):
                    feat_valid, _, _ = z_score_tensor(feat_valid, fit=False, mean=feat_mean, std=feat_std)
            else:
                feat_mean = torch.zeros(feat_train.shape[-1], dtype=torch.float32)
                feat_std = torch.ones(feat_train.shape[-1], dtype=torch.float32)
            
            # Fit linear regression from latent factors to target variables
            decoding_dict[f'from_{key}'] = {}
            decoding_dict[f'from_{key}']['model'] = LinearRegression()
            decoding_dict[f'from_{key}']['model'].fit(feat_train.detach().cpu(), target_dict['train'].detach().cpu())

            decoding_dict[f'from_{key}']['train'] = {}
            decoding_dict[f'from_{key}']['train']['y'] = target_dict['train'].reshape(-1, num_steps, n_z).detach().cpu()
            decoding_dict[f'from_{key}']['train']['yhat'] = torch.tensor(decoding_dict[f'from_{key}']['model'].predict(feat_train.detach().cpu()).reshape(-1, num_steps, n_z), dtype=torch.float32)

            # Compute CC between true and reconstructed target variables
            if compute_cc_flat:
                cc_vec_train, cc_mean_train = get_pearson_cc(x=decoding_dict[f'from_{key}']['train']['y'], xhat=decoding_dict[f'from_{key}']['train']['yhat'])
            else:
                cc_vec_train = []
                cc_mean_train = []
                x = decoding_dict[f'from_{key}']['train']['y']
                xhat = decoding_dict[f'from_{key}']['train']['yhat']
                for tr in range(x.shape[0]):
                    cc_vec_, cc_mean_ = get_pearson_cc(x=x[tr, :, :], xhat=xhat[tr, :, :])
                    cc_vec_train.append(cc_vec_.reshape(1, -1))
                    cc_mean_train.append(cc_mean_.reshape(1))
                cc_vec_train = torch.cat(cc_vec_train, dim=0).mean(dim=0)
                cc_mean_train = torch.cat(cc_mean_train, dim=0).mean(dim=0)

            decoding_dict[f'from_{key}']['train']['cc'] = {}
            decoding_dict[f'from_{key}']['train']['cc']['mean'] = cc_mean_train
            decoding_dict[f'from_{key}']['train']['cc']['vec'] = cc_vec_train

            # Logging
            self.logger.info(f'TRAIN, LR Decoding, from {key}, n_a {self.config.model.n_a}, n_x {self.config.model.n_x}, Decoding Correlation, vec: {cc_vec_train.numpy()},  mean: {cc_mean_train:.3f}')

            # Validate the learned LR model on validation set
            if isinstance(valid_loader, torch.utils.data.dataloader.DataLoader):
                decoding_dict[f'from_{key}']['valid'] = {}
                decoding_dict[f'from_{key}']['valid']['y'] = target_dict['valid'].reshape(-1, num_steps, n_z).detach().cpu()
                decoding_dict[f'from_{key}']['valid']['yhat'] = torch.tensor(decoding_dict[f'from_{key}']['model'].predict(feat_valid.detach().cpu()).reshape(-1, num_steps, n_z), dtype=torch.float32)

                # Compute CC between true and reconstructed target variables
                if compute_cc_flat:
                    cc_vec_valid, cc_mean_valid = get_pearson_cc(x=decoding_dict[f'from_{key}']['valid']['y'], xhat=decoding_dict[f'from_{key}']['valid']['yhat'])
                else:
                    cc_vec_valid = []
                    cc_mean_valid = []
                    x = decoding_dict[f'from_{key}']['valid']['y']
                    xhat = decoding_dict[f'from_{key}']['valid']['yhat']
                    for tr in range(x.shape[0]):
                        cc_vec_, cc_mean_ = get_pearson_cc(x=x[tr, :, :], xhat=xhat[tr, :, :])
                        cc_vec_valid.append(cc_vec_.reshape(1, -1))
                        cc_mean_valid.append(cc_mean_.reshape(1))
                    cc_vec_valid = torch.cat(cc_vec_valid, dim=0).mean(dim=0)
                    cc_mean_valid = torch.cat(cc_mean_valid, dim=0).mean(dim=0)

                decoding_dict[f'from_{key}']['valid']['cc'] = {}
                decoding_dict[f'from_{key}']['valid']['cc']['mean'] = cc_mean_valid
                decoding_dict[f'from_{key}']['valid']['cc']['vec'] = cc_vec_valid

                # Logging
                self.logger.info(f'VALID, LR Decoding, from {key}, n_a {self.config.model.n_a}, n_x {self.config.model.n_x}, Decoding Correlation, vec: {cc_vec_valid.numpy()}, mean: {cc_mean_valid:.3f}\n')
                    
        # Save decoding results
        if save_dec_results:
            dec_path = f'{self.config.model.save_dir}/decoding_results{extra_save_str}.pt'
            os.makedirs(os.path.dirname(dec_path), exist_ok=True)
            torch.save(decoding_dict, dec_path)


    def create_plots(self, s, y, m_s, m_y, model_vars, epoch, seq_id=0, prefix='train'):
        '''
        Creates and saves plots of inferred latent factors and reconstructions

        s: torch.Tensor, spiking activity, or discrete observations. (num_seq, num_steps, n_s)
        y: torch.Tensor, LFP, or continuous signals. (num_seq, num_steps, n_y)
        m_s: torch.Tensor, mask tensor for s denoting whether s is available for a time-step or not. (num_seq, num_steps)
        m_y: torch.Tensor, mask tensor for y denoting whether y is available for a time-step or not. (num_seq, num_steps)
        model_vars: dict, dictionary of inferred variables returned by MRINE.forward()
        epoch: int, epoch number for which the plots will be saved
        seq_id: int, sequence id in the batch to plot
        prefix: str, prefix for plot save names
        '''
        # Plot filtered and smoothed reconstructions
        self.create_y_plots(s, y, m_s, m_y, model_vars=model_vars, epoch=epoch, seq_id=seq_id, prefix=prefix, plot_smooth=False)
        self.create_y_plots(s, y, m_s, m_y, model_vars=model_vars, epoch=epoch, seq_id=seq_id, prefix=prefix, plot_smooth=True)

        # Plot filtered and smoothed embedding and latent factors 
        self.create_a_x_plots(d=model_vars['a_filter'], epoch=epoch, seq_id=seq_id, prefix=prefix, feat_name='a_filter')
        self.create_a_x_plots(d=model_vars['a_smooth'], epoch=epoch, seq_id=seq_id, prefix=prefix, feat_name='a_smooth')
        self.create_a_x_plots(d=model_vars['x_smooth'], epoch=epoch, seq_id=seq_id, prefix=prefix, feat_name='x_smooth')
        self.create_a_x_plots(d=model_vars['x_filter'], epoch=epoch, seq_id=seq_id, prefix=prefix, feat_name='x_filter')

        # Plot k-step ahead predicted distribution parameters
        self.create_k_step_ahead_plots(s=s, y=y, m_s=m_s, m_y=m_y, 
                                       likelihood_s=self.config.model.likelihood_s, likelihood_y=self.config.model.likelihood_y, 
                                       model_vars=model_vars, epoch=epoch, seq_id=seq_id, prefix=prefix)
        

    def create_y_plots(self, s, y, m_s, m_y, model_vars, epoch, seq_id=0, prefix='train', plot_smooth=False):
        '''
        Creates and saves plots of reconstructions

        s: torch.Tensor, spiking activity, or discrete observations. (num_seq, num_steps, n_s)
        y: torch.Tensor, LFP, or continuous signals. (num_seq, num_steps, n_y)
        m_s: torch.Tensor, mask tensor for s denoting whether s is available for a time-step or not. (num_seq, num_steps)
        m_y: torch.Tensor, mask tensor for y denoting whether y is available for a time-step or not. (num_seq, num_steps)
        model_vars: dict, dictionary of inferred variables returned by MRINE.forward()
        epoch: int, epoch number for which the plots will be saved
        seq_id: int, sequence id in the batch to plot
        prefix: str, prefix for plot save names
        '''

        # Disable gradient computation
        with torch.no_grad():
            for d, likelihood, data, mask in zip(['s', 'y'],
                                                [self.config.model.likelihood_s, self.config.model.likelihood_y],
                                                [s, y],
                                                [m_s, m_y]):
                fig = plt.figure(figsize=(20,15))

                x = data.detach().cpu()
                mask = mask.detach().cpu()

                # Plot for available timeteps
                num_seq, _, dim_x = x.shape 
                mask_bool = mask.type(torch.bool).unsqueeze(dim=-1).tile(1, 1, dim_x)
                x = x[mask_bool].reshape(num_seq, -1, dim_x)

                num_samples = np.shape(x)[1]
                color_index = range(num_samples)
                color_map = plt.cm.get_cmap('viridis')

                if likelihood.lower() == 'gaussian' and self.config.model.model_type in ['multi', 'single-gaussian']:
                    if plot_smooth:
                        x_hat = model_vars['data_dict_smooth'][d]['mu_smooth'].detach().cpu()
                    else:
                        x_hat = model_vars['data_dict_filter'][d]['mu_filter'].detach().cpu()
                    
                    x_hat = x_hat[mask_bool].reshape(num_seq, -1, dim_x)

                    # Plot the first 3 dimensions of true data in 3D plot
                    ax = fig.add_subplot(321, projection='3d')
                    ax_m = ax.scatter(x[seq_id, :, 0], x[seq_id, :, 1], x[seq_id, :, 2], c=color_index, vmin=0, vmax=num_samples, s=35, cmap=color_map, label='Noisy x')
                    ax.set_xlabel('Dim 1')
                    ax.set_ylabel('Dim 2')
                    ax.set_zlabel('Dim 3')
                    ax.set_title(f'True {d} embedding in 3d')
                    ax.legend()
                    fig.colorbar(ax_m)
                    
                    # Plot the first 3 dimensions of reconstruction in 3D plot
                    ax = fig.add_subplot(322, projection='3d')
                    ax_m = ax.scatter(x_hat[seq_id, :, 0], x_hat[seq_id, :, 1], x_hat[seq_id, :, 2], c=color_index, vmin=0, vmax=num_samples, s=35, cmap=color_map, label='Estimated x')
                    ax.set_title(f'Estimated {d} mu embedding in 3d')
                    ax.set_xlabel('Dim 1')
                    ax.set_ylabel('Dim 2')
                    ax.set_zlabel('Dim 3')
                    ax.legend()
                    fig.colorbar(ax_m)

                    # Plot dim 1
                    ax = fig.add_subplot(324)
                    ax.plot(np.arange(1, num_samples+1), x[seq_id, :, 0], 'g')
                    ax.plot(np.arange(1, num_samples+1), x_hat[seq_id, :, 0], 'b')
                    ax.set_title('Dim 1')

                    # Plot dim 2
                    ax = fig.add_subplot(325)
                    ax.plot(np.arange(1, num_samples+1), x[seq_id, :, 1], 'g')
                    ax.plot(np.arange(1, num_samples+1), x_hat[seq_id, :, 1] ,'b')
                    ax.set_title('Dim 2')

                    # Plot dim 3
                    ax = fig.add_subplot(326)
                    ax.plot(np.arange(1, num_samples+1), x[seq_id, :, 2], 'g', label='x_true')
                    ax.plot(np.arange(1, num_samples+1), x_hat[seq_id, :,2], 'b', label='x_mu')
                    ax.set_title('Dim 3')
                    ax.legend()

                elif likelihood.lower() == 'poisson' and self.config.model.model_type in ['multi', 'single-poisson']:
                    if plot_smooth:
                        x_hat = model_vars['data_dict_smooth'][d]['fr_smooth'].detach().cpu()
                    else:
                        x_hat = model_vars['data_dict_filter'][d]['fr_filter'].detach().cpu()
                    
                    x_hat = x_hat[mask_bool].reshape(num_seq, -1, dim_x)

                    # Plot the dimension with maximum spiking activity
                    dim_max_spiking = torch.argmax(torch.mean(x, dim=(0, 1)))
                    ax = fig.add_subplot(321)
                    ax.stem(x[seq_id, :, dim_max_spiking], markerfmt=' ', basefmt='C0', label='Spikes')
                    ax.plot(x_hat[seq_id, :, dim_max_spiking], label='Estimated FR')
                    ax.set_xlabel('Time')
                    ax.set_ylabel(f'Dim {dim_max_spiking+1}')
                    ax.legend()

                    # Plot dim 1
                    ax = fig.add_subplot(322)
                    ax.stem(x[seq_id, :, 0], markerfmt=' ', basefmt='C0', label='Spikes')
                    ax.plot(x_hat[seq_id, :, 0], label='Estimated FR')
                    ax.set_xlabel('Time')
                    ax.set_ylabel(f'Dim 1')
                    ax.legend()

                    # Plot dim 2
                    ax = fig.add_subplot(323)
                    ax.stem(x[seq_id, :, 1], markerfmt=' ', basefmt='C0', label='Spikes')
                    ax.plot(x_hat[seq_id, :, 1], label='Estimated FR')
                    ax.set_xlabel('Time')
                    ax.set_ylabel(f'Dim 2')
                    ax.legend()

                    # Plot dim 3
                    ax = fig.add_subplot(324)
                    ax.stem(x[seq_id, :, 2], markerfmt=' ', basefmt='C0', label='Spikes')
                    ax.plot(x_hat[seq_id, :, 2], label='Estimated FR')
                    ax.set_xlabel('Time')
                    ax.set_ylabel(f'Dim 3')
                    ax.legend()

                # Save plot
                if plot_smooth:
                    plot_name = f'{prefix}_{d}_embedding_smooth_{epoch}'
                else:
                    plot_name = f'{prefix}_{d}_embedding_filter_{epoch}'
                plt.savefig(f'{self.plot_save_dir}/{plot_name}.png')
                plt.close('all')


    def create_k_step_ahead_plots(self, s, y, m_s, m_y, likelihood_s, likelihood_y,
                                  model_vars, epoch, seq_id=0, prefix='train'):
        '''
        Creates and saves plots of k-step-ahead predicted distribution parameters

        s: torch.Tensor, spiking activity, or discrete observations. (num_seq, num_steps, n_s)
        y: torch.Tensor, LFP, or continuous signals. (num_seq, num_steps, n_y)
        m_s: torch.Tensor, mask tensor for s denoting whether s is available for a time-step or not. (num_seq, num_steps)
        m_y: torch.Tensor, mask tensor for y denoting whether y is available for a time-step or not. (num_seq, num_steps)
        likelihood_s: str, likelihood for modality s
        likelihood_y: str, likelihood for modality y
        model_vars: dict, dictionary of inferred variables returned by MRINE.forward()
        epoch: int, epoch number for which the plots will be saved
        seq_id: int, sequence id in the batch to plot
        prefix: str, prefix for plot save names
        '''

        # Disable gradient computation
        with torch.no_grad():
            num_k = len(self.config.loss.steps_ahead)
            fig = plt.figure(figsize=(20, 20))

            fig_num = 1
            for k in self.config.loss.steps_ahead:
                # Obtain k-step-ahead predicted distribution parameters
                data_dict_k_pred = self.mrine.get_k_step_ahead_prediction(model_vars, k)
                num_modalities = len(data_dict_k_pred.keys())

                for key, data, m, likelihood  in zip(['s', 'y'], [s, y], [m_s, m_y],
                                                    [likelihood_s, likelihood_y]): 
                    if data is not None:
                        # The first index of predictions are prediction k-th time-step from the initial condition, so plot it with true observation at time k 
                        x = data[:, k:, ...].detach().cpu() if k >= 0 else y.detach().cpu()
                        mask = m[:, k:, ...].detach().cpu() if k >= 0 else m_y.detach().cpu()

                        num_seq, _, dim_x = x.shape

                        # Plot only available time-steps
                        mask_bool = mask.type(torch.bool).unsqueeze(dim=-1).tile(1, 1, dim_x)
                        x = x[mask_bool].reshape(num_seq, -1, dim_x)
                        num_samples = x.shape[1] 
                        if likelihood.lower() == 'gaussian' and self.config.model.model_type in ['multi', 'single-gaussian']:
                            x_hat = data_dict_k_pred[key]['mu_pred'].detach().cpu()
                            x_hat = x_hat[mask_bool].reshape(num_seq, -1, dim_x)

                            # Plot dim 1
                            ax = fig.add_subplot(num_k, 3 * num_modalities, fig_num)
                            ax.plot(np.arange(k+1, num_samples+k+1), x[seq_id, :, 0], 'g', label=f'{k}-step x')
                            ax.plot(np.arange(k+1, num_samples+k+1), x_hat[seq_id, :, 0], 'b', label=f'{k}-step x pred')
                            ax.set_xlabel('Time')
                            ax.set_ylabel('Dim 1')
                            ax.set_title(f'k={k}, {key}')
                            ax.legend()
                            fig_num += 1

                            # Plot first 3 dimensions of predictions in 3D plot
                            color_index = range(x.shape[1])
                            color_map = plt.cm.get_cmap('viridis')
                            ax = fig.add_subplot(num_k, 3 * num_modalities, fig_num, projection='3d')
                            ax_m = ax.scatter(x_hat[seq_id, :, 0], x_hat[seq_id, :, 1], x_hat[seq_id, :, 2], c=color_index, vmin=0, vmax=x.shape[1], s=35, cmap=color_map, label='Estimated x')
                            ax.set_title(f'Estimated {key} mu embeddingin 3d')
                            ax.set_xlabel('Dim 1')
                            ax.set_ylabel('Dim 2')
                            ax.set_zlabel('Dim 3')
                            ax.legend()
                            fig.colorbar(ax_m)
                            fig_num += 2

                        elif likelihood.lower() == 'poisson' and self.config.model.model_type in ['multi', 'single-poisson']:
                            # Plot only available time-steps
                            x_hat = data_dict_k_pred[key]['fr_pred'].detach().cpu()
                            x_hat = x_hat[mask_bool].reshape(num_seq, -1, dim_x)
                            log_x_hat = torch.log(x_hat)

                            # Plot the maximum spiking dimension
                            dim_max_spiking = torch.argmax(torch.mean(x, dim=(0, 1)))
                            ax = fig.add_subplot(num_k, 3 * num_modalities, fig_num)
                            ax.stem(x[seq_id, :, dim_max_spiking], markerfmt=' ', basefmt='C0', label=f'{k}-step spike')
                            ax.plot(x_hat[seq_id, :, dim_max_spiking], label=f'{k}-step fr pred')

                            ax.set_xlabel('Time')
                            ax.set_ylabel(f'Dim {dim_max_spiking+1}')
                            ax.set_title(f'k={k}, {key}')
                            ax.legend()
                            fig_num += 1

                            color_index = range(x.shape[1])
                            color_map = plt.cm.get_cmap('viridis')
                            
                            # Plot first 3 dimensions of predictions in 3D plot
                            ax = fig.add_subplot(num_k, 3 * num_modalities, fig_num, projection='3d')
                            ax_m = ax.scatter(log_x_hat[seq_id, :, 0], log_x_hat[seq_id, :, 1], log_x_hat[seq_id, :, 2], c=color_index, vmin=0, vmax=x.shape[1], s=35, cmap=color_map, label='Estimated LogFR')
                            ax.set_xlabel('Dim 1')
                            ax.set_ylabel('Dim 2')
                            ax.set_zlabel('Dim 3')
                            ax.legend()
                            fig.colorbar(ax_m)
                            fig_num += 2

            # Save plot
            plot_name = f'{prefix}_k_step_embedding_{epoch}'
            plt.savefig(f'{self.plot_save_dir}/{plot_name}.png')
            plt.close('all')
    

    def create_a_x_plots(self, d, epoch, seq_id=0, prefix='train', feat_name='a'):
        '''
        d, torch.Tensor, latent factor to plot (num_seq, num_steps, n_x or n_a)
        epoch: int, epoch number for which the plots will be saved
        seq_id: int, sequence id in the batch to plot
        prefix: str, prefix for plot save names
        '''

        # Disable gradient computation
        with torch.no_grad():
            dim = d.shape[-1]
            d = d.detach().cpu().numpy()
            fig = plt.figure(figsize=(10,8))

            if dim > 2:
                # Plot first 3 dimensions in 3D plot
                ax = fig.add_subplot(221, projection='3d')
                ax.plot(d[seq_id, :, 0], d[seq_id, :, 1], d[seq_id, :, 2])
                ax.set_xlabel('Dim 1')
                ax.set_ylabel('Dim 2')
                ax.set_zlabel('Dim 3')
                ax.set_title(f'Embedded {feat_name} embedding in 3d')

                # View from top
                ax = fig.add_subplot(222)
                ax.plot(d[seq_id, :, 0], d[seq_id, :, 1])
                ax.set_xlabel('Dim 1')
                ax.set_ylabel('Dim 2')
                ax.set_title(f'Embedded {feat_name} embedding from top')
                
                # Plot dim 1
                ax = fig.add_subplot(223)
                ax.plot(range(d.shape[1]), d[seq_id, :, 0])
                ax.set_xlabel('Time')
                ax.set_ylabel('Dim 1')

                # Plot dim 2
                ax = fig.add_subplot(224)
                ax.plot(range(d.shape[1]), d[seq_id, :, 1])
                ax.set_xlabel('Time')
                ax.set_ylabel('Dim 2')
                fig.suptitle(f'Embedded {feat_name} embedding info', fontsize=16)

            elif dim == 2:
                # View from top
                ax = fig.add_subplot(221)
                ax.plot(d[seq_id, :, 0], d[seq_id, :, 1])
                ax.set_xlabel('Dim 1')
                ax.set_ylabel('Dim 2')
                ax.set_title(f'Embedded {feat_name} embedding from top')

                # Plot dim 1
                ax = fig.add_subplot(222)
                ax.plot(range(d.shape[1]), d[seq_id, :, 0])
                ax.set_xlabel('Time')
                ax.set_ylabel('Dim 1')

                # Plot dim 2
                ax = fig.add_subplot(223)
                ax.plot(range(d.shape[1]), d[seq_id, :, 1])
                ax.set_xlabel('Time')
                ax.set_ylabel('Dim 2')
                fig.suptitle(f'Embedded {feat_name} embedding info', fontsize=16)

            elif dim == 1:
                # Plot dim 1
                ax = fig.add_subplot(111)
                ax.plot(range(d.shape[1]), d[seq_id, :, 0])
                ax.set_xlabel('Time')
                ax.set_ylabel('Dim 1')

            # Save plot
            plot_name = f'{prefix}_embedded_{feat_name}_embedding_{epoch}'
            plt.savefig(f'{self.plot_save_dir}/{plot_name}.png')
            plt.close('all')


    def write_summary(self, epoch, prefix='train'):
        '''
        Logs metrics to tensorboard

        epoch: int, epoch number for which the metrics are logged
        prefix: str, prefix used when logging the metrics
        '''
        for key, val in self.metrics[prefix].items():
            self.writer.add_scalar(f'{prefix}/{key}', val.compute(), epoch)

        # Log the learning rate
        if prefix != 'valid': # no need to log it multiple times
            self.writer.add_scalar(f'learning_rate', self.optimizer.param_groups[0]['lr'], epoch)






        

     
