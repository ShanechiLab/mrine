import os 
import torch 
import datetime
import logging
from torch.utils.tensorboard import SummaryWriter

class BaseTrainer(object):
    '''
    Base trainer class
    '''

    def __init__(self, config):
        '''
        Initializes TrainerMRINE object

        config: CfgNode, config denoting the hyperparameters of MRINE
        '''
        
        self.config = config

        # Create save directories for plots and checkpoints
        self.ckpt_save_dir = f'{self.config.model.save_dir}/ckpts'; os.makedirs(self.ckpt_save_dir, exist_ok=True)
        self.plot_save_dir = f'{self.config.model.save_dir}/plots'; os.makedirs(self.plot_save_dir, exist_ok=True)

        # Training starts from the first epoch if not in resume_train mode
        self.start_epoch = 1
        self.writer = SummaryWriter(log_dir=f'{self.config.model.save_dir}/summary') # tensorboard writer
    

    def _save_config(self):
        '''
        Saves the config in the save directory denoted in the config
        '''
        config_save_path = f'{self.config.model.save_dir}/config.yaml'
        with open(config_save_path, 'w') as outfile:
            outfile.write(self.config.dump())
    

    def _get_optimizer(self, params):
        '''
        Creates Adam optimizer 
        
        params: list, list of parameters to be optimized
        '''
        optimizer = torch.optim.Adam(params=params, 
                                     lr=self.config.lr.init_lr)
        
        return optimizer


    def _get_metrics(self):
        # Must return metric_names as list, and metrics as dictionary
        pass


    def _reset_metrics(self, train_valid='train'):
        '''
        Resets all the metrics to be computed in the next epoch

        train_valid: str, which metrics to reset
        '''
        for _, metric in self.metrics[train_valid].items():
            metric.reset()


    def _update_metrics(self, loss_dict, batch_size, train_valid='train', verbose=True):
        '''
        Updates the metrics that are presented in the loss_dict

        loss_dict: dict, dictionary of computed loss values that are used to update metrics
        batch_size: int, size of the batch for which the loss values are computed
        train_valid: str, on which set of data, e.g. training or validation, the loss values are computed
        verbose: bool, if True and a metric does not exist in the loss_dict, raises warning  
        '''
        # Disable gradient computation
        with torch.no_grad():
            for key in self.metric_names:
                if key not in loss_dict:
                    if verbose:
                        self.logger.warning(f'{key} does not exist in loss_dict, metric cannot be updated!')
                    else:
                        pass
                else:
                    self.metrics[train_valid][key].update(loss_dict[key], batch_size)


    def _get_logger(self, prefix='mrine'):
        '''
        Initializes the logger object to log training progress
        
        prefix: str, name of the logger object 
        '''
        os.makedirs(self.config.model.save_dir, exist_ok=True)
        date_time = datetime.datetime.now().strftime("%m-%d_%H-%M")
        log_path = f'{self.config.model.save_dir}/{prefix}_{date_time}.log'

        # from: https://stackoverflow.com/a/56689445/16228104
        logger = logging.getLogger(f'{prefix.upper()} Logger')
        logger.setLevel(logging.DEBUG)

        # remove old handlers from logger (since logger is static object) so that in several calls, it doesn't overwrite to previous log files
        handlers = logger.handlers[:]
        for handler in handlers:
            logger.removeHandler(handler)
            handler.close()
        
        # create file handler which logs even debug messages
        fh = logging.FileHandler(log_path, mode='a')
        fh.setLevel(logging.DEBUG)
        
        # create console handler with a higher log level
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)

        # create formatter and add it to the handlers
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s', '%m/%d/%Y %I:%M:%S %p')
        ch.setFormatter(formatter)
        fh.setFormatter(formatter)

        # add the handlers to logger
        logger.addHandler(ch)
        logger.addHandler(fh)

        return logger


    def _get_lr_scheduler(self):
        '''
        Initializes the Cyclic learning rate scheduler (https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.CyclicLR.html)
        '''
        scheduler = torch.optim.lr_scheduler.CyclicLR(self.optimizer, 
                                                      base_lr=self.config.lr.base_lr, max_lr=self.config.lr.max_lr, 
                                                      mode=self.config.lr.mode, gamma=self.config.lr.gamma, step_size_up=self.config.lr.step_size_up, 
                                                      cycle_momentum=False)
        return scheduler

    
    def _load_ckpt(self, model, optimizer=None, lr_scheduler=None, load_dir=None, file_name=None):
        '''
        Loads a checkpoint provided in file_name saved in the save directory

        model: nn.Module, model to overwrite its parameters from the loaded checkpoint
        optimizer: optim.Adam, Adam optimizer to overwrite its first and second order statistics from the loaded checkpoint if resume_train is True
        lr_scheduler: optim.lr_scheduler.CyclicLR, CyclicLR object to load if resume_train is True
        load_dir: str, checkpoint loading directory. if None, save directory in the config will be used
        file_name: str, file name of the checkpoint to load without .pth extension
        '''
        if load_dir is None: 
            load_dir = self.config.model.save_dir
        
        # Optimizer and LR scheduler can be loaded only in resume_train mode, meaningless to load otherwise
        self.logger.warning('Optimizer and LR scheduler can be loaded only in resume_train mode, else they are re-initialized')
        load_path = f'{load_dir}/ckpts/{file_name}_ckpt.pth'
        self.logger.info(f'Loading model from: {load_path}...')

        # If checkpoint does not exist, return the initial model, optimizer and LR scheduler
        try: 
            ckpt = torch.load(load_path)
        except:
            self.logger.warning('Ckpt path does not exist! Loading is skipped.')
            return model, optimizer, lr_scheduler
        
        if self.config.load.resume_train:
            # Set start epoch as the number of the epoch that checkpoint was saved
            self.start_epoch = ckpt['epoch'] + 1 if isinstance(ckpt['epoch'], int) else 1

            # Try loading the optimizer
            if optimizer is not None:
                try:
                    optimizer.load_state_dict(ckpt['optimizer'])
                except:
                    self.logger.warning('Optimizer cannot be loaded!, check if optimizer type is consistent! Loading is skipped.')
            
            if lr_scheduler is not None:
                try:
                    lr_scheduler.load_state_dict(ckpt['lr_scheduler'])
                except:
                    self.logger.warning('LR scheduler cannot be loaded, check if scheduler type is consistent! Loading is skipped.')

        # Try loading the model, if architectures are not consistent, return the initial model
        try:
            model.load_state_dict(ckpt['state_dict'])
        except Exception as e:
            self.logger.warning(e)
            self.logger.warning('Given architecture in config does not match the architecture of given checkpoint! Loading is skipped.')
            return model, optimizer, lr_scheduler
        
        self.logger.info(f'Checkpoint succesfully loaded from {load_path}!')
        return model, optimizer, lr_scheduler
        

    def write_model_gradients(self, model, step, prefix='unclipped'):
        '''
        Log gradient norms to tensorboard

        model: nn.Module, model to overwrite its parameters from the loaded checkpoint
        step: int, training step for which the gradients will be logged
        prefix: str, prefix denoting whether clipped or unclipped gradients are being logged
        '''

        # Compute the total gradient norm with and without LDM parameters
        total_norm = 0
        total_norm_wo_ldm = 0
        for name, p in model.named_parameters():
            if p.grad is not None:
                grad_norm =  p.grad.detach().data.cpu().norm(2)
                total_norm += grad_norm ** 2
                if 'ldm' not in name.lower():
                    total_norm_wo_ldm += grad_norm ** 2
                self.writer.add_scalar('grads/' + name + f"/{prefix}_grad", grad_norm, step)
        
        # Log the gradient norms
        total_norm = total_norm ** 0.5
        total_norm_wo_ldm = total_norm_wo_ldm ** 0.5
        self.writer.add_scalar(f'grads/total_grad_{prefix}_norm', total_norm, step)
        self.writer.add_scalar(f'grads/total_grad_{prefix}_norm_wo_LDM', total_norm_wo_ldm, step)
        

    def _save_ckpt(self, epoch, model, optimizer, lr_scheduler=None, save_name=None):
        '''
        Saves the model in a .pth file

        epoch: int, epoch number for which the model is being saved
        model: nn.Module, model to overwrite its parameters from the loaded checkpoint
        optimizer: optim.Adam, Adam optimizer to save
        lr_scheduler: optim.lr_scheduler.CyclicLR, CyclicLR object to save
        save_name: str, file name of the checkpoint file
        '''

        if save_name is None:
            save_name = epoch
        save_path = f'{self.ckpt_save_dir}/{save_name}_ckpt.pth'

        if lr_scheduler is not None:
            torch.save({
                        'state_dict': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'lr_scheduler': lr_scheduler.state_dict(),
                        'epoch': epoch
                        }, save_path)
        else:
            torch.save({
                        'state_dict': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'epoch': epoch
                        }, save_path)


    def train_epoch(self, epoch, train_loader):
        '''
        Empty function for one epoch training
        '''
        pass


    def valid_epoch(self, epoch, train_loader):
        '''
        Empty function for one epoch validation (evaluation)
        '''
        pass


    def train(self, train_loader, valid_loader):
        '''
        Empty function for model training over course of epochs denoted in the config
        '''
        pass