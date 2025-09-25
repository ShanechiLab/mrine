from torchmetrics import Metric
import torch 

class Mean(Metric):
    '''
    Metric class to log metrics during training and to tensorboard
    '''
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.add_state("sum", default=torch.tensor(0, dtype=torch.float32), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0, dtype=torch.float32), dist_reduce_fx="sum")

    def update(self, value: torch.Tensor, batch_size: torch.Tensor):
        '''
        Updates the metrics with value which is computer over samples with size of batch_size
        '''
        value = value.clone().detach() if isinstance(value, torch.Tensor) else torch.tensor(value, dtype=torch.float32)
        batch_size = torch.tensor(batch_size, dtype=torch.float32)
        self.sum += value.cpu() * batch_size
        self.total += batch_size

    def reset(self):
        '''
        Resets the metric
        '''
        self.sum = torch.tensor(0, dtype=torch.float32)
        self.total = torch.tensor(0, dtype=torch.float32)

    def compute(self):
        '''
        Computes the metric
        '''
        return self.sum / self.total
