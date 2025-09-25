from torch.utils.data import Dataset
import torch 

class MRINEDataset(Dataset):
    '''
    MRINE Dataset. 

    s: torch.Tensor, spiking activity, or discrete observations. (num_seq, num_steps, n_s)
    y: torch.Tensor, LFP, or continuous signals. (num_seq, num_steps, n_y)
    m_s: torch.Tensor, mask tensor for s denoting whether s is available for a time-step or not. (num_seq, num_steps)
    m_y: torch.Tensor, mask tensor for y denoting whether y is available for a time-step or not. (num_seq, num_steps)
    target: torch.Tensor,  target variable to decode from inferred latent factors. (num_seq, num_steps)
    '''
    def __init__(self, s=None, y=None, m_s=None, m_y=None, target=None):
        self.s = s
        self.y = y
        self.m_s = m_s
        self.m_y = m_y
        self.target = target

        if self.s is None and self.y is not None:
            self.s = self.y
            self.m_s = self.m_y
        elif self.s is not None and self.y is None:
            self.y = self.s
            self.m_y = self.m_s
        if self.target is None:
            self.target = torch.ones_like(self.y)

    def __len__(self):
        return self.s.shape[0]

    def __getitem__(self, idx):
        data_sample = []
        for data in [self.s, self.y, self.m_s, self.m_y, self.target]:
            if len(data.shape) == 2:
                data_sample.append(data[idx, :])
            elif len(data.shape) == 3:
                data_sample.append(data[idx, :, :])
        return data_sample

