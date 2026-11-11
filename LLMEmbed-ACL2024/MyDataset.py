import torch
from torch.utils.data import Dataset, DataLoader

def _safe_torch_load(path):
    try:
        return torch.load(path, weights_only=False)
    except TypeError:
        return torch.load(path)

class MyDataset(Dataset):

    def __init__(self, mode, l_path, b_path, r_path):
        self.l_sents_reps = _safe_torch_load(l_path + f'{mode}_sents.pt')
        self.b_sents_reps = _safe_torch_load(b_path + f'{mode}_sents.pt')
        self.r_sents_reps = _safe_torch_load(r_path + f'{mode}_sents.pt')

        self.labels = _safe_torch_load(l_path + f'{mode}_labels.pt')
        
        self.sample_num = self.labels.shape[0]
        
    def __getitem__(self, index):
        return self.l_sents_reps[index], self.b_sents_reps[index], self.r_sents_reps[index], self.labels[index]        
        
    def __len__(self):
        return self.sample_num

