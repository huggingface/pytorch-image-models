import torch
import random
from torch.utils.data.sampler import Sampler

class BagSampler(Sampler):
    def __init__(self, dataset):
        halfway_point = int(len(dataset)/2)
        self.first_half_indices = list(range(halfway_point))
        self.second_half_indices = list(range(halfway_point, len(dataset)))
        
    def __iter__(self):
        random.shuffle(self.first_half_indices)
        random.shuffle(self.second_half_indices)
        return iter(self.first_half_indices + self.second_half_indices)
    
    def __len__(self):
        return len(self.first_half_indices) + len(self.second_half_indices)