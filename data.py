# Module containing methods for data generation

import pickle as pkl
import torch


data_dir = './gans_data/data_dg'


# dataset for GANs
class GANs_Dataset(torch.utils.data.Dataset):
    def __init__(self):
        """
        load training samples for GANs
        """
        
        with open(f'{data_dir}/image_list', 'rb') as f:
            self.image_list = pkl.load(f)
            
        with open(f'{data_dir}/classes_list', 'rb') as f:
            self.classes_list = pkl.load(f)
            
        with open(f'{data_dir}/annotations_list', 'rb') as f:
            self.annotations_list = pkl.load(f)
        
    def __getitem__(self, index):
        cls = self.classes_list[index]
        ann = self.annotations_list[index]
        
        cls = torch.cat([cls] + [torch.tensor([-1]) for _ in range(5 - cls.shape[0])], dim=0)
        ann = torch.cat([ann] + [torch.tensor([-1, -1, -1, -1]).unsqueeze(0) for _ in range(5 - ann.shape[0])], dim=0)
            
        return (self.image_list[index], cls, ann)
    
    def __len__(self):
        return len(self.image_list)