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
        return (self.image_list[index], self.classes_list[index], self.annotations_list[index])
    
    def __len__(self):
        return len(self.image_list)