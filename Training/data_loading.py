import torch
from monai.data import (
    Dataset,
    DataLoader,
)
from monai.transforms import (
    Compose
)
import h5py
import os
import random

def data_spilt(base_dir, num_seg, num_rl, num_holdout, seed=325):
    list = os.listdir(base_dir)
    list = [os.path.join(base_dir, item) for item in list]
    random.seed(seed)
    random.shuffle(list)
    return list[:num_seg], list[num_seg:num_seg+num_rl], list[-num_holdout:]

    
def readh5(file_name):
    with h5py.File(file_name, 'r') as h5_file:
            t2 = torch.tensor(h5_file['t2'][:])
            dwi = torch.tensor(h5_file['dwi'][:])
            lesion = torch.tensor(h5_file['lesion'][:])
    return {
        't2': t2,
        'dwi': dwi,
        'lesion': lesion,
    }
class ReadH5d():
    def __call__(self, file_name):
        return readh5(file_name)



def create_data_loader(data_list, transform, batch_size, drop_last=True, shuffle=True):
    set = Dataset(data_list, transform)
    return DataLoader(set, num_workers=8, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)




def readh5_with_adc(file_name):
    with h5py.File(file_name, 'r') as h5_file:
            t2 = torch.tensor(h5_file['t2'][:])
            dwi = torch.tensor(h5_file['dwi'][:])
            adc = torch.tensor(h5_file['adc'][:])
            lesion = torch.tensor(h5_file['lesion'][:])
    return {
        't2': t2,
        'dwi': dwi,
        'adc': adc,
        'lesion': lesion,
    }
class ReadH5d_with_adc():
    def __call__(self, file_name):
        return readh5_with_adc(file_name)

