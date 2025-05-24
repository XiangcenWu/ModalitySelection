from Training import data_spilt, ReadH5d, create_data_loader
from Training import train_seg_net, test_seg_net
from monai.transforms import *
from monai.networks.nets import SwinUNETR
from monai.losses import DiceFocalLoss, DiceLoss
import torch
batch_size=6
num_epoch=1000
device = 'cuda:0'

seg_list, rl_list, holdout_list = data_spilt('/raid/candi/xiangcen/miami-data/miama_h5', 925, 300, 100)
print(holdout_list[:5])

seg_list, rl_list, holdout_list = data_spilt('/raid/candi/xiangcen/miami-data/miama_h5_reinforce', 925, 300, 100)
print(holdout_list[:5])


