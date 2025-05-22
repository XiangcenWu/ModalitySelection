from Training import data_spilt, ReadH5d, create_data_loader
from Training import train_seg_net, test_seg_net
from monai.transforms import *
from monai.networks.nets import SwinUNETR
from monai.losses import DiceFocalLoss, DiceLoss
import torch
batch_size=6
num_epoch=1000
device = 'cuda:1'


seg_list, rl_list, holdout_list = data_spilt('/raid/candi/xiangcen/miami-data/miama_h5', 925, 300, 100)


inference_transform = ReadH5d()

inference_loader = create_data_loader(holdout_list, inference_transform, batch_size=1, drop_last=False, shuffle=True)


seg_model = SwinUNETR(
    img_size = (128, 128, 32),
    in_channels = 2,
    out_channels = 1,
    depths = (2, 2, 2, 2),
    num_heads = (3, 6, 12, 24),
    drop_rate = 0.,
    attn_drop_rate = 0.,
    dropout_path_rate = 0.,
    downsample="mergingv2",
    use_v2=True,
)
seg_model.load_state_dict(torch.load("/raid/candi/xiangcen/trained_models/SegModels/swinunetr.ptm", map_location=device))
seg_model.eval()




dice_t2, dice_dwi, dice_both = test_seg_net(seg_model, inference_loader, device=device)
print(f't2:{dice_t2.mean().item(), dice_t2.std().item()}, dwi:{dice_dwi.mean().item(), dice_dwi.std().item()}\
, both:{dice_both.mean().item(), dice_both.std().item()}')
