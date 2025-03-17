from Training import data_spilt, ReadH5d, create_data_loader
from Training import train_seg_net, test_seg_net
from monai.transforms import *
from monai.networks.nets import SwinUNETR
from monai.losses import DiceFocalLoss, DiceLoss
import torch
batch_size=6
num_epoch=1000



seg_list, rl_list, holdout_list = data_spilt('/raid/candi/xiangcen/miami-data/miama_h5', 925, 300, 100)


train_transform = Compose([
    ReadH5d(),
    # RandAffined(['t2', 'dwi', 'lesion'], spatial_size=(128, 128, 32), prob=0.25, shear_range=(0.1, 0.1, 0.1, 0.1, 0.1, 0.1), mode='nearest', padding_mode='zeros'),
    # RandGaussianSmoothd(['t2', 'dwi',], prob=0.1),
    # RandGaussianNoised(['t2', 'dwi',], prob=0.1, std=0.05),
    # RandAdjustContrastd(['t2', 'dwi',], prob=0.1, gamma=(0.5, 2.))
])
inference_transform = ReadH5d()

train_loader = create_data_loader(seg_list, train_transform, batch_size=batch_size, shuffle=True)
inference_loader = create_data_loader(holdout_list, inference_transform, batch_size=1, drop_last=False, shuffle=True)


model = SwinUNETR(
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
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)



for i in range(20):
    train_loss = train_seg_net(model, train_loader, optimizer, DiceLoss(sigmoid=True))
    dice_t2, dice_dwi, dice_both = test_seg_net(model, inference_loader)
    torch.save(model.state_dict(), "/raid/candi/xiangcen/trained_models/SegModels/swinunetr.ptm")
    print(f'epoch {i}, loss {train_loss}, t2:{dice_t2}, dwi:{dice_dwi}, both:{dice_both}')
