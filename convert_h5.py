import re
import h5py
import torch
import os
from monai.transforms import ScaleIntensityRangePercentiles, Resize
import SimpleITK as sitk
# import matplotlib.pyplot as plt


def bbox_range(arr, seg, ):
    assert arr.shape == seg.shape, "the shapes of img and seg are not equal"
    assert (seg == 0).all() == False, "the values in the seg are all zeros"
    len_x, len_y, len_z = seg.shape
    # rx, ry, rz = radius[0], radius[1], radius[2]
    for x_a in range(len_x):
        if (seg[x_a, :, :] == 0).all() != True:
            break
    for x_b in range(len_x - 1, -1, -1):
        if (seg[x_b, :, :] == 0).all() != True:
            break
    for y_a in range(len_y):
        if (seg[:, y_a, :] == 0).all() != True:
            break
    for y_b in range(len_y - 1, -1, -1):
        if (seg[:, y_b, :] == 0).all() != True:
            break
    for z_a in range(len_z):
        if (seg[:, :, z_a] == 0).all() != True:
            break
    for z_b in range(len_z - 1, -1, -1):
        if (seg[:, :, z_b] == 0).all() != True:
            break
    
    return x_a, x_b, y_a, y_b, z_a, z_b

def cropAbyB(crop_tensor: torch.tensor, crop_reference_tensor: torch.tensor) -> torch.tensor:
    x_a, x_b, y_a, y_b, z_a, z_b = bbox_range(crop_tensor, crop_reference_tensor)
    return crop_tensor[x_a:x_b, y_a:y_b, z_a:z_b]

def get_all_patient_name(text_file):
    
    with open(text_file, "r") as file:
        content = file.read()

    # Use regex to find all occurrences of "P-1000"
    # Use regex to find all occurrences of P- followed by digits
    matches = set(re.findall(r'P-\d+', content))
    return sorted(matches)

def get_all_files(file_name, base_dir):
    patient_name_list = get_all_patient_name(file_name)
    _t2 = 't2.nii.gz'
    _dwi = 'dwi_2000.nii.gz'
    _adc = 'adc.nii.gz'
    _lesion = 'lesion_mask.nii.gz'
    _gland = 'prostate_mask.nii.gz'
    
    _list = []

    for item in patient_name_list:
        _list.append({
            'name': item,
            't2': os.path.join(base_dir, item, _t2),
            'dwi': os.path.join(base_dir, item, _dwi),
            'adc': os.path.join(base_dir, item, _adc),
            'lesion': os.path.join(base_dir, item, _lesion),
            'gland': os.path.join(base_dir, item, _gland)
        })
        

    return _list



def save_h5(root_dir: str, file_nick_name: str, t2_tensor: torch.tensor, 
            dwi_tensor: torch.tensor, adc_tensor: torch.tensor, 
            lesion_tensor: torch.tensor, gland_tensor: torch.tensor):
    file_name = os.path.join(root_dir, file_nick_name + '.h5')
    with h5py.File(file_name, 'w') as h5_file:
        h5_file.create_dataset('t2', data=t2_tensor)
        h5_file.create_dataset('dwi', data=dwi_tensor)
        h5_file.create_dataset('adc', data=adc_tensor)
        h5_file.create_dataset('lesion', data=lesion_tensor)
        h5_file.create_dataset('gland', data=gland_tensor)



file_name = r"/raid/candi/xiangcen/miami-data/data-ROI-192-96/data_split_files/data_exclude_missing/path_list.txt"
base_dir = r"/raid/candi/xiangcen/miami-data/data-ROI-192-96"
data_list = get_all_files(file_name, base_dir)

print(len(data_list))


scaler = ScaleIntensityRangePercentiles(0, 100, 0, 1)
resizer = Resize((128, 128, 32), mode="trilinear")
resizer_lesion = Resize((128, 128, 32), mode="nearest-exact")
for item in data_list:

    item_name = item['name']
    t2_image = sitk.ReadImage(item['t2'])
    dwi_image = sitk.ReadImage(item['dwi'])
    adc_image = sitk.ReadImage(item['adc'])
    lesion_image = sitk.ReadImage(item['lesion'])
    gland_tensor = sitk.ReadImage(item['gland'])



    t2_tensor = sitk.GetArrayFromImage(t2_image)
    dwi_tensor = sitk.GetArrayFromImage(dwi_image)
    adc_tensor = sitk.GetArrayFromImage(adc_image)
    lesion_tensor = sitk.GetArrayFromImage(lesion_image)
    gland_tensor = sitk.GetArrayFromImage(gland_tensor)




    t2_tensor = torch.from_numpy(t2_tensor).permute(1, 2, 0)
    dwi_tensor = torch.from_numpy(dwi_tensor).permute(1, 2, 0)
    adc_tensor = torch.from_numpy(adc_tensor).permute(1, 2, 0)
    lesion_tensor = torch.from_numpy(lesion_tensor).permute(1, 2, 0)
    gland_tensor = torch.from_numpy(gland_tensor).permute(1, 2, 0)
    
    
    crop_reference_tensor = (gland_tensor + lesion_tensor) != 0
    crop_reference_tensor = crop_reference_tensor.float()
    
    
    t2_tensor = cropAbyB(t2_tensor, crop_reference_tensor)
    dwi_tensor = cropAbyB(dwi_tensor, crop_reference_tensor)
    adc_tensor = cropAbyB(adc_tensor, crop_reference_tensor)
    lesion_tensor = cropAbyB(lesion_tensor, crop_reference_tensor)
    gland_tensor = cropAbyB(gland_tensor, crop_reference_tensor)
    
    
    t2_tensor = resizer(t2_tensor.unsqueeze(0))
    dwi_tensor = resizer(dwi_tensor.unsqueeze(0))
    adc_tensor = resizer(adc_tensor.unsqueeze(0))
    lesion_tensor = resizer_lesion(lesion_tensor.unsqueeze(0))
    gland_tensor = resizer_lesion(gland_tensor.unsqueeze(0))


    t2_tensor = scaler(t2_tensor)
    dwi_tensor = scaler(dwi_tensor)
    adc_tensor = scaler(adc_tensor)
    lesion_tensor = (lesion_tensor != 0).float()
    gland_tensor = scaler(gland_tensor)
    


    # img_tensor = torch.cat([
    #     t2_tensor,
    #     dwi_tensor,
    #     adc_tensor
    # ])
    # lesion_tensor = lesion_tensor

    save_h5('/raid/candi/xiangcen/miami-data/miama_h5', item_name, t2_tensor, dwi_tensor, adc_tensor, lesion_tensor, gland_tensor)
    print(f'{item_name} done')

