import re
import h5py
import torch
import os
from monai.transforms import ScaleIntensityRangePercentiles
import SimpleITK as sitk
import matplotlib.pyplot as plt




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
    
    _list = []

    for item in patient_name_list:
        _list.append({
            'name': item,
            't2': os.path.join(base_dir, item, _t2),
            'dwi': os.path.join(base_dir, item, _dwi),
            'adc': os.path.join(base_dir, item, _adc),
            'lesion': os.path.join(base_dir, item, _lesion)
        })
        

    return _list



def save_h5(root_dir: str, file_nick_name: str, img_tensor: torch.tensor, lesion_tensor: torch.tensor):
    file_name = os.path.join(root_dir, file_nick_name + '.h5')
    with h5py.File(file_name, 'w') as h5_file:
        h5_file.create_dataset('img', data=img_tensor)
        h5_file.create_dataset('lesion', data=lesion_tensor)



file_name = r"E:\miami-data\data-ROI-192-96\data_split_files\data_exclude_missing\path_list.txt"
base_dir = r"E:\miami-data\data-ROI-192-96"
data_list = get_all_files(file_name, base_dir)

print(data_list, len(data_list))


scaler = ScaleIntensityRangePercentiles(0, 100, 0, 1)
for item in data_list:

    item_name = item['name']
    t2_image = sitk.ReadImage(item['t2'])
    dwi_image = sitk.ReadImage(item['dwi'])
    adc_image = sitk.ReadImage(item['adc'])
    lesion_image = sitk.ReadImage(item['lesion'])



    t2_tensor = sitk.GetArrayFromImage(t2_image)
    dwi_tensor = sitk.GetArrayFromImage(dwi_image)
    adc_tensor = sitk.GetArrayFromImage(adc_image)
    lesion_tensor = sitk.GetArrayFromImage(lesion_image)




    t2_tensor = torch.from_numpy(t2_tensor).permute(1, 2, 0)
    dwi_tensor = torch.from_numpy(dwi_tensor).permute(1, 2, 0)
    adc_tensor = torch.from_numpy(adc_tensor).permute(1, 2, 0)
    lesion_tensor = torch.from_numpy(lesion_tensor).permute(1, 2, 0)


    t2_tensor = scaler(t2_tensor)
    dwi_tensor = scaler(dwi_tensor)
    adc_tensor = scaler(adc_tensor)
    lesion_tensor = (lesion_tensor != 0).float()


    img_tensor = torch.stack([
        t2_tensor,
        dwi_tensor,
        adc_tensor
    ])
    lesion_tensor = lesion_tensor.unsqueeze(0)

    save_h5('miama_h5', item_name, img_tensor, lesion_tensor)

# print(img_tensor.shape, lesion_tensor.shape)
# fig, axes = plt.subplots(1, 4, figsize=(12, 3))  # Adjust figure size if needed


# index = 60
# axes[0].imshow(img_tensor[0, :, :, index], cmap='gray')  # Display image (use cmap='gray' for grayscale)
# axes[0].axis('off')  # Hide axes

# axes[1].imshow(img_tensor[1, :, :, index], cmap='gray')  # Display image (use cmap='gray' for grayscale)
# axes[1].axis('off')  # Hide axes

# axes[2].imshow(img_tensor[2, :, :, index], cmap='gray')  # Display image (use cmap='gray' for grayscale)
# axes[2].axis('off')  # Hide axes

# axes[3].imshow(lesion_tensor[0, :, :, index], cmap='gray')  # Display image (use cmap='gray' for grayscale)
# axes[3].axis('off')  # Hide axes

# plt.show()
