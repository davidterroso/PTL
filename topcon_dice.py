import os
import pandas as pd
import numpy as np
from PIL import Image

gt_path = './GT_topcon/'
results_path = './results_topcon/'
path_csv = './outputs/Topcon.csv'
outputs_path = './outputs/'
smooth = 1e-6

results = []

df = pd.read_csv(path_csv)

volume_names = df.iloc[:, 1] + '_' + df.iloc[:, 0]

volume_name_counts = volume_names.value_counts().rename("count").reset_index()
volume_name_counts.columns = ['volume_name', 'count']

print(volume_name_counts)

gt_vol = []
result_vol = []

total = len(volume_name_counts)

for index, row in volume_name_counts.iterrows():
    volume_name = row['volume_name']
    count = row['count']
    
    current_gt_vol = []
    current_result_vol = []
    
    matching_rows = df[(df.iloc[:, 1] + '_' + df.iloc[:, 0]) == volume_name]

    for idx in range(count):
        gt_file_path = os.path.join(gt_path, f"{volume_name}_{str(idx).zfill(3)}.tiff")
        result_file_path = os.path.join(results_path, f"{volume_name}_{str(idx).zfill(3)}.tiff")
        print(gt_file_path)
        print(result_file_path)

        gt_mask = Image.open(gt_file_path)
        gt_mask_array = np.array(gt_mask)

        result_mask = Image.open(result_file_path)
        result_mask_array = np.array(result_mask)
        
        current_gt_vol.append(gt_mask_array)
        current_result_vol.append(result_mask_array)

    current_gt_vol = np.stack(current_gt_vol, axis=0)
    current_result_vol = np.stack(current_result_vol, axis=0)
    
    gt_vol.append(current_gt_vol)
    result_vol.append(current_result_vol)

gt_vol = np.concatenate(gt_vol, axis=0)
result_vol = np.concatenate(result_vol, axis=0)

if((np.any(gt_vol==1)) or (np.any(gt_vol==2)) or (np.any(gt_vol==3)) or 
    np.any(result_vol==1) or np.any(result_vol==2) or np.any(result_vol==3)):
    binary_gt_mask = np.where(gt_vol == 0, 0, 1)
    binary_result_mask = np.where(result_vol == 0, 0, 1)
    total_intersection = np.sum(binary_gt_mask * binary_result_mask)
    total_union = np.sum(binary_gt_mask + binary_result_mask)
    total_dice_coefficient = (2. * total_intersection)/(total_union + smooth)
else:
    total_dice_coefficient = float("nan")

if np.any(gt_vol==1) or np.any(result_vol==1):
    binary_gt_IRF_mask = np.where(gt_vol == 1, 1, 0)
    binary_result_IRF_mask = np.where(result_vol == 1, 1, 0)
    IRF_intersection = np.sum(binary_gt_IRF_mask * binary_result_IRF_mask)
    IRF_union = np.sum(binary_gt_IRF_mask + binary_result_IRF_mask)
    IRF_dice_coefficient = (2. * IRF_intersection)/(IRF_union + smooth)
else:
    IRF_dice_coefficient = float("nan")

if np.any(gt_vol==2) or np.any(result_vol==2):
    binary_gt_SRF_mask = np.where(gt_vol == 2, 1, 0)
    binary_result_SRF_mask = np.where(result_vol == 2, 1, 0)
    SRF_intersection = np.sum(binary_gt_SRF_mask * binary_result_SRF_mask)
    SRF_union = np.sum(binary_gt_SRF_mask + binary_result_SRF_mask)
    SRF_dice_coefficient = (2. * SRF_intersection)/(SRF_union + smooth)
else:
    SRF_dice_coefficient = float("nan")

if np.any(gt_vol==3) or np.any(result_vol==3):
    binary_gt_PED_mask = np.where(gt_vol == 3, 1, 0)
    binary_result_PED_mask = np.where(result_vol == 3, 1, 0)
    PED_intersection = np.sum(binary_gt_PED_mask * binary_result_PED_mask)
    PED_union = np.sum(binary_gt_PED_mask + binary_result_PED_mask)
    PED_dice_coefficient = (2. * PED_intersection)/(PED_union + smooth)
else:
    PED_dice_coefficient = float("nan")

results.append({'total_dice_coefficient': total_dice_coefficient,
                'IRF_dice_coefficient': IRF_dice_coefficient,
                'SRF_dice_coefficient': SRF_dice_coefficient,
                'PED_dice_coefficient': PED_dice_coefficient})

count += 1
print(f"{count} / {total}")

results_df = pd.DataFrame(results)
results_df.to_csv(os.path.join(outputs_path, 'topcon_dice.csv'), index=False)