import os
import pandas as pd
import numpy as np
from PIL import Image

gt_path = './GT_mask/'
results_path = './results/'
path_csv = './outputs/test_data_.csv'
outputs_path = './outputs/'
smooth = 1e-6

count = 0
results = []
df = pd.read_csv(path_csv)
paths = df.iloc[:, 1] + '_' + df.iloc[:, 0] + '_' + df.iloc[:, 3].astype(str).str.zfill(3) + '.tiff'
total = len(paths)

for subpath in paths:
    gt_file_path = os.path.join(gt_path, subpath)
    result_file_path = os.path.join(results_path, subpath)

    gt_mask = Image.open(gt_file_path)
    gt_mask_array = np.array(gt_mask)

    result_mask = Image.open(result_file_path)
    result_mask_array = np.array(result_mask)

    if((np.any(gt_mask_array==1)) or (np.any(gt_mask_array==2)) or (np.any(gt_mask_array==3)) or 
       np.any(result_mask_array==1) or np.any(result_mask_array==2) or np.any(result_mask_array==3)):
        binary_gt_mask = np.where(gt_mask_array == 0, 0, 1)
        binary_result_mask = np.where(result_mask_array == 0, 0, 1)
        total_intersection = np.sum(binary_gt_mask * binary_result_mask)
        total_union = np.sum(binary_gt_mask + binary_result_mask)
        total_dice_coefficient = (2. * total_intersection)/(total_union)
    else: total_dice_coefficient = float("nan")

    if(np.any(gt_mask_array==1) or np.any(result_mask_array==1)):
        binary_gt_IRF_mask = np.where(gt_mask_array == 1, 1, 0)
        binary_result_IRF_mask = np.where(result_mask_array == 1, 1, 0)
        IRF_intersection = np.sum(binary_gt_IRF_mask * binary_result_IRF_mask)
        IRF_union = np.sum(binary_gt_IRF_mask + binary_result_IRF_mask)
        IRF_dice_coefficient = (2. * IRF_intersection)/(IRF_union)
    else: IRF_dice_coefficient = float("nan")

    if(np.any(gt_mask_array==2) or np.any(result_mask_array==2)):
        binary_gt_SRF_mask = np.where(gt_mask_array == 2, 1, 0)
        binary_result_SRF_mask = np.where(result_mask_array == 2, 1, 0)
        SRF_intersection = np.sum(binary_gt_SRF_mask * binary_result_SRF_mask)
        SRF_union = np.sum(binary_gt_SRF_mask + binary_result_SRF_mask)
        SRF_dice_coefficient = (2. * SRF_intersection)/(SRF_union)
    else: SRF_dice_coefficient = float("nan")

    if(np.any(gt_mask_array==3) or np.any(result_mask_array==3)):
        binary_gt_PED_mask = np.where(gt_mask_array == 3, 1, 0)
        binary_result_PED_mask = np.where(result_mask_array == 3, 1, 0)
        PED_intersection = np.sum(binary_gt_PED_mask * binary_result_PED_mask)
        PED_union = np.sum(binary_gt_PED_mask + binary_result_PED_mask)
        PED_dice_coefficient = (2. * PED_intersection)/(PED_union)
    else: PED_dice_coefficient = float("nan")

    volume = subpath.split('_')[-2]
    vendor = subpath.split('_')[-3]
    results.append({'vendor': vendor,
                    'volume': volume,
                    'name': subpath.replace('.tiff', ''),
                    'total_dice_coefficient': total_dice_coefficient,
                    'IRF_dice_coefficient': IRF_dice_coefficient,
                    'SRF_dice_coefficient': SRF_dice_coefficient,
                    'PED_dice_coefficient': PED_dice_coefficient
    })

    count += 1
    print(f"{count} / {total}")

results_df = pd.DataFrame(results)
results_df.to_csv(os.path.join(outputs_path, 'dice_coefficients_per_slice.csv'), index=False)

numeric_columns = ['total_dice_coefficient', 'IRF_dice_coefficient', 'SRF_dice_coefficient', 'PED_dice_coefficient']

results_df_per_volume = results_df.groupby(['volume'])[numeric_columns].mean()
results_df_per_volume.to_csv(os.path.join(outputs_path, 'dice_coefficients_per_volume.csv'))

results_df_per_vendor = results_df.groupby(['vendor'])[numeric_columns].mean()
results_df_per_vendor.to_csv(os.path.join(outputs_path, 'dice_coefficients_per_vendor.csv'))