import os
import pandas as pd
import numpy as np
from PIL import Image

# Define paths
gt_path = './GT_mask/'
results_path = './results/'
path_csv = './outputs/test_data_.csv'
outputs_path = './outputs/'
smooth = 1e-6

# Initialize variables
count = 0
results = []

# Read the CSV file
df = pd.read_csv(path_csv)

# Create paths and volume names
paths = df.iloc[:, 1] + '_' + df.iloc[:, 0] + '_' + df.iloc[:, 3].astype(str).str.zfill(3) + '.tiff'
volume_names = df.iloc[:, 1] + '_' + df.iloc[:, 0]

# Get volume name counts
volume_name_counts = volume_names.value_counts().rename("count").reset_index()
volume_name_counts.columns = ['volume_name', 'count']

# Get the total number of unique volume names
total = len(volume_name_counts)

# Iterate over each unique volume name
for index, row in volume_name_counts.iterrows():
    volume_name = row['volume_name']
    count = row['count']
    
    gt_vol = []
    result_vol = []
    
    # Find matching rows in the original dataframe
    matching_rows = df[(df.iloc[:, 1] + '_' + df.iloc[:, 0]) == volume_name]

    # Iterate over each matching row
    for idx, matching_row in enumerate(matching_rows.iterrows()):
        gt_file_path = os.path.join(gt_path, f"{matching_row[1].iloc[1]}_{matching_row[1].iloc[0]}_{str(idx).zfill(3)}.tiff")
        result_file_path = os.path.join(results_path, f"{matching_row[1].iloc[1]}_{matching_row[1].iloc[0]}_{str(idx).zfill(3)}.tiff")
        print(gt_file_path)
        print(result_file_path)

        # Load masks and convert to numpy arrays
        gt_mask = Image.open(gt_file_path)
        gt_mask_array = np.array(gt_mask)

        result_mask = Image.open(result_file_path)
        result_mask_array = np.array(result_mask)
        
        # Append masks to the volumes
        gt_vol.append(gt_mask_array)
        result_vol.append(result_mask_array)

    # Stack the masks to create 3D volumes
    gt_vol = np.stack(gt_vol, axis=0)
    result_vol = np.stack(result_vol, axis=0)

    # Compute dice coefficients
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

    # Append results
    results.append({'name': row['volume_name'],
                    'total_dice_coefficient': total_dice_coefficient,
                    'IRF_dice_coefficient': IRF_dice_coefficient,
                    'SRF_dice_coefficient': SRF_dice_coefficient,
                    'PED_dice_coefficient': PED_dice_coefficient})

    count += 1
    print(f"{count} / {total}")

results_df = pd.DataFrame(results)
results_df.to_csv(os.path.join(outputs_path, 'dice_coefficients_per_volume_as_a_whole.csv'), index=False)
