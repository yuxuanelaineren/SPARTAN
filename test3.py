import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib.colors import ListedColormap
import matplotlib.colors as mcolors
import seaborn as sns
from scipy import stats

################################################################################################
# SPARTAN HIPS vs UV-Vis
################################################################################################
# Set the directory path
HIPS_dir = '/Volumes/rvmartin/Active/SPARTAN-shared/Analysis_Data/Master_files/'
UV_dir = '/Volumes/rvmartin/Active/ren.yuxuan/BC_Comparison/SPARTAN_BC/BC_UV-Vis_SPARTAN/'
IBR_dir = '/Volumes/rvmartin/Active/SPARTAN-shared/Analysis_Data/Black_Carbon/IBR_by_site/'
out_dir = '/Volumes/rvmartin/Active/ren.yuxuan/BC_Comparison/SPARTAN_BC/'

################################################################################################
# calculate IBR BC
################################################################################################

# Constants
q = 1.5 # thickness
filter_surface_area = 3.142  # cm^2
absorption_cross_section = 0.060  # cm^2/ug, 0.1 in SOP, 0.075 for AEAZ

# Define a function to calculate EBC (ug)
def calculate_ebc(row):
    normalized_r = row['normalized_r']
    if normalized_r > 0:  # Check if normalized_r is positive
        return  -filter_surface_area / (q * absorption_cross_section) * np.log(normalized_r / 1)
    else:
        return np.nan  # Replace invalid values with NaN

# Initialize an empty DataFrame to store the combined data
IBR_df = pd.DataFrame()

# Loop through each subfolder in the specified directory
for subfolder in os.listdir(IBR_dir):
    subfolder_path = os.path.join(IBR_dir, subfolder)
    # Check if the subfolder is a directory
    if os.path.isdir(subfolder_path):
        # Look for Excel files in the subfolder
        for file in os.listdir(subfolder_path):
            # Skip files starting with a dot
            if file.startswith('.') or file.startswith('~'):
                continue
            if file.endswith('_IBR.xlsx'):
                # Extract site name from the file name
                site_name = file.split('_')[0]
                print(site_name)
                # Read the Excel file
                excel_path = os.path.join(subfolder_path, file)
                df = pd.read_excel(excel_path, engine='openpyxl', header=7)
                # Extract required columns
                df.columns = df.columns.str.strip()  # Remove leading/trailing whitespace
                required_columns = ['Cartridge ID', 'Sample ID#', 'Reflectance']
                df = df[required_columns]
                # Drop rows with blank values in 'Reflectance' column
                df.dropna(subset=['Reflectance'], inplace=True)
                # Add 'site' column with site name
                df['site'] = site_name
                # Concatenate data with the combined DataFrame
                IBR_df = pd.concat([IBR_df, df])

# Write the merged data to separate sheets in an Excel file
with pd.ExcelWriter(os.path.join(out_dir, 'BC_IBR_SPARTAN.xlsx'), index=False, engine='openpyxl') as writer:
    # Write the merged data
    IBR_df.to_excel(writer, sheet_name='raw', index=False)

# Normalize 'Reflectance' for each 'Cartridge ID'
IBR_df = IBR_df.copy()
# IBR_df = pd.read_excel(os.path.join(out_dir, 'BC_IBR_SPARTAN.xlsx'))
for cart_id, group in IBR_df.groupby('Cartridge ID'):
    print(cart_id)
    # Find the 'Reflectance' value for 'Sample ID#' ending with '-7'
    ref_7 = group.loc[group['Sample ID#'].str.endswith('-7'), 'Reflectance'].iloc[0]
    # Calculate normalized 'Reflectance' values for other 'Sample ID#'s
    group['normalized_r'] = group['Reflectance'] / ref_7
    # Calculate EBC (ug) for each row
    group['EBC_ug'] = group.apply(calculate_ebc, axis=1)

    # Update the DataFrame with the calculated EBC values
    IBR_df.loc[group.index, 'normalized_r'] = group['normalized_r']
    IBR_df.loc[group.index, 'EBC_ug'] = group['EBC_ug']

# Write the combined data with normalized 'Reflectance' to a new Excel file
with pd.ExcelWriter(os.path.join(out_dir, 'BC_IBR_SPARTAN.xlsx'), index=False, engine='openpyxl') as writer:
    # Write the merged data
    IBR_df.to_excel(writer, sheet_name='IBR_EBC', index=False)
