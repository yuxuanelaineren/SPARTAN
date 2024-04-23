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
IBR_dir = '/Volumes/rvmartin/Active/ren.yuxuan/BC_Comparison/SPARTAN_BC'
out_dir = '/Volumes/rvmartin/Active/ren.yuxuan/BC_Comparison/SPARTAN_BC/'

################################################################################################
# Combine HIPS and UV-Vis dataset, set site as country and city
################################################################################################
# Create an empty dataframe to store the combined data
combined_df = pd.DataFrame()

# Read the HIPS_df
HIPS_df = pd.read_excel(out_dir + 'BC_HIPS_SPARTAN.xlsx', sheet_name='All',
                        usecols=['FilterID', 'mass_ug', 'Volume_m3', 'BC_HIPS_ug', 'Site', 'Country', 'City'])

# Read the UV-Vis data from Analysis_ALL
IBR_df = pd.read_excel(os.path.join(IBR_dir, 'BC_IBR_SPARTAN.xlsx'), usecols=['Sample ID#', 'EBC_ug'])
# Drop the last two digits in the "Filter ID" column: 'AEAZ-0113-1' to 'AEAZ-0113'
IBR_df['FilterID'] = IBR_df['Sample ID#'].str[:-2]
# Merge DataFrames
merged_df = pd.merge(IBR_df, HIPS_df, on=['FilterID'], how='inner')

# Convert the relevant columns to numeric to handle any non-numeric values
merged_df['BC_HIPS_ug'] = pd.to_numeric(merged_df['BC_HIPS_ug'], errors='coerce')
merged_df['BC_IBR_ug'] = pd.to_numeric(merged_df['EBC_ug'], errors='coerce')
merged_df['mass_ug'] = pd.to_numeric(merged_df['mass_ug'], errors='coerce')
merged_df['Volume_m3'] = pd.to_numeric(merged_df['Volume_m3'], errors='coerce')

# Calculate BC concentrations
merged_df['BC_HIPS_(ug/m3)'] = merged_df['BC_HIPS_ug'] / merged_df['Volume_m3']
merged_df['BC_IBR_(ug/m3)'] = merged_df['BC_IBR_ug'] / merged_df['Volume_m3']

# Calculate BC fractions
merged_df['f_BC_HIPS'] = merged_df['BC_HIPS_ug'] / merged_df['mass_ug']
merged_df['f_BC_IBR'] = merged_df['BC_IBR_ug'] / merged_df['mass_ug']

# Drop rows where IBR BC < 0
merged_df = merged_df.loc[merged_df['BC_IBR_ug'] > 0]

# Write the merged data to separate sheets in an Excel file
with pd.ExcelWriter(os.path.join(out_dir, 'BC_HIPS_IBR_SPARTAN.xlsx'), engine='openpyxl') as writer:
    # Write the merged data
    merged_df.to_excel(writer, sheet_name='HIPS_IBR', index=False)