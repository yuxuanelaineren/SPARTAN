#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import calendar
import matplotlib.pyplot as plt
import cartopy.crs as ccrs  # cartopy must be >=0.19
import xarray as xr
import cartopy.feature as cfeature
import pandas as pd
import calendar
import numpy as np
from scipy.spatial.distance import cdist
import geopandas as gpd
from shapely.geometry import Point
from shapely.ops import nearest_points
from matplotlib import font_manager
import seaborn as sns
from scipy import stats

# Set the directory path
obs_dir = '/Volumes/rvmartin/Active/SPARTAN-shared/Analysis_Data/Master_files/'
site_dir = '/Volumes/rvmartin/Active/SPARTAN-shared/Site_Sampling/'
out_dir = '/Volumes/rvmartin/Active/ren.yuxuan/AMS/'
################################################################################################
# Extract IC from masterfile and lon/lat from site.details
################################################################################################
# Function to read and preprocess data from master files
def read_master_files(obs_dir):
    IC_dfs = []
    for filename in os.listdir(obs_dir):
        if filename.endswith('.csv'):
            master_data = pd.read_csv(os.path.join(obs_dir, filename), encoding='ISO-8859-1')
            IC_columns = ['FilterID', 'start_year', 'start_month', 'start_day', 'Mass_type', 'mass_ug', 'Volume_m3',
                          'IC_NO3_ug', 'IC_SO4_ug', 'IC_NH4_ug']
            if all(col in master_data.columns for col in IC_columns):
                # Select the specified columns
                master_data.columns = master_data.columns.str.strip()
                IC_df = master_data[IC_columns].copy()
                # Select PM2.5
                IC_df['Mass_type'] = pd.to_numeric(IC_df['Mass_type'], errors='coerce')
                IC_df = IC_df.loc[IC_df['Mass_type'] == 1]
                # Convert the relevant columns to numeric
                IC_df[['IC_NO3_ug', 'IC_SO4_ug', 'IC_NH4_ug', 'mass_ug', 'Volume_m3', 'start_year']] = IC_df[
                    ['IC_NO3_ug', 'IC_SO4_ug', 'IC_NH4_ug', 'mass_ug', 'Volume_m3', 'start_year']].apply(pd.to_numeric, errors='coerce')
                # Drop rows with NaN values
                IC_df = IC_df.dropna(subset=['start_year', 'Volume_m3', 'IC_NO3_ug'])
                IC_df = IC_df[IC_df['Volume_m3'] > 0]  # Exclude rows where Volume_m3 is 0
                # Calculate concentrations in extraction solution, ug/mL
                IC_df['NO3'] = IC_df['IC_NO3_ug'] / 6
                IC_df['SO4'] = IC_df['IC_SO4_ug'] / 6
                IC_df['NH4'] = IC_df['IC_NH4_ug'] / 6
                # Extract the site name and add as a column
                site_name = filename.split('_')[0]
                IC_df["Site"] = [site_name] * len(IC_df)
                # Append the current HIPS_df to the list
                IC_dfs.append(IC_df)
            else:
                print(f"Skipping {filename} because not all required columns are present.")
    return pd.concat(IC_dfs, ignore_index=True)

# Main script
if __name__ == "__main__":
    # Read data
    IC_df = read_master_files(obs_dir)
    site_df = pd.read_excel(os.path.join(site_dir, 'Site_details.xlsx'), usecols=['Site_Code', 'Country', 'City', 'Latitude', 'Longitude'])
    obs_df = pd.merge(IC_df, site_df, how="left", left_on="Site", right_on="Site_Code").drop("Site_Code", axis=1)

    # Write HIPS data to Excel
    with pd.ExcelWriter(os.path.join(out_dir, "IC_SPARTAN.xlsx"), engine='openpyxl', mode='w') as writer:
        obs_df.to_excel(writer, sheet_name='All', index=False)

    # Writ summary statistics to Excel
    site_counts = obs_df.groupby('Site')['FilterID'].count()
    for site, count in site_counts.items():
        print(f"{site}: {count} rows")
    summary_df = obs_df.groupby(['Country', 'City'])['NO3'].agg(['count', 'mean', 'std'])
    summary_df['stderr'] = summary_df['std'] / np.sqrt(summary_df['count']).pow(0.5)
    summary_df.rename(columns={'count': 'num_obs', 'mean': 'NO3_mean', 'std': 'NO3_stdv', 'stderr': 'NO3_stderr'},
                      inplace=True)
    with pd.ExcelWriter(os.path.join(out_dir, "IC_SPARTAN.xlsx"), engine='openpyxl', mode='a') as writer:
        summary_df.to_excel(writer, sheet_name='Summary', index=True)

################################################################################################
# Create a box plot for NO3 concentration in extraction
################################################################################################
# Set the directory path
obs_df = pd.read_excel(out_dir + 'IC_SPARTAN.xlsx', sheet_name='All')

# Set font properties
sns.set(font='Arial')

# Create the box plot
plt.figure(figsize=(12, 8))
ax = sns.boxplot(x='City', y='NO3', data=obs_df, color='white',
                 boxprops=dict(edgecolor='black', linewidth=1, facecolor='white'),  # Box properties
                 flierprops=dict(marker='o', markeredgecolor='black', markerfacecolor='black', markersize=4))  # Flier properties
ax.set_facecolor('white')

# Iterate over each line element to set border color and width
for line in ax.lines:
    line.set_color('black')  # set border color to black
    line.set_linewidth(1)  # set border width to 1
for spine in ax.spines.values():
    spine.set_edgecolor('black')
    spine.set_linewidth(1)
ax.grid(False)  # remove the grid
ax.tick_params(axis='both', direction='inout', length=6, width=1, colors='black')

# Set labels and title
plt.xlabel('City', fontsize=18)
plt.ylabel('Nitrate Concentration (µg/mL)', fontsize=18)
plt.title('Nitrate Concentration in Extraction Solution', fontsize=20)
plt.ylim([-0.5, 10])
plt.yticks(fontname='Arial', size=18)
plt.xticks(rotation=45, ha='right', size=16)

# Show plot
plt.tight_layout()
plt.savefig(out_dir + 'Box_NO3_Extraction_Solution.tiff', dpi=600)
plt.show()