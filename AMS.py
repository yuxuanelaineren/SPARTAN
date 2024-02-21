#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import calendar
import matplotlib.pyplot as plt
import cartopy.crs as ccrs  # cartopy must be >=0.19
import xarray as xr
import cartopy.feature as cfeature
import pandas as pd
from gamap_colormap import WhGrYlRd
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
# Extract IC from master file and lon/lat from site.details
################################################################################################
# Create an empty list to store individual HIPS DataFrames
IC_dfs = []

# Iterate over each file in the directory
for filename in os.listdir(obs_dir):
    if filename.endswith('.csv'):
        # Read the data from the master file
        master_data = pd.read_csv(os.path.join(obs_dir, filename), encoding='ISO-8859-1')

        # Specify the required columns
        IC_columns = ['FilterID', 'start_year', 'start_month', 'start_day', 'Mass_type', 'mass_ug', 'Volume_m3',
                        'IC_NO3_ug', 'IC_SO4_ug', 'IC_NH4_ug']
        # Check if all required columns are present
        if all(col in master_data.columns for col in IC_columns):
            # Remove leading/trailing whitespaces from column names
            master_data.columns = master_data.columns.str.strip() #Important!
            # Select the specified columns
            IC_df = master_data[IC_columns].copy()

            # Select PM2.5, rows where Mass_type is 1
            IC_df['Mass_type'] = pd.to_numeric(IC_df['Mass_type'], errors='coerce')
            IC_df = IC_df.loc[IC_df['Mass_type'] == 1]

            # Convert the relevant columns to numeric
            IC_df['IC_NO3_ug'] = pd.to_numeric(IC_df['IC_NO3_ug'], errors='coerce')
            IC_df['IC_SO4_ug'] = pd.to_numeric(IC_df['IC_SO4_ug'], errors='coerce')
            IC_df['IC_NH4_ug'] = pd.to_numeric(IC_df['IC_NH4_ug'], errors='coerce')

            # Extract the site name from the filename
            site_name = filename.split('_')[0]
            # Add the site name as a column in the selected data
            IC_df["Site"] = [site_name] * len(IC_df)

            # Append the current HIPS_df to the list
            IC_dfs.append(IC_df)
        else:
            print(f"Skipping {filename} because not all required columns are present.")

# Concatenate all HIPS DataFrames into a single DataFrame
IC_df = pd.concat(IC_dfs, ignore_index=True)

# Assuming your DataFrame is named obs_df
site_counts = IC_df.groupby('Site')['FilterID'].count()

# Print the number of rows for each site
for site, count in site_counts.items():
    print(f"{site}: {count} rows")

# Calculate concentrations in extraction solution, ug/mL
IC_df['NO3'] = IC_df['IC_SO4_ug'] / 6
IC_df['SO4'] = IC_df['IC_SO4_ug'] / 6
IC_df['NH4'] = IC_df['IC_SO4_ug'] / 6

# Read Site name and lon/lat from Site_detail.xlsx
site_df = pd.read_excel(os.path.join(site_dir, 'Site_details.xlsx'),
                        usecols=['Site_Code', 'Country', 'City', 'Latitude', 'Longitude'])

# Merge the dataframes based on the "Site" and "Site_Code" columns
obs_df = pd.merge(IC_df, site_df, how="left", left_on="Site", right_on="Site_Code")

# Drop the duplicate "Site_Code" column
obs_df.drop("Site_Code", axis=1, inplace=True)

# Write to excel file
with pd.ExcelWriter(os.path.join(out_dir, "IC_SPARTAN.xlsx"), engine='openpyxl') as writer:
    # Write the HIPS data to the 'HIPS_All' sheet
    obs_df.to_excel(writer, sheet_name='IC_All', index=False)


