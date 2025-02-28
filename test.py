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
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.colors as mcolors
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.dates as mdates

# Set the directory path
FTIR_dir = '/Volumes/rvmartin/Active/ren.yuxuan/Mass_Reconstruction/'
Residual_dir = '/Volumes/rvmartin/Active/SPARTAN-shared/Public_Data/RCFM/'
OMOC_dir = '/Volumes/rvmartin/Active/ren.yuxuan/RCFM/FTIR_OC_OMOC_Residual/OM_OC/'
site_dir = '/Volumes/rvmartin/Active/SPARTAN-shared/Site_Sampling/'
out_dir = '/Volumes/rvmartin/Active/ren.yuxuan/Mass_Reconstruction/'
Colocation_dir = '/Volumes/rvmartin/Active/ren.yuxuan/Mass_Reconstruction/Co-location/'
################################################################################################
# Calculate and plot MAL
################################################################################################
# Read the file
AEAZ_df = pd.read_csv('/Volumes/rvmartin/Active/SPARTAN-shared/Analysis_Data/Master_files/AEAZ_master.csv')
ILHA_df = pd.read_csv('/Volumes/rvmartin/Active/SPARTAN-shared/Analysis_Data/Master_files/ILHA_master.csv')
ILNZ_df = pd.read_csv('/Volumes/rvmartin/Active/SPARTAN-shared/Analysis_Data/Master_files/ILNZ_master.csv')
# Add the 'Site' column to each DataFrame
AEAZ_df['Site'] = 'Abu Dhabi'
ILHA_df['Site'] = 'Haifa'
ILNZ_df['Site'] = 'Rehovot'
# Concatenate the DataFrames into master_df
master_df = pd.concat([AEAZ_df, ILHA_df, ILNZ_df], ignore_index=True)
master_df[['Volume_m3', 'start_year']] = master_df[['Volume_m3', 'start_year']].apply(pd.to_numeric, errors='coerce')
master_df = master_df[master_df['start_year'].isin([2019, 2020, 2021, 2022, 2023, 2024])]
# Define the columns that need to be converted (all _XRF_ng columns)
XRF_columns = [col for col in master_df.columns if col.endswith('_XRF_ng')]
master_df[XRF_columns] = master_df[XRF_columns].apply(pd.to_numeric, errors='coerce')
# Perform conversion for each of these columns
for col in XRF_columns:
    # Convert mass to concentration (ng/m3 to µg/m³)
    master_df[f'{col.replace("_XRF_ng", "")}'] = master_df[col] / master_df['Volume_m3'] / 1000  # Divide by 1000 to convert ng to µg

# Drop NaN and negative values in the specified columns
columns_to_check = ['start_year', 'Volume_m3', 'Al', 'K', 'Mg', 'Na', 'Si', 'Ca', 'Fe', 'Ti']
master_df = master_df.dropna(subset=columns_to_check)  # Drop NaN values
master_df = master_df[(master_df[columns_to_check] > 0).all(axis=1)]  # Drop negative values
print(master_df.head())

# ==== Dust Equation ====
# Calcule MAL based on K, Mg, Na for each filetr
MAL_default = 0.72
CF_default = 1.14
# Calculate MAL based on K, Mg, Na for each filter
master_df['MAL_corrected'] = (
    (1.20 * master_df['K'] / master_df['Al']) +
    (1.66 * master_df['Mg'] / master_df['Al']) +
    (1.35 * master_df['Na'] / master_df['Al'])
) / 1.89
master_df['Soil_default'] = (
    (1.89 * master_df['Al'] * (1 + MAL_default)) +
    (2.14 * master_df['Si']) +
    (1.40 * master_df['Ca']) +
    (1.36 * master_df['Fe']) +
    (1.67 * master_df['Ti'])
) * CF_default
master_df['Soil_corrected'] = (
    (1.89 * master_df['Al'] * (1 + master_df['MAL_corrected'])) +
    (2.14 * master_df['Si']) +
    (1.40 * master_df['Ca']) +
    (1.36 * master_df['Fe']) +
    (1.67 * master_df['Ti'])
) * CF_default

# # ==== Al & Si Attenuation Correction ====
# abu_dhabi_df['dust_loading'] = abu_dhabi_df['Soil_default'] * abu_dhabi_df['Volume_m3'] / 3.53 # Convert µg/m³ to µg and then to µg/cm²
# abu_dhabi_df['A'] = 0.78 - 8.6e-4 * abu_dhabi_df['dust_loading'] + 4.0e-7 *abu_dhabi_df['dust_loading'] ** 2 # A = 0.78 - 8.6e-4 * dust_loading + 4.0e-7 * dust_loading ** 2 #
#
# # Correct Al & Si values
# abu_dhabi_df['Si_corrected'] = abu_dhabi_df['Si'] / abu_dhabi_df['A']
# abu_dhabi_df['Al_corrected'] = abu_dhabi_df['Al'] * 0.77 / abu_dhabi_df['A']  # 0.77 adjustment for Al calibration

# Calculate and print statistics for MAL_corrected and Soil_corrected
for col in ['MAL_corrected', 'Soil_corrected', 'Soil_default']:
    mean_val = master_df[col].mean()
    median_val = master_df[col].median()
    min_val = master_df[col].min()
    max_val = master_df[col].max()
    se_val = master_df[col].std() / (len(master_df[col]) ** 0.5)
    print(f"Statistics for {col}:\nMean: {mean_val:.2f}\nMedian: {median_val:.2f}\nMin: {min_val:.2f}\nMax: {max_val:.2f}\nSE: {se_val:.2f}\n")
mean_MAL_corrected = (
    (1.20 * (master_df['K']).mean() / (master_df['Al']).mean()) +
    (1.66 * (master_df['Mg']).mean() / (master_df['Al']).mean()) +
    (1.35 * (master_df['Na']).mean() / (master_df['Al']).mean())
) / 1.89

print(f"mean_MAL_corrected: {mean_MAL_corrected:.4f}")


nmb = (master_df['Soil_corrected'] - master_df['Soil_default']).sum() / master_df['Soil_default'].sum()
print(f"NMB: {nmb:.4f}")

# # Save results to a new Excel file
# output_file = os.path.join(out_dir, 'AbuDhabi_MAL.xlsx')
# master_df[['FilterID', 'MAL_corrected', 'Soil_corrected', 'Soil_default', 'Soil', 'Al', 'Si', 'Ca', 'Fe', 'Ti', 'K', 'Mg', 'Na']].to_excel(output_file, index=False)

# Combine date
master_df['start_year'] = master_df['start_year'].astype(str).str.strip()
# Convert columns to numeric, filling invalid values with NaN, and then replace NaNs with 0 or a valid default
master_df['start_year'] = pd.to_numeric(master_df['start_year'], errors='coerce', downcast='integer')
master_df['start_year'] = master_df['start_year'].astype(int)
master_df['start_month'] = master_df['start_month'].astype(int)
master_df['start_day'] = master_df['start_day'].astype(int)
master_df['Date'] = pd.to_datetime(
    master_df['start_year'].astype(str) + '-' + master_df['start_month'].astype(str) + '-' + master_df[
        'start_day'].astype(str))

# Ensure the 'date' column is in datetime format
master_df['Date'] = pd.to_datetime(master_df['Date'], errors='coerce')

# Drop rows where 'date' or 'MAL_corrected' is NaN
master_df = master_df.dropna(subset=['Date', 'MAL_corrected'])

# Create the plot
plt.figure(figsize=(10, 6))
plt.axhline(y=0.72, color='grey', linestyle='--', linewidth=1)
# Plot for each site with corresponding color
for site, color in zip(['Abu Dhabi', 'Haifa', 'Rehovot'], ['black', 'blue', 'red']):
    site_data = master_df[master_df['Site'] == site]  # Filter the data for the site
    plt.plot(site_data['Date'], site_data['MAL_corrected'], marker='o', color=color, linestyle='None', label=site)
# Add a legend
legend = plt.legend(facecolor='white', prop={'family': 'Arial','size': 12})
legend.get_frame().set_edgecolor('white')

# Add labels and title
plt.xlabel('Date', fontsize=16, family='Arial')
plt.ylabel('MAL', fontsize=16, family='Arial')
# plt.title('MAL: Abu Dhabi', fontsize=16, family='Arial')

# Adjust the font for the tick labels
plt.xticks(rotation=45, fontsize=16, family='Arial')
plt.yticks(fontsize=16, family='Arial')
plt.ylim([0, 15])
# plt.xticks([2019, 2020, 2021, 2022, 2023, 2024], fontname='Arial', size=16)
plt.yticks([0, 5, 10, 15], fontname='Arial', size=16)
# Remove grid lines if you want to get rid of them
plt.grid(False)

# Show the plot
plt.tight_layout()
# plt.savefig(out_dir + 'MAL_AbhDhabi_Haifa_Rehovot.svg', dpi=300)

plt.show()
