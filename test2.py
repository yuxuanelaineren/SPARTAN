#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
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
from scipy.io import loadmat

# Set the directory path
obs_dir = '/Volumes/rvmartin/Active/SPARTAN-shared/Analysis_Data/Master_files/'
site_dir = '/Volumes/rvmartin/Active/SPARTAN-shared/Site_Sampling/'
out_dir = '/Volumes/rvmartin/Active/ren.yuxuan/AMS/supportData/'
################################################################################################
# Extract XRF Sulfur and IC Sulfate from masterfile and lon/lat from site.details
################################################################################################
# Function to read and preprocess data from master files
def read_master_files(obs_dir):
    excluded_filters = [
        'AEAZ-0078', 'AEAZ-0086', 'AEAZ-0089', 'AEAZ-0090', 'AEAZ-0093', 'AEAZ-0097',
        'AEAZ-0106', 'AEAZ-0114', 'AEAZ-0115', 'AEAZ-0116', 'AEAZ-0141', 'AEAZ-0142',
        'BDDU-0346', 'BDDU-0347', 'BDDU-0349', 'BDDU-0350',
        'MXMC-0006', 'NGIL-0309'
    ]
    IC_dfs = []
    for filename in os.listdir(obs_dir):
        if filename.endswith('.csv'):
            master_data = pd.read_csv(os.path.join(obs_dir, filename), encoding='ISO-8859-1')
            IC_columns = ['FilterID', 'start_year', 'start_month', 'start_day', 'Mass_type', 'mass_ug', 'Volume_m3',
                            'S_XRF_ng', 'IC_SO4_ug', 'Flags']
            if all(col in master_data.columns for col in IC_columns):
                # Select the specified columns
                master_data.columns = master_data.columns.str.strip()
                IC_df = master_data[IC_columns].copy()
                # Exclude specific FilterID values
                IC_df = IC_df[~IC_df['FilterID'].isin(excluded_filters)]
                # Select PM2.5
                IC_df['Mass_type'] = pd.to_numeric(IC_df['Mass_type'], errors='coerce')
                IC_df = IC_df.loc[IC_df['Mass_type'] == 1]
                # Convert the relevant columns to numeric
                IC_df[['S_XRF_ng', 'IC_SO4_ug', 'mass_ug', 'Volume_m3', 'start_year']] = IC_df[
                    ['S_XRF_ng', 'IC_SO4_ug', 'mass_ug', 'Volume_m3', 'start_year']].apply(pd.to_numeric, errors='coerce')
                # Select year 2019 - 2023
                # IC_df = IC_df[IC_df['start_year'].isin([2019, 2020, 2021, 2022, 2023])]
                # Drop rows with NaN values
                IC_df = IC_df.dropna(subset=['start_year', 'Volume_m3', 'S_XRF_ng', 'IC_SO4_ug'])
                IC_df = IC_df[IC_df['Volume_m3'] > 0]
                IC_df = IC_df[IC_df['S_XRF_ng'] > 0]
                IC_df = IC_df[IC_df['IC_SO4_ug'] > 0]
                # Calculate BC concentrations, fractions, and BC/Sulfate
                IC_df['S_XRF_(ug/m3)'] = IC_df['S_XRF_ng'] / (IC_df['Volume_m3'] * 1000)
                IC_df['IC_SO4_(ug/m3)'] = IC_df['IC_SO4_ug'] / IC_df['Volume_m3']
                IC_df['PM25'] = IC_df['mass_ug'] / IC_df['Volume_m3']
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
    with pd.ExcelWriter(os.path.join(out_dir, "SO4_IC_XRF_SPARTAN.xlsx"), engine='openpyxl', mode='w') as writer:
        obs_df.to_excel(writer, sheet_name='All', index=False)

    # Writ summary statistics to Excel
    site_counts = obs_df.groupby('Site')['FilterID'].count()
    for site, count in site_counts.items():
        print(f"{site}: {count} rows")
    summary_df = obs_df.groupby(['Country', 'City'])['IC_SO4_(ug/m3)'].agg(['count', 'mean', 'median', 'std'])
    summary_df['stderr'] = summary_df['std'] / np.sqrt(summary_df['count']).pow(0.5)
    summary_df.rename(columns={'count': 'num_obs', 'mean': 'bc_mean', 'median': 'bc_median', 'std': 'bc_stdv', 'stderr': 'bc_stderr'},
        inplace=True)
    with pd.ExcelWriter(os.path.join(out_dir, "SO4_IC_XRF_SPARTAN.xlsx"), engine='openpyxl', mode='a') as writer:
        summary_df.to_excel(writer, sheet_name='Summary', index=True)
################################################################################################
# plot XRF S vs IC SO4, color cell by no. of pairs
################################################################################################
# Read the file
merged_df = pd.read_excel(os.path.join(out_dir, 'SO4_IC_XRF_SPARTAN.xlsx'))
# Rename to simplify coding
merged_df.rename(columns={"S_XRF_(ug/m3)": "XRF"}, inplace=True)
merged_df.rename(columns={"IC_SO4_(ug/m3)": "IC"}, inplace=True)
merged_df.rename(columns={"City": "city"}, inplace=True)

# Create figure and axes objects
fig, ax = plt.subplots(figsize=(7, 6))

# Create a 2D histogram to divide the area into squares and count data points in each square
hist, xedges, yedges = np.histogram2d(merged_df['XRF'], merged_df['IC'], bins=300)

# Determine the color for each square based on the number of pairs
colors = np.zeros_like(hist)
for i in range(len(hist)):
    for j in range(len(hist[i])):
        pairs = hist[i][j]
        colors[i][j] = pairs

# Define the custom color scheme gradient
colors = [(1, 1, 1), (0, 0.5, 1), (0, 1, 0), (1, 1, 0), (1, 0.5, 0), (1, 0, 0)]

# Create a custom colormap using the gradient defined
cmap = mcolors.LinearSegmentedColormap.from_list('custom_gradient', colors)


# Plot the 2D histogram with the specified color scheme
sns.set(font='Arial')
scatterplot = plt.imshow(hist.T, extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], cmap=cmap, origin='lower')

# Display the original data points as a scatter plot
# plt.scatter(merged_df['XRF'], merged_df['IC'], color='black', s=2, alpha=0.5)

# Set title, xlim, ylim, ticks, labels
plt.xlim([merged_df['IC'].min()-0.15, 10])
plt.ylim([merged_df['IC'].min()-0.1, 10])
plt.xticks([0, 2, 4, 6, 8, 10], fontname='Arial', size=18)
plt.yticks([0, 2, 4, 6, 8, 10], fontname='Arial', size=18)
ax.tick_params(axis='x', direction='out', width=1, length=5)
ax.tick_params(axis='y', direction='out', width=1, length=5)

# Add 1:1 line with grey dash
x = merged_df['IC']
y = merged_df['IC']
plt.plot([merged_df['IC'].min(), merged_df['IC'].max()], [3*merged_df['IC'].min(), 3*merged_df['IC'].max()], color='grey', linestyle='--', linewidth=1)

# Add number of data points to the plot
num_points = len(merged_df)
plt.text(0.6, 0.7, f'N = {num_points}', transform=ax.transAxes, fontsize=18)

# Perform linear regression with NaN handling
mask = ~np.isnan(merged_df['XRF']) & ~np.isnan(merged_df['IC'])
slope, intercept, r_value, p_value, std_err = stats.linregress(merged_df['XRF'][mask], merged_df['IC'][mask])
# Check for NaN in results
if np.isnan(slope) or np.isnan(intercept) or np.isnan(r_value):
    print("Linear regression results contain NaN values. Check the input data.")
else:
    # Add linear regression line and text
    # sns.regplot(x='XRF', y='IC', data=merged_df, scatter=False, ci=None, line_kws={'color': 'k', 'linestyle': '-', 'linewidth': 1})
    # Change the sign of the intercept for display
    intercept_display = abs(intercept)  # Use abs() to ensure a positive value
    intercept_sign = '-' if intercept < 0 else '+'  # Determine the sign for display

    # Update the text line with the adjusted intercept
    plt.text(0.6, 0.76, f"y = {slope:.2f}x {intercept_sign} {intercept_display:.2f}\n$r^2$ = {r_value ** 2:.2f}",
             transform=plt.gca().transAxes, fontsize=18)

plt.xlabel('XRF Sulfur (µg/m$^3$)', fontsize=18, color='black', fontname='Arial')
plt.ylabel('IC Sulfate (µg/m$^3$)', fontsize=18, color='black', fontname='Arial')

# Create the colorbar and specify font properties
cbar_ax = fig.add_axes([0.65, 0.2, 0.02, 0.4])
cbar = plt.colorbar(label='Number of Pairs', cax=cbar_ax)
cbar.ax.set_ylabel('Number of Pairs', fontsize=14, fontname='Arial')
cbar.outline.set_edgecolor('black')
cbar.outline.set_linewidth(1)
cbar.set_ticks([0, 10, 20, 30, 40], fontname='Arial', fontsize=14)

# show the plot
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "Sulfate_Comparison_IC_XRF.svg"), format="SVG", dpi=300)
plt.show()
