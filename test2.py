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

# Set the directory path
FTIR_dir = '/Volumes/rvmartin/Active/ren.yuxuan/Mass_Reconstruction/'
Residual_dir = '/Volumes/rvmartin/Active/SPARTAN-shared/Public_Data/RCFM/'
OMOC_dir = '/Volumes/rvmartin/Active/ren.yuxuan/RCFM/FTIR_OC_OMOC_Residual/OM_OC/'
site_dir = '/Volumes/rvmartin/Active/SPARTAN-shared/Site_Sampling/'
out_dir = '/Volumes/rvmartin/Active/ren.yuxuan/Mass_Reconstruction/'
################################################################################################
# Create scatter plot for Residual vs OM, colored by dust
################################################################################################
def get_city_index(city):
    for region, cities in region_mapping.items():
        if city in cities:
            return cities.index(city)
    return float('inf')  # If city is not found, place it at the end
def get_region_for_city(city):
    for region, cities in region_mapping.items():
        if city in cities:
            return region
    print(f"Region not found for city: {city}")
    return None
def map_city_to_marker(city):
    for region, cities in region_mapping.items():
        if city in cities:
            city_index = cities.index(city)
            assigned_marker = region_markers[region][city_index % len(region_markers[region])]
            return assigned_marker
    return None
# Read the file
compr_df = pd.read_excel(os.path.join(out_dir, 'FT-IR_OM_OC_Residual_Chris.xlsx'), sheet_name='OM_OC_batch234_Residual')
compr_df = compr_df[compr_df['batch'].isin(['batch2_2022_06_batch3_2023_03', 'batch4_2024_03'])]
compr_df.rename(columns={'RM_dry': 'Residual'}, inplace=True)
compr_df['DustRatio'] = compr_df['Soil'] / compr_df['PM2.5']
compr_df['Residual'] = compr_df.apply(lambda row: row['Residual'] if row['DustRatio'] < 0.4 else row['Residual'] + row['PM2.5']*(row['DustRatio'] - 0.4), axis=1)
# compr_df = compr_df[compr_df['OM'] > 0]
# compr_df = compr_df[compr_df['OC'] > 0]
# compr_df = compr_df[compr_df['Residual'] > 0]
# compr_df['Ratio'] = compr_df['OM'] / compr_df['OC']
# compr_df['OM'] = compr_df.apply(lambda row: row['OM'] if row['Ratio'] < 2.5 else row['OC']*2.5, axis=1)

# Step 1: Calculate monthly mean and standard error for each city
monthly_stats = compr_df.groupby(['City', 'start_month']).agg(
    OM_monthly_mean=('OM', 'mean'),
    OM_monthly_se=('OM', lambda x: x.std() / np.sqrt(len(x))),
    Residual_monthly_mean=('Residual', 'mean'),
    Residual_monthly_se=('Residual', lambda x: x.std() / np.sqrt(len(x)))
).reset_index()

# Step 2: Calculate annual statistics (mean and standard error) from monthly results
annual_stats = monthly_stats.groupby(['City']).agg(
    OM_mean=('OM_monthly_mean', 'mean'),
    OM_se=('OM_monthly_se', 'mean'),
    Residual_mean=('Residual_monthly_mean', 'mean'),
    Residual_se=('Residual_monthly_se', 'mean')
).reset_index()
# Rename the annual statistics DataFrame as summary_df and sort by OM_mean
summary_df = annual_stats.sort_values(by='OM_mean')

# Print the names of each city
unique_cities = compr_df['City'].unique()
for city in unique_cities:
    print(f"City: {city}")

# Classify 'city' based on 'region'
region_mapping = {
    'North America': ['Downsview', 'Halifax', 'Kelowna', 'Lethbridge', 'Sherbrooke', 'Baltimore', 'Bondville', 'Mammoth Cave', 'Norman', 'Pasadena', 'Fajardo', 'Mexico City'],
    'Australia': ['Melbourne'],
    'East Asia': ['Beijing', 'Seoul', 'Ulsan', 'Kaohsiung', 'Taipei'],
    'Central Asia': ['Abu Dhabi', 'Haifa', 'Rehovot'],
    'South Asia': ['Dhaka', 'Bandung', 'Delhi', 'Kanpur', 'Manila', 'Singapore', 'Hanoi'],
    'Africa': ['Bujumbura', 'Addis Ababa', 'Ilorin', 'Johannesburg', 'Pretoria'],
    'South America': ['Buenos Aires', 'Santiago', 'Palmira'],
}
region_mapping = {region: [city for city in cities if city in unique_cities] for region, cities in region_mapping.items()}

# Define custom palette for each region with 5 shades for each color
region_markers = {
    'North America': ['o', '^', 's', 'p', 'H', '*'],
    'Australia': ['o', '^', 's', 'p', 'H', '*'],
    'East Asia': ['o', '^', 's', 'p', 'H', '*'],
    'Central Asia': ['o', '^', 's', 'p', 'H', '*'],
    'South Asia': ['o', '^', 's', 'p', 'H', '*'],
    'Africa': ['o', '^', 's', 'p', 'H', '*'],
    'South America': ['o', '^', 's', 'p', 'H', '*'],
}

# Create an empty list to store the city_marker for each city
city_marker = []
city_marker_match = []

# Iterate over each unique city and map it to a marker
for city in unique_cities:
    marker = map_city_to_marker(city)
    if marker is not None:
        city_marker.append(marker)
        city_marker_match.append({'city': city, 'marker': marker})

print("City Marker:", city_marker)

# Create a colormap for DustRatio
blue_colors = [(0, 0, 0.6), (0, 0, 1), (0, 0.27, 0.8), (0.4, 0.5, 0.9), (0.431, 0.584, 1), (0.7, 0.76, 0.9)]
red_colors = [(0.9, 0.6, 0.6), (1, 0.4, 0.4), (1, 0, 0), (0.8, 0, 0), (0.5, 0, 0)]
colors = blue_colors + [(1, 1, 1)] + red_colors
cmap = mcolors.LinearSegmentedColormap.from_list("custom_cmap", colors)
# nnorm = plt.Normalize(vmin=0, vmax=0.4)
norm = plt.Normalize(vmin=compr_df['DustRatio'].min(), vmax=compr_df['DustRatio'].max())

# Create figure and axes objects
fig, ax = plt.subplots(figsize=(8, 6))

# Create scatter plot with white background, black border, and no grid
sns.set(font='Arial')
scatterplot = sns.scatterplot(x='OM', y='Residual', data=compr_df, hue='DustRatio', palette=cmap, s=20, alpha=1, edgecolor='k', style='City',  markers=city_marker)
scatterplot.set_facecolor('white')  # set background color to white
border_width = 1
for spine in scatterplot.spines.values():
    spine.set_edgecolor('black')  # set border color to black
    spine.set_linewidth(border_width)  # set border width
scatterplot.grid(False)  # remove the grid

# Sort the unique_cities list based on their appearance in region_mapping
unique_cities_sorted = sorted(unique_cities, key=get_city_index)

# Sort the legend labels based on x-axis order
handles, labels = ax.get_legend_handles_labels()
ordered_handles_labels = sorted(zip(summary_df['OM_mean'], handles, labels), key=lambda x: x[0])
_, ordered_handles, ordered_labels = zip(*ordered_handles_labels)

# Add the legend with ordered labels
legend = ax.legend(ordered_handles, ordered_labels, markerscale=0.7, prop={'family': 'Arial','size': 11.5}, loc='best', bbox_to_anchor=(1.05, 1), borderaxespad=0.)
legend.get_frame().set_edgecolor('black')

# Set title, xlim, ylim, ticks, labels
plt.title('FT-IR OM vs updated Residual, Dust Fraction < 0.4', fontsize=18, fontname='Arial', y=1.03)
# plt.title('Imposing OM/OC = 2.5 Threshold', fontsize=18, fontname='Arial', y=1.03)
plt.xlim([-9, 80])
plt.ylim([-9, 80])
plt.xticks([0, 20, 40, 60, 80], fontname='Arial', size=18)
plt.yticks([0, 20, 40, 60, 80], fontname='Arial', size=18)
scatterplot.tick_params(axis='x', direction='out', width=1, length=5)
scatterplot.tick_params(axis='y', direction='out', width=1, length=5)

# Add 1:1 line with grey dash
x = compr_df['OM']
y = compr_df['OM']
plt.plot([-10, 80], [-10, 80], color='grey', linestyle='--', linewidth=1)

# Define the range of x-values for the two segments
x_range = [compr_df['OM'].min(), compr_df['OM'].max()]
# Perform linear regression for all segments
mask = (compr_df['OM'] >= x_range[0]) & (compr_df['OM'] <= x_range[1])
slope, intercept, r_value, p_value, std_err = stats.linregress(compr_df['OM'][mask], compr_df['Residual'][mask])
# Plot regression lines
sns.regplot(x='OM', y='Residual', data=compr_df[mask],
            scatter=False, ci=None, line_kws={'color': 'black', 'linestyle': '-', 'linewidth': 1.5}, ax=ax)

# Add columns for normalized mean difference (NMD) and normalized root mean square difference (NRMSD)
def calculate_nmd_and_nrmsd(df, obs_col, sim_col):
    """
    Calculate normalized mean difference (NMD) and normalized root mean square difference (NRMSD).

    Args:
        df (pd.DataFrame): DataFrame containing observation and simulation columns.
        obs_col (str): Column name for observations.
        sim_col (str): Column name for simulations.

    Returns:
        dict: Dictionary containing NMD and NRMSD values.
    """
    obs = df[obs_col].values
    sim = df[sim_col].values
    # Remove rows with NaN values
    valid_indices = ~np.isnan(obs) & ~np.isnan(sim)
    obs = obs[valid_indices]
    sim = sim[valid_indices]
    # Check if there are valid data points
    if len(obs) == 0:
        return {'NMD (%)': np.nan, 'NRMSD (%)': np.nan}
    # Calculate NMD
    nmd = np.mean((sim - obs) / obs) * 100  # Percentage
    # Calculate NRMSD
    rmsd = np.sqrt(np.mean((sim - obs) ** 2))
    mean_obs = np.mean(obs)
    nrmsd = (rmsd / mean_obs) * 100  # Percentage
    return {'NMD (%)': nmd, 'NRMSD (%)': nrmsd}
# Perform the calculations for the entire dataset
nmd_nrmsd_results = calculate_nmd_and_nrmsd(compr_df, obs_col='OM', sim_col='Residual')
nmd = nmd_nrmsd_results['NMD (%)']
nrmsd = nmd_nrmsd_results['NRMSD (%)']

# Add text with linear regression equations and other statistics
intercept_display = abs(intercept)
intercept_sign = '-' if intercept < 0 else '+'
num_points = mask.sum()
plt.text(0.05, 0.65, f'y = {slope:.2f}x {intercept_sign} {intercept_display:.1f}\n$r^2$ = {r_value ** 2:.2f}\nN = {num_points}\nNMD = {nmd:.1f}%\nNRMSD = {nrmsd:.0f}%',
         transform=scatterplot.transAxes, fontsize=18, color='black')

# Set labels
plt.xlabel('FT-IR Organic Matter (µg/m$^3$)', fontsize=18, color='black', fontname='Arial')
plt.ylabel('Residual (µg/m$^3$)', fontsize=18, color='black', fontname='Arial')

# Create an axis for the colorbar (cax)
cax = fig.add_axes([0.63, 0.2, 0.015, 0.3])  # Position: [x, y, width, height]
cbar = fig.colorbar(
    plt.cm.ScalarMappable(cmap=cmap, norm=norm),
    cax=cax,  # Set the colorbar to the specified axis
)
cbar.set_label('Dust Fraction', fontsize=14, fontname='Arial', labelpad=5)
cbar.ax.tick_params(labelsize=12, width=1)
cbar.outline.set_edgecolor('black')
cbar.outline.set_linewidth(1)
cbar.set_ticks([0, 0.25, 0.5, 0.75, 1], fontname='Arial', fontsize=14)

# Show the plot
plt.tight_layout()
# plt.savefig(out_dir + 'OM_B234_vs_updatedResidual_ColorByDust<0.4.svg', dpi=300)

plt.show()