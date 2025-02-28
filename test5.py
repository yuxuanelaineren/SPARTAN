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

# Set the directory path
FTIR_dir = '/Volumes/rvmartin/Active/ren.yuxuan/Mass_Reconstruction/'
Residual_dir = '/Volumes/rvmartin/Active/SPARTAN-shared/Public_Data/RCFM/'
OMOC_dir = '/Volumes/rvmartin/Active/ren.yuxuan/RCFM/FTIR_OC_OMOC_Residual/OM_OC/'
site_dir = '/Volumes/rvmartin/Active/SPARTAN-shared/Site_Sampling/'
out_dir = '/Volumes/rvmartin/Active/ren.yuxuan/Mass_Reconstruction/'
Colocation_dir = '/Volumes/rvmartin/Active/ren.yuxuan/Mass_Reconstruction/Co-location/'
################################################################################################
# Create scatter plot for Residual vs OM, colored by region
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
def map_city_to_color(city):
    for region, cities in region_mapping.items():
        if city in cities:
            city_index = cities.index(city) % len(region_colors[region])
            assigned_color = region_colors[region][city_index]
            print(f"City: {city}, Region: {region}, Assigned Color: {assigned_color}")
            return assigned_color
    print(f"City not found in any region: {city}")
    return (0, 0, 0)
def map_city_to_marker(city):
    for region, cities in region_mapping.items():
        if city in cities:
            city_index = cities.index(city)
            assigned_marker = region_markers[region][city_index % len(region_markers[region])]
            return assigned_marker
    return None
# Read the file
compr_df = pd.read_excel(os.path.join(out_dir, 'FT-IR_OM_OC_vs_Residual_Chris_vs_sim_OMOC.xlsx'), sheet_name='All')
compr_df['Residual'] = compr_df['RM_dry']
compr_df['OM'] = compr_df['FTIR_OM']
compr_df = compr_df[compr_df['OM'] > 0]
compr_df = compr_df[compr_df['batch'].isin(['batch2_2022_06_batch3_2023_03', 'batch4_2024_03'])]
compr_df = compr_df[compr_df['Site'].isin(['AEAZ', 'ILHA', 'ILNZ'])]
# compr_df['Ratio'] = compr_df['OM'] / compr_df['FTIR_OC']
# compr_df['OM'] = compr_df.apply(lambda row: row['OM'] if row['Ratio'] < 2.5 else row['FTIR_OC']*2.5, axis=1)

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

# Define custom palette for each region with 5 shades for each color, https://rgbcolorpicker.com/0-1
region_colors = {
    'North America': [
        (0, 0, 0.6),  (0, 0, 1), (0, 0.27, 0.8), (0.4, 0.5, 0.9), (0.431, 0.584, 1), (0.7, 0.76, 0.9), (0.85, 0.9, 1)
    ],  # Blue shades
    # 'Central Asia': [
    #     (0.58, 0.1, 0.81), (0.66, 0.33, 0.83), (0.9, 0.4, 1), (0.73, 0.44, 0.8), (0.8, 0.55, 0.77), (0.88, 0.66, 0.74)
    # ],  # Purple shades
    'Central Asia': [
        'black', 'blue','red'
    ],  # Purple shades
    'Australia': [
        (0.6, 0.4, 0.2)
    ],  # Brown
    'East Asia': [
        (0, 0.5, 0), (0, 0.8, 0), (0, 1, 0), (0.56, 0.93, 0.56), (0.8, 0.9, 0.8)
    ],  # Green shades
    'South Asia': [
        (0.5, 0, 0), (0.8, 0, 0), (1, 0, 0), (1, 0.4, 0.4), (0.9, 0.6, 0.6)
    ],  # Red shades
    'Africa': [
        (1, 0.4, 0), (1, 0.6, 0.14), (1, 0.63, 0.48), (1, 0.85, 0.73), (1, 0.96, 0.85)
    ], # Orange shades
    'South America': [
        (1, 0.16, 0.827), (1, 0.42, 0.70), (0.8, 0.52, 0.7), (0.961, 0.643, 0.804), (1, 0.64, 0.64), (1, 0.76, 0.48)
    ]  # Pink shades
}

# Create an empty list to store the city_palette for each city
city_palette = []
city_color_match = []
# Iterate over each unique city and map it to a gradient
for city in unique_cities:
    city_color = map_city_to_color(city)
    if city_color is not None:
        city_palette.append(city_color)
        city_color_match.append({'city': city, 'color': city_color})  # Store both city name and color
print("City Palette:", city_palette)

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

# Calculate mean and standard error grouped by City
summary_stats = compr_df.groupby('City').agg(
    OM_mean=('OM', 'mean'),
    OM_se=('OM', lambda x: x.std() / np.sqrt(len(x))),
    Residual_mean=('Residual', 'mean'),
    Residual_se=('Residual', lambda x: x.std() / np.sqrt(len(x)))
).reset_index()

# Create figure and axes objects
fig, ax = plt.subplots(figsize=(8, 6))

# Create scatter plot with white background, black border, and no grid
sns.set(font='Arial')
scatterplot = sns.scatterplot(x='OM', y='Residual', data=compr_df, hue='City', palette=city_palette, s=20, alpha=1, edgecolor='k', style='City',  markers=city_marker)
scatterplot.set_facecolor('white')  # set background color to white
border_width = 1
for spine in scatterplot.spines.values():
    spine.set_edgecolor('black')  # set border color to black
    spine.set_linewidth(border_width)  # set border width
scatterplot.grid(False)  # remove the grid

# Sort the unique_cities list based on their appearance in region_mapping
unique_cities_sorted = sorted(unique_cities, key=get_city_index)

# Create legend with custom order
sorted_city_color_match = sorted(city_color_match, key=lambda x: (
    list(region_mapping.keys()).index(get_region_for_city(x['city'])),
    region_mapping[get_region_for_city(x['city'])].index(x['city'])
))

# Create legend handles with both color and marker for each city
legend_handles = []
for city_info in sorted_city_color_match:
    city = city_info['city']
    color = city_info['color']
    marker = map_city_to_marker(city)
    if marker is not None:
        handle = plt.Line2D([0], [0], marker=marker, color=color, linestyle='', markersize=6, label=city)
        legend_handles.append(handle)

# Create legend with custom handles
legend = plt.legend(handles=legend_handles, facecolor='white', bbox_to_anchor=(1.03, 0.50), loc='center left', fontsize=11.5)
legend.get_frame().set_edgecolor('black')

# Set title, xlim, ylim, ticks, labels
plt.title('FT-IR OM vs updated Residual', fontsize=18, fontname='Arial', y=1.03)
# plt.title('Imposing OM/OC = 2.5 Threshold', fontsize=18, fontname='Arial', y=1.03)
plt.xlim([-8, 52])
plt.ylim([-8, 52])
plt.xticks([0, 10, 20, 30, 40, 50], fontname='Arial', size=18)
plt.yticks([0, 10, 20, 30, 40, 50], fontname='Arial', size=18)
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
# # Plot regression lines
# sns.regplot(x='OM', y='Residual', data=compr_df[mask],
#             scatter=False, ci=None, line_kws={'color': 'black', 'linestyle': '-', 'linewidth': 1.5}, ax=ax)

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
    # nmd = np.mean((sim - obs) / obs) * 100  # Percentage
    nmd = np.sum(sim - obs) / np.sum(obs) * 100
    # Calculate NRMSD
    rmsd = np.sqrt(np.mean((sim - obs) ** 2))
    mean_obs = np.mean(obs)
    nrmsd = (rmsd / mean_obs) * 100  # Percentage
    return {'NMB (%)': nmd, 'NRMSD (%)': nrmsd}
# Perform the calculations for the entire dataset
nmd_nrmsd_results = calculate_nmd_and_nrmsd(compr_df, obs_col='OM', sim_col='Residual')
nmd = nmd_nrmsd_results['NMB (%)']
nrmsd = nmd_nrmsd_results['NRMSD (%)']

# Add text with linear regression equations and other statistics
intercept_display = abs(intercept)
intercept_sign = '-' if intercept < 0 else '+'
num_points = mask.sum()
plt.text(0.05, 0.70, f'y = {slope:.3f}x {intercept_sign} {intercept_display:.1f}\n$r^2$ = {r_value ** 2:.3f}\nN = {num_points}\nNMB = {nmd:.0f}%\nNRMSD = {nrmsd:.0f}%',
         transform=scatterplot.transAxes, fontsize=16, color='black')

# Set labels
plt.xlabel('FT-IR OM (µg/m$^3$)', fontsize=18, color='black', fontname='Arial')
plt.ylabel('Residual (µg/m$^3$)', fontsize=18, color='black', fontname='Arial')

# Show the plot
plt.tight_layout()
# plt.savefig(out_dir + 'FTIR_OM_vs_Residual_AbuDhabi_Haifa_Rehovot.svg', dpi=300)

plt.show()