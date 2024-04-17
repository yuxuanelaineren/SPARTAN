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

# Set the directory path
HIPS_dir = '/Volumes/rvmartin/Active/SPARTAN-shared/Analysis_Data/Master_files/'
UV_dir = '/Volumes/rvmartin/Active/ren.yuxuan/BC_Comparison/SPARTAN_BC/BC_UV-Vis_SPARTAN/'
out_dir = '/Volumes/rvmartin/Active/ren.yuxuan/BC_Comparison/SPARTAN_BC/'
################################################################################################
# Create scatter plot for all sites
################################################################################################
# Read the file
compr_df = pd.read_excel(os.path.join(out_dir, 'BC_HIPS_UV-Vis_SPARTAN.xlsx'))

# Drop rows where f_BC is greater than 1
compr_df = compr_df.loc[compr_df['f_BC_UV-Vis'] <= 1]
compr_df.rename(columns={"BC_HIPS_(ug/m3)": "HIPS"}, inplace=True)
compr_df.rename(columns={"BC_UV-Vis_(ug/m3)": "UV-Vis"}, inplace=True)
compr_df.rename(columns={"City": "city"}, inplace=True)

# Print the names of each city
unique_cities = compr_df['city'].unique()
for city in unique_cities:
    print(f"City: {city}")

# Classify 'city' based on 'region'
def get_region_for_city(city):
    for region, cities in region_mapping.items():
        if city in cities:
            return region
    print(f"Region not found for city: {city}")
    return None

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

region_markers = {
    'North America': ['o', 'o', 'o', 'p', 'H', '*'],
    'Australia': ['o', '^', 's', 'p', 'H', '*'],
    'East Asia': ['o', '^', 's', 'p', 'H', '*'],
    'Central Asia': ['o', '^', 's', 'p', 'H', '*'],
    'South Asia': ['o', '^', 's', 'p', 'H', '*'],
    'Africa': ['o', 'o', 'o', 'o', 'o', 'o'],
    'South America': ['o', '^', 's', 'p', 'H', '*'],
}
# Create an empty list to store the city_marker for each city
city_marker = []
city_marker_match = []

def map_city_to_marker(city):
    for region, cities in region_mapping.items():
        if city in cities:
            if region == 'North America':
                return 'd'
            elif region == 'Australia':
                return '*'
            elif region == 'East Asia':
                return '^'
            elif region == 'Central Asia':
                return 'p'
            elif region == 'South Asia':
                return 's'
            elif region == 'Africa':
                return 'o'
            elif region == 'South America':
                return 'o'
            else:
                return 'o'  # Default marker style
    print(f"City not found in any region: {city}")
    return 'o'

# Iterate over each unique city and map it to a marker
for city in unique_cities:
    marker = map_city_to_marker(city)
    if marker is not None:
        city_marker.append(marker)
        city_marker_match.append({'city': city, 'marker': marker})

# Create figure and axes objects
fig, ax = plt.subplots(figsize=(8, 6))
# Create scatter plot
sns.set(font='Arial')
scatterplot = sns.scatterplot(x='HIPS', y='UV-Vis', data=compr_df, hue='city',  s=80, alpha=1, edgecolor='k', style='city',  markers=city_marker)

# Set title, xlim, ylim, ticks, labels
plt.xlim([compr_df['HIPS'].min()-0.5, 35])
plt.ylim([compr_df['HIPS'].min()-0.5, 35])
plt.xticks([0, 10, 20, 30], fontname='Arial', size=18)
plt.yticks([0, 10, 20, 30], fontname='Arial', size=18)
scatterplot.tick_params(axis='x', direction='out', width=1, length=5)
scatterplot.tick_params(axis='y', direction='out', width=1, length=5)

# Add 1:1 line with grey dash
x = compr_df['HIPS']
y = compr_df['HIPS']
plt.plot([compr_df['HIPS'].min(), 11.5], [compr_df['HIPS'].min(), 11.5],
         color='grey', linestyle='--', linewidth=1)

# Perform linear regression for all segments
mask = ~np.isnan(compr_df['HIPS']) & ~np.isnan(compr_df['UV-Vis'])
slope, intercept, r_value, p_value, std_err = stats.linregress(compr_df['HIPS'][mask], compr_df['UV-Vis'][mask])

# Plot regression lines
sns.regplot(x='HIPS', y='UV-Vis', data=compr_df[mask],
            scatter=False, ci=None, line_kws={'color': 'black', 'linestyle': '-', 'linewidth': 1.5}, ax=ax)

# Add text with linear regression equations and other statistics
intercept_display = abs(intercept)
intercept_sign = '-' if intercept < 0 else '+'
plt.text(0.05, 0.66, f'y = {slope:.2f}x {intercept_sign} {intercept_display:.2f}\n$r^2$ = {r_value ** 2:.2f}',
         transform=scatterplot.transAxes, fontsize=18, color='black')

# Add the number of data points for each segment
num_points = mask.sum()
plt.text(0.05, 0.6, f'N = {num_points}', transform=scatterplot.transAxes, fontsize=18, color='black')

# Set labels
plt.xlabel('HIPS Black Carbon (µg/m$^3$)', fontsize=18, color='black', fontname='Arial')
plt.ylabel('UV-Vis Black Carbon (µg/m$^3$)', fontsize=18, color='black', fontname='Arial')

# Show the plot
plt.tight_layout()
# plt.savefig(out_dir + 'Fig_b_r_Scatter_{}_{}_{}_Sim_vs_SPARTAN_{}_{:02d}_AnnualMean.svg'.format(cres, inventory, deposition, species, year), dpi=300)

plt.show()