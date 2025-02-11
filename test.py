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
import matplotlib.lines as mlines
from matplotlib.lines import Line2D
from matplotlib.patches import Circle
from matplotlib.patches import Polygon

cres = 'C360'
year = 2019
species = 'BC'
inventory = 'CEDS'
deposition = 'noLUO'

# Set the directory path
sim_dir = '/Volumes/rvmartin2/Active/Shared/dandan.z/GCHP-v13.4.1/{}-CEDS01-fixed-vert-{}-output/monthly/'.format(cres.lower(), deposition) # CEDS, C360, noLUO
# sim_dir = '/Volumes/rvmartin2/Active/Shared/dandan.z/GCHP-v13.4.1/{}-EDGARv61-vert-{}-output/monthly/'.format(cres.lower(), deposition) # EDGAR, noLUO
# sim_dir = '/Volumes/rvmartin2/Active/Shared/dandan.z/GCHP-v13.4.1/{}-HTAPv3-vert-{}-output/monthly/'.format(cres.lower(), deposition) # HTAP, noLUO
# sim_dir = '/Volumes/rvmartin2/Active/Shared/dandan.z/GCHP-v13.4.1/{}-CEDS01-fixed-vert-{}-CSwinds-output/monthly/'.format(cres.lower(), deposition) # CEDS, C3720, noLUO
sim_dir = '/Volumes/rvmartin2/Active/Shared/dandan.z/GCHP-v13.4.1/{}-CEDS01-fixed-vert-{}-output/monthly/'.format(cres.lower(), deposition) # CEDS, C180, noLUO, GEOS-FP
obs_dir = '/Volumes/rvmartin/Active/SPARTAN-shared/Analysis_Data/Master_files/'
site_dir = '/Volumes/rvmartin/Active/SPARTAN-shared/Site_Sampling/'
out_dir = '/Volumes/rvmartin/Active/ren.yuxuan/BC_Comparison/{}_{}_{}_{}/'.format(cres.lower(), inventory, deposition, year)
support_dir = '/Volumes/rvmartin/Active/ren.yuxuan/BC_Comparison/supportData/'
otherMeas_dir = '/Volumes/rvmartin/Active/ren.yuxuan/BC_Comparison/otherMeasurements/'
################################################################################################
# Create scatter plot: sim vs meas, color blue and red with two lines, MAC 7 and 13 indicated as dots, Beijing grey out
################################################################################################
# Read the file
compr_df = pd.read_excel(os.path.join(out_dir, '{}_{}_{}_vs_SPARTAN_{}_{}_Summary.xlsx'.format(cres, inventory, deposition, species, year)), sheet_name='Annual')
compr_df['obs'] = 1 * compr_df['obs'] # 1 for MAC=10m2/g, 10/7 for MAC=7m2/g, 10/13 for MAC=13m2/g
compr_df['obs_high'] = (10 / 7) * compr_df['obs']
compr_df['obs_low'] = (10 / 13) * compr_df['obs']
# compr_df['obs_se'] = 1 * compr_df['obs_se']

# Print the names of each city
unique_cities = compr_df['city'].unique()
for city in unique_cities:
    print(f"City: {city}")

# Define the range of x-values for the two segments
x_range_1 = [compr_df['obs'].min(), 1.35*1] # 1 for MAC=10m2/g, 10/7 for MAC=7m2/g, 10/13 for MAC=13m2/g
x_range_1_high = [compr_df['obs_high'].min(), 1.35*(10 / 7)]
x_range_1_low = [compr_df['obs_low'].min(), 1.35*(10 / 13)]
x_range_2 = [1.4*1, compr_df['obs'].max()]
x_range_2_high = [1.4*(10 / 7), compr_df['obs_high'].max()]
x_range_2_low = [1.4*(10 / 13), compr_df['obs_low'].max()]

# Define custom blue and red colors
blue_colors = [(0.7, 0.76, 0.9),  (0.431, 0.584, 1), (0.4, 0.5, 0.9), (0, 0.27, 0.8),  (0, 0, 1), (0, 0, 0.6)]
red_colors = [(0.9, 0.6, 0.6), (1, 0.4, 0.4), (1, 0, 0), (0.8, 0, 0), (0.5, 0, 0)]
blue_cmap = LinearSegmentedColormap.from_list('blue_cmap', blue_colors)
red_cmap = LinearSegmentedColormap.from_list('red_cmap', red_colors)

# Map city to color based on observed values
def map_city_to_color(city, obs):
    if city == 'Beijing':  # Mark Beijing grey
        return 'grey'
    elif x_range_1[0] <= obs <= x_range_1[1]:
        index_within_range = sorted(compr_df[compr_df['obs'].between(x_range_1[0], x_range_1[1])]['obs'].unique()).index(obs)
        obs_index = index_within_range / (len(compr_df[compr_df['obs'].between(x_range_1[0], x_range_1[1])]['obs'].unique()) - 1)
        return blue_cmap(obs_index)
    elif x_range_2[0] <= obs <= x_range_2[1]:
        index_within_range = sorted(compr_df[compr_df['obs'].between(x_range_2[0], x_range_2[1])]['obs'].unique()).index(obs)
        obs_index = index_within_range / (len(compr_df[compr_df['obs'].between(x_range_2[0], x_range_2[1])]['obs'].unique()) - 1)
        return red_cmap(obs_index)
    else:
        return 'black'

# city_palette = [map_city_to_color(city, obs) for city, obs in zip(compr_df['city'], compr_df['obs'])]
city_palette = [map_city_to_color(city, obs) if city != 'Singapore' else blue_cmap(0.5)
                for city, obs in zip(compr_df['city'], compr_df['obs'])]
# Sort the cities in the legend based on observed values
sorted_cities = sorted(compr_df['city'].unique(), key=lambda city: compr_df.loc[compr_df['city'] == city, 'obs'].iloc[0])

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
sns.set(font='Arial')

# Add 1:1 line with grey dash
plt.plot([-0.5, 5.1], [-0.5, 5.1], color='grey', linestyle='--', linewidth=1, zorder=1)

# Plot horizontal uncertainty lines with markers at both ends
scatterplot = sns.scatterplot(x='obs_high', y='sim', data=compr_df, hue='city', palette=city_palette, s=50, alpha=1, edgecolor='k', style='city', markers=city_marker, zorder=3) # linestyle=(0, (3, 3))
sns.scatterplot(x='obs_low', y='sim', data=compr_df, hue='city', palette=city_palette, s=50, alpha=1, edgecolor='k', style='city', markers=city_marker, zorder=3)
for i, row in compr_df.iterrows():
    ax.plot([row['obs_low'], row['obs_high']], [row['sim'], row['sim']], color='black', alpha=0.7, linewidth=1.5, zorder=2)
# Overlay markers with vertical line style inside obs_high
for i, row in compr_df.iterrows():
    ax.plot(row['obs_low'], row['sim'], marker=r'$/$', markersize=3, color='black', alpha=0.7, linewidth=0, zorder=4)
    ax.plot(row['obs_high'], row['sim'], marker=r'$\backslash$', markersize=3, color='black', alpha=0.7, linewidth=0, zorder=4)


# Customize axis spines
for spine in ax.spines.values():
    spine.set_edgecolor('black')
    spine.set_linewidth(1)

# Customize legend markers
handles, labels = scatterplot.get_legend_handles_labels()
sorted_handles = [handles[list(labels).index(city)] for city in sorted_cities]
border_width = 1
# Customize legend order
legend = plt.legend(handles=sorted_handles, labels=sorted_cities, facecolor='white', bbox_to_anchor=(1.03, 0.50), loc='center left', fontsize=12, markerscale=1.25)
legend.get_frame().set_edgecolor('black')

# Set title, xlim, ylim, ticks, labels
# plt.title(f'GCHP-v13.4.1 {cres.lower()} {inventory} {deposition} vs SPARTAN', fontsize=16, fontname='Arial', y=1.03)  # PM$_{{2.5}}$
plt.xlim([-0.5, 11])
plt.ylim([-0.5, 11])
plt.xticks([0, 2, 4, 6, 8, 10], fontname='Arial', size=18)
plt.yticks([0, 2, 4, 6, 8, 10], fontname='Arial', size=18)
scatterplot.tick_params(axis='x', direction='out', width=1, length=5)
scatterplot.tick_params(axis='y', direction='out', width=1, length=5)

# Perform linear regression for the first segment
# mask_1 = (compr_df['obs'] >= x_range_1[0]) & (compr_df['obs'] <= x_range_1[1])
mask_1_high = (compr_df['obs_high'] >= x_range_1_high[0]) & (compr_df['obs_high'] <= x_range_1_high[1])
mask_1_low = (compr_df['obs_low'] >= x_range_1_low[0]) & (compr_df['obs_low'] <= x_range_1_low[1])
# slope_1, intercept_1, r_value_1, p_value_1, std_err_1 = stats.linregress(compr_df['obs'][mask_1], compr_df['sim'][mask_1])
slope_1_high, intercept_1_high, r_value_1_high, p_value_1_high, std_err_1_high = stats.linregress(compr_df['obs_high'][mask_1_high], compr_df['sim'][mask_1_high])
slope_1_low, intercept_1_low, r_value_1_low, p_value_1_low, std_err_1_low = stats.linregress(compr_df['obs_low'][mask_1_low], compr_df['sim'][mask_1_low])
# Perform linear regression for the second segment
# mask_2 = (compr_df['obs'] >= x_range_2[0]) & (compr_df['obs'] <= x_range_2[1])
mask_2_high = (compr_df['obs_high'] >= x_range_2_high[0]) & (compr_df['obs_high'] <= x_range_2_high[1])
mask_2_low = (compr_df['obs_low'] >= x_range_2_low[0]) & (compr_df['obs_low'] <= x_range_2_low[1])
# slope_2, intercept_2, r_value_2, p_value_2, std_err_2 = stats.linregress(compr_df['obs'][mask_2], compr_df['sim'][mask_2])
slope_2_high, intercept_2_high, r_value_2_high, p_value_2_high, std_err_2_high = stats.linregress(compr_df['obs_high'][mask_2_high], compr_df['sim'][mask_2_high])
slope_2_low, intercept_2_low, r_value_2_low, p_value_2_low, std_err_2_low = stats.linregress(compr_df['obs_low'][mask_2_low], compr_df['sim'][mask_2_low])

# # Plot regression lines
# # sns.regplot(x='obs', y='sim', data=compr_df[mask_1],scatter=False, ci=None, line_kws={'color': 'blue', 'linestyle': '-', 'linewidth': 1.5}, ax=ax)
# sns.regplot(x='obs_high', y='sim', data=compr_df[mask_1_high],
#             scatter=False, ci=None, line_kws={'color': 'lightblue', 'linestyle': '--', 'linewidth': 0.5}, ax=ax)
# sns.regplot(x='obs_low', y='sim', data=compr_df[mask_1_low],
#             scatter=False, ci=None, line_kws={'color': 'lightblue', 'linestyle': '--', 'linewidth': 0.5}, ax=ax)
# # Regression lines
# y_range_1 = [slope_1 * x_range_1[0] + intercept_1, slope_1 * x_range_1[1] + intercept_1]
# y_vals = np.linspace(y_range_1[0], y_range_1[1], 100)
# regression_high_x = (y_vals - intercept_1_high) / slope_1_high
# regression_low_x = (y_vals - intercept_1_low) / slope_1_low
# # Fill the area between regression_low and regression_high with light blue
# ax.fill_betweenx(y_vals, regression_low_x, regression_high_x, color='lightblue', alpha=0.5)

# Add text with linear regression equations for high (right)
intercept_display_1_high = abs(intercept_1_high)
intercept_display_2_high = abs(intercept_2_high)
intercept_display_1_low = abs(intercept_1_low)
intercept_display_2_low = abs(intercept_2_low)
intercept_sign_1_high = '-' if intercept_1_high < 0 else '+'
intercept_sign_2_high = '-' if intercept_2_high < 0 else '+'
intercept_sign_1_low = '-' if intercept_1_high < 0 else '+'
intercept_sign_2_low = '-' if intercept_2_high < 0 else '+'
num_points_1 = mask_1_high.sum()
num_points_2 = mask_2_high.sum()
plt.text(0.5, 0.73, f'y$_L$ = {slope_1_low:.1f}x$_L$ {intercept_sign_1_low} {intercept_display_1_low:.2f}\ny$_H$ = {slope_1_high:.1f}x$_H$ {intercept_sign_1_high} {intercept_display_1_high:.2f}\n$r^2$ = {r_value_1_high ** 2:.2f}\nN = {num_points_1}',
         transform=scatterplot.transAxes, fontsize=18, color='blue')
plt.text(0.5, 0.48, f'y$_L$ = {slope_2_low:.3f}x$_L$ {intercept_sign_2_low} {intercept_display_2_low:.1f}\ny$_H$ = {slope_2_high:.4f}x$_H$ {intercept_sign_2_high} {intercept_display_2_high:.1f}\n$r^2$ = {r_value_2_high ** 2:.5f}\nN = {num_points_2}',
         transform=scatterplot.transAxes, fontsize=18, color='red')

# Set labels
plt.xlabel('HIPS Measured Black Carbon (µg/m$^3$)', fontsize=18, color='black', fontname='Arial')
plt.ylabel('Simulated Black Carbon (µg/m$^3$)', fontsize=18, color='black', fontname='Arial')

# Show the plot
plt.tight_layout()
# plt.savefig(out_dir + 'Fig2_Scatter_{}_{}_{}_vs_SPARTAN_{}_{:02d}_MAC7+13_BeijingGrey.svg'.format(cres, inventory, deposition, species, year), dpi=300)
plt.show()