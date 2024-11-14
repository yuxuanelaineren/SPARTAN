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

cres = 'C360'
year = 2019
species = 'BC'
inventory = 'CEDS'
deposition = 'noLUO'

# Set the directory path
sim_dir = '/Volumes/rvmartin2/Active/Shared/dandan.z/GCHP-v13.4.1/{}-CEDS01-fixed-vert-{}-output/monthly/'.format(cres.lower(), deposition) # CEDS, noLUO
# sim_dir = '/Volumes/rvmartin2/Active/Shared/dandan.z/GCHP-v13.4.1/{}-EDGARv61-vert-{}-output/monthly/'.format(cres.lower(), deposition) # EDGAR, noLUO
# sim_dir = '/Volumes/rvmartin2/Active/Shared/dandan.z/GCHP-v13.4.1/{}-HTAPv3-vert-{}-output/monthly/'.format(cres.lower(), deposition) # HTAP, noLUO
# sim_dir = '/Volumes/rvmartin/Active/ren.yuxuan/BC_Comparison/WUCR3-C360/' # EDGAR, LUO
# sim_dir = '/Volumes/rvmartin/Active/ren.yuxuan/BC_Comparison/{}_{}_{}_{}/'.format(cres.lower(), inventory, deposition, year) # C720, HTAP, LUO
obs_dir = '/Volumes/rvmartin/Active/SPARTAN-shared/Analysis_Data/Master_files/'
site_dir = '/Volumes/rvmartin/Active/SPARTAN-shared/Site_Sampling/'
out_dir = '/Volumes/rvmartin/Active/ren.yuxuan/BC_Comparison/{}_{}_{}_{}/'.format(cres.lower(), inventory, deposition, year)
support_dir = '/Volumes/rvmartin/Active/ren.yuxuan/BC_Comparison/supportData/'
################################################################################################
# Create scatter plot: sim vs meas, color by GDP per Captia
################################################################################################
# Read the file
compr_df = pd.read_excel(os.path.join(out_dir, '{}_{}_{}_vs_SPARTAN_{}_{}.xlsx'.format(cres, inventory, deposition, species, year)), sheet_name='Annual')
compr_df['obs'] = 1 * compr_df['obs']
compr_df['obs_se'] = 1 * compr_df['obs_se']

# Print the names of each city
unique_cities = compr_df['city'].unique()
for city in unique_cities:
    print(f"City: {city}")

# Define the country groups
red_group = ["Burundi", "Ethiopia", "India", "Bangladesh", "Nigeria", "Indonesia", "South Africa", "China", "Mexico"]
blue_group = ['Taiwan', "Korea", "PuertoRico", "Israel", "UAE", "Canada", "Australia", "United States", "Singapore"]
# Add a new column to classify each country as 'red' or 'blue'
compr_df['color'] = compr_df['country'].apply(lambda x: 'red' if x in red_group else ('blue' if x in blue_group else 'other'))
# Define a custom function to map each city to a color based on its country
def map_city_to_color(city):
    if compr_df.loc[compr_df['city'] == city, 'color'].values[0] == 'red':
        return 'red'
    elif compr_df.loc[compr_df['city'] == city, 'color'].values[0] == 'blue':
        return 'blue'
    else:
        return 'grey'
city_palette = [map_city_to_color(city) for city in compr_df['city']]
# sorted_cities = sorted(compr_df['city'].unique(), key=lambda city: (compr_df.loc[compr_df['city'] == city, 'country'].iloc[0], compr_df.loc[compr_df['city'] == city, 'obs'].iloc[0]))
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
plt.plot([-0.5, 6.2], [-0.5, 6.2], color='grey', linestyle='--', linewidth=1, zorder=1)
# # Add error bars
# for i in range(len(compr_df)):
#     ax.errorbar(compr_df['obs'].iloc[i], compr_df['sim'].iloc[i],
#                 xerr=compr_df['obs_se'].iloc[i], yerr=compr_df['sim_se'].iloc[i],
#                 fmt='none', color='k', alpha=1, capsize=2, elinewidth=1, zorder=1) # color=city_palette[i], color='k'
# Create scatter plot
scatterplot = sns.scatterplot(x='obs', y='sim', data=compr_df, hue='city', palette=city_palette, s=80, alpha=1, edgecolor='k', style='city', markers=city_marker, zorder=2)
# Customize axis spines
for spine in ax.spines.values():
    spine.set_edgecolor('black')
    spine.set_linewidth(1)

# # Customize legend markers
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

# Perform regression for blue group
blue_mask = compr_df['color'] == 'blue'
slope_blue, intercept_blue, r_value_blue, _, _ = stats.linregress(compr_df.loc[blue_mask, 'obs'], compr_df.loc[blue_mask, 'sim'])
sns.regplot(x='obs', y='sim', data=compr_df[blue_mask], scatter=False, ci=None, line_kws={'color': 'blue', 'linestyle': '-', 'linewidth': 1.5}, ax=ax)
plt.text(0.6, 0.83, f'y = {slope_blue:.2f}x + {intercept_blue:.2f}\n$r^2$ = {r_value_blue**2:.2f}', color='blue', transform=ax.transAxes, fontsize=18)

# Perform regression for red group
red_mask = compr_df['color'] == 'red'
slope_red, intercept_red, r_value_red, _, _ = stats.linregress(compr_df.loc[red_mask, 'obs'], compr_df.loc[red_mask, 'sim'])
# sns.regplot(x='obs', y='sim', data=compr_df[red_mask], scatter=False, ci=None, line_kws={'color': 'red', 'linestyle': '-', 'linewidth': 1.5}, ax=ax)
plt.text(0.6, 0.61, f'y = {slope_red:.2f}x + {intercept_red:.2f}\n$r^2$ = {r_value_red**2:.2f}', color='red', transform=ax.transAxes, fontsize=18)
# Add the number of data points
num_points_1 = blue_mask.sum()
plt.text(0.6, 0.77, f'N = {num_points_1}', transform=scatterplot.transAxes, fontsize=18, color='blue')
num_points_2 = red_mask.sum()
plt.text(0.6, 0.55, f'N = {num_points_2}', transform=scatterplot.transAxes, fontsize=18, color='red')


# Set labels
plt.xlabel('HIPS Measured Black Carbon (µg/m$^3$)', fontsize=18, color='black', fontname='Arial')
plt.ylabel('Simulated Black Carbon (µg/m$^3$)', fontsize=18, color='black', fontname='Arial')

# Show the plot
plt.tight_layout()
# plt.savefig(out_dir + 'Scatter_{}_{}_{}_vs_SPARTAN_{}_{:02d}_GDPperCaptia.svg'.format(cres, inventory, deposition, species, year), dpi=300)
plt.show()