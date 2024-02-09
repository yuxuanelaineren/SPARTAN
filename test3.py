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


################################################################################################
# Create scatter plot for difference in sim and obs vs elevation
################################################################################################
# Set the directory path
sim_dir = '/Volumes/rvmartin/Active/ren.yuxuan/BC_Comparison/'
out_dir = '/Volumes/rvmartin/Active/ren.yuxuan/BC_Comparison/representative_bias/'
# Read the file
# diff_df = pd.read_excel(os.path.join(sim_dir, 'c360_BC/C360_LUO_Sim_vs_SPARTAN_BC_2018_Summary.xlsx'), sheet_name = 'Annual')
diff_df = pd.read_excel(os.path.join(out_dir, 'c360_HTAP_CEDS_HDI.xlsx'))

# Print the names of each city
unique_cities = diff_df['city'].unique()
for city in unique_cities:
    print(f"City: {city}")

# Classify 'city' based on 'region'
region_mapping = {
    'Central Asia': ['Abu Dhabi', 'Haifa', 'Rehovot'],
    'South Asia': ['Dhaka', 'Bandung', 'Delhi', 'Kanpur', 'Manila', 'Singapore', 'Hanoi'],
    'East Asia': ['Beijing', 'Seoul', 'Ulsan', 'Kaohsiung', 'Taipei'],
    'North America': ['Downsview', 'Halifax', 'Kelowna', 'Lethbridge', 'Sherbrooke',
                      'Baltimore', 'Bondville', 'Mammoth Cave', 'Norman', 'Pasadena', 'Fajardo', 'Mexico City'],
    'South America': ['Buenos Aires', 'Santiago', 'Palmira'],
    'Africa': ['Bujumbura', 'Addis Ababa', 'Ilorin', 'Johannesburg', 'Pretoria'],
    'Australia': ['Melbourne']
}
# Define custom palette for each region with 5 shades for each color
region_colors = {
    'Central Asia': [
        (1, 0.42, 0.70), (0.8, 0.52, 0.7), (1, 0.4, 0.4), (1, 0.64, 0.64), (1, 0.76, 0.48)
    ],  # Pink shades
    'South Asia': [
        (0.5, 0, 0), (0.8, 0, 0), (1, 0, 0), (1, 0.2, 0.2), (1, 0.48, 0.41), (1, 0.6, 0.6)
    ],  # Red shades
    'East Asia': [
        (1, 0.64, 0), (1, 0.55, 0.14), (1, 0.63, 0.48), (1, 0.74, 0.61), (1, 0.85, 0.73), (1, 0.96, 0.85)
    ],  # Orange shades
    'North America': [
        (0, 0, 0.5), (0, 0, 1), (0.39, 0.58, 0.93), (0, 0, 0.8), (0.54, 0.72, 0.97), (0.68, 0.85, 0.9)
    ],  # Blue shades
    'South America': [
        (0.58, 0.1, 0.81), (0.9, 0.4, 1), (0.66, 0.33, 0.83), (0.73, 0.44, 0.8), (0.8, 0.55, 0.77), (0.88, 0.66, 0.74)
    ],  # Purple shades
    'Africa': [
        (0, 0.5, 0), (0, 0.8, 0), (0, 1, 0), (0.56, 0.93, 0.56), (0.56, 0.93, 0.56), (0.8, 0.9, 0.8)
    ],  # Green shades
    'Australia': [
        (0.6, 0.4, 0.2)
    ]  # Brown
}

def map_city_to_color(city):
    for region, cities in region_mapping.items():
        if city in cities:
            city_index = cities.index(city) % len(region_colors[region])
            assigned_color = region_colors[region][city_index]
            print(f"City: {city}, Region: {region}, Assigned Color: {assigned_color}")
            return assigned_color
    print(f"City not found in any region: {city}")
    return (0, 0, 0)  # Default to black if city is not found
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

# Create figure and axes objects
fig, ax = plt.subplots(figsize=(8, 6))

# Create scatter plot with white background, black border, and no grid
sns.set(font='Arial')
# Plot 'difference_c360' as circles
sns.scatterplot(x='HDI', y='diff_f_HTAP', data=diff_df, hue='city', palette=city_palette, s=80, alpha=1, ax=ax, edgecolor='k', marker='o')
# Plot 'difference_c720' as triangles
sns.scatterplot(x='HDI', y='diff_f_CEDS', data=diff_df, hue='city', palette=city_palette, s=80, alpha=1, ax=ax, edgecolor='k', marker='^')

ax.set_facecolor('white')
border_width = 1
for spine in ax.spines.values():
    spine.set_edgecolor('black')  # set border color to black
    spine.set_linewidth(border_width)  # set border width
ax.grid(False)  # remove the grid


# Create a function to determine the index of a city in region_mapping
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
# Sort the unique_cities list based on their appearance in region_mapping
unique_cities_sorted = sorted(unique_cities, key=get_city_index)
# Create city legend with custom order
sorted_city_color_match = sorted(city_color_match, key=lambda x: (
    list(region_mapping.keys()).index(get_region_for_city(x['city'])),
    region_mapping[get_region_for_city(x['city'])].index(x['city'])
))
legend_labels = [city['city'] for city in sorted_city_color_match]
legend_handles_city = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=city['color'], markersize=8, label=city['city']) for city in sorted_city_color_match]
legend_city = ax.legend(handles=legend_handles_city, facecolor='white', bbox_to_anchor=(1.03, 0.50), loc='center left', fontsize=12)

# Set x-axis scale to logarithmic
# plt.xscale('log')
# Set title, xlim, ylim, ticks, labels
plt.title(f'HDI vs Normalized Difference',
          fontsize=18, fontname='Arial', y=1.03)  # PM$_{{2.5}}$
plt.xlim([0.4, 1])
plt.ylim([-3, 2])
# plt.xlim([annual_df['sim'].min()-0.5, annual_df['sim'].max()+0.5])
# plt.ylim([annual_df['sim'].min()-0.5, annual_df['sim'].max()+0.5])
plt.xticks([0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1], fontname='Arial', size=18)
plt.yticks([-3, -2, -1, 0, 1, 2], fontname='Arial', size=18)
ax.tick_params(axis='x', direction='out', width=1, length=5)
ax.tick_params(axis='y', direction='out', width=1, length=5)

# Add y = 0 with grey dash
plt.axhline(y=0, color='grey', linestyle='--', linewidth=1)

# Add number of data points to the plot
num_points = len(diff_df)
# plt.text(0.1, 0.7, f'N = {num_points}', transform=ax.transAxes, fontsize=18)
# plt.text(0.1, 0.7, f'N = {num_points}', transform=scatterplot.transAxes, fontsize=22)

plt.xlabel('Human Development Index (HDI)', fontsize=18, color='black', fontname='Arial')
plt.ylabel('Normalized Difference (Observation - Simulation)', fontsize=18, color='black', fontname='Arial')

# show the plot
plt.tight_layout()
plt.savefig(out_dir + 'Scatter_c360_HTAP_CEDS_BC_Normalized_Difference_HDI.tiff', dpi=600)
# plt.savefig(out_dir + 'Scatter_{}_Sim_vs_SPARTAN_{}_{:02d}_AnnualMean.tiff'.format(cres, species, year), dpi=600)
# plt.savefig('/Users/renyuxuan/Downloads/' + 'Scatter_{}_Sim_vs_SPARTAN_{}_{:02d}_MonMean.tiff'.format(cres, species, year), dpi=600)

plt.show()
