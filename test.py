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


cres = 'C360'
year = 2019
species = 'BC'
inventory = 'CEDS'
deposition = 'noLUO'

# Set the directory path
out_dir = '/Volumes/rvmartin/Active/ren.yuxuan/BC_Comparison/representative_bias/' + 'Beijing_c360_sim_vs_SPARTAN_differeny_year.xlsx'

################################################################################################
# Create scatter plot for monthly and annual data
################################################################################################
# Read the file
# compr_df = pd.read_excel('/Volumes/rvmartin/Active/ren.yuxuan/BC_Comparison/representative_bias/Beijing_c360_sim_vs_SPARTAN_differeny_year.xlsx', sheet_name='Mon')
compr_df = pd.read_excel('/Volumes/rvmartin/Active/ren.yuxuan/BC_Comparison/representative_bias/Beijing_c360_sim_vs_SPARTAN_differeny_year.xlsx', sheet_name='Annual')
compr_df[compr_df['city'] == 'Beijing']
# Drop rows where BC is greater than 1
# compr_df = compr_df.loc[compr_df['obs'] <= 20]

# Print the names of each city
unique_cities = compr_df['city'].unique()
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
        (0, 0, 0.6), (0, 0.27, 0.8), (0, 0, 1), (0.4, 0.5, 0.9), (0.431, 0.584, 1), (0.7, 0.76, 0.9)
    ],  # Blue shades
    'Central Asia': [
        (0.58, 0.1, 0.81), (0.66, 0.33, 0.83), (0.9, 0.4, 1), (0.73, 0.44, 0.8), (0.8, 0.55, 0.77), (0.88, 0.66, 0.74)
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

# Define the range of x-values for the two segments
# x_range_1 = [compr_df['obs'].min(), 2.3]
# x_range_2 = [2.3, compr_df['obs'].max()] # 2.4 to include Beijing

# Create figure and axes objects
fig, ax = plt.subplots(figsize=(8, 6))

# Create scatter plot with white background, black border, and no grid
sns.set(font='Arial')
scatterplot = sns.scatterplot(x='obs', y='sim', data=compr_df[compr_df['year_obs'] == 2019], hue='city', palette=city_palette, s=80, alpha=1, ax=ax, edgecolor='k', marker='o')
scatterplot = sns.scatterplot(x='obs', y='sim', data=compr_df[compr_df['year_obs'] == 2020], hue='city', palette=city_palette, s=80, alpha=1, ax=ax, edgecolor='k', marker='s')
scatterplot = sns.scatterplot(x='obs', y='sim', data=compr_df[compr_df['year_obs'] == 2021], hue='city', palette=city_palette, s=80, alpha=1, ax=ax, edgecolor='k', marker='^')
scatterplot = sns.scatterplot(x='obs', y='sim', data=compr_df[compr_df['year_obs'] == 2022], hue='city', palette=city_palette, s=80, alpha=1, ax=ax, edgecolor='k', marker='x')
scatterplot = sns.scatterplot(x='obs', y='sim', data=compr_df[compr_df['year_obs'] == 2023], hue='city', palette=city_palette, s=80, alpha=1, ax=ax, edgecolor='k', marker='*')


scatterplot.set_facecolor('white')  # set background color to white
border_width = 1
for spine in scatterplot.spines.values():
    spine.set_edgecolor('black')  # set border color to black
    spine.set_linewidth(border_width)  # set border width
scatterplot.grid(False)  # remove the grid

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
# Create legend with custom order
sorted_city_color_match = sorted(city_color_match, key=lambda x: (
    list(region_mapping.keys()).index(get_region_for_city(x['city'])),
    region_mapping[get_region_for_city(x['city'])].index(x['city'])
))
legend_labels = [city['city'] for city in sorted_city_color_match]
legend_handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=city['color'], markersize=8, label=city['city']) for city in sorted_city_color_match]
legend = plt.legend(handles=legend_handles, facecolor='white', bbox_to_anchor=(1.03, 0.50), loc='center left', fontsize=11.5)
# legend.get_frame().set_linewidth(0.0)

# Set title, xlim, ylim, ticks, labels
plt.title(f'GCHP-v13.4.1 {cres.lower()} {inventory} {deposition} vs SPARTAN',
          fontsize=16, fontname='Arial', y=1.03)  # PM$_{{2.5}}$
plt.xlim([-0.5, 100]) # 14 for edgar
plt.ylim([-0.5, 100])
plt.xticks([0, 2, 4, 6, 8, 10, 12], fontname='Arial', size=18)
plt.yticks([0, 2, 4, 6, 8, 10, 12], fontname='Arial', size=18)
# plt.yticks([0, 5, 10, 15, 20], fontname='Arial', size=18)
scatterplot.tick_params(axis='x', direction='out', width=1, length=5)
scatterplot.tick_params(axis='y', direction='out', width=1, length=5)

# Add 1:1 line with grey dash
x = compr_df['obs']
y = compr_df['obs']
plt.plot([compr_df['obs'].min(), compr_df['obs'].max()], [compr_df['obs'].min(), compr_df['obs'].max()],
         color='grey', linestyle='--', linewidth=1)


plt.xlabel('Observed Black Carbon (µg/m$^3$)', fontsize=18, color='black', fontname='Arial')
plt.ylabel('Simulated Black Carbon (µg/m$^3$)', fontsize=18, color='black', fontname='Arial')

# show the plot
plt.tight_layout()
# plt.savefig(out_dir + 'Scatter_{}_{}_{}_Sim_vs_SPARTAN_{}_{:02d}_MonMean.tiff'.format(cres, inventory, deposition, species, year), dpi=600)
# plt.savefig(out_dir + 'Scatter_{}_{}_{}_Sim_vs_SPARTAN_{}_{:02d}_AnnualMean_2reg.tiff'.format(cres, inventory, deposition, species, year), dpi=600)
# plt.savefig('/Users/renyuxuan/Downloads/' + 'Scatter_{}_Sim_vs_SPARTAN_{}_{:02d}_MonMean.tiff'.format(cres, species, year), dpi=600)

plt.show()

